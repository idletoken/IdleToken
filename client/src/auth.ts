// Email identity + session (acceptance P2). Interface-first (philosophy 14):
// the UI talks to an AuthProvider, never to a specific backend. Two real
// implementations behind the same interface:
//   - LocalAuthProvider — identity stays on this machine (hashed passwords,
//     real rejection of bad logins). Works fully offline; pairs via code mode.
//   - CloudAuthProvider — the platform gateway's /auth (email + JWT). Needed
//     for account-mode pairing (machines signed in to the same account find
//     each other) and the marketplace.
// The selector picks by the `platformUrl` setting; the UI is unchanged.
import { loadSettings } from "./settings";
import { SESSION_KEY, clearSecret, getSecret, setSecret } from "./secrets";

export interface Session {
  email: string;
  token: string;
  createdAt: number;
  // Which backend issued this session. Absent in sessions stored by older
  // builds — treat those as "local".
  provider?: "local" | "cloud";
  userId?: string; // platform user id (cloud only)
}

// Errors carry a stable code the UI maps to a localized message.
export type AuthErrorCode =
  | "auth.err.email"
  | "auth.err.weak"
  | "auth.err.exists"
  | "auth.err.invalid"
  | "auth.err.registerOnWeb"
  | "auth.err.unverified"
  | "auth.err.network"
  | "auth.err.server";

export class AuthError extends Error {
  code: AuthErrorCode;
  constructor(code: AuthErrorCode) {
    super(code);
    this.code = code;
  }
}

export interface AuthProvider {
  kind: "local" | "cloud";
  currentSession(): Session | null;
  signUp(email: string, password: string): Promise<Session>;
  signIn(email: string, password: string): Promise<Session>;
  signOut(): void;
}

// ---- helpers --------------------------------------------------------------
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MIN_PASSWORD = 8;

function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

async function sha256Hex(s: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Password hashing for the local (offline) identity.
 *
 * Until 2026-08-20 this was one round of SHA-256 over `salt:password` (audit
 * A-P2-2). SHA-256 is built to be fast, which is the opposite of what a
 * password hash is for: a consumer GPU tries billions of candidates a second
 * against a stolen table, and the salt only stops one attack from covering
 * every account at once — it does nothing about the speed.
 *
 * PBKDF2-SHA256 at the OWASP-2023 iteration count instead. Not argon2 or
 * scrypt, which are better, because both mean a WebAssembly dependency and
 * bundle size is a hard constraint here (principle 10); PBKDF2 is in the Web
 * Crypto API every platform already ships, so this costs nothing but the work
 * factor it is supposed to cost.
 */
const KDF_ITERATIONS = 210_000;

async function derivePasswordHash(password: string, saltHex: string): Promise<string> {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, [
    "deriveBits",
  ]);
  const salt = Uint8Array.from(saltHex.match(/../g) ?? [], (h) => parseInt(h, 16));
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", hash: "SHA-256", salt, iterations: KDF_ITERATIONS },
    key,
    256
  );
  return Array.from(new Uint8Array(bits))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Check a password against a stored record, whichever scheme wrote it.
 *
 * Records written before the change carry no `kdf` field and are verified with
 * the old SHA-256 — refusing them would lock people out of their own machine
 * over a change they did not make. `upgraded` says the record should be
 * rewritten with the real KDF, which the caller does using the password it has
 * in hand right now (the only moment it can).
 */
async function verifyPassword(
  user: StoredUser,
  password: string
): Promise<{ ok: boolean; upgraded?: StoredUser }> {
  if (user.kdf === "pbkdf2") {
    return { ok: (await derivePasswordHash(password, user.salt)) === user.hash };
  }
  if ((await sha256Hex(`${user.salt}:${password}`)) !== user.hash) return { ok: false };
  const salt = randomHex(16);
  return { ok: true, upgraded: { salt, hash: await derivePasswordHash(password, salt), kdf: "pbkdf2" } };
}

function randomHex(bytes: number): string {
  const a = new Uint8Array(bytes);
  crypto.getRandomValues(a);
  return Array.from(a)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// ---- local provider -------------------------------------------------------
interface StoredUser {
  salt: string;
  hash: string;
  /** Which scheme produced `hash`. Absent = the pre-2026-08-20 single-round
   *  SHA-256; see verifyPassword for why those still work. */
  kdf?: "pbkdf2";
}
type UserTable = Record<string, StoredUser>;

const USERS_KEY = "idletoken.auth.users";

function readUsers(): UserTable {
  try {
    return JSON.parse(localStorage.getItem(USERS_KEY) || "{}") as UserTable;
  } catch {
    return {};
  }
}

function writeUsers(t: UserTable) {
  localStorage.setItem(USERS_KEY, JSON.stringify(t));
}

export const localAuthProvider: AuthProvider = {
  kind: "local",

  currentSession(): Session | null {
    try {
      const raw = getSecret(SESSION_KEY);
      return raw ? (JSON.parse(raw) as Session) : null;
    } catch {
      return null;
    }
  },

  async signUp(email: string, password: string): Promise<Session> {
    const e = normalizeEmail(email);
    if (!EMAIL_RE.test(e)) throw new AuthError("auth.err.email");
    if (password.length < MIN_PASSWORD) throw new AuthError("auth.err.weak");
    const users = readUsers();
    if (users[e]) throw new AuthError("auth.err.exists");
    const salt = randomHex(16);
    users[e] = { salt, hash: await derivePasswordHash(password, salt), kdf: "pbkdf2" };
    writeUsers(users);
    return persistSession({ email: e, token: randomHex(24), createdAt: Date.now(), provider: "local" });
  },

  async signIn(email: string, password: string): Promise<Session> {
    const e = normalizeEmail(email);
    if (!EMAIL_RE.test(e)) throw new AuthError("auth.err.email");
    const users = readUsers();
    const user = users[e];
    if (!user) throw new AuthError("auth.err.invalid");
    const { ok, upgraded } = await verifyPassword(user, password);
    if (!ok) throw new AuthError("auth.err.invalid");
    // A correct password is the only moment the plaintext exists, so it is the
    // only moment an old record can be rehashed with the real KDF.
    if (upgraded) {
      users[e] = upgraded;
      writeUsers(users);
    }
    return persistSession({ email: e, token: randomHex(24), createdAt: Date.now(), provider: "local" });
  },

  signOut(): void {
    clearSecret(SESSION_KEY);
  },
};

function persistSession(s: Session): Session {
  setSecret(SESSION_KEY, JSON.stringify(s));
  return s;
}

// ---- cloud provider (platform gateway /auth) --------------------------------
// Talks to the platform's auth endpoints (platform/packages/gateway):
//   POST /auth/register {email,password} -> {user, verificationRequired}   409 = exists
//     (no token: accounts are created on the website, sessions only via login)
//   POST /auth/login    {email,password} -> {token, user:{id,email}}   401 = invalid
// The JWT is the session token; account-mode pairing (P3) presents it to prove
// two machines belong to the same account. Only identity crosses the wire —
// no user data or model content (philosophy 16).
const FETCH_TIMEOUT_MS = 10_000;

class CloudAuthProvider implements AuthProvider {
  kind = "cloud" as const;
  constructor(private baseUrl: string) {}

  currentSession(): Session | null {
    return localAuthProvider.currentSession();
  }

  signOut(): void {
    localAuthProvider.signOut();
  }

  // Registration is the website's job (see portalRegisterUrl) and the gateway
  // no longer answers /auth/register with a session at all — there is nothing
  // this could return. The UI never reaches it (the signup form only exists in
  // builds with no platform), so this is a loud guard, not a code path.
  async signUp(): Promise<Session> {
    throw new AuthError("auth.err.registerOnWeb");
  }

  async signIn(email: string, password: string): Promise<Session> {
    const e = normalizeEmail(email);
    if (!EMAIL_RE.test(e)) throw new AuthError("auth.err.email");
    return this.post("/auth/login", e, password);
  }

  private async post(path: string, email: string, password: string): Promise<Session> {
    let res: Response;
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
      res = await fetch(this.baseUrl.replace(/\/+$/, "") + path, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email, password }),
        signal: ctrl.signal,
      });
      clearTimeout(timer);
    } catch {
      // Unreachable / timed out / DNS — distinct from "wrong password" so the
      // user knows to check the platform URL or network, not their credentials.
      throw new AuthError("auth.err.network");
    }
    if (res.status === 409) throw new AuthError("auth.err.exists");
    if (res.status === 401) throw new AuthError("auth.err.invalid");
    if (res.status === 400) throw new AuthError("auth.err.invalid");
    // 403 = credentials were right but the email is not verified yet (gateway
    // `email_not_verified`). Must stay distinct from 401: "wrong password" and
    // "go click the link in your inbox" need completely different reactions.
    if (res.status === 403) throw new AuthError("auth.err.unverified");
    if (!res.ok) throw new AuthError("auth.err.server");
    let body: { token?: string; user?: { id?: string; email?: string } };
    try {
      body = await res.json();
    } catch {
      throw new AuthError("auth.err.server");
    }
    if (!body?.token) throw new AuthError("auth.err.server");
    return persistSession({
      email: body.user?.email || email,
      token: body.token,
      createdAt: Date.now(),
      provider: "cloud",
      userId: body.user?.id,
    });
  }
}

// Selector — the `platformUrl` setting picks the backend: set = cloud auth
// against the platform gateway (enables account-mode pairing), empty = local
// identity (fully offline; code-mode pairing still works). Reads settings lazily
// so a URL change takes effect on the next auth action without a restart.
export function getAuthProvider(): AuthProvider {
  const url = loadSettings().platformUrl.trim();
  return url ? new CloudAuthProvider(url) : localAuthProvider;
}

// ---- where accounts are actually created -----------------------------------
// Signing UP is the platform's job, not the client's: the website owns email
// verification, the signup credit grant and the terms you have to accept. A
// second signup form in the client can only be a weaker copy of it that drifts
// — so the client links out and keeps only sign-IN.
//
// `/auth/login` does not require a verified email (only API keys and inference
// do), so an account created on the website signs in here straight away.
const BUILT_IN_PORTAL_URL: string =
  (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_PORTAL_URL) || "";

/**
 * URL of the website's registration form, or null when there is no website to
 * send anyone to (offline build with no platformUrl) — in that case the local
 * provider's signUp is the ONLY way to get an identity, so the client keeps its
 * own form as the fallback.
 *
 * Derivation exists for self-hosters: an API at `api.example.com` implies a
 * portal at `example.com`, the same pair the production deploy uses. An
 * explicit VITE_PORTAL_URL always wins.
 */
export function portalRegisterUrl(): string | null {
  const base = BUILT_IN_PORTAL_URL || derivePortalFromApi(loadSettings().platformUrl.trim());
  if (!base) return null;
  try {
    const u = new URL(base);
    // A real route, not `/?auth=register`: it is the address of the signup page,
    // so it can be bookmarked, pasted to someone else, and read as what it is.
    // (The portal also still honours the old query form — see its App.tsx.)
    u.pathname = "/register";
    u.search = "";
    return u.toString();
  } catch {
    return null;
  }
}

function derivePortalFromApi(api: string): string {
  if (!api) return "";
  try {
    const u = new URL(api);
    u.hostname = u.hostname.replace(/^api\./, "");
    return u.origin;
  } catch {
    return "";
  }
}

/**
 * Ask the platform to resend the verification email. Unauthenticated on
 * purpose: the user gets here by being LOCKED OUT of login, so there is no
 * session to authenticate with — the gateway accepts `{email}` for exactly
 * this case.
 *
 * "sent" is what the gateway's 200 means, and it answers 200 even for an
 * address it has never seen (it must not reveal who is registered), so the
 * message the UI shows is deliberately about the mailbox, not the account.
 * The one thing it does report is the 60s cooldown — claiming "sent" after a
 * 429 would be a plain lie, and the user would sit waiting for mail that was
 * never sent.
 */
export type ResendResult = "sent" | "rate-limited" | "failed";

export async function resendVerification(email: string): Promise<ResendResult> {
  const url = loadSettings().platformUrl.trim();
  if (!url) return "failed";
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(url.replace(/\/+$/, "") + "/auth/resend-verification", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email: normalizeEmail(email) }),
      signal: ctrl.signal,
    });
    if (res.status === 429) return "rate-limited";
    return res.ok ? "sent" : "failed";
  } catch {
    return "failed";
  } finally {
    clearTimeout(timer);
  }
}

/** Open a URL in the user's real browser (never inside the app webview). */
export async function openExternal(url: string): Promise<void> {
  if (typeof window !== "undefined" && "__TAURI_INTERNALS__" in window) {
    const { open } = await import("@tauri-apps/plugin-shell");
    await open(url);
    return;
  }
  window.open(url, "_blank", "noopener,noreferrer");
}
