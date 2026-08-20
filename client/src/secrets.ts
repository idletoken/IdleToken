// Credentials the client has to keep between runs, held outside localStorage.
//
// Added 2026-08-20 (audit A-P2-2). The platform session JWT — the thing that
// can spend this account's Sparks — was a plain string in `localStorage`. In
// the desktop app it now lives in a 0600 file the host process owns
// (src-tauri/src/secrets.rs); in the browser dev build there is no host
// process, so localStorage remains, which is exactly as safe as a dev build is
// meant to be.
//
// The store is SYNCHRONOUS to read, on purpose. `AuthProvider.currentSession`
// and `platformGate` are called during render and cannot become async without
// rewriting every caller. So the host is read once at startup (`hydrate`) into
// a module-level cache, and writes go through to the host in the background.
// Until `hydrate` resolves the cache is empty and the app reads as signed out —
// which is why App awaits it before it decides what to render.
// The Tauri check is inlined rather than imported from platform.ts on purpose:
// auth.ts imports this module and platform.ts imports auth.ts, and a cycle
// through a module that runs code at import time is how a build starts failing
// on a "cannot access before initialization" that has nothing to do with the
// change that caused it.
function inTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/** Where the platform session lives. Same name as the old localStorage key, so
 *  the migration below can find what previous versions wrote. */
export const SESSION_KEY = "idletoken.auth.session";

let cache: Record<string, string> = {};
let hydrated = false;

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke: call } = await import("@tauri-apps/api/core");
  return call<T>(cmd, args);
}

/**
 * Read the store into the cache. Safe to call more than once; the second call
 * is a no-op so a re-render cannot cost a file read.
 *
 * Also performs the one-way migration out of localStorage: anything a previous
 * version left there is moved into the host store and then REMOVED, because a
 * copy left behind would mean the audit finding is still true for every
 * upgraded install.
 */
export async function hydrateSecrets(): Promise<void> {
  if (hydrated) return;
  hydrated = true;
  if (!inTauri()) {
    // Browser dev build: localStorage is the store, read it as-is.
    for (const k of [SESSION_KEY]) {
      const v = localStorage.getItem(k);
      if (v) cache[k] = v;
    }
    return;
  }
  try {
    cache = (await invoke<Record<string, string>>("secrets_load")) ?? {};
  } catch {
    cache = {};
  }
  for (const k of [SESSION_KEY]) {
    const legacy = localStorage.getItem(k);
    if (!legacy) continue;
    if (!cache[k]) {
      cache[k] = legacy;
      try {
        await invoke("secrets_set", { key: k, value: legacy });
      } catch {
        // Could not write the host store: keep the localStorage copy rather
        // than sign the user out. Removing it here would lose the session.
        continue;
      }
    }
    localStorage.removeItem(k);
  }
}

export function getSecret(key: string): string | null {
  return cache[key] ?? null;
}

export function setSecret(key: string, value: string): void {
  if (value) cache[key] = value;
  else delete cache[key];
  if (!inTauri()) {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
    return;
  }
  // Fire and forget: the cache is already correct, and a failed write means
  // the session does not survive a restart — not that this one is invalid.
  void invoke("secrets_set", { key, value }).catch(() => {});
}

export function clearSecret(key: string): void {
  setSecret(key, "");
}
