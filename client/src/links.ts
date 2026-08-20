// Every address the client can send someone to, in one file.
//
// Written 2026-08-20 (audit A-P1-4/A-P1-5). Before it, the client had exactly
// one outbound link — "create an account on the website" — while its copy told
// people to "report it" and to "open a GitHub issue" with nothing to click, and
// pointed at settings categories that had been deleted. Links scattered through
// the string table are links nobody re-checks; one file is one thing to check.
//
// Two kinds live here and they behave differently:
//   * FIXED  — the public repository. It does not depend on any setting, so it
//              is always available.
//   * DERIVED — anything on the platform (legal texts, password reset). These
//              are null when no platform is configured, and the UI must then
//              not render the entry at all rather than show a dead control.
import { loadSettings } from "./settings";

/** The public repository. Same host the updater's release feed comes from. */
export const REPO_URL = "https://github.com/idletoken/IdleToken";
/** Where a missing model, a bad download source or any other defect goes. */
export const ISSUES_URL = `${REPO_URL}/issues`;
/** Downloads + the user guide (the README is the guide). */
export const RELEASES_URL = `${REPO_URL}/releases`;
export const GUIDE_URL = `${REPO_URL}#readme`;
/** Third-party attribution, as Apache-2.0 §4(d) asks. */
export const NOTICE_URL = `${REPO_URL}/blob/main/NOTICE`;
/** How to report a security problem privately. */
export const SECURITY_URL = `${REPO_URL}/blob/main/SECURITY.md`;

/** The configured platform API base, or "" when this build has none. */
function platformBase(): string {
  return loadSettings().platformUrl.trim().replace(/\/+$/, "");
}

/**
 * The website, derived from the API host the same way `auth.portalRegisterUrl`
 * does it (`api.example.com` → `example.com`), or null offline.
 */
export function portalUrl(): string | null {
  const built = (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_PORTAL_URL) || "";
  const base = built || platformBase();
  if (!base) return null;
  try {
    const u = new URL(base);
    u.hostname = u.hostname.replace(/^api\./, "");
    return u.origin;
  } catch {
    return null;
  }
}

/**
 * The website's sign-in page, which is where "forgot my password" lives (the
 * client deliberately has no reset form of its own: the reset link arrives by
 * email and lands on the website).
 */
export function portalSignInUrl(): string | null {
  const base = portalUrl();
  return base ? `${base}/login` : null;
}

/**
 * The platform's legal texts. Unauthenticated by design — terms have to be
 * readable before anyone signs anything — and returned as markdown, which is
 * why the client renders them itself instead of opening a browser tab full of
 * JSON.
 */
export function legalUrl(doc: "tos" | "privacy", lang: "en" | "zh"): string | null {
  const base = platformBase();
  return base ? `${base}/legal/${doc}?lang=${lang}` : null;
}
