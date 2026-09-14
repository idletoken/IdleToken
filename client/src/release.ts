// What this build knows about its own age.
//
// The client has no updater and will not get one back (retired 2026-09-02):
// nothing here fetches an artifact, checks a signature, or installs anything.
// It reads two version strings the platform publishes and turns them into two
// statements, both of which a user can act on with a browser:
//
//   "a newer version exists"   → advice, shown as one quiet line in Settings.
//   "this build cannot share"  → a marketplace admission fact, enforced by the
//                                gateway and surfaced here so the sharing
//                                switch can explain itself instead of failing.
//
// Two rules that are easy to get wrong and expensive to get wrong:
//
//  1. NEVER render "you are up to date". The platform's `currentVersion` comes
//     from a deployment that may not have been redeployed since the last
//     release, so it can legitimately be older than what is installed. An
//     under-report is silent and harmless; an over-report is a false all-clear
//     about a security fix. When in doubt this module says nothing.
//
//  2. The floor is LISTING ELIGIBILITY, not permission to run the software. A
//     build below it must keep working normally — single machine, cluster,
//     local API, everything — and only decline to offer the machine to
//     strangers. Any copy derived from this module has to read that way.
import { PLATFORM_CLIENT_VERSION, platformRequest, replyJson } from "./platformHttp";
import { loadSettings } from "./settings";

export interface ReleasePolicy {
  /** Newest release the platform knows about. May legitimately lag this build. */
  currentVersion: string;
  /** Floor for putting this machine on the marketplace. Never a run-time gate. */
  minimumSupportedVersion: string;
}

export interface VersionStanding {
  /** This build's own version. */
  installed: string;
  /** Set only when the platform advertises something strictly newer. */
  newerVersion: string | null;
  /** True when this build is below the marketplace listing floor. */
  belowShareFloor: boolean;
  /** The floor itself, for copy that has to name it. */
  shareFloor: string | null;
}

/** Parse `major.minor.patch`, ignoring any prerelease/build suffix. */
function triple(v: string): [number, number, number] | null {
  const m = /^(\d+)\.(\d+)\.(\d+)/.exec(v.trim());
  if (!m) return null;
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

/** -1 / 0 / 1, or null when either side is unparseable. */
export function compareVersions(a: string, b: string): number | null {
  const x = triple(a);
  const y = triple(b);
  if (!x || !y) return null;
  for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] < y[i] ? -1 : 1;
  return 0;
}

/**
 * Turn the published policy into what this build should say about itself.
 *
 * Pure, so the two rules above are testable without a network. `policy === null`
 * (offline, signed out, old gateway) yields a standing that claims nothing:
 * no newer version, not below the floor. Silence is the correct output when
 * there is no information — the alternative is warning people at random.
 */
export function versionStanding(
  policy: ReleasePolicy | null,
  installed: string = PLATFORM_CLIENT_VERSION,
): VersionStanding {
  const quiet: VersionStanding = {
    installed, newerVersion: null, belowShareFloor: false, shareFloor: null,
  };
  if (!policy) return quiet;

  const vsCurrent = compareVersions(installed, policy.currentVersion);
  const vsFloor = compareVersions(installed, policy.minimumSupportedVersion);

  return {
    installed,
    // Strictly newer only. Equal says nothing, and OLDER-than-installed says
    // nothing either: that is a gateway behind the release, not a client ahead
    // of one, and there is nothing for the user to do about it.
    newerVersion: vsCurrent !== null && vsCurrent < 0 ? policy.currentVersion : null,
    // An unparseable floor is not a floor. Refusing to share on the strength of
    // a string we could not read would take the machine off the market over a
    // typo in a config file.
    belowShareFloor: vsFloor !== null && vsFloor < 0,
    shareFloor: vsFloor !== null ? policy.minimumSupportedVersion : null,
  };
}

const CACHE_MS = 6 * 60 * 60 * 1000;
let cached: { at: number; policy: ReleasePolicy | null } | null = null;

/**
 * Read the published policy. Public endpoint, no session required.
 *
 * Cached for six hours and never retried on a schedule: this is a line of text
 * in a settings page, not a service. Every failure resolves to null, because
 * "the platform is unreachable" must not become "your client is out of date".
 */
export async function fetchReleasePolicy(force = false): Promise<ReleasePolicy | null> {
  if (!force && cached && Date.now() - cached.at < CACHE_MS) return cached.policy;
  const base = loadSettings().platformUrl.trim().replace(/\/+$/, "");
  if (!base) {
    cached = { at: Date.now(), policy: null };
    return null;
  }
  let policy: ReleasePolicy | null = null;
  try {
    const res = await platformRequest(`${base}/release/client`, { timeoutMs: 8_000 });
    const body = res.ok ? replyJson<Partial<ReleasePolicy>>(res) : null;
    if (body && typeof body.currentVersion === "string"
        && typeof body.minimumSupportedVersion === "string") {
      policy = {
        currentVersion: body.currentVersion,
        minimumSupportedVersion: body.minimumSupportedVersion,
      };
    }
  } catch {
    // Offline, DNS, TLS, a proxy in the way. All of them mean "no information".
  }
  cached = { at: Date.now(), policy };
  return policy;
}

/** Test seam and settings-change hook: drop the cached answer. */
export function resetReleasePolicyCache(): void {
  cached = null;
}

/** The whole answer in one call, for callers that do not want two steps. */
export async function currentVersionStanding(force = false): Promise<VersionStanding> {
  return versionStanding(await fetchReleasePolicy(force));
}
