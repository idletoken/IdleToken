// How the diagnostics bundle is assembled. **There is exactly one copy**: the
// settings panel's "export diagnostics" button and the acceptance channel's
// report-diagnostics go through the same function, or we would be testing one
// path while users download another -- and this is precisely where it is decided
// whether an access token goes out, the last place that should have two
// implementations.
import type { AppSettings } from "./settings";
import { readProblems } from "./problems";

// The settings fields that go into the bundle (an **allowlist**). Anything that
// could carry a credential is excluded: apiToken, the platform session, any
// key or secret. One field missing costs one extra question while debugging;
// one field too many could hand a user's token to a stranger. A missed entry in
// a denylist is a leak, a missed entry in an allowlist is just a blank field --
// the same trade-off as the public mirror.
export const DIAG_SETTING_KEYS = [
  "modelId", "quant", "tier", "kvDir",
  "apiHost", "apiPort", "interStagePort", "discoveryPort",
  "resourcePreset", "maxVramMb", "maxRamMb", "computeMode",
  "clusterName", "preferCoordinator", "sameSubnetOnly", "bindNic",
  "platformUrl",
] as const satisfies readonly (keyof AppSettings)[];

/** The engine-side report plus the allowlisted settings = the file the user downloads. */
export function buildDiagnosticsBundle(
  report: Record<string, unknown>,
  settings: AppSettings,
): Record<string, unknown> {
  const picked: Record<string, unknown> = {};
  for (const k of DIAG_SETTING_KEYS) picked[k] = settings[k];
  // Recent problems (chat, downloads, cluster). The bundle exists so a user does
  // not have to open a console, and it used to carry hardware and settings but
  // not one word about what actually went wrong — the error lived in a banner
  // the next send wiped, so "what did it say?" had no answer. No prompts or
  // replies go in: see problems.ts for what is deliberately excluded.
  //
  // Always included (2026-08-15, the opt-out switch is gone): exporting the
  // bundle is itself the explicit act of sharing — it downloads a file the
  // user then sends by hand — and a diagnostics bundle without the failures
  // is the one kind that cannot diagnose anything.
  return {
    ...report,
    settings: picked,
    recent_problems: readProblems(),
  };
}

/** The filename carries a UTC timestamp: users often send several in a row, and identically named files have overwritten each other. */
export function diagnosticsFileName(now: Date = new Date()): string {
  return `idletoken-diagnostics-${now.toISOString().slice(0, 19).replace(/[:T]/g, "")}.json`;
}
