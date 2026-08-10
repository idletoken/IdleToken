import type { NodeSnapshot } from "../types";
import { tauriProvider } from "./tauri";
import { devFixtureProvider } from "./devFixture";

// Optional usage caps (MiB; 0/undefined = no cap) that the probe must honor,
// so the "max VRAM / RAM" setting really takes effect (philosophy 15).
export interface ProbeOptions {
  maxVramMb?: number;
  maxRamMb?: number;
}

// Interface-first (design philosophy 14 + 17): the UI talks to a provider, not
// to Tauri directly. `real` drives the native engine over the local sidecar;
// `dev-fixture` renders the shell in a plain browser during development. Swapping
// backends (e.g. a future remote/mock coordinator) means one new implementation.
export interface ResourceProvider {
  probe(opts?: ProbeOptions): Promise<NodeSnapshot>;
}

// Tauri injects `window.__TAURI_INTERNALS__` into the webview. Its absence means
// we are in a browser dev server, so there is no engine to talk to.
function runningInTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export function getResourceProvider(): ResourceProvider {
  return runningInTauri() ? tauriProvider : devFixtureProvider;
}
