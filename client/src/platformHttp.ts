// One way out to the platform gateway, for every caller in the client.
//
// Why not just `fetch` (2026-08-21): the webview's origin is `tauri://localhost`
// (`http://tauri.localhost` on Windows) and the gateway's CORS allowlist is the
// portal's domains, so a preflight from the desktop app comes back without an
// `Access-Control-Allow-Origin` header and the browser blocks the request before
// it leaves the machine. `fetch` then rejects with a bare TypeError, which the
// caller can only report as "cannot reach the server" — which is what every
// sign-in attempt from the desktop client said, about a server that was up.
//
// In Tauri the request goes through the Rust command `platform_http`, which has
// no origin and triggers no preflight. In a plain browser (the dev server, the
// acceptance dist) it falls back to `fetch`, where CORS applies and is correct:
// there the app really is a web page on an origin the gateway can decide about.
//
// The reply deliberately keeps `status` and the raw `text` instead of throwing
// on non-2xx. Every caller here distinguishes 401 from 403 from 409 from "the
// request never completed", and a helper that collapses those is how the
// original bug got its misleading message.
// Deliberately NOT imported from platform.ts, which exports the same two-line
// check: auth.ts uses this module, platform.ts uses this module, and
// platform.ts already imports auth.ts — taking `inTauri` from there would close
// that into an import cycle for the sake of two lines.
function inTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export interface PlatformReply {
  status: number;
  ok: boolean;
  text: string;
}

/** Thrown when the request never completed: DNS, TLS, timeout, refused. NOT
 *  for an HTTP status the server chose to send — that is a completed request
 *  and comes back as a PlatformReply. */
export class PlatformNetworkError extends Error {}

export const PLATFORM_TIMEOUT_MS = 20_000;

export async function platformRequest(
  url: string,
  init?: { method?: string; body?: string; bearer?: string; timeoutMs?: number }
): Promise<PlatformReply> {
  const method = init?.method ?? "GET";

  if (inTauri()) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const r = await invoke<{ status: number; body: string }>("platform_http", {
        method,
        url,
        body: init?.body ?? null,
        bearer: init?.bearer ?? null,
      });
      return { status: r.status, ok: r.status >= 200 && r.status < 300, text: r.body };
    } catch (e) {
      // The command's own Err — the request did not complete. Its message is
      // reqwest's, which names the actual cause (dns error, connection refused,
      // certificate, timed out) instead of the browser's opaque TypeError.
      throw new PlatformNetworkError(e instanceof Error ? e.message : String(e));
    }
  }

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), init?.timeoutMs ?? PLATFORM_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      method,
      headers: {
        ...(init?.body ? { "content-type": "application/json" } : {}),
        ...(init?.bearer ? { authorization: `Bearer ${init.bearer}` } : {}),
      },
      ...(init?.body ? { body: init.body } : {}),
      signal: ctrl.signal,
    });
    return { status: res.status, ok: res.ok, text: await res.text() };
  } catch (e) {
    throw new PlatformNetworkError(e instanceof Error ? e.message : String(e));
  } finally {
    clearTimeout(timer);
  }
}

/** Parse a reply body as JSON, or null when it is not JSON (an nginx error
 *  page, an empty 204). Callers decide what a null means for them — it is not
 *  always a failure, and it is never something to invent a value for. */
export function replyJson<T>(r: PlatformReply): T | null {
  try {
    return JSON.parse(r.text) as T;
  } catch {
    return null;
  }
}
