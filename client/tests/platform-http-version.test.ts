import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { version as PACKAGE_VERSION } from "../package.json";
import {
  IDLETOKEN_VERSION_HEADER,
  PLATFORM_CLIENT_VERSION,
  platformRequest,
} from "../src/platformHttp";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

test("the browser control-plane wrapper sends the package version on every request", async () => {
  const seen: Array<{ url: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    seen.push({ url: String(input), init });
    return new Response("{}", { status: 200 });
  }) as typeof fetch;

  await platformRequest("https://idletoken.ai/providers", { bearer: "secret-token" });
  await platformRequest("https://idletoken.ai/providers/p-1/heartbeat", {
    method: "POST",
    body: "{}",
    bearer: "secret-token",
  });

  assert.equal(PLATFORM_CLIENT_VERSION, PACKAGE_VERSION);
  assert.equal(seen.length, 2);
  for (const call of seen) {
    const headers = new Headers(call.init?.headers);
    assert.equal(headers.get(IDLETOKEN_VERSION_HEADER), PACKAGE_VERSION);
  }
  assert.equal(new Headers(seen[0].init?.headers).get("authorization"), "Bearer secret-token");
});

test("the compatibility header contains no bearer material", async () => {
  let headers = new Headers();
  globalThis.fetch = (async (_input: string | URL | Request, init?: RequestInit) => {
    headers = new Headers(init?.headers);
    return new Response("{}", { status: 200 });
  }) as typeof fetch;

  await platformRequest("https://idletoken.ai/providers", { bearer: "do-not-copy" });
  assert.equal(headers.get(IDLETOKEN_VERSION_HEADER), PACKAGE_VERSION);
  assert.equal(headers.get(IDLETOKEN_VERSION_HEADER)?.includes("do-not-copy"), false);
});
