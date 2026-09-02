import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";
import {
  DIAGNOSTIC_LIMITS,
  DIAG_SETTING_KEYS,
  buildDiagnosticsBundle,
  exportableSettings,
} from "../src/diagnostics";
import { DEFAULT_SETTINGS, type AppSettings } from "../src/settings";

const storage = new Map<string, string>();
globalThis.localStorage = {
  getItem(key: string) { return storage.get(key) ?? null; },
  setItem(key: string, value: string) { storage.set(key, value); },
  removeItem(key: string) { storage.delete(key); },
  clear() { storage.clear(); },
  key(index: number) { return [...storage.keys()][index] ?? null; },
  get length() { return storage.size; },
} as Storage;

const settings = (overrides: Partial<AppSettings> = {}): AppSettings => ({
  ...DEFAULT_SETTINGS,
  ...overrides,
});

beforeEach(() => storage.clear());

test("preserves allowlisted machine facts and useful errors", () => {
  const bundle = buildDiagnosticsBundle({
    schema: "idletoken-diagnostics/1",
    generatedAt: 1_788_112_800,
    app: { version: "0.1.30", os: "macos", arch: "arm64" },
    probe: {
      hostname: "private-macbook",
      gpu_name: "Apple M4 Pro",
      cpu_count: 14,
      unified_memory: true,
      vram_total: 25_769_803_776,
      ram_usable: 21_474_836_480,
      hw_status: 0,
      hw_reason: "CUDA driver unavailable after 2 attempts; see /Users/alice/.idletoken/engine.log",
    },
    advise: {
      nodes: 2,
      models: [{
        id: "qwen3.5-9b",
        label: "Qwen3.5 9B",
        quant: "Q4_K_M",
        mode: "gpu_only",
        max_ctx: 131_072,
        weight_bytes: 6_500_000_000,
        need_bytes: 8_000_000_000,
        shortfall_bytes: 0,
        available: true,
        single_node: false,
      }],
    },
    cluster: {
      phase: "ready",
      engine: "llamacpp",
      engine_state: "ready",
      cluster_size: 2,
      members: [{
        hostname: "private-worker",
        rpc_endpoint: "192.168.1.23:50052",
        role: "worker",
        state: "ready",
        gpu: "NVIDIA RTX 4090",
        stage: 1,
        layer_lo: 31,
        layer_hi: 61,
      }],
    },
  }, settings({
    apiHost: "127.0.0.1",
    kvDir: "/Users/alice/.idletoken/kv",
    bindNic: "192.168.1.23",
    platformUrl: "https://platform.idletoken.example/v1",
  }));

  assert.equal((bundle.probe as any).gpu_name, "Apple M4 Pro");
  // The useful half of the error survives; the path inside it does not. This is
  // also the only POSITIVE control for the path scrubber now that `kvDir` has
  // left the settings allowlist — and it has to live on a field like this one,
  // because the fixture in the next test puts its paths after a `prompt=`,
  // which PROMPT_ASSIGNMENT truncates to end-of-line before WINDOWS_PATH or
  // POSIX_PATH ever sees them.
  assert.match((bundle.probe as any).hw_reason, /^CUDA driver unavailable after 2 attempts/);
  assert.match((bundle.probe as any).hw_reason, /\[redacted-path\]/);
  assert.doesNotMatch((bundle.probe as any).hw_reason, /alice|engine\.log/);
  assert.equal((bundle.advise as any).models[0].max_ctx, 131_072);
  assert.deepEqual((bundle.cluster as any).members[0], {
    role: "worker",
    state: "ready",
    gpu: "NVIDIA RTX 4090",
    stage: 1,
    layer_lo: 31,
    layer_hi: 61,
  });
  assert.equal((bundle.settings as any).apiHost, "127.0.0.1");
  // `kvDir` left DIAG_SETTING_KEYS with the retired CPU-offload path, so it is
  // now dropped outright rather than redacted. Asserted as absent, not merely
  // deleted from this test: the setting is still populated above, so an
  // allowlist that let it through again would fail here. (It used to be the
  // only witness for `[redacted-path]`; that moved to `hw_reason` above rather
  // than being lost with it.)
  assert.equal(Object.hasOwn(bundle.settings as object, "kvDir"), false);
  assert.equal((bundle.settings as any).bindNic, "[redacted-ip]");
  assert.equal((bundle.settings as any).platformUrl, "https://platform.idletoken.example/v1");
  assert.doesNotMatch(JSON.stringify(bundle), /private-macbook|private-worker|192\.168\.1\.23/);
});

test("redacts nested report and recent-problem secrets at the final boundary", () => {
  const jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbGljZSJ9.signature";
  storage.set("idletoken.problems.v1", JSON.stringify([{
    at: "2026-08-31T00:00:00.000Z",
    kind: "chat",
    message: `prompt: private medical question; Bearer ${jwt}; alice@example.com; /Users/alice/private.txt; 192.168.1.9\u202e\nFORGED`,
    detail: { model: "qwen3.5-9b", quant: "Q4_K_M", injected: "secret" },
    prompt: "unknown fields do not leave",
  }]));

  const bundle = buildDiagnosticsBundle({
    schema: "idletoken-diagnostics/1",
    unknown: "private medical question",
    prompt: "private medical question",
    app: { version: "0.1.30", os: "macos", arch: "arm64", accessToken: "app-secret" },
    probe: {
      gpu_name: "Apple M4 Pro",
      error: `probe failed; prompt=private medical question; apiToken=api-secret; overflowKey=overflow-secret; providerKey=provider-secret; rendezvous_key=rendezvous-secret; ${jwt}; alice@example.com; C:\\Users\\alice\\secret.txt; /Users/alice/private.txt; 192.168.1.9; 2001:db8::1; https://admin:password@example.com/v1?api_key=query-secret\u200b\u202e\nFORGED`,
      nested: { prompt: "private medical question", overflowKey: "overflow-secret" },
    },
    cluster: {
      phase: "ready",
      cluster_size: 2,
      members: [{ role: "worker", state: "ready", rpc_endpoint: "203.0.113.8:50052" }],
    },
  }, settings({
    apiToken: "api-secret",
    overflowKey: "overflow-secret",
  }));

  const text = JSON.stringify(bundle);
  for (const secret of [
    "private medical question",
    "api-secret",
    "overflow-secret",
    "provider-secret",
    "rendezvous-secret",
    jwt,
    "alice@example.com",
    "C:\\Users\\alice\\secret.txt",
    "/Users/alice/private.txt",
    "192.168.1.9",
    "2001:db8::1",
    "query-secret",
  ]) assert.equal(text.includes(secret), false, `leaked ${secret}`);
  assert.doesNotMatch(text, /[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]/);
  assert.equal(Object.hasOwn(bundle, "unknown"), false);
  assert.equal(Object.hasOwn(bundle, "prompt"), false);
  assert.equal(Object.hasOwn(bundle.probe as object, "nested"), false);
  assert.equal((bundle.probe as any).gpu_name, "Apple M4 Pro");
  assert.match((bundle.probe as any).error, /probe failed/);
  assert.match((bundle.probe as any).error, /\[redacted/);
  // ⚠ The absence assertions above are weaker than they look. PROMPT_ASSIGNMENT
  // is `(prompt|messages|…)\s*[:=]\s*.*$`, so the `prompt=` early in this string
  // truncates everything after it to `[redacted]` in one step: the scrubbed
  // value is literally `probe failed; prompt=[redacted]`. Every later secret in
  // the fixture is therefore absent because it was swallowed, not because its
  // own rule fired. Individual rules are pinned in the test above, on inputs
  // that reach them.
  assert.match((bundle.recent_problems as any)[0].message, /^prompt: \[redacted\]/);
  assert.deepEqual((bundle.recent_problems as any)[0].detail, { model: "qwen3.5-9b", quant: "Q4_K_M" });
});

test("fails closed for cyclic, binary, accessor, and oversized values", () => {
  const members: unknown[] = [];
  members.push(members);
  const accessorProbe: Record<string, unknown> = { gpu_name: "must not survive an accessor sibling" };
  let getterCalls = 0;
  Object.defineProperty(accessorProbe, "error", {
    enumerable: true,
    get() { getterCalls += 1; return "prompt: getter secret"; },
  });

  const cyclic = buildDiagnosticsBundle({
    schema: "idletoken-diagnostics/1",
    probe: { gpu_name: new Uint8Array([115, 101, 99, 114, 101, 116]) },
    cluster: { phase: "ready", members },
    advise: { nodes: 1, models: Array.from({ length: DIAGNOSTIC_LIMITS.maxArrayItems + 1 }, () => ({ id: "x" })) },
  }, settings());
  assert.equal((cyclic.probe as any).gpu_name, "[omitted: unsupported value]");
  assert.deepEqual((cyclic.cluster as any).members, ["[omitted: cyclic value]"]);
  assert.equal((cyclic.advise as any).models, "[omitted: array limit]");
  assert.doesNotThrow(() => JSON.stringify(cyclic));

  const accessor = buildDiagnosticsBundle({ schema: "idletoken-diagnostics/1", probe: accessorProbe }, settings());
  assert.equal(accessor.probe, "[omitted: unsupported value]");
  assert.equal(getterCalls, 0);

  const tooManyKeys = Object.fromEntries(
    Array.from({ length: DIAGNOSTIC_LIMITS.maxObjectKeys + 1 }, (_, i) => [`unknown_${i}`, i]),
  );
  const oversized = buildDiagnosticsBundle({
    schema: "idletoken-diagnostics/1",
    probe: tooManyKeys,
    cluster: { error: "x".repeat(DIAGNOSTIC_LIMITS.maxStringLength + 1) },
  }, settings());
  assert.equal(oversized.probe, "[omitted: object limit]");
  assert.equal((oversized.cluster as any).error, "[omitted: string limit]");

  const overlongKey = "x".repeat(DIAGNOSTIC_LIMITS.maxKeyLength + 1);
  const keyLimited = buildDiagnosticsBundle({
    schema: "idletoken-diagnostics/1",
    probe: { gpu_name: "Apple M4 Pro", [overlongKey]: "ignored but still budgeted" },
  }, settings());
  assert.equal(keyLimited.probe, "[omitted: key limit]");
});

test("settings backup and acceptance bundle derive diagnostics from one settings allowlist", () => {
  const configured = settings({
    modelId: "qwen3.5-9b",
    quant: "Q4_K_M",
    apiToken: "api-secret",
    overflowKey: "overflow-secret",
  });
  const exported = exportableSettings(configured);
  const acceptanceBundle = buildDiagnosticsBundle({ schema: "idletoken-diagnostics/1" }, configured);
  const diagnosticSettings = acceptanceBundle.settings as Record<string, unknown>;

  assert.deepEqual(Object.keys(diagnosticSettings), [...DIAG_SETTING_KEYS]);
  for (const key of DIAG_SETTING_KEYS) assert.deepEqual(diagnosticSettings[key], exported[key]);
  for (const secretKey of ["apiToken", "overflowKey", "platformJwt", "providerKey"]) {
    assert.equal(Object.hasOwn(exported, secretKey), false);
    assert.equal(Object.hasOwn(diagnosticSettings, secretKey), false);
  }
});
