import { readFileSync } from "node:fs";
import {
  clusterCapacityVerdict,
  MODELS,
  estimateClusterCapacity,
  getManifest,
  getVariant,
  hybridRequirements,
  kvBytesForContext,
  poolRam,
} from "../src/models";
import { PRODUCT_CONTEXT_CAP, modelCtxMax, runtimeResourceBudget } from "../src/settings";

interface OracleCase {
  model: string;
  quant: string;
  ctx: number;
  nodes: number;
  /** "cuda" | "metal". The measured workspace differs between them by up to
   *  22x, so a case is only meaningful with its backend attached. */
  backend: "cuda" | "metal";
  weights: number;
  kv: number;
  overhead: number;
  need: number;
}

const oraclePath = process.argv[2];
if (!oraclePath) throw new Error("usage: resource_estimate_check <C-oracle.json>");
interface WddmCase {
  ram_total: number;
  budget: number;
}

const oracle = JSON.parse(readFileSync(oraclePath, "utf8")) as {
  cases: OracleCase[];
  wddm_cases: WddmCase[];
};
const models = new Map(MODELS.map((m) => [m.id, m]));
let failures = 0;

// Three product windows since 2026-09-02 (docs/ctx-tiers-2026-09.md), 128K the
// default. qwen3-8b is the between-tiers case: its ceiling is 163840, so asking
// for 256K clamps to the ceiling while asking for 128K does not.
if (
  PRODUCT_CONTEXT_CAP !== 1048576 ||
  modelCtxMax("qwen3.8-27b") !== 131072 ||
  modelCtxMax("qwen3.8-27b", 262144) !== 262144 ||
  modelCtxMax("qwen3.8-27b", 1048576) !== 1048576 ||
  modelCtxMax("deepseek-v4-flash") !== 131072 ||
  modelCtxMax("deepseek-v4-flash", 1048576) !== 1048576 ||
  modelCtxMax("qwen3-8b") !== 131072 ||
  modelCtxMax("qwen3-8b", 262144) !== 163840
) {
  console.error(
    "RESOURCE_ESTIMATE_MISMATCH exact 128K/256K/1M product contexts do not respect model ability",
  );
  failures++;
}

// The hardware strip and the runtime-capacity rows intentionally answer two
// different questions. These fixtures pin the second one to the exact cap
// semantics used by DGX/unified Linux, Apple Silicon and discrete Windows.
{
  const GiB = 1024 ** 3;
  const cases = [
    {
      label: "DGX unified balanced budget",
      resources: {
        os: "linux" as const,
        unified_memory: true,
        vram_usable: 115545133056,
        ram_usable: 115545133056,
        ram_expert_usable: 0,
      },
      caps: { maxVramMb: 91860, maxRamMb: 91860 },
      want: { vram_usable: 96322191360, ram_usable: 96322191360, ram_expert_usable: 0 },
    },
    {
      label: "Mac unified custom budget",
      resources: {
        os: "macos" as const,
        unified_memory: true,
        vram_usable: 40 * GiB,
        ram_usable: 40 * GiB,
        // A stale sidecar value still cannot create a second unified pool.
        ram_expert_usable: 20 * GiB,
      },
      caps: { maxVramMb: 32 * 1024, maxRamMb: 32 * 1024 },
      want: { vram_usable: 32 * GiB, ram_usable: 32 * GiB, ram_expert_usable: 0 },
    },
    {
      label: "Windows discrete GPU and expert-RAM budgets",
      resources: {
        os: "windows" as const,
        unified_memory: false,
        vram_usable: 24 * GiB,
        ram_usable: 60 * GiB,
        ram_expert_usable: 32 * GiB,
      },
      caps: { maxVramMb: 20 * 1024, maxRamMb: 28 * 1024 },
      want: { vram_usable: 20 * GiB, ram_usable: 28 * GiB, ram_expert_usable: 28 * GiB },
    },
  ];
  for (const c of cases) {
    const got = runtimeResourceBudget(c.resources, c.caps);
    if (
      got.vram_usable !== c.want.vram_usable ||
      got.ram_usable !== c.want.ram_usable ||
      got.ram_expert_usable !== c.want.ram_expert_usable
    ) {
      console.error(
        `RESOURCE_ESTIMATE_MISMATCH ${c.label}: ` +
        `want=${JSON.stringify(c.want)} client=${JSON.stringify(got)}`,
      );
      failures++;
    }
  }
}

function same(label: string, expected: number, actual: number, c: OracleCase): void {
  if (!Number.isSafeInteger(expected) || !Number.isSafeInteger(actual) || expected !== actual) {
    console.error(
      `RESOURCE_ESTIMATE_MISMATCH ${c.model}/${c.quant || "implicit"} ` +
      `ctx=${c.ctx} nodes=${c.nodes} backend=${c.backend} ${label}: ` +
      `C=${expected} client=${actual}`,
    );
    failures++;
  }
}

for (const c of oracle.cases) {
  const model = models.get(c.model);
  if (!model) {
    console.error(`RESOURCE_ESTIMATE_MISMATCH C registry model absent from client: ${c.model}`);
    failures++;
    continue;
  }
  const manifest = getManifest(c.model);
  const variant = getVariant(c.model, c.quant);
  const weights = variant
    ? variant.layer_weight_bytes + variant.shared_weight_bytes
    : manifest.layer_weight_bytes + manifest.shared_weight_bytes;
  const kv = kvBytesForContext(manifest, c.ctx, c.quant);
  const estimate = estimateClusterCapacity(
    model,
    { vram_usable: 0, ram_usable: 0, unified_memory: false },
    c.ctx,
    c.nodes,
    c.quant,
    c.backend,
  );
  same("weights", c.weights, weights, c);
  same("kv", c.kv, kv, c);
  same("overhead", c.overhead, estimate.needBytes - weights - kv, c);
  same("need", c.need, estimate.needBytes, c);
}

// Both backends must actually appear, or the loop above could be checking one
// of them twice and calling the matrix complete.
for (const want of ["cuda", "metal"]) {
  if (!oracle.cases.some((c) => c.backend === want)) {
    console.error(`RESOURCE_ESTIMATE_MISMATCH oracle has no ${want} cases`);
    failures++;
  }
}
// The native side owns the Windows formula and publishes the per-node result.
// The client must consume that field rather than raw free RAM, and must apply
// it independently to every member before summing the cluster.
if (!Array.isArray(oracle.wddm_cases) || oracle.wddm_cases.length < 14) {
  console.error("RESOURCE_ESTIMATE_MISMATCH native oracle lacks low-RAM or WDDM boundary cases");
  failures++;
} else {
  for (const c of oracle.wddm_cases) {
    const one = poolRam([{
      ramFree: c.ram_total,
      ramExpertFree: c.budget,
      unifiedMemory: false,
    }]);
    if (!one.complete || one.bytes !== c.budget) {
      console.error(
        `RESOURCE_ESTIMATE_MISMATCH WDDM per-node budget: total=${c.ram_total} ` +
        `native=${c.budget} client=${one.bytes} complete=${one.complete}`,
      );
      failures++;
    }
  }
  const a = oracle.wddm_cases.find((c) => c.ram_total === 48 * 1024 ** 3);
  const b = oracle.wddm_cases.find((c) => c.ram_total === 64 * 1024 ** 3);
  if (!a || !b) {
    console.error("RESOURCE_ESTIMATE_MISMATCH native oracle lacks 48/64-GiB WDDM cases");
    failures++;
  } else {
    const pair = poolRam([
      { ramFree: a.ram_total, ramExpertFree: a.budget, unifiedMemory: false },
      { ramFree: b.ram_total, ramExpertFree: b.budget, unifiedMemory: false },
    ]);
    if (!pair.complete || pair.bytes !== a.budget + b.budget) {
      console.error("RESOURCE_ESTIMATE_MISMATCH WDDM reserve was not applied per node");
      failures++;
    }
    const oldWindowsPeer = poolRam([{
      ramFree: a.ram_total,
      unifiedMemory: false,
    }]);
    if (oldWindowsPeer.complete || oldWindowsPeer.bytes !== 0) {
      console.error("RESOURCE_ESTIMATE_MISMATCH missing ramExpertFree was treated as usable RAM");
      failures++;
    }
  }
}
if (clusterCapacityVerdict(88.7, 31.3, 77.3, true) !== "hybrid-check") {
  console.error("RESOURCE_ESTIMATE_MISMATCH pooled MoE RAM was treated as a proved fit");
  failures++;
}
if (clusterCapacityVerdict(88.7, 31.3, 50, true) !== "short") {
  console.error("RESOURCE_ESTIMATE_MISMATCH definite MoE aggregate shortfall was hidden");
  failures++;
}
if (clusterCapacityVerdict(30, 31.3, 0, false) !== "fits") {
  console.error("RESOURCE_ESTIMATE_MISMATCH GPU-only fit became uncertain");
  failures++;
}
if (clusterCapacityVerdict(88.7, 31.3, 0, true, true, false) !== "unknown") {
  console.error("RESOURCE_ESTIMATE_MISMATCH missing expert-RAM report was treated as a shortfall");
  failures++;
}

// Hybrid is two independent requirements. This is the measured DSv4-shaped
// boundary from the Windows pair: 77.340027 GiB page-lockable versus
// 77.265625 GiB of expert weights (about 76 MiB spare). The old card added RAM
// to VRAM and compared that made-up pool with 88.7 GiB; the new calculation
// must retain the complete expert store as the RAM requirement and charge the
// model's own active expert count on the VRAM side.
{
  const GiB = 1024 ** 3;
  const fullGpu = Math.round(88.7 * GiB);
  const experts = Math.round(77.265625 * GiB);
  for (const active of [4, 6, 8, 10]) {
    const hybrid = hybridRequirements(fullGpu, {
      expertBytesTotal: experts,
      nExpert: 256,
      nExpertUsed: active,
      complete: true,
    });
    const transfer = Math.ceil((experts * active) / 256);
    if (
      !hybrid ||
      hybrid.ramNeedBytes !== experts ||
      hybrid.transferBytes !== transfer ||
      hybrid.poolExperts !== active ||
      hybrid.vramNeedBytes !== fullGpu - experts + transfer
    ) {
      console.error(
        `RESOURCE_ESTIMATE_MISMATCH top-${active} Hybrid VRAM/RAM requirements drifted`,
      );
      failures++;
    }
  }
  if (hybridRequirements(fullGpu, {
    expertBytesTotal: experts,
    nExpert: 256,
    nExpertUsed: 6,
    complete: false,
  }) !== null) {
    console.error("RESOURCE_ESTIMATE_MISMATCH incomplete GGUF expert layout was guessed");
    failures++;
  }
}
if (oracle.cases.length < 100) {
  console.error(`RESOURCE_ESTIMATE_MISMATCH oracle is unexpectedly small: ${oracle.cases.length}`);
  failures++;
}
if (failures) {
  console.error(`RESOURCE_ESTIMATE_CHECK_FAIL ${failures}`);
  process.exit(1);
}
console.log(`RESOURCE_ESTIMATE_CHECK_OK ${oracle.cases.length} exact cases`);
