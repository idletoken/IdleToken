import { readFileSync } from "node:fs";
import {
  MODELS,
  estimateClusterCapacity,
  getManifest,
  getVariant,
  kvBytesForContext,
} from "../src/models";
import { PRODUCT_CONTEXT_CAP, modelCtxMax } from "../src/settings";

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
const oracle = JSON.parse(readFileSync(oraclePath, "utf8")) as { cases: OracleCase[] };
const models = new Map(MODELS.map((m) => [m.id, m]));
let failures = 0;

if (
  PRODUCT_CONTEXT_CAP !== 1048576 ||
  modelCtxMax("qwen3.8-27b") !== 262144 ||
  modelCtxMax("qwen3.8-27b", true) !== 1048576 ||
  modelCtxMax("deepseek-v4-flash") !== 262144 ||
  modelCtxMax("deepseek-v4-flash", true) !== 1048576 ||
  modelCtxMax("qwen3-8b") !== 163840
) {
  console.error(
    "RESOURCE_ESTIMATE_MISMATCH exact 256K/1M product contexts do not respect model ability",
  );
  failures++;
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
if (oracle.cases.length < 100) {
  console.error(`RESOURCE_ESTIMATE_MISMATCH oracle is unexpectedly small: ${oracle.cases.length}`);
  failures++;
}
if (failures) {
  console.error(`RESOURCE_ESTIMATE_CHECK_FAIL ${failures}`);
  process.exit(1);
}
console.log(`RESOURCE_ESTIMATE_CHECK_OK ${oracle.cases.length} exact cases`);
