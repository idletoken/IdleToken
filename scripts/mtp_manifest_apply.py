#!/usr/bin/env python3
"""Apply a complete immutable-header MTP inventory and measured graph profiles.

Both manifest JSON and the compiled registry are updated together. Missing
variants, unknown revisions, unmeasured cells, or target-workspace growth abort
before writing any file. This tool never extrapolates between context tiers.
"""
import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
TIERS = ("f16", "q8_0", "q4_0")
CONTEXTS = {131072: "128k", 262144: "256k", 1048576: "1m"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", type=pathlib.Path, required=True)
    ap.add_argument("--memory", type=pathlib.Path, action="append", required=True)
    args = ap.parse_args()
    inventory = json.loads(args.inventory.read_text())
    pin = (ROOT / "scripts/llamacpp-patches/UPSTREAM").read_text().split()[1]
    if inventory.get("schema_version") != 1 or inventory.get("engine_commit") != pin:
        raise SystemExit("inventory does not match the pinned engine")
    if inventory.get("errors"):
        raise SystemExit("inventory has errors; refusing partial capability data")
    rows = {(r["model"], r["quant"]): r for r in inventory["rows"]}
    if len(rows) != len(inventory["rows"]):
        raise SystemExit("duplicate inventory entries")
    measurements = {}
    for path in args.memory:
        for row in json.loads(path.read_text()):
            gpu = [d for d in row["devices"] if not d["host"]]
            if len(gpu) != 1:
                raise SystemExit("profile must measure exactly one GPU backend")
            backend = "metal" if gpu[0]["name"].startswith("MTL") else "cuda" if gpu[0]["name"].startswith("CUDA") else None
            if (not backend or not row.get("no_alloc") or row.get("draft_max") != 3
                    or row.get("seq_slots") != 1 or row.get("mmproj_included")
                    or row.get("expert_cache_included")
                    or row.get("spec_types") != (["ngram-mod", "draft-mtp"]
                                                  if row["mtp"] else ["ngram-mod"])):
                raise SystemExit("measurement does not match the pinned product configuration")
            key = (row["model"], backend, row["ctx_size"], row["kv"], row["quant"], row["mtp"])
            if key in measurements:
                raise SystemExit(f"duplicate profile cell: {key}")
            measurements[key] = row

    registry_path = ROOT / "src/common/model.c"
    registry = registry_path.read_text()
    writes = {}
    expected = set()
    for path in sorted((ROOT / "models").glob("*.json")):
        model = json.loads(path.read_text())
        geometry = set()
        for variant in model["variants"]:
            key = model["id"], variant["quant"]
            expected.add(key)
            if key not in rows:
                raise SystemExit(f"missing inventory cell: {key}")
            row = rows[key]
            if (row["repo"], row["revision"], row["gguf"]) != (
                    variant["repo"], variant["revision"], variant["gguf"]):
                raise SystemExit(f"inventory revision/file mismatch: {key}")
            variant["mtp_layers"] = row["mtp_layers"]
            variant["mtp_weight_bytes"] = row["mtp_weight_bytes"]
            if row["mtp_layers"]:
                geometry.add(row["mtp_kv_bytes_per_token"])
            pattern = (r'(\{[^{}]*\.quant\s*=\s*"' + re.escape(variant["quant"]) +
                       r'"[^{}]*\.gguf\s*=\s*"' + re.escape(variant["gguf"]) + r'"[^{}]*)(\})')
            def update_variant(match):
                prefix = re.sub(r",?\s*\.mtp_(?:layers|weight_bytes)\s*=\s*\d+(?:ull)?", "", match[1]).rstrip()
                if row["mtp_layers"]:
                    prefix += f', .mtp_layers = {row["mtp_layers"]}, .mtp_weight_bytes = {row["mtp_weight_bytes"]}ull'
                return prefix + " }"
            registry, count = re.subn(pattern, update_variant, registry)
            if count != 1:
                raise SystemExit(f"expected exactly one compiled variant for {key}, found {count}")
        if geometry:
            if len(geometry) != 1 or next(iter(geometry)) <= 0:
                raise SystemExit(f"variant draft geometry differs within {model['id']}")
            if model.get("mtp_draft") and model["mtp_draft"]["kv_bytes_per_token"] != next(iter(geometry)):
                raise SystemExit(f"embedded and standalone draft geometry differs within {model['id']}")
            model["mtp_kv_bytes_per_token"] = next(iter(geometry))
            for backend in ("cuda", "metal"):
                for ctx, tag in CONTEXTS.items():
                    gpu_values, host_values = [], []
                    for kv in TIERS:
                        key = model["id"], backend, ctx, kv
                        candidates = [v for k, v in measurements.items() if k[:4] == key and k[-1]]
                        expected_quants = {v["quant"] for v in model["variants"] if v["mtp_layers"]}
                        if {v["quant"] for v in candidates} != expected_quants:
                            raise SystemExit(f"incomplete per-precision baseline/MTP measurements: {key}")
                        gpu_max = host_max = 0
                        for mtp in candidates:
                            base = measurements.get(key + (mtp["quant"], False))
                            if not base:
                                raise SystemExit(f"missing paired baseline: {key}, {mtp['quant']}")
                            for device in mtp["devices"]:
                                before = next((d for d in base["devices"] if d["name"] == device["name"]), None)
                                if not before or device["target_compute_bytes"] > before["target_compute_bytes"]:
                                    raise SystemExit(f"target workspace grew with MTP: {key}; add variant-specific target accounting")
                            gpu_max = max(gpu_max, sum(d["draft_compute_bytes"] for d in mtp["devices"] if not d["host"]))
                            host_max = max(host_max, sum(d["draft_compute_bytes"] for d in mtp["devices"] if d["host"]))
                            draft_kv = sum(d["draft_context_bytes"] for d in mtp["devices"])
                            if draft_kv != ctx * model["mtp_kv_bytes_per_token"]:
                                raise SystemExit(f"draft cache geometry does not match engine measurement: {key}")
                        gpu_values.append(gpu_max)
                        host_values.append(host_max)
                    # A model may have an admitted standalone head for other
                    # precisions. Keep that measured envelope when refreshing
                    # the embedded-head allocation evidence.
                    for kind, values in (("compute", gpu_values), ("host_compute", host_values)):
                        field = f"mtp_{kind}_bytes_{tag}_{backend}"
                        previous = model.get(field, [0, 0, 0]) if model.get("mtp_draft") else [0, 0, 0]
                        model[field] = [max(old, new) for old, new in zip(previous, values)]
            pattern = r'([ \t]*\.id\s*=\s*"' + re.escape(model["id"]) + r'"[\s\S]*?\n    \})'
            match = re.search(pattern, registry)
            if not match:
                raise SystemExit(f"compiled model missing: {model['id']}")
            block = match[0]
            block = re.sub(r"\s*\.mtp_\w+\s*=\s*(?:\{[^}]*\}|\d+ull),", "", block)
            fields = [f'.mtp_kv_bytes_per_token = {model["mtp_kv_bytes_per_token"]}ull,']
            for backend in ("cuda", "metal"):
                for tag in CONTEXTS.values():
                    for kind in ("compute", "host_compute"):
                        key = f"mtp_{kind}_bytes_{tag}_{backend}"
                        fields.append(f'.{key} = {{ ' + ", ".join(f"{v}ull" for v in model[key]) + " },")
            block = block[:-6] + "\n        " + "\n        ".join(fields) + "\n    }"
            registry = re.sub(pattern, lambda _: block, registry, count=1)
        writes[path] = json.dumps(model, indent=2, ensure_ascii=False) + "\n"
    if set(rows) != expected:
        raise SystemExit("inventory contains entries outside the current curated variants")
    writes[registry_path] = registry
    for path, body in writes.items():
        path.write_text(body)
    print(f"Updated {len(expected)} verified variants and {len(writes) - 1} manifests.")


if __name__ == "__main__":
    main()
