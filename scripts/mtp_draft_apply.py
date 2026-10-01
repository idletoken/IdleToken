#!/usr/bin/env python3
"""Admit audited standalone MTP assets only after complete CUDA/Metal profiles.

Update JSON manifests and native registry together. Embedded MTP profiles remain
in the per-model envelope. Any missing precision/context/KV/backend pair aborts
before a file is written; no capability is inferred from another precision.
"""
import argparse
import json
import pathlib
import re

from mtp_draft_measure import artifact_fingerprints

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTEXTS = {131072: "128k", 262144: "256k", 1048576: "1m"}
KV = ("f16", "q8_0", "q4_0")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=pathlib.Path, required=True)
    p.add_argument("--memory", type=pathlib.Path, action="append", required=True)
    p.add_argument("--report", type=pathlib.Path, required=True)
    args = p.parse_args()
    inventory = json.loads(args.inventory.read_text())
    pin = (ROOT / "scripts/llamacpp-patches/UPSTREAM").read_text().split()[1]
    if (inventory.get("schema_version") != 1
            or inventory.get("engine_commit") != pin or inventory.get("errors")):
        raise SystemExit("inventory has errors or does not match pinned engine")
    model_ids = [row["model"] for row in inventory["rows"]]
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise SystemExit("inventory has missing or duplicate model entries")
    cells, engines = {}, {}
    for file in args.memory:
        for row in json.loads(file.read_text()):
            key = (row["model"], row["quant"], row["backend"], row["ctx_size"], row["kv"], row["mtp"])
            if key in cells:
                raise SystemExit("duplicate allocation profile: " + repr(key))
            gpu = [d for d in row["devices"] if not d["host"]]
            if (row.get("backend") not in ("cuda", "metal")
                    or row.get("engine_commit") != pin or not row.get("no_alloc")
                    or not re.fullmatch(r"[0-9a-f]{64}", row.get("engine_sha256", ""))
                    or row.get("mtp_external", False) != row["mtp"]
                    or row.get("draft_max") != 3 or row.get("seq_slots") != 1
                    or row.get("mmproj_included") or row.get("expert_cache_included")
                    or len(gpu) != 1 or row["spec_types"] != (["ngram-mod", "draft-mtp"] if row["mtp"] else ["ngram-mod"])
                    or not gpu[0]["name"].startswith("MTL" if row["backend"] == "metal" else "CUDA")):
                raise SystemExit("allocation profile violates product configuration")
            cells[key] = row
            engines.setdefault(row["backend"], set()).add(row["engine_sha256"])
    if any(len(values) != 1 for values in engines.values()):
        raise SystemExit("a backend matrix mixes different engine builds")
    registry_path = ROOT / "src/common/model.c"
    registry = registry_path.read_text()
    writes, expected, summaries = {}, set(), []
    for entry in inventory["rows"]:
        path = ROOT / "models" / (entry["model"] + ".json")
        model = json.loads(path.read_text())
        audited = {v["quant"]: v for v in entry["variants"]}
        if len(audited) != len(model["variants"]) or set(audited) != {v["quant"] for v in model["variants"]}:
            raise SystemExit("audit does not cover every selected precision")
        descriptor = dict(entry["draft"])
        if model.get("mtp_kv_bytes_per_token", descriptor["kv_bytes_per_token"]) != descriptor["kv_bytes_per_token"]:
            raise SystemExit("embedded and standalone MTP cache geometry differ")
        symbol = re.sub(r"[^A-Z0-9]", "_", model["id"].upper()) + "_MTP_DRAFT"
        model_pattern = r'([ \t]*\.id\s*=\s*"' + re.escape(model["id"]) + r'"[\s\S]*?\n    \})'
        match = re.search(model_pattern, registry)
        if not match:
            raise SystemExit("native model entry missing")
        block = match[0]
        maximum_weight = descriptor["weight_bytes"]
        external = [v for v in model["variants"] if not v.get("mtp_layers", 0)]
        for variant in model["variants"]:
            proof = audited[variant["quant"]]
            if not proof["compatible"] or any(proof[k] != variant[k] for k in ("repo", "gguf", "revision", "sha256")):
                raise SystemExit("audit provenance no longer matches manifest")
            variant["mtp_draft_compatible"] = True
            variant_pattern = (r'(\{[^{}]*\.quant\s*=\s*"' + re.escape(variant["quant"]) +
                               r'"[^{}]*\.gguf\s*=\s*"' + re.escape(variant["gguf"]) + r'"[^{}]*)(\})')
            def update_variant(found):
                old = re.sub(r",?\s*\.mtp_draft_compatible\s*=\s*\d+", "", found[1]).rstrip()
                return old + ", .mtp_draft_compatible = 1 }"
            registry, count = re.subn(variant_pattern, update_variant, registry)
            if count != 1:
                raise SystemExit("native precision entry is missing or ambiguous")
        for backend in ("cuda", "metal"):
            for ctx, tag in CONTEXTS.items():
                gpu_values, host_values = [], []
                for kv in KV:
                    draft_gpu = draft_host = 0
                    for variant in external:
                        prefix = (model["id"], variant["quant"], backend, ctx, kv)
                        if any(prefix + (enabled,) not in cells for enabled in (False, True)):
                            raise SystemExit("missing paired allocation profile: " + repr(prefix))
                        baseline, mtp = (cells[prefix + (enabled,)] for enabled in (False, True))
                        expected.update(prefix + (enabled,) for enabled in (False, True))
                        for row in (baseline, mtp):
                            if row["target_sha256"] != variant["sha256"] or row["draft_sha256"] != descriptor["sha256"]:
                                raise SystemExit("allocation profile artifact hashes differ")
                            if any(row.get(k) != v for k, v in artifact_fingerprints(
                                    entry, audited[variant["quant"]]).items()):
                                raise SystemExit("allocation profile full-shard fingerprints differ")
                        if sum(d["draft_context_bytes"] for d in mtp["devices"]) != ctx * descriptor["kv_bytes_per_token"]:
                            raise SystemExit("measured standalone draft KV geometry differs")
                        target_context_delta = sum(d["target_context_bytes"] for d in mtp["devices"]) - sum(d["target_context_bytes"] for d in baseline["devices"])
                        recurrent = model["n_layers"] - model["n_layers"] // model["kv"]["full_attention_interval"]
                        expected_rollback = 3 * recurrent * model["kv"]["state_bytes_per_layer"]
                        if target_context_delta != expected_rollback:
                            raise SystemExit("measured target rollback differs from native planner geometry")
                        gpu_workspace = host_workspace = gpu_model_delta = 0
                        for device in mtp["devices"]:
                            before = next((d for d in baseline["devices"] if d["name"] == device["name"]), None)
                            if before is None:
                                raise SystemExit("paired allocation devices differ")
                            extra_model = device["model_bytes"] - before["model_bytes"]
                            workspace = device["draft_compute_bytes"] + max(0, device["target_compute_bytes"] - before["target_compute_bytes"])
                            if device["host"]:
                                if extra_model:
                                    raise SystemExit("standalone MTP weights escaped local GPU accounting")
                                host_workspace += workspace
                            else:
                                gpu_model_delta += extra_model
                                gpu_workspace += workspace
                        if gpu_model_delta < descriptor["weight_bytes"]:
                            raise SystemExit("measured GPU allocation omits draft weights")
                        maximum_weight = max(maximum_weight, gpu_model_delta)
                        draft_gpu, draft_host = max(draft_gpu, gpu_workspace), max(draft_host, host_workspace)
                    gpu_values.append(draft_gpu)
                    host_values.append(draft_host)
                for kind, values in (("compute", gpu_values), ("host_compute", host_values)):
                    field = "mtp_%s_bytes_%s_%s" % (kind, tag, backend)
                    old = model.get(field, [0, 0, 0])
                    model[field] = [max(a, b) for a, b in zip(old, values)]
                    block = re.sub(r"\n\s*\." + field + r"\s*=\s*\{[^}]*\},", "", block)
        descriptor["weight_bytes"] = maximum_weight
        model["mtp_draft"] = descriptor
        model["mtp_kv_bytes_per_token"] = descriptor["kv_bytes_per_token"]
        block = re.sub(r"\n\s*\.mtp_(?:draft|kv_bytes_per_token)\s*=\s*[^,]+,", "", block)
        fields = ["        .mtp_draft = &%s," % symbol,
                  "        .mtp_kv_bytes_per_token = %sull," % descriptor["kv_bytes_per_token"]]
        for backend in ("cuda", "metal"):
            for tag in CONTEXTS.values():
                for kind in ("compute", "host_compute"):
                    field = "mtp_%s_bytes_%s_%s" % (kind, tag, backend)
                    fields.append("        .%s = { %s }," % (field, ", ".join(str(v) + "ull" for v in model[field])))
        block = block[:-6] + "\n" + "\n".join(fields) + "\n    }"
        registry = re.sub(model_pattern, lambda _: block, registry, count=1)
        writes[path] = json.dumps(model, indent=2, ensure_ascii=False) + "\n"
        summaries.append({"model": model["id"], "variants": len(audited), "external_variants": len(external),
                          "draft_weight_bytes": maximum_weight, "draft_file_bytes": descriptor["bytes"]})
    if expected != set(cells):
        raise SystemExit("allocation matrix includes unknown or unused cells")
    start, end = "/* BEGIN CURATED STANDALONE MTP ASSETS */", "/* END CURATED STANDALONE MTP ASSETS */"
    registry = re.sub(re.escape(start) + r"[\s\S]*?" + re.escape(end) + r"\n\n", "", registry)
    lines = [start]
    # A partial refresh must retain descriptors admitted by earlier complete
    # measurements. Rebuild definitions from the resulting catalog, not only
    # this invocation's selected inventory rows.
    definitions = []
    for path in sorted((ROOT / "models").glob("*.json")):
        model = json.loads(writes.get(path, path.read_text()))
        if model.get("mtp_draft"):
            symbol = re.sub(r"[^A-Z0-9]", "_", model["id"].upper()) + "_MTP_DRAFT"
            definitions.append((symbol, model["mtp_draft"]))
    for symbol, asset in definitions:
        if asset.get("parts"):
            lines.append("static const idletoken_model_mtp_part %s_PARTS[] = {" % symbol)
            for part in asset["parts"]:
                lines.append('    { .gguf = "%s", .bytes = %sull, .sha256 = "%s" },' % (part["file"], part["bytes"], part["sha256"]))
            lines.append("};")
        lines.append("static const idletoken_model_mtp_draft %s = {" % symbol)
        for field in ("repo", "gguf", "sha256", "revision"):
            lines.append('    .%s = "%s",' % (field, asset[field]))
        for field in ("bytes", "weight_bytes", "kv_bytes_per_token"):
            lines.append("    .%s = %sull," % (field, asset[field]))
        lines.append("    .layers = %s," % asset["layers"])
        if asset.get("parts"):
            lines.append("    .parts = %s_PARTS, .n_parts = %s," % (symbol, len(asset["parts"])))
        lines.append("};\n")
    lines.extend([end, "", ""])
    registry = registry.replace("static const idletoken_model_spec MODELS[] = {", "\n".join(lines) + "static const idletoken_model_spec MODELS[] = {", 1)
    writes[registry_path] = registry
    for path, text in writes.items():
        path.write_text(text)
    args.report.write_text(json.dumps({"engine_commit": pin, "engine_sha256": {k: next(iter(v)) for k, v in engines.items()},
                                      "allocation_profiles": len(cells), "models": summaries}, indent=2) + "\n")
    print("Admitted %s assets after %s paired allocation profiles." % (len(summaries), len(cells)))


if __name__ == "__main__":
    main()
