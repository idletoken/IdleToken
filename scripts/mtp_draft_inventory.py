#!/usr/bin/env python3
"""Audit immutable standalone MTP assets against every selected base precision.

The input contains explicitly curated candidates, not model-name heuristics.
This only downloads bounded GGUF headers. It preserves complete tokenizer value
hashes (never truncated vocabulary samples), checks the complete draft tensor
directory against a known integrated MTP export, and emits portable headers for
the pinned engine's no_alloc allocation measurements. It does not enable assets.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import re
import time
import urllib.request
from urllib.parse import quote

import measure_gguf as mg
import mtp_manifest_inventory as mi

ROOT = pathlib.Path(__file__).resolve().parents[1]


def source_index(asset, cache):
    revision = asset["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("artifact revision must be immutable")
    url = "%s/api/models/%s/revision/%s?blobs=true" % (
        mg.HF_ENDPOINT, quote(asset["repo"], safe="/"), revision)
    path = cache / (hashlib.sha256(url.encode()).hexdigest() + ".source.json")
    if path.exists():
        info = json.loads(path.read_text())
    else:
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "idletoken-mtp-inventory/1"})
                with mg._OPENER.open(request, timeout=90) as response:
                    info = json.load(response)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
        path.write_text(json.dumps(info))
    if info.get("sha") != revision:
        raise ValueError("source API resolved a different revision")
    return {entry["rfilename"]: entry for entry in info["siblings"]}


def fingerprint(path):
    reader = mg.Reader(path.read_bytes())
    if reader.take(4) != b"GGUF" or reader.u32() != 3:
        raise ValueError("expected GGUF v3")
    reader.u64()
    count = reader.u64()
    values = {}
    for _ in range(count):
        key, kind = reader.string(), reader.u32()
        start = reader.i
        reader.skip_value(kind)
        values[key] = {"type": kind, "sha256": hashlib.sha256(
            reader.b[start:reader.i]).hexdigest()}
    return values


def fetch_set(asset, cache, portable):
    entries = [{"file": asset["gguf"], "bytes": asset["bytes"] - sum(
        p["bytes"] for p in asset.get("parts", [])), "sha256": asset["sha256"]}]
    entries.extend(asset.get("parts", []))
    sources = source_index(asset, cache)
    headers, specs = [], []
    portable.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(entries):
        source = sources.get(entry["file"], {})
        if (source.get("lfs", {}).get("sha256") != entry["sha256"]
                or source.get("size") != entry["bytes"]):
            raise ValueError("immutable source hash or size mismatch: " + entry["file"])
        header = mi.fetch(asset["repo"], asset["revision"], entry["file"], cache, small=i > 0)
        if header["file_bytes"] != entry["bytes"]:
            raise ValueError("immutable file size mismatch: " + entry["file"])
        raw = (cache / (header["key"] + ".hdr")).read_bytes()
        # A metadata-only shard may end before the next aligned tensor offset.
        if (len(raw) != min(header["data_start"], header["file_bytes"])
                or hashlib.sha256(raw).hexdigest() != header["header_sha256"]):
            raise ValueError("cached GGUF header fingerprint mismatch: " + entry["file"])
        name = pathlib.Path(entry["file"]).name
        (portable / (name + ".hdr")).write_bytes(raw)
        specs.append({"name": name, "total": header["file_bytes"], "header": len(raw)})
        headers.append(header)
    (portable / "spec.json").write_text(json.dumps(specs))
    return headers


def tensor_map(headers):
    result = {}
    for header in headers:
        for name, dims, _ in header["tensors"]:
            if name in result:
                raise ValueError("duplicate tensor: " + name)
            result[name] = list(dims)
    return result


def audit_candidate(candidate, cache, portable):
    model = json.loads((ROOT / "models" / (candidate["model"] + ".json")).read_text())
    descriptor = candidate["draft"]
    draft = fetch_set(descriptor, cache, portable / model["id"] / "draft")
    draft_meta = draft[0]["metadata"]
    arch = draft_meta["general.architecture"]
    layers = draft_meta.get(arch + ".nextn_predict_layers", 0)
    blocks = draft_meta.get(arch + ".block_count", 0)
    if arch not in mi.MTP_ARCHES or layers != 1 or blocks != model["n_layers"] + 1:
        raise ValueError("draft architecture or NextN geometry mismatch")
    draft_tensors = tensor_map(draft)
    if draft_tensors.get("token_embd.weight") != [model["n_embd"], model["n_vocab"]]:
        raise ValueError("standalone embedding geometry does not match the curated model")
    if draft_meta.get("general.license", "").lower() != model["license"].lower():
        raise ValueError("standalone GGUF license does not match the source model")
    if "blk.0.attn_norm.weight" in draft_tensors:
        raise ValueError("asset contains a full target stack, not a standalone draft")
    reference = fetch_set(candidate["reference"], cache, portable / model["id"] / "reference")
    prefix = "blk.%d." % model["n_layers"]
    reference_tensors = {k: v for k, v in tensor_map(reference).items() if k.startswith(prefix)}
    if not reference_tensors or any(draft_tensors.get(k) != v for k, v in reference_tensors.items()):
        raise ValueError("standalone draft is missing or changes a complete integrated NextN tensor")
    unexpected = [k for k in draft_tensors if k.startswith("blk.") and k not in reference_tensors]
    if unexpected:
        raise ValueError("unexpected draft tensors: " + repr(unexpected))
    for name in ("token_embd.weight", "output_norm.weight"):
        if name not in draft_tensors:
            raise ValueError("missing standalone shared tensor " + name)
    draft_fp = fingerprint(cache / (draft[0]["key"] + ".hdr"))
    # Drafting consumes target token IDs directly. The padding token is not
    # inserted by this path; an absent add_bos=false has the same pinned-engine
    # default. Keep both differences in the evidence instead of hiding them.
    token_exceptions = {"tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token"}
    rows = []
    for variant in model["variants"]:
        first = mi.fetch(variant["repo"], variant["revision"], variant["gguf"], cache)
        asset = dict(variant, bytes=first["file_bytes"] + sum(p["bytes"] for p in variant.get("parts", [])))
        base = fetch_set(asset, cache, portable / model["id"] / variant["quant"])
        meta = first["metadata"]
        if meta["general.architecture"] != arch or meta[arch + ".block_count"] - meta.get(arch + ".nextn_predict_layers", 0) != model["n_layers"]:
            raise ValueError("target architecture mismatch: " + variant["quant"])
        fp = fingerprint(cache / (first["key"] + ".hdr"))
        token_keys = {k for k in fp.keys() | draft_fp.keys() if k.startswith("tokenizer.ggml.")}
        token_diff = [k for k in sorted(token_keys) if fp.get(k) != draft_fp.get(k)]
        if any(k not in token_exceptions for k in token_diff):
            raise ValueError("token IDs or tokenizer rules differ: " + repr(token_diff))
        false_value = {"type": 7, "sha256": hashlib.sha256(b"\0").hexdigest()}
        if any(values.get("tokenizer.ggml.add_bos_token", false_value) != false_value
               for values in (fp, draft_fp)):
            raise ValueError("this audit only admits the pinned qwen no-BOS tokenizer")
        geometry_diff = [k for k in sorted(meta.keys() | draft_meta.keys())
                         if k.startswith(arch + ".") and k not in (
                             arch + ".nextn_predict_layers", arch + ".block_count")
                         and meta.get(k) != draft_meta.get(k)]
        if geometry_diff:
            raise ValueError("target and draft architecture parameters differ: " + repr(geometry_diff))
        base_tensors = tensor_map(base)
        shared = {}
        for name in ("token_embd.weight", "output_norm.weight", "output.weight"):
            target_shape = base_tensors.get(name, base_tensors.get("token_embd.weight") if name == "output.weight" else None)
            draft_shape = draft_tensors.get(name, draft_tensors.get("token_embd.weight") if name == "output.weight" else None)
            if target_shape != draft_shape:
                raise ValueError("shared tensor shape differs: " + name)
            shared[name] = target_shape
        rows.append({"quant": variant["quant"], "repo": variant["repo"],
                     "revision": variant["revision"], "gguf": variant["gguf"],
                     "sha256": variant["sha256"], "compatible": True,
                     "embedded_mtp_layers": variant.get("mtp_layers", 0),
                     "base_headers": [p["header_sha256"] for p in base],
                     "tokenizer_differences_allowed": token_diff,
                     "shared_shapes": shared})
    weight_bytes = sum(max(0, h["file_bytes"] - h["data_start"]) for h in draft)
    key = arch + ".attention."
    kv = draft_meta[key + "head_count_kv"] * (draft_meta[key + "key_length"] + draft_meta[key + "value_length"]) * 2
    descriptor = dict(descriptor, layers=layers, weight_bytes=weight_bytes, kv_bytes_per_token=kv)
    return {"model": model["id"], "draft": descriptor, "license": candidate["license"],
            "gguf_license": draft_meta["general.license"],
            "source_license_url": candidate["source_license_url"],
            "draft_headers": [p["header_sha256"] for p in draft],
            "draft_tensors": draft_tensors, "nextn_tensor_count": len(reference_tensors),
            "tokenizer_fingerprints": {k: v for k, v in draft_fp.items() if k.startswith("tokenizer.ggml.")},
            "variants": rows, "portable": str(portable / model["id"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=pathlib.Path, required=True)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--portable", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(args.candidates.read_text())
    rows, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        pending = {pool.submit(audit_candidate, c, args.cache, args.portable): c["model"] for c in candidates}
        for result in concurrent.futures.as_completed(pending):
            model = pending[result]
            try:
                rows.append(result.result())
                print(model + ": compatible", flush=True)
            except Exception as error:
                errors.append({"model": model, "error": str(error)})
                print(model + ": " + str(error), flush=True)
    output = {"schema_version": 1, "engine_commit": mi.ENGINE_COMMIT,
              "rows": sorted(rows, key=lambda r: r["model"]), "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
