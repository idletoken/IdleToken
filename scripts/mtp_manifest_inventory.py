#!/usr/bin/env python3
"""Inventory MTP tensors at the immutable revisions in the curated manifests.

Only GGUF headers are fetched. Cached JSON records retain tensor offsets and
metadata; positive variants also retain portable headers for no_alloc memory
measurement. No tensor data is required or accepted as measurement evidence.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.request
from urllib.parse import quote

import measure_gguf as mg

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE_COMMIT = (ROOT / "scripts/llamacpp-patches/UPSTREAM").read_text().split()[1]
# Architectures with an actual DECODER_MTP graph in the pinned engine.
MTP_ARCHES = {
    "deepseek2", "deepseek32", "deepseek4", "glm-dsa", "qwen35", "qwen35moe",
    "qwen3next", "hy-v3", "mimo2", "nemotron-h-moe", "step35", "bailingmoe3",
    "cohere2moe",
}


def fetch(repo, revision, name, cache, small=False):
    key = hashlib.sha256(f"{repo}/{revision}/{name}".encode()).hexdigest()
    cached = cache / (key + ".json")
    if cached.exists():
        return json.loads(cached.read_text())
    url = f"{mg.HF_ENDPOINT}/{repo}/resolve/{revision}/{quote(name)}"
    want = 256 << 10 if small else 16 << 20
    while want <= 256 << 20:
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={
                    "Range": f"bytes=0-{want - 1}",
                    "User-Agent": "idletoken-mtp-inventory/1",
                })
                with mg._OPENER.open(req, timeout=90) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or not re.fullmatch(r"bytes 0-\d+/\d+", content_range):
                        raise ValueError("server did not honor bounded header Range")
                    total = int(content_range.rsplit("/", 1)[1])
                    buf = response.read(want + 1)
                    if len(buf) > want:
                        raise ValueError("server returned more bytes than requested")
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
        try:
            tensors, start, metadata = mg.parse_header(buf, keep_kv=True)
        except mg.Short:
            want *= 2
            continue
        # Tokenizer text is irrelevant to capability evidence and huge.
        metadata = {k: v for k, v in metadata.items() if not k.startswith("tokenizer.")}
        result = dict(repo=repo, revision=revision, file=name, file_bytes=total,
                      data_start=start, metadata=metadata, tensors=tensors,
                      header_sha256=hashlib.sha256(buf[:start]).hexdigest(), key=key)
        # Preserve all headers temporarily: the first part may contain no
        # tensor directory; whether MTP exists is decided over the whole set.
        (cache / (key + ".hdr")).write_bytes(buf[:start])
        cached.write_text(json.dumps(result, ensure_ascii=True))
        return result
    raise ValueError(f"header too large: {repo}/{name}")


def inventory(manifest, variant, cache):
    repo = variant.get("repo", manifest.get("repo"))
    revision = variant.get("revision", manifest.get("revision"))
    if not repo or not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{manifest['id']}/{variant.get('quant')}: immutable revision missing")
    name = variant.get("gguf", manifest.get("default_gguf"))
    first = fetch(repo, revision, name, cache)
    meta = first["metadata"]
    arch = meta["general.architecture"]
    blocks = int(meta.get(arch + ".block_count", 0))
    layers = int(meta.get(arch + ".nextn_predict_layers", 0))
    result = dict(model=manifest["id"], quant=variant.get("quant", ""),
                  repo=repo, revision=revision, gguf=name, arch=arch,
                  block_count=blocks, nextn_predict_layers=layers,
                  first_file_bytes=first["file_bytes"], first_header_bytes=first["data_start"],
                  mtp_layers=0, mtp_weight_bytes=0, mtp_kv_bytes_per_token=0,
                  headers=[first["header_sha256"]])
    if layers == 0:
        result["reason"] = "no_nextn_metadata"
        return result
    if not 0 < layers < blocks:
        raise ValueError(f"invalid MTP layer geometry: {result}")
    parts = [first]
    for part in variant.get("parts", []):
        header = fetch(repo, revision, part["file"], cache, small=True)
        if header["file_bytes"] != part["bytes"]:
            raise ValueError(f"immutable shard size differs from manifest: {part['file']}")
        parts.append(header)
    result["headers"] = [p["header_sha256"] for p in parts]
    names = set()
    total = 0
    for part in parts:
        tensors = sorted(part["tensors"], key=lambda t: t[2])
        for i, (name, dims, offset) in enumerate(tensors):
            names.add(name)
            match = re.match(r"blk\.(\d+)\.", name)
            if match and int(match[1]) >= blocks - layers:
                end = tensors[i + 1][2] if i + 1 < len(tensors) else part["file_bytes"] - part["data_start"]
                if end <= offset:
                    raise ValueError(f"invalid tensor offsets: {name}")
                total += end - offset
    required = [f"blk.{i}.nextn.{suffix}.weight"
                for i in range(blocks - layers, blocks)
                for suffix in ("eh_proj", "enorm", "hnorm")]
    missing = [n for n in required if n not in names]
    result["missing_tensors"] = missing
    if missing:
        result["reason"] = "missing_nextn_tensors"
    elif arch not in MTP_ARCHES:
        result["reason"] = "no_pinned_engine_mtp_graph"
    else:
        rank = int(meta.get(arch + ".attention.kv_lora_rank", 0))
        if rank:
            per_layer = (rank + int(meta.get(arch + ".rope.dimension_count", 0))) * 2
        else:
            key = int(meta.get(arch + ".attention.key_length", 0))
            val = int(meta.get(arch + ".attention.value_length", key))
            per_layer = int(meta.get(arch + ".attention.head_count_kv", 0)) * (key + val) * 2
        if per_layer <= 0:
            raise ValueError("supported MTP head has missing draft cache geometry")
        result.update(mtp_layers=layers, mtp_weight_bytes=total,
                      mtp_kv_bytes_per_token=per_layer * layers, reason="supported")
        portable = cache / "portable" / manifest["id"] / variant.get("quant", "default")
        portable.mkdir(parents=True, exist_ok=True)
        spec = []
        for part in parts:
            base = pathlib.Path(part["file"]).name
            (portable / (base + ".hdr")).write_bytes((cache / (part["key"] + ".hdr")).read_bytes())
            spec.append(dict(name=base, total=part["file_bytes"], header=part["data_start"]))
        (portable / "spec.json").write_text(json.dumps(spec))
        result["portable"] = str(portable)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", action="append")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in sorted((ROOT / "models").glob("*.json")):
        manifest = json.loads(path.read_text())
        if args.model and manifest["id"] not in args.model:
            continue
        jobs.extend((manifest, variant) for variant in manifest.get("variants", [manifest]))
    # Probe positive controls first to unblock allocation measurements.
    jobs.sort(key=lambda j: j[0]["id"] not in ("qwen3.8-27b", "glm-5.2"))
    rows, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(inventory, m, v, args.cache): (m, v) for m, v in jobs}
        for future in concurrent.futures.as_completed(pending):
            m, v = pending[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"{len(rows)}/{len(jobs)} {m['id']} {v.get('quant')}: {row['reason']}", flush=True)
            except Exception as exc:
                errors.append(dict(model=m["id"], quant=v.get("quant"), error=str(exc)))
                print(f"ERROR {m['id']} {v.get('quant')}: {exc}", file=sys.stderr, flush=True)
            args.output.write_text(json.dumps(dict(schema_version=1, engine_commit=ENGINE_COMMIT,
                rows=sorted(rows, key=lambda r:(r["model"],r["quant"])), errors=errors), indent=2)+"\n")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
