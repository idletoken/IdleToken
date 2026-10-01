#!/usr/bin/env python3
"""Measure paired standalone MTP allocations on one real GPU backend.

Only --idletoken-memory-json consumes the sparse portable-header stand-ins.
Each enabled base precision is measured independently at every product context
and target KV type; the draft always uses f16 KV and depth three.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import time


def artifact_fingerprints(model, variant):
    """Bind cache reuse to every shard, including metadata-only first files."""
    draft = {key: model["draft"].get(key, []) for key in (
        "repo", "revision", "gguf", "bytes", "sha256", "parts")}
    draft["headers"] = model["draft_headers"]
    target = {key: variant[key] for key in (
        "repo", "revision", "gguf", "sha256", "base_headers")}
    return {name + "_artifact_sha256": hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for name, value in (("draft", draft), ("target", target))}


def file_sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True)
    p.add_argument("--inventory", type=pathlib.Path, required=True)
    p.add_argument("--portable", type=pathlib.Path, required=True)
    p.add_argument("--standins", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--backend", choices=("cuda", "metal"), required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--model", action="append")
    args = p.parse_args()
    engine_sha256 = file_sha256(args.engine)
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith("LLAMA_ARG_")}
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("errors"):
        raise SystemExit("refusing inventory with unresolved compatibility errors")
    args.output.mkdir(parents=True, exist_ok=True)

    def standin(model, name):
        source = args.portable / model / name
        dest = args.standins / model / name
        dest.mkdir(parents=True, exist_ok=True)
        entries = json.loads((source / "spec.json").read_text())
        for entry in entries:
            with (dest / entry["name"]).open("wb") as stream:
                stream.write((source / (entry["name"] + ".hdr")).read_bytes())
                stream.truncate(entry["total"])
        return str(dest / entries[0]["name"])

    jobs = []
    for model in inventory["rows"]:
        if args.model and model["model"] not in args.model:
            continue
        draft = standin(model["model"], "draft")
        for variant in model["variants"]:
            if not variant["compatible"] or variant.get("embedded_mtp_layers", 0):
                continue
            target = standin(model["model"], variant["quant"])
            for ctx in (131072, 262144, 1048576):
                for kv in ("f16", "q8_0", "q4_0"):
                    for enabled in (False, True):
                        jobs.append((model, variant, target, draft, ctx, kv, enabled))

    def measure(job):
        model, variant, target, draft, ctx, kv, enabled = job
        name = "%s-%s-%s-%s-%s-%s" % (model["model"], variant["quant"], args.backend,
                                         ctx, kv, "mtp" if enabled else "baseline")
        dest = args.output / (name + ".json")
        fingerprints = artifact_fingerprints(model, variant)
        if dest.exists():
            existing = json.loads(dest.read_text())
            if (existing.get("engine_commit") == inventory["engine_commit"]
                    and existing.get("engine_sha256") == engine_sha256
                    and existing.get("target_sha256") == variant["sha256"]
                    and existing.get("draft_sha256") == model["draft"]["sha256"]
                    and all(existing.get(k) == v for k, v in fingerprints.items())):
                return existing
            raise ValueError("existing measurement provenance differs: " + name)
        spec = ["ngram-mod", "draft-mtp"] if enabled else ["ngram-mod"]
        argv = [args.engine, "--idletoken-managed", "--idletoken-memory-json", "-m", target, "-ngl", "999",
                "--fit", "off", "-c", str(ctx), "-np", "1", "-ctk", kv, "-ctv", kv,
                "-fa", "on", "--spec-type", ",".join(spec), "--spec-draft-n-max", "3",
                "-ctkd", "f16", "-ctvd", "f16"]
        if enabled:
            argv += ["-md", draft, "--spec-draft-device", "MTL0" if args.backend == "metal" else "CUDA0",
                     "--spec-draft-ngl", "all"]
        start = time.monotonic()
        run = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, env=environment)
        if run.returncode:
            (args.output / (name + ".log")).write_bytes(run.stderr)
            raise ValueError(name + ": " + run.stderr[-3000:].decode(errors="replace"))
        row = json.loads(run.stdout)
        if not row.get("no_alloc") or row.get("mtp_external", False) != enabled:
            raise ValueError("engine did not measure requested external MTP mode")
        row.update(model=model["model"], quant=variant["quant"], kv=kv, backend=args.backend,
                   engine_commit=inventory["engine_commit"], spec_types=spec,
                   engine_sha256=engine_sha256,
                   target_sha256=variant["sha256"], draft_sha256=model["draft"]["sha256"],
                   seconds=time.monotonic() - start, command=argv)
        row.update(fingerprints)
        dest.write_text(json.dumps(row) + "\n")
        return row

    rows = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for count, row in enumerate(pool.map(measure, jobs), 1):
            rows.append(row)
            if count % 18 == 0:
                print("%s/%s %s %s" % (count, len(jobs), row["model"], row["quant"]), flush=True)
    if file_sha256(args.engine) != engine_sha256:
        raise SystemExit("engine changed during the allocation measurement")
    (args.output / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("Measured %s paired allocation profiles." % len(rows), flush=True)


if __name__ == "__main__":
    main()
