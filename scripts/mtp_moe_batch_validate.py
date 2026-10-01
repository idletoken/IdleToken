#!/usr/bin/env python3
"""Run the same independent MoE numeric oracle on Metal or CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SOURCE = Path(__file__).resolve().with_name("mtp_moe_batch_probe.cpp")
HOOK = Path(__file__).resolve().with_name("mtp_moe_batch_probe.cmake")
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "mtp-moe-batch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe")
    parser.add_argument("--device")
    parser.add_argument("--prefix")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--self-test", action="store_true", help="check portable probe/hash references without a GPU")
    args = parser.parse_args()
    if not SOURCE.is_file():
        parser.error("canonical probe is missing: " + str(SOURCE))
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if args.self_test:
        assert HOOK.is_file(), "CMake probe hook is missing"
        assert '${CMAKE_CURRENT_LIST_DIR}/mtp_moe_batch_probe.cpp' in HOOK.read_text()
        print(json.dumps(dict(passed=True, source_file=SOURCE.name, source_sha256=source_sha,
                              hook_file=HOOK.name, output_directory=str(args.out))))
        return
    if not all((args.probe, args.device, args.prefix)):
        parser.error("--probe, --device and --prefix are required for GPU validation")
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    base = {k: v for k, v in os.environ.items() if not k.startswith("GGML_MOE_")}
    base.update(GGML_NO_BACKTRACE="1", GGML_MOE_POOL_DEBUG="1")
    cases = [
        ("default", 40, {}, None, 12),
        ("mapped-off", 40, {"GGML_MOE_MAPPED_READBACK": "0"}, None, 12),
        ("issue-off", 40, {"GGML_MOE_LAYER_ISSUE": "0"}, None, 12),
        ("both-off", 40, {"GGML_MOE_MAPPED_READBACK": "0", "GGML_MOE_LAYER_ISSUE": "0"}, None, 12),
        ("pinned-dma", 40, {"GGML_MOE_MAPPED_READBACK": "0", "GGML_MOE_PINNED_READBACK": "1"}, None, 12),
        ("foreign-ids-owner", 40, {}, "cpu-ids", 12),
        ("strided-ids", 40, {}, "strided-ids", 12),
        ("cap-below-active", 7, {}, None, 36),
        ("cap-below-union", 31, {}, None, 24),
        ("cap-exact-union", 32, {}, None, 12),
        ("multi-only", 40, {}, "multi-only", 12),
        ("multi-only-off", 40, {"GGML_MOE_LAYER_ISSUE": "0"}, "multi-only", 12),
    ]
    expected_digests = {}
    records = []
    for name, cap, extra, option, expected_bypass in cases:
        command = [args.probe, args.device] + ([option] if option else [])
        env = {**base, "GGML_MOE_POOL_EXPERTS": str(cap), **extra}
        p = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        stem = args.prefix + "-" + name
        (OUT / (stem + ".stdout.log")).write_text(p.stdout)
        (OUT / (stem + ".stderr.log")).write_text(p.stderr)
        assert p.returncode == 0, (name, p.returncode, p.stderr[-2000:])
        done = re.search(r"MOE_BATCH_PROBE_OK cases=36 .*max_abs=([^ ]+) digest=([0-9a-f]+)", p.stdout)
        assert done, (name, p.stdout)
        digest = done.group(2)
        group = "multi-only" if option == "multi-only" else "mixed-shapes"
        expected_digests.setdefault(group, digest)
        assert digest == expected_digests[group], (name, digest, expected_digests[group])
        pools = re.findall(r"moe-pool (.*): cap (\d+), hits (\d+), misses (\d+), prefetched (\d+), bypassed batches (\d+)", p.stderr)
        assert len(pools) == 3, (name, pools)
        for weight, actual_cap, hits, misses, prefetched, bypass in pools:
            assert int(actual_cap) == cap, (name, actual_cap, cap)
            assert int(bypass) == expected_bypass, (name, bypass, expected_bypass)
        prefetch = sum(int(pool[4]) for pool in pools)
        if name in ("issue-off", "both-off", "cap-below-active", "multi-only-off"):
            assert prefetch == 0, (name, prefetch)
        if name in ("default", "multi-only"):
            assert prefetch > 0, "default path never issued siblings"
        if args.device.startswith("CUDA") and name == "default":
            assert re.search(r"moe-readback: .*kind=ids bytes=4096 .*mapped=1", p.stderr)
        records.append(dict(case=name, cap=cap, environment=extra, option=option,
                            max_abs=float(done.group(1)), output_digest=digest,
                            prefetched_experts=prefetch,
                            bypassed_batches_per_weight=expected_bypass, passed=True))
        print("PASS", args.prefix, name, "prefetched", prefetch, flush=True)
    by_name = {item["case"]: item for item in records}
    assert (by_name["default"]["prefetched_experts"] >
            by_name["foreign-ids-owner"]["prefetched_experts"]), "multi-token owner guard was not exercised"
    result = dict(passed=True, device=args.device,
                  source_sha256=source_sha,
                  probe_sha256=hashlib.sha256(Path(args.probe).read_bytes()).hexdigest(),
                  cases=records,
                  scope="Real owner-GPU GGML MUL_MAT_ID graphs against a host dot-product oracle; identical output across existing optimization-off controls, context shape changes, strided IDs, owner mismatch and active/union capacity boundaries. These are correctness/branch controls, not model speed measurements.")
    (OUT / (args.prefix + "-validation.json")).write_text(json.dumps(result, indent=2) + "\n")
    print("MOE_SMALL_BATCH_VALIDATION_OK cases=12 graphs=432")


if __name__ == "__main__":
    main()
