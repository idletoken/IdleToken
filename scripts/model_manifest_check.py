#!/usr/bin/env python3
"""Diff models/<id>.json against the compiled registry in src/common/model.c.

The two are hand-maintained copies of the same numbers. The manifest feeds the
platform catalogue, the pricing table and the client; the registry feeds the
resource planner inside coord/worker. When they drift, nothing crashes — the
cluster is simply sized from numbers the engine disagrees with. That is why
this is a gate and not a comment.

Usage:  python3 scripts/model_manifest_check.py [path/to/model_registry_dump]
        (builds nothing; the caller compiles the dumper)
"""
import json
import os
import subprocess
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# manifest key -> registry key. Only fields that exist on both sides and mean
# exactly the same thing; anything needing interpretation stays out.
SCALARS = [
    ("n_layers", "n_layers"),
    ("n_embd", "n_embd"),
    ("hc_streams", "hc_streams"),
    ("n_vocab", "n_vocab"),
    ("layer_weight_bytes", "layer_weight_bytes"),
    ("shared_weight_bytes", "shared_weight_bytes"),
    ("context_max", "context_max"),
    ("overhead_base_bytes", "overhead_base_bytes"),
    ("default_gguf", "default_gguf"),
]
KV = [
    ("bytes_per_token_per_layer", "kv_bytes_per_token_per_layer"),
    ("state_bytes_per_layer", "state_bytes_per_layer"),
    ("full_attention_interval", "full_attention_interval"),
]


def main(argv):
    dumper = argv[0] if argv else os.path.join(REPO, "build", "model_registry_dump")
    if not os.path.exists(dumper):
        print("model_manifest_check: no dumper at %s" % dumper)
        return 2
    reg = json.loads(subprocess.check_output([dumper]).decode())

    problems = []
    checked = 0
    mdir = os.path.join(REPO, "models")
    for fn in sorted(os.listdir(mdir)):
        if not fn.endswith(".json"):
            continue
        man = json.load(open(os.path.join(mdir, fn)))
        mid = man.get("id")
        if mid not in reg:
            problems.append("%s: id %r has a manifest but no registry entry" % (fn, mid))
            continue
        r = reg[mid]
        checked += 1
        for mk, rk in SCALARS:
            if mk not in man:
                continue
            if man[mk] != r[rk]:
                problems.append("%s: %s manifest=%r registry=%r" % (mid, mk, man[mk], r[rk]))
        for mk, rk in KV:
            if mk in man.get("kv", {}) and man["kv"][mk] != r[rk]:
                problems.append("%s: kv.%s manifest=%r registry=%r"
                                % (mid, mk, man["kv"][mk], r[rk]))
        # `available` gates whether the planner will serve the model at all —
        # a manifest claiming ready while the engine says not (or vice versa)
        # is the difference between "offered to users" and "actually works".
        if bool(man.get("available")) != bool(r["available"]):
            problems.append("%s: available manifest=%r registry=%r"
                            % (mid, man.get("available"), bool(r["available"])))
        # `deployment` decides whether the coordinator will let this model span
        # more than one machine. Both copies must say the same thing, and BOTH
        # must say something: a model added without it is refused at runtime,
        # which is a fine failure mode but a terrible way to discover the typo.
        md, rd = man.get("deployment"), r.get("deployment")
        if md not in ("single-node", "cluster"):
            problems.append("%s: manifest deployment=%r (want 'single-node' or 'cluster')" % (mid, md))
        elif md != rd:
            problems.append("%s: deployment manifest=%r registry=%r" % (mid, md, rd))
        # Integrity gate (threat-model.md, plan A2): a model offered to users
        # (available=true) must pin a SHA-256 for every downloadable file —
        # variants each carry their own; variant-less models pin default_gguf
        # at the top level. Without the hash the client/scripts download
        # unverified, which reopens exactly the hole the gate closed. Models
        # still under curation stay available=false until the hash is recorded
        # (locally renamed files: record it by hand from the merged file).
        if bool(man.get("available")):
            vs = man.get("variants") or []
            if vs:
                for v in vs:
                    if not v.get("sha256"):
                        problems.append("%s[%s]: available but variant has no sha256"
                                        " (run scripts/manifest_sha256.py)" % (mid, v.get("quant")))
            elif not man.get("sha256"):
                problems.append("%s: available but no top-level sha256 for default_gguf"
                                " (run scripts/manifest_sha256.py, or record it by hand)" % mid)
        mv = {v["quant"]: v for v in man.get("variants", [])}
        rv = {v["quant"]: v for v in r.get("variants", [])}
        if set(mv) != set(rv):
            problems.append("%s: variant quants differ manifest=%s registry=%s"
                            % (mid, sorted(mv), sorted(rv)))
        for q in sorted(set(mv) & set(rv)):
            for f in ("layer_weight_bytes", "shared_weight_bytes", "gguf"):
                if mv[q][f] != rv[q][f]:
                    problems.append("%s[%s]: %s manifest=%r registry=%r"
                                    % (mid, q, f, mv[q][f], rv[q][f]))

    # --- The third copy: the client's registry -------------------------------
    # This script used to compare only two, models/*.json against
    # src/common/model.c. On 2026-08-08 a third turned up: the imports and the
    # MANIFESTS array at the top of client/src/models.ts. It was missing
    # qwen3.5-4b/9b/27b/35b-a3b -- the engine could run them and `--advise` listed
    # them as runnable, yet they were absent from the client's dropdown and users
    # could not select them. The README telling people to "pick Qwen3.5-4B" was
    # therefore an empty statement, and no check would have caught it.
    #
    # Only the id sets are compared: details such as byte counts cannot drift,
    # because the client imports the very same JSON. What does drift is precisely
    # whether a given manifest was imported at all.
    ts = os.path.join(REPO, "client", "src", "models.ts")
    if os.path.exists(ts):
        src = open(ts, encoding="utf-8").read()
        m = re.search(r"const MANIFESTS\s*=\s*\[(.*?)\]", src, re.S)
        if not m:
            problems.append("client/src/models.ts: MANIFESTS array not found (did the structure change?)")
        else:
            idents = set(re.findall(r"[A-Za-z_$][\w$]*", m.group(1)))
            imported = dict(re.findall(
                r'import\s+([A-Za-z_$][\w$]*)\s+from\s+"\.\./\.\./models/([^"]+)\.json"', src))
            listed = {f for ident, f in imported.items() if ident in idents}
            have = {fn[:-5] for fn in os.listdir(mdir) if fn.endswith(".json")}
            missing = sorted(have - listed)
            extra = sorted(listed - have)
            if missing:
                problems.append("client/src/models.ts MANIFESTS is missing: %s"
                                " (the engine can run them but the client cannot select them)" % ", ".join(missing))
            if extra:
                problems.append("client/src/models.ts MANIFESTS has extra entries: %s" % ", ".join(extra))

    if problems:
        print("MODEL_MANIFEST_CHECK_FAIL (%d)" % len(problems))
        for p in problems:
            print("  " + p)
        return 1
    print("%d manifests match the compiled registry" % checked)
    print("MODEL_MANIFEST_CHECK_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
