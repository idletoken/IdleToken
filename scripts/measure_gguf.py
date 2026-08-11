#!/usr/bin/env python3
"""Measure a remote GGUF's layer / shared weight bytes without downloading it.

`models/<id>.json` and the engine registry (src/common/model.c) both need
`layer_weight_bytes` and `shared_weight_bytes` per precision. Those numbers
drive the fit judgment, the split, and the capability table -- and they are NOT
guessable: on 2026-08-11 four of qwen3-8b's five variants turned out to be
round numbers somebody estimated, 0.4-0.6 GB high each, and one of them
(BF16) named a file that does not exist in the repo at all. Hence this script:
measuring is now cheaper than guessing.

How it works: a GGUF's whole tensor directory lives in the file header, so one
HTTP Range request for the first few MiB is enough to see every tensor's name
and offset. Sizes come from the gaps between consecutive offsets, and the last
tensor's size from the file size. Tensors named `blk.<N>.*` are per-layer
weights; everything else (embeddings, output head, norms) is shared.

    scripts/measure_gguf.py unsloth/Qwen3.5-4B-GGUF Q5_K_M Q6_K
    scripts/measure_gguf.py unsloth/Qwen3.5-4B-GGUF --all-files

Self-check: `layer + shared + data_start == file_size` must hold exactly. It is
not a sanity heuristic -- it is the whole reason the result can be trusted, so
a mismatch is a hard error rather than a warning.

Two traps this script exists to avoid (both cost an afternoon once):
  * Do NOT take the file size from `curl -I`: after the redirect to the CDN you
    get the Content-Length of the redirect body (a value like 1465), which makes
    the last tensor come out negative. The size comes from the HF API instead.
  * `resolve/main/<file>` is a 302 to the CDN, so the Range request must follow
    redirects.
"""
import argparse
import json
import struct
import sys
import urllib.error
import urllib.request

HF_API = "https://huggingface.co/api/models/{repo}?blobs=true"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{name}"

# GGUF metadata value types, and how many bytes the fixed-width ones take.
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
T_STRING, T_ARRAY = 8, 9


class Short(Exception):
    """The header ran past what we fetched; caller refetches a bigger prefix."""


class Reader:
    def __init__(self, buf):
        self.b, self.i = buf, 0

    def take(self, n):
        if self.i + n > len(self.b):
            raise Short()
        out = self.b[self.i : self.i + n]
        self.i += n
        return out

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def skip_value(self, t):
        if t in FIXED:
            self.take(FIXED[t])
        elif t == T_STRING:
            self.string()
        elif t == T_ARRAY:
            et, n = self.u32(), self.u64()
            if et in FIXED:
                self.take(FIXED[et] * n)  # bulk-skip; arrays here reach millions of entries
            else:
                for _ in range(n):
                    self.skip_value(et)
        else:
            raise ValueError(f"unknown GGUF value type {t}")


def http_get(url, byte_range=None):
    req = urllib.request.Request(url, headers={"User-Agent": "idletoken-measure/1"})
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    # urlopen follows the 302 to the CDN and re-sends the Range header.
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def parse_header(buf):
    """-> (tensors[(name, offset)], data_start). Raises Short if buf is too small."""
    r = Reader(buf)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file (bad magic)")
    r.u32()  # version
    n_tensors, n_kv = r.u64(), r.u64()
    alignment = 32
    for _ in range(n_kv):
        key = r.string()
        t = r.u32()
        if key == "general.alignment":
            # Needed for data_start, so read it rather than skipping it.
            if t != 4:
                raise ValueError("general.alignment is not UINT32")
            alignment = r.u32()
        else:
            r.skip_value(t)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        r.take(8 * n_dims)  # dims
        r.u32()  # ggml type
        tensors.append((name, r.u64()))
    data_start = (r.i + alignment - 1) // alignment * alignment
    return tensors, data_start


def measure(repo, name, file_size):
    """-> (layer_bytes, shared_bytes, data_start)."""
    want = 16 << 20
    while True:
        buf = http_get(HF_FILE.format(repo=repo, name=name), (0, want - 1))
        try:
            tensors, data_start = parse_header(buf)
            break
        except Short:
            if want >= (256 << 20):
                raise SystemExit(f"{name}: header exceeds 256 MiB — refusing to keep fetching")
            want *= 2
    tensors.sort(key=lambda t: t[1])
    layer = shared = 0
    for i, (tname, off) in enumerate(tensors):
        end = tensors[i + 1][1] if i + 1 < len(tensors) else file_size - data_start
        size = end - off
        if size < 0:
            raise SystemExit(f"{name}: tensor {tname} has negative size — bad file_size?")
        if tname.startswith("blk."):
            layer += size
        else:
            shared += size
    # The contract that makes the result trustworthy. Never downgrade to a warning.
    if layer + shared + data_start != file_size:
        raise SystemExit(
            f"{name}: self-check FAILED — layer({layer}) + shared({shared}) + "
            f"header({data_start}) = {layer + shared + data_start} != file_size({file_size})"
        )
    return layer, shared, data_start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="HF repo, e.g. unsloth/Qwen3.5-4B-GGUF")
    ap.add_argument("quants", nargs="*", help="quant names to measure, e.g. Q5_K_M Q6_K")
    ap.add_argument("--all-files", action="store_true", help="measure every .gguf in the repo")
    ap.add_argument("--json", action="store_true", help="emit manifest-shaped variant objects")
    args = ap.parse_args()

    api = json.loads(http_get(HF_API.format(repo=args.repo)))
    sizes = {
        s["rfilename"]: s.get("size")
        for s in api.get("siblings", [])
        if s["rfilename"].endswith(".gguf")
    }
    if not sizes:
        raise SystemExit(f"{args.repo}: no .gguf files listed")

    stem = args.repo.split("/")[-1].removesuffix("-GGUF")
    if args.all_files:
        picks = sorted(sizes)
    else:
        picks = []
        for q in args.quants:
            fn = f"{stem}-{q}.gguf"
            if fn not in sizes:
                # A missing file is the failure mode that put a 404 in the
                # shipped precision menu. Say so loudly and keep going.
                print(f"MISSING  {fn} is not in {args.repo}", file=sys.stderr)
                continue
            picks.append(fn)

    out = []
    for fn in picks:
        layer, shared, hdr = measure(args.repo, fn, sizes[fn])
        quant = fn.removesuffix(".gguf").removeprefix(stem + "-")
        out.append(
            {
                "quant": quant,
                "layer_weight_bytes": layer,
                "shared_weight_bytes": shared,
                "repo": args.repo,
                "gguf": fn,
            }
        )
        print(
            f"{fn:<40} layer={layer:>14,} shared={shared:>13,} header={hdr:>10,} "
            f"total={sizes[fn]:>14,}",
            file=sys.stderr,
        )
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
