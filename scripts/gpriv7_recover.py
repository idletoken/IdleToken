#!/usr/bin/env python3
"""G-PRIV-7 embedding-recovery attacker (v2 rebuild plan §5.1 / F2).

Given a captured RPC byte stream, the public GGUF, and the prompt's token ids,
try to recover the RAW EMBEDDING ROWS of the prompt tokens from the stream.

Why this works, and why it is the whole point of the layer-0 privacy invariant:
llama.cpp pins the embedding LOOKUP to the host CPU, but the RESULT of that
lookup is the input to layer 0. If layer 0 lives on a remote worker, that f32
tensor -- one row per prompt token, each row == the token's embedding -- is
shipped over RPC via SET_TENSOR. The embedding table is PUBLIC (it is in the
GGUF anyone can download), so a row of hidden-size f32 on the wire is
equivalent to plaintext: dequantize token_embd, and every prompt token's row
matches byte-for-byte.

Method (mirrors the pivot-doc appendix, which measured 17/17 recovery with a
20-decoy false-positive control):
  - dequantize token_embd.weight from the GGUF -> (n_vocab, n_embd) f32;
  - for each prompt token id, take the first K f32 of its row (K*4 bytes) and
    search the captured stream for that exact byte prefix;
  - for a set of DECOY token ids NOT in the prompt, do the same as a
    false-positive control -- a decoy hit would mean the match is coincidence,
    not recovery.

Output (one JSON line on stdout):
  {"n_prompt": N, "recovered": R, "n_decoy": D, "decoy_hits": H,
   "recovered_ids": [...], "n_embd": E}

Exact byte-prefix matching is deliberate: it was verified (pivot appendix and
this repo's own measurement) that gguf-py's dequantization reproduces the exact
f32 bytes llama.cpp puts on the wire for this model family. If that ever stops
holding, the positive control in the gate goes red -- which is the correct
fail-closed signal, not a reason to loosen the match into something that would
also match noise.
"""
import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--tokens", required=True,
                    help="comma-separated prompt token ids")
    ap.add_argument("--decoys", default="",
                    help="comma-separated decoy token ids (not in the prompt)")
    ap.add_argument("--gguf-py", default="",
                    help="path to the gguf-py package (for dequantize)")
    ap.add_argument("--prefix-floats", type=int, default=8)
    args = ap.parse_args()

    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)
    try:
        import numpy as np
        from gguf import GGUFReader, dequantize
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": "import: %s" % e}))
        return 3

    reader = GGUFReader(args.gguf)
    emb = None
    shape = None
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            emb = dequantize(t.data, t.tensor_type)
            shape = t.shape
            break
    if emb is None:
        print(json.dumps({"error": "no token_embd.weight in the GGUF"}))
        return 3
    n_embd = int(shape[0])
    n_vocab = int(shape[1])
    emb = emb.reshape(n_vocab, n_embd).astype(np.float32)

    with open(args.capture, "rb") as f:
        cap = f.read()

    prompt = [int(x) for x in args.tokens.split(",") if x.strip() != ""]
    decoys = [int(x) for x in args.decoys.split(",") if x.strip() != ""]
    k = max(1, args.prefix_floats)

    def hit(tid: int) -> bool:
        if tid < 0 or tid >= n_vocab:
            return False
        return cap.find(emb[tid][:k].tobytes()) >= 0

    # De-dup the prompt: the same token appearing twice is one recoverable row.
    uniq = list(dict.fromkeys(prompt))
    recovered = [t for t in uniq if hit(t)]
    decoy_hits = [d for d in decoys if hit(d)]

    print(json.dumps({
        "n_prompt": len(uniq),
        "recovered": len(recovered),
        "n_decoy": len(decoys),
        "decoy_hits": len(decoy_hits),
        "recovered_ids": recovered,
        "n_embd": n_embd,
        "capture_bytes": len(cap),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
