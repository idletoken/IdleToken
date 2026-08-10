#!/usr/bin/env python3
"""GGUF layer-sharding tool for IdleToken.

A pipeline worker only runs layers [lo, hi), so it should not need the whole
80 GB GGUF locally. ds4's loader (see docs/ investigation) mmaps the file and
requires the FULL tensor directory to be present, but never reads the bytes of
layers it skips. So a worker can load a *sparse* file that is the original
apparent size (bounds checks are arithmetic) yet only physically holds the
header + shared tensors + its assigned layers; the rest are holes.

This tool has three modes:

  index <master.gguf> [-o manifest.json]
      Parse the GGUF directory and emit a manifest: file_size, tensor_data_pos,
      alignment, and every tensor's {name, layer, offset, bytes}. Self-validates
      that the reconstructed directory sums to the real file size.

  materialize <master.gguf> <manifest.json> <lo> <hi> <out.gguf>
      Write a sparse partial GGUF holding only the header + shared tensors +
      layers [lo, hi). Apparent size == original; disk usage ~= what's needed.

  ranges <manifest.json> <lo> <hi>
      Print the byte ranges [start,end) a worker for [lo,hi) must fetch (header
      + shared + those layers). This is exactly what the C fetcher pulls over
      HTTP range requests; kept here so the logic has one source of truth.

The quant block geometry table is copied verbatim from ds4.c gguf_types[] so the
byte sizes are byte-exact.
"""
import json
import os
import struct
import sys

# (block_elems, block_bytes) per GGUF tensor type id — verbatim from ds4.c.
GGUF_TYPES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 110), 20: (256, 50), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2),
}

# GGUF metadata value types.
_MV_U8, _MV_I8, _MV_U16, _MV_I16, _MV_U32, _MV_I32, _MV_F32, _MV_BOOL, \
    _MV_STRING, _MV_ARRAY, _MV_U64, _MV_I64, _MV_F64 = range(13)
_SCALAR_SIZE = {_MV_U8: 1, _MV_I8: 1, _MV_U16: 2, _MV_I16: 2, _MV_U32: 4,
                _MV_I32: 4, _MV_F32: 4, _MV_BOOL: 1, _MV_U64: 8, _MV_I64: 8,
                _MV_F64: 8}


class _Cur:
    def __init__(self, buf):
        self.b = buf
        self.p = 0

    def need(self, n):
        if self.p + n > len(self.b):
            raise EOFError(f"GGUF cursor past end (need {n} at {self.p}/{len(self.b)})")

    def u32(self):
        self.need(4); v = struct.unpack_from("<I", self.b, self.p)[0]; self.p += 4; return v

    def u64(self):
        self.need(8); v = struct.unpack_from("<Q", self.b, self.p)[0]; self.p += 8; return v

    def s(self):
        n = self.u64(); self.need(n); v = self.b[self.p:self.p + n]; self.p += n
        return v.decode("utf-8", "replace")


def _skip_value(c, vtype):
    if vtype == _MV_STRING:
        c.s(); return
    if vtype == _MV_ARRAY:
        it = c.u32(); ln = c.u64()
        sz = _SCALAR_SIZE.get(it)
        if sz is not None:
            c.need(sz * ln); c.p += sz * ln
        else:
            for _ in range(ln):
                _skip_value(c, it)
        return
    sz = _SCALAR_SIZE.get(vtype)
    if sz is None:
        raise ValueError(f"unknown GGUF metadata value type {vtype}")
    c.need(sz); c.p += sz


def _read_value_u32(c, vtype):
    """Read a value we care about (alignment is U32); skip otherwise."""
    if vtype == _MV_U32:
        return c.u32()
    _skip_value(c, vtype)
    return None


def _align_up(x, a):
    return (x + a - 1) // a * a


def _layer_of(name):
    """blk.<N>.* -> N, else -1 (shared/global tensor)."""
    if not name.startswith("blk."):
        return -1
    rest = name[4:]
    dot = rest.find(".")
    if dot <= 0 or not rest[:dot].isdigit():
        return -1
    return int(rest[:dot])


def parse_gguf(path, header_bytes=None):
    """Parse the GGUF header + tensor directory. Reads only the header region
    (a few MB), not the tensor data. Returns the manifest dict."""
    file_size = os.path.getsize(path)
    # The directory lives at the start; read a generous chunk, grow if needed.
    read_n = header_bytes or min(file_size, 64 * 1024 * 1024)
    with open(path, "rb") as f:
        buf = f.read(read_n)
        while True:
            try:
                return _parse_from(buf, file_size)
            except EOFError:
                if len(buf) >= file_size:
                    raise
                read_n = min(file_size, len(buf) * 2)
                f.seek(0)
                buf = f.read(read_n)


def _parse_from(buf, file_size):
    c = _Cur(buf)
    magic = c.b[:4]; c.p = 4
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF file (magic={magic!r})")
    version = c.u32()
    n_tensors = c.u64()
    n_kv = c.u64()
    alignment = 32  # GGUF default
    for _ in range(n_kv):
        key = c.s()
        vtype = c.u32()
        val = _read_value_u32(c, vtype)
        if key == "general.alignment" and val is not None:
            alignment = val
    tensors = []
    for _ in range(n_tensors):
        name = c.s()
        ndim = c.u32()
        elems = 1
        for _ in range(ndim):
            elems *= c.u64()
        ttype = c.u32()
        rel = c.u64()
        if ttype not in GGUF_TYPES:
            raise ValueError(f"tensor {name}: unknown type {ttype}")
        be, bb = GGUF_TYPES[ttype]
        nbytes = ((elems + be - 1) // be) * bb
        tensors.append({"name": name, "type": ttype, "layer": _layer_of(name),
                        "rel": rel, "bytes": nbytes})
    tensor_data_pos = _align_up(c.p, alignment)
    max_end = tensor_data_pos
    for t in tensors:
        t["offset"] = tensor_data_pos + t["rel"]
        del t["rel"]
        max_end = max(max_end, t["offset"] + t["bytes"])
    return {
        "version": version,
        "file_size": file_size,
        "tensor_data_pos": tensor_data_pos,
        "alignment": alignment,
        "n_tensors": n_tensors,
        "reconstructed_end": max_end,
        "tensors": tensors,
    }


def needed_ranges(man, lo, hi):
    """Byte ranges a worker for layers [lo,hi) must have: the whole header/
    directory region + every shared tensor + every tensor in [lo,hi)."""
    ranges = [(0, man["tensor_data_pos"])]  # header + directory
    for t in man["tensors"]:
        L = t["layer"]
        if L == -1 or (lo <= L < hi):
            ranges.append((t["offset"], t["offset"] + t["bytes"]))
    ranges.sort()
    # merge adjacent/overlapping
    merged = []
    for s, e in ranges:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def cmd_index(argv):
    path = argv[0]
    out = None
    if len(argv) >= 3 and argv[1] == "-o":
        out = argv[2]
    man = parse_gguf(path)
    end, fsz = man["reconstructed_end"], man["file_size"]
    # Self-check: the reconstructed directory must fit within the file, and the
    # last tensor should end at (or one alignment pad before) EOF.
    ok = end <= fsz and (fsz - end) < man["alignment"] + 1
    shared = sum(t["bytes"] for t in man["tensors"] if t["layer"] == -1)
    n_layers = 1 + max((t["layer"] for t in man["tensors"]), default=-1)
    sys.stderr.write(
        f"[gguf_shard] {os.path.basename(path)}: {man['n_tensors']} tensors, "
        f"{n_layers} layers, file={fsz} recon_end={end} "
        f"shared={shared/1e9:.2f}GB validate={'OK' if ok else 'MISMATCH'}\n")
    if not ok:
        sys.stderr.write(f"[gguf_shard] VALIDATION FAILED: recon_end {end} vs file {fsz}\n")
        return 2
    text = json.dumps(man)
    if out:
        with open(out, "w") as f:
            f.write(text)
        sys.stderr.write(f"[gguf_shard] wrote {out}\n")
    else:
        print(text)
    return 0


def _load_manifest(p):
    with open(p) as f:
        return json.load(f)


def cmd_ranges(argv):
    man = _load_manifest(argv[0])
    lo, hi = int(argv[1]), int(argv[2])
    rs = needed_ranges(man, lo, hi)
    total = sum(e - s for s, e in rs)
    sys.stderr.write(f"[gguf_shard] layers [{lo},{hi}): {len(rs)} ranges, "
                     f"{total/1e9:.2f}GB of {man['file_size']/1e9:.2f}GB\n")
    for s, e in rs:
        print(f"{s} {e}")
    return 0


def cmd_idx(argv):
    """C-friendly line-based manifest for the worker's fetcher (no JSON parse in
    C). Line 1: `file_size tensor_data_pos n_tensors`. Then one line per tensor:
    `layer offset bytes` (layer -1 = shared/global)."""
    man = parse_gguf(argv[0])
    out = None
    if len(argv) >= 3 and argv[1] == "-o":
        out = argv[2]
    lines = [f"{man['file_size']} {man['tensor_data_pos']} {man['n_tensors']}"]
    for t in man["tensors"]:
        lines.append(f"{t['layer']} {t['offset']} {t['bytes']}")
    text = "\n".join(lines) + "\n"
    if out:
        with open(out, "w") as f:
            f.write(text)
        sys.stderr.write(f"[gguf_shard] wrote {out} ({man['n_tensors']} tensors)\n")
    else:
        sys.stdout.write(text)
    return 0


def cmd_materialize(argv):
    master, manifest, lo, hi, out = argv[0], argv[1], int(argv[2]), int(argv[3]), argv[4]
    man = _load_manifest(manifest)
    rs = needed_ranges(man, lo, hi)
    fsz = man["file_size"]
    total = sum(e - s for s, e in rs)
    sys.stderr.write(f"[gguf_shard] materialize [{lo},{hi}) -> {out}: "
                     f"copying {total/1e9:.2f}GB of {fsz/1e9:.2f}GB (sparse)\n")
    with open(master, "rb") as src, open(out, "wb") as dst:
        dst.truncate(fsz)  # full apparent size; holes stay unallocated (sparse)
        for s, e in rs:
            src.seek(s)
            dst.seek(s)
            remaining = e - s
            while remaining > 0:
                chunk = src.read(min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    break
                dst.write(chunk)
                remaining -= len(chunk)
    sys.stderr.write(f"[gguf_shard] done: {out} apparent={fsz} bytes\n")
    return 0


def main(argv):
    if not argv:
        sys.stderr.write(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "index":
        return cmd_index(rest)
    if cmd == "idx":
        return cmd_idx(rest)
    if cmd == "ranges":
        return cmd_ranges(rest)
    if cmd == "materialize":
        return cmd_materialize(rest)
    sys.stderr.write(f"unknown command: {cmd}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
