#!/usr/bin/env python3
"""Generate metadata-only GGUF fixtures for the ds4x config unit tests.

A fixture is a valid GGUF v3 header (magic + KV section + empty tensor
directory) with realistic metadata for each architecture the ds4x backend
speaks. No tensor bytes — ds4x_config_from_gguf never reads them.

Usage: make_test_gguf.py <outdir>     (writes glm_dsa.gguf, deepseek2.gguf,
                                       unknown_arch.gguf)
"""
import struct
import sys
from pathlib import Path

# GGUF metadata value type ids
T_U32, T_F32, T_STR, T_ARR = 4, 6, 8, 9


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


T_I32 = 5


def kv(key: str, vtype: int, value) -> bytes:
    out = _str(key) + struct.pack("<I", vtype)
    if vtype == T_U32:
        out += struct.pack("<I", value)
    elif vtype == T_F32:
        out += struct.pack("<f", value)
    elif vtype == T_STR:
        out += _str(value)
    elif vtype == T_ARR:  # value = (item_type, list)
        it, items = value
        out += struct.pack("<IQ", it, len(items))
        for x in items:
            if it == T_STR:
                out += _str(x)
            elif it == T_I32:
                out += struct.pack("<i", x)
            elif it == T_U32:
                out += struct.pack("<I", x)
            else:
                raise ValueError(it)
    else:
        raise ValueError(vtype)
    return out


def _bytes_to_unicode():
    """GPT-2 byte↔unicode map (must match ds4x_tokenizer.c build_byte_unicode)."""
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    b2u, n = {}, 0
    for b in range(256):
        if b in printable:
            b2u[b] = b
        else:
            b2u[b] = 256 + n
            n += 1
    return {b: chr(cp) for b, cp in b2u.items()}


def write_tokenizer_gguf(path):
    """Tiny byte-level BPE tokenizer GGUF: all 256 byte tokens + a few merges
    that build 'hello', plus bos/eos control tokens. Exercises vocab load,
    decode (byte-unicode reversal), special tokens, and BPE merge encode."""
    b2u = _bytes_to_unicode()
    tokens = [b2u[b] for b in range(256)]          # ids 0..255 = one per byte
    types = [1] * 256                              # TT_NORMAL
    # merge tokens (in byte-unicode space; ASCII letters are printable = self)
    extra = ["he", "ll", "hell", "hello"]
    for e in extra:
        tokens.append(e); types.append(1)
    bos_id, eos_id = len(tokens), len(tokens) + 1
    tokens += ["<bos>", "<eos>"]; types += [3, 3]  # TT_CONTROL
    # ChatML special tokens (USER_DEFINED) so the special-token-aware encode +
    # ds4x_tok_chat_apply are exercised (Qwen3 ships these as added tokens).
    tokens += ["<|im_start|>", "<|im_end|>"]; types += [4, 4]  # TT_USER_DEFINED
    merges = ["h e", "l l", "he ll", "hell o"]     # ranks by order

    a = "tokenizer.ggml"
    kvs = [
        kv("general.architecture", T_STR, "glm_dsa"),
        kv(f"{a}.model", T_STR, "gpt2"),
        kv(f"{a}.tokens", T_ARR, (T_STR, tokens)),
        kv(f"{a}.token_type", T_ARR, (T_I32, types)),
        kv(f"{a}.merges", T_ARR, (T_STR, merges)),
        kv(f"{a}.bos_token_id", T_U32, bos_id),
        kv(f"{a}.eos_token_id", T_U32, eos_id),
    ]
    write_gguf(Path(path), kvs)


def write_gguf(path: Path, kvs: list[bytes]) -> None:
    hdr = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))  # version 3, 0 tensors
    path.write_bytes(hdr + b"".join(kvs))


def main(outdir: str) -> None:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)

    # GLM-5.2 shape (zai-org/GLM-5.2 config.json, 2026-07 research):
    # explicit nope/rope keys + vocab_size key.
    a = "glm_dsa"
    write_gguf(d / "glm_dsa.gguf", [
        kv("general.architecture", T_STR, a),
        kv(f"{a}.block_count", T_U32, 78),
        kv(f"{a}.embedding_length", T_U32, 6144),
        kv(f"{a}.attention.head_count", T_U32, 64),
        kv(f"{a}.attention.head_count_kv", T_U32, 1),
        kv(f"{a}.attention.kv_lora_rank", T_U32, 512),
        kv(f"{a}.attention.q_lora_rank", T_U32, 2048),
        kv(f"{a}.attention.qk_nope_head_dim", T_U32, 192),
        kv(f"{a}.rope.dimension_count", T_U32, 64),
        kv(f"{a}.attention.v_head_dim", T_U32, 256),
        kv(f"{a}.expert_count", T_U32, 256),
        kv(f"{a}.expert_used_count", T_U32, 8),
        kv(f"{a}.expert_shared_count", T_U32, 1),
        kv(f"{a}.expert_feed_forward_length", T_U32, 2048),
        kv(f"{a}.leading_dense_block_count", T_U32, 3),
        kv(f"{a}.feed_forward_length", T_U32, 12288),
        kv(f"{a}.rope.freq_base", T_F32, 8_000_000.0),
        kv(f"{a}.context_length", T_U32, 1_048_576),
        kv(f"{a}.vocab_size", T_U32, 154_880),
        kv(f"{a}.index_topk", T_U32, 2048),
        kv(f"{a}.index_topk_freq", T_U32, 4),
    ])

    # deepseek2 shape (Kimi K2.x / DeepSeek V3 lineage) exercising the
    # ALTERNATE spellings: key_length/value_length instead of nope/v_head_dim,
    # vocab via len(tokenizer.ggml.tokens) instead of vocab_size.
    a = "deepseek2"
    write_gguf(d / "deepseek2.gguf", [
        kv("general.architecture", T_STR, a),
        kv(f"{a}.block_count", T_U32, 61),
        kv(f"{a}.embedding_length", T_U32, 7168),
        kv(f"{a}.attention.head_count", T_U32, 64),
        kv(f"{a}.attention.head_count_kv", T_U32, 1),
        kv(f"{a}.attention.kv_lora_rank", T_U32, 512),
        kv(f"{a}.attention.q_lora_rank", T_U32, 1536),
        kv(f"{a}.attention.key_length", T_U32, 192),   # nope 128 + rope 64
        kv(f"{a}.rope.dimension_count", T_U32, 64),
        kv(f"{a}.attention.value_length", T_U32, 128),
        kv(f"{a}.expert_count", T_U32, 384),
        kv(f"{a}.expert_used_count", T_U32, 8),
        kv(f"{a}.expert_shared_count", T_U32, 1),
        kv(f"{a}.expert_feed_forward_length", T_U32, 2048),
        kv(f"{a}.leading_dense_block_count", T_U32, 1),
        kv(f"{a}.feed_forward_length", T_U32, 18432),
        kv(f"{a}.rope.freq_base", T_F32, 50_000.0),
        kv(f"{a}.context_length", T_U32, 262_144),
        kv("tokenizer.ggml.tokens", T_ARR, (T_STR, [f"t{i}" for i in range(1000)])),
    ])

    # ds4x must refuse architectures outside its coverage (MLA-MoE + GQA
    # dense/MoE). "mamba" is a state-space arch it does not implement.
    write_gguf(d / "unknown_arch.gguf", [
        kv("general.architecture", T_STR, "mamba"),
        kv("mamba.block_count", T_U32, 32),
    ])

    write_tokenizer_gguf(d / "tokenizer.gguf")

    print(f"wrote 4 fixtures to {d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/fixtures")
