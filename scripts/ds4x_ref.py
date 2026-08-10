#!/usr/bin/env python3
"""Numpy reference for the ds4x MLA-MoE layer forward (Phase B.2 oracle).

Generates a test-vector bundle: a tiny random-weight model in the exact shape
family ds4x speaks (MLA attention + optional dense-lead + MoE swiglu), the
input tokens, and the expected hidden states after every layer. The C CPU
forward (src/ds4x/ds4x_forward.c) must reproduce `expect` to fp32 tolerance —
that is the alignment gate BEFORE any CUDA work. Real-model alignment against
llama.cpp running GLM-5.2 happens later on the DGX (G-GLM ladder).

Math spec (C side mirrors this exactly — keep in sync):
  rmsnorm(x, w) = x * w / sqrt(mean(x^2) + 1e-6)
  rope: neox pairs (i, i+d/2), theta_i = pos * base^(-2i/d)
  MLA:  q via optional LoRA (q_a -> rmsnorm -> q_b); kv_a -> latent(rank)+k_rope
        latent rmsnormed and cached; per-head k_nope/v reconstructed via kv_b
        score = (q_nope.k_nope + q_rope.k_rope)/sqrt(nope+rope), causal softmax
  MoE:  softmax router, top-k renormalized, swiglu experts + shared expert
        (score-function variants like sigmoid routing are a Phase B checklist
        item to verify per-arch against llama.cpp before real weights)

Bundle format (little-endian):
  8s magic "DS4XVEC1", u32 n_records,
  then per record: u64 name_len, name, u32 ndim, u64 dims[ndim], f32 data.

Also writes a REAL tiny GGUF next to the bundle (<out.bin>.gguf): same weights
under llama.cpp tensor names + glm_dsa metadata, so ds4x_model_load() is
tested end-to-end (file → weights → forward → same expect).

Usage: ds4x_ref.py <out.bin>
"""
import struct
import sys

import numpy as np

ALIGN = 32
T_U32, T_F32_KV, T_STR = 4, 6, 8

# reference-record name → llama.cpp tensor name (per layer)
GGUF_NAMES = {
    "attn_norm": "attn_norm.weight",
    "q_a": "attn_q_a.weight",
    "q_a_norm": "attn_q_a_norm.weight",
    "q_b": "attn_q_b.weight",
    "q_proj": "attn_q.weight",
    "kv_a": "attn_kv_a_mqa.weight",
    "kv_a_norm": "attn_kv_a_norm.weight",
    # kv_b is emitted as the SPLIT absorb form (attn_k_b/attn_v_b) below, to
    # exercise ds4x_model.c's split-tensor assembly against ground truth.
    "attn_out": "attn_output.weight",
    "ffn_norm": "ffn_norm.weight",
    "gate": "ffn_gate.weight",
    "up": "ffn_up.weight",
    "down": "ffn_down.weight",
    "router": "ffn_gate_inp.weight",
    "e_score_bias": "exp_probs_b.bias",
    "e_gate": "ffn_gate_exps.weight",
    "e_up": "ffn_up_exps.weight",
    "e_down": "ffn_down_exps.weight",
    "s_gate": "ffn_gate_shexp.weight",
    "s_up": "ffn_up_shexp.weight",
    "s_down": "ffn_down_shexp.weight",
}


def q8_0_encode(mat):
    """Quantize a 2D fp32 matrix (last dim ×32) to ggml Q8_0. Returns
    (raw_bytes, dequantized_fp32) — the dequant uses the SAME f16 scale + int8
    the bytes carry, so ds4x's on-the-fly dequant reproduces it bit-for-bit."""
    m = np.ascontiguousarray(mat, dtype=np.float32).reshape(-1, mat.shape[-1])
    n_out, n_in = m.shape
    nb = n_in // 32
    raw = bytearray()
    deq = np.empty((n_out, n_in), dtype=np.float32)
    for o in range(n_out):
        for b in range(nb):
            blk = m[o, b * 32:(b + 1) * 32]
            amax = float(np.max(np.abs(blk)))
            d = amax / 127.0
            d16 = np.float16(d)
            df = np.float32(d16)
            if df != 0:
                qs = np.clip(np.round(blk / d), -127, 127).astype(np.int8)
            else:
                qs = np.zeros(32, dtype=np.int8)
            raw += d16.tobytes() + qs.tobytes()
            deq[o, b * 32:(b + 1) * 32] = df * qs.astype(np.float32)
    return bytes(raw), deq.reshape(mat.shape)


def _s(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv_u32(k, v):
    return _s(k) + struct.pack("<II", T_U32, v)


def _kv_f32(k, v):
    return _s(k) + struct.pack("<If", T_F32_KV, v)


def write_model_gguf(path, cfg, W, qbytes=None):
    qbytes = qbytes or {}
    a = "glm_dsa"
    kvs = [
        _s("general.architecture") + struct.pack("<I", T_STR) + _s(a),
        _kv_u32(f"{a}.block_count", cfg.n_layer),
        _kv_u32(f"{a}.embedding_length", cfg.n_embd),
        _kv_u32(f"{a}.attention.head_count", cfg.n_head),
        _kv_u32(f"{a}.attention.head_count_kv", 1),
        _kv_u32(f"{a}.attention.kv_lora_rank", cfg.kv_rank),
        _kv_u32(f"{a}.attention.q_lora_rank", cfg.q_rank),
        _kv_u32(f"{a}.attention.qk_nope_head_dim", cfg.nope),
        _kv_u32(f"{a}.rope.dimension_count", cfg.rope_dim),
        _kv_u32(f"{a}.attention.v_head_dim", cfg.v_dim),
        _kv_u32(f"{a}.expert_count", cfg.n_expert),
        _kv_u32(f"{a}.expert_used_count", cfg.top_k),
        _kv_u32(f"{a}.expert_shared_count", cfg.n_shared),
        _kv_u32(f"{a}.expert_feed_forward_length", cfg.ff_exp),
        _kv_u32(f"{a}.leading_dense_block_count", cfg.dense_lead),
        _kv_u32(f"{a}.feed_forward_length", cfg.ff_dense),
        _kv_u32(f"{a}.expert_gating_func", cfg.gating_func),
        _kv_u32(f"{a}.expert_weights_norm", cfg.weights_norm),
        _kv_f32(f"{a}.expert_weights_scale", cfg.weights_scale),
        _kv_f32(f"{a}.rope.freq_base", cfg.rope_theta),
        _kv_u32(f"{a}.vocab_size", 1000),
        _kv_u32("general.alignment", ALIGN),
    ]
    # tensors: (name, arr) with GGUF dims innermost-first (= reversed shape)
    tensors = []
    for name in ("token_embd.weight", "output_norm.weight", "output.weight"):
        key = {"token_embd.weight": "token_embd",
               "output_norm.weight": "output_norm",
               "output.weight": "output"}[name]
        if key in W:
            tensors.append((name, np.ascontiguousarray(W[key], dtype=np.float32)))
    for il in range(cfg.n_layer):
        for rec_name, gguf_suffix in GGUF_NAMES.items():
            key = f"blk.{il}.{rec_name}"
            if key in W:
                tensors.append((f"blk.{il}.{gguf_suffix}",
                                np.ascontiguousarray(W[key], dtype=np.float32)))
        # emit kv_b as split attn_k_b / attn_v_b (llama.cpp shapes):
        #   combined kv_b = [n_head, nope+v_dim, kv_rank]
        #   k_b ggml {nope, kv_rank, n_head}  → numpy [n_head, kv_rank, nope]
        #   v_b ggml {kv_rank, v_dim, n_head} → numpy [n_head, v_dim, kv_rank]
        kvb = W[f"blk.{il}.kv_b"].reshape(cfg.n_head, cfg.nope + cfg.v_dim, cfg.kv_rank)
        k_part = kvb[:, : cfg.nope, :]                       # [nh, nope, kv_rank]
        v_part = kvb[:, cfg.nope:, :]                        # [nh, v_dim, kv_rank]
        k_b = np.ascontiguousarray(k_part.transpose(0, 2, 1), dtype=np.float32)
        v_b = np.ascontiguousarray(v_part, dtype=np.float32)
        tensors.append((f"blk.{il}.attn_k_b.weight", k_b))
        tensors.append((f"blk.{il}.attn_v_b.weight", v_b))
    infos, data, off = b"", b"", 0
    for name, arr in tensors:
        off = (off + ALIGN - 1) // ALIGN * ALIGN
        infos += _s(name) + struct.pack("<I", arr.ndim)
        for d in reversed(arr.shape):
            infos += struct.pack("<Q", d)
        if name in qbytes:
            blob = qbytes[name]                       # Q8_0 raw bytes
            infos += struct.pack("<IQ", 8, off)       # type Q8_0
        else:
            blob = arr.tobytes()
            infos += struct.pack("<IQ", 0, off)       # type F32
        pad = off - len(data)
        data += b"\0" * pad + blob
        off += len(blob)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + infos
    pad = (len(head) + ALIGN - 1) // ALIGN * ALIGN - len(head)
    with open(path, "wb") as f:
        f.write(head + b"\0" * pad + data)

EPS = 1e-6


def rmsnorm(x, w):
    return x * w / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(v, pos, base):
    """neox-style: rotate pairs (i, i+d/2). v: [d], returns rotated copy."""
    d = v.shape[-1]
    half = d // 2
    out = v.copy()
    for i in range(half):
        theta = pos * base ** (-2.0 * i / d)
        a, b = v[i], v[i + half]
        out[i] = a * np.cos(theta) - b * np.sin(theta)
        out[i + half] = a * np.sin(theta) + b * np.cos(theta)
    return out


def rope_partial(v, pos, base, rdim):
    """Rotate only the first `rdim` dims of a head, leave the rest untouched.
    qwen35 does this (rope.dimension_count 64 < head_dim 256); mirrors the C
    forward, where rope_dim_partial == 0 means "rotate the whole head"."""
    out = v.copy()
    out[:rdim] = rope(v[:rdim], pos, base)
    return out


class Cfg:
    n_layer = 2          # layer 0 dense-lead, layer 1 MoE
    n_vocab = 1000
    n_embd = 64
    n_head = 4
    kv_rank = 32
    q_rank = 48
    nope = 16
    rope_dim = 8
    v_dim = 16
    n_expert = 8
    top_k = 2
    n_shared = 1
    ff_exp = 32
    ff_dense = 96
    dense_lead = 1
    rope_theta = 10000.0
    n_tokens = 5
    # router weighting (DeepSeek-V3 lineage: sigmoid + bias + norm + scale)
    gating_func = 2      # 1=softmax, 2=sigmoid
    weights_norm = 1
    weights_scale = 2.5


def forward(cfg, W, x):
    """x: [n_tokens, n_embd] -> hidden after all layers. Sequential tokens
    with a growing MLA cache, mirroring the C implementation."""
    n_tok = x.shape[0]
    qd = cfg.nope + cfg.rope_dim
    scale = 1.0 / np.sqrt(qd)
    h = x.copy()
    for il in range(cfg.n_layer):
        p = f"blk.{il}."
        # cache: latent [t, kv_rank], k_rope [t, rope_dim]
        lat_cache = np.zeros((n_tok, cfg.kv_rank), dtype=h.dtype)
        rope_cache = np.zeros((n_tok, cfg.rope_dim), dtype=h.dtype)
        attn_out = np.zeros_like(h)
        for t in range(n_tok):
            cur = rmsnorm(h[t], W[p + "attn_norm"])
            # q path: LoRA (q_rank>0) or full projection (q_rank==0)
            if cfg.q_rank > 0:
                qa = cur @ W[p + "q_a"].T
                qa = rmsnorm(qa, W[p + "q_a_norm"])
                q = (qa @ W[p + "q_b"].T).reshape(cfg.n_head, qd)
            else:
                q = (cur @ W[p + "q_proj"].T).reshape(cfg.n_head, qd)
            # kv path
            kv = cur @ W[p + "kv_a"].T
            lat = rmsnorm(kv[: cfg.kv_rank], W[p + "kv_a_norm"])
            k_r = rope(kv[cfg.kv_rank:], t, cfg.rope_theta)
            lat_cache[t] = lat
            rope_cache[t] = k_r
            # per-head attention over cache [0..t]
            kvb = W[p + "kv_b"].reshape(cfg.n_head, cfg.nope + cfg.v_dim, cfg.kv_rank)
            heads = []
            for hd in range(cfg.n_head):
                q_n = q[hd, : cfg.nope]
                q_r = rope(q[hd, cfg.nope:], t, cfg.rope_theta)
                k_n = lat_cache[: t + 1] @ kvb[hd, : cfg.nope].T      # [t+1, nope]
                v = lat_cache[: t + 1] @ kvb[hd, cfg.nope:].T          # [t+1, v]
                s = (k_n @ q_n + rope_cache[: t + 1] @ q_r) * scale
                s = np.exp(s - s.max())
                s /= s.sum()
                heads.append(s @ v)
            attn_out[t] = np.concatenate(heads) @ W[p + "attn_out"].T
        h = h + attn_out

        ffn_out = np.zeros_like(h)
        for t in range(n_tok):
            cur = rmsnorm(h[t], W[p + "ffn_norm"])
            if il < cfg.dense_lead:
                y = (silu(cur @ W[p + "gate"].T) * (cur @ W[p + "up"].T)) @ W[p + "down"].T
            else:
                # exact llama.cpp build_moe_ffn order (see ds4x_forward.c)
                logits = cur @ W[p + "router"].T
                if cfg.gating_func == 2:
                    probs = 1.0 / (1.0 + np.exp(-logits))
                else:
                    probs = np.exp(logits - logits.max())
                    probs /= probs.sum()
                bias = W[p + "e_score_bias"]
                sel = probs + bias
                top = np.argsort(-sel)[: cfg.top_k]
                wt = probs[top].astype(h.dtype)            # UNBIASED weights
                if cfg.weights_norm:
                    wt = wt / max(wt.sum(), 6.103515625e-5)
                if cfg.weights_scale not in (0.0, 1.0):
                    wt = wt * cfg.weights_scale
                y = np.zeros(cfg.n_embd, dtype=h.dtype)
                for j, e in enumerate(top):
                    ex = (silu(cur @ W[p + "e_gate"][e].T) * (cur @ W[p + "e_up"][e].T)) @ W[p + "e_down"][e].T
                    y += wt[j] * ex
                if cfg.n_shared > 0:   # a MoE need not have a shared expert
                    y += (silu(cur @ W[p + "s_gate"].T) * (cur @ W[p + "s_up"].T)) @ W[p + "s_down"].T
            ffn_out[t] = y
        h = h + ffn_out
    return h


# ===================== GQA (qwen3) reference =============================
# Standard GQA dense transformer: separate Q/K/V, per-head Q/K RMSNorm (qk_norm,
# Qwen3), neox rope on the whole head_dim, KV heads broadcast, dense SwiGLU FFN.
# The C GQA branch (ds4x_forward.c) mirrors this op-for-op.

class QwenCfg:
    arch = "qwen3"
    n_layer = 2
    n_vocab = 1000
    n_embd = 64
    n_head = 4
    n_head_kv = 2        # GQA: 2 query heads share each KV head
    head_dim = 16
    ff_dense = 96
    rope_theta = 1000000.0
    n_tokens = 5
    qk_norm = 1


def gqa_forward(cfg, W, x):
    n_tok = x.shape[0]
    hdim = cfg.head_dim
    grp = cfg.n_head // cfg.n_head_kv
    scale = 1.0 / np.sqrt(hdim)
    h = x.copy()
    for il in range(cfg.n_layer):
        p = f"blk.{il}."
        kcache = np.zeros((n_tok, cfg.n_head_kv, hdim), dtype=h.dtype)
        vcache = np.zeros((n_tok, cfg.n_head_kv, hdim), dtype=h.dtype)
        attn = np.zeros_like(h)
        for t in range(n_tok):
            cur = rmsnorm(h[t], W[p + "attn_norm"])
            q = (cur @ W[p + "q_proj"].T).reshape(cfg.n_head, hdim)
            k = (cur @ W[p + "k_proj"].T).reshape(cfg.n_head_kv, hdim)
            v = (cur @ W[p + "v_proj"].T).reshape(cfg.n_head_kv, hdim)
            if cfg.qk_norm:
                q = np.stack([rmsnorm(q[i], W[p + "q_norm"]) for i in range(cfg.n_head)])
                k = np.stack([rmsnorm(k[i], W[p + "k_norm"]) for i in range(cfg.n_head_kv)])
            q = np.stack([rope(q[i], t, cfg.rope_theta) for i in range(cfg.n_head)])
            k = np.stack([rope(k[i], t, cfg.rope_theta) for i in range(cfg.n_head_kv)])
            kcache[t] = k
            vcache[t] = v
            heads = []
            for i in range(cfg.n_head):
                kvh = i // grp
                s = (kcache[: t + 1, kvh] @ q[i]) * scale
                s = np.exp(s - s.max())
                s /= s.sum()
                heads.append(s @ vcache[: t + 1, kvh])
            attn[t] = np.concatenate(heads) @ W[p + "attn_out"].T
        h = h + attn
        ffn = np.zeros_like(h)
        for t in range(n_tok):
            cur = rmsnorm(h[t], W[p + "ffn_norm"])
            ffn[t] = (silu(cur @ W[p + "gate"].T) * (cur @ W[p + "up"].T)) @ W[p + "down"].T
        h = h + ffn
    return h


def _gguf_bytes(kvs, tensors):
    infos, data, off = b"", b"", 0
    for name, arr in tensors:
        off = (off + ALIGN - 1) // ALIGN * ALIGN
        infos += _s(name) + struct.pack("<I", arr.ndim)
        for d in reversed(arr.shape):
            infos += struct.pack("<Q", d)
        blob = arr.tobytes()
        infos += struct.pack("<IQ", 0, off)       # type F32
        pad = off - len(data)
        data += b"\0" * pad + blob
        off += len(blob)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + infos
    pad = (len(head) + ALIGN - 1) // ALIGN * ALIGN - len(head)
    return head + b"\0" * pad + data


QWEN_NAMES = {
    "attn_norm": "attn_norm.weight",
    "q_proj": "attn_q.weight",
    "k_proj": "attn_k.weight",
    "v_proj": "attn_v.weight",
    "q_norm": "attn_q_norm.weight",
    "k_norm": "attn_k_norm.weight",
    "attn_out": "attn_output.weight",
    "ffn_norm": "ffn_norm.weight",
    "gate": "ffn_gate.weight",
    "up": "ffn_up.weight",
    "down": "ffn_down.weight",
}


def write_qwen3_gguf(path, cfg, W):
    a = cfg.arch
    kvs = [
        _s("general.architecture") + struct.pack("<I", T_STR) + _s(a),
        _kv_u32(f"{a}.block_count", cfg.n_layer),
        _kv_u32(f"{a}.embedding_length", cfg.n_embd),
        _kv_u32(f"{a}.attention.head_count", cfg.n_head),
        _kv_u32(f"{a}.attention.head_count_kv", cfg.n_head_kv),
        _kv_u32(f"{a}.attention.key_length", cfg.head_dim),
        _kv_u32(f"{a}.feed_forward_length", cfg.ff_dense),
        _kv_f32(f"{a}.rope.freq_base", cfg.rope_theta),
        _kv_u32(f"{a}.vocab_size", cfg.n_vocab),
        _kv_u32("general.alignment", ALIGN),
    ]
    tensors = []
    for name, key in (("token_embd.weight", "token_embd"),
                      ("output_norm.weight", "output_norm"),
                      ("output.weight", "output")):
        if key in W:
            tensors.append((name, np.ascontiguousarray(W[key], dtype=np.float32)))
    for il in range(cfg.n_layer):
        for rec_name, suffix in QWEN_NAMES.items():
            key = f"blk.{il}.{rec_name}"
            if key in W:
                tensors.append((f"blk.{il}.{suffix}",
                                np.ascontiguousarray(W[key], dtype=np.float32)))
    with open(path, "wb") as f:
        f.write(_gguf_bytes(kvs, tensors))


# ============ Gated DeltaNet (Qwen3.5/3.6 hybrid) reference ================
# Transcribed operation-for-operation from transformers Qwen3NextGatedDeltaNet;
# the exact order is documented in docs/linear-attention-design.md §2. Getting
# any of L2-norm / gate sign / conv-then-activation wrong yields a model that
# runs and talks nonsense (cf. the Q4_K bit-unpacking bug), so this file is the
# judge — the C side must match it, not the other way round.

def l2norm(v, eps=1e-6):
    return v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + eps)


def softplus(x):
    # log1p(exp(-|x|)) + max(x,0) — the numerically stable form
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


class GdnCfg:
    """Tiny hybrid model: layer_types = [linear, linear, linear, full] (3:1)."""
    arch = "qwen3next"
    n_layer = 4
    layer_types = ["linear", "linear", "linear", "full"]
    n_vocab = 1000
    n_embd = 64
    # full-attention (GQA) layer
    n_head = 4
    n_head_kv = 2
    head_dim = 16
    # linear-attention (GDN) layer
    lin_k_heads = 2
    lin_v_heads = 4          # > k_heads: each k head serves 2 v heads
    lin_k_dim = 8
    lin_v_dim = 8
    conv_kernel = 4
    ff_dense = 96
    rope_theta = 1000000.0
    n_tokens = 6
    qk_norm = 1
    # qwen35-only features of the FULL layer. Without these the fixture never
    # exercised the interleaved [q|gate] layout or the sigmoid output gate —
    # the exact pair that produced garbage on real weights and was only caught
    # by the llama.cpp comparison (design doc §4d).
    attn_out_gate = 1        # q_proj emits [q_h | gate_h] interleaved per head
    rope_dim_partial = 8     # rope only the first 8 of head_dim 16

    @property
    def key_dim(self):   return self.lin_k_heads * self.lin_k_dim
    @property
    def value_dim(self): return self.lin_v_heads * self.lin_v_dim


def gdn_layer(cfg, W, p, h, state=None):
    """One Gated DeltaNet layer over a chunk of the sequence.

    `state` = (S, conv_win) carried across chunks, exactly like the C runner
    must do: S is the per-v-head outer-product accumulator and conv_win holds
    the last kernel-1 raw channel rows the causal conv still needs. Passing it
    in/out is what makes "prefill in one shot" == "decode token by token";
    that equality is the contract the C implementation is held to.
    Returns (out, new_state)."""
    n_tok = h.shape[0]
    kd, vd = cfg.key_dim, cfg.value_dim
    conv_ch = kd * 2 + vd

    # 1) projections
    # separate tensors, matching real qwen35 GGUF (attn_qkv / attn_gate /
    # ssm_beta / ssm_alpha) — see docs/linear-attention-design.md §4b
    qkv = h @ W[p + "in_proj_qkv"].T            # [T, kd*2 + vd]
    z   = h @ W[p + "in_proj_z"].T              # [T, vd]
    b   = h @ W[p + "in_proj_b"].T              # [T, v_heads]
    a   = h @ W[p + "in_proj_a"].T              # [T, v_heads]
    query = qkv[:, 0:kd]
    key   = qkv[:, kd:kd * 2]
    value = qkv[:, kd * 2:]

    # 2) causal depthwise conv1d over concat(q,k,v), then silu
    mixed = np.concatenate([query, key, value], axis=-1)   # [T, conv_ch]
    convw = W[p + "conv1d_w"]                              # [conv_ch, kernel]
    convb = W[p + "conv1d_b"]                              # [conv_ch]
    # Prepend the carried window (zeros at sequence start) so a chunk boundary
    # is indistinguishable from processing the whole sequence at once.
    if state is None:
        conv_win = np.zeros((cfg.conv_kernel - 1, conv_ch), dtype=mixed.dtype)
        S = np.zeros((cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim), dtype=h.dtype)
    else:
        S, conv_win = state[0].copy(), state[1].copy()
    padded = np.concatenate([conv_win, mixed], axis=0)     # [K-1+T, conv_ch]
    out = np.zeros_like(mixed)
    for t in range(n_tok):
        acc = np.zeros(conv_ch, dtype=mixed.dtype)
        for j in range(cfg.conv_kernel):
            acc += convw[:, j] * padded[t + j]
        out[t] = acc + convb
    new_conv_win = padded[n_tok:]                          # last K-1 raw rows
    mixed = silu(out)
    query = mixed[:, 0:kd]
    key   = mixed[:, kd:kd * 2]
    value = mixed[:, kd * 2:]

    # 3) gates
    beta = 1.0 / (1.0 + np.exp(-b))                                   # sigmoid
    # ssm_a is used DIRECTLY (it already holds the negative coefficient); the
    # transformers reference writes -exp(A_log) because its checkpoint stores
    # log|A|. Verified against llama.cpp's graph — see design doc §4d.
    g = W[p + "A_log"] * softplus(a + W[p + "dt_bias"])               # < 0

    # 4) per-head L2 norm on q,k
    q = query.reshape(n_tok, cfg.lin_k_heads, cfg.lin_k_dim)
    k = key.reshape(n_tok, cfg.lin_k_heads, cfg.lin_k_dim)
    q = l2norm(q)
    k = l2norm(k)
    v = value.reshape(n_tok, cfg.lin_v_heads, cfg.lin_v_dim)

    # 5) delta rule; state S per value head: [k_dim, v_dim]
    oscale = 1.0 / np.sqrt(cfg.lin_v_dim)
    core = np.zeros((n_tok, cfg.lin_v_heads, cfg.lin_v_dim), dtype=h.dtype)
    for t in range(n_tok):
        for vh in range(cfg.lin_v_heads):
            # STRIDED key-head sharing: value heads vh and vh+k_heads share key
            # head vh. This said `vh // grp` until 2026-07-28 and the C forward
            # said the same, so the gate was green while BOTH were wrong — the
            # fixture even has k=2/v=4. An oracle that shares the assumption
            # under test is not an oracle; only llama.cpp on a real 16k/32v
            # model (Qwen3.5-4B) could tell the two apart. See design doc §4j.
            kh = vh % cfg.lin_k_heads
            kt, qt, vt = k[t, kh], q[t, kh], v[t, vh]
            decay = np.exp(g[t, vh])
            S[vh] = S[vh] * decay                    # decay FIRST …
            mem = S[vh].T @ kt                       # … then read the state
            S[vh] = S[vh] + beta[t, vh] * np.outer(kt, vt - mem)
            core[t, vh] = (S[vh].T @ qt) * oscale    # 1/sqrt(d_v) output scale
    core = core.reshape(n_tok, cfg.value_dim)

    # 6) gated RMSNorm -- PER HEAD over v_head_dim (in a real GGUF, ssm_norm.weight
    #    is only v_head_dim long and is shared across heads; normalizing over the
    #    whole v_dim is wrong, see the design doc §4b)
    core = core.reshape(n_tok, cfg.lin_v_heads, cfg.lin_v_dim)
    core = rmsnorm(core, W[p + "ssm_norm"])          # broadcast to every head
    core = core.reshape(n_tok, cfg.value_dim) * silu(z)
    return core @ W[p + "out_proj"].T, (S, new_conv_win)


def gdn_forward(cfg, W, x):
    """Hybrid stack: per-layer dispatch on cfg.layer_types."""
    h = x.copy()
    for il in range(cfg.n_layer):
        p = f"blk.{il}."
        attn = np.zeros_like(h)
        if cfg.layer_types[il] == "linear":
            cur = np.stack([rmsnorm(h[t], W[p + "attn_norm"]) for t in range(h.shape[0])])
            attn, _ = gdn_layer(cfg, W, p, cur)
        else:
            # full GQA layer (same math as gqa_forward's inner block)
            n_tok, hdim = h.shape[0], cfg.head_dim
            grp = cfg.n_head // cfg.n_head_kv
            scale = 1.0 / np.sqrt(hdim)
            kc = np.zeros((n_tok, cfg.n_head_kv, hdim), dtype=h.dtype)
            vc = np.zeros((n_tok, cfg.n_head_kv, hdim), dtype=h.dtype)
            gated = getattr(cfg, "attn_out_gate", 0)
            rdim = getattr(cfg, "rope_dim_partial", 0) or hdim
            for t in range(n_tok):
                cur = rmsnorm(h[t], W[p + "attn_norm"])
                # [q_h | gate_h] interleaved per head → stride 2*hdim
                qraw = (cur @ W[p + "q_proj"].T).reshape(cfg.n_head, hdim * (2 if gated else 1))
                q = qraw[:, :hdim]
                gate = qraw[:, hdim:] if gated else None
                kk = (cur @ W[p + "k_proj"].T).reshape(cfg.n_head_kv, hdim)
                vv = (cur @ W[p + "v_proj"].T).reshape(cfg.n_head_kv, hdim)
                if cfg.qk_norm:
                    q = np.stack([rmsnorm(q[i], W[p + "q_norm"]) for i in range(cfg.n_head)])
                    kk = np.stack([rmsnorm(kk[i], W[p + "k_norm"]) for i in range(cfg.n_head_kv)])
                # partial rope: only the first rdim of each head rotates
                q  = np.stack([rope_partial(q[i], t, cfg.rope_theta, rdim) for i in range(cfg.n_head)])
                kk = np.stack([rope_partial(kk[i], t, cfg.rope_theta, rdim) for i in range(cfg.n_head_kv)])
                kc[t], vc[t] = kk, vv
                heads = []
                for i in range(cfg.n_head):
                    kvh = i // grp
                    s = (kc[: t + 1, kvh] @ q[i]) * scale
                    s = np.exp(s - s.max()); s /= s.sum()
                    heads.append(s @ vc[: t + 1, kvh])
                o = np.concatenate(heads)
                if gated:      # SIGMOID, not silu — the GDN z-gate is the silu one
                    o = o * (1.0 / (1.0 + np.exp(-gate.reshape(-1))))
                attn[t] = o @ W[p + "attn_out"].T
        h = h + attn
        ffn = np.zeros_like(h)
        for t in range(h.shape[0]):
            cur = rmsnorm(h[t], W[p + "ffn_norm"])
            ffn[t] = (silu(cur @ W[p + "gate"].T) * (cur @ W[p + "up"].T)) @ W[p + "down"].T
        h = h + ffn
    return h


def gdn_main(out_path):
    cfg = GdnCfg()
    rng = np.random.default_rng(11)

    def mat(*shape):
        return rng.normal(0, 0.05, size=shape)

    kd, vd = cfg.key_dim, cfg.value_dim
    conv_ch = kd * 2 + vd
    W = {}
    for il in range(cfg.n_layer):
        p = f"blk.{il}."
        W[p + "attn_norm"] = 1.0 + mat(cfg.n_embd)
        if cfg.layer_types[il] == "linear":
            W[p + "in_proj_qkv"] = mat(kd * 2 + vd, cfg.n_embd)
            W[p + "in_proj_z"]   = mat(vd, cfg.n_embd)
            W[p + "in_proj_b"]   = mat(cfg.lin_v_heads, cfg.n_embd)
            W[p + "in_proj_a"]   = mat(cfg.lin_v_heads, cfg.n_embd)
            W[p + "conv1d_w"]     = mat(conv_ch, cfg.conv_kernel)
            W[p + "conv1d_b"]     = mat(conv_ch)
            # A_log = log(A), A ~ U(0,16) → exp(A_log) = A > 0, and g = -A*softplus(..) < 0
            # ssm_a stores -A directly (NOT log A) — matches the real GGUF, and
            # keeps exp(g) a decay in (0,1) instead of blowing the state up.
            W[p + "A_log"]        = -rng.uniform(0.05, 1.5, size=cfg.lin_v_heads)
            W[p + "dt_bias"]      = np.ones(cfg.lin_v_heads)
            W[p + "ssm_norm"]     = 1.0 + mat(cfg.lin_v_dim)   # shared across heads
            W[p + "out_proj"]     = mat(cfg.n_embd, vd)
        else:
            hdim = cfg.head_dim
            # [q_h | gate_h] per head when gated → 2× rows
            qmul = 2 if getattr(cfg, "attn_out_gate", 0) else 1
            W[p + "q_proj"]  = mat(cfg.n_head * hdim * qmul, cfg.n_embd)
            W[p + "k_proj"]  = mat(cfg.n_head_kv * hdim, cfg.n_embd)
            W[p + "v_proj"]  = mat(cfg.n_head_kv * hdim, cfg.n_embd)
            W[p + "q_norm"]  = 1.0 + mat(hdim)
            W[p + "k_norm"]  = 1.0 + mat(hdim)
            W[p + "attn_out"] = mat(cfg.n_embd, cfg.n_head * hdim)
        W[p + "ffn_norm"] = 1.0 + mat(cfg.n_embd)
        W[p + "gate"] = mat(cfg.ff_dense, cfg.n_embd)
        W[p + "up"]   = mat(cfg.ff_dense, cfg.n_embd)
        W[p + "down"] = mat(cfg.n_embd, cfg.ff_dense)
    W = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in W.items()}

    x = rng.normal(0, 0.5, size=(cfg.n_tokens, cfg.n_embd)).astype(np.float32)
    expect = gdn_forward(cfg, W, x)
    assert np.isfinite(expect).all(), "GDN reference produced NaN/Inf"

    records = dict(W)
    records["input"] = x
    records["expect"] = expect
    # layer_types as floats: 0 = linear, 1 = full
    records["layer_types"] = np.array(
        [0.0 if t == "linear" else 1.0 for t in cfg.layer_types], dtype=np.float32)
    records["config"] = np.array([
        cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_head_kv, cfg.head_dim,
        cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim,
        cfg.conv_kernel, cfg.ff_dense, cfg.rope_theta, cfg.n_tokens,
        cfg.qk_norm, cfg.n_vocab,
        getattr(cfg, "attn_out_gate", 0), getattr(cfg, "rope_dim_partial", 0),
    ], dtype=np.float64)

    with open(out_path, "wb") as f:
        f.write(b"DS4XVEC1")
        f.write(struct.pack("<I", len(records)))
        for name, arr in records.items():
            a = np.ascontiguousarray(arr, dtype=np.float32)
            nb = name.encode()
            f.write(struct.pack("<Q", len(nb)) + nb)
            f.write(struct.pack("<I", a.ndim))
            for d in a.shape:
                f.write(struct.pack("<Q", d))
            f.write(a.tobytes())
    print(f"wrote {len(records)} gated-deltanet records to {out_path} "
          f"(layers {cfg.layer_types}, expect mean±std "
          f"{expect.mean():.4f}±{expect.std():.4f})")


def qwen_main(out_path):
    cfg = QwenCfg()
    rng = np.random.default_rng(7)

    def mat(*shape):
        return rng.normal(0, 0.05, size=shape)

    hdim = cfg.head_dim
    W = {}
    for il in range(cfg.n_layer):
        p = f"blk.{il}."
        W[p + "attn_norm"] = 1.0 + mat(cfg.n_embd)
        W[p + "q_proj"] = mat(cfg.n_head * hdim, cfg.n_embd)
        W[p + "k_proj"] = mat(cfg.n_head_kv * hdim, cfg.n_embd)
        W[p + "v_proj"] = mat(cfg.n_head_kv * hdim, cfg.n_embd)
        W[p + "q_norm"] = 1.0 + mat(hdim)
        W[p + "k_norm"] = 1.0 + mat(hdim)
        W[p + "attn_out"] = mat(cfg.n_embd, cfg.n_head * hdim)
        W[p + "ffn_norm"] = 1.0 + mat(cfg.n_embd)
        W[p + "gate"] = mat(cfg.ff_dense, cfg.n_embd)
        W[p + "up"] = mat(cfg.ff_dense, cfg.n_embd)
        W[p + "down"] = mat(cfg.n_embd, cfg.ff_dense)
    W = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in W.items()}

    x = rng.normal(0, 0.5, size=(cfg.n_tokens, cfg.n_embd)).astype(np.float32)
    expect = gqa_forward(cfg, W, x)
    assert np.isfinite(expect).all(), "gqa reference produced NaN/Inf"

    tok_embd = rng.normal(0, 0.05, size=(cfg.n_vocab, cfg.n_embd)).astype(np.float32)
    out_norm = (1.0 + rng.normal(0, 0.05, size=cfg.n_embd)).astype(np.float32)
    out_w = rng.normal(0, 0.05, size=(cfg.n_vocab, cfg.n_embd)).astype(np.float32)
    tokens = rng.integers(0, cfg.n_vocab, size=cfg.n_tokens).astype(np.int32)
    he = gqa_forward(cfg, W, tok_embd[tokens])
    last = rmsnorm(he[-1], out_norm)
    expect_logits = (out_w @ last).astype(np.float32)
    W["token_embd"] = tok_embd
    W["output_norm"] = out_norm
    W["output"] = out_w

    records = dict(W)
    records["input"] = x
    records["expect"] = expect
    records["tokens"] = tokens.astype(np.float32)
    records["expect_logits"] = expect_logits
    records["config"] = np.array([
        cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_head_kv, cfg.head_dim,
        cfg.ff_dense, cfg.rope_theta, cfg.n_tokens, cfg.qk_norm, cfg.n_vocab,
    ], dtype=np.float64)

    with open(out_path, "wb") as f:
        f.write(b"DS4XVEC1")
        f.write(struct.pack("<I", len(records)))
        for name, arr in records.items():
            a = np.ascontiguousarray(arr, dtype=np.float32)
            nb = name.encode()
            f.write(struct.pack("<Q", len(nb)) + nb)
            f.write(struct.pack("<I", a.ndim))
            for d in a.shape:
                f.write(struct.pack("<Q", d))
            f.write(a.tobytes())
    write_qwen3_gguf(out_path + ".gguf", cfg, W)
    print(f"wrote {len(records)} qwen3 records to {out_path} (+ tiny GGUF) "
          f"(expect mean±std {expect.mean():.4f}±{expect.std():.4f})")


def main(out_path, q_rank=None, quant=None, n_shared=None):
    cfg = Cfg()
    if q_rank is not None:
        cfg.q_rank = q_rank        # 0 = non-LoRA q (full attn_q projection)
    if n_shared is not None:
        cfg.n_shared = n_shared    # 0 = MoE with NO shared expert (Qwen3-MoE)
    rng = np.random.default_rng(42)

    def mat(*shape):
        return rng.normal(0, 0.05, size=shape)

    qd = cfg.nope + cfg.rope_dim
    W = {}
    for il in range(cfg.n_layer):
        p = f"blk.{il}."
        W[p + "attn_norm"] = 1.0 + mat(cfg.n_embd)
        if cfg.q_rank > 0:
            W[p + "q_a"] = mat(cfg.q_rank, cfg.n_embd)
            W[p + "q_a_norm"] = 1.0 + mat(cfg.q_rank)
            W[p + "q_b"] = mat(cfg.n_head * qd, cfg.q_rank)
        else:
            W[p + "q_proj"] = mat(cfg.n_head * qd, cfg.n_embd)
        W[p + "kv_a"] = mat(cfg.kv_rank + cfg.rope_dim, cfg.n_embd)
        W[p + "kv_a_norm"] = 1.0 + mat(cfg.kv_rank)
        W[p + "kv_b"] = mat(cfg.n_head * (cfg.nope + cfg.v_dim), cfg.kv_rank)
        W[p + "attn_out"] = mat(cfg.n_embd, cfg.n_head * cfg.v_dim)
        W[p + "ffn_norm"] = 1.0 + mat(cfg.n_embd)
        if il < cfg.dense_lead:
            W[p + "gate"] = mat(cfg.ff_dense, cfg.n_embd)
            W[p + "up"] = mat(cfg.ff_dense, cfg.n_embd)
            W[p + "down"] = mat(cfg.n_embd, cfg.ff_dense)
        else:
            W[p + "router"] = mat(cfg.n_expert, cfg.n_embd)
            # small nonzero bias so it actually affects selection in the test
            W[p + "e_score_bias"] = rng.normal(0, 0.3, size=cfg.n_expert)
            W[p + "e_gate"] = mat(cfg.n_expert, cfg.ff_exp, cfg.n_embd)
            W[p + "e_up"] = mat(cfg.n_expert, cfg.ff_exp, cfg.n_embd)
            W[p + "e_down"] = mat(cfg.n_expert, cfg.n_embd, cfg.ff_exp)
            if cfg.n_shared > 0:
                W[p + "s_gate"] = mat(cfg.ff_exp, cfg.n_embd)
                W[p + "s_up"] = mat(cfg.ff_exp, cfg.n_embd)
                W[p + "s_down"] = mat(cfg.n_embd, cfg.ff_exp)

    # fp32 end-to-end so the reference matches the C side's arithmetic width
    # (fp64-vs-fp32 drift would eat the whole comparison budget).
    W = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in W.items()}

    # --quant q8_0: quantize the layer matrices whose inner dim is ×32 to Q8_0
    # for the GGUF, and replace the fp32 weights with their DEQUANTIZED form so
    # expect/bundle use the exact values ds4x's on-the-fly dequant produces.
    qbytes = {}
    if quant == "q8_0":
        for il in range(cfg.n_layer):
            for rec_name, gguf_suffix in GGUF_NAMES.items():
                key = f"blk.{il}.{rec_name}"
                arr = W.get(key)
                if arr is None or arr.ndim < 2 or arr.shape[-1] % 32 != 0:
                    continue
                raw, deq = q8_0_encode(arr)
                W[key] = deq
                qbytes[f"blk.{il}.{gguf_suffix}"] = raw

    x = rng.normal(0, 0.5, size=(cfg.n_tokens, cfg.n_embd)).astype(np.float32)
    expect = forward(cfg, W, x)
    # macOS Accelerate BLAS is known to pollute FP status flags (spurious
    # RuntimeWarnings in matmul); what matters is that the values are finite.
    assert np.isfinite(expect).all(), "reference forward produced NaN/Inf"

    # ---- full pipeline: tokens → embed → forward → output head → logits ---
    tok_embd = rng.normal(0, 0.05, size=(cfg.n_vocab, cfg.n_embd)).astype(np.float32)
    out_norm = (1.0 + rng.normal(0, 0.05, size=cfg.n_embd)).astype(np.float32)
    out_w = rng.normal(0, 0.05, size=(cfg.n_vocab, cfg.n_embd)).astype(np.float32)
    tokens = rng.integers(0, cfg.n_vocab, size=cfg.n_tokens).astype(np.int32)
    xe = tok_embd[tokens]                       # [n_tok, n_embd]
    he = forward(cfg, W, xe)
    last = rmsnorm(he[-1], out_norm)
    expect_logits = (out_w @ last).astype(np.float32)   # logits for last token
    W["token_embd"] = tok_embd
    W["output_norm"] = out_norm
    W["output"] = out_w

    records = dict(W)
    records["input"] = x
    records["expect"] = expect
    records["tokens"] = tokens.astype(np.float32)
    records["expect_logits"] = expect_logits
    records["config"] = np.array([
        cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.kv_rank, cfg.q_rank,
        cfg.nope, cfg.rope_dim, cfg.v_dim, cfg.n_expert, cfg.top_k,
        cfg.n_shared, cfg.ff_exp, cfg.ff_dense, cfg.dense_lead,
        cfg.rope_theta, cfg.n_tokens,
        cfg.gating_func, cfg.weights_norm, cfg.weights_scale,
    ], dtype=np.float64)

    with open(out_path, "wb") as f:
        f.write(b"DS4XVEC1")
        f.write(struct.pack("<I", len(records)))
        for name, arr in records.items():
            a = np.ascontiguousarray(arr, dtype=np.float32)
            nb = name.encode()
            f.write(struct.pack("<Q", len(nb)) + nb)
            f.write(struct.pack("<I", a.ndim))
            for d in a.shape:
                f.write(struct.pack("<Q", d))
            f.write(a.tobytes())
    write_model_gguf(out_path + ".gguf", cfg, W, qbytes=qbytes)
    print(f"wrote {len(records)} records to {out_path} (+ tiny GGUF) "
          f"(expect mean±std {expect.mean():.4f}±{expect.std():.4f})")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "build/fixtures/ds4x_vectors.bin"
    # "--arch qwen3" emits the GQA (dense) fixture instead of the MLA-MoE one.
    if "--arch" in sys.argv:
        _a = sys.argv[sys.argv.index("--arch") + 1]
        if _a == "qwen3":
            qwen_main(out)
            sys.exit(0)
        if _a in ("qwen3next", "gdn"):
            gdn_main(out)
            sys.exit(0)
    # optional: "--q-rank N" (N=0 exercises the non-LoRA q projection path)
    qr = None
    if "--q-rank" in sys.argv:
        qr = int(sys.argv[sys.argv.index("--q-rank") + 1])
    ns = None
    if "--n-shared" in sys.argv:
        ns = int(sys.argv[sys.argv.index("--n-shared") + 1])
    q = None
    if "--quant" in sys.argv:
        q = sys.argv[sys.argv.index("--quant") + 1]
    main(out, q_rank=qr, quant=q, n_shared=ns)
