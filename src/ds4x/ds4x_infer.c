/* ds4x_infer.c — standalone CPU inference over a ds4x model (Phase B tool).
 *
 * Loads a GGUF, prefills a token-id sequence, then greedy-decodes N tokens.
 * No GPU, no cluster — reuses the exact verified ds4x path (model load →
 * runner → embed → output head). Its job is the G-GLM alignment harness:
 * on the DGX, feed the SAME token ids to this tool and to llama.cpp and
 * compare logits / argmax token by token, WITHOUT standing up the whole
 * pipeline first. Tokenizer is out of scope here (feed ids directly) — that
 * keeps the comparison purely numeric.
 *
 * Usage:
 *   ds4x_infer <model.gguf> --tokens 1,2,3 [--n-predict N] [--ctx C] [--quiet]
 *   ds4x_infer <model.gguf> --text "..."  [--n-predict N] ...   (needs a GGUF
 *                                          that also carries tokenizer.ggml.*)
 *   ds4x_infer <model.gguf> --selftest <vectors.bin>   (argmax vs reference)
 *
 * C only. No C++. Builds anywhere the ds4x reference builds. */
/* clock_gettime/struct timespec are hidden under strict -std=c99; must precede
 * every include (same trap as MADV_WILLNEED in ds4x_model.c). */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "idletoken_ds4x.h"
#include "idletoken_ds4x_tok.h"
#ifdef IDLETOKEN_DS4X_CUDA
#include "idletoken_ds4x_cuda.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int argmax(const float *v, uint32_t n) {
    int best = 0;
    for (uint32_t i = 1; i < n; i++) if (v[i] > v[best]) best = (int)i;
    return best;
}

/* Parse "1,2,3" into out[], returns count (<= cap). */
static uint32_t parse_tokens(const char *s, int32_t *out, uint32_t cap) {
    uint32_t n = 0;
    while (*s && n < cap) {
        out[n++] = (int32_t)strtol(s, (char **)&s, 10);
        while (*s == ',' || *s == ' ') s++;
    }
    return n;
}

/* Read the "tokens"/"expect_logits" records from a ds4x_ref.py bundle for the
 * selftest. Minimal reader — mirrors ds4x_forward_test's loader. */
static int read_bundle_record(const char *path, const char *want,
                              float **out, uint64_t *nelem) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    uint32_t nrec;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "DS4XVEC1", 8) ||
        fread(&nrec, 4, 1, f) != 1) { fclose(f); return -1; }
    for (uint32_t i = 0; i < nrec; i++) {
        uint64_t nl; char name[128];
        uint32_t ndim; uint64_t dims[4], total = 1;
        if (fread(&nl, 8, 1, f) != 1 || nl >= sizeof(name)) { fclose(f); return -1; }
        if (fread(name, 1, (size_t)nl, f) != nl) { fclose(f); return -1; }
        name[nl] = 0;
        if (fread(&ndim, 4, 1, f) != 1 || ndim > 4) { fclose(f); return -1; }
        for (uint32_t d = 0; d < ndim; d++) { if (fread(&dims[d], 8, 1, f) != 1) { fclose(f); return -1; } total *= dims[d]; }
        if (!strcmp(name, want)) {
            float *buf = malloc((size_t)total * 4);
            if (!buf || fread(buf, 4, (size_t)total, f) != total) { free(buf); fclose(f); return -1; }
            *out = buf; *nelem = total; fclose(f); return 0;
        }
        if (fseek(f, (long)(total * 4), SEEK_CUR) != 0) { fclose(f); return -1; }
    }
    fclose(f);
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> --tokens 1,2,3 [--n-predict N] "
                        "[--ctx C] [--quiet] | --selftest <vectors.bin>\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const char *tokspec = NULL, *selftest = NULL, *textspec = NULL;
    uint32_t n_predict = 8, ctx = 0;
    int quiet = 0;
    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokspec = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) textspec = argv[++i];
        else if (!strcmp(argv[i], "--n-predict") && i + 1 < argc) n_predict = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--selftest") && i + 1 < argc) selftest = argv[++i];
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

#ifdef IDLETOKEN_DS4X_CUDA
    /* Same knob the worker uses (IDLETOKEN_MAX_VRAM_MB), so a cap can be exercised
     * from the CLI without standing up a cluster — and so this tool cannot
     * quietly fill a card the worker would have stayed off. */
    {
        const char *cap = getenv("IDLETOKEN_MAX_VRAM_MB");
        if (cap && *cap) {
            uint64_t mb = strtoull(cap, NULL, 10);
            ds4x_cuda_set_budget(mb * 1024ull * 1024ull);
            fprintf(stderr, "ds4x_infer: VRAM budget %llu MB (IDLETOKEN_MAX_VRAM_MB)\n",
                    (unsigned long long)mb);
        }
    }
#endif

    char err[256];
    /* Startup is dominated by these two loads; time them separately so the
     * next optimization targets whichever actually costs (the tokenizer once
     * hid a 30 s O(n^2) behind what looked like "weight loading"). */
    struct timespec ta, tb, tc;
    clock_gettime(CLOCK_MONOTONIC, &ta);
    ds4x_model *m = ds4x_model_load(path, 0, 0, err, sizeof(err));
    if (!m) { fprintf(stderr, "load: %s\n", err); return 1; }
    const ds4x_config *c = ds4x_model_config(m);
    clock_gettime(CLOCK_MONOTONIC, &tb);

    /* Tokenizer is optional (the model GGUF must carry tokenizer.ggml.* for
     * --text or decoded output; the tiny test model has none). */
    ds4x_tokenizer *tok = ds4x_tok_load(path, err, sizeof(err));
    clock_gettime(CLOCK_MONOTONIC, &tc);
    fprintf(stderr, "ds4x load: weights %.2f s, tokenizer %.2f s\n",
            (double)(tb.tv_sec - ta.tv_sec) + (double)(tb.tv_nsec - ta.tv_nsec) / 1e9,
            (double)(tc.tv_sec - tb.tv_sec) + (double)(tc.tv_nsec - tb.tv_nsec) / 1e9);

    int32_t toks[4096];
    uint32_t n_tok = 0;
    float *ref_logits = NULL; uint64_t ref_n = 0;
    if (textspec) {
        if (!tok) { fprintf(stderr, "--text needs tokenizer.ggml.* in the GGUF (%s)\n", err); return 1; }
        int64_t ne = ds4x_tok_encode(tok, textspec, toks, 4096);
        if (ne <= 0) { fprintf(stderr, "encode produced no tokens\n"); return 1; }
        n_tok = (uint32_t)ne;
    } else if (selftest) {
        float *tf = NULL; uint64_t tn = 0;
        if (read_bundle_record(selftest, "tokens", &tf, &tn) != 0 ||
            read_bundle_record(selftest, "expect_logits", &ref_logits, &ref_n) != 0) {
            fprintf(stderr, "selftest: cannot read tokens/expect_logits from %s\n", selftest);
            return 1;
        }
        n_tok = (uint32_t)tn;
        for (uint32_t i = 0; i < n_tok && i < 4096; i++) toks[i] = (int32_t)tf[i];
        free(tf);
        n_predict = 0;   /* just check prefill argmax vs the reference */
    } else if (tokspec) {
        n_tok = parse_tokens(tokspec, toks, 4096);
    } else {
        fprintf(stderr, "need --tokens or --selftest\n"); return 1;
    }
    if (n_tok == 0) { fprintf(stderr, "no input tokens\n"); return 1; }

    if (ctx == 0) ctx = n_tok + n_predict;
    ds4x_runner *r = ds4x_runner_create(m, ctx, err, sizeof(err));
    if (!r) { fprintf(stderr, "runner: %s\n", err); return 1; }

    float *hid = malloc((size_t)ctx * c->n_embd * sizeof(float));
    float *logits = malloc((size_t)c->n_vocab * sizeof(float));
    if (!hid || !logits) { fprintf(stderr, "oom\n"); return 1; }

    /* prefill */
    if (ds4x_embed_tokens(m, toks, n_tok, hid) != 0) { fprintf(stderr, "embed\n"); return 1; }
    if (ds4x_runner_run(r, hid, n_tok, 0) != 0) { fprintf(stderr, "prefill\n"); return 1; }
    if (ds4x_output_logits(m, hid + (size_t)(n_tok - 1) * c->n_embd, logits) != 0) {
        fprintf(stderr, "output\n"); return 1;
    }
    int next = argmax(logits, c->n_vocab);

    if (selftest) {
        int ref_arg = argmax(ref_logits, (uint32_t)ref_n);
        double mx = 0;
        for (uint32_t i = 0; i < c->n_vocab && i < ref_n; i++) {
            double d = logits[i] - ref_logits[i]; if (d < 0) d = -d;
            if (d > mx) mx = d;
        }
        printf("selftest: argmax %d (ref %d), max|Δlogit| %.3e\n", next, ref_arg, mx);
        free(ref_logits);
        if (next != ref_arg || mx >= 1e-4) { printf("DS4X_INFER_SELFTEST_FAIL\n"); return 1; }
        printf("DS4X_INFER_SELFTEST_OK\n");
        return 0;
    }

    if (!quiet) printf("prefill %u tokens → next %d\n", n_tok, next);
    /* greedy decode, collecting generated ids */
    int32_t gen[4096];
    uint32_t ngen = 0;
    uint32_t pos = n_tok;
    for (uint32_t step = 0; step < n_predict && pos < ctx; step++) {
        float *slot = hid + (size_t)pos * c->n_embd;
        if (ngen < 4096) gen[ngen++] = next;
        if (ds4x_embed_tokens(m, &next, 1, slot) != 0) break;
        if (ds4x_runner_run(r, slot, 1, pos) != 0) break;
        if (ds4x_output_logits(m, slot, logits) != 0) break;
        if (tok && next == ds4x_tok_eos(tok)) { pos++; break; }
        next = argmax(logits, c->n_vocab);
        if (!quiet) printf("  step %u @pos %u → %d\n", step, pos, next);
        pos++;
    }
    if (getenv("IDLETOKEN_DS4X_PROF")) {
        double pj = 0, cv = 0, rc = 0, po = 0;
        ds4x_prof_report(&pj, &cv, &rc, &po);
        const double tot = pj + cv + rc + po;
        fprintf(stderr, "ds4x prof (linear path %.3fs): projections %.3fs (%.1f%%), "
                        "conv+gates %.3fs (%.1f%%), recurrence %.3fs (%.1f%%), "
                        "norm+out_proj %.3fs (%.1f%%)\n",
                tot, pj, tot > 0 ? 100.0 * pj / tot : 0.0,
                cv, tot > 0 ? 100.0 * cv / tot : 0.0,
                rc, tot > 0 ? 100.0 * rc / tot : 0.0,
                po, tot > 0 ? 100.0 * po / tot : 0.0);
    }
    {   /* Non-zero only under IDLETOKEN_DS4X_GDN_CHECK=1. Printed unconditionally
         * when it ran, so a check run can never be mistaken for a clean one. */
        double gmx = 0; uint64_t gn = 0;
        ds4x_gdn_check_report(&gmx, &gn);
        if (gn > 0)
            fprintf(stderr, "ds4x gdn check: %llu chunks, max |Δ| cpu vs gpu "
                            "recurrence = %.3e\n", (unsigned long long)gn, gmx);
    }
    printf("generated %u tokens\n", pos - n_tok);
    if (tok && ngen > 0) {
        char *text = ds4x_tok_decode(tok, gen, ngen, 0);
        if (text) { printf("text: %s\n", text); free(text); }
    }

    free(hid); free(logits);
    ds4x_runner_free(r);
#ifdef IDLETOKEN_DS4X_CUDA
    {   /* Where did matvec time go? Kernel vs copies/launch/sync — printed so
         * the next optimization targets a measured number, not a hunch. */
        double kms = 0, tms = 0; uint64_t nc = 0;
        ds4x_cuda_stats(&kms, &tms, &nc);
        if (nc > 0)
            fprintf(stderr,
                    "ds4x cuda: %llu matvecs  kernel %.0f ms (%.3f ms/call)  "
                    "total %.0f ms (%.3f ms/call)  overhead %.0f%%\n",
                    (unsigned long long)nc, kms, kms / (double)nc,
                    tms, tms / (double)nc,
                    tms > 0 ? 100.0 * (tms - kms) / tms : 0.0);
        uint64_t nrows = 0;
        ds4x_cuda_matmul_stats(&kms, &tms, &nc, &nrows);
        if (nc > 0)
            fprintf(stderr,
                    "ds4x cuda: %llu matmuls (%llu token-rows)  kernel %.0f ms "
                    "(%.3f ms/call)  total %.0f ms  overhead %.0f%%\n",
                    (unsigned long long)nc, (unsigned long long)nrows, kms,
                    kms / (double)nc, tms, tms > 0 ? 100.0 * (tms - kms) / tms : 0.0);
        /* Its own bucket, not folded into matmuls: the whole point of fusing is
         * that three matmuls become one call, so it has to be visible as the
         * matmul count dropping and this count rising. */
        ds4x_cuda_ffn_stats(&kms, &tms, &nc, &nrows);
        if (nc > 0)
            fprintf(stderr,
                    "ds4x cuda: %llu ffns (%llu token-rows)  kernel %.0f ms "
                    "(%.3f ms/call)  total %.0f ms  overhead %.0f%%\n",
                    (unsigned long long)nc, (unsigned long long)nrows, kms,
                    kms / (double)nc, tms, tms > 0 ? 100.0 * (tms - kms) / tms : 0.0);
        ds4x_cuda_proj_stats(&kms, &tms, &nc, &nrows);
        if (nc > 0)
            fprintf(stderr,
                    "ds4x cuda: %llu projfans (%llu token-rows)  kernel %.0f ms "
                    "(%.3f ms/call)  total %.0f ms  overhead %.0f%%\n",
                    (unsigned long long)nc, (unsigned long long)nrows, kms,
                    kms / (double)nc, tms, tms > 0 ? 100.0 * (tms - kms) / tms : 0.0);
        ds4x_cuda_gdn_stats(&kms, &tms, &nc);
        if (nc > 0)
            fprintf(stderr,
                    "ds4x cuda: %llu gdn chunks  kernel %.0f ms (%.3f ms/call)  "
                    "total %.0f ms (%.3f ms/call)  overhead %.0f%%\n",
                    (unsigned long long)nc, kms, kms / (double)nc,
                    tms, tms / (double)nc,
                    tms > 0 ? 100.0 * (tms - kms) / tms : 0.0);
    }
#endif
    ds4x_model_free(m);
    if (tok) ds4x_tok_free(tok);
    return 0;
}
