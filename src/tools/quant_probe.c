/* quant_probe — what precision does the coordinator think a GGUF is?
 *
 * The single question `/idletoken/v1/stats` answers with its `quant` field,
 * asked without loading a model or touching a byte of VRAM. It exists because
 * "the coordinator reports an empty precision" was reproducible on a real node
 * but not attributable from the outside: the answer depends on whether the run
 * uses the static registry manifest or the auto-generated one, and only the
 * coordinator knows which.
 *
 * Reads the GGUF HEADER only (idletoken_model_from_gguf), so it runs in
 * milliseconds against a 10 GB weight file on the machine that actually serves
 * it.
 *
 *   quant_probe <path-to.gguf> [--model-id ID]
 *
 * Prints the same two facts the coordinator settles at startup: the model id it
 * resolves to, and the precision that id + file name name. Exit 0 always —
 * this is a probe, not a gate; the caller reads the output.
 */
#include <stdio.h>
#include <string.h>

#include "idletoken_model.h"
#include "idletoken_model_auto.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path-to.gguf> [--model-id ID]\n", argv[0]);
        return 2;
    }
    const char *gguf = argv[1];
    const char *model_id = NULL;
    for (int i = 2; i < argc; i++)
        if (!strcmp(argv[i], "--model-id") && i + 1 < argc) model_id = argv[++i];

    const idletoken_model_spec *m = NULL;
    idletoken_auto_model am;
    char err[256] = "";
    if (model_id) {
        m = idletoken_model_get(model_id);
        if (!m) { printf("model_id=%s NOT IN REGISTRY\n", model_id); return 0; }
        printf("source=registry\n");
    } else {
        /* Exactly what coord_main.c does without --model-id. */
        if (idletoken_model_from_gguf(gguf, &am, err, sizeof(err)) != 0) {
            printf("auto manifest FAILED: %s\n", err);
            return 0;
        }
        m = &am.spec;
        printf("source=auto-manifest\n");
    }
    printf("id=%s\n", m->id ? m->id : "(null)");
    printf("n_variants=%u\n", (unsigned)m->n_variants);
    printf("registry_row_for_id=%s\n",
           (m->id && idletoken_model_get(m->id)) ? "present" : "absent");
    printf("quant=%s\n", idletoken_model_quant_from_gguf(m, gguf));
    return 0;
}
