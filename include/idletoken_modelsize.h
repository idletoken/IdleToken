/* idletoken_modelsize.h — ONE answer to "how big is the model we are about to
 * load?", for every consumer of idletoken_llm_model_size.
 *
 * WHY THIS EXISTS (2026-08-19, measured on a 16 GiB discrete card — see
 * results/llamacpp-multislot-big-win-20260819.md §4). The coordinator sized its
 * memory budget from `variants[default_variant]` while `--llama-gguf` decided
 * which file the engine actually opened. Serving Qwen3.5-27B Q4_K_M (15.59 GiB)
 * with no `--quant` therefore planned for the manifest default IQ2_XXS
 * (7.98 GiB), understated the weights by 7.6 GiB, and derived 2 sequence slots
 * where 1 is correct. Nothing reconciled the two numbers, and the manifest was
 * not even missing the right one: `Q4_K_M -> 16740812704` matches the file byte
 * for byte.
 *
 * The engine's own fit check caught that particular machine, but it is a backup
 * path: it fires only when llama.cpp cannot fit the model AT ALL, and says
 * nothing about a machine with room for the weights and not for the extra slots
 * the wrong budget invented — which is the shape of the 08-18 Windows freeze.
 *
 * THE RULE: the budget follows the file that will really be loaded, and the
 * resolver says out loud where its number came from. Three sources, in
 * priority order (idletoken_model_size_resolve):
 *
 *   1. an explicit GGUF path      — stat it; the bytes on disk are not an
 *                                   opinion. Split sets are summed.
 *   2. (model, quant)             — the manifest variant for that precision.
 *   3. model only                 — the default variant, WITH A WARNING: if the
 *                                   engine opens a different quant, the slot
 *                                   count and the tensor split are both wrong.
 *
 * Every caller uses this function. Three call sites resolving the precision
 * "the same way" is how the advisor and the planner drift into disagreeing
 * about what a machine can run — the lesson already written down in
 * idletoken_mode_decide_quant's comment.
 */
#ifndef IDLETOKEN_MODELSIZE_H
#define IDLETOKEN_MODELSIZE_H

#include <stddef.h>
#include <stdint.h>

#include "idletoken_model.h"
#include "idletoken_plan.h"

/* Bytes a GGUF occupies on disk. A split set ("-00001-of-00004.gguf") is summed
 * over all its parts, because llama.cpp loads the whole set and a size taken
 * from part 1 alone understates the model several-fold — the same class of
 * mistake this file exists to end.
 *
 * `path` must name part 1 of a split set (the part carrying the header);
 * pointing at a later part is a user error and is reported as one.
 *
 * Returns the byte count, or 0 with a sentence in `why` (missing file, missing
 * sibling part, wrong part). Never partially succeeds: an incomplete download
 * returns 0 rather than the sum of what happens to be present. */
uint64_t idletoken_gguf_bytes_on_disk(const char *path, char *why, size_t why_cap);

/* Parse the "-NNNNN-of-NNNNN.gguf" split convention out of a file's BASENAME.
 * `*idx` is this file's 1-based part index, `*total` the part count. Returns 0
 * (and touches nothing) when the name is not a split name. */
int idletoken_gguf_split_parts(const char *basename, unsigned *idx, unsigned *total);

/* Fill `out` with the scheduler's view of the model that will actually be
 * loaded. See the header comment for the three sources.
 *
 *   spec        the registry/manifest entry (required — it carries the shape:
 *               layer count and KV bytes per token, neither of which varies
 *               with the quantization)
 *   quant       the requested precision, or NULL/"" for unspecified
 *   gguf_path   the file the engine will open, or NULL/"" when none is known
 *   why         (optional) receives the source sentence WITHOUT a prefix, e.g.
 *               "the GGUF on disk Qwen3.5-27B-Q4_K_M.gguf (15.59 GiB, manifest
 *               quant Q4_K_M)". Callers print it after "budget from: " so a log
 *               read months later can answer "what was it budgeting against?".
 *               Warnings are appended to the same sentence after " — WARNING:"
 *               so that one print cannot drop them.
 *
 * Returns 0 on success (including every warned case — a warning is a fact about
 * the answer, not a failure to produce one), or -1 when `spec` or `out` is NULL
 * or the spec carries no layers.
 *
 * A GGUF path that cannot be sized does NOT fail the call: it degrades to the
 * manifest with a loud warning, because the engine is about to open that same
 * path and will complain about it far more precisely than we can. What must
 * never happen is degrading QUIETLY. */
int idletoken_model_size_resolve(const idletoken_model_spec *spec,
                                 const char *quant,
                                 const char *gguf_path,
                                 idletoken_llm_model_size *out,
                                 char *why, size_t why_cap);

#endif /* IDLETOKEN_MODELSIZE_H */
