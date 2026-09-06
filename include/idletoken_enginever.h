/* idletoken_enginever.h — report the llama.cpp engine build version (v2 WS-C3).
 *
 * The one hard cluster invariant that replaced OS homogeneity (v2 plan §1.4):
 * every node runs the SAME llama.cpp build. ggml-RPC already fail-closes on a
 * protocol mismatch ("RPC server version mismatch"), but that error surfaces
 * mid-load with no machine name attached; the coordinator therefore compares
 * version strings at HELLO time and refuses with a sentence naming the machine
 * that has to upgrade.
 *
 * The version string is what `llama-server --version` prints on its first
 * line with the "version: " prefix stripped — e.g. "1 (0a50d99)" — so it
 * carries the pinned commit SHA and changes whenever the pin does.
 *
 * C only. No C++. */
#ifndef IDLETOKEN_ENGINEVER_H
#define IDLETOKEN_ENGINEVER_H

#include <stddef.h>

/* Max bytes of an engine version string on the wire (incl. NUL). */
#define IDLETOKEN_ENGINE_VERSION_MAX 64

/* Runs `llama_server_bin --version` and writes the first line (without the
 * "version: " prefix) to `out`. Returns 0, or -1 when the binary cannot be
 * run or prints nothing recognizable — callers must treat that as "cannot
 * prove the invariant" and refuse, not guess.
 *
 * TEST ONLY: IDLETOKEN_TEST_ENGINE_VERSION overrides the answer and prints a
 * loud banner — it exists so the version-mismatch refusal can be exercised
 * without actually building two engine versions. Never set it outside a test
 * harness. */
int idletoken_engine_version(const char *llama_server_bin,
                             char *out, size_t cap);

/* Does this engine binary carry the node-local MoE RPC command set (patch
 * 0005-rpc-node-local-moe.patch)? The version string above is a function of
 * the upstream pin only, so two builds of the same pin with different patch
 * series report the SAME version; and a 0005 llama-server sends
 * GET_DEVICE_TYPE to every rpc-server it registers, which an older
 * rpc-server answers by dropping the connection mid-load. Capabilities must
 * therefore be read from the ENGINE binary that will run, not from the build
 * of the supervisor around it (2026-09-05: a new worker advertised the new
 * command set for an old rpc-server and the cluster died in the engine).
 *
 * The probe scans the binary for a string that exists only in the patched
 * sources ("selected-expert ranges"). Returns 1 when present, 0 when absent,
 * -1 when the file cannot be read. Cheap: one sequential read, no execution. */
int idletoken_engine_has_node_local_moe(const char *engine_bin);

#endif /* IDLETOKEN_ENGINEVER_H */
