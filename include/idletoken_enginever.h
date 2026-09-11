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

/* Does an engine binary carry the withdrawn patch 0007 (staged CUDA host
 * buffers)? Retained for inspecting old binaries only. The staged pageable
 * tail measured too slow to count as usable expert capacity, so this result
 * must never bypass the page-lock admission cap. Same scan as above,
 * different needle ("staged host buffer:"). 1 / 0 / -1 as above. */
int idletoken_engine_has_host_staging(const char *engine_bin);

/* Engine capability flags a worker reports in RESOURCE_REPORT (the u32 that
 * was reserved before 2026-09-07; old workers send 0 = no capabilities). */
#define IDLETOKEN_ENGINE_FLAG_HOST_STAGING 0x1u   /* legacy patch 0007 marker; never bypasses admission */

#include <stdint.h>

/* Runs `rpc_server_bin --pinned-probe` (patch 0006) and returns the bytes of
 * host memory this machine's GPU can page-lock, or 0 when it cannot be
 * measured (no such flag in the binary, no GPU, unified memory). The planner
 * caps the routed experts it keeps in a node's RAM at this number: the engine
 * stores them in the GPU's page-locked host buffers and silently drops to
 * pageable memory — several times slower over PCIe — when the lock fails
 * (2026-09-07, a 48 GiB Windows test node at 31.4 GiB).
 *
 * What the number IS (established 2026-09-07 evening, DXGI QueryVideoMemoryInfo
 * on both testbed boxes + Microsoft's "Calculating Graphics Memory" page): on
 * Windows the ceiling is WDDM's per-process NON-LOCAL memory budget, and the
 * OS computes the underlying SharedSystemMemory as
 *   MIN(RAM * 80 %, MAX(RAM - 16 GiB, RAM * 50 %))
 * with the budget 0.75 GiB below that. cudaHostAlloc fails exactly when the
 * process's non-local usage would cross the budget (a 64 GiB Windows test
 * node: 63.66 GiB RAM -> 47.66 shared -> 46.91 budget, failure at 46 +
 * 1 GiB; a 48 GiB Windows test node: 47.75 ->
 * 31.75 -> 31.00, failure at 30 + 1). The 16 GiB is the fixed-reserve term in
 * the middle branch, not the discrete card's VRAM; the integrated GPU and the
 * software renderer report the same shared figure, while the "50 %" folklore
 * is the same formula below 32 GiB of RAM. Not configurable: the OS sets it and
 * a driver may only lower it. Linux has no such budget. The probe stays the
 * source of truth (a future Windows build could change the formula), but the
 * formula predicts it to within the probe's 512 MiB granularity. */
uint64_t idletoken_engine_probe_pinned(const char *rpc_server_bin);

/* Read the cached probe without starting a new allocation test. This is the
 * only safe operation for a client's ordinary resource refresh: a fresh probe
 * deliberately locks most of system RAM for up to a minute. Returns 0 for a
 * missing, malformed, or RAM-size-mismatched cache entry. */
uint64_t idletoken_engine_cached_pinned_ceiling(uint64_t ram_total);

/* The probe above, cached per machine: the result is keyed by `ram_total`
 * and kept in the user's cache directory (LOCALAPPDATA on Windows,
 * XDG_CACHE_HOME/~/.cache elsewhere), because measuring means locking most of
 * the RAM for up to a minute. Returns 0 when the probe cannot measure. */
uint64_t idletoken_engine_pinned_ceiling(const char *rpc_server_bin, uint64_t ram_total);

#endif /* IDLETOKEN_ENGINEVER_H */
