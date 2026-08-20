/* IdleToken Cluster — resource probe.
 *
 * Reports usable VRAM and RAM after subtracting system overhead and safety
 * margins, per docs/architecture.md §5.1. The numbers map 1:1 to the
 * RESOURCE_REPORT wire message (docs/wire-protocol.md).
 *
 * GPU via NVML (Linux: linked nvml.h; Windows: nvml.dll loaded from the
 * driver at runtime). RAM via /proc/meminfo (Linux) or GlobalMemoryStatusEx
 * (Windows). */

#ifndef IDLETOKEN_RESOURCE_H
#define IDLETOKEN_RESOURCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define IDLETOKEN_VRAM_SAFETY_BYTES     (1ull * 1024ull * 1024ull * 1024ull)  /* 1.0 GB */
#define IDLETOKEN_VRAM_WORKSPACE_BYTES  (3ull * 1024ull * 1024ull * 1024ull / 2ull) /* 1.5 GB */
/* Metal working-set reserve, CALIBRATED 2026-08-15 (M4 16 GiB; the llamacpp
 * engine's whole non-weight footprint measured ~133 MiB, and the scheduler
 * charges width-scaled engine buffers separately — plan.c). The old value
 * reused the 1.5 GiB CUDA workspace guess, which double-reserved ~2.3 GiB on
 * a 16 GiB Mac and pushed fitting models into refusals.
 * results/resource-calibration-20260815.md. */
#define IDLETOKEN_METAL_WORKSPACE_BYTES (512ull * 1024ull * 1024ull)          /* 0.5 GB */
#define IDLETOKEN_RAM_SAFETY_BYTES      (4ull * 1024ull * 1024ull * 1024ull)  /* 4.0 GB */

/* Headroom the machine keeps for itself, as an ABSOLUTE amount — applied on
 * top of the "total − used_other − safety" subtraction, and only ever as a
 * backstop.
 *
 * HISTORY, because the shape of this rule changed for a reason (2026-08-16).
 * It used to be a proportional ceiling (70% of RAM), introduced after two
 * hangs on win-a: available memory sat at 128–730 MiB for a whole run while
 * the subtraction believed ~10 GiB would be left over. That was measured
 * against **ds4**, which ALLOCATED the weights — unreclaimable memory, so
 * planning to the last byte really did starve the machine.
 *
 * llama.cpp mmaps the weights instead: they are page cache, and the kernel
 * evicts them under pressure. Measured 2026-08-16 on DGX (119.6 GiB RAM,
 * GLM-5.2 IQ2_XXS = 222 GiB — nearly 2x physical): the model loaded, served,
 * and generated at 0.91 tok/s, RSS plateaued at physical RAM, and
 * **MemAvailable never fell below 113 GiB** across 245 samples. The failure
 * the proportional ceiling defends against does not occur on this engine.
 * (docs/resource-budget-rethink-2026-08.md §5.)
 *
 * Two problems with the old shape, both real:
 *   - it BOUND on 2 of 3 testbed machines, i.e. one hardcoded percentage —
 *     not the careful subtraction under it — decided the whole product's
 *     memory budget;
 *   - the risk it guards (the OS being starved) needs a roughly FIXED reserve,
 *     while a proportion wastes 36 GiB on a 119 GiB box and leaves a 16 GiB
 *     laptop only 4.8.
 *
 * So: leave a mostly-fixed amount and let the subtraction (which measures what
 * is actually in use) govern normally.
 *
 * `total/8` clamped to [4, 12] GiB rather than a flat number, because both
 * extremes are wrong at one end: a flat 8 GiB takes HALF of a 16 GiB laptop,
 * while a flat 4 GiB is thin on a 119 GiB box hosting a 200 GiB working set.
 * The clamp keeps it within a range a human can sanity-check:
 *     16 GiB → 4    64 GiB → 8    119 GiB → 12
 * Override with IDLETOKEN_OS_HEADROOM_GIB for measurement; values outside
 * 1..64 are ignored, so a typo cannot disable the backstop. */
#define IDLETOKEN_OS_HEADROOM_MIN_BYTES (4ull * 1024ull * 1024ull * 1024ull)
#define IDLETOKEN_OS_HEADROOM_MAX_BYTES (12ull * 1024ull * 1024ull * 1024ull)

/* Hardware floor. A node below any of these cannot serve layers, and finding
 * that out at model-load time (or worse: as garbage tokens) is exactly the
 * experience this floor exists to prevent.
 *
 * cc 7.5 = Turing. Say plainly what this is and is not: it is a SUPPORT
 * boundary, not a claim that older cards cannot run the engine. The pinned
 * llama.cpp does support Volta -- ggml-cuda/common.cuh defines
 * GGML_CUDA_CC_VOLTA (700) and gates a dedicated VOLTA_MMA_AVAILABLE
 * tensor-core path on it -- and a real Tesla V100 (cc 7.0, driver 580.173.02)
 * passed every probe and floor check on 2026-08-17 once the floor was lowered.
 * We choose not to carry it.
 *
 * Why 7.5 and not lower, now that the old reason is gone (it used to read
 * "the tensor-core / fp16 paths the ds4 and ds4x kernels are written against",
 * and those kernels were shelved on 2026-08-16 -- Makefile: DS4X_*_OBJ :=
 * # shelved -- so the floor was guarding a dead code path):
 *
 *   - NVIDIA is removing the ground under pre-7.5. CUDA 12.9 already warns
 *     "support for offline compilation for architectures prior to sm_75 will
 *     be removed in a future release", and CUDA 13.0 has removed sm_70
 *     outright (measured with `nvcc --list-gpu-arch` on both: 12.9 emits 70,
 *     13.0 starts at 75). Supporting Volta means pinning the whole Linux
 *     build to a 12.x toolkit for as long as we promise it.
 *   - Every supported architecture is a testbed obligation, not just a
 *     -gencode flag. Nothing in the acceptance ladder covers Volta.
 *
 * So the floor is 7.5 because that is where our toolchain and our testing can
 * both hold, and anyone with older hardware can build for it themselves.
 *
 * The driver floor tracks the CUDA toolkit this build is compiled against,
 * because that is what the runtime demands. It used to be two hand-written
 * constants -- Windows 527.41 (CUDA 12.x) and Linux 580.65 (CUDA 13.0) -- with
 * a comment saying "rebuild against a different toolkit => update these".
 * Nobody does. On 2026-08-16 a Linux build made with CUDA 12.4 still refused
 * an RTX 4090 on driver 550.142, quoting the 580.65 floor of a toolkit it was
 * not built with: a perfectly capable machine turned away with a sentence that
 * was simply false for those bytes. The same coupling had already misfired on
 * Windows (docs/cross-machine-pp-analysis.md: a CUDA 13.3 build refused a
 * 555.85 laptop until it was rebuilt against 12.8).
 *
 * So derive it. CUDART_VERSION is defined by the CUDA headers this
 * translation unit is compiled with, and minor-version compatibility means
 * the floor is a function of the MAJOR version only: any 12.x runs on the
 * 12.0 minimum driver. Values are NVIDIA's published per-toolkit minimums.
 *
 * ⚠ Residual gap, deliberately left visible rather than papered over: what
 * actually loads CUDA at inference time is llama.cpp's backend, built by
 * scripts/build_llamacpp.sh with whatever toolkit that machine has -- not
 * necessarily the one coord/worker were compiled against. When they differ,
 * this floor describes the wrong binary. It is right whenever both are built
 * on the same machine (every path we ship today), and it can no longer be
 * wrong merely because someone forgot to edit a constant. */
#define IDLETOKEN_MIN_CC_MAJOR   7
#define IDLETOKEN_MIN_CC_MINOR   5

/* CUDART_VERSION comes from the CUDA headers; include them when this machine
 * has them. Guarded by __has_include because the macOS build has no CUDA at
 * all, and the header is not on the include path of every target. */
#if defined(__has_include)
#  if __has_include(<cuda_runtime_api.h>)
#    include <cuda_runtime_api.h>
#  endif
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION < 12000
  /* Fail loudly rather than hand back a 12.x floor that is simply wrong for an
   * older toolkit (it would refuse machines that work). No silent fallback. */
  #error "IdleToken needs CUDA 12 or newer; this build sees an older CUDART_VERSION"
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
  /* CUDA 13.x. Linux figure verified on the DGX Spark (driver 580.126.09). */
  #ifdef _WIN32
    #define IDLETOKEN_MIN_DRIVER_MAJOR 580
    #define IDLETOKEN_MIN_DRIVER_MINOR 88
  #else
    #define IDLETOKEN_MIN_DRIVER_MAJOR 580
    #define IDLETOKEN_MIN_DRIVER_MINOR 65
  #endif
#else
  /* CUDA 12.x, and the fallback when no CUDA header is visible (macOS, or a
   * target built without the CUDA include path). Falling back to the LOWER
   * floor is the deliberate direction: the higher floor's only justification
   * is a CUDA 13 build, and a translation unit that cannot see the CUDA
   * headers is not one. The Windows figure is the one verified in practice --
   * a 555.85 laptop runs the 12.8 build. */
  #ifdef _WIN32
    #define IDLETOKEN_MIN_DRIVER_MAJOR 527
    #define IDLETOKEN_MIN_DRIVER_MINOR 41
  #else
    #define IDLETOKEN_MIN_DRIVER_MAJOR 525
    #define IDLETOKEN_MIN_DRIVER_MINOR 60
  #endif
#endif
/* Hard constraint (docs/architecture.md §2): every node needs a GPU with at least this much VRAM
 * (CUDA context + workspace + at least one layer). Unified-memory hosts are
 * exempt — their "VRAM" is host RAM. */
#define IDLETOKEN_MIN_VRAM_BYTES (4ull * 1024ull * 1024ull * 1024ull)

/* Apple Silicon floor. The VRAM floor above is waived for unified-memory hosts
 * (their "VRAM" is host RAM), so a Mac would otherwise have no floor at all.
 * The number it is applied to is Metal's recommendedMaxWorkingSetSize, which
 * on the machines measured so far sits near 74% of physical RAM — so this
 * admits an 8 GB Mac and refuses anything smaller. Same reasoning as the CUDA
 * floor: enough for the runtime's own state plus at least one model layer. */
#define IDLETOKEN_MIN_APPLE_WORKING_SET_BYTES (4ull * 1024ull * 1024ull * 1024ull)

/* Which GPU stack this node runs. Decides which hardware floor applies:
 * compute capability and driver version are CUDA-only concepts and are left
 * zero on Apple Silicon, so idletoken_hw_check must dispatch on this BEFORE
 * it reads them.
 *
 * Deliberately NOT on the wire. RESOURCE_REPORT (docs/wire-protocol.md) is
 * unchanged: the hardware floor is checked locally by the worker before it
 * even sends HELLO, so the coordinator never needs the vendor to decide
 * whether a node may join. What the coordinator DOES need — `unified_memory`
 * and the byte counts — it already gets. This becomes a wire field the day a
 * heterogeneous cluster has to plan around per-vendor throughput rather than
 * per-node bytes. */
typedef enum {
    IDLETOKEN_GPU_VENDOR_UNKNOWN = 0,
    IDLETOKEN_GPU_VENDOR_NVIDIA  = 1,
    IDLETOKEN_GPU_VENDOR_APPLE   = 2,   /* Apple Silicon, Metal, unified memory */
} idletoken_gpu_vendor;

typedef struct {
    /* GPU */
    char     gpu_name[64];
    idletoken_gpu_vendor gpu_vendor;
    uint8_t  cc_major;          /* CUDA only; 0 on Metal */
    uint8_t  cc_minor;          /* CUDA only; 0 on Metal */
    char     driver_version[32]; /* NVML "580.126.09" — empty if unreadable */
    bool     unified_memory;     /* DGX Spark / Grace: 1, discrete VRAM: 0 */
    uint64_t vram_total;
    uint64_t vram_used_other;    /* in use by *other* processes */
    uint64_t vram_usable;        /* total - used_other - safety - workspace */

    /* Host */
    char     hostname[64];
    uint64_t ram_total;
    uint64_t ram_used_other;     /* MemTotal - MemAvailable */
    uint64_t ram_usable;         /* ram_total - ram_used_other - safety */
    /* Measured ceiling on pinned (cudaHostAlloc) host memory; 0 = unknown.
     *
     * HYBRID spills every layer that will not fit in VRAM into pinned memory,
     * so THIS, not ram_usable, is what bounds a node's host-side share. It is
     * far below physical RAM and not derivable from it: measured 47616 MiB on a
     * 65190 MiB box (73.0%) and 23552 MiB on a 48897 MiB box (48.2%), same GPU
     * model, each exactly reproducible across runs. Neither physical memory
     * (the smaller box still had 22 GiB free when it failed), nor the commit
     * limit, nor pagefile size predicts it.
     *
     * Measuring means allocating until failure, so it is done once and cached
     * (see worker_main.c). 0 means "not measured" and callers must treat the
     * host side as unconstrained — the behaviour before this field existed. */
    uint64_t ram_pinnable;
    uint32_t cpu_count;

    /* Storage at the configured GGUF dir */
    uint64_t disk_avail;
} idletoken_resource_report;

/* Run the probe. `gguf_dir` may be NULL to skip disk probing. Returns 0 on
 * success, -1 on error (errno or fprintf'd diagnostic). Partial fields may
 * be left at zero if a sub-probe fails. */
int idletoken_resource_probe(idletoken_resource_report *out, const char *gguf_dir);

/* Why a machine cannot take part. Kept as a code (not just a string) so the
 * client can localize; `reason` carries the human sentence for CLI/log use. */
typedef enum {
    IDLETOKEN_HW_OK = 0,
    IDLETOKEN_HW_NO_GPU,          /* NVML absent or no device: no driver at all */
    IDLETOKEN_HW_CC_TOO_LOW,      /* pre-Turing card */
    IDLETOKEN_HW_DRIVER_TOO_OLD,  /* driver older than the shipped CUDA needs */
    IDLETOKEN_HW_VRAM_TOO_SMALL,  /* < IDLETOKEN_MIN_VRAM_BYTES */
    IDLETOKEN_HW_GPU_UNSUPPORTED, /* a GPU, but not one we have a backend for
                                   * (e.g. an Intel Mac's AMD card: Metal is
                                   * present, unified memory is not) */
    /* Append-only: the client mirrors these numbers (client/src/types.ts) and
     * old builds keep sending the old ones. Inserting in the middle would
     * relabel every refusal already in the field. */
    IDLETOKEN_HW_MACOS_SEALED,    /* Apple Silicon: capable, but the macOS
                                   * compute-node line is parked before v0.1.
                                   * See idletoken_macos_node_sealed() in
                                   * idletoken_proto.h for why and how to lift. */
} idletoken_hw_status;

/* --- "this machine will not join, and retrying will not help" contract -----
 *
 * Two refusals are deterministic: the hardware floor (idletoken_hw_check) and the
 * coordinator turning this node away at HELLO (wrong OS family, say). Both are
 * decisions, not crashes — restarting the worker just repeats them.
 *
 * The worker therefore exits with IDLETOKEN_EXIT_JOIN_REFUSED and prints ONE
 * line carrying the reason:
 *
 *     idletoken-worker: JOIN_REFUSED: <one-line reason>
 *
 * The client's sidecar supervisor (client/src-tauri/src/engine.rs) scans stderr
 * for that marker, shows the reason in the UI, and does NOT restart. Without it
 * a refused machine burned five backoff restarts and ended as "crashed" with the
 * reason buried in a log ring buffer nobody opens.
 *
 * Change the marker here and in engine.rs together; the gate that covers this
 * is G_HOMO (see scripts/acceptance.sh). */
#define IDLETOKEN_JOIN_REFUSED_MARK "JOIN_REFUSED: "
#define IDLETOKEN_EXIT_JOIN_REFUSED 2

/* Verdict on whether this machine can serve layers at all. Writes a one-line
 * explanation into `reason` (what is wrong AND what is required) — the product
 * promise is "say what you need", not "fail mysteriously later". Callers must
 * refuse to join the cluster on anything but IDLETOKEN_HW_OK: a node that limps
 * along on an unsupported GPU produces garbage tokens, not an error.
 *
 * Test hooks (probe-time, see resource.c): IDLETOKEN_FORCE_NO_NVML=1,
 * IDLETOKEN_FAKE_CC=6.1, IDLETOKEN_FAKE_DRIVER=470.00. */
idletoken_hw_status idletoken_hw_check(const idletoken_resource_report *r,
                                 char *reason, size_t cap);

/* Apply user-configured usage limits (from the client's settings panel):
 * clamp `vram_usable`/`ram_usable` down so we never plan to use more than the
 * machine's owner allows. A limit of 0 means "no limit" and is ignored. This
 * is what makes the "max VRAM / RAM" setting really take effect (it flows into
 * RESOURCE_REPORT and therefore into the coordinator's layer split). */
void idletoken_resource_apply_caps(idletoken_resource_report *r,
                                uint64_t max_vram_bytes,
                                uint64_t max_ram_bytes);

/* Pretty-print for diagnostics. */
void idletoken_resource_print(const idletoken_resource_report *r);

/* Machine-readable single-line JSON of the same fields, for the desktop
 * client's Rust sidecar (and any other tooling) to consume without scraping
 * the human-formatted output. All byte counts are raw integers. */
void idletoken_resource_print_json(const idletoken_resource_report *r);

#endif /* IDLETOKEN_RESOURCE_H */
