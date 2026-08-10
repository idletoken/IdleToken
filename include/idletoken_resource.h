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
#define IDLETOKEN_RAM_SAFETY_BYTES      (4ull * 1024ull * 1024ull * 1024ull)  /* 4.0 GB */

/* Blunt proportional ceiling on `ram_usable`, applied on top of the
 * "total − used_other − safety" subtraction and independent of it.
 *
 * Why a second, cruder rule: the subtraction *looks* precise and is not. It
 * charges only what other processes hold at probe time, so anything our own
 * load costs beyond the raw weight bytes is unaccounted — and on Windows that
 * turned out to be large. Measured on win-a (63.7 GiB, 28 layers): available
 * memory sat at 128–730 MiB for the whole run, not as a loading dip but as the
 * steady state, while the formula believed 53.7 GiB were usable and ~10 GiB
 * would be left over. A machine that plans to the last byte hangs the moment
 * anything else asks for memory; the user hit exactly that twice, screen on and
 * keyboard dead.
 *
 * Parallax (GradientHQ) caps parameter memory at a flat 50% of device memory
 * and never subtracts anything — cruder than us and it does not hang. The
 * lesson taken is not the number but the shape: keep a floor of headroom that
 * no amount of estimation error can eat into.
 *
 * 70% of a 64 GiB box leaves ~19 GiB, which still admits the 28-layer split
 * this cluster actually runs. It is a ceiling, not a target: the subtraction
 * still wins whenever it is lower. Override with IDLETOKEN_RAM_MAX_PCT for
 * measurement; values outside 10..95 are ignored. */
#define IDLETOKEN_RAM_MAX_PCT           70

/* Hardware floor. A node below any of these cannot serve layers, and finding
 * that out at model-load time (or worse: as garbage tokens) is exactly the
 * experience this floor exists to prevent.
 *
 * cc 7.5 = Turing = RTX 20 series. Everything older lacks the tensor-core /
 * fp16 paths the ds4 and ds4x kernels are written against.
 *
 * The driver floor tracks the CUDA toolkit each shipped package is BUILT with,
 * because that is what the runtime demands:
 *   - Windows package: ds4cuda.dll built with CUDA 12.8 -> 527.41 (CUDA 12.x
 *     minimum; minor-version compatibility covers 12.0..12.9, which is why a
 *     555.85 laptop runs a 12.8 build).
 *   - Linux package: built with CUDA 13.0 -> 580.65.
 * Rebuild against a different toolkit => update these. */
#define IDLETOKEN_MIN_CC_MAJOR   7
#define IDLETOKEN_MIN_CC_MINOR   5
#ifdef _WIN32
  #define IDLETOKEN_MIN_DRIVER_MAJOR 527
  #define IDLETOKEN_MIN_DRIVER_MINOR 41
#else
  #define IDLETOKEN_MIN_DRIVER_MAJOR 580
  #define IDLETOKEN_MIN_DRIVER_MINOR 65
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
} idletoken_hw_status;

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
