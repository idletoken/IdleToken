/* IdleToken Cluster — resource probe (Linux NVML + Windows runtime nvml.dll
 * + macOS Metal/mach).
 *
 * Numbers report what's *actually free for our worker*, after deducting
 * what the system and other processes already use, plus safety margins. */

/* _GNU_SOURCE comes via Makefile. */

#include "idletoken_resource.h"
#include "idletoken_proto.h"   /* idletoken_macos_node_sealed() — the seal is
                                * shared with the coordinator's handshake */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <sys/statvfs.h>
  #include <sys/utsname.h>
  #include <unistd.h>
#endif

#ifdef __APPLE__
  #include <mach/mach.h>
  #include <mach/mach_host.h>
  #include <sys/sysctl.h>
  #include "idletoken_mac_gpu.h"
#endif

/* GPU probe backends:
 *   - Linux: link NVML from the CUDA install (nvml.h + -lnvidia-ml, always
 *     present at /usr/local/cuda/include/nvml.h on the DGX Spark).
 *   - Windows: load nvml.dll from the NVIDIA driver at *runtime* (see the
 *     _WIN32 probe_gpu below), so a driver-only machine with no CUDA toolkit
 *     can still report its GPU (driver-only one-click start, architecture.md §2). No nvml.h needed.
 *   - macOS: Metal via src/platform/mac/mac_gpu.m. No NVML equivalent exists
 *     and none is needed — see the Darwin probe_gpu below.
 * NVML ships with every NVIDIA driver, so neither path adds a toolkit dep. */
#if !defined(_WIN32) && !defined(__APPLE__)
  #define IDLETOKEN_HAVE_NVML 1
  #include <nvml.h>
#endif

/* OS headroom backstop: see IDLETOKEN_OS_HEADROOM_BYTES in
 * idletoken_resource.h for why this is an absolute amount and no longer a
 * percentage (2026-08-16, measured).
 * WARNING: this must sit **before** `#if defined(_WIN32)` -- both the Windows
 * and the Linux probe call it. The first version put it inside the Windows
 * branch, which compiled fine on Windows and only blew up at link time on
 * Linux (undefined reference).
 * Combined with the subtraction by taking the smaller of the two, so the
 * subtraction wins whenever it is the more conservative bound — which, unlike
 * under the old proportional rule, is now the normal case. */
static void ram_apply_os_headroom(idletoken_resource_report *r) {
    uint64_t headroom = r->ram_total / 8;
    if (headroom < IDLETOKEN_OS_HEADROOM_MIN_BYTES) headroom = IDLETOKEN_OS_HEADROOM_MIN_BYTES;
    if (headroom > IDLETOKEN_OS_HEADROOM_MAX_BYTES) headroom = IDLETOKEN_OS_HEADROOM_MAX_BYTES;
    const char *e = getenv("IDLETOKEN_OS_HEADROOM_GIB");
    if (e && e[0]) {
        unsigned long v = strtoul(e, NULL, 10);
        /* out-of-range values are ignored: a typo must not disable the backstop */
        if (v >= 1 && v <= 64) headroom = (uint64_t)v * 1024ull * 1024ull * 1024ull;
    }
    uint64_t ceiling = r->ram_total > headroom ? r->ram_total - headroom : 0;
    if (r->ram_usable > ceiling) r->ram_usable = ceiling;
}

#if defined(_WIN32)
/* ---- Windows GPU probe via runtime-loaded nvml.dll -----------------------
 * nvml.dll installs into System32 with the NVIDIA display driver, so a box
 * that has *only* the driver — no CUDA toolkit — can still probe its GPU.
 * We declare the tiny slice of the NVML ABI we call here instead of pulling
 * in nvml.h (which only exists with the toolkit). x86_64 has a single calling
 * convention, so these plain typedefs are ABI-correct against the shipped DLL. */

typedef int   nvmlReturn_t;
#define HAI_NVML_SUCCESS 0
typedef void *nvmlDevice_t;
typedef struct { unsigned long long total, free, used; } hai_nvmlMemory_t;

typedef nvmlReturn_t (*fn_nvmlInit)(void);
typedef nvmlReturn_t (*fn_nvmlShutdown)(void);
typedef const char  *(*fn_nvmlErrorString)(nvmlReturn_t);
typedef nvmlReturn_t (*fn_nvmlGetCount)(unsigned int *);
typedef nvmlReturn_t (*fn_nvmlGetHandle)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*fn_nvmlGetName)(nvmlDevice_t, char *, unsigned int);
typedef nvmlReturn_t (*fn_nvmlGetCC)(nvmlDevice_t, int *, int *);
typedef nvmlReturn_t (*fn_nvmlGetMem)(nvmlDevice_t, hai_nvmlMemory_t *);
typedef nvmlReturn_t (*fn_nvmlGetDriver)(char *, unsigned int);

static HMODULE nvml_load(void) {
    /* Test hook: pretend the driver is absent, so the "no driver" path can be
     * exercised on a machine that does have one (G-HW gate). */
    if (getenv("IDLETOKEN_FORCE_NO_NVML")) return NULL;
    HMODULE h = LoadLibraryA("nvml.dll");   /* default search hits System32 */
    if (!h) h = LoadLibraryA(
        "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll"); /* legacy */
    return h;
}

static int probe_gpu(idletoken_resource_report *r) {
    HMODULE nv = nvml_load();
    if (!nv) {
        fprintf(stderr, "idletoken-probe: nvml.dll not found — is the NVIDIA "
                        "driver installed?\n");
        return -1;
    }

    fn_nvmlInit        init  = (fn_nvmlInit)(void *)GetProcAddress(nv, "nvmlInit_v2");
    fn_nvmlShutdown    down  = (fn_nvmlShutdown)(void *)GetProcAddress(nv, "nvmlShutdown");
    fn_nvmlErrorString estr  = (fn_nvmlErrorString)(void *)GetProcAddress(nv, "nvmlErrorString");
    fn_nvmlGetCount    cnt   = (fn_nvmlGetCount)(void *)GetProcAddress(nv, "nvmlDeviceGetCount_v2");
    fn_nvmlGetHandle   hnd   = (fn_nvmlGetHandle)(void *)GetProcAddress(nv, "nvmlDeviceGetHandleByIndex_v2");
    fn_nvmlGetName     gname = (fn_nvmlGetName)(void *)GetProcAddress(nv, "nvmlDeviceGetName");
    fn_nvmlGetCC       gcc   = (fn_nvmlGetCC)(void *)GetProcAddress(nv, "nvmlDeviceGetCudaComputeCapability");
    fn_nvmlGetMem      gmem  = (fn_nvmlGetMem)(void *)GetProcAddress(nv, "nvmlDeviceGetMemoryInfo");
    fn_nvmlGetDriver   gdrv  = (fn_nvmlGetDriver)(void *)GetProcAddress(nv, "nvmlSystemGetDriverVersion");

    if (!init || !down || !cnt || !hnd || !gname || !gcc || !gmem) {
        fprintf(stderr, "idletoken-probe: nvml.dll is missing expected symbols "
                        "(driver too old?)\n");
        FreeLibrary(nv);
        return -1;
    }
    #define ESTR(s) (estr ? estr(s) : "nvml error")

    nvmlReturn_t st = init();
    if (st != HAI_NVML_SUCCESS) {
        fprintf(stderr, "idletoken-probe: nvmlInit failed: %s\n", ESTR(st));
        FreeLibrary(nv);
        return -1;
    }

    unsigned int count = 0;
    st = cnt(&count);
    if (st != HAI_NVML_SUCCESS || count == 0) {
        fprintf(stderr, "idletoken-probe: no NVIDIA GPU found (%s)\n",
                st == HAI_NVML_SUCCESS ? "count=0" : ESTR(st));
        down(); FreeLibrary(nv);
        return -1;
    }
    if (count > 1) {
        fprintf(stderr, "idletoken-probe: %u GPUs visible; using device 0. "
                        "Multi-GPU per node is out of scope for v0.1.\n", count);
    }

    nvmlDevice_t dev = NULL;
    st = hnd(0, &dev);
    if (st != HAI_NVML_SUCCESS) {
        fprintf(stderr, "idletoken-probe: nvmlDeviceGetHandleByIndex: %s\n", ESTR(st));
        down(); FreeLibrary(nv);
        return -1;
    }

    char name[96] = {0};
    if (gname(dev, name, sizeof(name)) == HAI_NVML_SUCCESS)
        snprintf(r->gpu_name, sizeof(r->gpu_name), "%s", name);

    /* Driver version decides whether the shipped CUDA runtime can load at all
     * (see IDLETOKEN_MIN_DRIVER_* ) — probe it here rather than discovering the
     * mismatch as a cudaErrorInsufficientDriver deep inside model load. */
    if (gdrv) {
        char drv[80] = {0};
        if (gdrv(drv, (unsigned)sizeof(drv)) == HAI_NVML_SUCCESS)
            snprintf(r->driver_version, sizeof(r->driver_version), "%s", drv);
    }

    int major = 0, minor = 0;
    if (gcc(dev, &major, &minor) == HAI_NVML_SUCCESS) {
        r->cc_major = (uint8_t)major;
        r->cc_minor = (uint8_t)minor;
    }

    /* NVML answered, so this is an NVIDIA card whatever else fails below. */
    r->gpu_vendor = IDLETOKEN_GPU_VENDOR_NVIDIA;

    /* Discrete Windows GPUs (RTX 5060 Ti / 2070) are never unified memory. */
    r->unified_memory = false;

    hai_nvmlMemory_t mem = {0};
    st = gmem(dev, &mem);
    if (st == HAI_NVML_SUCCESS) {
        r->vram_total = mem.total;
        /* At probe time our worker has not allocated yet, so NVML's "used" is
         * exactly what *other* processes (compositor, browser, ...) hold. */
        r->vram_used_other = mem.used;
        uint64_t reserved = IDLETOKEN_VRAM_SAFETY_BYTES + IDLETOKEN_VRAM_WORKSPACE_BYTES;
        if (r->vram_total > r->vram_used_other + reserved)
            r->vram_usable = r->vram_total - r->vram_used_other - reserved;
        else
            r->vram_usable = 0;
    } else {
        fprintf(stderr, "idletoken-probe: nvmlDeviceGetMemoryInfo: %s\n", ESTR(st));
    }

    #undef ESTR
    down();
    FreeLibrary(nv);
    return 0;
}
#elif defined(__APPLE__)
/* ---- macOS GPU probe via Metal -------------------------------------------
 * Everything NVML answers on the other two platforms is either unavailable or
 * meaningless here, and the reasons differ per field:
 *
 *   "how much VRAM"        — there is none. The GPU reads host RAM. What binds
 *                            us instead is recommendedMaxWorkingSetSize: the
 *                            bytes Apple says we may keep resident before the
 *                            system starts evicting our resources.
 *   "how much do OTHER
 *    processes hold"       — no such query, and charging it here would
 *                            DOUBLE-COUNT: another app's Metal buffers are host
 *                            RAM, so probe_host has already subtracted them in
 *                            ram_used_other. Left at 0 deliberately.
 *   compute capability /
 *   driver version         — CUDA concepts with no Metal analogue. Left 0/"",
 *                            which is why idletoken_hw_check dispatches on
 *                            gpu_vendor before it looks at either.
 *
 * Measured on the M4 / 16 GiB test machine (macOS 26.5):
 *   recommendedMaxWorkingSetSize = 11.84 GiB (74% of physical)
 *   maxBufferLength              =  8.88 GiB (55% of physical — a per-buffer
 *                                   cap, which is why ds4_metal.m splits the
 *                                   mapped model into views) */
static int probe_gpu(idletoken_resource_report *r) {
    /* Same test hook as the NVML paths: pretend the GPU is absent. */
    if (getenv("IDLETOKEN_FORCE_NO_NVML")) {
        fprintf(stderr, "idletoken-probe: no Metal device (forced)\n");
        return -1;
    }

    idletoken_mac_gpu_info g;
    if (idletoken_mac_gpu_probe(&g) != 0) {
        fprintf(stderr, "idletoken-probe: no Metal device found. IdleToken "
                        "needs a GPU to serve layers.\n");
        return -1;
    }

    snprintf(r->gpu_name, sizeof(r->gpu_name), "%s", g.name);
    r->unified_memory = g.unified_memory ? true : false;

    /* An Intel Mac's AMD/Intel GPU reports a Metal device but not Apple7 and
     * not unified memory. Record it as UNKNOWN so hw_check refuses with a
     * specific sentence instead of letting it join and produce garbage: the
     * ds4 Metal kernels are written against unified memory. */
    r->gpu_vendor = (g.apple_silicon && g.unified_memory)
                        ? IDLETOKEN_GPU_VENDOR_APPLE
                        : IDLETOKEN_GPU_VENDOR_UNKNOWN;

    r->vram_total      = g.recommended_working_set;
    r->vram_used_other = 0;   /* see the double-counting note above */

    /* The GPU may not keep more resident than Apple recommends, and it may not
     * use memory the host probe has already ruled out (other processes, the
     * 4 GiB safety floor, the 70% proportional ceiling). Take the smaller.
     *
     * Only the Metal workspace reserve is subtracted, not
     * IDLETOKEN_VRAM_SAFETY_BYTES: that 1 GiB stands for the CUDA context,
     * which has no Metal counterpart. The reserve itself is CALIBRATED
     * (2026-08-15, this machine): the llamacpp engine's whole non-weight
     * footprint is ~133 MiB and the width-scaled part is charged by the
     * scheduler — the earlier 1.5 GiB CUDA-derived guess stacked with that
     * into a ~2.3 GiB double reservation on a 16 GiB Mac. */
    uint64_t budget = g.recommended_working_set;
    if (r->ram_usable < budget) budget = r->ram_usable;
    r->vram_usable = budget > IDLETOKEN_METAL_WORKSPACE_BYTES
                         ? budget - IDLETOKEN_METAL_WORKSPACE_BYTES
                         : 0;
    /* No `ram_usable > 0` guard on that clamp, deliberately. The first version
     * had one — meaning "only clamp if the host probe produced something" — and
     * it inverted the worst case into the best one: on a Mac with no free
     * memory (ram_usable == 0) the clamp was skipped and the node advertised
     * the full 10.3 GiB it definitely did not have. A failed host probe must
     * report zero usable, not unlimited. */
    return 0;
}
#elif defined(IDLETOKEN_HAVE_NVML)
static int probe_gpu(idletoken_resource_report *r) {
    /* Test hook: pretend the driver is absent (G-HW gate) — same contract as
     * the Windows path, where it short-circuits the nvml.dll load. */
    if (getenv("IDLETOKEN_FORCE_NO_NVML")) {
        fprintf(stderr, "idletoken-probe: NVML unavailable — is the NVIDIA "
                        "driver installed?\n");
        return -1;
    }
    nvmlReturn_t st = nvmlInit_v2();
    if (st != NVML_SUCCESS) {
        fprintf(stderr, "idletoken-probe: nvmlInit_v2 failed: %s\n", nvmlErrorString(st));
        return -1;
    }

    unsigned int count = 0;
    st = nvmlDeviceGetCount_v2(&count);
    if (st != NVML_SUCCESS || count == 0) {
        fprintf(stderr, "idletoken-probe: no NVIDIA GPU found (%s)\n",
                st == NVML_SUCCESS ? "count=0" : nvmlErrorString(st));
        nvmlShutdown();
        return -1;
    }
    if (count > 1) {
        fprintf(stderr, "idletoken-probe: %u GPUs visible; using device 0. "
                        "Multi-GPU per node is out of scope for v0.1.\n", count);
    }

    nvmlDevice_t dev;
    st = nvmlDeviceGetHandleByIndex_v2(0, &dev);
    if (st != NVML_SUCCESS) {
        fprintf(stderr, "idletoken-probe: nvmlDeviceGetHandleByIndex_v2: %s\n",
                nvmlErrorString(st));
        nvmlShutdown();
        return -1;
    }

    /* NVML answered, so this is an NVIDIA card whatever else fails below. */
    r->gpu_vendor = IDLETOKEN_GPU_VENDOR_NVIDIA;

    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {0};
    if (nvmlDeviceGetName(dev, name, sizeof(name)) == NVML_SUCCESS) {
        snprintf(r->gpu_name, sizeof(r->gpu_name), "%s", name);
    }

    int major = 0, minor = 0;
    if (nvmlDeviceGetCudaComputeCapability(dev, &major, &minor) == NVML_SUCCESS) {
        r->cc_major = (uint8_t)major;
        r->cc_minor = (uint8_t)minor;
    }

    /* Driver version decides whether the shipped CUDA runtime can load at all
     * (see IDLETOKEN_MIN_DRIVER_*). */
    {
        char drv[80] = {0};
        if (nvmlSystemGetDriverVersion(drv, (unsigned)sizeof(drv)) == NVML_SUCCESS)
            snprintf(r->driver_version, sizeof(r->driver_version), "%s", drv);
    }

    /* Unified-memory detection: NVML doesn't expose this directly. Heuristic:
     * if the device name contains "GB10" (Grace-Blackwell) or if memory_total
     * matches host RAM order-of-magnitude AND nvmlDeviceGetMemoryInfo returns
     * 0 used despite there clearly being device-tied processes, treat as unified.
     * For v0.1 we just look at the device name. */
    r->unified_memory = (strstr(r->gpu_name, "GB10") != NULL ||
                         strstr(r->gpu_name, "GH200") != NULL ||
                         strstr(r->gpu_name, "Grace") != NULL);

    nvmlMemory_v2_t mem;
    mem.version = nvmlMemory_v2;
    st = nvmlDeviceGetMemoryInfo_v2(dev, &mem);
    if (st == NVML_SUCCESS) {
        r->vram_total = mem.total;
        /* "used" reported by NVML is total - free, which includes our own
         * future allocations. For "used by *other* processes", we sum the
         * per-process memory used reported by NVML and subtract anything
         * tagged with our PID. v0.1 approximation: use NVML's "used". */
        uint64_t used_others = mem.used;
        pid_t self = getpid();
        /* try to remove our own usage from "used_others" */
        unsigned int n_procs = 64;
        nvmlProcessInfo_t procs[64];
        if (nvmlDeviceGetComputeRunningProcesses_v3(dev, &n_procs, procs) == NVML_SUCCESS) {
            for (unsigned int i = 0; i < n_procs; i++) {
                if ((pid_t)procs[i].pid == self) {
                    if (used_others >= procs[i].usedGpuMemory) used_others -= procs[i].usedGpuMemory;
                }
            }
        }
        r->vram_used_other = used_others;

        uint64_t reserved = IDLETOKEN_VRAM_SAFETY_BYTES + IDLETOKEN_VRAM_WORKSPACE_BYTES;
        if (r->vram_total > r->vram_used_other + reserved) {
            r->vram_usable = r->vram_total - r->vram_used_other - reserved;
        } else {
            r->vram_usable = 0;
        }
    } else {
        fprintf(stderr, "idletoken-probe: nvmlDeviceGetMemoryInfo_v2: %s\n", nvmlErrorString(st));
    }

    nvmlShutdown();
    return 0;
}
#else  /* neither Windows nor Linux/NVML — no known GPU probe backend */
static int probe_gpu(idletoken_resource_report *r) {
    /* Every supported platform (Linux w/ NVML, Windows w/ nvml.dll) has a real
     * probe above; this branch is a safety net for an unexpected build config. */
    (void)r;
    fprintf(stderr, "idletoken-probe: GPU probe unavailable in this build "
                    "(no NVML backend); host + disk only.\n");
    return -1;
}
#endif /* GPU probe backend */

/* Host RAM + CPU + hostname. Returns 0 on success. */
static int probe_host(idletoken_resource_report *r) {
#ifdef _WIN32
    char cname[MAX_COMPUTERNAME_LENGTH + 1] = {0};
    DWORD clen = sizeof(cname);
    if (GetComputerNameA(cname, &clen)) {
        snprintf(r->hostname, sizeof(r->hostname), "%s", cname);
    }

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    r->cpu_count = (uint32_t)si.dwNumberOfProcessors;

    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) {
        fprintf(stderr, "idletoken-probe: GlobalMemoryStatusEx failed\n");
        return -1;
    }
    r->ram_total = (uint64_t)ms.ullTotalPhys;
    uint64_t ram_avail = (uint64_t)ms.ullAvailPhys;
    if (ram_avail > r->ram_total) ram_avail = r->ram_total;
    r->ram_used_other = r->ram_total - ram_avail;

    if (r->ram_total > r->ram_used_other + IDLETOKEN_RAM_SAFETY_BYTES) {
        r->ram_usable = r->ram_total - r->ram_used_other - IDLETOKEN_RAM_SAFETY_BYTES;
    } else {
        r->ram_usable = 0;
    }
    ram_apply_os_headroom(r);
    return 0;
#elif defined(__APPLE__)
    struct utsname un;
    if (uname(&un) == 0) {
        size_t n = strnlen(un.nodename, sizeof(un.nodename));
        if (n >= sizeof(r->hostname)) n = sizeof(r->hostname) - 1;
        memcpy(r->hostname, un.nodename, n);
        r->hostname[n] = '\0';
    }

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    r->cpu_count = ncpu > 0 ? (uint32_t)ncpu : 0;

    {
        uint64_t memsize = 0;
        size_t len = sizeof(memsize);
        if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) != 0 || memsize == 0) {
            fprintf(stderr, "idletoken-probe: sysctl hw.memsize: %s\n", strerror(errno));
            return -1;
        }
        r->ram_total = memsize;
    }

    /* macOS has no MemAvailable. Reconstruct what Activity Monitor calls
     * "Memory Used" from the mach VM counters and treat the rest as available:
     *
     *   used = internal (anonymous/app pages)
     *        - purgeable (app pages the kernel may drop on demand)
     *        + wired     (unpageable kernel/driver pages)
     *        + compressed
     *
     * File-backed pages (external_page_count) are deliberately NOT counted as
     * used — they are the page cache, and on this project that matters more
     * than usual: the GGUF is mmap'd, so at steady state a large share of
     * "used-looking" memory is our own model file and is reclaimable. Counting
     * it would make a warm machine look full and refuse work it can do. */
    {
        vm_size_t page_size = 0;
        if (host_page_size(mach_host_self(), &page_size) != KERN_SUCCESS || page_size == 0)
            page_size = 16384;   /* Apple Silicon default; sysconf agrees */

        vm_statistics64_data_t vm;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                              (host_info64_t)&vm, &count) != KERN_SUCCESS) {
            fprintf(stderr, "idletoken-probe: host_statistics64 failed\n");
            return -1;
        }

        uint64_t internal   = (uint64_t)vm.internal_page_count;
        uint64_t purgeable  = (uint64_t)vm.purgeable_count;
        uint64_t app_pages  = internal > purgeable ? internal - purgeable : 0;
        uint64_t used_pages = app_pages + (uint64_t)vm.wire_count +
                              (uint64_t)vm.compressor_page_count;

        uint64_t used = used_pages * (uint64_t)page_size;
        if (used > r->ram_total) used = r->ram_total;
        r->ram_used_other = used;
    }

    if (r->ram_total > r->ram_used_other + IDLETOKEN_RAM_SAFETY_BYTES) {
        r->ram_usable = r->ram_total - r->ram_used_other - IDLETOKEN_RAM_SAFETY_BYTES;
    } else {
        r->ram_usable = 0;
    }
    ram_apply_os_headroom(r);
    return 0;
#else
    struct utsname un;
    if (uname(&un) == 0) {
        size_t n = strnlen(un.nodename, sizeof(un.nodename));
        if (n >= sizeof(r->hostname)) n = sizeof(r->hostname) - 1;
        memcpy(r->hostname, un.nodename, n);
        r->hostname[n] = '\0';
    }

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    r->cpu_count = ncpu > 0 ? (uint32_t)ncpu : 0;

    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) {
        fprintf(stderr, "idletoken-probe: cannot read /proc/meminfo: %s\n", strerror(errno));
        return -1;
    }

    uint64_t mem_total_kb = 0, mem_available_kb = 0;
    char line[256];
    int found = 0;
    while (found < 2 && fgets(line, sizeof(line), fp)) {
        unsigned long long v = 0;
        if (sscanf(line, "MemTotal: %llu kB", &v) == 1) {
            mem_total_kb = (uint64_t)v;
            found++;
        } else if (sscanf(line, "MemAvailable: %llu kB", &v) == 1) {
            mem_available_kb = (uint64_t)v;
            found++;
        }
    }
    fclose(fp);

    if (mem_total_kb == 0) {
        fprintf(stderr, "idletoken-probe: failed to parse MemTotal\n");
        return -1;
    }

    r->ram_total = mem_total_kb * 1024ull;
    uint64_t ram_avail = mem_available_kb * 1024ull;
    if (ram_avail > r->ram_total) ram_avail = r->ram_total;
    r->ram_used_other = r->ram_total - ram_avail;

    if (r->ram_total > r->ram_used_other + IDLETOKEN_RAM_SAFETY_BYTES) {
        r->ram_usable = r->ram_total - r->ram_used_other - IDLETOKEN_RAM_SAFETY_BYTES;
    } else {
        r->ram_usable = 0;
    }
    ram_apply_os_headroom(r);
    return 0;
#endif /* _WIN32 */
}

static int probe_disk(idletoken_resource_report *r, const char *dir) {
    if (!dir) return 0;
#ifdef _WIN32
    ULARGE_INTEGER free_avail;
    if (!GetDiskFreeSpaceExA(dir, &free_avail, NULL, NULL)) {
        fprintf(stderr, "idletoken-probe: GetDiskFreeSpaceEx(%s) failed\n", dir);
        return -1;
    }
    r->disk_avail = (uint64_t)free_avail.QuadPart;
    return 0;
#else
    struct statvfs s;
    if (statvfs(dir, &s) != 0) {
        fprintf(stderr, "idletoken-probe: statvfs(%s): %s\n", dir, strerror(errno));
        return -1;
    }
    r->disk_avail = (uint64_t)s.f_bavail * (uint64_t)s.f_frsize;
    return 0;
#endif
}

int idletoken_resource_probe(idletoken_resource_report *out, const char *gguf_dir) {
    if (!out) { errno = EINVAL; return -1; }
    memset(out, 0, sizeof(*out));

    /* Order matters: host first so probe_gpu can fall back to RAM numbers
     * on unified-memory hosts where NVML doesn't report VRAM. */
    int r2 = probe_host(out);
    int r1 = probe_gpu(out);
    int r3 = probe_disk(out, gguf_dir);

    /* Unified-memory hosts (DGX Spark GB10, Grace Hopper, etc.) share one
     * physical pool between CPU and GPU. NVML reports "Not Supported" for
     * memory info there. Treat the GPU pool as aliased to host RAM so the
     * coordinator's `needed(tier) <= sum(vram_usable + ram_usable)` math
     * does not double-count or zero out. */
    if (out->unified_memory && out->vram_total == 0 && out->ram_total > 0) {
        out->vram_total      = out->ram_total;
        out->vram_used_other = out->ram_used_other;
        out->vram_usable     = out->ram_usable;
    }

    /* Test hooks (G-HW gate): pretend this machine has an older card / driver
     * so the refusal paths can be exercised on hardware that is actually fine.
     * Applied after the real probe so everything else stays truthful. */
    {
        const char *fake_cc = getenv("IDLETOKEN_FAKE_CC");
        if (fake_cc && *fake_cc) {
            unsigned mj = 0, mn = 0;
            if (sscanf(fake_cc, "%u.%u", &mj, &mn) >= 1) {
                fprintf(stderr,
                        "*** TEST OVERRIDE *** IDLETOKEN_FAKE_CC=%s replaces the "
                        "real compute capability of this GPU. Never set this "
                        "outside a test harness.\n", fake_cc);
                out->cc_major = (uint8_t)mj;
                out->cc_minor = (uint8_t)mn;
            }
        }
        const char *fake_drv = getenv("IDLETOKEN_FAKE_DRIVER");
        if (fake_drv && *fake_drv) {
            fprintf(stderr,
                    "*** TEST OVERRIDE *** IDLETOKEN_FAKE_DRIVER=%s replaces the "
                    "real driver version of this node. Never set this outside a "
                    "test harness.\n", fake_drv);
            snprintf(out->driver_version, sizeof(out->driver_version), "%s", fake_drv);
        }
        /* G_MACSEAL needs an Apple-vendor report, and the only machines that
         * produce one are the ones the seal keeps out of the ladder. Forcing
         * the vendor byte lets the refusal be exercised on the CUDA nodes that
         * actually run the gates. */
        const char *fake_vendor = getenv("IDLETOKEN_FAKE_VENDOR");
        if (fake_vendor && !strcmp(fake_vendor, "apple")) {
            fprintf(stderr,
                    "*** TEST OVERRIDE *** IDLETOKEN_FAKE_VENDOR=apple replaces "
                    "the real GPU vendor of this node. Never set this outside a "
                    "test harness.\n");
            out->gpu_vendor = IDLETOKEN_GPU_VENDOR_APPLE;
            if (out->gpu_name[0] == '\0')
                snprintf(out->gpu_name, sizeof(out->gpu_name), "Apple M-series (forced)");
        }
    }

    return (r1 == 0 && r2 == 0 && r3 == 0) ? 0 : -1;
}

/* Compare a NVML driver string ("580.126.09") against the floor. Unreadable
 * versions are NOT treated as too old: some virtualized/WSL stacks leave the
 * string empty while the driver works, and refusing a working machine over a
 * missing cosmetic field would be worse than the risk it guards. */
static int driver_below_floor(const char *ver) {
    if (!ver || !*ver) return 0;
    unsigned mj = 0, mn = 0;
    if (sscanf(ver, "%u.%u", &mj, &mn) < 1) return 0;
    if (mj != IDLETOKEN_MIN_DRIVER_MAJOR) return mj < IDLETOKEN_MIN_DRIVER_MAJOR;
    return mn < IDLETOKEN_MIN_DRIVER_MINOR;
}

idletoken_hw_status idletoken_hw_check(const idletoken_resource_report *r,
                                 char *reason, size_t cap) {
    if (!r) return IDLETOKEN_HW_NO_GPU;
    #define SAY(...) do { if (reason && cap) snprintf(reason, cap, __VA_ARGS__); } while (0)

    /* Apple Silicon: a different stack, so a different floor. Compute
     * capability and driver version below are CUDA concepts and are zero here;
     * running the NVIDIA checks on a Mac would refuse every Mac ever made. */
    if (r->gpu_vendor == IDLETOKEN_GPU_VENDOR_APPLE) {
        /* Sealed before the size floor on purpose: "this Mac is too small" is
         * the wrong sentence to tell someone whose Mac is plenty big. */
        if (idletoken_macos_node_sealed()) {
            /* Keep this under ~250 chars: every caller's `reason` buffer is
             * 256, and the half that gets truncated is the actionable half. */
            SAY("%s runs Metal, but macOS compute nodes are sealed in this build "
                "(no numerical baseline of its own, no Metal kernels for small "
                "models, unverified multi-Mac clusters). This Mac can still run "
                "the client and control a Windows or Linux cluster.", r->gpu_name);
            return IDLETOKEN_HW_MACOS_SEALED;
        }
        if (r->vram_total < IDLETOKEN_MIN_APPLE_WORKING_SET_BYTES) {
            SAY("%s can keep only %.1f GB resident for the GPU; IdleToken needs "
                "at least %.0f GB (roughly an 8 GB Mac). This machine can still "
                "run the client and control the cluster.",
                r->gpu_name, (double)r->vram_total / 1073741824.0,
                (double)IDLETOKEN_MIN_APPLE_WORKING_SET_BYTES / 1073741824.0);
            return IDLETOKEN_HW_VRAM_TOO_SMALL;
        }
        SAY("SEALED PATH (" IDLETOKEN_MACOS_UNSEAL_ENV " is set): %s (Metal, %.1f GB "
            "working set) meets the floor, but no gate covers anything it does.",
            r->gpu_name, (double)r->vram_total / 1073741824.0);
        return IDLETOKEN_HW_OK;   /* SAY is #undef'd once, at the end of the function */
    }

    /* A GPU we recognise as present but have no backend for. Today that is an
     * Intel Mac: Metal answers, unified memory does not, and the ds4 Metal
     * kernels assume unified memory. Say so instead of failing at load time. */
    if (r->gpu_vendor == IDLETOKEN_GPU_VENDOR_UNKNOWN && r->gpu_name[0] != '\0' &&
        r->cc_major == 0 && r->cc_minor == 0) {
        SAY("%s is not a GPU IdleToken can compute on — it needs either an "
            "NVIDIA card (RTX 20 series or newer) on Windows or Linux. "
            "This machine can still run the client and control the cluster.",
            r->gpu_name);
        return IDLETOKEN_HW_GPU_UNSUPPORTED;
    }

    if (r->gpu_name[0] == '\0' || (r->cc_major == 0 && r->cc_minor == 0)) {
        SAY("no usable NVIDIA GPU detected. IdleToken needs an NVIDIA card "
            "(RTX 20 series or newer) with the NVIDIA driver installed "
            "(driver %d.%d or newer); install the driver from NVIDIA and start "
            "IdleToken again.",
            IDLETOKEN_MIN_DRIVER_MAJOR, IDLETOKEN_MIN_DRIVER_MINOR);
        return IDLETOKEN_HW_NO_GPU;
    }
    if (r->cc_major < IDLETOKEN_MIN_CC_MAJOR ||
        (r->cc_major == IDLETOKEN_MIN_CC_MAJOR && r->cc_minor < IDLETOKEN_MIN_CC_MINOR)) {
        SAY("%s is compute capability %u.%u; IdleToken needs %d.%d or newer "
            "(RTX 20 series / Turing and later). This card cannot serve layers.",
            r->gpu_name, r->cc_major, r->cc_minor,
            IDLETOKEN_MIN_CC_MAJOR, IDLETOKEN_MIN_CC_MINOR);
        return IDLETOKEN_HW_CC_TOO_LOW;
    }
    if (driver_below_floor(r->driver_version)) {
        SAY("NVIDIA driver %s is too old for this build; IdleToken needs %d.%d or "
            "newer. Update the driver and start IdleToken again.",
            r->driver_version, IDLETOKEN_MIN_DRIVER_MAJOR, IDLETOKEN_MIN_DRIVER_MINOR);
        return IDLETOKEN_HW_DRIVER_TOO_OLD;
    }
    if (!r->unified_memory && r->vram_total > 0 &&
        r->vram_total < IDLETOKEN_MIN_VRAM_BYTES) {
        SAY("%s has %.1f GB of VRAM; IdleToken needs at least %.0f GB per machine "
            "(CUDA context + workspace + one model layer).",
            r->gpu_name, (double)r->vram_total / 1073741824.0,
            (double)IDLETOKEN_MIN_VRAM_BYTES / 1073741824.0);
        return IDLETOKEN_HW_VRAM_TOO_SMALL;
    }
    SAY("%s (cc %u.%u, driver %s) meets the hardware floor.",
        r->gpu_name, r->cc_major, r->cc_minor,
        r->driver_version[0] ? r->driver_version : "unknown");
    #undef SAY
    return IDLETOKEN_HW_OK;
}

void idletoken_resource_apply_caps(idletoken_resource_report *r,
                                uint64_t max_vram_bytes,
                                uint64_t max_ram_bytes) {
    if (!r) return;
    if (max_vram_bytes > 0 && r->vram_usable > max_vram_bytes)
        r->vram_usable = max_vram_bytes;
    if (max_ram_bytes > 0 && r->ram_usable > max_ram_bytes)
        r->ram_usable = max_ram_bytes;
}

static void fmt_bytes(uint64_t b, char out[32]) {
    const double GB = 1024.0 * 1024.0 * 1024.0;
    const double MB = 1024.0 * 1024.0;
    if (b >= (uint64_t)GB) snprintf(out, 32, "%.2f GiB", (double)b / GB);
    else if (b >= (uint64_t)MB) snprintf(out, 32, "%.2f MiB", (double)b / MB);
    else snprintf(out, 32, "%llu B", (unsigned long long)b);
}

void idletoken_resource_print(const idletoken_resource_report *r) {
    char vt[32], vu[32], vo[32], rt[32], ru[32], ro[32], da[32];
    fmt_bytes(r->vram_total,      vt);
    fmt_bytes(r->vram_usable,     vu);
    fmt_bytes(r->vram_used_other, vo);
    fmt_bytes(r->ram_total,       rt);
    fmt_bytes(r->ram_usable,      ru);
    fmt_bytes(r->ram_used_other,  ro);
    fmt_bytes(r->disk_avail,      da);

    printf("IdleToken resource probe\n");
    printf("  host:        %s  (%u CPUs)\n", r->hostname, r->cpu_count);
    printf("  gpu:         %s  (cc %u.%u, driver %s)%s\n",
           r->gpu_name, r->cc_major, r->cc_minor,
           r->driver_version[0] ? r->driver_version : "unknown",
           r->unified_memory ? "  [unified memory]" : "");
    if (r->gpu_vendor == IDLETOKEN_GPU_VENDOR_APPLE) {
        printf("  vram total : %s   (Metal recommended working set, unified pool)\n", vt);
        printf("  vram other : %s   (not queryable; already charged to ram other)\n", vo);
        printf("  vram usable: %s   (min(working set, ram usable) - 0.5G Metal workspace)\n", vu);
    } else if (r->unified_memory) {
        printf("  vram total : %s   (aliased to host RAM, unified pool)\n", vt);
        printf("  vram other : %s\n", vo);
        printf("  vram usable: %s   (= ram_usable on unified-memory hosts)\n", vu);
    } else {
        printf("  vram total : %s\n", vt);
        printf("  vram other : %s\n", vo);
        printf("  vram usable: %s   (after 1.0G safety + 1.5G workspace)\n", vu);
    }
    printf("  ram total  : %s\n", rt);
    printf("  ram other  : %s\n", ro);
    printf("  ram usable : %s   (after 4.0G safety)\n", ru);
    printf("  disk avail : %s\n", da);
}

/* Emit `s` as a JSON string literal (with surrounding quotes), escaping the
 * characters JSON requires. Keeps the client's parser happy on odd hostnames. */
static void json_str(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
            case '"':  fputs("\\\"", stdout); break;
            case '\\': fputs("\\\\", stdout); break;
            case '\n': fputs("\\n", stdout);  break;
            case '\r': fputs("\\r", stdout);  break;
            case '\t': fputs("\\t", stdout);  break;
            default:
                if (*p < 0x20) printf("\\u%04x", *p);
                else putchar(*p);
        }
    }
    putchar('"');
}

void idletoken_resource_print_json(const idletoken_resource_report *r) {
    fputs("{\"hostname\":", stdout);       json_str(r->hostname);
    fputs(",\"os\":", stdout);
#ifdef _WIN32
    json_str("windows");
#elif defined(__linux__)
    json_str("linux");
#elif defined(__APPLE__)
    json_str("macos");
#else
    json_str("unknown");
#endif
    fputs(",\"cpu_count\":", stdout);      printf("%u", r->cpu_count);
    fputs(",\"gpu_name\":", stdout);       json_str(r->gpu_name);
    /* Which GPU stack, so the client can label the node and pick the right
     * "what you need" sentence without parsing gpu_name. */
    fputs(",\"gpu_vendor\":", stdout);
    json_str(r->gpu_vendor == IDLETOKEN_GPU_VENDOR_NVIDIA ? "nvidia" :
             r->gpu_vendor == IDLETOKEN_GPU_VENDOR_APPLE  ? "apple"  : "unknown");
    printf(",\"cc_major\":%u,\"cc_minor\":%u", r->cc_major, r->cc_minor);
    fputs(",\"driver_version\":", stdout); json_str(r->driver_version);
    {   /* Verdict travels with the numbers so the client can gate the UI on a
         * code instead of re-deriving the rule in TypeScript. */
        char why[256] = "";
        idletoken_hw_status hw = idletoken_hw_check(r, why, sizeof(why));
        printf(",\"hw_status\":%d", (int)hw);
        fputs(",\"hw_reason\":", stdout); json_str(why);
    }
    printf(",\"unified_memory\":%s", r->unified_memory ? "true" : "false");
    printf(",\"vram_total\":%llu,\"vram_used_other\":%llu,\"vram_usable\":%llu",
           (unsigned long long)r->vram_total,
           (unsigned long long)r->vram_used_other,
           (unsigned long long)r->vram_usable);
    printf(",\"ram_total\":%llu,\"ram_used_other\":%llu,\"ram_usable\":%llu",
           (unsigned long long)r->ram_total,
           (unsigned long long)r->ram_used_other,
           (unsigned long long)r->ram_usable);
    printf(",\"disk_avail\":%llu", (unsigned long long)r->disk_avail);
    fputs("}\n", stdout);
}
