/* IdleToken Cluster — macOS GPU facts, the few that need Objective-C.
 *
 * resource.c is plain C and stays that way; this is the one wall it cannot see
 * through. Everything else about a Mac (RAM, CPU count, free memory) comes from
 * sysctl and mach, which C reaches directly.
 *
 * Linked into the worker only — the worker already links Metal for the ds4
 * engine, so this costs nothing extra. The coordinator never probes hardware. */

#ifndef IDLETOKEN_MAC_GPU_H
#define IDLETOKEN_MAC_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char     name[64];          /* MTLDevice.name, e.g. "Apple M4" */
    int      unified_memory;    /* MTLDevice.hasUnifiedMemory */
    int      apple_silicon;     /* supportsFamily(MTLGPUFamilyApple7+) */
    uint64_t recommended_working_set;  /* recommendedMaxWorkingSetSize */
    uint64_t max_buffer_length;        /* maxBufferLength — caps ONE allocation */
    uint64_t current_allocated;        /* currentAllocatedSize, this process */
} idletoken_mac_gpu_info;

/* Fill `out` from the system default Metal device.
 * Returns 0 on success, -1 when there is no Metal device at all (headless VM,
 * or a Mac too old for the frameworks). Never partially fills: on -1, `out` is
 * zeroed. */
int idletoken_mac_gpu_probe(idletoken_mac_gpu_info *out);

#ifdef __cplusplus
}
#endif

#endif /* IDLETOKEN_MAC_GPU_H */
