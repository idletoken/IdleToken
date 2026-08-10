/* IdleToken Cluster — macOS GPU probe (Objective-C, Metal).
 *
 * The counterpart of the NVML probes in resource.c. Metal has no NVML: there
 * is no "how much of this GPU is another process using" query, because on
 * Apple Silicon there is no separate GPU pool to be using. What Metal does
 * expose, and what actually bounds us:
 *
 *   recommendedMaxWorkingSetSize — how many bytes of resources Apple says we
 *     may keep resident before the system starts evicting ours. This, not
 *     physical RAM, is the ceiling on a Mac node's share.
 *   maxBufferLength — the largest SINGLE MTLBuffer. Separate limit, and the
 *     one that bites first: ds4_metal.m already splits the mapped model into
 *     views because of it (DS4_METAL_MAX_MODEL_VIEWS).
 *
 * Measured on the M4 / 16 GiB test machine, macOS 26.5: see resource.c's
 * Darwin branch for the numbers this returns there. */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "idletoken_mac_gpu.h"

#include <string.h>

int idletoken_mac_gpu_probe(idletoken_mac_gpu_info *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return -1;

        const char *name = dev.name ? [dev.name UTF8String] : NULL;
        if (name) snprintf(out->name, sizeof(out->name), "%s", name);

        out->unified_memory = dev.hasUnifiedMemory ? 1 : 0;

        /* Apple7 is the first family that covers Apple Silicon Macs (M1). An
         * Intel Mac's AMD/Intel GPU answers NO here, which is the distinction
         * that matters: the ds4 Metal kernels assume unified memory, and an
         * Intel Mac would have to copy 80 GiB across PCIe to run them. */
        out->apple_silicon = [dev supportsFamily:MTLGPUFamilyApple7] ? 1 : 0;

        out->recommended_working_set = (uint64_t)dev.recommendedMaxWorkingSetSize;
        out->max_buffer_length       = (uint64_t)dev.maxBufferLength;
        out->current_allocated       = (uint64_t)dev.currentAllocatedSize;
    }
    return 0;
}
