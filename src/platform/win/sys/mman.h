/* IdleToken Windows platform shim: <sys/mman.h>
 *
 * Minimal mmap/munmap/posix_madvise backed by Win32 file mapping, covering the
 * subset ds4 uses (read-only MAP_PRIVATE mapping of the GGUF model). Only
 * compiled into the include path on Windows builds; Linux/macOS use the real
 * <sys/mman.h>. Implementation in src/platform/win/win_compat.c. */
#ifndef IDLETOKEN_WIN_SYS_MMAN_H
#define IDLETOKEN_WIN_SYS_MMAN_H
#ifdef _WIN32

#include <stddef.h>

#define PROT_NONE   0x0
#define PROT_READ   0x1
#define PROT_WRITE  0x2
#define PROT_EXEC   0x4

#define MAP_FILE      0x00
#define MAP_SHARED    0x01
#define MAP_PRIVATE   0x02
#define MAP_ANONYMOUS 0x20
#define MAP_ANON      MAP_ANONYMOUS

#define MAP_FAILED ((void *)-1)

/* posix_madvise / madvise advice — accepted but treated as advisory no-ops. */
#define POSIX_MADV_NORMAL     0
#define POSIX_MADV_RANDOM     1
#define POSIX_MADV_SEQUENTIAL 2
#define POSIX_MADV_WILLNEED   3
#define POSIX_MADV_DONTNEED   4
#define MADV_NORMAL     POSIX_MADV_NORMAL
#define MADV_RANDOM     POSIX_MADV_RANDOM
#define MADV_SEQUENTIAL POSIX_MADV_SEQUENTIAL
#define MADV_WILLNEED   POSIX_MADV_WILLNEED
#define MADV_DONTNEED   POSIX_MADV_DONTNEED

#ifdef __cplusplus
extern "C" {
#endif

/* 64-bit offset regardless of Windows' 32-bit off_t (LLP64). */
void *mmap(void *addr, size_t length, int prot, int flags, int fd, long long offset);
int   munmap(void *addr, size_t length);
int   posix_madvise(void *addr, size_t length, int advice);
int   madvise(void *addr, size_t length, int advice);

#ifdef __cplusplus
}
#endif

#endif /* _WIN32 */
#endif /* IDLETOKEN_WIN_SYS_MMAN_H */
