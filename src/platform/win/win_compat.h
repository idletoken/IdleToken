/* IdleToken Windows platform shim: misc POSIX bits missing from MinGW-w64.
 * Force-included on Windows builds (gcc -include win_compat.h) so ds4's direct
 * sysconf() calls resolve. Implementation in win_compat.c. */
#ifndef IDLETOKEN_WIN_COMPAT_H
#define IDLETOKEN_WIN_COMPAT_H
#ifdef _WIN32

#include <stddef.h>
#include <stdio.h>
#include <sys/types.h>   /* ssize_t */

/* 64-bit file sizes. MinGW's default `struct stat` carries a 32-bit st_size,
 * which silently wraps the 81 GB GGUF: 86,720,111,488 mod 2^32 = 820,765,568
 * — exactly the bogus size the worker preflight rejected on win-b. ds4.c
 * also sizes its model mmap with fstat(), so everything compiled behind this
 * shim must see 64-bit sizes. Include the real <sys/stat.h> FIRST so the
 * remap only rewrites consumer code, not the header's own declarations. */
#include <sys/stat.h>
#define stat  _stati64
#define fstat _fstati64

/* sysconf names ds4 uses. */
#define _SC_PAGESIZE         1
#define _SC_PAGE_SIZE        1
#define _SC_NPROCESSORS_ONLN 2
#define _SC_NPROCESSORS_CONF 2

/* fcntl: only the close-on-exec dance is used, which is a no-op on Windows. */
#define F_GETFD    1
#define F_SETFD    2
#define FD_CLOEXEC 1

#ifdef __cplusplus
extern "C" {
#endif

/* Opt-in parent-death (Windows PR_SET_PDEATHSIG analogue): exit when the
 * launching client dies. Reads IDLETOKEN_DIE_WITH_PARENT + IDLETOKEN_PARENT_PID;
 * no-op when either is unset. */
void    idletoken_die_with_parent(void);

/* Best-effort idempotent inbound firewall allow (architecture §9, productization).
 * Checks `netsh advfirewall firewall show rule` first; adds only when missing.
 * Adding needs elevation — otherwise prints the exact netsh command once so
 * the user/installer can run it. IDLETOKEN_NO_FIREWALL_RULE=1 skips everything.
 * Include the port in rule_name so a port change provisions a fresh rule. */
void    idletoken_win_ensure_firewall_rule(const char *rule_name,
                                        const char *protocol, int port);

long    sysconf(int name);
int     fcntl(int fd, int cmd, ...);
/* POSIX env setters. MSVCRT only has _putenv_s / the (deprecated) putenv, and
 * MinGW does not declare setenv at all — the coord uses it to hand ds4 its
 * lock-file path before loading the tokenizer. */
int     setenv(const char *name, const char *value, int overwrite);
int     unsetenv(const char *name);
/* POSIX sleep(3). Windows has Sleep(milliseconds) with different case and
 * units — a bare `sleep(5)` compiles to an implicit-declaration warning and
 * then a link error. */
unsigned sleep(unsigned seconds);
int     dprintf(int fd, const char *fmt, ...);
ssize_t pread(int fd, void *buf, size_t count, long long offset);
FILE   *fmemopen(void *buf, size_t size, const char *mode);

#ifdef __cplusplus
}
#endif

#endif /* _WIN32 */
#endif /* IDLETOKEN_WIN_COMPAT_H */
