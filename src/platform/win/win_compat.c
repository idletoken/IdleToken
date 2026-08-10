/* IdleToken Windows platform shim implementations.
 * Backs the declarations in src/platform/win/{sys/mman.h,sys/file.h,win_compat.h}.
 * Windows-only translation unit (guarded by _WIN32). */
#ifdef _WIN32

#include <windows.h>
#include <io.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

#include "sys/mman.h"
#include "sys/file.h"
#include "win_compat.h"

/* --- parent-death: die with the launching client ------------------------- *
 * Windows analogue of Linux PR_SET_PDEATHSIG. The client supervisor spawns us
 * with IDLETOKEN_DIE_WITH_PARENT=1 and IDLETOKEN_PARENT_PID=<client pid>; we open a
 * SYNCHRONIZE handle to that process and a background thread waits on it, so a
 * crashed/SIGKILLed client never orphans the engine. Opt-in only: scripted
 * (nohup) deploys don't set the vars and must outlive their launching shell. */

static DWORD WINAPI idletoken_parent_watch(LPVOID arg) {
    HANDLE parent = (HANDLE)arg;
    WaitForSingleObject(parent, INFINITE); /* returns when the parent exits */
    ExitProcess(0);                         /* take the engine down hard */
    return 0;
}

void idletoken_die_with_parent(void) {
    if (!getenv("IDLETOKEN_DIE_WITH_PARENT")) return;
    const char *ppid_s = getenv("IDLETOKEN_PARENT_PID");
    if (!ppid_s || !*ppid_s) return;
    DWORD ppid = (DWORD)strtoul(ppid_s, NULL, 10);
    if (ppid == 0) return;

    HANDLE parent = OpenProcess(SYNCHRONIZE, FALSE, ppid);
    if (!parent) { ExitProcess(0); return; } /* parent already gone/inaccessible */
    if (WaitForSingleObject(parent, 0) == WAIT_OBJECT_0) { ExitProcess(0); }

    HANDLE t = CreateThread(NULL, 0, idletoken_parent_watch, parent, 0, NULL);
    if (t) CloseHandle(t); /* detached; `parent` stays open for the wait thread */
    /* If the thread couldn't start, we simply forgo parent-death — no orphan
     * risk beyond what the client's graceful-exit hook already covers. */
}

/* --- firewall: self-provision inbound allow rules -------------------------- *
 * Real-machine bring-up hit silently-filtered ports twice (UDP beacon 14097,
 * HC inter-stage 14322/14323) and needed manual netsh. Product behavior:
 * the engine provisions its own inbound rules on startup. Idempotent —
 * `show rule` exits 0 when the rule already exists. ADD requires elevation;
 * when we are not elevated the add fails and we print the exact command once
 * (the client installer, which runs elevated, can also pre-provision). */

void idletoken_win_ensure_firewall_rule(const char *rule_name,
                                     const char *protocol, int port) {
    if (getenv("IDLETOKEN_NO_FIREWALL_RULE")) return;
    char cmd[512];
    snprintf(cmd, sizeof cmd,
             "netsh advfirewall firewall show rule name=\"%s\" >NUL 2>&1",
             rule_name);
    if (system(cmd) == 0) return;   /* already provisioned */
    snprintf(cmd, sizeof cmd,
             "netsh advfirewall firewall add rule name=\"%s\" dir=in "
             "action=allow protocol=%s localport=%d profile=any >NUL 2>&1",
             rule_name, protocol, port);
    if (system(cmd) == 0) {
        fprintf(stderr, "idletoken/win: firewall rule \"%s\" added (%s %d inbound)\n",
                rule_name, protocol, port);
    } else {
        fprintf(stderr,
                "idletoken/win: could not add firewall rule \"%s\" (not elevated?). "
                "Run once as admin:\n"
                "  netsh advfirewall firewall add rule name=\"%s\" dir=in "
                "action=allow protocol=%s localport=%d profile=any\n",
                rule_name, rule_name, protocol, port);
    }
}

/* --- mmap family ---------------------------------------------------------- */

void *mmap(void *addr, size_t length, int prot, int flags, int fd, long long offset) {
    (void)addr;
    if (length == 0) return MAP_FAILED;

    DWORD flProtect;
    DWORD dwDesiredAccess;
    if (prot & PROT_WRITE) {
        if (flags & MAP_PRIVATE) {          /* copy-on-write */
            flProtect       = PAGE_WRITECOPY;
            dwDesiredAccess = FILE_MAP_COPY;
        } else {
            flProtect       = PAGE_READWRITE;
            dwDesiredAccess = FILE_MAP_WRITE;
        }
    } else {                                 /* read-only (ds4's model mmap) */
        flProtect       = PAGE_READONLY;
        dwDesiredAccess = FILE_MAP_READ;
    }

    HANDLE hfile = (HANDLE)_get_osfhandle(fd);
    if (hfile == INVALID_HANDLE_VALUE) return MAP_FAILED;

    /* max size 0/0 => whole file. */
    HANDLE hmap = CreateFileMappingA(hfile, NULL, flProtect, 0, 0, NULL);
    if (!hmap) return MAP_FAILED;

    ULARGE_INTEGER off;
    off.QuadPart = (ULONGLONG)offset;
    void *p = MapViewOfFile(hmap, dwDesiredAccess, off.HighPart, off.LowPart, length);
    /* The view keeps its own reference to the mapping; safe to close now. */
    CloseHandle(hmap);
    if (getenv("IDLETOKEN_MMAP_DEBUG")) {
        fprintf(stderr, "idletoken/win: mmap length=%llu offset=%llu -> %p (err=%lu)\n",
                (unsigned long long)length, (unsigned long long)offset, p,
                p ? 0UL : (unsigned long)GetLastError());
    }
    if (!p) return MAP_FAILED;
    return p;
}

int munmap(void *addr, size_t length) {
    (void)length;
    return UnmapViewOfFile(addr) ? 0 : -1;
}

/* Advisory only; Windows has PrefetchVirtualMemory but a no-op is correct. */
int posix_madvise(void *addr, size_t length, int advice) {
    (void)addr; (void)length; (void)advice;
    return 0;
}
int madvise(void *addr, size_t length, int advice) {
    (void)addr; (void)length; (void)advice;
    return 0;
}

/* --- flock ---------------------------------------------------------------- */

int flock(int fd, int operation) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;

    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));

    if (operation & LOCK_UN) {
        return UnlockFileEx(h, 0, 0xFFFFFFFFu, 0xFFFFFFFFu, &ov) ? 0 : -1;
    }
    DWORD flags = 0;
    if (operation & LOCK_EX) flags |= LOCKFILE_EXCLUSIVE_LOCK;
    if (operation & LOCK_NB) flags |= LOCKFILE_FAIL_IMMEDIATELY;
    return LockFileEx(h, flags, 0, 0xFFFFFFFFu, 0xFFFFFFFFu, &ov) ? 0 : -1;
}

/* --- sysconf -------------------------------------------------------------- */

long sysconf(int name) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    switch (name) {
        case _SC_PAGESIZE:         return (long)si.dwPageSize;
        case _SC_NPROCESSORS_ONLN: return (long)si.dwNumberOfProcessors;
        default:                   return -1;
    }
}

/* --- fcntl: close-on-exec is a no-op (we never fork/exec) ------------------ */

int fcntl(int fd, int cmd, ...) {
    (void)fd; (void)cmd;
    return 0;
}

/* --- dprintf: formatted write to a raw fd --------------------------------- */

int dprintf(int fd, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) return -1;

    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) return -1;
    va_start(ap, fmt);
    vsnprintf(buf, (size_t)n + 1, fmt, ap);
    va_end(ap);

    int w = _write(fd, buf, (unsigned)n);
    free(buf);
    return w;
}

/* --- pread: positioned read, does not move the fd's file pointer ----------- */

ssize_t pread(int fd, void *buf, size_t count, long long offset) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;

    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ULARGE_INTEGER o;
    o.QuadPart   = (ULONGLONG)offset;
    ov.Offset     = o.LowPart;
    ov.OffsetHigh = o.HighPart;

    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD)count, &got, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        return -1;
    }
    return (ssize_t)got;
}

/* --- fmemopen: MinGW has none; back it with a temp file -------------------
 * Correct for read modes (ds4 snapshot restore reads `buf` back). Write modes
 * land in the temp file and are NOT reflected into `buf` — acceptable because
 * the worker inference path never takes snapshots. TODO: real writeback if a
 * Windows snapshot-save path is ever needed. */

FILE *fmemopen(void *buf, size_t size, const char *mode) {
    FILE *fp = tmpfile();
    if (!fp) return NULL;
    if (buf && size && strchr(mode, 'r')) {
        fwrite(buf, 1, size, fp);
        rewind(fp);
    }
    return fp;
}

/* --- setenv / unsetenv ----------------------------------------------------
 * _putenv_s always overwrites, so honour `overwrite` by checking first. Setting
 * an empty value is how MSVCRT deletes a variable, which is exactly what
 * unsetenv wants. */

int setenv(const char *name, const char *value, int overwrite) {
    if (!name || !*name || strchr(name, '=')) return -1;
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value ? value : "") == 0 ? 0 : -1;
}

int unsetenv(const char *name) {
    if (!name || !*name || strchr(name, '=')) return -1;
    return _putenv_s(name, "") == 0 ? 0 : -1;
}

unsigned sleep(unsigned seconds) {
    Sleep(seconds * 1000u);
    return 0;   /* never interrupted by a signal on Windows */
}

#endif /* _WIN32 */
