/* IdleToken Windows platform shim: <sys/file.h> — flock() via LockFileEx.
 * ds4 uses flock for the single-process lock file (DS4_LOCK_FILE). */
#ifndef IDLETOKEN_WIN_SYS_FILE_H
#define IDLETOKEN_WIN_SYS_FILE_H
#ifdef _WIN32

#define LOCK_SH 1  /* shared lock   */
#define LOCK_EX 2  /* exclusive     */
#define LOCK_NB 4  /* non-blocking  */
#define LOCK_UN 8  /* unlock        */

#ifdef __cplusplus
extern "C" {
#endif

int flock(int fd, int operation);

#ifdef __cplusplus
}
#endif

#endif /* _WIN32 */
#endif /* IDLETOKEN_WIN_SYS_FILE_H */
