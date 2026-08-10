#ifdef _WIN32
#include <io.h>
#include <stdio.h>
#include <time.h>
#include <process.h>
#include <stdlib.h>
#include <BaseTsd.h>
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef SSIZE_T ssize_t;
#endif
#ifndef SSIZE_MAX
#define SSIZE_MAX 0x7fffffffffffffffLL
#endif
#ifndef __attribute__
#define __attribute__(x)
#endif
#ifndef getpid
#define getpid _getpid
#endif
#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#endif
#define st_blksize st_rdev
#define _SC_PAGESIZE 1
#define _SC_NPROCESSORS_ONLN 2
static __inline long sysconf(int n){if(n==_SC_PAGESIZE)return 4096;if(n==_SC_NPROCESSORS_ONLN)return 8;return -1;}
#ifndef CLOCK_MONOTONIC
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
#endif
static __inline int clock_gettime(int clk,struct timespec *ts){(void)clk;return timespec_get(ts,TIME_UTC)==TIME_UTC?0:-1;}
static __inline ssize_t pread(int fd,void *buf,size_t count,long long offset){__int64 cur=_lseeki64(fd,0,SEEK_CUR);_lseeki64(fd,offset,SEEK_SET);int n=_read(fd,buf,(unsigned)count);_lseeki64(fd,cur,SEEK_SET);return (ssize_t)n;}
#endif
