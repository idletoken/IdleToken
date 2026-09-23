/* The same bounded resolver queue under real OS lookup and controlled stalls.
 * The injected resolver exists only in this test translation unit. */
#include "../common/platform_proxy.h"
static int test_resolve(const char *, const char *, const char *, const char *, idletoken_proxy_routes *);
#define IDLETOKEN_PROXY_RESOLVE test_resolve
#include "../common/platform_proxy.c"
#include <assert.h>

static pthread_mutex_t test_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t start = PTHREAD_COND_INITIALIZER;
static int waiting, go, stall, calls, peak, active, failures, budget = 5000;
static const char *target = "https://example.test/resolver-test";
static int test_resolve(const char *url, const char *scheme, const char *host, const char *port, idletoken_proxy_routes *routes) {
    pthread_mutex_lock(&test_lock);
    calls++; active++; if (active > peak) peak = active;
    pthread_mutex_unlock(&test_lock);
    int rc;
    if (stall == 1) {
#ifdef _WIN32
        Sleep(500);
#else
        struct timespec pause = {0, 500000000}; nanosleep(&pause, NULL);
#endif
        rc = add_route(routes, "");
    } else rc = native_proxies(url, scheme, host, port, routes);
    pthread_mutex_lock(&test_lock); active--; pthread_mutex_unlock(&test_lock);
    return rc;
}
static void *run(void *unused) {
    (void)unused;
    pthread_mutex_lock(&test_lock); waiting++; pthread_cond_broadcast(&start);
    while (!go) pthread_cond_wait(&start, &test_lock);
    pthread_mutex_unlock(&test_lock);
    idletoken_proxy_routes routes;
    int rc = idletoken_platform_proxies(target, budget, &routes);
    pthread_mutex_lock(&test_lock); if (rc) failures++; pthread_mutex_unlock(&test_lock);
    return NULL;
}
static void batch(int expected) {
    pthread_t threads[16];
    waiting = go = failures = 0;
    int64_t began = idletoken_platform_now_ms();
    for (int i = 0; i < 16; i++) assert(!pthread_create(&threads[i], NULL, run, NULL));
    pthread_mutex_lock(&test_lock);
    while (waiting < 16) pthread_cond_wait(&start, &test_lock);
    go = 1; pthread_cond_broadcast(&start); pthread_mutex_unlock(&test_lock);
    for (int i = 0; i < 16; i++) pthread_join(threads[i], NULL);
    printf("{\"stall\":%d,\"queries\":16,\"failures\":%d,\"elapsed_ms\":%lld,\"peak_native\":%d}\n",
           stall, failures, (long long)(idletoken_platform_now_ms() - began), peak);
    assert(failures == expected && peak <= 2);
    assert(idletoken_platform_now_ms() - began < budget + 600);
}
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--idletoken-system-proxy") && getenv("IDLETOKEN_TEST_RESOLVER_HANG")) {
        /* A genuinely stuck native child, not a slow resolver that will return. */
        for (;;) {
#ifdef _WIN32
            Sleep(1000);
#else
            sleep(1);
#endif
        }
    }
    int helper = idletoken_platform_proxy_helper(argc, argv);
    if (helper >= 0) return helper;
    const char *names[] = {"HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy","NO_PROXY","no_proxy"};
    for (unsigned i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
#ifdef _WIN32
        _putenv_s(names[i], "");
#else
        unsetenv(names[i]);
#endif
    }
    char url[256]; assert(!idletoken_platform_url(target, NULL, url, sizeof(url)));
    batch(0);
    stall = 1; budget = 120; calls = 0;
    batch(16);
    assert(calls == 2); /* queued timeouts must not spawn more workers */
    for (;;) {
        pthread_mutex_lock(&resolver_lock); int running = active_resolvers; pthread_mutex_unlock(&resolver_lock);
        if (!running) break;
#ifdef _WIN32
        Sleep(10);
#else
        struct timespec pause = {0, 10000000}; nanosleep(&pause, NULL);
#endif
    }
    stall = 0; budget = 5000;
    batch(0);
    /* Poison both native slots, then prove they are reaped without restarting
     * the caller. The test-only variable is read only by this probe binary. */
#ifdef _WIN32
    _putenv_s("IDLETOKEN_TEST_RESOLVER_HANG", "1");
#else
    setenv("IDLETOKEN_TEST_RESOLVER_HANG", "1", 1);
#endif
    stall = 2; budget = 200;
    batch(16);
    int64_t until = idletoken_platform_now_ms() + 6000;
    for (;;) {
        pthread_mutex_lock(&resolver_lock); int running = active_resolvers; pthread_mutex_unlock(&resolver_lock);
        if (!running) break;
        assert(idletoken_platform_now_ms() < until);
#ifdef _WIN32
        Sleep(20);
#else
        struct timespec pause = {0, 20000000}; nanosleep(&pause, NULL);
#endif
    }
#ifdef _WIN32
    _putenv_s("IDLETOKEN_TEST_RESOLVER_HANG", "");
#else
    unsetenv("IDLETOKEN_TEST_RESOLVER_HANG");
#endif
    stall = 0; budget = 5000;
    batch(0);
    puts("PLATFORM_PROXY_STRESS_OK");
    return 0;
}
