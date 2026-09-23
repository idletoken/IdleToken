#ifndef IDLETOKEN_PLATFORM_PROXY_H
#define IDLETOKEN_PLATFORM_PROXY_H
#define IDLETOKEN_PROXY_ROUTES 8
typedef struct {
    unsigned count;
    char urls[IDLETOKEN_PROXY_ROUTES][2048]; /* empty string = explicitly direct */
} idletoken_proxy_routes;
/* Per-request policy discovery with bounded native PAC evaluation. */
int idletoken_platform_proxies(const char *url, int budget_ms,
                              idletoken_proxy_routes *routes);
int idletoken_platform_proxies_cancel(const char *url, int budget_ms, idletoken_proxy_routes *routes,
                                     int (*cancelled)(void *), void *context);
#endif
