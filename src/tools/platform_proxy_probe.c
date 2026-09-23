/* Native PAC and live policy probe. It supplies test configuration directly
 * to the production resolver, without changing the user's OS proxy settings. */
#include "../common/platform_proxy.c"

int main(int argc, char **argv) {
    int helper = idletoken_platform_proxy_helper(argc, argv);
    if (helper >= 0) return helper;
    if (argc != 3) return 2;
    for (;;) {
        idletoken_proxy_routes routes = {0};
        int result = -1;
        if (!strcmp(argv[1], "system")) {
            result = idletoken_platform_proxies(argv[2], 1500, &routes);
        } else {
#ifdef _WIN32
            WCHAR pac[2048];
            WINHTTP_CURRENT_USER_IE_PROXY_CONFIG cfg = {0};
            if (MultiByteToWideChar(CP_UTF8, 0, argv[1], -1, pac, 2048)) {
                cfg.lpszAutoConfigUrl = pac;
                result = windows_auto_proxy(argv[2], "http", &cfg, &routes);
            }
#elif defined(__APPLE__)
            CFURLRef target = CFURLCreateWithBytes(NULL, (UInt8 *)argv[2], strlen(argv[2]), kCFStringEncodingUTF8, NULL);
            CFURLRef pac = CFURLCreateWithBytes(NULL, (UInt8 *)argv[1], strlen(argv[1]), kCFStringEncodingUTF8, NULL);
            const void *keys[] = {kCFProxyTypeKey, kCFProxyAutoConfigurationURLKey};
            const void *values[] = {kCFProxyTypeAutoConfigurationURL, pac};
            CFDictionaryRef entry = CFDictionaryCreate(NULL, keys, values, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
            const void *items[] = {entry};
            CFArrayRef list = CFArrayCreate(NULL, items, 1, &kCFTypeArrayCallBacks);
            result = mac_routes(list, target, &routes, 0);
            CFRelease(list); CFRelease(entry); CFRelease(pac); CFRelease(target);
#endif
        }
        printf("%d %u\n", result, routes.count);
        for (unsigned i = 0; i < routes.count; i++) puts(routes.urls[i][0] ? routes.urls[i] : "DIRECT");
        fflush(stdout);
        if (getchar() == EOF) break;
    }
    return 0;
}
