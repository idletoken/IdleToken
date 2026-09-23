#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "platform_proxy.h"
#include "idletoken_platform_http.h"
#include <curl/curl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <winhttp.h>
#define strcasecmp _stricmp
#else
#include <strings.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <CFNetwork/CFNetwork.h>
#include <mach-o/dyld.h>
#endif
#ifndef _WIN32
#include <fcntl.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <poll.h>
#include <signal.h>
#include <grp.h>
#include <sys/resource.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#include <limits.h>
#include <errno.h>
#endif

static int add_route(idletoken_proxy_routes *r, const char *url) {
    if (r->count == IDLETOKEN_PROXY_ROUTES || strlen(url) >= sizeof(r->urls[0]) || strpbrk(url, "\r\n\t ")) return -1;
    if (!strcmp(url, "direct://")) url = "";
    if (*url && !strstr(url, "://")) {
        if (snprintf(r->urls[r->count], sizeof(r->urls[0]), "http://%s", url) >= (int)sizeof(r->urls[0])) return -1;
    } else snprintf(r->urls[r->count], sizeof(r->urls[0]), "%s", url);
    r->count++;
    return 0;
}

static const char *first_env(const char *a, const char *b) {
    const char *v = getenv(a);
    return v && *v ? v : getenv(b);
}

static int loopback(const char *host) {
    unsigned char bytes[16];
    if (!strcasecmp(host, "localhost") || !strcasecmp(host, "localhost.")) return 1;
    if (inet_pton(AF_INET, host, bytes) == 1) return bytes[0] == 127;
    if (inet_pton(AF_INET6, host, bytes) == 1) {
        static const unsigned char v6[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
        static const unsigned char mapped[12] = {0,0,0,0,0,0,0,0,0,0,255,255};
        return !memcmp(bytes, v6, 16) || (!memcmp(bytes, mapped, 12) && bytes[12] == 127);
    }
    return 0;
}

/* NO_PROXY supports DNS suffixes, optional ports, IPv6 and numeric CIDRs.
 * Windows bypass lists also use semicolons, *.suffix and <local>. */
static int bypass(const char *list, const char *host, const char *port) {
    if (!list || !*list || strlen(list) > 16384) return 0;
    char *copy = strdup(list), *save = NULL;
    if (!copy) return 0;
    int result = 0;
    for (char *entry = strtok_r(copy, ",; \t", &save); entry; entry = strtok_r(NULL, ",; \t", &save)) {
        if (!strcmp(entry, "*") || (!strcmp(entry, "<local>") && !strchr(host, '.') && !strchr(host, ':'))) { result = 1; break; }
        char *slash = strchr(entry, '/');
        if (slash) {
            *slash++ = 0; char *end;
            long bits = strtol(slash, &end, 10);
            unsigned char network[16], address[16];
            int family = strchr(entry, ':') ? AF_INET6 : AF_INET;
            if (*slash && !*end && bits >= 0 && bits <= (family == AF_INET ? 32 : 128) &&
                inet_pton(family, entry, network) == 1 && inet_pton(family, host, address) == 1) {
                int match = 1;
                for (long bit = 0; bit < bits; bit++) if ((network[bit / 8] ^ address[bit / 8]) & (0x80 >> (bit % 8))) { match = 0; break; }
                if (match) { result = 1; break; }
            }
            continue;
        }
        char *entry_port = NULL;
        if (*entry == '[') {
            char *close = strchr(++entry, ']');
            if (!close) continue;
            if (close[1] == ':') entry_port = close + 2;
            *close = 0;
        } else {
            char *colon = strchr(entry, ':');
            if (colon && !strchr(colon + 1, ':')) { *colon = 0; entry_port = colon + 1; }
        }
        if (entry_port && strcmp(entry_port, port)) continue;
        if (*entry == '*') entry++;
        if (*entry == '.') entry++;
        size_t h = strlen(host), n = strlen(entry);
        if (n && h >= n && !strcasecmp(host + h - n, entry) && (h == n || host[h - n - 1] == '.')) { result = 1; break; }
    }
    free(copy);
    return result;
}

#ifdef _WIN32
static char *utf8(LPCWSTR str) {
    if (!str) return NULL;
    int n = WideCharToMultiByte(CP_UTF8, 0, str, -1, NULL, 0, NULL, NULL);
    char *out = n > 0 ? malloc((size_t)n) : NULL;
    if (out) WideCharToMultiByte(CP_UTF8, 0, str, -1, out, n, NULL, NULL);
    return out;
}
static int windows_list(LPCWSTR input, const char *scheme, idletoken_proxy_routes *routes) {
    char *text = utf8(input), *save = NULL;
    if (!text) return -1;
    char socks[2048] = "";
    int result = 0;
    for (char *item = strtok_r(text, "; ", &save); item; item = strtok_r(NULL, "; ", &save)) {
        char *eq = strchr(item, '=');
        if (eq) {
            *eq = 0;
            if (!strcasecmp(item, "socks")) {
                if (snprintf(socks, sizeof(socks), "socks5h://%s", eq + 1) >= (int)sizeof(socks)) { result = -1; break; }
                continue;
            }
            if (strcasecmp(item, scheme)) continue;
            item = eq + 1;
        }
        if (!strcasecmp(item, "DIRECT")) item = "";
        if (add_route(routes, item)) { result = -1; break; }
    }
    free(text);
    /* A per-scheme setting without a match explicitly leaves this scheme direct. */
    if (!result && !routes->count) result = add_route(routes, socks);
    return result;
}
typedef struct {
    HANDLE completed;
    volatile LONG references;
    DWORD error;
} windows_proxy_query;

static void windows_proxy_query_release(windows_proxy_query *query) {
    if (InterlockedDecrement(&query->references) == 0) {
        CloseHandle(query->completed);
        free(query);
    }
}

static void CALLBACK windows_proxy_complete(HINTERNET handle, DWORD_PTR context,
                                            DWORD status, void *info, DWORD length) {
    (void)handle;
    windows_proxy_query *query = (windows_proxy_query *)context;
    if (!query) return;
    if (status == WINHTTP_CALLBACK_STATUS_HANDLE_CLOSING) {
        /* Closing is the final callback, including after a cancelled lookup. */
        windows_proxy_query_release(query);
    } else if (status == WINHTTP_CALLBACK_STATUS_GETPROXYFORURL_COMPLETE) {
        query->error = ERROR_SUCCESS;
        SetEvent(query->completed);
    } else if (status == WINHTTP_CALLBACK_STATUS_REQUEST_ERROR) {
        query->error = info && length >= sizeof(WINHTTP_ASYNC_RESULT)
            ? ((WINHTTP_ASYNC_RESULT *)info)->dwError : ERROR_WINHTTP_INTERNAL_ERROR;
        SetEvent(query->completed);
    }
}

static int windows_proxy_entries(const WINHTTP_PROXY_RESULT *proxies, idletoken_proxy_routes *routes) {
    for (DWORD i = 0; i < proxies->cEntries; i++) {
        const WINHTTP_PROXY_RESULT_ENTRY *entry = &proxies->pEntries[i];
        if (!entry->fProxy) {
            if (add_route(routes, "")) return -1;
            continue;
        }
        const char *prefix = entry->ProxyScheme == INTERNET_SCHEME_HTTP ? "http" :
            entry->ProxyScheme == INTERNET_SCHEME_HTTPS ? "https" :
            entry->ProxyScheme == INTERNET_SCHEME_SOCKS ? "socks5h" : NULL;
        char *host = utf8(entry->pwszProxy), route[2048];
        if (!prefix || !host || !*host || !entry->ProxyPort) { free(host); return -1; }
        int brackets = strchr(host, ':') != NULL && host[0] != '[';
        int length = snprintf(route, sizeof(route), "%s://%s%s%s:%u", prefix,
                              brackets ? "[" : "", host, brackets ? "]" : "", entry->ProxyPort);
        free(host);
        if (length < 0 || length >= (int)sizeof(route) || add_route(routes, route)) return -1;
    }
    return routes->count ? 0 : -1;
}

static int windows_auto_proxy(const char *url, const char *scheme,
                              const WINHTTP_CURRENT_USER_IE_PROXY_CONFIG *cfg,
                              idletoken_proxy_routes *routes) {
    int result = -1;
    HINTERNET session = WinHttpOpen(L"IdleToken", WINHTTP_ACCESS_TYPE_NO_PROXY, NULL, NULL, WINHTTP_FLAG_ASYNC);
    HINTERNET resolver = NULL;
    windows_proxy_query *query = calloc(1, sizeof(*query));
    int len = MultiByteToWideChar(CP_UTF8, 0, url, -1, NULL, 0);
    WCHAR *wide = len > 0 ? malloc((size_t)len * sizeof(WCHAR)) : NULL;
    if (query) { query->references = 1; query->completed = CreateEventW(NULL, TRUE, FALSE, NULL); }
    if (!session || !query || !query->completed || !wide) goto done;
    MultiByteToWideChar(CP_UTF8, 0, url, -1, wide, len);
    WinHttpSetTimeouts(session, 3000, 3000, 3000, 3000);
    if (WinHttpSetStatusCallback(session, windows_proxy_complete,
        WINHTTP_CALLBACK_FLAG_REQUEST_ERROR | WINHTTP_CALLBACK_FLAG_GETPROXYFORURL_COMPLETE |
        WINHTTP_CALLBACK_FLAG_HANDLES, 0) == WINHTTP_INVALID_STATUS_CALLBACK ||
        WinHttpCreateProxyResolver(session, &resolver) != ERROR_SUCCESS) goto done;
    DWORD_PTR context = (DWORD_PTR)query;
    if (!WinHttpSetOption(resolver, WINHTTP_OPTION_CONTEXT_VALUE, &context, sizeof(context))) goto done;
    InterlockedIncrement(&query->references);
    WINHTTP_AUTOPROXY_OPTIONS options = {0};
    options.dwFlags = cfg->lpszAutoConfigUrl ? WINHTTP_AUTOPROXY_CONFIG_URL : WINHTTP_AUTOPROXY_AUTO_DETECT;
    options.lpszAutoConfigUrl = cfg->lpszAutoConfigUrl;
    options.dwAutoDetectFlags = cfg->lpszAutoConfigUrl ? 0 : WINHTTP_AUTO_DETECT_TYPE_DHCP | WINHTTP_AUTO_DETECT_TYPE_DNS_A;
    options.fAutoLogonIfChallenged = FALSE;
    /* The old WinHttpGetProxyForUrl loses DIRECT entries in mixed PAC lists. */
    DWORD error = WinHttpGetProxyForUrlEx(resolver, wide, &options, context);
    if (error == ERROR_IO_PENDING) {
        error = WaitForSingleObject(query->completed, 3500) == WAIT_OBJECT_0
            ? query->error : ERROR_WINHTTP_TIMEOUT;
    }
    if (error == ERROR_SUCCESS) {
        WINHTTP_PROXY_RESULT proxies = {0};
        if (WinHttpGetProxyResult(resolver, &proxies) == ERROR_SUCCESS) {
            result = windows_proxy_entries(&proxies, routes);
            WinHttpFreeProxyResult(&proxies);
        }
    } else if (!cfg->lpszAutoConfigUrl && error == ERROR_WINHTTP_AUTODETECTION_FAILED) {
        /* No discovered WPAD is an ordinary default. Explicit PAC errors fail. */
        result = cfg->lpszProxy ? windows_list(cfg->lpszProxy, scheme, routes) : add_route(routes, "");
    }
done:
    if (resolver) WinHttpCloseHandle(resolver);
    if (session) WinHttpCloseHandle(session);
    if (query) windows_proxy_query_release(query);
    free(wide);
    return result;
}
static int system_proxies(const char *url, const char *scheme, const char *host, const char *port, idletoken_proxy_routes *routes) {
    WINHTTP_CURRENT_USER_IE_PROXY_CONFIG cfg = {0};
    if (!WinHttpGetIEProxyConfigForCurrentUser(&cfg)) {
        /* A service or scheduled-task account may have no per-user proxy
         * configuration at all (ERROR_FILE_NOT_FOUND). That is "no proxy is
         * configured", not "the configured proxy failed": the documented
         * fallback is the machine-wide WinHTTP setting (netsh winhttp), and
         * with none of that either the answer is direct. Every other failure
         * still refuses rather than silently going direct. */
        if (GetLastError() != ERROR_FILE_NOT_FOUND) return -1;
        WINHTTP_PROXY_INFO machine = {0};
        if (!WinHttpGetDefaultProxyConfiguration(&machine)) return -1;
        int fallback;
        char *machine_exceptions = utf8(machine.lpszProxyBypass);
        if (machine.dwAccessType == WINHTTP_ACCESS_TYPE_NAMED_PROXY && machine.lpszProxy) {
            fallback = bypass(machine_exceptions, host, port) ? add_route(routes, "")
                     : windows_list(machine.lpszProxy, scheme, routes);
        } else fallback = add_route(routes, "");
        free(machine_exceptions);
        if (machine.lpszProxy) GlobalFree(machine.lpszProxy);
        if (machine.lpszProxyBypass) GlobalFree(machine.lpszProxyBypass);
        return fallback;
    }
    int result = -1;
    char *exceptions = utf8(cfg.lpszProxyBypass);
    if (bypass(exceptions, host, port)) result = add_route(routes, "");
    else if (cfg.lpszAutoConfigUrl || cfg.fAutoDetect) {
        result = windows_auto_proxy(url, scheme, &cfg, routes);
    } else result = cfg.lpszProxy ? windows_list(cfg.lpszProxy, scheme, routes) : add_route(routes, "");
    free(exceptions);
    if (cfg.lpszProxy) GlobalFree(cfg.lpszProxy);
    if (cfg.lpszProxyBypass) GlobalFree(cfg.lpszProxyBypass);
    if (cfg.lpszAutoConfigUrl) GlobalFree(cfg.lpszAutoConfigUrl);
    return result;
}
#elif defined(__APPLE__)
typedef struct { CFArrayRef proxies; int finished; } pac_result;
static void pac_complete(void *info, CFArrayRef proxies, CFErrorRef error) {
    pac_result *r = info;
    if (!error && proxies) r->proxies = CFRetain(proxies);
    r->finished = 1;
}
static int mac_routes(CFArrayRef proxies, CFURLRef target, idletoken_proxy_routes *out, int depth) {
    if (!proxies || depth > 1) return -1;
    for (CFIndex i = 0; i < CFArrayGetCount(proxies) && out->count < IDLETOKEN_PROXY_ROUTES; i++) {
        CFDictionaryRef item = CFArrayGetValueAtIndex(proxies, i);
        CFStringRef type = CFDictionaryGetValue(item, kCFProxyTypeKey);
        if (!type) return -1;
        if (CFEqual(type, kCFProxyTypeNone)) { if (add_route(out, "")) return -1; }
        else if (CFEqual(type, kCFProxyTypeAutoConfigurationURL) || CFEqual(type, kCFProxyTypeAutoConfigurationJavaScript)) {
            pac_result result = {0};
            CFStreamClientContext context = {0, &result, NULL, NULL, NULL};
            CFRunLoopSourceRef source = NULL;
            if (CFEqual(type, kCFProxyTypeAutoConfigurationURL)) {
                CFURLRef pac = CFDictionaryGetValue(item, kCFProxyAutoConfigurationURLKey);
                if (pac) source = CFNetworkExecuteProxyAutoConfigurationURL(pac, target, pac_complete, &context);
            } else {
                CFStringRef script = CFDictionaryGetValue(item, kCFProxyAutoConfigurationJavaScriptKey);
                if (script) source = CFNetworkExecuteProxyAutoConfigurationScript(script, target, pac_complete, &context);
            }
            if (!source) return -1;
            CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopDefaultMode);
            int64_t until = idletoken_platform_now_ms() + 4000;
            while (!result.finished && idletoken_platform_now_ms() < until) CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, true);
            CFRunLoopSourceInvalidate(source); CFRelease(source);
            int rc = mac_routes(result.proxies, target, out, depth + 1);
            if (result.proxies) CFRelease(result.proxies);
            if (rc) return -1;
        } else {
            const char *prefix = CFEqual(type, kCFProxyTypeSOCKS) ? "socks5h" :
                (CFEqual(type, kCFProxyTypeHTTP) || CFEqual(type, kCFProxyTypeHTTPS)) ? "http" : NULL;
            CFStringRef h = CFDictionaryGetValue(item, kCFProxyHostNameKey);
            CFNumberRef p = CFDictionaryGetValue(item, kCFProxyPortNumberKey);
            char host[512], route[2048]; int port = 0;
            if (!prefix || !h || !p || !CFStringGetCString(h, host, sizeof(host), kCFStringEncodingUTF8) ||
                !CFNumberGetValue(p, kCFNumberIntType, &port) || port < 1 || port > 65535) return -1;
            snprintf(route, sizeof(route), "%s://%s%s%s:%d", prefix, strchr(host, ':') ? "[" : "", host, strchr(host, ':') ? "]" : "", port);
            if (add_route(out, route)) return -1;
        }
    }
    return out->count ? 0 : -1;
}
static int system_proxies(const char *url, const char *scheme, const char *host, const char *port, idletoken_proxy_routes *routes) {
    (void)scheme; (void)host; (void)port;
    CFDictionaryRef settings = CFNetworkCopySystemProxySettings();
    CFURLRef target = CFURLCreateWithBytes(NULL, (const UInt8 *)url, strlen(url), kCFStringEncodingUTF8, NULL);
    if (!settings || !target) { if (settings) CFRelease(settings); if (target) CFRelease(target); return -1; }
    CFArrayRef proxies = CFNetworkCopyProxiesForURL(target, settings);
    int result = mac_routes(proxies, target, routes, 0);
    if (proxies) CFRelease(proxies);
    CFRelease(target); CFRelease(settings);
    return result;
}
#elif defined(__linux__)
static int linux_proxy_warning;
/* Layout of GLib's GLogField; the helper is built without GLib headers. */
struct proxy_log_field { const char *key; const void *value; ssize_t length; };
/* Only the resolver's OWN warnings mean "this PAC verdict cannot be trusted".
 * A desktop emits GLib warnings from many domains that have nothing to do with
 * proxy policy: no session bus under a systemd user unit, a missing GSettings
 * schema on a non-GNOME desktop, a dconf hiccup. Treating every one of them as
 * a failed resolution blocked every platform request on such a machine even
 * when the answer was DIRECT.
 *
 * The bundled libproxy 0.5.4 logs under G_LOG_DOMAIN "pxbackend" (from its
 * src/backend/meson.build); a malformed PAC surfaces as
 *   0x10 pxbackend  px_manager_expand_pac: Unable to set PAC ... while online
 * (0x10 = G_LOG_LEVEL_WARNING). Every benign message observed on a real GNOME
 * box is either GLib-GIO/GObject infrastructure or a pxbackend *info* line
 * (0x80), which never reaches this mask. So the block set is the resolver's
 * own domains; a warning from any other domain, or with no domain field, is
 * treated as desktop noise. The message text is never parsed -- it can carry
 * the PAC URL or credentials. The pac-bad positive control in
 * scripts/platform_proxy_gate.py pins the fail-closed side: narrow this and
 * a bad PAC silently becomes a direct connection. */
/* INVERSE allowlist, and deliberately so. Measured on the bundled libproxy
 * 0.5.4 (DGX, 2026-09-20): EVERY PAC failure -- server unreachable, DNS failure,
 * malformed JavaScript -- is a WARNING under domain "pxbackend" ("Unable to
 * download/set PAC ..."); a valid DIRECT verdict emits nothing. The only benign
 * WARNING from broken desktop config (garbage GSETTINGS_BACKEND, missing session
 * bus, non-GNOME desktop) is under "GLib-GIO". So ignore the GLib ecosystem and
 * treat EVERY other domain -- and an absent domain -- as a failure.
 *
 * Listing the failure domains instead (pxbackend/pacrunner/duktape) works for
 * this exact build but fails OPEN for a resolver that renames its domain: a bad
 * PAC would become a silent direct connection, which hard rule #10 forbids.
 * Failing CLOSED here is a loud, recoverable "system proxy resolution failed".
 * The resolver process execs fresh and loads only libproxy, its config plugins
 * and GLib, so nothing unrelated can emit under a benign domain. The message
 * text is never parsed -- it can carry the PAC URL or credentials. */
static int proxy_log_from_resolver(const void *fields, size_t count) {
    static const char *benign[] = { "glib", "gio", "gvfs", "dconf", "gsettings" };
    const struct proxy_log_field *field = fields;
    for (size_t i = 0; field && i < count; i++) {
        if (!field[i].key || strcmp(field[i].key, "GLIB_DOMAIN")) continue;
        const char *value = field[i].value;
        if (!value) return 1; /* a warning with no domain: fail closed */
        size_t n = field[i].length < 0 ? strlen(value) : (size_t)field[i].length;
        char domain[64];
        if (n >= sizeof(domain)) n = sizeof(domain) - 1;
        for (size_t k = 0; k < n; k++) domain[k] = (char)tolower((unsigned char)value[k]);
        domain[n] = 0;
        for (size_t b = 0; b < sizeof(benign) / sizeof(benign[0]); b++)
            if (!strncmp(domain, benign[b], strlen(benign[b]))) return 0;
        return 1; /* pxbackend / pacrunner / duktape / anything unknown */
    }
    return 1; /* no GLIB_DOMAIN field at all: fail closed */
}
static int proxy_log_writer(unsigned int level, const void *fields, size_t count, void *context) {
    (void)context;
    /* GLib ERROR/CRITICAL/WARNING. Only the domain field is inspected; message
     * text may contain PAC URLs or credentials and is never parsed or logged.
     * This is an isolated resolver helper process. */
    if ((level & ((1u << 2) | (1u << 3) | (1u << 4))) && proxy_log_from_resolver(fields, count)) linux_proxy_warning = 1;
    return 1; /* G_LOG_WRITER_HANDLED */
}
static int linux_proxies(const char *url, idletoken_proxy_routes *routes) {
    /* Packaged helpers carry a private resolver: distro 0.4 silently accepts
     * some failed PACs as DIRECT. Never replace the host's library. */
    char exe[4096], bundled[8192];
    void *lib = NULL;
    ssize_t exe_len = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (exe_len > 0) {
        exe[exe_len] = 0;
        char *slash = strrchr(exe, '/');
        if (slash) {
            *slash = 0;
            const char *locations[] = {
                "../lib/IdleToken/network", /* installed package */
                "client/src-tauri/runtime/linux/network", /* source coordinator */
                "../client/src-tauri/runtime/linux/network", /* source agent */
            };
            for (unsigned i = 0; i < sizeof(locations)/sizeof(locations[0]); i++) {
                snprintf(bundled, sizeof(bundled), "%s/%s/libproxy.so.1", exe, locations[i]);
                /* An installed but unloadable library is a broken package. */
                if (access(bundled, F_OK) == 0) {
                    lib = dlopen(bundled, RTLD_NOW | RTLD_LOCAL);
                    if (!lib) return -1;
                    break;
                }
            }
        }
    }
    if (!lib) lib = dlopen("libproxy.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        /* A headless host with no desktop has no desktop proxy configuration.
         * A desktop missing its resolver must not silently become direct. */
        if (getenv("XDG_CURRENT_DESKTOP") || getenv("DESKTOP_SESSION")) return -1;
        return add_route(routes, "");
    }
    void *(*create)(void) = (void *(*)(void))dlsym(lib, "px_proxy_factory_new");
    char **(*resolve)(void *, const char *) = (char **(*)(void *, const char *))dlsym(lib, "px_proxy_factory_get_proxies");
    void (*destroy)(void *) = (void (*)(void *))dlsym(lib, "px_proxy_factory_free");
    void (*release)(char **) = (void (*)(char **))dlsym(lib, "px_proxy_factory_free_proxies");
    void (*set_writer)(int (*)(unsigned int, const void *, size_t, void *), void *, void (*)(void *)) =
        (void (*)(int (*)(unsigned int, const void *, size_t, void *), void *, void (*)(void *)))dlsym(lib, "g_log_set_writer_func");
    int result = -1;
    /* libproxy can return direct:// after a PAC download or JavaScript error,
     * with the failure exposed only as a GLib warning. Treat that warning as
     * a failed resolution; it is not an explicit DIRECT decision by the PAC.
     * Require the error channel instead of accepting an unverifiable route. */
    if (create && resolve && destroy && release && set_writer) {
        linux_proxy_warning = 0;
        set_writer(proxy_log_writer, NULL, NULL);
        void *factory = create();
        if (factory) {
            char **values = resolve(factory, url);
            if (values && values[0]) {
                result = 0;
                for (unsigned i = 0; values[i] && i < IDLETOKEN_PROXY_ROUTES; i++) if (add_route(routes, values[i])) { result = -1; break; }
            }
            if (values) release(values);
            destroy(factory);
            if (linux_proxy_warning) { memset(routes, 0, sizeof(*routes)); result = -1; }
        }
    }
    dlclose(lib);
    return result;
}
static int system_proxies(const char *url, const char *scheme, const char *host, const char *port, idletoken_proxy_routes *routes) {
    (void)scheme; (void)host; (void)port;
    return linux_proxies(url, routes);
}
#else
#error Unsupported platform for public network proxy discovery
#endif

#ifndef _WIN32
static int native_proxies(const char *url, const char *scheme, const char *host, const char *port, idletoken_proxy_routes *routes) {
    (void)scheme; (void)host; (void)port;
    char exe[4096];
#ifdef __APPLE__
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size)) return -1;
#else
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0 || n >= (ssize_t)sizeof(exe) - 1) return -1;
    exe[n] = 0;
#endif
    struct stat st;
    const char *user_home = getenv("HOME");
    int drop = geteuid() == 0 && user_home && stat(user_home, &st) == 0 && st.st_uid != 0;
    struct rlimit limits;
    unsigned long fd_limit = getrlimit(RLIMIT_NOFILE, &limits) == 0 ? limits.rlim_cur : 65536;
    /* A GUI-launched process on macOS commonly inherits RLIM_INFINITY here
     * (and Linux before close_range walks the same loop): closing two billion
     * descriptors one by one never reaches exec inside the 4 s budget, so
     * every native resolution failed on exactly the launch path users take.
     * A helper that only resolves proxy policy and exits can tolerate a
     * leaked descriptor above this bound; it cannot tolerate never starting. */
    if (fd_limit > 65536) fd_limit = 65536;
    int pipes[2], input[2];
    if (pipe(pipes)) return -1;
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, input)) { close(pipes[0]); close(pipes[1]); return -1; }
#ifdef SO_NOSIGPIPE
    int no_sigpipe = 1;
    setsockopt(input[1], SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif
    int null_fd = open("/dev/null", O_WRONLY);
    if (null_fd < 0) { close(pipes[0]); close(pipes[1]); close(input[0]); close(input[1]); return -1; }
    pid_t child = fork();
    if (child == 0) {
        dup2(pipes[1], STDOUT_FILENO); dup2(input[0], STDIN_FILENO); dup2(null_fd, STDERR_FILENO);
#ifdef SYS_close_range
        if (syscall(SYS_close_range, 3u, ~0u, 0) != 0)
#endif
            for (unsigned long fd = 3; fd < fd_limit; fd++) close((int)fd);
        if (drop && (setgroups(0, NULL) || setgid(st.st_gid) || setuid(st.st_uid))) _exit(126);
        /* OS resolver and PAC runtime run only after a fresh exec. No
         * thread can remain wedged in native DNS after this child is killed. */
        execl(exe, exe, "--idletoken-system-proxy", (char *)NULL);
        _exit(127);
    }
    close(pipes[1]); close(input[0]); close(null_fd);
    if (child < 0) { close(pipes[0]); close(input[1]); return -1; }
    /* The URL can carry private query parameters; it belongs in a pipe, not argv. */
    size_t sent = 0, length = strlen(url);
    while (sent < length) {
        ssize_t n = send(input[1], url + sent, length - sent,
#ifdef MSG_NOSIGNAL
                         MSG_NOSIGNAL
#else
                         0
#endif
        );
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        sent += (size_t)n;
    }
    close(input[1]);
    char data[IDLETOKEN_PROXY_ROUTES * 2048 + 32];
    size_t used = 0; int status = 0, ended = 0;
    int64_t until = idletoken_platform_now_ms() + 4000;
    while (idletoken_platform_now_ms() < until && used + 1 < sizeof(data)) {
        struct pollfd fd = { pipes[0], POLLIN, 0 };
        if (poll(&fd, 1, 50) > 0) {
            ssize_t got = read(pipes[0], data + used, sizeof(data) - used - 1);
            if (got <= 0) { ended = 1; break; }
            used += (size_t)got;
        }
    }
    close(pipes[0]);
    if (!ended) kill(child, SIGKILL);
    /* EOF can precede exit; reaping is separately bounded. */
    int reaped = 0;
    for (int i = 0; i < 20; i++) {
        if (waitpid(child, &status, WNOHANG) == child) { reaped = 1; break; }
        struct timespec pause = {0, 10000000}; nanosleep(&pause, NULL);
    }
    if (!reaped) { kill(child, SIGKILL); waitpid(child, &status, 0); return -1; }
    if (!ended || !WIFEXITED(status) || WEXITSTATUS(status)) return -1;
    data[used] = 0; char *save = NULL;
    for (char *line = strtok_r(data, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) if (add_route(routes, line)) return -1;
    return routes->count ? 0 : -1;
}
#else
static int native_proxies(const char *url, const char *scheme, const char *host, const char *port, idletoken_proxy_routes *routes) {
    (void)scheme; (void)host; (void)port;
    WCHAR exe[4096], command[4200];
    DWORD length = GetModuleFileNameW(NULL, exe, 4096);
    if (!length || length >= 4096) return -1;
    _snwprintf(command, 4200, L"\"%ls\" --idletoken-system-proxy", exe);
    HANDLE in_read = NULL, in_write = NULL, out_read = NULL, out_write = NULL, null_handle = INVALID_HANDLE_VALUE;
    SECURITY_ATTRIBUTES sa = {sizeof(sa), NULL, TRUE};
    PROCESS_INFORMATION process = {0};
    STARTUPINFOEXW startup = {0}; startup.StartupInfo.cb = sizeof(startup);
    int result = -1;
    if (!CreatePipe(&in_read, &in_write, &sa, 8192) || !CreatePipe(&out_read, &out_write, &sa, 32768)) goto done;
    SetHandleInformation(in_write, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0);
    null_handle = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE | FILE_SHARE_READ, &sa, OPEN_EXISTING, 0, NULL);
    if (null_handle == INVALID_HANDLE_VALUE) goto done;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = in_read; startup.StartupInfo.hStdOutput = out_write; startup.StartupInfo.hStdError = null_handle;
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(NULL, 1, 0, &bytes);
    startup.lpAttributeList = malloc(bytes);
    if (!startup.lpAttributeList) goto done;
    if (!InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &bytes)) { free(startup.lpAttributeList); startup.lpAttributeList = NULL; goto done; }
    HANDLE inherit[] = {in_read, out_write, null_handle};
    if (!UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                   inherit, sizeof(inherit), NULL, NULL)) goto done;
    if (!CreateProcessW(exe, command, NULL, NULL, TRUE, CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT,
                        NULL, NULL, &startup.StartupInfo, &process)) goto done;
    CloseHandle(in_read); in_read = NULL; CloseHandle(out_write); out_write = NULL;
    DWORD written;
    if (!WriteFile(in_write, url, (DWORD)strlen(url), &written, NULL) || written != strlen(url)) goto done;
    CloseHandle(in_write); in_write = NULL;
    char data[IDLETOKEN_PROXY_ROUTES * 2048 + 32]; size_t used = 0;
    int ended = 0;
    int64_t until = idletoken_platform_now_ms() + 4000;
    while (idletoken_platform_now_ms() < until && used + 1 < sizeof(data)) {
        DWORD available = 0;
        if (!PeekNamedPipe(out_read, NULL, 0, NULL, &available, NULL)) { ended = GetLastError() == ERROR_BROKEN_PIPE; break; }
        if (available) {
            DWORD count = available < sizeof(data) - used - 1 ? available : (DWORD)(sizeof(data) - used - 1);
            DWORD got = 0;
            if (!ReadFile(out_read, data + used, count, &got, NULL) || !got) break;
            used += got;
        } else Sleep(10);
    }
    DWORD status = 1;
    if (!ended || WaitForSingleObject(process.hProcess, 200) != WAIT_OBJECT_0 ||
        !GetExitCodeProcess(process.hProcess, &status) || status) goto done;
    data[used] = 0; char *save = NULL;
    for (char *line = strtok_r(data, "\r\n", &save); line; line = strtok_r(NULL, "\r\n", &save)) if (add_route(routes, line)) goto done;
    result = routes->count ? 0 : -1;
done:
    if (process.hProcess) {
        if (WaitForSingleObject(process.hProcess, 0) != WAIT_OBJECT_0) TerminateProcess(process.hProcess, 2);
        WaitForSingleObject(process.hProcess, 1000);
        CloseHandle(process.hProcess); CloseHandle(process.hThread);
    }
    if (startup.lpAttributeList) { DeleteProcThreadAttributeList(startup.lpAttributeList); free(startup.lpAttributeList); }
    if (in_read) CloseHandle(in_read);
    if (in_write) CloseHandle(in_write);
    if (out_read) CloseHandle(out_read);
    if (out_write) CloseHandle(out_write);
    if (null_handle != INVALID_HANDLE_VALUE) CloseHandle(null_handle);
    return result;
}
#endif

int idletoken_platform_proxy_helper(int argc, char **argv) {
    if (argc != 2 || strcmp(argv[1], "--idletoken-system-proxy")) return -1;
    char url[4096];
    size_t len = fread(url, 1, sizeof(url) - 1, stdin);
    if (!len || len == sizeof(url) - 1) return 2;
    url[len] = 0;
    CURLU *u = curl_url(); char *scheme = NULL, *host = NULL, *port = NULL;
    int result = 2;
    idletoken_proxy_routes routes = {0};
    if (!u || curl_url_set(u, CURLUPART_URL, url, 0) || curl_url_get(u, CURLUPART_SCHEME, &scheme, 0) ||
        curl_url_get(u, CURLUPART_HOST, &host, 0) || curl_url_get(u, CURLUPART_PORT, &port, CURLU_DEFAULT_PORT)) goto done;
    if (host[0] == '[') { size_t n = strlen(host); memmove(host, host + 1, n - 2); host[n - 2] = 0; }
    if (system_proxies(url, scheme, host, port, &routes)) goto done;
    for (unsigned i = 0; i < routes.count; i++) puts(routes.urls[i][0] ? routes.urls[i] : "direct://");
    result = 0;
done:
    curl_free(scheme); curl_free(host); curl_free(port); if (u) curl_url_cleanup(u);
    return result;
}

/* Native resolvers run in killable children with a four-second ceiling.
 * At most two may exist; queueing callers share their original deadline.
 * Their heap-owned requests outlive timed-out callers; no thread cancellation
 * or stack callback lifetime assumptions cross a system-library boundary. */
typedef struct {
    char url[4096], scheme[16], host[512], port[16];
    idletoken_proxy_routes routes;
    int result, done, abandoned;
} proxy_query;
static pthread_mutex_t resolver_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned active_resolvers;
/* Every request re-reads the OS policy (the Linux gate switches the system
 * proxy port in-process and expects the very next request to follow it).
 * A verdict cache would need a change signal the three OSes do not share;
 * the cost of re-reading is instead kept out of the request's own clocks by
 * the transport, which gives discovery its own bounded budget. */
static void *resolve_native(void *arg) {
    proxy_query *query = arg;
#ifndef IDLETOKEN_PROXY_RESOLVE
#define IDLETOKEN_PROXY_RESOLVE native_proxies
#endif
    int result = IDLETOKEN_PROXY_RESOLVE(query->url, query->scheme, query->host, query->port, &query->routes);
    pthread_mutex_lock(&resolver_lock);
    active_resolvers--;
    query->result = result; query->done = 1;
    if (query->abandoned) free(query);
    pthread_mutex_unlock(&resolver_lock);
    return NULL;
}

int idletoken_platform_proxies_cancel(const char *url, int budget_ms, idletoken_proxy_routes *routes,
                                     int (*cancelled)(void *), void *context) {
    const int64_t until = idletoken_platform_now_ms() + budget_ms;
    memset(routes, 0, sizeof(*routes));
    CURLU *u = curl_url();
    char *scheme = NULL, *host = NULL, *port = NULL;
    int result = -1;
    if (!u || curl_url_set(u, CURLUPART_URL, url, 0) || curl_url_get(u, CURLUPART_SCHEME, &scheme, 0) ||
        curl_url_get(u, CURLUPART_HOST, &host, 0) || curl_url_get(u, CURLUPART_PORT, &port, CURLU_DEFAULT_PORT)) goto done;
    if (host[0] == '[') { size_t n = strlen(host); memmove(host, host + 1, n - 2); host[n - 2] = 0; }
    if (loopback(host) || bypass(first_env("no_proxy", "NO_PROXY"), host, port)) { result = add_route(routes, ""); goto done; }
    const char *proxy = !strcmp(scheme, "https") ? first_env("https_proxy", "HTTPS_PROXY") : first_env("http_proxy", "HTTP_PROXY");
    if (!proxy || !*proxy) proxy = first_env("all_proxy", "ALL_PROXY");
    if (proxy && *proxy) { result = add_route(routes, proxy); goto done; }
    proxy_query *query = calloc(1, sizeof(*query));
    if (!query) goto done;
    if (strlen(url) >= sizeof(query->url) || strlen(host) >= sizeof(query->host)) { free(query); goto done; }
    snprintf(query->url, sizeof(query->url), "%s", url);
    snprintf(query->scheme, sizeof(query->scheme), "%s", scheme);
    snprintf(query->host, sizeof(query->host), "%s", host);
    snprintf(query->port, sizeof(query->port), "%s", port);
    int started = 0;
    for (;;) {
        /* Queue inside the caller's original budget. A third ordinary request
         * must not fail merely because two native lookups are still running.
         * Timed-out native calls keep their slots until they really return. */
        int expired = idletoken_platform_now_ms() >= until || (cancelled && cancelled(context));
        pthread_mutex_lock(&resolver_lock);
        if (expired) {
            if (started && !query->done) query->abandoned = 1;
            else free(query);
            pthread_mutex_unlock(&resolver_lock); break;
        }
        if (!started && active_resolvers < 2) {
            pthread_t thread;
            if (pthread_create(&thread, NULL, resolve_native, query)) {
                pthread_mutex_unlock(&resolver_lock); free(query); break;
            }
            active_resolvers++; pthread_detach(thread); started = 1;
        }
        if (started && query->done) {
            result = query->result; *routes = query->routes; free(query);
            pthread_mutex_unlock(&resolver_lock); break;
        }
        pthread_mutex_unlock(&resolver_lock);
#ifdef _WIN32
        Sleep(10);
#else
        struct timespec pause = {0, 10000000}; nanosleep(&pause, NULL);
#endif
    }
done:
    curl_free(scheme); curl_free(host); curl_free(port); if (u) curl_url_cleanup(u);
    return result;
}

int idletoken_platform_proxies(const char *url, int budget_ms, idletoken_proxy_routes *routes) {
    return idletoken_platform_proxies_cancel(url, budget_ms, routes, NULL, NULL);
}
