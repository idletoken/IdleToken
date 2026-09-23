/* Wire-level test driver for the production transport. No network mocks.
 *
 *   probe <url> <total_ms> [receipt] [cancel_ms] [upload_bytes] [body_idle_ms] [body_total_ms]
 *
 * Without upload_bytes every budget equals total_ms and the body is ten
 * bytes. With upload_bytes the request carries that many bytes, total_ms
 * becomes the "start answering" budget only (headers_ms) and the upload is
 * bounded by the body budgets, which is how the agent's sealed-result POST is
 * configured. That is the shape that must survive a slow uplink. */
#include "idletoken_platform_http.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cancelled(void *arg) {
    return idletoken_platform_now_ms() >= *(int64_t *)arg;
}
int main(int argc, char **argv) {
    int helper = idletoken_platform_proxy_helper(argc, argv);
    if (helper >= 0) return helper;
    if (argc < 3) return 2;
    int total = atoi(argv[2]);
    int64_t cancel_at = idletoken_platform_now_ms() + (argc > 4 ? atoi(argv[4]) : 0);
    size_t upload = argc > 5 ? (size_t)strtoull(argv[5], NULL, 10) : 0;
    int body_idle = argc > 6 ? atoi(argv[6]) : total;
    int body_total = argc > 7 ? atoi(argv[7]) : total;
    char url[4096];
    if (idletoken_platform_url(argv[1], NULL, url, sizeof(url))) return 2;
    char *filler = NULL;
    if (upload) {
        filler = malloc(upload);
        if (!filler) return 2;
        memset(filler, 'u', upload);
    }
    idletoken_platform_http_request request = {
        .url = url, .method = "POST",
        .body = upload ? filler : "probe-body", .body_len = upload ? upload : 10,
        .connect_ms = total, .headers_ms = total, .total_ms = upload ? 0 : total,
        .body_idle_ms = body_idle, .body_total_ms = body_total,
        .receipt = argc > 3 && atoi(argv[3]), .max_response = 4u * 1024u * 1024u,
        .cancelled = argc > 4 && atoi(argv[4]) != 0 ? cancelled : NULL,
        .cancel_context = &cancel_at,
    };
    int64_t start = idletoken_platform_now_ms();
    idletoken_platform_http_response response;
    int result = idletoken_platform_http(&request, &response);
    printf("{\"result\":%d,\"status\":%d,\"truncated\":%d,\"cancelled\":%d,\"elapsed_ms\":%lld,\"remaining_ms\":%lld}\n",
        result, response.status, response.truncated, response.cancelled,
        (long long)(idletoken_platform_now_ms() - start),
        (long long)(response.delivery_deadline_ms ? response.delivery_deadline_ms - idletoken_platform_now_ms() : 0));
    if (response.body) fwrite(response.body, 1, response.len, stdout);
    if (result) fprintf(stderr, "%s\n", response.error);
    free(response.body);
    free(filler);
    return 0;
}
