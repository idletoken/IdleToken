/* api_test.c — unit tests for the API surface helpers:
 *   - idletoken_anthropic_to_openai (system promotion, non-leading-system
 *     demotion, sampling passthrough, tools / tool_use / tool_result);
 *   - idletoken_oai_resp_to_anthropic_content (tool_calls -> tool_use blocks,
 *     finish_reason -> stop_reason);
 *   - idletoken_ip_is_overlay (IPv4 CGNAT + Tailscale IPv6 /48);
 *   - idletoken_hex64_valid (cluster RPC PSK format);
 *   - idletoken_http_path_strip_query / idletoken_http_auth_value_matches.
 * Pure C, runs anywhere:
 *   cc -Wall -Wextra -std=c99 -Iinclude src/common/apiconv.c src/common/net.c \
 *      src/common/http.c src/tools/api_test.c -o api_test && ./api_test
 * Prints API_TEST_OK on success. */
#include "idletoken_apiconv.h"
#include "idletoken_net.h"
#include "idletoken_http.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks = 0, failures = 0;
static void ok(int cond, const char *what) {
    checks++;
    if (cond) { printf("  [ok] %s\n", what); }
    else      { failures++; printf("  [FAIL] %s\n", what); }
}

/* Convert helper: run the translation, assert non-NULL, return it. */
static char *conv(const char *body, int want_stream, int dflt_max) {
    size_t ol = 0;
    char *r = idletoken_anthropic_to_openai(body, strlen(body), want_stream,
                                            dflt_max, &ol);
    if (r && ol != strlen(r)) { free(r); return NULL; }
    return r;
}

int main(void) {
    /* ---- overlay detection ------------------------------------------- */
    printf("== idletoken_ip_is_overlay ==\n");
    ok(idletoken_ip_is_overlay("100.64.0.1") == 1,      "100.64.0.1 is overlay");
    ok(idletoken_ip_is_overlay("100.127.255.255") == 1, "100.127.255.255 is overlay");
    ok(idletoken_ip_is_overlay("100.64.0.1:50052") == 1, "ip:port form still matches");
    ok(idletoken_ip_is_overlay("100.63.0.1") == 0,      "100.63.x is not");
    ok(idletoken_ip_is_overlay("100.128.0.1") == 0,     "100.128.x is not");
    ok(idletoken_ip_is_overlay("192.168.1.1") == 0,     "192.168.1.1 is not");
    ok(idletoken_ip_is_overlay("fd7a:115c:a1e0::1") == 1, "Tailscale IPv6 /48 is overlay");
    ok(idletoken_ip_is_overlay("FD7A:115C:A1E0::1") == 1, "case-insensitive IPv6 match");
    ok(idletoken_ip_is_overlay("fd7a:115c:a1e0:ab12::7") == 1, "deeper /48 address matches");
    ok(idletoken_ip_is_overlay("[fd7a:115c:a1e0::1]:14100") == 1, "bracketed [v6]:port matches");
    ok(idletoken_ip_is_overlay("fd7a:115c:a1e00::1") == 0, "longer hextet a1e00 must NOT match");
    ok(idletoken_ip_is_overlay("fd00::1") == 0,         "other ULA space stays allowed");
    ok(idletoken_ip_is_overlay("fe80::") == 0,          "link-local is not overlay");
    ok(idletoken_ip_is_overlay("") == 0,                "empty string is not");
    ok(idletoken_ip_is_overlay(NULL) == 0,              "NULL is not");

    /* ---- PSK hex64 ---------------------------------------------------- */
    printf("== idletoken_hex64_valid ==\n");
    {
        char h[80];
        memset(h, 'a', 64); h[64] = '\0';
        ok(idletoken_hex64_valid(h) == 1, "64 lowercase hex accepted");
        memset(h, 'F', 64); h[32] = '7'; h[64] = '\0';
        ok(idletoken_hex64_valid(h) == 1, "mixed-case hex accepted");
        h[63] = 'g';
        ok(idletoken_hex64_valid(h) == 0, "non-hex char rejected");
        memset(h, 'a', 63); h[63] = '\0';
        ok(idletoken_hex64_valid(h) == 0, "63 chars rejected");
        memset(h, 'a', 65); h[65] = '\0';
        ok(idletoken_hex64_valid(h) == 0, "65 chars rejected");
        ok(idletoken_hex64_valid("") == 0, "empty rejected");
    }

    /* ---- HTTP query strip + auth match -------------------------------- */
    printf("== http helpers ==\n");
    {
        char p1[64] = "/v1/messages?beta=true";
        idletoken_http_path_strip_query(p1);
        ok(strcmp(p1, "/v1/messages") == 0, "query string stripped");
        char p2[64] = "/v1/chat/completions";
        idletoken_http_path_strip_query(p2);
        ok(strcmp(p2, "/v1/chat/completions") == 0, "bare path unchanged");
        char p3[8] = "/x?";
        idletoken_http_path_strip_query(p3);
        ok(strcmp(p3, "/x") == 0, "trailing bare '?' stripped");

        ok(idletoken_http_auth_value_matches("Bearer tok123", "tok123") == 1,
           "Bearer <token> matches");
        ok(idletoken_http_auth_value_matches("bEaReR  tok123", "tok123") == 1,
           "scheme case-insensitive + extra spaces");
        ok(idletoken_http_auth_value_matches("tok123", "tok123") == 1,
           "bare token matches");
        ok(idletoken_http_auth_value_matches("Bearer wrong", "tok123") == 0,
           "wrong token refused");
        ok(idletoken_http_auth_value_matches("Bearer tok1234", "tok123") == 0,
           "prefix token refused");
    }

    /* ---- anthropic -> openai: basics ---------------------------------- */
    printf("== idletoken_anthropic_to_openai: basics ==\n");
    {
        char *r = conv("{\"system\":\"Be terse\","
                       "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                       "\"max_tokens\":50}", 0, 0);
        ok(r != NULL, "simple body converts");
        ok(r && strstr(r, "{\"role\":\"system\",\"content\":\"Be terse\"}") == r + 13,
           "system promoted to leading message");
        ok(r && strstr(r, "{\"role\":\"user\",\"content\":\"hi\"}") != NULL,
           "user message carried");
        ok(r && strstr(r, "\"max_tokens\":50") != NULL, "max_tokens carried");
        ok(r && strstr(r, "\"stream\"") == NULL, "no stream flag when not asked");
        free(r);
    }
    {
        char *r = conv("{\"system\":[{\"type\":\"text\",\"text\":\"A\"},"
                                    "{\"type\":\"text\",\"text\":\"B\"}],"
                       "\"messages\":[{\"role\":\"user\",\"content\":\"q\"}]}",
                       1, 128);
        ok(r && strstr(r, "\"content\":\"A\\nB\"") != NULL,
           "system text blocks joined with \\n");
        ok(r && strstr(r, "\"max_tokens\":128") != NULL,
           "max_tokens falls back to the default");
        ok(r && strstr(r, "\"stream\":true") != NULL &&
               strstr(r, "\"include_usage\":true") != NULL,
           "stream + usage flags on streaming requests");
        free(r);
    }
    {
        char *r = conv("{\"messages\":["
                       "{\"role\":\"user\",\"content\":\"a\"},"
                       "{\"role\":\"system\",\"content\":\"reminder\"}]}", 0, 0);
        ok(r && strstr(r, "{\"role\":\"user\",\"content\":\"reminder\"}") != NULL &&
               strstr(r, "\"role\":\"system\"") == NULL,
           "non-leading system demoted to user");
        free(r);
    }
    {
        size_t ol = 0;
        ok(idletoken_anthropic_to_openai("{\"messages\":[]}", 15, 0, 0, &ol) == NULL,
           "no usable message -> NULL");
    }

    /* ---- anthropic -> openai: sampling passthrough --------------------- */
    printf("== idletoken_anthropic_to_openai: sampling ==\n");
    {
        char *r = conv("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                       "\"temperature\":0.7,\"top_p\":0.9,\"top_k\":40,"
                       "\"stop_sequences\":[\"END\",\"\\n\\n\"]}", 0, 0);
        ok(r && strstr(r, "\"temperature\":0.7") != NULL, "temperature passes through");
        ok(r && strstr(r, "\"top_p\":0.9") != NULL,       "top_p passes through");
        ok(r && strstr(r, "\"top_k\":40") != NULL,        "top_k passes through");
        ok(r && strstr(r, "\"stop\":[\"END\",\"\\n\\n\"]") != NULL,
           "stop_sequences becomes stop");
        free(r);
    }

    /* ---- anthropic -> openai: tools ------------------------------------ */
    printf("== idletoken_anthropic_to_openai: tools ==\n");
    {
        const char *body =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}],"
             "\"tools\":[{\"name\":\"get_weather\","
                         "\"description\":\"Look up weather\","
                         "\"input_schema\":{\"type\":\"object\","
                           "\"properties\":{\"city\":{\"type\":\"string\"}}}}],"
             "\"tool_choice\":{\"type\":\"any\"}}";
        ok(idletoken_body_has_tools(body, strlen(body)) == 1, "body_has_tools sees tools");
        char *r = conv(body, 0, 0);
        ok(r && strstr(r, "\"tools\":[{\"type\":\"function\",\"function\":{"
                          "\"name\":\"get_weather\"") != NULL,
           "tool translated to type:function");
        ok(r && strstr(r, "\"description\":\"Look up weather\"") != NULL,
           "tool description carried");
        ok(r && strstr(r, "\"parameters\":{\"type\":\"object\","
                          "\"properties\":{\"city\":{\"type\":\"string\"}}}") != NULL,
           "input_schema becomes parameters verbatim");
        ok(r && strstr(r, "\"tool_choice\":\"required\"") != NULL,
           "tool_choice any -> required");
        free(r);
        ok(idletoken_body_has_tools("{\"tools\":[]}", 12) == 0,
           "empty tools array is not tools");
    }
    {
        /* assistant tool_use + user tool_result round trip */
        const char *body =
            "{\"messages\":["
             "{\"role\":\"user\",\"content\":\"weather in Paris\"},"
             "{\"role\":\"assistant\",\"content\":["
               "{\"type\":\"text\",\"text\":\"Checking.\"},"
               "{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"get_weather\","
                "\"input\":{\"city\":\"Paris\"}}]},"
             "{\"role\":\"user\",\"content\":["
               "{\"type\":\"tool_result\",\"tool_use_id\":\"tu_1\","
                "\"content\":\"22C sunny\"}]}]}";
        char *r = conv(body, 0, 0);
        ok(r != NULL, "tool round-trip body converts");
        ok(r && strstr(r, "\"tool_calls\":[{\"id\":\"tu_1\",\"type\":\"function\","
                          "\"function\":{\"name\":\"get_weather\","
                          "\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]") != NULL,
           "tool_use becomes tool_calls with stringified arguments");
        ok(r && strstr(r, "{\"role\":\"assistant\",\"content\":\"Checking.\"") != NULL,
           "assistant text kept next to its tool_calls");
        ok(r && strstr(r, "{\"role\":\"tool\",\"tool_call_id\":\"tu_1\","
                          "\"content\":\"22C sunny\"}") != NULL,
           "tool_result becomes a role:tool message");
        free(r);
    }
    {
        /* tool_result with block-array content + trailing user text */
        const char *body =
            "{\"messages\":["
             "{\"role\":\"user\",\"content\":["
               "{\"type\":\"tool_result\",\"tool_use_id\":\"tu_9\","
                "\"content\":[{\"type\":\"text\",\"text\":\"out1\"},"
                             "{\"type\":\"text\",\"text\":\"out2\"}]},"
               "{\"type\":\"text\",\"text\":\"continue\"}]}]}";
        char *r = conv(body, 0, 0);
        ok(r && strstr(r, "{\"role\":\"tool\",\"tool_call_id\":\"tu_9\","
                          "\"content\":\"out1\\nout2\"}") != NULL,
           "tool_result text blocks flattened");
        const char *toolmsg = r ? strstr(r, "\"role\":\"tool\"") : NULL;
        const char *usermsg = r ? strstr(r, "{\"role\":\"user\",\"content\":\"continue\"}") : NULL;
        ok(toolmsg && usermsg && toolmsg < usermsg,
           "tool message precedes the user text of the same turn");
        free(r);
    }

    /* ---- openai response -> anthropic ---------------------------------- */
    printf("== idletoken_oai_resp_to_anthropic_content ==\n");
    {
        const char *resp =
            "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
             "\"content\":\"Using the tool now.\","
             "\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\","
               "\"function\":{\"name\":\"get_weather\","
                "\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]},"
             "\"finish_reason\":\"tool_calls\"}],"
             "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}";
        char sr[16] = "";
        size_t ol = 0;
        char *r = idletoken_oai_resp_to_anthropic_content(resp, strlen(resp),
                                                          sr, sizeof(sr), &ol);
        ok(r != NULL, "tool-call response converts");
        ok(r && strstr(r, "{\"type\":\"text\",\"text\":\"Using the tool now.\"}") != NULL,
           "assistant text becomes a text block");
        ok(r && strstr(r, "{\"type\":\"tool_use\",\"id\":\"call_1\","
                          "\"name\":\"get_weather\","
                          "\"input\":{\"city\":\"Paris\"}}") != NULL,
           "tool_calls becomes tool_use with parsed input");
        ok(strcmp(sr, "tool_use") == 0, "finish_reason tool_calls -> stop_reason tool_use");
        free(r);
    }
    {
        const char *resp =
            "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
             "\"content\":\"plain answer\"},\"finish_reason\":\"stop\"}]}";
        char sr[16] = "";
        char *r = idletoken_oai_resp_to_anthropic_content(resp, strlen(resp),
                                                          sr, sizeof(sr), NULL);
        ok(r && strcmp(r, "[{\"type\":\"text\",\"text\":\"plain answer\"}]") == 0,
           "text-only response converts to one text block");
        ok(strcmp(sr, "end_turn") == 0, "finish_reason stop -> end_turn");
        free(r);
    }
    {
        /* malformed arguments must degrade to {} — never to invalid JSON */
        const char *resp =
            "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
             "\"content\":\"\","
             "\"tool_calls\":[{\"id\":\"c2\",\"type\":\"function\","
               "\"function\":{\"name\":\"f\",\"arguments\":\"{broken\"}}]},"
             "\"finish_reason\":\"tool_calls\"}]}";
        char sr[16] = "";
        char *r = idletoken_oai_resp_to_anthropic_content(resp, strlen(resp),
                                                          sr, sizeof(sr), NULL);
        ok(r && strstr(r, "\"input\":{}") != NULL,
           "unparseable arguments degrade to an empty input object");
        ok(r && strstr(r, "\"type\":\"text\"") == NULL,
           "no empty text block when the reply is tool-only");
        free(r);
    }

    /* ---- unescape ------------------------------------------------------ */
    printf("== idletoken_json_unescape ==\n");
    {
        size_t ol = 0;
        char *u = idletoken_json_unescape("a\\nb\\\"c\\u0041", 13, &ol);
        ok(u && ol == 6 && memcmp(u, "a\nb\"cA", 6) == 0,
           "escapes and \\u0041 decode");
        free(u);
    }

    printf("%d checks, %d failures\n", checks, failures);
    if (failures) { printf("API_TEST_FAILED\n"); return 1; }
    printf("API_TEST_OK\n");
    return 0;
}
