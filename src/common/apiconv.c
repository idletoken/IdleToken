/* apiconv.c — see include/idletoken_apiconv.h. C99, pure functions. */
#include "idletoken_apiconv.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- tiny growing string builder ----------------------------------------- */

typedef struct { char *p; size_t len, cap; int oom; } sb_t;

static void sb_put(sb_t *b, const char *s, size_t n) {
    if (b->oom || n == 0) return;
    if (b->len + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap : 512;
        while (nc < b->len + n + 1) nc *= 2;
        char *np = realloc(b->p, nc);
        if (!np) { b->oom = 1; return; }
        b->p = np;
        b->cap = nc;
    }
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void sb_cstr(sb_t *b, const char *s) { sb_put(b, s, strlen(s)); }

/* Escape raw bytes for embedding inside a JSON string literal. Used where the
 * two protocols disagree about nesting: an Anthropic "input" OBJECT becomes
 * the OpenAI "arguments" STRING. */
static void sb_put_json_escaped(sb_t *b, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  sb_cstr(b, "\\\""); break;
        case '\\': sb_cstr(b, "\\\\"); break;
        case '\n': sb_cstr(b, "\\n");  break;
        case '\r': sb_cstr(b, "\\r");  break;
        case '\t': sb_cstr(b, "\\t");  break;
        default:
            if (c < 0x20) {
                char u[8];
                snprintf(u, sizeof(u), "\\u%04x", (unsigned)c);
                sb_cstr(b, u);
            } else {
                sb_put(b, (const char *)&s[i], 1);
            }
        }
    }
}

/* ---- JSON scanning -------------------------------------------------------- */

static const char *skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

long idletoken_json_value_len(const char *v, const char *end) {
    if (!v || v >= end) return -1;
    if (*v == '"') {
        const char *p = v + 1;
        int esc = 0;
        while (p < end) {
            if (esc) esc = 0;
            else if (*p == '\\') esc = 1;
            else if (*p == '"') return (long)(p - v + 1);
            p++;
        }
        return -1;
    }
    if (*v == '{' || *v == '[') {
        int depth = 0, in_str = 0, esc = 0;
        for (const char *p = v; p < end; p++) {
            char ch = *p;
            if (esc)            esc = 0;
            else if (in_str)    { if (ch == '\\') esc = 1; else if (ch == '"') in_str = 0; }
            else if (ch == '"') in_str = 1;
            else if (ch == '{' || ch == '[') depth++;
            else if (ch == '}' || ch == ']') { if (--depth == 0) return (long)(p - v + 1); }
        }
        return -1;
    }
    /* number / true / false / null: runs until a structural byte */
    const char *p = v;
    while (p < end && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
    return p == v ? -1 : (long)(p - v);
}

const char *idletoken_json_obj_get(const char *json, size_t len, const char *key) {
    if (!json || !key) return NULL;
    const char *end = json + len;
    const char *p = skip_ws(json, end);
    if (p >= end || *p != '{') return NULL;
    p = skip_ws(p + 1, end);
    size_t klen = strlen(key);
    while (p < end && *p != '}') {
        if (*p != '"') return NULL;
        long kl = idletoken_json_value_len(p, end);
        if (kl < 2) return NULL;
        const char *kspan = p + 1;
        size_t kspan_len = (size_t)kl - 2;
        p = skip_ws(p + kl, end);
        if (p >= end || *p != ':') return NULL;
        p = skip_ws(p + 1, end);
        if (p >= end) return NULL;
        if (kspan_len == klen && memcmp(kspan, key, klen) == 0) return p;
        long vl = idletoken_json_value_len(p, end);
        if (vl < 0) return NULL;
        p = skip_ws(p + vl, end);
        if (p < end && *p == ',') p = skip_ws(p + 1, end);
    }
    return NULL;
}

int idletoken_json_obj_str(const char *json, size_t len, const char *key,
                           const char **out, size_t *out_len) {
    const char *v = idletoken_json_obj_get(json, len, key);
    if (!v || *v != '"') return -1;
    long vl = idletoken_json_value_len(v, json + len);
    if (vl < 2) return -1;
    *out = v + 1;
    *out_len = (size_t)vl - 2;
    return 0;
}

/* Iterate elements of the array starting at `arr` (points at '['): *iter is a
 * byte offset into it, start with 0. Yields the element value span. */
static int arr_next(const char *arr, const char *end, size_t *iter,
                    const char **el, long *el_len) {
    long al = idletoken_json_value_len(arr, end);
    if (al < 2 || *arr != '[') return 0;
    const char *aend = arr + al - 1;   /* the closing ']' */
    const char *p = *iter ? arr + *iter : arr + 1;
    p = skip_ws(p, aend);
    if (p >= aend) return 0;
    if (*p == ',') p = skip_ws(p + 1, aend);
    if (p >= aend) return 0;
    long vl = idletoken_json_value_len(p, aend);
    if (vl < 0) return 0;
    *el = p;
    *el_len = vl;
    *iter = (size_t)(p + vl - arr);
    return 1;
}

char *idletoken_json_unescape(const char *span, size_t len, size_t *out_len) {
    char *out = malloc(len + 1);
    if (!out) return NULL;
    size_t o = 0;
    for (size_t i = 0; i < len; i++) {
        char c = span[i];
        if (c != '\\') { out[o++] = c; continue; }
        if (++i >= len) break;
        switch (span[i]) {
        case '"':  out[o++] = '"';  break;
        case '\\': out[o++] = '\\'; break;
        case '/':  out[o++] = '/';  break;
        case 'b':  out[o++] = '\b'; break;
        case 'f':  out[o++] = '\f'; break;
        case 'n':  out[o++] = '\n'; break;
        case 'r':  out[o++] = '\r'; break;
        case 't':  out[o++] = '\t'; break;
        case 'u': {
            if (i + 4 >= len) { i = len; break; }
            unsigned cp = 0;
            int okhex = 1;
            for (int k = 1; k <= 4; k++) {
                char h = span[i + k];
                unsigned d;
                if      (h >= '0' && h <= '9') d = (unsigned)(h - '0');
                else if (h >= 'a' && h <= 'f') d = (unsigned)(h - 'a' + 10);
                else if (h >= 'A' && h <= 'F') d = (unsigned)(h - 'A' + 10);
                else { okhex = 0; break; }
                cp = cp * 16 + d;
            }
            if (!okhex) break;
            i += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < len &&
                span[i + 1] == '\\' && span[i + 2] == 'u') {
                unsigned lo = 0;
                int ok2 = 1;
                for (int k = 3; k <= 6; k++) {
                    char h = span[i + k];
                    unsigned d;
                    if      (h >= '0' && h <= '9') d = (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') d = (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') d = (unsigned)(h - 'A' + 10);
                    else { ok2 = 0; break; }
                    lo = lo * 16 + d;
                }
                if (ok2 && lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    i += 6;
                }
            }
            if (cp >= 0xD800 && cp <= 0xDFFF) break;   /* unpaired surrogate */
            if (cp < 0x80) {
                out[o++] = (char)cp;
            } else if (cp < 0x800) {
                out[o++] = (char)(0xC0 | (cp >> 6));
                out[o++] = (char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                out[o++] = (char)(0xE0 | (cp >> 12));
                out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                out[o++] = (char)(0x80 | (cp & 0x3F));
            } else {
                out[o++] = (char)(0xF0 | (cp >> 18));
                out[o++] = (char)(0x80 | ((cp >> 12) & 0x3F));
                out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                out[o++] = (char)(0x80 | (cp & 0x3F));
            }
            break;
        }
        default: out[o++] = span[i]; break;
        }
    }
    out[o] = '\0';
    if (out_len) *out_len = o;
    return out;
}

/* ---- request translation -------------------------------------------------- */

/* True when the value span looks like a bare JSON number (passthrough gate:
 * never copy an arbitrary value into the upstream body unchecked). */
static int span_is_number(const char *v, long vl) {
    if (vl <= 0) return 0;
    for (long i = 0; i < vl; i++) {
        char c = v[i];
        if (!((c >= '0' && c <= '9') || c == '-' || c == '+' ||
              c == '.' || c == 'e' || c == 'E'))
            return 0;
    }
    return 1;
}

/* Copy a top-level numeric field through, as `,"name":<raw>`. */
static void put_num_field(sb_t *b, const char *body, size_t len,
                          const char *key, const char *out_name) {
    const char *v = idletoken_json_obj_get(body, len, key);
    if (!v) return;
    long vl = idletoken_json_value_len(v, body + len);
    if (!span_is_number(v, vl)) return;
    sb_cstr(b, ",\"");
    sb_cstr(b, out_name);
    sb_cstr(b, "\":");
    sb_put(b, v, (size_t)vl);
}

static int top_int_field(const char *body, size_t len, const char *key, int dflt) {
    const char *v = idletoken_json_obj_get(body, len, key);
    if (!v) return dflt;
    long vl = idletoken_json_value_len(v, body + len);
    if (vl <= 0) return dflt;
    int sign = 1;
    long i = 0;
    if (v[0] == '-') { sign = -1; i = 1; }
    int val = 0, digits = 0;
    for (; i < vl && v[i] >= '0' && v[i] <= '9'; i++) { val = val * 10 + (v[i] - '0'); digits++; }
    return digits ? sign * val : dflt;
}

/* Flatten a "content" value (string, or an array of text blocks) into `tsb`
 * as escaped JSON-string text, joining blocks with a literal \n. Non-text
 * blocks are skipped by the CALLERS that handle them (tool_use/tool_result);
 * here they are simply not text. Returns the number of text pieces taken. */
static int flatten_text_content(sb_t *tsb, const char *cv, const char *end) {
    if (!cv) return 0;
    if (*cv == '"') {
        long vl = idletoken_json_value_len(cv, end);
        if (vl < 2) return 0;
        sb_put(tsb, cv + 1, (size_t)vl - 2);
        return 1;
    }
    if (*cv != '[') return 0;
    int pieces = 0;
    size_t it = 0;
    const char *el;
    long el_len;
    while (arr_next(cv, end, &it, &el, &el_len)) {
        if (*el != '{') continue;
        const char *ty, *tx;
        size_t tyl, txl;
        if (idletoken_json_obj_str(el, (size_t)el_len, "type", &ty, &tyl) != 0)
            continue;
        if (tyl != 4 || memcmp(ty, "text", 4) != 0) continue;
        if (idletoken_json_obj_str(el, (size_t)el_len, "text", &tx, &txl) != 0)
            continue;
        if (pieces) sb_cstr(tsb, "\\n");
        sb_put(tsb, tx, txl);
        pieces++;
    }
    return pieces;
}

int idletoken_body_has_tools(const char *body, size_t len) {
    if (!body || len == 0) return 0;
    const char *v = idletoken_json_obj_get(body, len, "tools");
    if (!v || *v != '[') return 0;
    const char *p = skip_ws(v + 1, body + len);
    return (p < body + len && *p != ']') ? 1 : 0;
}

/* One message whose content is a BLOCK ARRAY. Emits into `b`:
 *  - assistant: one message with joined text + "tool_calls" from tool_use
 *    blocks (input object stringified into "arguments");
 *  - user (or anything else): one {"role":"tool"} message per tool_result
 *    block, then one message with the joined text (if any).
 * Returns how many OpenAI messages were emitted. `*n_msgs` is the running
 * count (for comma placement + the demote rule). */
static int emit_block_message(sb_t *b, const char *role, size_t rl,
                              const char *cv, const char *end, int *n_msgs) {
    const int is_assistant = (rl == 9 && memcmp(role, "assistant", 9) == 0);
    int emitted = 0;

    sb_t tsb = {0};
    int text_pieces = flatten_text_content(&tsb, cv, end);

    if (is_assistant) {
        sb_t tcsb = {0};
        int n_calls = 0;
        size_t it = 0;
        const char *el;
        long el_len;
        while (arr_next(cv, end, &it, &el, &el_len)) {
            if (*el != '{') continue;
            const char *ty;
            size_t tyl;
            if (idletoken_json_obj_str(el, (size_t)el_len, "type", &ty, &tyl) != 0)
                continue;
            if (tyl != 8 || memcmp(ty, "tool_use", 8) != 0) continue;
            const char *id, *nm;
            size_t idl, nml;
            if (idletoken_json_obj_str(el, (size_t)el_len, "id", &id, &idl) != 0 ||
                idletoken_json_obj_str(el, (size_t)el_len, "name", &nm, &nml) != 0)
                continue;
            const char *inp = idletoken_json_obj_get(el, (size_t)el_len, "input");
            if (n_calls) sb_cstr(&tcsb, ",");
            sb_cstr(&tcsb, "{\"id\":\"");
            sb_put(&tcsb, id, idl);
            sb_cstr(&tcsb, "\",\"type\":\"function\",\"function\":{\"name\":\"");
            sb_put(&tcsb, nm, nml);
            sb_cstr(&tcsb, "\",\"arguments\":\"");
            if (inp && *inp == '{') {
                long il = idletoken_json_value_len(inp, end);
                if (il > 0) sb_put_json_escaped(&tcsb, inp, (size_t)il);
                else        sb_cstr(&tcsb, "{}");
            } else {
                sb_cstr(&tcsb, "{}");
            }
            sb_cstr(&tcsb, "\"}}");
            n_calls++;
        }
        if (text_pieces > 0 || n_calls > 0) {
            if (*n_msgs) sb_cstr(b, ",");
            sb_cstr(b, "{\"role\":\"assistant\",\"content\":\"");
            if (tsb.p) sb_put(b, tsb.p, tsb.len);
            sb_cstr(b, "\"");
            if (n_calls) {
                sb_cstr(b, ",\"tool_calls\":[");
                if (tcsb.p) sb_put(b, tcsb.p, tcsb.len);
                sb_cstr(b, "]");
            }
            sb_cstr(b, "}");
            (*n_msgs)++;
            emitted++;
        }
        if (tcsb.oom) b->oom = 1;
        free(tcsb.p);
    } else {
        /* tool_result blocks answer the PREVIOUS assistant turn's tool calls,
         * so their {"role":"tool"} messages must precede this turn's text. */
        size_t it = 0;
        const char *el;
        long el_len;
        while (arr_next(cv, end, &it, &el, &el_len)) {
            if (*el != '{') continue;
            const char *ty;
            size_t tyl;
            if (idletoken_json_obj_str(el, (size_t)el_len, "type", &ty, &tyl) != 0)
                continue;
            if (tyl != 11 || memcmp(ty, "tool_result", 11) != 0) continue;
            const char *tid;
            size_t tidl;
            if (idletoken_json_obj_str(el, (size_t)el_len, "tool_use_id",
                                       &tid, &tidl) != 0)
                continue;
            if (*n_msgs) sb_cstr(b, ",");
            sb_cstr(b, "{\"role\":\"tool\",\"tool_call_id\":\"");
            sb_put(b, tid, tidl);
            sb_cstr(b, "\",\"content\":\"");
            {
                sb_t rsb = {0};
                const char *rcv = idletoken_json_obj_get(el, (size_t)el_len,
                                                         "content");
                flatten_text_content(&rsb, rcv, el + el_len);
                if (rsb.p) sb_put(b, rsb.p, rsb.len);
                if (rsb.oom) b->oom = 1;
                free(rsb.p);
            }
            sb_cstr(b, "\"}");
            (*n_msgs)++;
            emitted++;
        }
        if (text_pieces > 0) {
            /* Demote a non-leading system role to user (see header). */
            const int demote = (*n_msgs > 0 && rl == 6 &&
                                memcmp(role, "system", 6) == 0);
            if (*n_msgs) sb_cstr(b, ",");
            sb_cstr(b, "{\"role\":\"");
            if (demote) sb_cstr(b, "user");
            else        sb_put(b, role, rl);
            sb_cstr(b, "\",\"content\":\"");
            if (tsb.p) sb_put(b, tsb.p, tsb.len);
            sb_cstr(b, "\"}");
            (*n_msgs)++;
            emitted++;
        }
    }
    if (tsb.oom) b->oom = 1;
    free(tsb.p);
    return emitted;
}

static void emit_tools(sb_t *b, const char *body, size_t len) {
    const char *end = body + len;
    const char *tv = idletoken_json_obj_get(body, len, "tools");
    if (!tv || *tv != '[') return;
    sb_t tb = {0};
    int n_tools = 0;
    size_t it = 0;
    const char *el;
    long el_len;
    while (arr_next(tv, end, &it, &el, &el_len)) {
        if (*el != '{') continue;
        const char *nm, *ds;
        size_t nml, dsl;
        if (idletoken_json_obj_str(el, (size_t)el_len, "name", &nm, &nml) != 0)
            continue;   /* server-side tool entries carry no name; skip */
        if (n_tools) sb_cstr(&tb, ",");
        sb_cstr(&tb, "{\"type\":\"function\",\"function\":{\"name\":\"");
        sb_put(&tb, nm, nml);
        sb_cstr(&tb, "\"");
        if (idletoken_json_obj_str(el, (size_t)el_len, "description",
                                   &ds, &dsl) == 0) {
            sb_cstr(&tb, ",\"description\":\"");
            sb_put(&tb, ds, dsl);
            sb_cstr(&tb, "\"");
        }
        sb_cstr(&tb, ",\"parameters\":");
        {
            const char *sc = idletoken_json_obj_get(el, (size_t)el_len,
                                                    "input_schema");
            long scl = sc ? idletoken_json_value_len(sc, el + el_len) : -1;
            if (sc && *sc == '{' && scl > 0) sb_put(&tb, sc, (size_t)scl);
            else sb_cstr(&tb, "{\"type\":\"object\"}");
        }
        sb_cstr(&tb, "}}");
        n_tools++;
    }
    if (n_tools > 0) {
        sb_cstr(b, ",\"tools\":[");
        if (tb.p) sb_put(b, tb.p, tb.len);
        sb_cstr(b, "]");
        /* tool_choice: auto -> auto, any -> required, none -> none,
         * tool(name) -> the named function. Anything else: leave it to the
         * engine's default rather than forwarding a shape it may reject. */
        const char *tc = idletoken_json_obj_get(body, len, "tool_choice");
        if (tc && *tc == '{') {
            long tcl = idletoken_json_value_len(tc, end);
            const char *ty, *nm;
            size_t tyl, nml;
            if (tcl > 0 &&
                idletoken_json_obj_str(tc, (size_t)tcl, "type", &ty, &tyl) == 0) {
                if (tyl == 4 && !memcmp(ty, "auto", 4))
                    sb_cstr(b, ",\"tool_choice\":\"auto\"");
                else if (tyl == 3 && !memcmp(ty, "any", 3))
                    sb_cstr(b, ",\"tool_choice\":\"required\"");
                else if (tyl == 4 && !memcmp(ty, "none", 4))
                    sb_cstr(b, ",\"tool_choice\":\"none\"");
                else if (tyl == 4 && !memcmp(ty, "tool", 4) &&
                         idletoken_json_obj_str(tc, (size_t)tcl, "name",
                                                &nm, &nml) == 0) {
                    sb_cstr(b, ",\"tool_choice\":{\"type\":\"function\","
                               "\"function\":{\"name\":\"");
                    sb_put(b, nm, nml);
                    sb_cstr(b, "\"}}");
                }
            }
        }
    }
    if (tb.oom) b->oom = 1;
    free(tb.p);
}

char *idletoken_anthropic_to_openai(const char *body, size_t len,
                                    int want_stream, int default_max_tokens,
                                    size_t *out_len) {
    if (!body || len == 0) return NULL;
    const char *end = body + len;
    sb_t b = {0};
    sb_cstr(&b, "{\"messages\":[");
    int n_msgs = 0;

    {   /* Anthropic's system prompt is a top-level field, not part of
         * messages: string, or an array of text blocks (joined). */
        const char *sv = idletoken_json_obj_get(body, len, "system");
        sb_t ssb = {0};
        if (flatten_text_content(&ssb, sv, end) > 0) {
            sb_cstr(&b, "{\"role\":\"system\",\"content\":\"");
            if (ssb.p) sb_put(&b, ssb.p, ssb.len);
            sb_cstr(&b, "\"}");
            n_msgs++;
        }
        if (ssb.oom) b.oom = 1;
        free(ssb.p);
    }

    const char *mv = idletoken_json_obj_get(body, len, "messages");
    if (mv && *mv == '[') {
        size_t it = 0;
        const char *el;
        long el_len;
        while (arr_next(mv, end, &it, &el, &el_len)) {
            if (*el != '{') continue;
            const char *role;
            size_t rl;
            if (idletoken_json_obj_str(el, (size_t)el_len, "role", &role, &rl) != 0)
                continue;
            const char *cv = idletoken_json_obj_get(el, (size_t)el_len, "content");
            if (!cv) continue;
            if (*cv == '"') {
                long cl = idletoken_json_value_len(cv, el + el_len);
                if (cl < 2) continue;
                /* Chat templates (Qwen3.5 measured) accept a system message
                 * only in position 0 and 400 on any later one — but Claude
                 * Code sends mid-conversation {"role":"system"} reminders
                 * after the user turn. Rewrite any non-leading system role
                 * to "user": consecutive user turns render fine, and the
                 * reminder text carries its own <system-reminder> tags. */
                const int demote = (n_msgs > 0 && rl == 6 &&
                                    memcmp(role, "system", 6) == 0);
                if (n_msgs) sb_cstr(&b, ",");
                sb_cstr(&b, "{\"role\":\"");
                if (demote) sb_cstr(&b, "user");
                else        sb_put(&b, role, rl);
                sb_cstr(&b, "\",\"content\":\"");
                sb_put(&b, cv + 1, (size_t)cl - 2);
                sb_cstr(&b, "\"}");
                n_msgs++;
            } else if (*cv == '[') {
                emit_block_message(&b, role, rl, cv, el + el_len, &n_msgs);
            }
        }
    }

    if (n_msgs == 0) {
        /* single-content fallback (old clients): a bare top-level "content" */
        const char *cspan;
        size_t cl;
        if (idletoken_json_obj_str(body, len, "content", &cspan, &cl) == 0 &&
            cl > 0) {
            sb_cstr(&b, "{\"role\":\"user\",\"content\":\"");
            sb_put(&b, cspan, cl);
            sb_cstr(&b, "\"}");
            n_msgs++;
        }
    }
    if (n_msgs == 0 || b.oom) { free(b.p); return NULL; }
    sb_cstr(&b, "]");

    emit_tools(&b, body, len);

    put_num_field(&b, body, len, "temperature", "temperature");
    put_num_field(&b, body, len, "top_p", "top_p");
    put_num_field(&b, body, len, "top_k", "top_k");
    {   /* stop_sequences -> OpenAI "stop" (an array of strings either way) */
        const char *sv = idletoken_json_obj_get(body, len, "stop_sequences");
        long svl = sv ? idletoken_json_value_len(sv, end) : -1;
        if (sv && *sv == '[' && svl > 0) {
            sb_cstr(&b, ",\"stop\":");
            sb_put(&b, sv, (size_t)svl);
        }
    }

    int max_tokens = top_int_field(body, len, "max_tokens", 0);
    if (max_tokens <= 0 && default_max_tokens > 0) max_tokens = default_max_tokens;
    if (max_tokens > 0) {
        char mt[48];
        snprintf(mt, sizeof(mt), ",\"max_tokens\":%d", max_tokens);
        sb_cstr(&b, mt);
    }
    if (want_stream)
        sb_cstr(&b, ",\"stream\":true,\"stream_options\":{\"include_usage\":true}");
    sb_cstr(&b, "}");
    if (b.oom) { free(b.p); return NULL; }
    if (out_len) *out_len = b.len;
    return b.p;
}

/* ---- response translation ------------------------------------------------- */

int idletoken_oai_resp_message(const char *resp, size_t len,
                               const char **out, size_t *out_len) {
    if (!resp || len == 0) return -1;
    const char *end = resp + len;
    const char *cv = idletoken_json_obj_get(resp, len, "choices");
    if (!cv || *cv != '[') return -1;
    size_t it = 0;
    const char *ch0;
    long ch0_len;
    if (!arr_next(cv, end, &it, &ch0, &ch0_len) || *ch0 != '{') return -1;
    const char *m = idletoken_json_obj_get(ch0, (size_t)ch0_len, "message");
    if (!m || *m != '{') return -1;
    long ml = idletoken_json_value_len(m, ch0 + ch0_len);
    if (ml < 0) return -1;
    *out = m;
    *out_len = (size_t)ml;
    return 0;
}

int idletoken_oai_next_tool_call(const char *msg, size_t len, size_t *iter,
                                 idletoken_tool_call *out) {
    if (!msg || len == 0) return 0;
    const char *end = msg + len;
    const char *tv = idletoken_json_obj_get(msg, len, "tool_calls");
    if (!tv || *tv != '[') return 0;
    const char *el;
    long el_len;
    while (arr_next(tv, end, iter, &el, &el_len)) {
        if (*el != '{') continue;
        const char *id;
        size_t idl;
        if (idletoken_json_obj_str(el, (size_t)el_len, "id", &id, &idl) != 0) {
            id = "";
            idl = 0;
        }
        const char *fn = idletoken_json_obj_get(el, (size_t)el_len, "function");
        if (!fn || *fn != '{') continue;
        long fnl = idletoken_json_value_len(fn, el + el_len);
        if (fnl < 0) continue;
        const char *nm, *ar;
        size_t nml, arl;
        if (idletoken_json_obj_str(fn, (size_t)fnl, "name", &nm, &nml) != 0)
            continue;
        if (idletoken_json_obj_str(fn, (size_t)fnl, "arguments", &ar, &arl) != 0) {
            ar = "";
            arl = 0;
        }
        out->id = id;     out->id_len = idl;
        out->name = nm;   out->name_len = nml;
        out->args = ar;   out->args_len = arl;
        return 1;
    }
    return 0;
}

/* Append `,"input":<object>` from an escaped "arguments" span: unescape, and
 * embed only when the result is one whole JSON object — anything else (an
 * empty string, truncated JSON, a bare scalar) becomes {} rather than a
 * syntax error in OUR response. */
static void put_tool_input(sb_t *b, const char *args, size_t args_len) {
    sb_cstr(b, ",\"input\":");
    size_t ul = 0;
    char *un = idletoken_json_unescape(args, args_len, &ul);
    int ok = 0;
    if (un) {
        const char *p = skip_ws(un, un + ul);
        if (p < un + ul && *p == '{') {
            long vl = idletoken_json_value_len(p, un + ul);
            if (vl > 0 && skip_ws(p + vl, un + ul) == un + ul) {
                sb_put(b, p, (size_t)vl);
                ok = 1;
            }
        }
    }
    if (!ok) sb_cstr(b, "{}");
    free(un);
}

char *idletoken_oai_resp_to_anthropic_content(const char *resp, size_t len,
                                              char *stop_reason, size_t sr_cap,
                                              size_t *out_len) {
    if (stop_reason && sr_cap) snprintf(stop_reason, sr_cap, "max_tokens");
    const char *msg;
    size_t mlen;
    if (idletoken_oai_resp_message(resp, len, &msg, &mlen) != 0) return NULL;

    if (stop_reason && sr_cap) {
        /* finish_reason lives on the choice, not the message */
        const char *cv = idletoken_json_obj_get(resp, len, "choices");
        size_t it = 0;
        const char *ch0;
        long ch0_len;
        if (cv && *cv == '[' &&
            arr_next(cv, resp + len, &it, &ch0, &ch0_len) && *ch0 == '{') {
            const char *fr;
            size_t frl;
            if (idletoken_json_obj_str(ch0, (size_t)ch0_len, "finish_reason",
                                       &fr, &frl) == 0) {
                if (frl == 4 && !memcmp(fr, "stop", 4))
                    snprintf(stop_reason, sr_cap, "end_turn");
                else if (frl == 10 && !memcmp(fr, "tool_calls", 10))
                    snprintf(stop_reason, sr_cap, "tool_use");
            }
        }
    }

    const char *text = "";
    size_t textl = 0;
    idletoken_json_obj_str(msg, mlen, "content", &text, &textl);

    sb_t b = {0};
    sb_cstr(&b, "[");
    int n_blocks = 0;
    size_t it = 0;
    idletoken_tool_call tc;
    int have_calls = 0;
    {   /* probe once so an all-tools reply gets no empty text block */
        size_t probe = 0;
        have_calls = idletoken_oai_next_tool_call(msg, mlen, &probe, &tc);
    }
    if (textl > 0 || !have_calls) {
        sb_cstr(&b, "{\"type\":\"text\",\"text\":\"");
        sb_put(&b, text, textl);
        sb_cstr(&b, "\"}");
        n_blocks++;
    }
    while (idletoken_oai_next_tool_call(msg, mlen, &it, &tc)) {
        if (n_blocks) sb_cstr(&b, ",");
        sb_cstr(&b, "{\"type\":\"tool_use\",\"id\":\"");
        sb_put(&b, tc.id, tc.id_len);
        sb_cstr(&b, "\",\"name\":\"");
        sb_put(&b, tc.name, tc.name_len);
        sb_cstr(&b, "\"");
        put_tool_input(&b, tc.args, tc.args_len);
        sb_cstr(&b, "}");
        n_blocks++;
    }
    sb_cstr(&b, "]");
    if (b.oom) { free(b.p); return NULL; }
    if (out_len) *out_len = b.len;
    return b.p;
}
