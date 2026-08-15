/* IdleToken Cluster — API-face conversion: Anthropic /v1/messages bodies to
 * and from the OpenAI chat shape the llama-server sidecar speaks. Pure
 * functions, no sockets, no globals — extracted from the coordinator relay so
 * `make apitest` can drive them directly.
 *
 * JSON convention (same as the relay in coord_main.c): string values move
 * around as RAW STILL-ESCAPED spans. The bytes inside a JSON string literal
 * are valid in any other JSON string literal, so nothing here unescapes and
 * re-escapes on the way through — except where the PROTOCOLS disagree about
 * nesting (a tool_use "input" object vs. a stringified "arguments" field),
 * which is exactly what the escape/unescape helpers below exist for. */
#ifndef IDLETOKEN_APICONV_H
#define IDLETOKEN_APICONV_H

#include <stddef.h>

/* ---- JSON object scanning (depth-aware, escape-aware) -------------------- */

/* Value position of `key` at the TOP level of the object `json` starts with
 * (leading whitespace tolerated), or NULL. Unlike a first-occurrence scan
 * this cannot be fooled by the same key nested inside another value — which
 * matters here, because tool schemas embed arbitrary user-chosen keys. */
const char *idletoken_json_obj_get(const char *json, size_t len, const char *key);

/* Byte length of one whole JSON value starting at `v` (string, object, array,
 * number, or literal), or -1 when unterminated. */
long idletoken_json_value_len(const char *v, const char *end);

/* Raw span (still escaped, quotes excluded) of `key` at the top level of
 * `json` when its value is a string. Returns 0, or -1 when absent or not a
 * string. */
int idletoken_json_obj_str(const char *json, size_t len, const char *key,
                           const char **out, size_t *out_len);

/* ---- request: Anthropic /v1/messages -> OpenAI chat body ----------------- */

/* Translate an Anthropic /v1/messages body into an OpenAI chat body for the
 * sidecar:
 *   - top-level "system" (string or text blocks) becomes the leading system
 *     message; any later {"role":"system"} is demoted to "user" (chat
 *     templates 400 on non-leading system — measured with Qwen3.5);
 *   - content-block arrays are translated, not truncated: text blocks are
 *     joined, assistant tool_use blocks become OpenAI "tool_calls" (with the
 *     input object stringified into "arguments"), user tool_result blocks
 *     become {"role":"tool"} messages carrying tool_call_id;
 *   - top-level "tools" (name/description/input_schema) becomes OpenAI
 *     "tools" (type:function, parameters = input_schema), and "tool_choice"
 *     is mapped best-effort (auto/any/none/tool);
 *   - temperature, top_p, top_k and stop_sequences (-> "stop") pass through;
 *   - max_tokens is carried over, defaulting to `default_max_tokens` (pass 0
 *     for no default), and streaming requests ask the engine to include
 *     usage in the final SSE chunk.
 * Returns a malloc'd body (caller frees) or NULL when no usable message was
 * found. */
char *idletoken_anthropic_to_openai(const char *body, size_t len,
                                    int want_stream, int default_max_tokens,
                                    size_t *out_len);

/* Does the body carry a non-empty top-level "tools" array? (The streaming
 * relay uses this to pick the tool-aware path.) */
int idletoken_body_has_tools(const char *body, size_t len);

/* ---- response: OpenAI chat.completion -> Anthropic message --------------- */

/* Span of choices[0].message inside an OpenAI chat.completion body. 0 / -1. */
int idletoken_oai_resp_message(const char *resp, size_t len,
                               const char **out, size_t *out_len);

/* One entry of a message's tool_calls[], as raw escaped spans. */
typedef struct {
    const char *id;   size_t id_len;
    const char *name; size_t name_len;
    const char *args; size_t args_len;   /* escaped text of the "arguments"
                                          * JSON-string (itself JSON) */
} idletoken_tool_call;

/* Iterate tool_calls[] of a message span (idletoken_oai_resp_message). Start
 * with *iter = 0; returns 1 and fills `out` per call, 0 when exhausted. */
int idletoken_oai_next_tool_call(const char *msg, size_t len, size_t *iter,
                                 idletoken_tool_call *out);

/* Build the Anthropic "content" array (JSON text, brackets included) from an
 * OpenAI chat.completion response: the assistant text as a text block, plus
 * one tool_use block per tool call (with "input" as the parsed object), and
 * map finish_reason into `stop_reason` ("end_turn" / "max_tokens" /
 * "tool_use"). Returns malloc'd text or NULL (OOM / no message). */
char *idletoken_oai_resp_to_anthropic_content(const char *resp, size_t len,
                                              char *stop_reason, size_t sr_cap,
                                              size_t *out_len);

/* Unescape a JSON string span into malloc'd NUL-terminated bytes (\uXXXX
 * emitted as UTF-8; unpaired surrogates dropped). NULL on OOM. */
char *idletoken_json_unescape(const char *span, size_t len, size_t *out_len);

#endif /* IDLETOKEN_APICONV_H */
