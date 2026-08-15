#!/usr/bin/env python3
"""IdleToken's single measuring stick. Every performance number that reaches a
commit message or the docs must come from this file.

Why it exists: this repo has retracted its own conclusions four times over **how
the measurement was defined** --
  - "readback is 86% of the time" (GPU execution had been charged to readback,
    retracted in 2581e08)
  - "the recurrence is 45.7%" (a timing error, corrected in ab52c2d)
  - "multi-slot concurrency is 1.41x" (did not reproduce, recorded as a negative
    result in 4c7ecd9)
  - "0.489 s/token is the decode rate" (it was prefill amortized in, corrected in
    581454b)
Not once was the engine wrong; the **ruler** was. The ad hoc timing code
scattered through the check scripts shared no rules, so the same mistake was made
again the moment it moved to another script. Hence one set of rules here, written
as **assertions** wherever possible rather than comments -- discipline written in
a comment cannot fail, discipline written as an assertion can.

The methodology and the origin of each rule are in docs/bench-methodology.md.

Usage (must run **on a cluster node itself**, see "why not from the control
machine" below):
    python3 bench.py --pp 512 --tg 64 --repeat 3 --warmup 1
    python3 bench.py --pp 512 --tg 64 --concurrency 2 --compare serial
    python3 bench.py --pp 128,512,2048 --tg 64 --json /tmp/bench.json

Why requests must not go from the control machine (a development Mac) straight to
the coordinator: the control machine's IP changes, it is not necessarily on the
cluster's subnet, and Windows blocks ICMP by default so ping proves nothing. One
entire benchmark round once read "11.04s", which was curl's connection timeout
rather than inference time -- it looked far too much like a plausible performance
number and fooled six rounds of debugging. The same reasoning appears in
testbed_api_post in scripts/testbed-lib.sh.
"""

import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time
import uuid
from urllib.parse import urlparse

# How much variance within one run makes a conclusion unsafe. Empirically,
# cross-machine PP fluctuates within 5%; past 15% the environment is not clean
# (background processes, thermal throttling, network jitter), and reporting a
# median then is lying to yourself.
SPREAD_WARN = 0.15

# Tolerance, in tokens, when calibrating the prompt length. The tokenizer is not
# ours and an exact hit is not achievable; but **what gets reported is always the
# measured prompt_tokens the server returns**, and --pp is only the target.
PP_TOLERANCE = 8


# ---------------------------------------------------------------------------
# HTTP: http.client rather than urllib, because the concurrency case must be able
# to separate "establish the connection" from "send the request".
# ---------------------------------------------------------------------------

class Client:
    """One connection to the coordinator. Separating connect() from send() is what
    makes concurrent measurement meaningful."""

    def __init__(self, base_url, api_token=None):
        u = urlparse(base_url)
        if u.scheme not in ("http", ""):
            raise SystemExit(f"only http is supported (the ruler runs on local loopback): {base_url}")
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or 80
        self.api_token = api_token
        self.conn = None

    def connect(self):
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=900)
        self.conn.connect()          # actually establish the TCP connection, no lazy connect

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            finally:
                self.conn = None

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_token:
            h["Authorization"] = "Bearer " + self.api_token
        return h

    def get_json(self, path):
        c = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            c.request("GET", path, headers=self._headers())
            r = c.getresponse()
            body = r.read()
            if r.status != 200:
                raise SystemExit(f"GET {path} returned HTTP {r.status}: {body[:200]!r}")
            return json.loads(body)
        finally:
            c.close()

    def post_json(self, path, payload):
        """A non-streaming request. Used for calibration only, never for timing."""
        body = json.dumps(payload).encode()
        self.conn.request("POST", path, body=body, headers=self._headers())
        r = self.conn.getresponse()
        raw = r.read()
        if r.status != 200:
            raise SystemExit(f"POST {path} returned HTTP {r.status}: {raw[:300]!r}\n"
                             f"  That duration is **not** inference time; do not use it as a performance number.")
        return json.loads(raw)


def stream_request(client, payload, barrier=None):
    """Issue one streaming request and return the arrival timestamps of each token.

    This is the heart of the ruler. Three definitions are fixed here:
      1. t_send is taken **at the moment the request goes out**, not when the
         thread starts -- thread startup jitters by tens of milliseconds, and
         charging that to TTFT is tens of milliseconds of fictitious difference.
      2. t_first is when the **first delta frame** arrives. The coordinator emits
         one frame per token produced (sse_delta in coord_main.c), so that is
         where time to first token ends.
      3. The token count always comes from the server's usage.completion_tokens in
         the trailer, never from counting frames -- a UTF-8 sequence continued
         across frames makes frame count and token count disagree (see the carry
         logic in coord_main.c).
    """
    body = json.dumps(dict(payload, stream=True)).encode()
    headers = client._headers()
    headers["Accept"] = "text/event-stream"

    if barrier is not None:
        barrier.wait()               # connections are up; only now is the starting line aligned

    t_send = time.monotonic()
    client.conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
    resp = client.conn.getresponse()
    if resp.status != 200:
        raw = resp.read()
        raise SystemExit(f"streaming request returned HTTP {resp.status}: {raw[:300]!r}\n"
                         f"  That duration is **not** inference time; do not use it as a performance number.")

    t_first = None
    t_last = None
    usage = None
    finish_reason = None
    text = []

    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        now = time.monotonic()
        try:
            frame = json.loads(data)
        except json.JSONDecodeError:
            continue
        if frame.get("usage"):
            usage = frame["usage"]
        ch = (frame.get("choices") or [{}])[0]
        if ch.get("finish_reason"):
            finish_reason = ch["finish_reason"]
        delta = (ch.get("delta") or {}).get("content")
        if delta is not None:
            # An empty string still counts as a frame: the coordinator emits an
            # empty delta while a UTF-8 sequence is incomplete, and the token had
            # genuinely been produced at that moment.
            if t_first is None:
                t_first = now
            t_last = now
            text.append(delta)

    t_end = time.monotonic()
    # Drain the rest of the response (the chunked terminator). Timing is already
    # captured, so this is pure cleanup: breaking out at [DONE] leaves the
    # connection in the "Request-sent" state, and the next request on that same
    # connection raises ResponseNotReady straight away. The real coordinator
    # happens to tolerate it while a stub server blows up immediately -- which is
    # to say it is a latent bug that only fires on some server implementations.
    try:
        resp.read()
    except Exception:                        # noqa: BLE001 -- cleanup must not affect this measurement
        pass

    if usage is None:
        raise SystemExit("no usage trailer in the stream -- wrong server version, or the request was interrupted."
                         "\n  Without an authoritative token count there is no throughput; this round is void.")
    if t_first is None:
        raise SystemExit("not a single delta arrived, yet the stream ended normally. That is an engine bug, not a performance number.")

    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "finish_reason": finish_reason,
        "ttft_s": t_first - t_send,
        "gen_span_s": t_last - t_first,
        "wall_s": t_end - t_send,
        "t_send": t_send,
        "t_first": t_first,
        "t_last": t_last,
        "t_end": t_end,
        "text": "".join(text),
    }


def derive(sample):
    """Turn raw timestamps into two throughputs. These two formulas are the easiest
    thing in the whole methodology to get wrong."""
    n_out = sample["completion_tokens"]
    n_in = sample["prompt_tokens"]

    # Prefill throughput: prompt tokens divided by time to first token.
    sample["prefill_tps"] = n_in / sample["ttft_s"] if sample["ttft_s"] > 0 else 0.0

    # Generation throughput: **minus one**. The first token's cost is prefill, not
    # decode; counting it in the numerator amortizes prefill into the decode rate,
    # which is exactly how "0.489 s/token turned out to be prefill amortized in"
    # came about. The denominator likewise spans only first token to last token.
    span = sample["gen_span_s"]
    sample["gen_tps"] = (n_out - 1) / span if (n_out > 1 and span > 0) else 0.0
    sample["ms_per_token"] = (span * 1000.0 / (n_out - 1)) if (n_out > 1 and span > 0) else 0.0
    return sample


# ---------------------------------------------------------------------------
# Prompt-length calibration: use the server's own tokenizer, never guess
# ---------------------------------------------------------------------------

FILLER = "The quick brown fox jumps over the lazy dog. "   # coarse unit, roughly 10 tokens
PAD = "word "                                              # fine unit, 1 token


def make_prompt(n_filler, n_pad, nonce, filler=FILLER):
    """The nonce goes **at the front**: KV prefix reuse compares prefixes, and only
    a leading nonce guarantees the prefixes differ. At the end, a common prefix of
    several thousand tokens still hits the cache and "cold cache" becomes a
    fiction."""
    return f"[{nonce}] " + filler * n_filler + PAD * n_pad


def calibrate(client, model, target_pp, nonce_fn, verbose=True, filler=FILLER):
    """Calibrate (n_filler, n_pad) so that prompt_tokens lands as close to
    target_pp as possible.

    Why not "roughly 8K": the tier boundaries (8K/32K/128K) are acceptance
    thresholds at specific numbers, and being a few hundred tokens off can cross a
    chunk boundary -- at which point two runs are not measuring the same thing.

    Two stages: a binary search over ~10-token sentences first (fewer requests),
    then a binary search over the pad count.

    `nonce_fn` must be **the same one used for the real measurement**. The first
    version hardcoded "[calib]" during calibration and used 12 random hex digits
    when measuring -- different token counts, so however precise the calibration
    was, the prompt actually sent was 5 tokens longer (the 128 case measured 133).
    Calibration and the thing being measured have to be identical.
    """
    def probe(n_filler, n_pad=0):
        r = client.post_json("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user",
                          "content": make_prompt(n_filler, n_pad, nonce_fn(), filler)}],
            "max_tokens": 1,
        })
        return r["usage"]["prompt_tokens"]

    # Coarse stage: find the largest filler count **not exceeding** the target and
    # leave the remainder to the fine stage.
    lo, hi = 0, 1
    while probe(hi) < target_pp:
        lo, hi = hi, hi * 2
        if hi > 1 << 20:
            raise SystemExit(f"calibration failed: filler doubled to {hi} without reaching {target_pp} tokens")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if probe(mid) <= target_pp:
            lo = mid
        else:
            hi = mid - 1
    n_filler = lo

    # Fine stage: **binary search again**, over the pad count, without assuming how
    # many tokens one pad is worth. The first version assumed PAD was exactly 1
    # token and filled the gap in a single step -- on DSv4's tokenizer it is
    # actually 2, so the 128 case calibrated to 134, overshooting by a factor of
    # two. The tokenizer is not ours, so every "it should be N" assumption has to
    # become "measure what it is".
    base = probe(n_filler)
    if base == target_pp:
        return n_filler, 0
    lo, hi = 0, max(1, (target_pp - base)) * 4 + 8      # generous upper bound; the per-pad cost is unknown
    while probe(n_filler, hi) < target_pp:
        hi *= 2
        if hi > 1 << 20:
            break
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if probe(n_filler, mid) <= target_pp:
            lo = mid
        else:
            hi = mid - 1
    # lo is the largest pad not exceeding the target; one step past it may land
    # closer, so compare both.
    cands = [(abs(probe(n_filler, p) - target_pp), p) for p in (lo, lo + 1)]
    err, pad = min(cands)

    if err > PP_TOLERANCE and verbose:
        print(f"   ! prompt length calibrated only to +/-{err} tokens (target {target_pp})", file=sys.stderr)
    return n_filler, pad


# ---------------------------------------------------------------------------
# Environment gate: prove, before and after, that this cluster was serving nobody
# but us for the duration
# ---------------------------------------------------------------------------

def snapshot(client):
    return client.get_json("/idletoken/v1/stats")


def check_env(before, after, expected_requests, cold_cache, strict):
    """Compare the before and after snapshots. Two things:
      1. The delta in requests must equal exactly what we sent -- anything extra
         means **another process is using this cluster** (a load test left over
         from the previous round, a client polling, the platform probing for
         liveness). Those requests contend for the same GPU, and the resulting
         numbers are garbage. This used to rely on clearing the field by hand
         before measuring, which people forget.
      2. In cold-cache mode the delta in cache_hits must be 0. KV prefix reuse is
         on by default now (kv-cache-design.md), and without isolation you are
         measuring cache hit rate rather than compute.
    """
    problems = []
    d_req = after.get("requests", 0) - before.get("requests", 0)
    if d_req != expected_requests:
        problems.append(
            f"the cluster served {d_req} requests during this run while we sent only {expected_requests}"
            f" -- something else is using this cluster, so this round's numbers are not trustworthy")
    d_hit = after.get("cache_hits", 0) - before.get("cache_hits", 0)
    if cold_cache:
        if d_hit != 0:
            problems.append(
                f"{d_hit} KV prefix hits occurred in cold-cache mode"
                f" -- this measures the cache rather than compute (is the nonce not working?)")
    else:
        # --warm-cache is the mirror image: zero hits means this case is **not
        # exercising the cache path at all**, and the numbers it reports would be
        # read as "performance with a working cache" -- worse than not reporting.
        # This assertion has a history: the first --warm-cache simply resent the
        # same prompt, while kv_slot_extends in coord_main.c requires
        # n_prompt > h->len -- and the same prompt is shorter than "last round's
        # prompt plus the tokens it generated", so it **could never hit**. That
        # switch quietly measured a full round of cold cache while calling itself
        # warm.
        if d_hit == 0:
            problems.append(
                "--warm-cache produced zero KV prefix hits -- the cache path was never exercised. "
                "KV reuse is a **multi-turn continuation** cache (the new prompt must strictly extend "
                "the previous turn's history), not a same-prompt replay cache")
    if problems:
        # The gate is evaluated after the case has run, so the numbers are already
        # on screen. It must say explicitly that they are void -- otherwise the
        # likeliest outcome is someone copying that tok/s line out of the scrollback
        # into a commit message long after the failure has scrolled away.
        head = ("environment gate failed -- every number printed above for this case is VOID:"
                if strict else "! environment gate warning (--no-strict allowed it through; numbers are indicative only):")
        msg = head + "\n" + "\n".join("  - " + p for p in problems)
        if strict:
            raise SystemExit(msg + "\n  Clear the field and re-run; pass --no-strict if you really mean to keep these numbers.")
        print(msg, file=sys.stderr)


# The coordinator's ttft_ms is an EWMA with a half-life of 8 requests
# (coord_main.c: *0.875 + tt*0.125). With fewer measurements than this it still
# carries memory of the previous round, and cross-checking against it produces
# only false alarms -- and a check that cries wolf ends up ignored by everyone.
TTFT_XCHECK_MIN_SAMPLES = 8


def cross_check_ttft(stats_after, measured_ttft_s, n_samples):
    """Cross-check our client-side TTFT against the one the server records itself.

    The two are defined differently -- the server's starts at exec_start and
    excludes queueing and the HTTP round trip, so the client's **should be
    slightly larger**. If the client's is instead clearly smaller, we have taken
    some earlier moment for the first token; if it is far larger, the time went
    into queueing or the network rather than inference.
    """
    srv = stats_after.get("avg_ttft_ms", 0) / 1000.0
    if srv <= 0:
        return None
    if n_samples < TTFT_XCHECK_MIN_SAMPLES:
        return None          # the EWMA has not converged; skip the cross-check silently
    ratio = measured_ttft_s / srv
    if ratio < 0.9:
        print(f"    ! client TTFT {measured_ttft_s*1000:.0f}ms < server {srv*1000:.0f}ms"
              f" -- it cannot beat the server; the first-token detection is wrong", file=sys.stderr)
    elif ratio > 1.5:
        print(f"    ! client TTFT {measured_ttft_s*1000:.0f}ms is {ratio:.1f}x the server's"
              f" {srv*1000:.0f}ms -- the time went into queueing or the network, not inference", file=sys.stderr)
    return srv


# ---------------------------------------------------------------------------
# Run one case
# ---------------------------------------------------------------------------

CONTINUE_TURN = "Please continue."


def run_batch(base_url, api_token, model, prompt_maker, tg, concurrency,
              barrier_wait=True, warm_cache=False):
    """Send `concurrency` requests concurrently and return a sample for each.

    The crux of concurrency is here: **establish every connection first, then
    align the starting line with a barrier**. Without the barrier, jitter from
    thread creation and the TCP handshake (tens of milliseconds) staggers the
    arrivals, and by the time the second request lands the first is already
    decoding -- which measures not concurrency but "serial execution with a bit of
    overlap". That is the shape of "multi-slot concurrency 1.41x did not
    reproduce".

    With `warm_cache=True`, each thread first issues a **priming turn** (not
    timed) and then measures "the priming turn plus its reply plus a new user
    message" -- so the new prompt strictly extends the slot's history and
    satisfies kv_slot_extends in coord_main.c. Priming finishes **before** the
    barrier, so the starting line stays aligned.
    """
    clients = [Client(base_url, api_token) for _ in range(concurrency)]
    for c in clients:
        c.connect()
    barrier = threading.Barrier(concurrency) if (barrier_wait and concurrency > 1) else None

    results = [None] * concurrency
    errors = [None] * concurrency
    n_primed = concurrency if warm_cache else 0

    def work(i):
        try:
            base = prompt_maker(i)
            msgs = [{"role": "user", "content": base}]
            if warm_cache:
                # Priming turn: leave the prompt and reply in the slot as the
                # history the next turn will extend.
                prime = stream_request(clients[i], {
                    "model": model, "messages": msgs, "max_tokens": tg})
                msgs = msgs + [{"role": "assistant", "content": prime["text"]},
                               {"role": "user", "content": CONTINUE_TURN}]
                # Reconnect on a clean connection before the barrier: the measured
                # request must not inherit any connection state from the priming
                # turn, and the reconnect has to happen **before** the barrier or
                # the handshake time leaks into the starting line.
                clients[i].close()
                clients[i].connect()
            payload = {"model": model, "messages": msgs, "max_tokens": tg}
            results[i] = derive(stream_request(clients[i], payload, barrier))
        except BaseException as e:          # noqa: BLE001 -- the barrier raises BrokenBarrierError
            errors[i] = e
            if barrier is not None:
                barrier.abort()             # do not leave other threads waiting forever on a dead peer

    threads = [threading.Thread(target=work, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for c in clients:
        c.close()

    for e in errors:
        if e is not None:
            raise e
    # Priming turns also occupy the cluster and are counted in /idletoken/v1/stats requests
    # -- the environment gate needs to know about them.
    return results, concurrency + n_primed


def summarize(samples, tg):
    """Collapse a set of repeated measurements into one row. Median, not mean --
    a single thermal throttle skews a mean."""
    gen = [s["gen_tps"] for s in samples]
    ttft = [s["ttft_s"] for s in samples]
    pre = [s["prefill_tps"] for s in samples]
    out = [s["completion_tokens"] for s in samples]

    row = {
        "n": len(samples),
        "prompt_tokens": samples[0]["prompt_tokens"],
        "completion_tokens_median": statistics.median(out),
        "gen_tps_median": statistics.median(gen),
        "gen_tps_min": min(gen),
        "gen_tps_max": max(gen),
        "ms_per_token_median": statistics.median([s["ms_per_token"] for s in samples]),
        "ttft_s_median": statistics.median(ttft),
        "prefill_tps_median": statistics.median(pre),
    }

    warns = []
    med = row["gen_tps_median"]
    if med > 0 and len(gen) > 1:
        spread = (max(gen) - min(gen)) / med
        row["spread"] = spread
        if spread > SPREAD_WARN:
            warns.append(f"generation throughput spread {spread*100:.0f}% > {SPREAD_WARN*100:.0f}%"
                         f" ({min(gen):.2f}-{max(gen):.2f} tok/s) -- the environment is not clean; do not draw conclusions from this")

    # Differing output lengths mean two runs did not measure the same thing. The
    # engine has no ignore_eos yet, so all we can do here is **detect it and say
    # so**, not enforce it. (Adding ignore_eos is a small engine-side change.)
    if len(set(out)) > 1:
        warns.append(f"output lengths differ across runs {sorted(set(out))} -- throughputs at different lengths are not directly comparable")
    short = [s for s in samples if s["finish_reason"] == "stop"]
    if short:
        warns.append(f"{len(short)}/{len(samples)} runs hit EOS early (generated fewer than max_tokens={tg})"
                     f" -- the denominator is smaller than you think; use a longer --tg or another prompt")
    row["warnings"] = warns
    return row


def print_row(label, row):
    print(f"  {label}")
    print(f"    prompt {row['prompt_tokens']} tok → prefill {row['prefill_tps_median']:.1f} tok/s"
          f"   TTFT {row['ttft_s_median']*1000:.0f} ms")
    print(f"    decode {row['gen_tps_median']:.2f} tok/s"
          f"   ({row['ms_per_token_median']:.1f} ms/token)"
          f"   [{row['gen_tps_min']:.2f}~{row['gen_tps_max']:.2f}, n={row['n']}]")
    for w in row["warnings"]:
        print(f"    ! {w}")


# ---------------------------------------------------------------------------

def parse_list(s, name):
    try:
        return [int(x) for x in str(s).split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--{name} accepts only comma-separated integers: {s}")


def main():
    ap = argparse.ArgumentParser(
        description="IdleToken performance ruler (methodology: docs/bench-methodology.md)")
    ap.add_argument("--base-url", default=os.environ.get("IDLETOKEN_BENCH_URL",
                                                         "http://127.0.0.1:8000"),
                    help="must be local loopback; measuring from a control machine measures the network, not inference")
    ap.add_argument("--model", default=os.environ.get("IDLETOKEN_BENCH_MODEL", "qwen3-8b"))
    ap.add_argument("--api-token", default=os.environ.get("IDLETOKEN_API_TOKEN"))
    ap.add_argument("--pp", default="512", help="prompt tokens; comma-separated for several cases")
    ap.add_argument("--tg", default="64", help="generated tokens; comma-separated, paired with --pp (broadcast when length 1)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="warm-up requests discarded per case (default 1; the first request carries one-off costs)")
    ap.add_argument("--repeat", type=int, default=3, help="measurements per case (default 3, median reported)")
    ap.add_argument("--concurrency", type=int, default=1, help="requests in flight at once")
    ap.add_argument("--compare", choices=["serial"], default=None,
                    help="also run the concurrent case serially for comparison and report the speedup")
    ap.add_argument("--warm-cache", action="store_true",
                    help="exercise the KV cache-hit path: each thread issues a priming turn first and the "
                         "measured request is its continuation (--pp then sets only the first turn's length); "
                         "cold cache by default")
    ap.add_argument("--prompt-file",
                    help="repeat this UTF-8 corpus instead of the default English filler; used by versioned pricing workloads")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="warn instead of exiting when the environment gate fails")
    ap.add_argument("--json", dest="json_out", help="write the full result (environment snapshots included) to a file")
    args = ap.parse_args()

    pps = parse_list(args.pp, "pp")
    tgs = parse_list(args.tg, "tg")
    if len(tgs) == 1:
        tgs = tgs * len(pps)
    if len(pps) == 1 and len(tgs) > 1:
        pps = pps * len(tgs)
    if len(pps) != len(tgs):
        raise SystemExit(f"--pp has {len(pps)} cases and --tg has {len(tgs)}; they cannot be paired")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    filler = FILLER
    if args.prompt_file:
        try:
            with open(args.prompt_file, encoding="utf-8") as f:
                filler = f.read()
        except OSError as exc:
            raise SystemExit(f"cannot read --prompt-file {args.prompt_file}: {exc}")
        if not filler.strip():
            raise SystemExit(f"--prompt-file is empty: {args.prompt_file}")
        if not filler.endswith("\n"):
            filler += "\n"

    probe = Client(args.base_url, args.api_token)
    health = probe.get_json("/health")
    if health.get("status") != "ok":
        raise SystemExit(f"cluster is not ready: {health}")
    env_before = snapshot(probe)

    print(f"== IdleToken bench  model={args.model}  url={args.base_url}")
    print(f"   cluster: ctx_size={env_before.get('ctx_size')} "
          f"seq_slots={env_before.get('seq_slots')} "
          f"concurrency={env_before.get('concurrency')}")
    print(f"   setup: cold_cache={'no (--warm-cache)' if args.warm_cache else 'yes'} "
          f"warmup={args.warmup} repeat={args.repeat} concurrency={args.concurrency}")

    calib_client = Client(args.base_url, args.api_token)
    calib_client.connect()
    sent = 0
    report = {"config": vars(args), "env_before": env_before, "rows": []}

    for pp, tg in zip(pps, tgs):
        print(f"\n-- pp={pp} tg={tg}")
        # A cold cache needs a distinct nonce per request, a warm one needs a fixed
        # prefix. Calibration and measurement share the same generator.
        nonce_fn = (lambda: "warm") if args.warm_cache else (lambda: uuid.uuid4().hex[:12])
        n_filler, n_pad = calibrate(calib_client, args.model, pp, nonce_fn, filler=filler)

        # Calibration issued a number of max_tokens=1 requests, and the environment
        # gate has to account for them. Rather than guess the count, take the
        # snapshot at the end of calibration as the new baseline.
        env_before_run = snapshot(probe)

        def prompt_maker(i, _f=n_filler, _p=n_pad, _nf=nonce_fn, _fill=filler):
            return make_prompt(_f, _p, _nf(), _fill)

        def batch(conc):
            return run_batch(args.base_url, args.api_token, args.model, prompt_maker,
                             tg, conc, warm_cache=args.warm_cache)

        run_sent = 0
        for _ in range(args.warmup):
            _, n = batch(args.concurrency)
            run_sent += n

        # Keep the results **per batch**, do not flatten them. A concurrency window
        # is only meaningful within one batch: flattening `repeat` concurrent runs
        # into one list and taking min(t_first)/max(t_last) measures "from the start
        # of the first batch to the end of the last", gaps between batches included
        # -- which is how the first version scored a stub server that should have
        # shown 2x at 0.99x.
        batches = []
        for _ in range(args.repeat):
            b, n = batch(args.concurrency)
            batches.append(b)
            run_sent += n
        conc_samples = [s for b in batches for s in b]

        row = summarize(conc_samples, tg)
        row.update({
            "pp_target": pp,
            "tg": tg,
            "concurrency": args.concurrency,
            "filler_repeats": n_filler,
            "pad_repeats": n_pad,
        })
        # Compare the target against the prompt length **actually sent**, not the
        # value calibration returned. Any inconsistency between calibration and
        # measurement (nonce shape, message template, chat template) surfaces only
        # here -- which is exactly what caused the 133 vs 128 case.
        # In warm-cache mode the measured prompt is "first turn + reply + follow-up"
        # and is legitimately longer than --pp, so the mismatch is expected.
        pp_err = abs(row["prompt_tokens"] - pp)
        if pp_err > PP_TOLERANCE and not args.warm_cache:
            row["warnings"].append(
                f"measured prompt is {row['prompt_tokens']} tok, off the {pp} target by {pp_err}"
                f" -- calibration and measurement are not sending the same thing")
        label = f"{args.concurrency}-way concurrent (per stream)" if args.concurrency > 1 else "single stream"
        print_row(label, row)

        # Wall clock of one concurrent batch: from the starting line (earliest send)
        # to the finish line (latest completion).
        conc_walls = [max(s["t_end"] for s in b) - min(s["t_send"] for s in b)
                      for b in batches]

        if args.concurrency > 1:
            # Aggregate cluster throughput: total output of every stream in a batch
            # divided by that batch's decode window. Not "per-stream tps x streams"
            # -- that assumes the streams overlap completely, whereas in a PP
            # cluster they usually overlap only partly, so the product is
            # systematically too high.
            aggs = []
            for b in batches:
                w = max(s["t_last"] for s in b) - min(s["t_first"] for s in b)
                if w > 0:
                    aggs.append(sum(s["completion_tokens"] - 1 for s in b) / w)
            row["agg_gen_tps"] = statistics.median(aggs) if aggs else 0.0
            print(f"    cluster total {row['agg_gen_tps']:.2f} tok/s (aggregate output within one batch)")

        if args.compare == "serial" and args.concurrency > 1:
            # Serial control: the same `concurrency` requests, one after another,
            # repeated `repeat` times.
            ser_groups = []
            for _ in range(args.repeat):
                g = []
                for _i in range(args.concurrency):
                    s, n = batch(1)
                    g.extend(s)
                    run_sent += n
                ser_groups.append(g)
            ser = [s for g in ser_groups for s in g]
            srow = summarize(ser, tg)
            print_row("serial control (per stream)", srow)

            # The speedup compares **wall clock to complete the same set of
            # requests**, not tps -- the two tps figures have different denominators,
            # and dividing them would report "each stream is slower but total output
            # is higher" as a slowdown. Take each median, then divide.
            ser_wall = statistics.median([sum(s["wall_s"] for s in g) for g in ser_groups])
            con_wall = statistics.median(conc_walls)
            row["serial"] = srow
            row["serial_wall_s"] = ser_wall
            row["concurrent_wall_s"] = con_wall
            row["speedup_vs_serial"] = ser_wall / con_wall if con_wall > 0 else 0.0
            print(f"    concurrency speedup {row['speedup_vs_serial']:.2f}x"
                  f" (serial {ser_wall:.2f}s vs concurrent {con_wall:.2f}s, the same {args.concurrency} requests)")

        env_after_run = snapshot(probe)
        row["server_ttft_s"] = cross_check_ttft(env_after_run, row["ttft_s_median"],
                                                len(conc_samples))
        check_env(env_before_run, env_after_run, run_sent,
                  cold_cache=not args.warm_cache, strict=args.strict)
        sent += run_sent
        report["rows"].append(row)

    calib_client.close()
    report["env_after"] = snapshot(probe)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nfull results -> {args.json_out}")

    print("\nBENCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
