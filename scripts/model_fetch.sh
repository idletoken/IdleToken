#!/usr/bin/env bash
# Fetch a model's weights (acceptance G-FETCH).
#
# The target machine may have no working route to huggingface.co — in China
# that is the normal case, not the exception. A product that says "download the
# GGUF yourself" has already lost the non-expert user, so this script:
#   1. picks a reachable endpoint (explicit HF_ENDPOINT, then HF direct, then
#      the mirror) and SAYS which one it used,
#   2. resumes instead of restarting (these files are 0.5-80 GB),
#   3. verifies what it got (GGUF magic + full length), and
#   4. fails with something the user can act on: no route / not found / disk.
#
# Usage:  scripts/model_fetch.sh <model-id> [quant] [dest-dir]
#         scripts/model_fetch.sh qwen3.5-0.8b Q4_K_M ~/work/qwen
# Env:    HF_ENDPOINT   force one endpoint (skips probing)
#         IDLETOKEN_MIRRORS  space-separated fallback list (default hf-mirror.com)
#
# Contract: last line MODEL_FETCH_OK or MODEL_FETCH_FAIL: <reason>.
set -u

cd "$(dirname "$0")/.." || exit 1

MODEL="${1:-}"
QUANT="${2:-}"
DEST="${3:-.}"
[ -n "$MODEL" ] || { echo "MODEL_FETCH_FAIL: usage: model_fetch.sh <model-id> [quant] [dest-dir]"; exit 1; }

die() { echo "MODEL_FETCH_FAIL: $*"; exit 1; }

MANIFEST="models/$MODEL.json"
[ -f "$MANIFEST" ] || die "unknown model '$MODEL' (no $MANIFEST). Known: $(ls models/*.json | xargs -n1 basename | sed 's/.json//' | tr '\n' ' ')"

# --- resolve repo / filename / expected size from the manifest --------------
read -r REPO FILE EXPECT <<EOF
$(python3 - "$MANIFEST" "$QUANT" <<'PY'
import json, sys
man = json.load(open(sys.argv[1]))
want = sys.argv[2]
variants = man.get("variants") or []
v = None
if variants:
    v = next((x for x in variants if x.get("quant") == want), None) if want else None
    if v is None:
        dq = man.get("default_quant")
        v = next((x for x in variants if x.get("quant") == dq), variants[0])
repo = (v or {}).get("repo") or (man.get("sources") or [{}])[0].get("repo", "")
gguf = (v or {}).get("gguf") or man.get("default_gguf", "")
size = int((v or man).get("layer_weight_bytes", 0)) + int((v or man).get("shared_weight_bytes", 0))
print(repo, gguf, size)
PY
)
EOF
[ -n "${REPO:-}" ] && [ -n "${FILE:-}" ] || die "manifest $MANIFEST has no repo/filename for quant '${QUANT:-default}'"
echo "model:    $MODEL ${QUANT:+($QUANT)}"
echo "repo:     $REPO"
echo "file:     $FILE"

mkdir -p "$DEST" 2>/dev/null || die "cannot create destination $DEST"
OUT="$DEST/$FILE"

# --- pick an endpoint -------------------------------------------------------
# Order: caller's explicit choice, then the real thing, then mirrors. Probing
# beats guessing: the same machine may reach HF over a VPN today and not
# tomorrow, and a silent 30-minute stall is the worst possible answer.
if [ -n "${HF_ENDPOINT:-}" ]; then
    ENDPOINTS="$HF_ENDPOINT"
else
    ENDPOINTS="https://huggingface.co ${IDLETOKEN_MIRRORS:-https://hf-mirror.com}"
fi

url_for() { echo "$1/$REPO/resolve/main/$FILE"; }

PICKED=""
CLEN=0
for ep in $ENDPOINTS; do
    url=$(url_for "$ep")
    # -L: resolve/main is a redirect to a CDN. -I alone can 405 on some CDNs,
    # so ask for the first byte instead and read the total from Content-Range.
    hdr=$(curl -sS -L -m 20 -r 0-0 -o /dev/null -D - "$url" 2>/dev/null)
    code=$(printf '%s' "$hdr" | awk '/^HTTP\/[0-9.]+ [0-9]+/ {c=$2} END {print c}')
    case "$code" in
        200|206)
            total=$(printf '%s' "$hdr" | tr -d '\r' | awk -F'/' '/^[Cc]ontent-[Rr]ange:/ {print $2}' | tail -1)
            [ -n "$total" ] || total=$(printf '%s' "$hdr" | tr -d '\r' | awk '/^[Cc]ontent-[Ll]ength:/ {print $2}' | tail -1)
            PICKED="$ep"; CLEN="${total:-0}"
            break ;;
        404) die "the file does not exist on $ep ($REPO/$FILE). Check the model manifest's repo/filename." ;;
        401|403) echo "  $ep: needs authentication (gated repo) — trying the next endpoint" ;;
        "") echo "  $ep: no response (blocked or offline) — trying the next endpoint" ;;
        *)  echo "  $ep: HTTP $code — trying the next endpoint" ;;
    esac
done
[ -n "$PICKED" ] || die "no reachable endpoint for $REPO/$FILE. Tried: $ENDPOINTS. Check the network, or set HF_ENDPOINT to a mirror you can reach."
echo "endpoint: $PICKED$([ "$PICKED" = "https://huggingface.co" ] || echo "  (mirror)")"
[ "${CLEN:-0}" -gt 0 ] && echo "size:     $((CLEN / 1024 / 1024)) MiB"

# --- disk check before spending an hour on a doomed download ---------------
if [ "${CLEN:-0}" -gt 0 ]; then
    have=$(df -Pk "$DEST" 2>/dev/null | awk 'NR==2 {print $4 * 1024}')
    if [ -n "$have" ] && [ "$have" -lt "$CLEN" ]; then
        die "not enough free space in $DEST: need $((CLEN/1024/1024)) MiB, have $((have/1024/1024)) MiB. Free space or pass a different destination."
    fi
fi

# --- download (resumable) ---------------------------------------------------
if [ -f "$OUT" ] && [ "${CLEN:-0}" -gt 0 ]; then
    got=$(stat -c %s "$OUT" 2>/dev/null || stat -f %z "$OUT" 2>/dev/null || echo 0)
    if [ "$got" = "$CLEN" ]; then
        echo "already complete: $OUT"
    else
        echo "resuming at $((got / 1024 / 1024)) MiB of $((CLEN / 1024 / 1024)) MiB"
    fi
fi
# `-C -` resumes; on a complete file curl exits 33 ("range not supported" is
# how it reports "nothing left to get"), which is success for our purposes.
curl -fL -C - --retry 3 --retry-delay 2 -o "$OUT" "$(url_for "$PICKED")"
rc=$?
case "$rc" in
    0|33) ;;
    22) die "server refused the request (HTTP error) for $(url_for "$PICKED")" ;;
    28) die "download timed out against $PICKED — the endpoint is reachable but slow; rerun to resume" ;;
    *)  die "download failed (curl exit $rc). Rerun to resume from where it stopped." ;;
esac

# --- verify -----------------------------------------------------------------
got=$(stat -c %s "$OUT" 2>/dev/null || stat -f %z "$OUT" 2>/dev/null || echo 0)
[ "$got" -gt 0 ] || die "downloaded file is empty: $OUT"
if [ "${CLEN:-0}" -gt 0 ] && [ "$got" != "$CLEN" ]; then
    die "short file: got $got bytes, server advertised $CLEN. Rerun to resume."
fi
magic=$(head -c 4 "$OUT" | tr -d '\0')
[ "$magic" = "GGUF" ] || die "not a GGUF file (magic '$magic'). The endpoint may have served an error page — delete $OUT and retry."

# Manifest sizes are recorded from real files, so a large deviation means the
# manifest and the repo have drifted apart. Warn (the download is still valid)
# rather than fail — the planner uses these numbers to decide what fits.
if [ "${EXPECT:-0}" -gt 0 ]; then
    lo=$((EXPECT * 3 / 4)); hi=$((EXPECT * 5 / 4))
    if [ "$got" -lt "$lo" ] || [ "$got" -gt "$hi" ]; then
        echo "  WARN: file is $((got/1024/1024)) MiB but the manifest expects ~$((EXPECT/1024/1024)) MiB —"
        echo "        models/$MODEL.json may be out of date (planning numbers come from it)."
    fi
fi

echo "saved:    $OUT ($((got / 1024 / 1024)) MiB, GGUF verified)"
echo MODEL_FETCH_OK
