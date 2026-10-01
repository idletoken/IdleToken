#!/usr/bin/env bash
# An explicit positive MTP model is required; a non-MTP smoke fixture cannot
# turn missing MTP coverage into a pass. Reuse the owned-process gate launcher.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/testbed-lib.sh"
if [ -z "${IDLETOKEN_MTP_GATE_GGUF:-}" ]; then
    echo 'G_MTP_SKIP: set IDLETOKEN_MTP_GATE_GGUF to a complete MTP-capable GGUF'
    exit 0
fi
export IDLETOKEN_SPEC_GATE_GGUF="$IDLETOKEN_MTP_GATE_GGUF"
export IDLETOKEN_SPEC_TYPE=ngram-mod,draft-mtp
export IDLETOKEN_SPEC_GATE_NAME=G_MTP
exec bash "$ROOT/scripts/spec_faithful_gate.sh"
