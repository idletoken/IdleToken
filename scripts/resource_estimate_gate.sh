#!/usr/bin/env bash
# Cross-check every client resource number against the compiled native planner.
# The matrix covers every curated model/precision at exact 256K and 1M (when
# declared by that model), across 1/2/4 nodes. Any one-byte drift fails closed.
set -eu
cd "$(dirname "$0")/.."

command -v cc >/dev/null 2>&1 || { echo "RESOURCE_ESTIMATE_CHECK_FAIL no C compiler"; exit 2; }
command -v pnpm >/dev/null 2>&1 || { echo "RESOURCE_ESTIMATE_CHECK_FAIL no pnpm"; exit 2; }
mkdir -p build
cc -Wall -Wextra -std=c99 -Iinclude \
    src/common/model.c src/common/modelsize.c src/common/plan.c \
    src/tools/model_capacity_dump.c -o build/model_capacity_dump
./build/model_capacity_dump > build/model_capacity_oracle.json
(cd client && pnpm exec vite build --config vite.resource-check.config.ts >/dev/null)
node build/client-resource-check/resource_estimate_check.mjs \
    build/model_capacity_oracle.json
