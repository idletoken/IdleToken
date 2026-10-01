# MoE small-batch correctness probe

This standalone probe executes real GGML `MUL_MAT_ID` graphs with three expert
weight tensors in host memory and their operations on the selected GPU. Every
output is checked against independent host dot products. It needs no model,
tokenizer, network service or downloaded weights. It measures correctness and
branch coverage, not generation speed.

The validator covers 12 configurations and 432 graphs: token counts 1, 2, 4, 65,
128 and 129; shape changes; strided routing ids; ids on a different backend;
cache capacity below the active expert count, below the batch's expert union,
and exactly at the union; existing mapped-readback, pinned-DMA and sibling-issue
diagnostic controls. All configurations must have identical output bytes within
each input group. The multi-token-only case requires actual sibling prefetches;
disabling sibling issue must remove them without changing numeric results.

Build the pinned, patched engine first. Add the probe to that existing CMake
build using an absolute path to the included hook:

```sh
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DCMAKE_PROJECT_INCLUDE:FILEPATH="$PWD/scripts/mtp_moe_batch_probe.cmake"
cmake --build vendor/llama.cpp/build --config Release --target idletoken-moe-batch-probe
python3 scripts/mtp_moe_batch_validate.py \
  --probe vendor/llama.cpp/build/idletoken-moe-batch-probe \
  --device MTL0 --prefix metal --out results/mtp-moe-batch
```

For Windows/MSVC, configure the existing CUDA build from its normal developer
shell. The hook links the same `ggml` CMake target and dependencies as the engine:

```powershell
$hook = (Resolve-Path scripts/mtp_moe_batch_probe.cmake).Path
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build "-DCMAKE_PROJECT_INCLUDE:FILEPATH=$hook"
cmake --build vendor/llama.cpp/build --config Release --target idletoken-moe-batch-probe
python scripts/mtp_moe_batch_validate.py --probe vendor/llama.cpp/build/Release/idletoken-moe-batch-probe.exe --device CUDA0 --prefix cuda --out results/mtp-moe-batch
```

The executable location depends on the generator: a single-configuration build
places it at the build root, while Visual Studio normally uses `Release/`.
The validator sets its own bounded expert-cache diagnostic environment and
clears inherited `GGML_MOE_*` settings. CUDA validation additionally requires the
default mapped pinned-readback allocation to appear in the engine log. Metal
validation exercises expert pools and events but cannot validate CUDA mapping.

Remove the temporary hook from the build configuration after validation:

```sh
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -U CMAKE_PROJECT_INCLUDE
```

The optional `--out` controls all logs and the validation JSON. Its default is
`results/mtp-moe-batch` under the repository, so running the script does not write
logs beside its source. Existing `--probe`, `--device` and `--prefix` invocations
remain supported. Probe source and binary hashes are recorded separately.
Check portable source references without a GPU or engine build:

```sh
python3 scripts/mtp_moe_batch_validate.py --self-test
```

This self-check verifies the canonical probe and CMake hook names and prints the
same canonical source hash used by GPU validation. Changing `--out`, the working
directory or the repository location does not redirect source hashing to a log
directory. The historical validation files under `results/` remain evidence;
these files under `scripts/` are the reusable test source.
