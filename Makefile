# IdleToken Cluster top-level Makefile.
#
# Builds two binaries:
#   idletoken-worker  — inference shard. Links the GPU ds4 backend of the host
#                       platform: CUDA on Linux, Metal on macOS.
#   idletoken-coord   — scheduler + API server. CPU-only ds4 (-DDS4_NO_GPU),
#                       identical on both platforms (it never touches a GPU).
#
# Compute hosts: Linux + NVIDIA CUDA, or macOS + Apple Silicon (Metal).
# Windows is built by scripts/build_*_win.bat, not from here.
#
# The Metal path is the SAME ds4 engine: vendor/ds4 implements ds4_gpu.h twice
# (ds4_cuda.cu / ds4_metal.m) and every IdleToken PP extension sits above that
# interface, so nothing under src/ is backend-specific. The only Mac-only file
# is src/platform/mac/mac_gpu.m, the Metal facts the C resource probe needs.

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Linux)
  IDLETOKEN_GPU := cuda
else ifeq ($(UNAME_S),Darwin)
  IDLETOKEN_GPU := metal
  ifneq ($(UNAME_M),arm64)
    $(info IdleToken Cluster needs Apple Silicon on macOS: the ds4 Metal kernels)
    $(info assume unified memory, which an Intel Mac's GPU does not have.)
    $(error unsupported mac architecture: $(UNAME_M))
  endif
else
  $(info IdleToken Cluster builds on Linux + NVIDIA CUDA, or macOS + Apple Silicon.)
  $(info Detected host: $(UNAME_S). Use ./scripts/sync-to-spark.sh to build remotely.)
  $(error unsupported build host)
endif

CC              ?= cc
CUDA_HOME       ?= /usr/local/cuda
NVCC            ?= $(CUDA_HOME)/bin/nvcc
ifeq ($(IDLETOKEN_GPU),metal)
  NATIVE_CPU_FLAG ?= -mcpu=native
else
  NATIVE_CPU_FLAG ?= -march=native
endif

DS4 := vendor/ds4

# _GNU_SOURCE / -fno-finite-math-only are glibc- and gcc-shaped; the vendored
# ds4 Makefile applies them only off-Darwin and we follow it rather than find
# out the hard way which header they perturb on macOS.
# -MMD -MP: the compiler writes a .d file listing every header the object
# actually depends on, and we -include those at the bottom. Before 2026-08-12
# each pattern rule carried a HAND-WRITTEN list of headers instead, which is a
# trap rather than a mechanism: a header missing from the list means `make`
# does nothing after you edit it and quietly links the OLD object. That is not
# a build annoyance, it is a false green — a negative control run this way
# "passed" against a binary that never contained the change being tested.
CFLAGS_BASE := -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99 \
               -MMD -MP -I$(DS4) -Iinclude -Ivendor/tweetnacl
ifeq ($(IDLETOKEN_GPU),cuda)
  CFLAGS_BASE += -D_GNU_SOURCE -fno-finite-math-only
endif

ifeq ($(IDLETOKEN_GPU),cuda)
  # Worker needs NVML headers (in CUDA toolkit, not in default sysroot).
  # IDLETOKEN_DS4X_CUDA turns on the ds4x GPU matvec (the worker already links
  # cudart); without it a ds4x cluster would silently serve on CPU only.
  CFLAGS_WORKER := $(CFLAGS_BASE) -I$(CUDA_HOME)/include -DIDLETOKEN_DS4X_CUDA
else
  # No -DIDLETOKEN_DS4X_CUDA: ds4x has no Metal kernel yet, so small models run
  # on its C reference path. That is a speed gap, not a correctness one — and
  # it is LOUD rather than silent, because ds4x_cuda_available() is compiled
  # out entirely instead of returning 0 at runtime.
  CFLAGS_WORKER := $(CFLAGS_BASE)
endif
CFLAGS_COORD  := $(CFLAGS_BASE) -DDS4_NO_GPU
ifeq ($(IDLETOKEN_GPU),cuda)
  # resource.c (now in the coord too, v2 WS-B2) includes nvml.h on Linux.
  CFLAGS_COORD += -I$(CUDA_HOME)/include
endif

# Objective-C for the Metal sources. -fobjc-arc matches vendor/ds4's Makefile;
# mixing ARC and non-ARC translation units in one binary is legal but the
# vendored .m assumes ARC.
OBJCFLAGS := -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc \
             -MMD -MP -I$(DS4) -Iinclude

NVCCFLAGS := -O3 --use_fast_math \
             -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread

# libnvidia-ml.so.1 ships with the NVIDIA driver and is usually in the linker
# default path; CUDA_HOME/lib64 is a fallback for non-standard installs.
CUDA_LDLIBS := -lm -Xcompiler -pthread \
               -L$(CUDA_HOME)/targets/sbsa-linux/lib \
               -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lnvidia-ml

METAL_LDLIBS := -lm -pthread -framework Foundation -framework Metal

WORKER_BUILD := build/worker
COORD_BUILD  := build/coord

# vendor objects. ds4_cuda.o / ds4_metal.o are the two implementations of the
# same ds4_gpu.h; exactly one is linked.
ifeq ($(IDLETOKEN_GPU),cuda)
  DS4_GPU_OBJ := $(WORKER_BUILD)/vendor/ds4_cuda.o
else
  DS4_GPU_OBJ := $(WORKER_BUILD)/vendor/ds4_metal.o
endif
DS4_WORKER_OBJ := $(WORKER_BUILD)/vendor/ds4.o \
                  $(DS4_GPU_OBJ) \
                  $(WORKER_BUILD)/vendor/rax.o \
                  $(WORKER_BUILD)/vendor/tweetnacl.o
DS4_COORD_OBJ  := $(COORD_BUILD)/vendor/ds4.o \
                  $(COORD_BUILD)/vendor/rax.o \
                  $(COORD_BUILD)/vendor/tweetnacl.o

# common (resource.c is worker-only — pulls in NVML; http.c is coord-only)
# nodecrypt.c + privacy.c: token-id encryption on the coord<->worker link
# (docs/inter-node-encryption.md). privacy.c owns the XSalsa20-Poly1305
# primitive and was already used by the privacy proxy; nodecrypt.c adds the
# counter-nonce framing. TweetNaCl is vendored with no external dependency,
# which is the whole reason this is cheap -- see the design's §4.
COMMON_SRC_SHARED   := src/common/net.c src/common/discovery.c src/common/model.c \
                       src/common/nodecrypt.c src/common/privacy.c src/common/enginever.c
# advise.c (capability table) needs plan.c, which used to be coord-only — the
# worker now links both so `--advise` can answer "what can THIS machine run?"
# with the planner's own verdict instead of a second estimate.
COMMON_SRC_WORKER   := $(COMMON_SRC_SHARED) src/common/resource.c src/common/weights.c src/common/gguf.c \
                       src/common/plan.c src/common/advise.c
# resource.c + model_auto.c joined the coordinator for the llamacpp mode
# (v2 WS-B2/B4): the coord now probes ITS OWN machine (single-machine fit
# check + ctx sizing) and builds a runtime model spec from any GGUF header.
# On Linux that pulls NVML into the coord link — see the idletoken-coord rule.
COMMON_SRC_COORD    := $(COMMON_SRC_SHARED) src/common/http.c src/common/plan.c src/common/gguf.c \
                       src/common/advise.c src/common/resource.c src/common/model_auto.c \
                       src/common/apiconv.c
WORKER_COMMON_OBJ   := $(patsubst src/common/%.c,$(WORKER_BUILD)/common/%.o,$(COMMON_SRC_WORKER))

# ds4x generic CPU backend (small models: Qwen3 GQA, GLM/Kimi MLA). Pure C, no
# CUDA — links into the worker so a ds4x cluster serves on CPU (a CUDA kernel is
# a later speed-up, not a correctness gate). small-model-design.md §S-C.
DS4X_UNITS          := ds4x_config ds4x_model ds4x_forward ds4x_runner ds4x_quant
DS4X_WORKER_OBJ     := $(patsubst %,$(WORKER_BUILD)/ds4x/%.o,$(DS4X_UNITS))
ifeq ($(IDLETOKEN_GPU),cuda)
  DS4X_WORKER_OBJ   += $(WORKER_BUILD)/ds4x/ds4x_cuda.o
endif

# macOS-only: the Metal facts resource.c cannot reach from plain C. Worker only
# (the coordinator never probes hardware), and it costs no new framework — the
# worker already links Metal for ds4.
ifeq ($(IDLETOKEN_GPU),metal)
  PLATFORM_WORKER_OBJ := $(WORKER_BUILD)/platform/mac_gpu.o
else
  PLATFORM_WORKER_OBJ :=
endif
# Coord only needs the GGUF byte-BPE tokenizer (prompt encode + detokenize for
# ds4x models); embed/lm_head run on the workers.
DS4X_COORD_OBJ      := $(COORD_BUILD)/ds4x/ds4x_tokenizer.o
COORD_COMMON_OBJ    := $(patsubst src/common/%.c,$(COORD_BUILD)/common/%.o,$(COMMON_SRC_COORD))

# binary-specific
WORKER_MAIN_OBJ := $(WORKER_BUILD)/worker_main.o
COORD_MAIN_OBJ  := $(COORD_BUILD)/coord_main.o $(COORD_BUILD)/llama_sidecar.o

.PHONY: all worker coord clean check info plantest disctest autotest apitest

all: worker coord

# Unit tests for the planning core (mode decision + layer split). Pure C —
# also runs on the mac control machine with plain cc.
plantest:
	$(CC) -Wall -Wextra -std=c99 -Iinclude src/common/plan.c src/common/model.c src/common/advise.c src/tools/plan_test.c -o build/plan_test
	./build/plan_test

# Unit tests for the API surface helpers: Anthropic<->OpenAI translation
# (apiconv.c, incl. tool_use/tool_result), overlay-address detection, PSK hex
# validation, HTTP query stripping and Authorization matching. Pure C — also
# runs on the mac control machine with plain cc.
apitest:
	@mkdir -p build
	$(CC) -Wall -Wextra -std=c99 -D_GNU_SOURCE -Iinclude src/common/apiconv.c \
	    src/common/net.c src/common/http.c src/tools/api_test.c -o build/api_test
	./build/api_test

# Unit tests for the open-model intake (GGUF header -> runtime model spec,
# v2 WS-B4). Pure C — reuses the metadata-only fixtures.
autotest:
	@mkdir -p build
	python3 scripts/make_test_gguf.py build/fixtures
	$(CC) -Wall -Wextra -std=c99 -Iinclude src/common/gguf.c src/common/model.c \
	    src/common/model_auto.c src/tools/model_auto_test.c -o build/model_auto_test
	./build/model_auto_test build/fixtures

# Unit tests for the ds4x runtime config (GGUF-metadata-driven, Phase B).
# Pure C — fixtures are metadata-only GGUFs generated by a small script.
ds4xtest:
	@mkdir -p build
	python3 scripts/make_test_gguf.py build/fixtures
	$(CC) -Wall -Wextra -std=c99 -Iinclude src/common/gguf.c src/common/model.c \
	    src/ds4x/ds4x_config.c src/tools/ds4x_config_test.c -o build/ds4x_config_test
	./build/ds4x_config_test build/fixtures
	$(CC) -Wall -Wextra -std=c99 -Iinclude src/ds4x/ds4x_quant.c \
	    src/tools/ds4x_quant_test.c -o build/ds4x_quant_test -lm
	./build/ds4x_quant_test
	python3 scripts/ds4x_ref.py build/fixtures/ds4x_vectors.bin
	$(CC) -Wall -Wextra -std=c99 -Iinclude src/common/model.c \
	    src/ds4x/ds4x_config.c src/ds4x/ds4x_forward.c src/ds4x/ds4x_model.c \
	    src/ds4x/ds4x_runner.c src/ds4x/ds4x_quant.c src/common/gguf.c \
	    src/tools/ds4x_forward_test.c -o build/ds4x_forward_test -lm
	./build/ds4x_forward_test build/fixtures/ds4x_vectors.bin build/fixtures/ds4x_vectors.bin.gguf
	python3 scripts/ds4x_ref.py build/fixtures/ds4x_noqlora.bin --q-rank 0
	./build/ds4x_forward_test build/fixtures/ds4x_noqlora.bin build/fixtures/ds4x_noqlora.bin.gguf
	python3 scripts/ds4x_ref.py build/fixtures/ds4x_q8.bin --quant q8_0
	./build/ds4x_forward_test build/fixtures/ds4x_q8.bin build/fixtures/ds4x_q8.bin.gguf
	python3 scripts/ds4x_ref.py build/fixtures/ds4x_qwen3.bin --arch qwen3
	$(CC) -Wall -Wextra -std=c99 -Iinclude src/common/model.c \
	    src/ds4x/ds4x_config.c src/ds4x/ds4x_forward.c src/ds4x/ds4x_model.c \
	    src/ds4x/ds4x_runner.c src/ds4x/ds4x_quant.c src/common/gguf.c \
	    src/tools/ds4x_gqa_test.c -o build/ds4x_gqa_test -lm
	./build/ds4x_gqa_test build/fixtures/ds4x_qwen3.bin build/fixtures/ds4x_qwen3.bin.gguf
	python3 scripts/ds4x_ref.py build/fixtures/ds4x_gdn.bin --arch qwen3next
	$(CC) -Wall -Wextra -std=c99 -Iinclude src/common/model.c \
	    src/ds4x/ds4x_config.c src/ds4x/ds4x_forward.c src/ds4x/ds4x_model.c \
	    src/ds4x/ds4x_runner.c src/ds4x/ds4x_quant.c src/common/gguf.c \
	    src/tools/ds4x_gdn_test.c -o build/ds4x_gdn_test -lm
	./build/ds4x_gdn_test build/fixtures/ds4x_gdn.bin
	$(CC) -Wall -Wextra -std=c99 -Iinclude src/common/gguf.c \
	    src/ds4x/ds4x_tokenizer.c src/tools/ds4x_tok_test.c -o build/ds4x_tok_test
	./build/ds4x_tok_test build/fixtures
	$(CC) -Wall -Wextra -std=c99 -Iinclude src/common/model.c \
	    src/ds4x/ds4x_config.c src/ds4x/ds4x_forward.c src/ds4x/ds4x_model.c \
	    src/ds4x/ds4x_runner.c src/ds4x/ds4x_tokenizer.c src/ds4x/ds4x_quant.c \
	    src/common/gguf.c \
	    src/ds4x/ds4x_infer.c -o build/ds4x_infer -lm
	./build/ds4x_infer build/fixtures/ds4x_vectors.bin.gguf --selftest build/fixtures/ds4x_vectors.bin

# CUDA parity gate: GPU dequant+matvec must match the CPU reference for every
# quant type (small-model-design §S-C). Needs nvcc + a device; on a machine
# without CUDA the binary reports SKIP (loudly) instead of a false green.
.PHONY: cudatest
cudatest:
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -Iinclude -c -o build/ds4x_cuda.o src/ds4x/ds4x_cuda.cu
	$(CC) -Wall -Wextra -std=c99 -Iinclude -c -o build/ds4x_quant_for_cuda.o src/ds4x/ds4x_quant.c
	@# ds4x_forward.c WITHOUT -DIDLETOKEN_DS4X_CUDA: the test's reference must be
	@# the CPU path itself (ds4x_gdn_recur_cpu), never a copy of it in the test.
	$(CC) -Wall -Wextra -std=c99 -Iinclude -c -o build/ds4x_forward_for_cuda.o src/ds4x/ds4x_forward.c
	$(CC) -Wall -Wextra -std=c99 -Iinclude -c -o build/ds4x_cuda_test.o src/tools/ds4x_cuda_test.c
	$(NVCC) $(NVCCFLAGS) -o build/ds4x_cuda_test \
	    build/ds4x_cuda_test.o build/ds4x_quant_for_cuda.o \
	    build/ds4x_forward_for_cuda.o build/ds4x_cuda.o $(CUDA_LDLIBS)
	./build/ds4x_cuda_test

# CUDA-accelerated standalone inference tool (same binary as ds4x_infer, but
# with the GPU matvec compiled in). Run with IDLETOKEN_DS4X_CUDA=1 to use the GPU;
# without the env var it behaves exactly like the CPU build.
XI_CFLAGS := -O3 -march=native -Wall -Wextra -std=c99 -Iinclude -DIDLETOKEN_DS4X_CUDA
XI_CSRC   := src/common/model.c src/ds4x/ds4x_config.c src/ds4x/ds4x_forward.c \
             src/ds4x/ds4x_model.c src/ds4x/ds4x_runner.c src/ds4x/ds4x_tokenizer.c \
             src/ds4x/ds4x_quant.c src/common/gguf.c src/ds4x/ds4x_infer.c
XI_OBJ    := $(patsubst %.c,build/xi/%.o,$(notdir $(XI_CSRC)))

.PHONY: ds4xinfer-cuda
ds4xinfer-cuda:
	@mkdir -p build/xi
	$(NVCC) $(NVCCFLAGS) -Iinclude -c -o build/xi/ds4x_cuda.o src/ds4x/ds4x_cuda.cu
	@for f in $(XI_CSRC); do \
	    echo "  CC $$f"; \
	    $(CC) $(XI_CFLAGS) -c -o build/xi/`basename $$f .c`.o $$f || exit 1; \
	done
	$(NVCC) $(NVCCFLAGS) -o build/ds4x_infer_cuda $(XI_OBJ) build/xi/ds4x_cuda.o $(CUDA_LDLIBS)

# Unit tests for LAN discovery + verification-code pairing. Pure C (+pthread) —
# runs on the mac control machine and Linux nodes; no GPU/model needed.
disctest:
	@mkdir -p build
	$(CC) -Wall -Wextra -std=c99 -D_GNU_SOURCE -Iinclude \
	    src/common/net.c src/common/discovery.c src/tools/discovery_test.c \
	    -o build/discovery_test -lpthread
	./build/discovery_test

worker: idletoken-worker
coord:  idletoken-coord

idletoken-worker: $(WORKER_MAIN_OBJ) $(WORKER_COMMON_OBJ) $(DS4X_WORKER_OBJ) $(DS4_WORKER_OBJ) $(PLATFORM_WORKER_OBJ)
ifeq ($(IDLETOKEN_GPU),cuda)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
else
	$(CC) -o $@ $^ $(METAL_LDLIBS)
endif

# The coord links the hardware probe since v2 WS-B2 (llamacpp-mode fit check):
# NVML on Linux (ships with the driver), the Metal facts object on macOS.
ifeq ($(IDLETOKEN_GPU),metal)
  PLATFORM_COORD_OBJ := $(COORD_BUILD)/platform/mac_gpu.o
  COORD_PROBE_LDLIBS := -framework Foundation -framework Metal
else
  PLATFORM_COORD_OBJ :=
  COORD_PROBE_LDLIBS := -L$(CUDA_HOME)/targets/sbsa-linux/lib \
                        -L$(CUDA_HOME)/lib64 -lnvidia-ml
endif

idletoken-coord: $(COORD_MAIN_OBJ) $(COORD_COMMON_OBJ) $(DS4X_COORD_OBJ) $(DS4_COORD_OBJ) $(PLATFORM_COORD_OBJ)
	$(CC) $(CFLAGS_COORD) -o $@ $^ -lm -pthread $(COORD_PROBE_LDLIBS)

$(COORD_BUILD)/platform/mac_gpu.o: src/platform/mac/mac_gpu.m include/idletoken_mac_gpu.h | $(COORD_BUILD)/platform
	$(CC) $(OBJCFLAGS) -c -o $@ $<

# --- vendor/ds4 sources (worker variant, GPU on) ----------------------------

$(WORKER_BUILD)/vendor/ds4.o: $(DS4)/ds4.c $(DS4)/ds4.h $(DS4)/ds4_gpu.h | $(WORKER_BUILD)/vendor
	$(CC) $(CFLAGS_WORKER) -c -o $@ $<

$(WORKER_BUILD)/vendor/ds4_cuda.o: $(DS4)/ds4_cuda.cu $(DS4)/ds4_gpu.h $(DS4)/ds4_iq2_tables_cuda.inc | $(WORKER_BUILD)/vendor
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(WORKER_BUILD)/vendor/ds4_metal.o: $(DS4)/ds4_metal.m $(DS4)/ds4_gpu.h | $(WORKER_BUILD)/vendor
	$(CC) $(OBJCFLAGS) -c -o $@ $<

$(WORKER_BUILD)/platform/mac_gpu.o: src/platform/mac/mac_gpu.m include/idletoken_mac_gpu.h | $(WORKER_BUILD)/platform
	$(CC) $(OBJCFLAGS) -c -o $@ $<

$(WORKER_BUILD)/vendor/rax.o: $(DS4)/rax.c $(DS4)/rax.h $(DS4)/rax_malloc.h | $(WORKER_BUILD)/vendor
	$(CC) $(CFLAGS_WORKER) -c -o $@ $<

# TweetNaCl is third-party and deliberately compiled without -Wall/-Wextra:
# it is a published, audited reference implementation, and its style (single
# letters, implicit conversions) produces noise we must not "fix".
$(WORKER_BUILD)/vendor/tweetnacl.o: vendor/tweetnacl/tweetnacl.c vendor/tweetnacl/tweetnacl.h | $(WORKER_BUILD)/vendor
	$(CC) -O3 -std=c99 -w -Ivendor/tweetnacl -c -o $@ $<

# --- vendor/ds4 sources (coord variant, CPU only) ---------------------------

$(COORD_BUILD)/vendor/ds4.o: $(DS4)/ds4.c $(DS4)/ds4.h $(DS4)/ds4_gpu.h | $(COORD_BUILD)/vendor
	$(CC) $(CFLAGS_COORD) -c -o $@ $<

$(COORD_BUILD)/vendor/rax.o: $(DS4)/rax.c $(DS4)/rax.h $(DS4)/rax_malloc.h | $(COORD_BUILD)/vendor
	$(CC) $(CFLAGS_COORD) -c -o $@ $<

$(COORD_BUILD)/vendor/tweetnacl.o: vendor/tweetnacl/tweetnacl.c vendor/tweetnacl/tweetnacl.h | $(COORD_BUILD)/vendor
	$(CC) -O3 -std=c99 -w -Ivendor/tweetnacl -c -o $@ $<

# --- our sources -----------------------------------------------------------

$(WORKER_BUILD)/common/%.o: src/common/%.c include/idletoken_proto.h include/idletoken_net.h include/idletoken_discovery.h include/idletoken_resource.h include/idletoken_weights.h | $(WORKER_BUILD)/common
	$(CC) $(CFLAGS_WORKER) -c -o $@ $<

$(COORD_BUILD)/common/%.o: src/common/%.c include/idletoken_proto.h include/idletoken_net.h include/idletoken_discovery.h include/idletoken_http.h | $(COORD_BUILD)/common
	$(CC) $(CFLAGS_COORD) -c -o $@ $<

$(COORD_BUILD)/ds4x/%.o: src/ds4x/%.c include/idletoken_ds4x_tok.h include/idletoken_gguf.h | $(COORD_BUILD)/ds4x
	$(CC) $(CFLAGS_COORD) -c -o $@ $<

$(WORKER_BUILD)/ds4x/%.o: src/ds4x/%.c include/idletoken_ds4x.h include/idletoken_ds4x_quant.h include/idletoken_gguf.h include/idletoken_model.h | $(WORKER_BUILD)/ds4x
	$(CC) $(CFLAGS_WORKER) -c -o $@ $<

$(WORKER_BUILD)/ds4x/ds4x_cuda.o: src/ds4x/ds4x_cuda.cu include/idletoken_ds4x_cuda.h | $(WORKER_BUILD)/ds4x
	$(NVCC) $(NVCCFLAGS) -Iinclude -c -o $@ $<

$(WORKER_BUILD)/%.o: src/worker/%.c include/idletoken_proto.h include/idletoken_net.h | $(WORKER_BUILD)
	$(CC) $(CFLAGS_WORKER) -c -o $@ $<

$(COORD_BUILD)/%.o: src/coord/%.c include/idletoken_proto.h include/idletoken_net.h | $(COORD_BUILD)
	$(CC) $(CFLAGS_COORD) -c -o $@ $<

# --- dirs ------------------------------------------------------------------

$(WORKER_BUILD) $(WORKER_BUILD)/vendor $(WORKER_BUILD)/common $(WORKER_BUILD)/ds4x $(WORKER_BUILD)/platform $(COORD_BUILD) $(COORD_BUILD)/vendor $(COORD_BUILD)/common $(COORD_BUILD)/ds4x $(COORD_BUILD)/platform:
	@mkdir -p $@

# --- helpers ---------------------------------------------------------------

info:
	@echo "DS4 vendor path:  $(DS4)"
	@echo "CUDA_HOME:        $(CUDA_HOME)"
	@echo "NVCC:             $(NVCC)"
	@echo "CFLAGS_WORKER:    $(CFLAGS_WORKER)"
	@echo "CFLAGS_COORD:     $(CFLAGS_COORD)"

clean:
	rm -rf build idletoken-worker idletoken-coord

# --- header dependencies (generated by -MMD, see CFLAGS_BASE) ---------------
#
# Absent on a first build; `-include` is silent about that, and the objects get
# built anyway. The nvcc-compiled objects (ds4_cuda.o, ds4x_cuda.o) are NOT
# covered — they keep the explicit header prerequisites on their own rules.
ALL_DEPS := $(WORKER_MAIN_OBJ:.o=.d) $(WORKER_COMMON_OBJ:.o=.d) \
            $(DS4X_WORKER_OBJ:.o=.d) $(DS4_WORKER_OBJ:.o=.d) \
            $(PLATFORM_WORKER_OBJ:.o=.d) \
            $(COORD_MAIN_OBJ:.o=.d) $(COORD_COMMON_OBJ:.o=.d) \
            $(DS4X_COORD_OBJ:.o=.d) $(DS4_COORD_OBJ:.o=.d)
-include $(ALL_DEPS)
