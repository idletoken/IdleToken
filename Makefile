# IdleToken Cluster top-level Makefile.
#
# The inference engine is llama.cpp, and it is NOT built from here: it is a
# sidecar. scripts/build_llamacpp.sh builds the pinned commit plus our patches,
# and the two binaries below spawn llama-server / ggml rpc-server as separate
# processes. Nothing here links libllama.
#
# Builds two binaries:
#   idletoken-worker  — supervises the rpc-server on a compute node: pairing,
#                       credentials, NIC selection, resource reporting.
#   idletoken-coord   — scheduler + API server. Drives llama-server on the
#                       coordinator and speaks the OpenAI/Anthropic APIs.
#
# ds4 (our own generic kernels) is SHELVED since 2026-08-16 and is NOT part of
# either binary: the call sites link src/common/ds4_stub.c and no ds4 object is
# compiled. See the IDLETOKEN_WITH_DS4 block further down for the switch that
# links the real thing again, which is for archaeology, not for shipping. The
# ds4x* targets in this file are frozen leftovers of that line.
#
# Compute hosts: Linux + NVIDIA CUDA, or macOS + Apple Silicon (Metal).
# Windows is built by scripts/build_*_win.bat, not from here.
#
# macOS is Apple Silicon only because the engine's Mac backend is Metal on
# unified memory. The only Mac-specific file under src/ is
# src/platform/mac/mac_gpu.m, the Metal facts the C resource probe needs.

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Linux)
  IDLETOKEN_GPU := cuda
else ifeq ($(UNAME_S),Darwin)
  IDLETOKEN_GPU := metal
  ifneq ($(UNAME_M),arm64)
    $(info IdleToken Cluster needs Apple Silicon on macOS: the Mac compute path)
    $(info is Metal on unified memory, which an Intel Mac's GPU does not have.)
    $(error unsupported mac architecture: $(UNAME_M))
  endif
else
  $(info IdleToken Cluster builds on Linux + NVIDIA CUDA, or macOS + Apple Silicon.)
  $(error unsupported build host)
endif

CC              ?= cc
CXX             ?= c++
CUDA_HOME       ?= /usr/local/cuda
NVCC            ?= $(CUDA_HOME)/bin/nvcc
ifeq ($(IDLETOKEN_GPU),metal)
  NATIVE_CPU_FLAG ?= -mcpu=native
else
  NATIVE_CPU_FLAG ?= -march=native
endif

DS4 := vendor/ds4

# The public mirror intentionally has no frozen ds4/ds4x headers or sources.
# sync-public.sh flips this marker in the mirror's Makefile only.  The private
# tree leaves it at zero and therefore keeps compiling ds4_stub.c against the
# real historical headers as a signature check.
IDLETOKEN_PUBLIC_SOURCE := 1
PUBLIC_LEGACY_COMPAT_DIR := build/public-compat

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
CFLAGS_BASE := -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99 -MMD -MP
ifeq ($(IDLETOKEN_PUBLIC_SOURCE),1)
  # Generated wrapper names satisfy untouched historical includes; their only
  # content is the public, refusing ABI.  Put this directory first so a stray
  # local legacy header cannot silently change a public build.
  CFLAGS_BASE += -I$(PUBLIC_LEGACY_COMPAT_DIR)
endif
CFLAGS_BASE += -I$(DS4) -Iinclude -Ivendor/tweetnacl -Ivendor/blake2
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
# Overflow routing's trust anchor (docs/api-surface.md §5.1c): the ed25519 key
# the platform signs its encryption key with, pinned into the coordinator the
# way the client pins its updater key. Release builds pass it:
#
#   make coord IDLETOKEN_PLATFORM_VERIFY_KEY_B64=<base64 32 bytes>
#
# Left empty here on purpose. A wrong default would be worse than none: an
# unpinned build refuses to enable overflow, which is visible, while a build
# pinning a stale key would fail verification against the live platform and read
# as "the platform is down".
IDLETOKEN_PLATFORM_VERIFY_KEY_B64 ?=
ifneq ($(IDLETOKEN_PLATFORM_VERIFY_KEY_B64),)
  CFLAGS_COORD += -DIDLETOKEN_PLATFORM_VERIFY_KEY_B64='"$(IDLETOKEN_PLATFORM_VERIFY_KEY_B64)"'
endif
# ⚠ The pin is a compile-time input make cannot see; the stamp below
# (COORD_PIN_STAMP, defined once COORD_BUILD exists) makes it a real
# prerequisite. Without that, setting the pin on a warm tree changes nothing.

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

# The platform verify-key pin (CFLAGS_COORD, above) is a compile-time input
# that make cannot see. Set it on a tree whose build/coord is already warm and
# every object is still "up to date", so the link quietly reuses an overflow.o
# compiled WITHOUT it — and the result is a coordinator that refuses to enable
# sharing, which is precisely the bug the pin was added to fix. That is how the
# 0.1.19 macOS bundle shipped an unpinned coordinator while the Linux one,
# built in a colder tree, came out pinned (2026-08-23).
#
# The stamp turns the VALUE into a real prerequisite. Its recipe runs on every
# invocation but only replaces the file when the content differs, so warm
# builds stay warm and a changed pin rebuilds exactly what depends on it.
COORD_PIN_STAMP := $(COORD_BUILD)/.verify-key-pin
$(COORD_PIN_STAMP): FORCE | $(COORD_BUILD)
	@printf '%s' '$(IDLETOKEN_PLATFORM_VERIFY_KEY_B64)' > $@.new
	@cmp -s $@.new $@ 2>/dev/null || mv -f $@.new $@
	@rm -f $@.new
FORCE:
.PHONY: FORCE

# vendor objects. ds4_cuda.o / ds4_metal.o are the two implementations of the
# same ds4_gpu.h; exactly one is linked.
ifeq ($(IDLETOKEN_GPU),cuda)
  DS4_GPU_OBJ := $(WORKER_BUILD)/vendor/ds4_cuda.o
else
  DS4_GPU_OBJ := $(WORKER_BUILD)/vendor/ds4_metal.o
endif
# ds4/ds4x are SHELVED (2026-08-16): the engine is llama.cpp, and these are our
# own generic kernels — not a selectable backend, not tested, not published.
# Default builds link src/common/ds4_stub.c instead: the ~66 call sites per main
# file stay untouched (that code is frozen, not to be edited), the real objects
# are simply never compiled.
#
# What this buys, measured: a Windows worker no longer needs the CUDA Toolkit to
# build (a test machine with only the driver installed could not build at all
# before this), and the build stops spending ~10 minutes in nvcc on kernels that
# never execute.
#
# IDLETOKEN_WITH_DS4=1 links the real thing again — for archaeology on the
# frozen line, not for shipping. tweetnacl is NOT part of this: it is the
# pairing crypto, needed either way.
IDLETOKEN_WITH_DS4 ?= 0
CFLAGS_WORKER += -DIDLETOKEN_WITH_DS4=$(IDLETOKEN_WITH_DS4)
ifeq ($(IDLETOKEN_WITH_DS4),1)
ifeq ($(IDLETOKEN_PUBLIC_SOURCE),1)
$(error IDLETOKEN_WITH_DS4=1 is unavailable in the public source distribution)
endif
DS4_WORKER_OBJ := $(WORKER_BUILD)/vendor/ds4.o \
                  $(DS4_GPU_OBJ) \
                  $(WORKER_BUILD)/vendor/rax.o \
                  $(WORKER_BUILD)/vendor/tweetnacl.o
DS4_COORD_OBJ  := $(COORD_BUILD)/vendor/ds4.o \
                  $(COORD_BUILD)/vendor/rax.o \
                  $(COORD_BUILD)/vendor/tweetnacl.o
else
ifeq ($(IDLETOKEN_PUBLIC_SOURCE),1)
DS4_WORKER_OBJ := $(WORKER_BUILD)/common/legacy_backend_refusal.o \
                  $(WORKER_BUILD)/vendor/tweetnacl.o
DS4_COORD_OBJ  := $(COORD_BUILD)/common/legacy_backend_refusal.o \
                  $(COORD_BUILD)/vendor/tweetnacl.o
else
DS4_WORKER_OBJ := $(WORKER_BUILD)/common/ds4_stub.o \
                  $(WORKER_BUILD)/vendor/tweetnacl.o
DS4_COORD_OBJ  := $(COORD_BUILD)/common/ds4_stub.o \
                  $(COORD_BUILD)/vendor/tweetnacl.o
endif
endif
# BLAKE2b: the nonce of the libsodium-shape sealed box the coordinator seals
# overflow requests with (src/common/sodium_seal.c). Not part of the ds4 switch
# above -- like tweetnacl it is crypto we need either way.
DS4_COORD_OBJ  += $(COORD_BUILD)/vendor/blake2b.o

# common (resource.c is worker-only — pulls in NVML; http.c is coord-only)
# nodecrypt.c + privacy.c: token-id encryption on the coord<->worker link
# (docs/inter-node-encryption.md). privacy.c owns the XSalsa20-Poly1305
# primitive and was already used by the privacy proxy; nodecrypt.c adds the
# counter-nonce framing. TweetNaCl is vendored with no external dependency,
# which is the whole reason this is cheap -- see the design's §4.
# modelsize.c: the one place that answers "how big is the model we are about to
# load?" (T8). Shared because the coordinator's budget and the worker's
# capability advisor must not resolve the precision differently — that drift is
# how the advisor starts promising what the planner refuses.
COMMON_SRC_SHARED   := src/common/net.c src/common/discovery.c src/common/model.c \
                       src/common/nodecrypt.c src/common/privacy.c src/common/enginever.c \
                       src/common/modelsize.c src/common/gguf.c
# advise.c (capability table) needs plan.c, which used to be coord-only — the
# worker now links both so `--advise` can answer "what can THIS machine run?"
# with the planner's own verdict instead of a second estimate.
COMMON_SRC_WORKER   := $(COMMON_SRC_SHARED) src/common/resource.c src/common/weights.c \
                       src/common/plan.c src/common/advise.c
# resource.c + model_auto.c joined the coordinator for the llamacpp mode
# (v2 WS-B2/B4): the coord now probes ITS OWN machine (single-machine fit
# check + ctx sizing) and builds a runtime model spec from any GGUF header.
# On Linux that pulls NVML into the coord link — see the idletoken-coord rule.
# b64.c: the base64 the sealed envelope is spelled in. Shared with the platform
# agent (Makefile.platform) so the side that seals and the side that opens
# cannot drift.
COMMON_SRC_COORD    := $(COMMON_SRC_SHARED) src/common/http.c src/common/plan.c \
                       src/common/weights.c \
                       src/common/advise.c src/common/resource.c src/common/model_auto.c \
                       src/common/apiconv.c src/common/b64.c src/common/sodium_seal.c \
                       src/common/admission.c
WORKER_COMMON_OBJ   := $(patsubst src/common/%.c,$(WORKER_BUILD)/common/%.o,$(COMMON_SRC_WORKER))

# ds4x generic CPU backend (small models: Qwen3 GQA, GLM/Kimi MLA). Pure C, no
# CUDA — links into the worker so a ds4x cluster serves on CPU (a CUDA kernel is
# a later speed-up, not a correctness gate). small-model-design.md §S-C.
DS4X_UNITS          := ds4x_config ds4x_model ds4x_forward ds4x_runner ds4x_quant
ifeq ($(IDLETOKEN_WITH_DS4),1)
DS4X_WORKER_OBJ     := $(patsubst %,$(WORKER_BUILD)/ds4x/%.o,$(DS4X_UNITS))
ifeq ($(IDLETOKEN_GPU),cuda)
  DS4X_WORKER_OBJ   += $(WORKER_BUILD)/ds4x/ds4x_cuda.o
endif
else
DS4X_WORKER_OBJ     :=   # shelved — ds4_stub.c satisfies the call sites
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
ifeq ($(IDLETOKEN_WITH_DS4),1)
DS4X_COORD_OBJ      := $(COORD_BUILD)/ds4x/ds4x_tokenizer.o
else
DS4X_COORD_OBJ      :=   # shelved — ds4_stub.c satisfies the call sites
endif
COORD_COMMON_OBJ    := $(patsubst src/common/%.c,$(COORD_BUILD)/common/%.o,$(COMMON_SRC_COORD))

# binary-specific
WORKER_MAIN_OBJ := $(WORKER_BUILD)/worker_main.o
COORD_MAIN_OBJ  := $(COORD_BUILD)/coord_main.o $(COORD_BUILD)/llama_sidecar.o \
                   $(COORD_BUILD)/overflow.o

.PHONY: all worker coord clean info

all: worker coord
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

$(COORD_BUILD)/vendor/blake2b.o: vendor/blake2/blake2b.c vendor/blake2/blake2.h | $(COORD_BUILD)/vendor
	$(CC) $(CFLAGS_COORD) -c -o $@ $<

$(COORD_BUILD)/vendor/tweetnacl.o: vendor/tweetnacl/tweetnacl.c vendor/tweetnacl/tweetnacl.h | $(COORD_BUILD)/vendor
	$(CC) -O3 -std=c99 -w -Ivendor/tweetnacl -c -o $@ $<

# --- our sources -----------------------------------------------------------

$(WORKER_BUILD)/common/%.o: src/common/%.c include/idletoken_proto.h include/idletoken_net.h include/idletoken_discovery.h include/idletoken_resource.h include/idletoken_weights.h | $(WORKER_BUILD)/common
	$(CC) $(CFLAGS_WORKER) -c -o $@ $<

$(COORD_BUILD)/common/%.o: src/common/%.c include/idletoken_proto.h include/idletoken_net.h include/idletoken_discovery.h include/idletoken_http.h | $(COORD_BUILD)/common
	$(CC) $(CFLAGS_COORD) -c -o $@ $<

# Public-source builds retain the old call sites but publish neither their
# frozen headers nor implementation.  Generate four one-line compatibility
# include names as build artifacts; all resolve to the neutral refusal ABI.
PUBLIC_LEGACY_COMPAT_HEADERS := \
    $(PUBLIC_LEGACY_COMPAT_DIR)/ds4.h \
    $(PUBLIC_LEGACY_COMPAT_DIR)/idletoken_ds4x.h \
    $(PUBLIC_LEGACY_COMPAT_DIR)/idletoken_ds4x_tok.h \
    $(PUBLIC_LEGACY_COMPAT_DIR)/idletoken_ds4x_cuda.h
ifeq ($(IDLETOKEN_PUBLIC_SOURCE),1)
$(PUBLIC_LEGACY_COMPAT_HEADERS): include/idletoken_legacy_backend_refusal.h
	@mkdir -p $(PUBLIC_LEGACY_COMPAT_DIR)
	@printf '%s\n' '#include "idletoken_legacy_backend_refusal.h"' > $@

$(WORKER_MAIN_OBJ) $(COORD_BUILD)/coord_main.o: $(PUBLIC_LEGACY_COMPAT_HEADERS)
endif

$(COORD_BUILD)/ds4x/%.o: src/ds4x/%.c include/idletoken_ds4x_tok.h include/idletoken_gguf.h | $(COORD_BUILD)/ds4x
	$(CC) $(CFLAGS_COORD) -c -o $@ $<

$(WORKER_BUILD)/ds4x/%.o: src/ds4x/%.c include/idletoken_ds4x.h include/idletoken_ds4x_quant.h include/idletoken_gguf.h include/idletoken_model.h | $(WORKER_BUILD)/ds4x
	$(CC) $(CFLAGS_WORKER) -c -o $@ $<

$(WORKER_BUILD)/ds4x/ds4x_cuda.o: src/ds4x/ds4x_cuda.cu include/idletoken_ds4x_cuda.h | $(WORKER_BUILD)/ds4x
	$(NVCC) $(NVCCFLAGS) -Iinclude -c -o $@ $<

$(WORKER_BUILD)/%.o: src/worker/%.c include/idletoken_proto.h include/idletoken_net.h | $(WORKER_BUILD)
	$(CC) $(CFLAGS_WORKER) -c -o $@ $<

$(COORD_BUILD)/%.o: src/coord/%.c include/idletoken_proto.h include/idletoken_net.h $(COORD_PIN_STAMP) | $(COORD_BUILD)
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
