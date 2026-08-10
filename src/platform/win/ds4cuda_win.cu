/* Windows CUDA build wrapper. nvcc does not forward -D/-Xcompiler to the pass
   that trips CUDA 13.3 CCCL's MSVC-preprocessor check, so we define the
   escape macro IN SOURCE (seen by every preprocessor pass) before the real
   translation unit is pulled in. Compiled with -I vendor/ds4. */
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING 1
#include "ds4_cuda.cu"
