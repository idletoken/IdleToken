# Add this standalone correctness probe to an existing engine build without
# editing its CMakeLists.txt. Works with the engine's CUDA/MSVC and Metal setup.
if(PROJECT_NAME STREQUAL "llama.cpp" AND NOT TARGET idletoken-moe-batch-probe)
    add_executable(idletoken-moe-batch-probe "${CMAKE_CURRENT_LIST_DIR}/mtp_moe_batch_probe.cpp")
    target_compile_features(idletoken-moe-batch-probe PRIVATE cxx_std_17)
    target_include_directories(idletoken-moe-batch-probe PRIVATE "${CMAKE_SOURCE_DIR}/ggml/include")
    target_link_libraries(idletoken-moe-batch-probe PRIVATE ggml)
endif()
