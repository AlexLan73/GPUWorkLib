# ════════════════════════════════════════════════════════════════════════════
# vector_algebra Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# ROCm-only module: Cholesky-based matrix inversion using rocBLAS + rocSOLVER.
# Task_11 v2: два режима симметризации (Roundtrip / GpuKernel via hiprtc).
#
#   - CholeskyInverterROCm  — POTRF + POTRI on AMD GPU
#   - SymmetrizeMode        — Roundtrip (CPU) или GpuKernel (hiprtc)
#   - Batched variant       — multiple matrices in one call
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME vector_algebra)

# ════════════════════════════════════════════════════════════════════════════
# Check ROCm + rocBLAS + rocSOLVER dependencies
# ════════════════════════════════════════════════════════════════════════════

if(NOT ROCM_ENABLED)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (ROCm disabled)")
    return()
endif()

if(NOT rocblas_FOUND)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (rocBLAS not found)")
    return()
endif()

if(NOT rocsolver_FOUND)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (rocSOLVER not found)")
    return()
endif()

message(STATUS "Configuring module: ${MODULE_NAME} (ROCm + rocBLAS + rocSOLVER + hiprtc)")

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC
    src/cholesky_inverter_rocm.cpp
    src/symmetrize_gpu_rocm.cpp
)

# Required for linking into a shared library (.so / Python .pyd)
set_property(TARGET ${MODULE_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(${MODULE_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
        drvgpu
        hip::host
        roc::rocblas
        roc::rocsolver
)

# hiprtc for GPU kernel compilation (SymmetrizeMode::GpuKernel)
if(hiprtc_FOUND)
    target_link_libraries(${MODULE_NAME} PRIVATE hiprtc::hiprtc)
else()
    message(WARNING "  ${MODULE_NAME}: hiprtc not found — GpuKernel mode will fail at runtime")
endif()

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

target_compile_definitions(${MODULE_NAME} PUBLIC ENABLE_ROCM=1)

# Alias
add_library(GPUWorkLib::${MODULE_NAME} ALIAS ${MODULE_NAME})

# ════════════════════════════════════════════════════════════════════════════
# Install
# ════════════════════════════════════════════════════════════════════════════

install(TARGETS ${MODULE_NAME}
    EXPORT GPUWorkLibTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/${MODULE_NAME}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

message(STATUS "  ${MODULE_NAME}: configured (ROCm only)")
message(STATUS "    - CholeskyInverterROCm: POTRF + POTRI via rocSOLVER")
message(STATUS "    - SymmetrizeMode: Roundtrip (CPU) + GpuKernel (hiprtc)")
message(STATUS "    - Inputs: CPU vector<complex<float>>, void* (ROCm), cl_mem (ZeroCopy)")
message(STATUS "    - Batched: multiple matrices in one call")
