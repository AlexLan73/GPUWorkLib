# ============================================================================
# Strategies Module - CMakeLists.txt
# ============================================================================
#
# ROCm-only module: Antenna array processor pipeline.
# d_S -> GEMM -> Window+FFT -> post-FFT scenarios -> AntennaResult
#
# Dependencies:
#   - hipBLAS (GEMM)
#   - hipFFT (batch C2C FFT)
#   - hiprtc (kernel compilation)
#   - statistics module (debug point stats)
#   - fft_maxima module (AllMaxima pipeline for Step2.2)
#
# ============================================================================

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME strategies)

# ============================================================================
# Check ROCm dependency
# ============================================================================

if(NOT ROCM_ENABLED)
    message(STATUS "  ${MODULE_NAME}: skipped (ROCm disabled)")
    return()
endif()

message(STATUS "Configuring module: ${MODULE_NAME} (ROCm)")

# ============================================================================
# Find hipBLAS and hipFFT
# ============================================================================

find_package(hipblas QUIET)
if(NOT hipblas_FOUND)
    message(FATAL_ERROR "  ${MODULE_NAME}: hipBLAS NOT FOUND")
endif()

find_package(hipfft QUIET)
if(NOT hipfft_FOUND)
    message(FATAL_ERROR "  ${MODULE_NAME}: hipFFT NOT FOUND")
endif()

# ============================================================================
# Library
# ============================================================================

add_library(${MODULE_NAME} STATIC
    src/antenna_processor_v1.cpp
    src/weight_generator.cpp
)

set_property(TARGET ${MODULE_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(${MODULE_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
        ${CMAKE_SOURCE_DIR}/modules/fft_func/include
        ${CMAKE_SOURCE_DIR}/modules/statistics/include
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
        drvgpu
        statistics
        fft_func
        hip::host
        roc::hipblas
        hip::hipfft
)

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

# Alias
add_library(GPUWorkLib::${MODULE_NAME} ALIAS ${MODULE_NAME})

# ============================================================================
# Install
# ============================================================================

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

# ============================================================================
# Summary
# ============================================================================

message(STATUS "  ${MODULE_NAME}: configured (ROCm only)")
message(STATUS "    - Pipeline: GEMM -> Window+FFT -> Step2.1/2.2/2.3")
message(STATUS "    - GEMM: hipBLAS Cgemm")
message(STATUS "    - FFT: hipFFT batch C2C")
message(STATUS "    - Kernels: hiprtc (hamming, pad, magnitudes, minmax)")
