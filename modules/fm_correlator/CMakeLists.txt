# ════════════════════════════════════════════════════════════════════════════
# FM Correlator Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# ROCm-only module: frequency-domain correlation for M-sequence phase
# modulated signals using hipFFT + HIP kernels (hiprtc).
#
# Pipeline:
#   apply_cyclic_shifts → C2C FFT → R2C FFT → multiply_conj_fused
#   → C2R IFFT → extract_magnitudes → peaks
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.21)

set(MODULE_NAME fm_correlator)

# ════════════════════════════════════════════════════════════════════════════
# Check ROCm dependency
# ════════════════════════════════════════════════════════════════════════════

if(NOT ROCM_ENABLED)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (ROCm disabled)")
    return()
endif()

message(STATUS "Configuring module: ${MODULE_NAME} (ROCm)")

# ════════════════════════════════════════════════════════════════════════════
# hipFFT (required)
# ════════════════════════════════════════════════════════════════════════════

find_package(hipfft QUIET)
if(NOT hipfft_FOUND)
    message(FATAL_ERROR "  ${MODULE_NAME}: hipfft NOT FOUND — install hipfft-dev")
endif()
message(STATUS "  ${MODULE_NAME}: hipFFT FOUND")

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC
    src/fm_correlator.cpp
    src/fm_correlator_processor_rocm.cpp
)

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
        hip::hipfft
)

target_compile_definitions(${MODULE_NAME} PUBLIC ENABLE_ROCM=1)
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

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
message(STATUS "    - Pipeline: cyclic_shifts → C2C FFT → R2C FFT → multiply_conj → C2R IFFT → extract")
message(STATUS "    - Kernels: hiprtc (apply_cyclic_shifts, multiply_conj_fused, extract_magnitudes, generate_test_inputs)")
