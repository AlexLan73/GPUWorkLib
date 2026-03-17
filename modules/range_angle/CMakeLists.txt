# ════════════════════════════════════════════════════════════════════════════
# range_angle Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# ROCm-only module: LFM dechirp → range FFT → transpose → beam FFT → peak search.
#
# Pipeline:
#   DechirpWindowOp  — multiply by conj(LFM ref) + Hann window (hiprtc)
#   RangeFftOp       — batched 1-D forward FFT (hipFFT)
#   TransposeOp      — [n_ant × n_range_bins] → [n_range_bins × n_ant] (hiprtc)
#   BeamFftOp        — 2-D spatial FFT + fftshift + |.|^2 (hipFFT + hiprtc)
#   PeakSearchOp     — GPU max reduction → TargetInfo list (hiprtc)
#
# Ref03 Architecture:
#   Layer 1: GpuContext (per-module stream + shared buffers)
#   Layer 2: IGpuOperation interface
#   Layer 3: GpuKernelOp base
#   Layer 4: BufferSet<N> (private per-Op buffers)
#   Layer 5: Concrete Ops in include/operations/
#   Layer 6: RangeAngleProcessor Facade
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME range_angle)

# ════════════════════════════════════════════════════════════════════════════
# Check ROCm dependency
# ════════════════════════════════════════════════════════════════════════════

if(NOT ROCM_ENABLED)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (ROCm disabled)")
    return()
endif()

message(STATUS "Configuring module: ${MODULE_NAME} (ROCm + hipFFT + hiprtc)")

# ════════════════════════════════════════════════════════════════════════════
# hipFFT dependency
# ════════════════════════════════════════════════════════════════════════════

find_package(hipfft QUIET HINTS ${ROCM_PATH}/lib/cmake/hipfft)
if(hipfft_FOUND)
    message(STATUS "  ${MODULE_NAME}: hipFFT FOUND")
else()
    message(FATAL_ERROR "  ${MODULE_NAME}: hipFFT NOT FOUND — install hipfft-dev (rocm-dev)")
endif()

# ════════════════════════════════════════════════════════════════════════════
# HIP language (needed for .hip stub files)
# ════════════════════════════════════════════════════════════════════════════

enable_language(HIP)

# ════════════════════════════════════════════════════════════════════════════
# Sources
# ════════════════════════════════════════════════════════════════════════════

# Host C++ sources
set(RANGE_ANGLE_SOURCES
    src/range_angle_processor.cpp
)

# HIP stub files (offline syntax reference only — actual kernels use hiprtc)
set(RANGE_ANGLE_HIP_SOURCES
    src/dechirp_window_kernel.hip
    src/transpose_kernel.hip
    src/fftshift2d_kernel.hip
)

set_source_files_properties(${RANGE_ANGLE_HIP_SOURCES} PROPERTIES LANGUAGE HIP)

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC
    ${RANGE_ANGLE_SOURCES}
    ${RANGE_ANGLE_HIP_SOURCES}
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
        hip::hipfft
)

# hiprtc for runtime kernel compilation
if(hiprtc_FOUND)
    target_link_libraries(${MODULE_NAME} PRIVATE hiprtc::hiprtc)
else()
    message(WARNING "  ${MODULE_NAME}: hiprtc not found — GPU kernels will fail at runtime")
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
message(STATUS "    - RangeAngleProcessor: Process() -> TargetInfo list")
message(STATUS "    - DechirpWindowOp: LFM dechirp + Hann window (hiprtc)")
message(STATUS "    - RangeFftOp: batched range FFT (hipFFT)")
message(STATUS "    - TransposeOp: [n_ant x n_rbins] -> [n_rbins x n_ant] (hiprtc)")
message(STATUS "    - BeamFftOp: 2-D spatial FFT + fftshift2d (hipFFT + hiprtc)")
message(STATUS "    - PeakSearchOp: GPU max reduction -> TargetInfo (hiprtc)")
message(STATUS "    - Status: STUB — all Execute() methods are TODO")
