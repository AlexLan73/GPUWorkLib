# ============================================================================
# GPUWorkLib - Main CMakeLists.txt
# DrvGPU library + GPUWorkLib executable
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(GPUWorkLib VERSION 1.1.0 LANGUAGES CXX)

# Modules
include(cmake/platform-detection.cmake)
include(cmake/gpu-config.cmake)
include(cmake/dependencies.cmake)

# Required for linking static libs into Python shared module (.so)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════╗")
message(STATUS "║  GPUWorkLib - Multi-GPU Library (DrvGPU)                 ║")
message(STATUS "║  Platform: ${PLATFORM_NAME}  |  OpenCL: ${OPENCL_ENABLED}               ║")
message(STATUS "╚══════════════════════════════════════════════════════════╝")
message(STATUS "")

# DrvGPU library (корневая)
add_subdirectory(DrvGPU)

# FFT Functions module (FFTProcessor + SpectrumMaximaFinder)
add_subdirectory(modules/fft_func)

# Signal Generators module
add_subdirectory(modules/signal_generators)

# LCH Farrow module (standalone fractional delay processor)
add_subdirectory(modules/lch_farrow)

# Filters module (FIR + IIR GPU filters)
add_subdirectory(modules/filters)

# Heterodyne module (LFM dechirp processing)
add_subdirectory(modules/heterodyne)

# Statistics module (ROCm only: mean, median, variance, std)
add_subdirectory(modules/statistics)

# vector_algebra module (ROCm only: Cholesky matrix inversion via rocBLAS+rocSOLVER)
add_subdirectory(modules/vector_algebra)

# FM Correlator module (ROCm only: frequency-domain correlation for M-sequence FM)
add_subdirectory(modules/fm_correlator)

# Strategies module (ROCm only: antenna array processor pipeline)
add_subdirectory(modules/strategies)

# Capon module (ROCm only: MVDR beamformer — Capon spatial spectrum + adaptive beamforming)
add_subdirectory(modules/capon)

# RangeAngle module (ROCm only: LFM dechirp + range FFT + beam FFT + peak search)
add_subdirectory(modules/range_angle)

# Main executable (src/)
add_subdirectory(src)

# Python bindings (optional)
option(BUILD_PYTHON "Build Python bindings (pybind11)" OFF)
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

message(STATUS "✅ GPUWorkLib configured successfully!")
message(STATUS "")
