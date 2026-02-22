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

message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════╗")
message(STATUS "║  GPUWorkLib - Multi-GPU Library (DrvGPU)                 ║")
message(STATUS "║  Platform: ${PLATFORM_NAME}  |  OpenCL: ${OPENCL_ENABLED}               ║")
message(STATUS "╚══════════════════════════════════════════════════════════╝")
message(STATUS "")

# DrvGPU library (корневая)
add_subdirectory(DrvGPU)

# FFT Maxima module
add_subdirectory(modules/fft_maxima)

# Signal Generators module
add_subdirectory(modules/signal_generators)

# FFT Processor module
add_subdirectory(modules/fft_processor)

# LCH Farrow module (standalone fractional delay processor)
add_subdirectory(modules/lch_farrow)

# Filters module (FIR + IIR GPU filters)
add_subdirectory(modules/filters)

# Heterodyne module (LFM dechirp processing)
add_subdirectory(modules/heterodyne)

# Main executable (src/)
add_subdirectory(src)

# Python bindings (optional)
option(BUILD_PYTHON "Build Python bindings (pybind11)" OFF)
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

message(STATUS "✅ GPUWorkLib configured successfully!")
message(STATUS "")
