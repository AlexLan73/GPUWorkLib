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

# Main executable (src/)
add_subdirectory(src)

message(STATUS "✅ GPUWorkLib configured successfully!")
message(STATUS "")
