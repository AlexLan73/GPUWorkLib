# ════════════════════════════════════════════════════════════════════════════
# FFT Maxima Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# Multi-GPU FFT processing with maxima search
# Uses clFFT library with callbacks for maximum performance
#
# Components:
#   - fft_maxima_core:    Core functionality (shared)
#   - fft_maxima_release: Release implementation (callbacks)
#   - fft_maxima:         Full library (both implementations)
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.16)

# Module name
set(MODULE_NAME fft_maxima)

message(STATUS "Configuring module: ${MODULE_NAME}")

# ════════════════════════════════════════════════════════════════════════════
# Find dependencies
# ════════════════════════════════════════════════════════════════════════════

# OpenCL
find_package(OpenCL REQUIRED)

# clFFT (используем переменные из cmake/dependencies.cmake)
if(NOT CLFFT_FOUND)
    message(FATAL_ERROR "clFFT not found! Please check cmake/dependencies.cmake")
endif()
message(STATUS "clFFT found: ${CLFFT_INCLUDE_DIR}")

# ════════════════════════════════════════════════════════════════════════════
# Source files
# ════════════════════════════════════════════════════════════════════════════

set(CORE_SOURCES
    src/antenna_fft_core.cpp
)

set(RELEASE_SOURCES
    src/antenna_fft_release.cpp
)

# Note: antenna_fft_debug.cpp removed - file does not exist
set(ALL_SOURCES
    ${CORE_SOURCES}
    ${RELEASE_SOURCES}
)

# Headers (for IDE integration)
set(HEADERS
    include/antenna_fft_core.h
    include/antenna_fft_release.h
    include/fft_logger.h
    include/fft_result_writer.hpp
    include/interface/antenna_fft_params.h
    include/kernels/fft_kernel_sources.hpp
)

# ════════════════════════════════════════════════════════════════════════════
# Core library (shared between Release and Debug)
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME}_core STATIC ${CORE_SOURCES})

target_include_directories(${MODULE_NAME}_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
    PRIVATE
        ${OpenCL_INCLUDE_DIRS}
        ${CLFFT_INCLUDE_DIR}
)

target_link_libraries(${MODULE_NAME}_core
    PUBLIC
        drvgpu
        OpenCL::OpenCL
        ${CLFFT_LIB}
)

target_compile_features(${MODULE_NAME}_core PUBLIC cxx_std_17)

# Header-only spdlog + fmt to avoid LNK2019 (inherited from drvgpu)
target_compile_definitions(${MODULE_NAME}_core PUBLIC SPDLOG_HEADER_ONLY FMT_HEADER_ONLY)

# ════════════════════════════════════════════════════════════════════════════
# Release library (callbacks - production)
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME}_release STATIC ${RELEASE_SOURCES})

target_include_directories(${MODULE_NAME}_release
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${OpenCL_INCLUDE_DIRS}
        ${CLFFT_INCLUDE_DIR}
)

target_link_libraries(${MODULE_NAME}_release
    PUBLIC
        ${MODULE_NAME}_core
    PRIVATE
        OpenCL::OpenCL
        ${CLFFT_LIB}
)

target_compile_features(${MODULE_NAME}_release PUBLIC cxx_std_17)

# Header-only spdlog + fmt to avoid LNK2019
target_compile_definitions(${MODULE_NAME}_release PUBLIC SPDLOG_HEADER_ONLY FMT_HEADER_ONLY)

# ════════════════════════════════════════════════════════════════════════════
# Full library (both implementations)
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC ${ALL_SOURCES})

target_include_directories(${MODULE_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
    PRIVATE
        ${OpenCL_INCLUDE_DIRS}
        ${CLFFT_INCLUDE_DIR}
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
        drvgpu
        OpenCL::OpenCL
        ${CLFFT_LIB}
)

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

# Header-only spdlog + fmt to avoid LNK2019
target_compile_definitions(${MODULE_NAME} PUBLIC SPDLOG_HEADER_ONLY FMT_HEADER_ONLY)

# ════════════════════════════════════════════════════════════════════════════
# Aliases for cleaner usage
# ════════════════════════════════════════════════════════════════════════════

add_library(GPUWorkLib::${MODULE_NAME} ALIAS ${MODULE_NAME})
add_library(GPUWorkLib::${MODULE_NAME}_core ALIAS ${MODULE_NAME}_core)
add_library(GPUWorkLib::${MODULE_NAME}_release ALIAS ${MODULE_NAME}_release)

# ════════════════════════════════════════════════════════════════════════════
# Install rules
# ════════════════════════════════════════════════════════════════════════════

install(TARGETS ${MODULE_NAME} ${MODULE_NAME}_core ${MODULE_NAME}_release
    EXPORT GPUWorkLibTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/fft_maxima
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

if(BUILD_TESTS)
    add_executable(test_${MODULE_NAME} tests/test_fft_maxima.hpp)

    target_link_libraries(test_${MODULE_NAME}
        PRIVATE
            ${MODULE_NAME}
            drv_gpu_opencl
            OpenCL::OpenCL
            ${CLFFT_LIB}
    )

    target_include_directories(test_${MODULE_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/include
    )

    # Add test to CTest
    add_test(NAME ${MODULE_NAME}_test COMMAND test_${MODULE_NAME})

    message(STATUS "    - Test:    test_${MODULE_NAME}")
endif()

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

message(STATUS "  ${MODULE_NAME}: configured")
message(STATUS "    - Core:    ${MODULE_NAME}_core (shared base)")
message(STATUS "    - Release: ${MODULE_NAME}_release (callbacks)")
message(STATUS "    - Full:    ${MODULE_NAME} (all)")
