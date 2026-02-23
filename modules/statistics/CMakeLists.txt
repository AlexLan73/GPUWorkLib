# ════════════════════════════════════════════════════════════════════════════
# Statistics Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# ROCm-only module: mean, median, variance, std for complex signal data.
# Uses HIP kernels (hiprtc) and rocPRIM (radix sort for median).
#
# IMPORTANT: This module compiles ONLY when ROCM_ENABLED=TRUE.
# On Windows (no ROCm), this module is completely skipped.
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME statistics)

# ════════════════════════════════════════════════════════════════════════════
# Check ROCm dependency
# ════════════════════════════════════════════════════════════════════════════

if(NOT ROCM_ENABLED)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (ROCm disabled)")
    return()
endif()

message(STATUS "Configuring module: ${MODULE_NAME} (ROCm)")

# ════════════════════════════════════════════════════════════════════════════
# Find rocPRIM (header-only, part of ROCm SDK)
# ════════════════════════════════════════════════════════════════════════════

find_package(rocprim QUIET)
if(NOT rocprim_FOUND)
    message(WARNING "  ${MODULE_NAME}: rocPRIM not found! Median computation disabled.")
    message(STATUS "   Ubuntu: sudo apt install rocprim-dev")
endif()

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC
    src/statistics_processor.cpp
)

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
)

# rocPRIM (header-only)
if(rocprim_FOUND)
    target_link_libraries(${MODULE_NAME} PUBLIC roc::rocprim)
    target_compile_definitions(${MODULE_NAME} PRIVATE HAS_ROCPRIM=1)
    message(STATUS "  ${MODULE_NAME}: rocPRIM ENABLED")
else()
    message(STATUS "  ${MODULE_NAME}: rocPRIM disabled (median not available)")
endif()

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

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
message(STATUS "    - Operations: Mean, Median, Variance, STD (Welford)")
message(STATUS "    - Input: complex<float>, per-beam")
