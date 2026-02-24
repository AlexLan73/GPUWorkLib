# ════════════════════════════════════════════════════════════════════════════
# Statistics Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# ROCm-only module: mean, median, variance, std for complex signal data.
#
# Sources:
#   statistics_processor.cpp  — compiled by g++ (host logic, hiprtc kernels)
#   statistics_sort_gpu.hip   — compiled by HIP/clang++ (rocprim segmented sort)
#
# Median uses rocprim::segmented_radix_sort_keys:
#   - All 256 beams sorted in ONE GPU call (fully parallel)
#   - ~30ms for 256×1.3M floats on RDNA4 vs ~2000ms CPU sort
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
# HIP language (needed to compile statistics_sort_gpu.hip with clang++)
# ════════════════════════════════════════════════════════════════════════════

enable_language(HIP)

# ════════════════════════════════════════════════════════════════════════════
# rocPRIM (header-only, required for segmented sort in .hip file)
# ════════════════════════════════════════════════════════════════════════════

find_package(rocprim QUIET)
if(rocprim_FOUND)
    message(STATUS "  ${MODULE_NAME}: rocPRIM FOUND — GPU median enabled")
else()
    message(FATAL_ERROR "  ${MODULE_NAME}: rocPRIM NOT FOUND — install rocprim-dev")
endif()

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

# Mark the .hip file explicitly so CMake uses the HIP compiler for it
set_source_files_properties(
    src/statistics_sort_gpu.hip
    PROPERTIES LANGUAGE HIP
)

add_library(${MODULE_NAME} STATIC
    src/statistics_processor.cpp   # g++: host logic + hiprtc kernels
    src/statistics_sort_gpu.hip    # HIP: rocprim segmented sort
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
        roc::rocprim
)

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
message(STATUS "    - Median: rocprim::segmented_radix_sort_keys (GPU, all beams parallel)")
message(STATUS "    - Input: complex<float>, per-beam")
