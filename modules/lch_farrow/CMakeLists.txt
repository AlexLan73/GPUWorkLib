# ════════════════════════════════════════════════════════════════════════════
# LCH Farrow Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# Standalone fractional delay processor (Lagrange 48x5)
# Applies per-antenna fractional delay to complex signal on GPU
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME lch_farrow)

message(STATUS "Configuring module: ${MODULE_NAME}")

# ════════════════════════════════════════════════════════════════════════════
# Dependencies
# ════════════════════════════════════════════════════════════════════════════

find_package(OpenCL REQUIRED)

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

set(MODULE_SOURCES src/lch_farrow.cpp)

# ROCm support (conditional)
if(ROCM_ENABLED)
    list(APPEND MODULE_SOURCES src/lch_farrow_rocm.cpp)
    message(STATUS "  ${MODULE_NAME}: ROCm support ENABLED")
endif()

add_library(${MODULE_NAME} STATIC ${MODULE_SOURCES})

target_include_directories(${MODULE_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
    PRIVATE
        ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
        drvgpu
        OpenCL::OpenCL
)

# ROCm linkage
if(ROCM_ENABLED)
    target_link_libraries(${MODULE_NAME} PUBLIC hip::host)
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

message(STATUS "  ${MODULE_NAME}: configured")
message(STATUS "    - Lagrange 48x5 fractional delay processor")
message(STATUS "    - GPUProfiler integration")
