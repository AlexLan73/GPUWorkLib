# ════════════════════════════════════════════════════════════════════════════
# Filters Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# GPU FIR & IIR filters for multi-channel parallel signal processing
#   - FIR: Direct-form convolution, __constant coefficients
#   - IIR: Biquad cascade, Direct Form II Transposed
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME filters)

message(STATUS "Configuring module: ${MODULE_NAME}")

# ════════════════════════════════════════════════════════════════════════════
# Dependencies
# ════════════════════════════════════════════════════════════════════════════

find_package(OpenCL REQUIRED)

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC
    src/fir_filter.cpp
    src/iir_filter.cpp
)

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
message(STATUS "    - FIR: direct-form, __constant coeffs, multi-channel parallel")
message(STATUS "    - IIR: biquad cascade, DFII-T, per-channel parallel")
message(STATUS "    - GPUProfiler integration")
