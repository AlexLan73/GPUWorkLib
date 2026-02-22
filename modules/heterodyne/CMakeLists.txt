# ════════════════════════════════════════════════════════════════════════════
# Heterodyne Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# LFM Dechirp processing on GPU (OpenCL)
# Pipeline: conj(LFM) -> multiply -> FFT -> peak -> range
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME heterodyne)

message(STATUS "Configuring module: ${MODULE_NAME}")

# ════════════════════════════════════════════════════════════════════════════
# Dependencies
# ════════════════════════════════════════════════════════════════════════════

find_package(OpenCL REQUIRED)

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC
    src/heterodyne_dechirp.cpp
    src/heterodyne_processor_opencl.cpp
    src/heterodyne_processor_rocm.cpp
)

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
        signal_generators
        spectrum_maxima
        OpenCL::OpenCL
)

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

# ════════════════════════════════════════════════════════════════════════════
# Kernel paths
# ════════════════════════════════════════════════════════════════════════════

set(HETERODYNE_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl")

target_compile_definitions(${MODULE_NAME} PRIVATE
    HETERODYNE_KERNELS_DIR="${HETERODYNE_KERNELS_DIR}"
)

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
message(STATUS "    - Kernels dir: ${HETERODYNE_KERNELS_DIR}")
message(STATUS "    - Pipeline: LfmConjugate -> Dechirp -> FFT -> Peak -> Range")
