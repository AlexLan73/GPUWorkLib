# ════════════════════════════════════════════════════════════════════════════
# fft_func Module - CMakeLists.txt  [BRANCH: nvidia]
# ════════════════════════════════════════════════════════════════════════════
#
# Ветка: nvidia (Windows / NVIDIA GPU / OpenCL + clFFT)
#
# Реализации:
#   OpenCL/clFFT — FFTProcessor, SpectrumMaximaFinder
#   Без ROCm (только для AMD, который в ветке main)
#
# CMake-таргеты:
#   fft_func                    — основной
#   GPUWorkLib::fft_func        — алиас
#   GPUWorkLib::spectrum_maxima — алиас (backward compat)
#   GPUWorkLib::fft_processor   — алиас (backward compat)
#   GPUWorkLib::fft_maxima      — алиас (backward compat)
#
# Сборка: cmake --preset windows-nvidia
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME fft_func)

message(STATUS "Configuring module: ${MODULE_NAME} [nvidia branch - OpenCL/clFFT]")

# ════════════════════════════════════════════════════════════════════════════
# Зависимости
# ════════════════════════════════════════════════════════════════════════════

find_package(OpenCL REQUIRED)

if(NOT CLFFT_FOUND)
    message(FATAL_ERROR "${MODULE_NAME}: clFFT not found! Set CLFFT_DIR or CLFFT_LIB.")
endif()

# ════════════════════════════════════════════════════════════════════════════
# Sources — только OpenCL/clFFT
# ════════════════════════════════════════════════════════════════════════════

set(FFT_FUNC_SOURCES
    src/fft_processor.cpp
    src/spectrum_maxima_finder.cpp
    src/spectrum_maxima_finder_process.cpp
    src/spectrum_maxima_finder_all_maxima.cpp
    src/spectrum_processor_opencl.cpp
    src/all_maxima_pipeline_opencl.cpp
    src/spectrum_processor_factory.cpp
)

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC ${FFT_FUNC_SOURCES})

set_property(TARGET ${MODULE_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

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
target_compile_definitions(${MODULE_NAME} PUBLIC ENABLE_CLFFT=1)

set_target_properties(${MODULE_NAME} PROPERTIES
    LINK_INTERFACE_MULTIPLICITY 2
)

# ════════════════════════════════════════════════════════════════════════════
# Aliases (backward compatibility)
# ════════════════════════════════════════════════════════════════════════════

add_library(GPUWorkLib::${MODULE_NAME}      ALIAS ${MODULE_NAME})
add_library(GPUWorkLib::spectrum_maxima     ALIAS ${MODULE_NAME})
add_library(GPUWorkLib::fft_processor       ALIAS ${MODULE_NAME})
add_library(GPUWorkLib::fft_maxima          ALIAS ${MODULE_NAME})

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

message(STATUS "  ${MODULE_NAME}: configured [nvidia branch]")
message(STATUS "    - FFTProcessor (OpenCL/clFFT)")
message(STATUS "    - SpectrumMaximaFinder (OpenCL/clFFT)")
message(STATUS "    - SpectrumProcessorFactory (OpenCL)")
