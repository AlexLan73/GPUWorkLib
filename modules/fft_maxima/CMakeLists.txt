# ════════════════════════════════════════════════════════════════════════════
# FFT Maxima Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# SpectrumMaximaFinder — поиск максимумов в FFT-спектре
# (Legacy antenna_fft_core/release удалён 2026-02-18)
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.27)

set(MODULE_NAME fft_maxima)

message(STATUS "Configuring module: ${MODULE_NAME}")

# ════════════════════════════════════════════════════════════════════════════
# Find dependencies
# ════════════════════════════════════════════════════════════════════════════

find_package(OpenCL REQUIRED)

if(NOT CLFFT_FOUND)
    message(FATAL_ERROR "clFFT not found! Please check cmake/dependencies.cmake")
endif()
message(STATUS "clFFT found: ${CLFFT_INCLUDE_DIR}")

# ════════════════════════════════════════════════════════════════════════════
# SpectrumMaximaFinder library
# ════════════════════════════════════════════════════════════════════════════

set(SPECTRUM_MAXIMA_SOURCES
    src/spectrum_maxima_finder.cpp
    src/spectrum_maxima_finder_process.cpp
    src/spectrum_maxima_finder_all_maxima.cpp
    src/spectrum_processor_factory.cpp
    src/spectrum_processor_opencl.cpp
    src/all_maxima_pipeline_opencl.cpp
)

add_library(spectrum_maxima STATIC ${SPECTRUM_MAXIMA_SOURCES})

target_include_directories(spectrum_maxima
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
    PRIVATE
        ${OpenCL_INCLUDE_DIRS}
        ${CLFFT_INCLUDE_DIR}
)

target_link_libraries(spectrum_maxima
    PRIVATE
        drvgpu
        OpenCL::OpenCL
        ${CLFFT_LIB}
)

set_target_properties(spectrum_maxima PROPERTIES
    LINK_INTERFACE_MULTIPLICITY 2
)

target_compile_features(spectrum_maxima PUBLIC cxx_std_17)

add_library(GPUWorkLib::spectrum_maxima ALIAS spectrum_maxima)

# Alias for backward compatibility (some CMake may reference fft_maxima)
add_library(GPUWorkLib::${MODULE_NAME} ALIAS spectrum_maxima)

# ════════════════════════════════════════════════════════════════════════════
# ROCm / hipFFT Support (conditional)
# ════════════════════════════════════════════════════════════════════════════

if(ROCM_ENABLED AND HIPFFT_FOUND)
    target_sources(spectrum_maxima PRIVATE
        src/spectrum_processor_rocm.cpp
        src/all_maxima_pipeline_rocm.cpp
    )
    target_link_libraries(spectrum_maxima PUBLIC
        hip::hipfft
    )
    message(STATUS "  ${MODULE_NAME}: ROCm/hipFFT ENABLED")
else()
    message(STATUS "  ${MODULE_NAME}: ROCm/hipFFT disabled")
endif()

# ════════════════════════════════════════════════════════════════════════════
# Install rules
# ════════════════════════════════════════════════════════════════════════════

install(TARGETS spectrum_maxima
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
# Summary
# ════════════════════════════════════════════════════════════════════════════

message(STATUS "  ${MODULE_NAME}: configured (spectrum_maxima)")
