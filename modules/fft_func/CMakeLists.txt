# ════════════════════════════════════════════════════════════════════════════
# fft_func Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# Объединяет: fft_processor + fft_maxima
#
# Два бэкенда:
#   ROCm/hipFFT  — активен при ROCM_ENABLED=ON (AMD GPU, основной)
#   OpenCL/clFFT — активен при ENABLE_CLFFT=ON (NVIDIA, ветка opencl-clfft)
#
# CMake-таргеты:
#   fft_func                    — основной
#   GPUWorkLib::fft_func        — алиас
#   GPUWorkLib::spectrum_maxima — алиас (backward compat)
#   GPUWorkLib::fft_processor   — алиас (backward compat)
#   GPUWorkLib::fft_maxima      — алиас (backward compat)
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME fft_func)

message(STATUS "Configuring module: ${MODULE_NAME}")

# ════════════════════════════════════════════════════════════════════════════
# Options
# ════════════════════════════════════════════════════════════════════════════

option(ENABLE_CLFFT "Enable OpenCL/clFFT backend (disabled on AMD RDNA4+)" OFF)

# ════════════════════════════════════════════════════════════════════════════
# Sources
# ════════════════════════════════════════════════════════════════════════════

set(FFT_FUNC_SOURCES "")

# ─── OpenCL / clFFT sources (опциональные) ───────────────────────────────────
if(ENABLE_CLFFT)
    if(NOT CLFFT_FOUND)
        message(FATAL_ERROR "${MODULE_NAME}: ENABLE_CLFFT=ON but clFFT not found!")
    endif()
    find_package(OpenCL REQUIRED)
    list(APPEND FFT_FUNC_SOURCES
        src/fft_processor.cpp
        src/spectrum_maxima_finder.cpp
        src/spectrum_maxima_finder_process.cpp
        src/spectrum_maxima_finder_all_maxima.cpp
        src/spectrum_processor_opencl.cpp
        src/all_maxima_pipeline_opencl.cpp
        src/spectrum_processor_factory.cpp
    )
    message(STATUS "  ${MODULE_NAME}: OpenCL/clFFT ENABLED")
else()
    message(STATUS "  ${MODULE_NAME}: OpenCL/clFFT disabled (ENABLE_CLFFT=OFF)")
endif()

# ─── ROCm / hipFFT sources (основные) ────────────────────────────────────────
if(ROCM_ENABLED AND HIPFFT_FOUND)
    list(APPEND FFT_FUNC_SOURCES
        src/fft_processor_rocm.cpp
        src/complex_to_mag_phase_rocm.cpp
        src/spectrum_processor_rocm.cpp
        src/all_maxima_pipeline_rocm.cpp
    )
    # spectrum_processor_factory нужна для ROCm даже если clFFT выключен
    if(NOT ENABLE_CLFFT)
        list(APPEND FFT_FUNC_SOURCES
            src/spectrum_processor_factory.cpp
        )
    endif()
    message(STATUS "  ${MODULE_NAME}: ROCm/hipFFT ENABLED")
else()
    message(STATUS "  ${MODULE_NAME}: ROCm/hipFFT disabled")
endif()

if(NOT FFT_FUNC_SOURCES)
    message(FATAL_ERROR "${MODULE_NAME}: no sources! Enable ENABLE_CLFFT or ROCM_ENABLED.")
endif()

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
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
        drvgpu
)

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

# ─── clFFT linkage ───────────────────────────────────────────────────────────
if(ENABLE_CLFFT)
    target_include_directories(${MODULE_NAME} PRIVATE
        ${OpenCL_INCLUDE_DIRS}
        ${CLFFT_INCLUDE_DIR}
    )
    target_link_libraries(${MODULE_NAME} PUBLIC
        OpenCL::OpenCL
        ${CLFFT_LIB}
    )
    target_compile_definitions(${MODULE_NAME} PUBLIC ENABLE_CLFFT=1)
endif()

# ─── ROCm / hipFFT linkage ───────────────────────────────────────────────────
if(ROCM_ENABLED AND HIPFFT_FOUND)
    if(hipfft_FOUND)
        target_link_libraries(${MODULE_NAME} PUBLIC hip::hipfft)
    else()
        target_link_libraries(${MODULE_NAME} PUBLIC ${HIPFFT_LIB})
        target_include_directories(${MODULE_NAME} PRIVATE ${HIPFFT_INCLUDE_DIR})
    endif()
endif()

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

message(STATUS "  ${MODULE_NAME}: configured")
message(STATUS "    - FFTProcessorROCm, ComplexToMagPhaseROCm, SpectrumProcessorROCm")
message(STATUS "    - AllMaximaPipelineROCm, SpectrumProcessorFactory")
if(ENABLE_CLFFT)
    message(STATUS "    - FFTProcessor (OpenCL/clFFT), SpectrumMaximaFinder (OpenCL)")
endif()
