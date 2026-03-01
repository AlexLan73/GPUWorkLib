# ════════════════════════════════════════════════════════════════════════════
# FFT Processor Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# FFT 1/n лучей с вариантами вывода (complex, magnitude+phase, freq)
# Использует clFFT с pre-callback для padding
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME fft_processor)

message(STATUS "Configuring module: ${MODULE_NAME}")

# ════════════════════════════════════════════════════════════════════════════
# Dependencies
# ════════════════════════════════════════════════════════════════════════════

find_package(OpenCL REQUIRED)

if(NOT CLFFT_FOUND)
    message(FATAL_ERROR "clFFT not found! Please check cmake/dependencies.cmake")
endif()

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC
    src/fft_processor.cpp
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
        OpenCL::OpenCL
        ${CLFFT_LIB}
)

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

# ════════════════════════════════════════════════════════════════════════════
# ROCm / hipFFT Support (conditional)
# ════════════════════════════════════════════════════════════════════════════

if(ROCM_ENABLED AND HIPFFT_FOUND)
    target_sources(${MODULE_NAME} PRIVATE
        src/fft_processor_rocm.cpp
        src/complex_to_mag_phase_rocm.cpp
    )
    if(hipfft_FOUND)
        target_link_libraries(${MODULE_NAME} PUBLIC hip::hipfft)
    else()
        # Fallback: manual link
        target_link_libraries(${MODULE_NAME} PUBLIC ${HIPFFT_LIB})
        target_include_directories(${MODULE_NAME} PRIVATE ${HIPFFT_INCLUDE_DIR})
    endif()
    message(STATUS "  ${MODULE_NAME}: ROCm/hipFFT ENABLED")
else()
    message(STATUS "  ${MODULE_NAME}: ROCm/hipFFT disabled")
endif()

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
message(STATUS "    - Modes: COMPLEX, MAGNITUDE_PHASE, MAGNITUDE_PHASE_FREQ")
