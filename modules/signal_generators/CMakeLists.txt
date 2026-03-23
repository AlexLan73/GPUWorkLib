# ════════════════════════════════════════════════════════════════════════════
# Signal Generators Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# CW, LFM, Noise генераторы сигналов на GPU (OpenCL)
# Strategy + Factory + DI via IBackend*
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME signal_generators)

message(STATUS "Configuring module: ${MODULE_NAME}")

# ════════════════════════════════════════════════════════════════════════════
# Dependencies
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# Library (ROCm-only: OpenCL generators removed from main branch)
# ════════════════════════════════════════════════════════════════════════════

if(NOT ROCM_ENABLED)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (ROCm disabled)")
    return()
endif()

add_library(${MODULE_NAME} STATIC
    src/signal_generator_factory.cpp
    src/cw_generator_rocm.cpp
    src/lfm_generator_rocm.cpp
    src/noise_generator_rocm.cpp
    src/lfm_conjugate_generator_rocm.cpp
    src/lfm_generator_analytical_delay_rocm.cpp
    src/form_signal_generator_rocm.cpp
    src/delayed_form_signal_generator_rocm.cpp
    src/script_generator_rocm.cpp
    src/form_script_generator_rocm.cpp
)

target_compile_definitions(${MODULE_NAME} PUBLIC ENABLE_ROCM=1)

target_include_directories(${MODULE_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
        drvgpu
        lch_farrow
        hip::host
)

# stdc++fs нужна только на GCC < 9 (не MSVC, не Clang с libc++)
if(NOT MSVC AND NOT APPLE)
    target_link_libraries(${MODULE_NAME} PRIVATE stdc++fs)
endif()

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

# ════════════════════════════════════════════════════════════════════════════
# Kernel cache paths (for on-disk kernel storage)
# ════════════════════════════════════════════════════════════════════════════

set(SIG_GEN_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")

target_compile_definitions(${MODULE_NAME} PRIVATE
    SIGNAL_GENERATORS_KERNELS_DIR="${SIG_GEN_KERNELS_DIR}"
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
message(STATUS "    - Generators: CW, LFM, Noise, Script, FormSignal, FormScript, DelayedFormSignal, LfmAnalyticalDelay, LfmConjugate")
message(STATUS "    - Kernels dir: ${SIG_GEN_KERNELS_DIR}")
message(STATUS "    - Factory + Service (DI via IBackend*)")
if(ROCM_ENABLED)
    message(STATUS "    - ROCm: FormSignalGeneratorROCm (HIP)")
endif()
