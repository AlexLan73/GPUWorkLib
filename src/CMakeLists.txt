# ============================================================================
# Main CMakeLists for src/
# src/CMakeLists.txt
# ============================================================================
# НАЗНАЧЕНИЕ: Главная сборка (header-only модули)
# ============================================================================

message(STATUS "")
message(STATUS "🔧 Processing: src/ (Main build)")
message(STATUS "")

# ============================================================================
# ПОДМОДУЛИ (header-only)
# ============================================================================

# OpenCL Manager Library
#add_subdirectory(ManagerOpenCL)

# GPU Module (header-only)
#add_subdirectory(GPU)

# Tests Module (header-only)
#add_subdirectory(Test)

# ============================================================================
# ИСХОДНЫЕ ФАЙЛЫ ГЛАВНОГО ПРИЛОЖЕНИЯ
# ============================================================================

set(MAIN_SOURCES
    main.cpp
)

# ============================================================================
# СОЗДАНИЕ ИСПОЛНЯЕМОГО ФАЙЛА
# ============================================================================

add_executable(GPUWorkLib ${MAIN_SOURCES})

message(STATUS "🔨 Creating main executable: GPUWorkLib")

# ============================================================================
# ЛИНКОВКА С DrvGPU БИБЛИОТЕКОЙ
# ============================================================================

# Линкуем с DrvGPU библиотекой (если она существует как цель)
if(TARGET DrvGPU::drvgpu)
    target_link_libraries(GPUWorkLib PRIVATE DrvGPU::drvgpu)
    message(STATUS "✅ Linked: DrvGPU::drvgpu")
endif()

# VectorOps module (для test_vector_ops.hpp)
if(TARGET DrvGPU::VectorOps)
    target_link_libraries(GPUWorkLib PRIVATE DrvGPU::VectorOps)
    message(STATUS "✅ Linked: DrvGPU::VectorOps")
endif()

# Search3 module (для test_search_3)
if(TARGET DrvGPU::Search)
    target_link_libraries(GPUWorkLib PRIVATE DrvGPU::Search)
    message(STATUS "✅ Linked: DrvGPU::Search")
endif()

# Signal Generators module (для test_signal_generators)
if(TARGET GPUWorkLib::signal_generators)
    target_link_libraries(GPUWorkLib PRIVATE GPUWorkLib::signal_generators)
    message(STATUS "Linked: GPUWorkLib::signal_generators")
endif()

# FFT Functions module
if(TARGET GPUWorkLib::fft_func)
    target_link_libraries(GPUWorkLib PRIVATE GPUWorkLib::fft_func)
    message(STATUS "✅ Linked: GPUWorkLib::fft_func")
endif()

# LchFarrow module (для test_lch_farrow)
if(TARGET GPUWorkLib::lch_farrow)
    target_link_libraries(GPUWorkLib PRIVATE GPUWorkLib::lch_farrow)
    message(STATUS "✅ Linked: GPUWorkLib::lch_farrow")
endif()

# Filters module (для test_fir_basic, test_iir_basic)
if(TARGET GPUWorkLib::filters)
    target_link_libraries(GPUWorkLib PRIVATE GPUWorkLib::filters)
    message(STATUS "✅ Linked: GPUWorkLib::filters")
endif()

# Heterodyne module (для test_heterodyne)
if(TARGET GPUWorkLib::heterodyne)
    target_link_libraries(GPUWorkLib PRIVATE GPUWorkLib::heterodyne)
    message(STATUS "✅ Linked: GPUWorkLib::heterodyne")
endif()

# Statistics module (ROCm only: mean, median, variance, std)
if(TARGET GPUWorkLib::statistics)
    target_link_libraries(GPUWorkLib PRIVATE GPUWorkLib::statistics)
    message(STATUS "✅ Linked: GPUWorkLib::statistics")
endif()

# Vector Algebra module (ROCm only: Cholesky matrix inversion)
if(TARGET GPUWorkLib::vector_algebra)
    target_link_libraries(GPUWorkLib PRIVATE GPUWorkLib::vector_algebra)
    message(STATUS "✅ Linked: GPUWorkLib::vector_algebra")
endif()

# FM Correlator module (ROCm only: frequency-domain FM correlation)
if(TARGET GPUWorkLib::fm_correlator)
    target_link_libraries(GPUWorkLib PRIVATE GPUWorkLib::fm_correlator)
    message(STATUS "✅ Linked: GPUWorkLib::fm_correlator")
endif()

# Strategies module (ROCm only: antenna beamforming pipeline)
if(TARGET GPUWorkLib::strategies)
    target_link_libraries(GPUWorkLib PRIVATE GPUWorkLib::strategies)
    message(STATUS "✅ Linked: GPUWorkLib::strategies")
endif()

# spectrum_maxima alias теперь указывает на fft_func (backward compat)
# Линковка уже выполнена выше через GPUWorkLib::fft_func

# ============================================================================
# OpenCL - ПРАВИЛЬНЫЙ СПОСОБ через imported target
# ============================================================================
if(OPENCL_ENABLED)
    if(TARGET OpenCL::OpenCL)
        target_link_libraries(GPUWorkLib PRIVATE OpenCL::OpenCL)
        message(STATUS "✅ Linked: OpenCL::OpenCL (imported target)")
    elseif(OpenCL_LIBRARY)
        # Windows: используем найденную библиотеку
        target_link_libraries(GPUWorkLib PRIVATE "${OpenCL_LIBRARY}")
        message(STATUS "✅ Linked: ${OpenCL_LIBRARY}")
    elseif(OpenCL_LIBRARIES)
        # Fallback для старых версий CMake
        target_link_libraries(GPUWorkLib PRIVATE ${OpenCL_LIBRARIES})
        message(STATUS "✅ Linked: ${OpenCL_LIBRARIES}")
    else()
        message(WARNING "⚠️  OpenCL library not found, skipping link")
    endif()
endif()


# CUDA (опционально)
if(CUDA_FOUND)
    target_link_libraries(GPUWorkLib PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(GPUWorkLib PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(GPUWorkLib PRIVATE CUDA_ENABLED=1)
    message(STATUS "✅ Linked: CUDA")
endif()

# nlohmann_json
if(NLOHMANN_JSON_FOUND AND TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(GPUWorkLib PRIVATE nlohmann_json::nlohmann_json)
    message(STATUS "✅ Linked: nlohmann_json")
endif()

# ============================================================================
# ПОДКЛЮЧЕНИЕ INCLUDE ДИРЕКТОРИЙ
# ============================================================================

target_include_directories(GPUWorkLib PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/interface
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/DrvGPU
    ${CMAKE_SOURCE_DIR}/modules/example/include
    ${CMAKE_SOURCE_DIR}/modules/search_maxim/include
    ${CMAKE_SOURCE_DIR}/modules/fft_func/include
    ${CMAKE_SOURCE_DIR}/modules/fft_func/tests
    ${CMAKE_SOURCE_DIR}/modules/signal_generators/include
    ${CMAKE_SOURCE_DIR}/modules/lch_farrow/include
    ${CMAKE_SOURCE_DIR}/modules/filters/include
    ${CMAKE_SOURCE_DIR}/modules/heterodyne/include
    ${CMAKE_SOURCE_DIR}/modules/statistics/include
    ${CMAKE_SOURCE_DIR}/modules/vector_algebra/include
    ${CMAKE_SOURCE_DIR}/modules/fm_correlator/include
    ${CMAKE_SOURCE_DIR}/modules/strategies/include
)
#    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL
#    ${CMAKE_SOURCE_DIR}/include/GPU
#    ${CMAKE_SOURCE_DIR}/include/Test

# OpenCL includes
if(OPENCL_ENABLED)
    target_include_directories(GPUWorkLib PRIVATE ${OpenCL_INCLUDE_DIRS})
endif()

# ============================================================================
# COMPILER FLAGS
# ============================================================================

if(MSVC)
    target_compile_options(GPUWorkLib PRIVATE /W4 /arch:AVX2 /Gy)
else()
    target_compile_options(GPUWorkLib PRIVATE -Wall -Wextra -march=native)
endif()

# HIP fat binaries use absolute 32-bit relocations (R_X86_64_32) incompatible with PIE.
# Disable PIE when ROCm is enabled.
if(ROCM_ENABLED)
    target_link_options(GPUWorkLib PRIVATE -no-pie)
endif()

# ============================================================================
# COMPILE DEFINITIONS
# ============================================================================

if(OPENCL_ENABLED)
    target_compile_definitions(GPUWorkLib PRIVATE
        OPENCL_ENABLED=1
        CL_TARGET_OPENCL_VERSION=300
    )
endif()


# ============================================================================
# SET PROPERTIES
# ============================================================================

set_target_properties(GPUWorkLib PROPERTIES
    VERSION 1.0.0
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ============================================================================
# POST-BUILD: Копирование configGPU.json в каталог запуска (Ref03)
# ============================================================================
if(EXISTS "${CMAKE_SOURCE_DIR}/DrvGPU/config/configGPU.json")
    add_custom_command(TARGET GPUWorkLib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/DrvGPU/config/configGPU.json"
            "$<TARGET_FILE_DIR:GPUWorkLib>/configGPU.json"
        COMMENT "Copying configGPU.json to executable directory"
    )
    message(STATUS "✅ Will copy configGPU.json to output directory")
else()
    message(WARNING "⚠️  configGPU.json not found, will be created at runtime")
endif()


message(STATUS "✅ Main executable configured: GPUWorkLib")
message(STATUS "   Output: ${CMAKE_BINARY_DIR}/GPUWorkLib")
message(STATUS "")
