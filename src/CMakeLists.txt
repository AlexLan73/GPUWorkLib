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

# clFFT (опционально - если не найден, продолжаем без него)
if(CLFFT_FOUND AND CLFFT_LIB)
    target_link_libraries(GPUWorkLib PRIVATE "${CLFFT_LIB}")
    message(STATUS "✅ Linked: clFFT")
else()
    message(STATUS "⏭️  clFFT not found - FFT support disabled")
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
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/interface
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)
#    ${CMAKE_SOURCE_DIR}/include/ManagerOpenCL
#    ${CMAKE_SOURCE_DIR}/include/GPU
#    ${CMAKE_SOURCE_DIR}/include/Test

# OpenCL includes
if(OPENCL_ENABLED)
    target_include_directories(GPUWorkLib PRIVATE ${OpenCL_INCLUDE_DIRS})
endif()

# clFFT includes
if(CLFFT_FOUND)
    target_include_directories(GPUWorkLib PRIVATE ${CLFFT_INCLUDE_DIR})
endif()

# ============================================================================
# COMPILER FLAGS
# ============================================================================

if(MSVC)
    target_compile_options(GPUWorkLib PRIVATE /W4 /arch:AVX2 /Gy)
else()
    target_compile_options(GPUWorkLib PRIVATE -Wall -Wextra -march=native)
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

if(CLFFT_FOUND)
    target_compile_definitions(GPUWorkLib PRIVATE
        CLFFT_VERSION=2
    )
endif()

# ============================================================================
# SET PROPERTIES
# ============================================================================

set_target_properties(GPUWorkLib PROPERTIES
    VERSION 1.0.0
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ============================================================================
# POST-BUILD: Копирование DLL на Windows (если есть)
# ============================================================================

if(IS_WINDOWS AND CLFFT_FOUND)
    # Если DLL уже найден в dependencies.cmake
    if(CLFFT_DLL)
        add_custom_command(TARGET GPUWorkLib POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CLFFT_DLL}"
                "$<TARGET_FILE_DIR:GPUWorkLib>"
            COMMENT "Copying clFFT.dll to executable directory"
        )
        message(STATUS "✅ Will copy clFFT.dll to output directory")
    else()
        # Fallback: ищем clFFT.dll рядом с lib или в clFFT/bin
        set(CLFFT_DLL_PATHS
            "${CMAKE_SOURCE_DIR}/clFFT/lib"
            "${CMAKE_SOURCE_DIR}/clFFT/bin"
            "${CMAKE_SOURCE_DIR}/clFFT"
        )
        find_file(CLFFT_DLL NAMES clFFT.dll PATHS ${CLFFT_DLL_PATHS} NO_DEFAULT_PATH)
        if(CLFFT_DLL)
            add_custom_command(TARGET GPUWorkLib POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CLFFT_DLL}"
                    "$<TARGET_FILE_DIR:GPUWorkLib>"
                COMMENT "Copying clFFT.dll to executable directory"
            )
            message(STATUS "✅ Will copy clFFT.dll: ${CLFFT_DLL}")
        else()
            message(WARNING "⚠️  clFFT.dll not found for copying")
        endif()
    endif()
endif()

message(STATUS "✅ Main executable configured: GPUWorkLib")
message(STATUS "   Output: ${CMAKE_BINARY_DIR}/GPUWorkLib")
message(STATUS "")
