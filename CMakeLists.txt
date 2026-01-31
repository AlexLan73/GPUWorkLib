# ============================================================================
# LCH-Farrow01: GPU-Accelerated Radar Signal Generator
# Main CMakeLists.txt - ГЛАВНЫЙ КОРНЕВОЙ ФАЙЛ
# ============================================================================
# НАЗНАЧЕНИЕ: Только подключение подмодулей и вывод информации
# Все настройки ОТДЕЛЕНЫ в отдельные файлы!
# ============================================================================

cmake_minimum_required(VERSION 3.20)

project(GPUWorkLib VERSION 1.0.0 LANGUAGES CXX)

# VS CODE INTELLISENSE FIX
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# INCLUDE МОДУЛИ КОНФИГУРАЦИИ
# ============================================================================
message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║  GPUWorkLib: GPU-Accelerated Library                          ║")
message(STATUS "║  Loading configuration modules...                             ║")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "")

# 1. Определение платформы (OS detection)
include(cmake/platform-detection.cmake)

# 2. Конфигурация GPU (CUDA/OpenCL выбор)
include(cmake/gpu-config.cmake)

# 3. Поиск и конфигурация зависимостей (dependencies)
include(cmake/dependencies.cmake)

# 4. Опции компилятора и оптимизации
include(cmake/compiler-options.cmake)

# 5. Отладочная информация
include(cmake/debug-config.cmake)

# ============================================================================
# ВЫВОД ИТОГОВОЙ КОНФИГУРАЦИИ
# ============================================================================
message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║           CONFIGURATION SUMMARY                               ║")
message(STATUS "╠════════════════════════════════════════════════════════════════╣")
message(STATUS "║ Platform: ${PLATFORM_NAME}")
message(STATUS "║ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "║ C++ Standard: C${CMAKE_CXX_STANDARD}")
message(STATUS "║ Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "╠════════════════════════════════════════════════════════════════╣")
message(STATUS "║ CUDA Support: ${CUDA_ENABLED}")
message(STATUS "║ OpenCL Support: ${OPENCL_ENABLED}")
message(STATUS "║ clFFT Support: ${CLFFT_FOUND}")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "")

# ============================================================================
# БИБЛИОТЕКА DrvGPU
# ============================================================================
# Добавляем DrvGPU как подмодуль
add_subdirectory(include/DrvGPU)

# ============================================================================
# ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ============================================================================
add_subdirectory(src)

message(STATUS "")
message(STATUS "✅ Project configured successfully!")
message(STATUS "")
