# ════════════════════════════════════════════════════════════════════
# Упрощённый CMakeLists.txt для DrvGPU с новой структурой папок
# ════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.27)
project(DrvGPU VERSION 1.0.0 LANGUAGES CXX)

# ════════════════════════════════════════════════════════════════════
# Настройки
# ════════════════════════════════════════════════════════════════════

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ════════════════════════════════════════════════════════════════════
# Опции
# ════════════════════════════════════════════════════════════════════

option(BUILD_SHARED_LIBS "Build shared library" OFF)
option(BUILD_EXAMPLES "Build examples" ON)

# ════════════════════════════════════════════════════════════════════
# Найти OpenCL (используем переменные из основного проекта)
# ════════════════════════════════════════════════════════════════════

# Проверяем - если OpenCL уже найден в основном проекте, используем его
if(DEFINED OpenCL_INCLUDE_DIRS AND DEFINED OpenCL_LIBRARIES)
    message(STATUS "Using OpenCL from parent project")
    message(STATUS "   Include: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "   Library: ${OpenCL_LIBRARIES}")
    set(OPENCL_FOUND_PARENT TRUE)
else()
    # Иначе ищем сами
    message(STATUS "OpenCL not found in parent, searching...")
    find_package(OpenCL REQUIRED)
    set(OPENCL_FOUND_PARENT FALSE)
endif()

find_package(Threads REQUIRED)

# ════════════════════════════════════════════════════════════════════
# Исходники
# ════════════════════════════════════════════════════════════════════

# DrvGPU Core
set(DRVGPU_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/drv_gpu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/module_registry.cpp"
)

# Memory Module (абстрактная память)
set(DRVGPU_MEMORY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/memory/memory_manager.hpp"
)

# OpenCL Backend
set(DRVGPU_OPENCL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_core.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/command_queue_pool.cpp"
)



# ════════════════════════════════════════════════════════════════════
# Библиотека DrvGPU
# ════════════════════════════════════════════════════════════════════

add_library(drvgpu
    ${DRVGPU_CORE_SOURCES}
    ${DRVGPU_MEMORY_SOURCES}
    ${DRVGPU_OPENCL_SOURCES}
)

# Include директории
target_include_directories(drvgpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/memory
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl
        ${OpenCL_INCLUDE_DIRS}
)

# Линковка
if(OPENCL_FOUND_PARENT)
    target_link_libraries(drvgpu
        PUBLIC
            ${OpenCL_LIBRARIES}
            Threads::Threads
    )
else()
    target_link_libraries(drvgpu
        PUBLIC
            OpenCL::OpenCL
            Threads::Threads
    )
endif()

# Алиас
add_library(DrvGPU::drvgpu ALIAS drvgpu)

# ════════════════════════════════════════════════════════════════════
# Примеры
# ════════════════════════════════════════════════════════════════════

if(BUILD_EXAMPLES)
    # Single GPU пример
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/single_gpu.hpp")
        add_executable(example_single_gpu single_gpu.hpp)
        target_link_libraries(example_single_gpu PRIVATE DrvGPU::drvgpu)
    endif()
    
    # Multi-GPU пример
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu.hpp")
        add_executable(example_multi_gpu multi_gpu.hpp)
        target_link_libraries(example_multi_gpu PRIVATE DrvGPU::drvgpu)
    endif()
endif()

# ════════════════════════════════════════════════════════════════════
# Информация
# ════════════════════════════════════════════════════════════════════

message(STATUS "")
message(STATUS "═══════════════════════════════════════")
message(STATUS "DrvGPU Library Configuration")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared libs:    ${BUILD_SHARED_LIBS}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  OpenCL version: ${OpenCL_VERSION_STRING}")
message(STATUS "═══════════════════════════════════════")
message(STATUS "")
