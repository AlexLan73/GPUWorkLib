# ════════════════════════════════════════════════════════════════════
# Упрощённый CMakeLists.txt для DrvGPU с новой структурой папок
# ════════════════════════════════════════════════════════════════════

# DrvGPU as subdirectory - no project() to avoid subproject

# Политика: find_package использует <PACKAGENAME>_ROOT (spdlog_ROOT и т.д.)
# CMP0144: Используем нижний регистр для имени пакета в переменной _ROOT
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

# ════════════════════════════════════════════════════════════════════
# Настройки
# ════════════════════════════════════════════════════════════════════

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ════════════════════════════════════════════════════════════════════
# Опции
# ════════════════════════════════════════════════════════════════════

option(BUILD_SHARED_LIBS "Build shared library" OFF)
option(BUILD_EXAMPLES "Build standalone example executables" OFF)
option(BUILD_EXTERNAL_CONTEXT_SUPPORT "Build External Context support" ON)

# ════════════════════════════════════════════════════════════════════
# Найти OpenCL и spdlog
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

# Найти spdlog (из vcpkg или системы)
# spdlog_ROOT может быть установлен извне (например, из vcpkg)
# Используем spdlog_ROOT вместо SPDLOG_ROOT согласно политике CMP0144
if(DEFINED ENV{SPDLOG_ROOT})
    set(spdlog_ROOT "$ENV{SPDLOG_ROOT}")
    message(STATUS "Using spdlog from spdlog_ROOT: ${spdlog_ROOT}")
endif()

# spdlog: используем из parent или ищем
if(NOT spdlog_FOUND AND NOT TARGET spdlog::spdlog)
    find_package(spdlog CONFIG QUIET)
endif()
if(NOT spdlog_FOUND AND NOT TARGET spdlog::spdlog)
    message(STATUS "spdlog: using header-only mode (no link)")
endif()

find_package(Threads REQUIRED)

# ════════════════════════════════════════════════════════════════════
# Исходники
# ════════════════════════════════════════════════════════════════════

# DrvGPU Core
set(DRVGPU_CORE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/drv_gpu.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/module_registry.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/memory_manager.cpp"
)

# Memory Module (абстрактная память)
set(DRVGPU_MEMORY_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/memory_manager.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/gpu_buffer.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/i_memory_buffer.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/memory_type.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory/svm_buffer.hpp"
)
set(DRVGPU_MEMORY_HEADERS ${DRVGPU_MEMORY_SOURCES})

# OpenCL Backend (БАЗОВЫЙ)
set(DRVGPU_OPENCL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_core.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/command_queue_pool.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_backend.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_core.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/command_queue_pool.hpp"
)

set(DRVGPU_OPENCL_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_core.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_backend.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/command_queue_pool.hpp"
)

# OpenCL Backend EXTERNAL CONTEXT
if(BUILD_EXTERNAL_CONTEXT_SUPPORT)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_backend_external.cpp")
        set(DRVGPU_EXTERNAL_CONTEXT_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_backend_external.cpp"
        )
        set(DRVGPU_EXTERNAL_CONTEXT_HEADERS
            "${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_backend_external.hpp"
        )
    else()
        set(DRVGPU_EXTERNAL_CONTEXT_SOURCES "")
        set(DRVGPU_EXTERNAL_CONTEXT_HEADERS "")
        message(WARNING "BUILD_EXTERNAL_CONTEXT_SUPPORT is ON but opencl_backend_external.cpp not found")
    endif()
else()
    set(DRVGPU_EXTERNAL_CONTEXT_SOURCES "")
    set(DRVGPU_EXTERNAL_CONTEXT_HEADERS "")
endif()

# Common Headers
set(DRVGPU_COMMON_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/drv_gpu.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu_manager.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/module_registry.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/i_compute_module.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/i_backend.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/backend_type.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/gpu_device_info.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/load_balancing.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/logger_interface.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/config_logger.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/default_logger.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/logger.hpp"
)

# Logger Sources
set(DRVGPU_LOGGER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/common/config_logger.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/default_logger.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/logger.cpp"
)

# ════════════════════════════════════════════════════════════════════
# Библиотека DrvGPU
# ════════════════════════════════════════════════════════════════════

add_library(drvgpu
    # Core
    ${DRVGPU_CORE_SOURCES}
    
    # Logger
    ${DRVGPU_LOGGER_SOURCES}
    
    # OpenCL Backend
    ${DRVGPU_OPENCL_SOURCES}
    
    # External Context (if enabled)
    ${DRVGPU_EXTERNAL_CONTEXT_SOURCES}
    
    # Headers (для IDE visibility)
    ${DRVGPU_COMMON_HEADERS}
    ${DRVGPU_MEMORY_HEADERS}
    ${DRVGPU_OPENCL_HEADERS}
    ${DRVGPU_EXTERNAL_CONTEXT_HEADERS}
)

# Include директории
target_include_directories(drvgpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/memory
        ${CMAKE_CURRENT_SOURCE_DIR}/backends
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl
        ${OpenCL_INCLUDE_DIRS}
)

# Линковка spdlog — только header-only, чтобы избежать LNK2019 (несовместимость ABI с vcpkg spdlogd.lib)
if(TARGET spdlog::spdlog_header_only)
    target_link_libraries(drvgpu PUBLIC spdlog::spdlog_header_only)
    message(STATUS "Using spdlog: HEADER-ONLY (target)")
elseif(spdlog_FOUND AND TARGET spdlog::spdlog)
    # Заголовки от find_package, но НЕ линкуем spdlog::spdlog — только include + SPDLOG_HEADER_ONLY
    get_target_property(SPDLOG_INC spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(drvgpu PUBLIC ${SPDLOG_INC})
    message(STATUS "Using spdlog: HEADER-ONLY (no link, avoid ABI mismatch)")
elseif(spdlog_FOUND)
    target_include_directories(drvgpu PUBLIC "${spdlog_INCLUDE_DIR}")
    message(STATUS "Using spdlog: HEADER-ONLY (no link)")
elseif(DEFINED spdlog_ROOT)
    target_include_directories(drvgpu PUBLIC "${spdlog_ROOT}/include")
    message(STATUS "Using spdlog: HEADER-ONLY (${spdlog_ROOT})")
else()
    find_path(SPDLOG_INCLUDE_DIR spdlog/spdlog.h PATHS /usr/include /usr/local/include)
    if(SPDLOG_INCLUDE_DIR)
        target_include_directories(drvgpu PUBLIC "${SPDLOG_INCLUDE_DIR}")
        message(STATUS "Using spdlog: system headers (header-only)")
    else()
        message(WARNING "spdlog not found! Install: sudo apt install libspdlog-dev")
    endif()
endif()

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

# Compile definitions
if(BUILD_EXTERNAL_CONTEXT_SUPPORT)
    target_compile_definitions(drvgpu PUBLIC DRVGPU_EXTERNAL_CONTEXT_SUPPORT)
endif()

# Header-only spdlog + fmt to avoid LNK2019 (vcpkg libs use dllimport/другая ABI)
target_compile_definitions(drvgpu PUBLIC SPDLOG_HEADER_ONLY FMT_HEADER_ONLY)

# Алиас
add_library(DrvGPU::drvgpu ALIAS drvgpu)

# ════════════════════════════════════════════════════════════════════
# Модули (VectorOps и др.) - в корне проекта
# ════════════════════════════════════════════════════════════════════
add_subdirectory(${CMAKE_SOURCE_DIR}/modules ${CMAKE_BINARY_DIR}/modules)

# ════════════════════════════════════════════════════════════════════
# Примеры
# ════════════════════════════════════════════════════════════════════

if(BUILD_EXAMPLES)
    # Single GPU пример (из DrvGPU/tests/)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/single_gpu.hpp")
        add_executable(example_single_gpu 
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/single_gpu.hpp"
        )
        target_link_libraries(example_single_gpu PRIVATE DrvGPU::drvgpu)
        target_include_directories(example_single_gpu PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}
        )
    endif()
    
    # Multi-GPU пример (из DrvGPU/tests/)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/multi_gpu.hpp")
        add_executable(example_multi_gpu 
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/multi_gpu.hpp"
        )
        target_link_libraries(example_multi_gpu PRIVATE DrvGPU::drvgpu)
        target_include_directories(example_multi_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
    endif()
    
    # External Context пример (из DrvGPU/tests/)
    if(BUILD_EXTERNAL_CONTEXT_SUPPORT)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/example_external_context_usage.hpp")
            add_executable(example_external_context 
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/example_external_context_usage.hpp"
            )
            target_link_libraries(example_external_context PRIVATE DrvGPU::drvgpu)
            target_include_directories(example_external_context PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
        endif()
    endif()
endif()

# ════════════════════════════════════════════════════════════════════
# Информация
# ════════════════════════════════════════════════════════════════════

message(STATUS "")
message(STATUS "═══════════════════════════════════════")
message(STATUS "DrvGPU Library Configuration")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  Version:             ${PROJECT_VERSION}")
message(STATUS "  Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared libs:         ${BUILD_SHARED_LIBS}")
message(STATUS "  Build examples:      ${BUILD_EXAMPLES}")
message(STATUS "  External Context:    ${BUILD_EXTERNAL_CONTEXT_SUPPORT}")
message(STATUS "  OpenCL version:      ${OpenCL_VERSION_STRING}")
message(STATUS "  OpenCL include:      ${OpenCL_INCLUDE_DIRS}")
message(STATUS "  OpenCL library:      ${OpenCL_LIBRARIES}")
if(spdlog_FOUND)
    message(STATUS "  spdlog:              FOUND")
else()
    message(STATUS "  spdlog:              HEADER-ONLY (${spdlog_ROOT})")
endif()
message(STATUS "═══════════════════════════════════════")
message(STATUS "")
