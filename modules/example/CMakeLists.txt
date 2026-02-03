# ════════════════════════════════════════════════════════════════════════════
# modules/vector_ops/CMakeLists.txt
# Vector Operations Module - Примитивные операции с векторами на GPU
# ════════════════════════════════════════════════════════════════════════════

project(VectorOpsModule VERSION 1.0.0 LANGUAGES CXX)

message(STATUS "[VectorOps] Configuring Vector Operations Module")

# ════════════════════════════════════════════════════════════════════════════
# Исходные файлы
# ════════════════════════════════════════════════════════════════════════════

set(VECTOR_OPS_HEADERS
    include/vector_ops_module.hpp
)

set(VECTOR_OPS_SOURCES
    src/vector_ops_module-part1.cpp
    src/vector_ops_module-part2.cpp
    src/vector_ops_module-part3.cpp
)

set(VECTOR_OPS_KERNELS
    kernels/vector_ops.cl
)

# ════════════════════════════════════════════════════════════════════════════
# Создание библиотеки модуля
# ════════════════════════════════════════════════════════════════════════════

add_library(vector_ops_module STATIC
    ${VECTOR_OPS_HEADERS}
    ${VECTOR_OPS_SOURCES}
)

# ════════════════════════════════════════════════════════════════════════════
# Alias для удобного использования
# ════════════════════════════════════════════════════════════════════════════

add_library(DrvGPU::VectorOps ALIAS vector_ops_module)

# ════════════════════════════════════════════════════════════════════════════
# Настройка include директорий
# ════════════════════════════════════════════════════════════════════════════

target_include_directories(vector_ops_module
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ════════════════════════════════════════════════════════════════════════════
# Линковка с основной библиотекой DrvGPU
# ════════════════════════════════════════════════════════════════════════════

target_link_libraries(vector_ops_module
    PUBLIC
        drvgpu
)

# ════════════════════════════════════════════════════════════════════════════
# C++ стандарт
# ════════════════════════════════════════════════════════════════════════════

target_compile_features(vector_ops_module PUBLIC cxx_std_17)

# ════════════════════════════════════════════════════════════════════════════
# Копирование OpenCL kernels в build директорию
# ════════════════════════════════════════════════════════════════════════════

# Создаём директорию для kernels в build
set(KERNELS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/modules/example/kernels)
file(MAKE_DIRECTORY ${KERNELS_OUTPUT_DIR})

# Копируем все .cl файлы
foreach(KERNEL_FILE ${VECTOR_OPS_KERNELS})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_FILE}
        ${KERNELS_OUTPUT_DIR}/${KERNEL_FILE}
        COPYONLY
    )
    message(STATUS "[VectorOps] Kernel copied: ${KERNEL_FILE}")
endforeach()

# ════════════════════════════════════════════════════════════════════════════
# Определение пути к kernels (для runtime загрузки)
# ════════════════════════════════════════════════════════════════════════════

# Путь к kernels в build директории (для запуска из build)
target_compile_definitions(vector_ops_module PRIVATE
    VECTOR_OPS_KERNELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
)

message(STATUS "[VectorOps] Vector Operations Module configured ✅")
