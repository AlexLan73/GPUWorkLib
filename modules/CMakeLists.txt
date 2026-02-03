# ════════════════════════════════════════════════════════════════════════════
# modules/CMakeLists.txt
# Главный CMake для всех вычислительных модулей DrvGPU
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.27)

# ════════════════════════════════════════════════════════════════════════════
# Опции для включения/выключения модулей
# ════════════════════════════════════════════════════════════════════════════

option(DRVGPU_BUILD_MODULE_EXAMPLE "Build Vector Operations Module" ON)
option(DRVGPU_BUILD_MODULE_SEARCH "Search for maximum in spectrum" ON)
option(DRVGPU_BUILD_MODULE_MATRIX     "Build Matrix Module"            OFF)
option(DRVGPU_BUILD_MODULE_FFT        "Build FFT Module"               OFF)
option(DRVGPU_BUILD_MODULE_CONVOLUTION "Build Convolution Module"      OFF)
option(DRVGPU_BUILD_MODULE_SORT       "Build Sort Module"              OFF)

# ════════════════════════════════════════════════════════════════════════════
# Вспомогательная функция для добавления модулей
# ════════════════════════════════════════════════════════════════════════════

function(add_drvgpu_module MODULE_NAME MODULE_DIR)
    message(STATUS "[Modules] Adding module: ${MODULE_NAME}")
    add_subdirectory(${MODULE_DIR})
endfunction()

# ════════════════════════════════════════════════════════════════════════════
# Подключение модулей (по условиям)
# ════════════════════════════════════════════════════════════════════════════

message(STATUS "")
message(STATUS "════════════════════════════════════════════════════════════")
message(STATUS "DrvGPU Compute Modules Configuration")
message(STATUS "════════════════════════════════════════════════════════════")

if(DRVGPU_BUILD_MODULE_EXAMPLE)
    add_drvgpu_module("VectorOps" "${CMAKE_CURRENT_SOURCE_DIR}/example")
endif()

if(DRVGPU_BUILD_MODULE_SEARCH)
    add_drvgpu_module("SearchMaxim" "${CMAKE_CURRENT_SOURCE_DIR}/search_maxim")
endif()

if(DRVGPU_BUILD_MODULE_MATRIX)
    message(STATUS "[Modules] Matrix module is disabled (not implemented yet)")
endif()

if(DRVGPU_BUILD_MODULE_FFT)
    message(STATUS "[Modules] FFT module is disabled (not implemented yet)")
endif()

if(DRVGPU_BUILD_MODULE_CONVOLUTION)
    message(STATUS "[Modules] Convolution module is disabled (not implemented yet)")
endif()

if(DRVGPU_BUILD_MODULE_SORT)
    message(STATUS "[Modules] Sort module is disabled (not implemented yet)")
endif()

message(STATUS "════════════════════════════════════════════════════════════")
message(STATUS "")
