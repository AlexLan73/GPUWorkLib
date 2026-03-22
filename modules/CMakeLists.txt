# ════════════════════════════════════════════════════════════════════════════
# modules/CMakeLists.txt
# Главный CMake для всех вычислительных модулей DrvGPU
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.27)

# ════════════════════════════════════════════════════════════════════════════
# Опции для включения/выключения модулей
# ════════════════════════════════════════════════════════════════════════════

# Legacy module options — all removed or migrated to dedicated CMakeLists.txt
# Modern modules (fft_func, statistics, etc.) are added from root CMakeLists.txt

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

# No legacy modules — all modern modules added from root CMakeLists.txt

message(STATUS "════════════════════════════════════════════════════════════")
message(STATUS "")
