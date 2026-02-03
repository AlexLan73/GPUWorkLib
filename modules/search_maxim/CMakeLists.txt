# ════════════════════════════════════════════════════════════════════════════
# modules/search_maxim/CMakeLists.txt
# Search FFT Module - FFT обработка с поиском максимумов
# ════════════════════════════════════════════════════════════════════════════

project(SearchModule VERSION 1.0.0 LANGUAGES CXX)

message(STATUS "[Search] Configuring Search FFT Module")

# ════════════════════════════════════════════════════════════════════════════
# clFFT (из родительского проекта или найти)
# ════════════════════════════════════════════════════════════════════════════
if(NOT CLFFT_FOUND)
    find_library(CLFFT_LIB NAMES clFFT PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
    find_path(CLFFT_INCLUDE_DIR NAMES clFFT.h PATHS /usr/include /usr/local/include
        ${CMAKE_SOURCE_DIR}/clFFT/include)
    if(CLFFT_LIB AND CLFFT_INCLUDE_DIR)
        set(CLFFT_FOUND TRUE)
    endif()
endif()
if(NOT CLFFT_FOUND)
    message(FATAL_ERROR "clFFT required for search_maxim module. Install: sudo apt install libclfft-dev")
endif()

# ════════════════════════════════════════════════════════════════════════════
# Исходные файлы
# ════════════════════════════════════════════════════════════════════════════

set(SEARCH_HEADERS
    include/search_3_module.hpp
    include/search_3_params.hpp
    include/search_3_result.hpp
)

set(SEARCH_SOURCES
    src/search_3_module-part1.cpp
    src/search_3_module-part2.cpp
    src/search_3_module-part3.cpp
    src/search_3_module-part4.cpp
)

set(SEARCH_KERNELS
    kernels/search_3_fft.cl
)

# ════════════════════════════════════════════════════════════════════════════
# Создание библиотеки модуля
# ════════════════════════════════════════════════════════════════════════════

add_library(search_3_module STATIC
    ${SEARCH_HEADERS}
    ${SEARCH_SOURCES}
)

# ════════════════════════════════════════════════════════════════════════════
# Alias для удобного использования
# ════════════════════════════════════════════════════════════════════════════

add_library(DrvGPU::Search ALIAS search_3_module)

# ════════════════════════════════════════════════════════════════════════════
# Настройка include директорий
# ════════════════════════════════════════════════════════════════════════════

target_include_directories(search_3_module
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ════════════════════════════════════════════════════════════════════════════
# Линковка с основной библиотекой DrvGPU и clFFT
# ════════════════════════════════════════════════════════════════════════════

target_link_libraries(search_3_module
    PUBLIC
        drvgpu
        ${CLFFT_LIB}
)
target_include_directories(search_3_module PRIVATE ${CLFFT_INCLUDE_DIR})

# ════════════════════════════════════════════════════════════════════════════
# C++ стандарт
# ════════════════════════════════════════════════════════════════════════════

target_compile_features(search_3_module PUBLIC cxx_std_17)

# ════════════════════════════════════════════════════════════════════════════
# Копирование OpenCL kernels в build директорию
# ════════════════════════════════════════════════════════════════════════════

set(KERNELS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/modules/search_maxim/kernels)
file(MAKE_DIRECTORY ${KERNELS_OUTPUT_DIR})

foreach(KERNEL_FILE ${SEARCH_KERNELS})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_FILE}
        ${KERNELS_OUTPUT_DIR}/${KERNEL_FILE}
        COPYONLY
    )
    message(STATUS "[Search] Kernel copied: ${KERNEL_FILE}")
endforeach()

# ════════════════════════════════════════════════════════════════════════════
# Определение пути к kernels
# ════════════════════════════════════════════════════════════════════════════

target_compile_definitions(search_3_module PRIVATE
    SEARCH_3_KERNELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
)

message(STATUS "[Search] Search FFT Module configured ✅")
