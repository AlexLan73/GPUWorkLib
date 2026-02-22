# ============================================================================
# python/CMakeLists.txt - Python bindings for GPUWorkLib (pybind11)
# ============================================================================
# Included from root CMakeLists.txt via add_subdirectory(python)
# Requires: -DBUILD_PYTHON=ON -DPython3_EXECUTABLE=path/to/python
# ============================================================================

message(STATUS "")
message(STATUS "Configuring Python bindings (pybind11)")

# ============================================================================
# Find Python + pybind11
# ============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

message(STATUS "  Python: ${Python3_EXECUTABLE} (${Python3_VERSION})")
message(STATUS "  NumPy:  ${Python3_NumPy_INCLUDE_DIRS}")

# pybind11 (installed via pip)
if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_RESULT
    )
    if(NOT pybind11_RESULT EQUAL 0)
        message(FATAL_ERROR "pybind11 not found. Install: pip install pybind11")
    endif()
    string(REPLACE "\"" "" _pybind11_cmakedir "${_pybind11_cmakedir}")
    set(pybind11_DIR "${_pybind11_cmakedir}" CACHE PATH "pybind11 cmake dir")
endif()

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "  pybind11: ${pybind11_VERSION}")

# ============================================================================
# pybind11 module
# ============================================================================

pybind11_add_module(gpuworklib
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_worklib_bindings.cpp
)

# OpenCL: prefer imported target, fallback to variable
if(TARGET OpenCL::OpenCL)
    set(_opencl_lib OpenCL::OpenCL)
elseif(OpenCL_LIBRARY)
    set(_opencl_lib "${OpenCL_LIBRARY}")
elseif(OpenCL_LIBRARIES)
    set(_opencl_lib "${OpenCL_LIBRARIES}")
else()
    find_package(OpenCL REQUIRED)
    set(_opencl_lib OpenCL::OpenCL)
endif()

target_link_libraries(gpuworklib PRIVATE
    drvgpu
    signal_generators
    fft_processor
    spectrum_maxima
    lch_farrow
    filters
    heterodyne
    ${_opencl_lib}
)

if(CLFFT_FOUND AND CLFFT_LIB)
    target_link_libraries(gpuworklib PRIVATE "${CLFFT_LIB}")
endif()

target_include_directories(gpuworklib PRIVATE
    ${CMAKE_SOURCE_DIR}/DrvGPU
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/interface
    ${CMAKE_SOURCE_DIR}/modules/signal_generators/include
    ${CMAKE_SOURCE_DIR}/modules/fft_processor/include
    ${CMAKE_SOURCE_DIR}/modules/fft_maxima/include
    ${CMAKE_SOURCE_DIR}/modules/lch_farrow/include
    ${CMAKE_SOURCE_DIR}/modules/filters/include
    ${CMAKE_SOURCE_DIR}/modules/heterodyne/include
    ${Python3_NumPy_INCLUDE_DIRS}
)

if(CLFFT_FOUND)
    target_include_directories(gpuworklib PRIVATE ${CLFFT_INCLUDE_DIR})
endif()

target_compile_definitions(gpuworklib PRIVATE
    OPENCL_ENABLED=1
    CL_TARGET_OPENCL_VERSION=300
)

if(CLFFT_FOUND)
    target_compile_definitions(gpuworklib PRIVATE CLFFT_VERSION=2)
endif()

if(MSVC)
    target_compile_options(gpuworklib PRIVATE /W3 /arch:AVX2 /wd4251)
endif()

set_target_properties(gpuworklib PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ============================================================================
# Copy clFFT.dll next to .pyd (Windows)
# ============================================================================

if(WIN32 AND CLFFT_FOUND AND CLFFT_DLL)
    add_custom_command(TARGET gpuworklib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CLFFT_DLL}" "$<TARGET_FILE_DIR:gpuworklib>"
        COMMENT "Copying clFFT.dll for Python module"
    )
endif()

message(STATUS "  gpuworklib Python module configured")
message(STATUS "")
