# ════════════════════════════════════════════════════════════════════════════
# Capon Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# ROCm-only модуль: алгоритм Кейпона (MVDR beamformer) для GPU.
#
# Алгоритм:
#   R = (1/N) * Y * Y^H + μI     — rocBLAS CGEMM + HIP kernel (диаг. загрузка)
#   R^{-1}                        — rocSOLVER POTRF + POTRI (Cholesky inversion)
#   z[m] = 1/(u^H * R^{-1} * u)  — rocBLAS CGEMM + HIP kernel (рельеф)
#   Y_out = (R^{-1}*U)^H * Y     — rocBLAS CGEMM (адаптивное ДО)
#
# Зависимости: rocBLAS (GEMM), rocSOLVER (POTRF+POTRI), hiprtc (custom kernels)
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME capon)

# ════════════════════════════════════════════════════════════════════════════
# Проверка зависимостей
# ════════════════════════════════════════════════════════════════════════════

if(NOT ROCM_ENABLED)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (ROCm disabled)")
    return()
endif()

if(NOT rocblas_FOUND)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (rocBLAS not found)")
    return()
endif()

if(NOT rocsolver_FOUND)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (rocSOLVER not found)")
    return()
endif()

message(STATUS "Configuring module: ${MODULE_NAME} (ROCm + rocBLAS + rocSOLVER + hiprtc)")

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC
    src/capon_processor.cpp   # g++: host logic + hiprtc kernels
)

# Required for linking into a shared library (.so / Python .pyd)
set_property(TARGET ${MODULE_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(${MODULE_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
        ${CMAKE_SOURCE_DIR}/modules/vector_algebra/include  # CholeskyInverterROCm
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
        drvgpu
        vector_algebra   # CholeskyInverterROCm — инверсия ковариационной матрицы
        hip::host
        roc::rocblas
        roc::rocsolver
)

# hiprtc для компиляции GPU kernels в runtime
if(hiprtc_FOUND)
    target_link_libraries(${MODULE_NAME} PRIVATE hiprtc::hiprtc)
else()
    message(WARNING "  ${MODULE_NAME}: hiprtc not found — GPU kernels will fail at runtime")
endif()

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

target_compile_definitions(${MODULE_NAME} PUBLIC ENABLE_ROCM=1)

# Alias
add_library(GPUWorkLib::${MODULE_NAME} ALIAS ${MODULE_NAME})

# ════════════════════════════════════════════════════════════════════════════
# Install
# ════════════════════════════════════════════════════════════════════════════

install(TARGETS ${MODULE_NAME}
    EXPORT GPUWorkLibTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/${MODULE_NAME}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

message(STATUS "  ${MODULE_NAME}: configured (ROCm only)")
message(STATUS "    - CaponProcessor: ComputeRelief + AdaptiveBeamform")
message(STATUS "    - CovarianceMatrixOp: rocBLAS CGEMM (R = Y*Y^H/N + muI)")
message(STATUS "    - CaponInvertOp: rocSOLVER POTRF+POTRI (Cholesky inversion)")
message(STATUS "    - CaponReliefOp: z[m] = 1/Re(u^H * R^{-1} * u)")
message(STATUS "    - AdaptBeamformOp: Y_out = (R^{-1}*U)^H * Y")
message(STATUS "    - Input: complex<float>, P channels x N samples x M directions")
