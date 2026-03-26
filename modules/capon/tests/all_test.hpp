#pragma once

/**
 * @file all_test.hpp
 * @brief Индексный файл тестов модуля capon
 *
 * main.cpp вызывает этот файл — НЕ отдельные тесты напрямую.
 * Включать/выключать тесты здесь.
 *
 * NOTE: Capon — ROCm-only модуль. Все тесты под #if ENABLE_ROCM.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM
#include "test_capon_rocm.hpp"
#include "test_capon_reference_data.hpp"
#include "test_capon_opencl_to_rocm.hpp"
#include "test_capon_hip_opencl_to_rocm.hpp"
#include "capon_benchmark.hpp"
#include "test_capon_benchmark_rocm.hpp"
#endif

namespace capon_all_test {

inline void run() {
#if ENABLE_ROCM
  test_capon_rocm::run();
  test_capon_reference_data::run();
  test_capon_opencl_to_rocm::run();         // OpenCL cl_mem → ZeroCopy → ROCm Capon
  test_capon_hip_opencl_to_rocm::run();     // hipMalloc → OpenCL writes → ROCm Capon
  // Benchmark (запускается только при is_prof=true в configGPU.json):
  // test_capon_benchmark_rocm::run();
#endif
}

}  // namespace capon_all_test
