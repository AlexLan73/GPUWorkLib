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
#include "capon_benchmark.hpp"
#include "test_capon_benchmark_rocm.hpp"
#endif

namespace capon_all_test {

inline void run() {
#if ENABLE_ROCM
  test_capon_rocm::run();
  // Benchmark (запускается только при is_prof=true в configGPU.json):
  // test_capon_benchmark_rocm::run();
#endif
}

}  // namespace capon_all_test
