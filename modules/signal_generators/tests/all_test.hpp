#pragma once

/**
 * @file all_test.hpp
 * @brief Перечень тестов модуля signal_generators
 *
 * main.cpp вызывает этот файл — НЕ отдельные тесты напрямую.
 * Включить/закомментировать нужные тесты здесь.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-17
 */

#if ENABLE_ROCM
#include "test_signal_generators_rocm_basic.hpp"
#include "test_form_signal_rocm.hpp"
#include "test_signal_generators_benchmark_rocm.hpp"
#endif

namespace signal_generators_all_test {

inline void run() {
#if ENABLE_ROCM
    // CW, LFM, Noise, LfmConjugate: GPU vs CPU
    test_signal_generators_rocm_basic::run();

    // FormSignalGeneratorROCm: getX on HIP
    test_form_signal_rocm::run();

    // ROCm Benchmarks (GpuBenchmarkBase)
    //   test_signal_generators_benchmark_rocm::run();
#endif
}

}  // namespace signal_generators_all_test
