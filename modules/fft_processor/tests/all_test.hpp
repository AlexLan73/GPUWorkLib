#pragma once

/**
 * @file all_test.hpp
 * @brief Перечень тестов модуля fft_processor
 *
 * main.cpp вызывает этот файл — НЕ отдельные тесты напрямую.
 * Включить/закомментировать нужные тесты здесь.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "test_fft_processor.hpp"
#include "test_fft_vs_cpu.hpp"
#include "test_fft_benchmark.hpp"
#if ENABLE_ROCM
#include "test_fft_processor_rocm.hpp"
#include "test_complex_to_mag_phase_rocm.hpp"
#include "test_fft_benchmark_rocm.hpp"
#include "test_fft_matrix_rocm.hpp"
#endif

namespace fft_processor_all_test {

inline void run() {
    // FFTProcessor (OpenCL/clFFT)
    test_fft_processor::run();
    test_fft_vs_cpu::run();
//    test_fft_benchmark::run();  // benchmark — долго, запускать отдельно

    // FFTProcessorROCm: hipFFT-based FFT (ROCm only)
#if ENABLE_ROCM
    test_fft_processor_rocm::run();
#endif

    // ComplexToMagPhaseROCm: direct complex->mag+phase (ROCm only)
#if ENABLE_ROCM
    test_complex_to_mag_phase_rocm::run();
#endif

    // FFTProcessorROCm Benchmark (GpuBenchmarkBase — hipFFT timing)
#if ENABLE_ROCM
//    test_fft_benchmark_rocm::run();
#endif

    // FFT Matrix Benchmark — beams × nFFT table (ROCm)
#if ENABLE_ROCM
    test_fft_matrix_rocm::run();
#endif
}

}  // namespace fft_processor_all_test
