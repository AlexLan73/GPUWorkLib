#pragma once

/**
 * @file all_test.hpp
 * @brief Перечень тестов модуля fft_func (FFTProcessor + SpectrumMaximaFinder)
 *
 * Объединяет тесты из бывших модулей fft_processor и fft_maxima.
 * - OpenCL/clFFT тесты: компилируются и запускаются только при ENABLE_CLFFT=1
 * - ROCm/hipFFT тесты: компилируются и запускаются только при ENABLE_ROCM=1
 *
 * main.cpp вызывает этот файл — НЕ отдельные тесты напрямую.
 * Включить/закомментировать нужные тесты здесь.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-11
 */

// ─── OpenCL / clFFT тесты (только при ENABLE_CLFFT=1) ────────────────────────
#if ENABLE_CLFFT
#include "test_fft_processor.hpp"
#include "test_fft_vs_cpu.hpp"
#include "test_spectrum_maxima.hpp"
#include "test_fft_maxima_benchmark.hpp"
// #include "test_fft_benchmark.hpp"         // benchmark — долго
// #include "test_large_batch.hpp"            // требует много GPU памяти
// #include "test_batch_all_maxima.hpp"       // падает на gfx1201
#endif  // ENABLE_CLFFT

// ─── Тесты AllMaxima (через OpenCL SpectrumMaximaFinder) ──────────────────────
#if ENABLE_CLFFT
#include "test_find_all_maxima.hpp"
#include "test_gpu_generator_integration.hpp"
#endif

// ─── ROCm / hipFFT тесты (только при ENABLE_ROCM=1) ──────────────────────────
#if ENABLE_ROCM
#include "test_fft_processor_rocm.hpp"
#include "test_complex_to_mag_phase_rocm.hpp"
#include "test_fft_matrix_rocm.hpp"
#include "test_spectrum_maxima_rocm.hpp"
// #include "test_fft_benchmark_rocm.hpp"         // benchmark — долго
// #include "test_fft_maxima_benchmark_rocm.hpp"   // benchmark — долго
#endif  // ENABLE_ROCM

namespace fft_func_all_test {

inline void run() {
    // ─── OpenCL/clFFT тесты ───────────────────────────────────────────────
#if ENABLE_CLFFT
    // FFTProcessor (OpenCL/clFFT)
    test_fft_processor::run();
    test_fft_vs_cpu::run();
    // test_fft_benchmark::run();   // benchmark — долго, запускать отдельно

    // SpectrumMaximaFinder (OpenCL/clFFT)
    test_spectrum_maxima::run();
    // test_large_batch::run();
    // test_batch_all_maxima::run();
    // test_fft_maxima_benchmark::run();
#endif  // ENABLE_CLFFT

    // ─── Тесты AllMaxima через OpenCL SpectrumMaximaFinder ───────────────
#if ENABLE_CLFFT
    // FindAllMaxima: полный pipeline, AllMaxima (OpenCL)
    test_find_all_maxima::run();

    // Интеграция CwGenerator → SpectrumMaximaFinder (GPU→GPU, OpenCL)
    test_gpu_generator_integration::run();
#endif

    // ─── ROCm/hipFFT тесты ───────────────────────────────────────────────
#if ENABLE_ROCM
    // FFTProcessorROCm: hipFFT-based FFT
    test_fft_processor_rocm::run();

    // ComplexToMagPhaseROCm: direct complex->mag+phase
    test_complex_to_mag_phase_rocm::run();

    // FFT Matrix Benchmark — beams × nFFT table
    test_fft_matrix_rocm::run();

    // SpectrumMaximaFinder ROCm
    test_spectrum_maxima_rocm::run();

    // Benchmarks (раскомментировать для запуска):
    // test_fft_benchmark_rocm::run();
    // test_fft_maxima_benchmark_rocm::run();
#endif  // ENABLE_ROCM
}

}  // namespace fft_func_all_test
