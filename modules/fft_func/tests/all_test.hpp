#pragma once

/**
 * @file all_test.hpp
 * @brief Тесты модуля fft_func — ветка nvidia (OpenCL/clFFT)
 *
 * Ветка: nvidia (Windows / NVIDIA GPU)
 * Все тесты используют OpenCL + clFFT. ROCm тестов нет.
 *
 * main.cpp вызывает этот файл — НЕ отдельные тесты напрямую.
 */

// ─── OpenCL / clFFT тесты ────────────────────────────────────────────────────
#include "test_fft_processor.hpp"
#include "test_fft_vs_cpu.hpp"
#include "test_spectrum_maxima.hpp"
#include "test_fft_maxima_benchmark.hpp"
// #include "test_fft_benchmark.hpp"         // benchmark — долго
// #include "test_large_batch.hpp"            // требует много GPU памяти
// #include "test_batch_all_maxima.hpp"       // падает на gfx1201

// ─── AllMaxima pipeline (OpenCL) ─────────────────────────────────────────────
#include "test_find_all_maxima.hpp"
#include "test_gpu_generator_integration.hpp"

namespace fft_func_all_test {

inline void run() {
    // FFTProcessor (OpenCL/clFFT)
    test_fft_processor::run();
    test_fft_vs_cpu::run();
    // test_fft_benchmark::run();   // benchmark — долго, запускать отдельно

    // SpectrumMaximaFinder (OpenCL/clFFT)
    test_spectrum_maxima::run();
    // test_large_batch::run();
    // test_batch_all_maxima::run();
    // test_fft_maxima_benchmark::run();

    // FindAllMaxima: полный pipeline (OpenCL)
    test_find_all_maxima::run();

    // Интеграция CwGenerator → SpectrumMaximaFinder (GPU→GPU, OpenCL)
    test_gpu_generator_integration::run();
}

}  // namespace fft_func_all_test
