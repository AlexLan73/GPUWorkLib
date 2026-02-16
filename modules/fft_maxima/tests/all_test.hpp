#pragma once

/**
 * @file all_test.hpp
 * @brief Перечень тестов модуля fft_maxima (SpectrumMaximaFinder)
 *
 * main.cpp вызывает этот файл — НЕ отдельные тесты напрямую.
 * Включить/закомментировать нужные тесты здесь.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "test_spectrum_maxima.hpp"
#include "test_large_batch.hpp"
#include "test_gpu_generator_integration.hpp"
#include "test_find_all_maxima.hpp"
#include "test_benchmark_all_maxima.hpp"
#include "test_batch_all_maxima.hpp"
// Legacy (antenna_fft) — вернуться отдельно
// #include "test_fft_maxima.hpp"
// #include "test_fft_svm.hpp"
// #include "test_external_context_fft.hpp"

namespace fft_maxima_all_test {

inline void run() {
    // Process: ONE_PEAK, TWO_PEAKS, batch
    // test_spectrum_maxima::run();
    // test_large_batch::run();  // НОВЫЙ API с batch processing

    // Интеграция CwGenerator → SpectrumMaximaFinder (GPU→GPU)
    // test_gpu_generator_integration::run();

    // FindAllMaxima: полный pipeline, AllMaxima
    // test_find_all_maxima::run();

    // BATCH: тесты для batch-обработки FindAllMaxima
    test_batch_all_maxima::run();

    // BENCHMARK: 10 лучей × 500k точек
    // test_benchmark_all_maxima::run();

    // Legacy (AntennaFFTCore)
    // test_fft_max::run();
    // test_fft_svm::run();
    // test_external_context_fft::run();
}

}  // namespace fft_maxima_all_test
