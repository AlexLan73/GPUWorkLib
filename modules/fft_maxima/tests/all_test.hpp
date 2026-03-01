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
#include "test_spectrum_maxima_rocm.hpp"
#include "test_fft_maxima_benchmark.hpp"
#if ENABLE_ROCM
#include "test_fft_maxima_benchmark_rocm.hpp"
#endif

namespace fft_maxima_all_test {

inline void run() {
    // Process: ONE_PEAK, TWO_PEAKS, batch
    // test_spectrum_maxima::run();
    // test_large_batch::run();  // НОВЫЙ API с batch processing

    // Интеграция CwGenerator → SpectrumMaximaFinder (GPU→GPU)
    // test_gpu_generator_integration::run();

    // FindAllMaxima: полный pipeline, AllMaxima
    // test_find_all_maxima::run();

    // BATCH: тесты для batch-обработки FindAllMaxima (OpenCL = падает на gfx1201)
    // test_batch_all_maxima::run();

    // BENCHMARK: DEPRECATED (старый стиль) — заменён test_fft_maxima_benchmark
    // test_benchmark_all_maxima::run();

    // BENCHMARK: Process + FindAllMaxima (GpuBenchmarkBase, OpenCL)
    // test_fft_maxima_benchmark::run();

    // ROCm: SpectrumProcessorROCm тесты (запускать на Linux + AMD GPU)
#if ENABLE_ROCM
    test_spectrum_maxima_rocm::run();
#endif

    // ROCm: Benchmark (GpuBenchmarkBase) — раскомментировать для запуска:
#if ENABLE_ROCM
//  test_fft_maxima_benchmark_rocm::run();
#endif
}

}  // namespace fft_maxima_all_test
