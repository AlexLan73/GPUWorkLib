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
#if ENABLE_ROCM
#include "test_fft_processor_rocm.hpp"
#endif

namespace fft_processor_all_test {

inline void run() {
    // FFTProcessor: Complex, MagPhase режимы
    test_fft_processor::run();

    // FFTProcessor vs CPU reference (pocketfft)
    test_fft_vs_cpu::run();

    // FFTProcessorROCm: hipFFT-based FFT (ROCm only)
#if ENABLE_ROCM
    test_fft_processor_rocm::run();
#endif
}

}  // namespace fft_processor_all_test
