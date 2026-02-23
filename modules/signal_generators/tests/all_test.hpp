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

#include "test_signal_generators.hpp"
#include "test_form_signal.hpp"
#include "test_form_script.hpp"
#include "test_delayed_form_signal.hpp"
#include "test_lfm_analytical_delay.hpp"
#include "test_form_signal_rocm.hpp"

namespace signal_generators_all_test {

inline void run() {
    // Signal Generators: CW, LFM, Noise
    // test_signal_generators::run();

    // FormSignalGenerator: getX formula, multi-channel, noise
    test_form_signal::run();

    // FormScriptGenerator: DSL + on-disk kernel cache (Этап 2)
    test_form_script::run();

    // DelayedFormSignalGenerator: Farrow 48×5 fractional delay
    test_delayed_form_signal::run();

    // LfmGeneratorAnalyticalDelay: analytical LFM with per-antenna delay
    test_lfm_analytical_delay::run();

    // FormSignalGeneratorROCm: getX on HIP (Linux + AMD GPU only)
    // test_form_signal_rocm::run();
}

}  // namespace signal_generators_all_test
