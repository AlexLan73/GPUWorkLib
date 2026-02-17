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

namespace signal_generators_all_test {

inline void run() {
    // Signal Generators: CW, LFM, Noise
    // test_signal_generators::run();

    // FormSignalGenerator: getX formula, multi-channel, noise
    test_form_signal::run();

    // FormScriptGenerator: DSL + on-disk kernel cache (Этап 2)
    test_form_script::run();
}

}  // namespace signal_generators_all_test
