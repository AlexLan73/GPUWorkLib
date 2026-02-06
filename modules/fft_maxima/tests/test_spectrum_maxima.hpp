#pragma once
/**
 * @file test_spectrum_maxima.hpp
 * @brief Тест для SpectrumMaximaFinder с синусоидами
 *
 * Генерирует синусоиды для 5 антен, обрабатывает через FFT,
 * ищет максимум спектра и сравнивает с аналитическим расчётом.
 *
 * По плану: период = 1000/(2.5 * (номер_антены + 1))
 * Частоты: 2.5, 5.0, 7.5, 10.0, 12.5 Hz
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-06
 */

#include "spectrum_maxima_finder.h"
#include "drv_gpu.hpp"
#include "common/backend_type.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <string>

namespace test_spectrum_maxima {

using namespace antenna_fft;
using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════════════════
// Вспомогательные структуры
// ════════════════════════════════════════════════════════════════════════════

struct ExpectedResult {
    float frequency;        // Ожидаемая частота (Hz)
    float expected_bin;     // Ожидаемый bin (дробный)
    uint32_t nearest_bin;   // Ближайший целый bin
};

// ════════════════════════════════════════════════════════════════════════════
// Генерация тестовых данных
// ════════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateTestData(const SpectrumParams& params) {
    std::vector<std::complex<float>> data(params.antenna_count * params.n_point);

    std::cout << "\n📊 Генерация тестовых данных:\n";
    std::cout << "────────────────────────────────────────\n";

    for (uint32_t antenna = 0; antenna < params.antenna_count; ++antenna) {
        float freq = 2.5f * (antenna + 1);
        std::cout << "  Антена " << antenna << ": freq = " << freq << " Hz\n";

        for (uint32_t t = 0; t < params.n_point; ++t) {
            float phase = 2.0f * M_PI * freq * t / params.sample_rate;
            float value = std::sin(phase);
            size_t idx = antenna * params.n_point + t;
            data[idx] = std::complex<float>(value, 0.0f);
        }
    }

    std::cout << "────────────────────────────────────────\n";
    std::cout << "  Всего точек: " << data.size() << "\n\n";
    return data;
}

// ════════════════════════════════════════════════════════════════════════════
// Аналитический расчёт
// ════════════════════════════════════════════════════════════════════════════

inline std::vector<ExpectedResult> CalculateExpected(const SpectrumParams& params) {
    std::vector<ExpectedResult> expected;

    std::cout << "📐 Аналитический расчёт:\n";
    std::cout << "────────────────────────────────────────\n";
    std::cout << "  nFFT = " << params.nFFT << "\n";
    std::cout << "  sample_rate = " << params.sample_rate << " Hz\n";
    std::cout << "  bin_width = " << (params.sample_rate / params.nFFT) << " Hz\n\n";

    for (uint32_t antenna = 0; antenna < params.antenna_count; ++antenna) {
        ExpectedResult result;
        result.frequency = 2.5f * (antenna + 1);
        result.expected_bin = result.frequency * params.nFFT / params.sample_rate;
        result.nearest_bin = static_cast<uint32_t>(std::round(result.expected_bin));

        std::cout << "  Антена " << antenna << ":\n";
        std::cout << "    Частота: " << result.frequency << " Hz\n";
        std::cout << "    Ожидаемый bin: " << result.expected_bin << "\n";
        std::cout << "    Ближайший bin: " << result.nearest_bin << "\n";

        expected.push_back(result);
    }

    std::cout << "────────────────────────────────────────\n\n";
    return expected;
}

// ════════════════════════════════════════════════════════════════════════════
// Проверка результатов
// ════════════════════════════════════════════════════════════════════════════

inline bool ValidateResults(
    const std::vector<SpectrumResult>& results,
    const std::vector<ExpectedResult>& expected,
    const SpectrumParams& params) {

    std::cout << "🔍 ПРОВЕРКА РЕЗУЛЬТАТОВ:\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    bool all_passed = true;

    for (size_t i = 0; i < results.size(); ++i) {
        const SpectrumResult& result = results[i];
        const ExpectedResult& exp = expected[i];

        float bin_error = std::abs(static_cast<float>(result.center_point.index) - exp.expected_bin);
        float freq_error = std::abs(result.interpolated.refined_frequency - exp.frequency);

        bool bin_ok = (bin_error < 1.5f);
        bool freq_ok = (freq_error < 0.5f);
        bool passed = bin_ok && freq_ok;

        if (!passed) all_passed = false;

        std::cout << "\n  Антена " << i << ":\n";
        std::cout << "  ├─ Ожидаемая частота:  " << std::fixed << std::setprecision(2)
                  << exp.frequency << " Hz\n";
        std::cout << "  ├─ Найденная частота:  " << result.interpolated.refined_frequency << " Hz\n";
        std::cout << "  ├─ Ошибка частоты:     " << freq_error << " Hz "
                  << (freq_ok ? "✅" : "❌") << "\n";
        std::cout << "  ├─ Ожидаемый bin:      " << exp.expected_bin << "\n";
        std::cout << "  ├─ Найденный bin:      " << result.center_point.index << "\n";
        std::cout << "  ├─ Ошибка bin:         " << bin_error << " "
                  << (bin_ok ? "✅" : "❌") << "\n";
        std::cout << "  ├─ Magnitude:          " << result.center_point.magnitude << "\n";
        std::cout << "  ├─ freq_offset:        " << result.interpolated.freq_offset << "\n";
        std::cout << "  └─ Статус:             " << (passed ? "✅ PASS" : "❌ FAIL") << "\n";
    }

    std::cout << "\n════════════════════════════════════════════════════════════\n";
    std::cout << "  ИТОГО: " << (all_passed ? "✅ ВСЕ ТЕСТЫ ПРОШЛИ!" : "❌ ЕСТЬ ОШИБКИ!") << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n\n";

    return all_passed;
}

// ════════════════════════════════════════════════════════════════════════════
// Вывод профилирования
// ════════════════════════════════════════════════════════════════════════════

inline void PrintProfiling(const ProfilingData& profiling) {
    std::cout << "⏱️  GPU ПРОФИЛИРОВАНИЕ:\n";
    std::cout << "────────────────────────────────────────\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Upload (Host→GPU):       " << std::setw(8) << profiling.upload_time_ms << " ms\n";
    std::cout << "  FFT (with pre-callback): " << std::setw(8) << profiling.fft_time_ms << " ms\n";
    std::cout << "  Post-kernel:             " << std::setw(8) << profiling.post_kernel_time_ms << " ms\n";
    std::cout << "  Download (GPU→Host):     " << std::setw(8) << profiling.download_time_ms << " ms\n";
    std::cout << "────────────────────────────────────────\n";
    std::cout << "  TOTAL:                   " << std::setw(8) << profiling.total_time_ms << " ms\n";
    std::cout << "────────────────────────────────────────\n\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Главная функция теста
// ════════════════════════════════════════════════════════════════════════════

inline int run() {
    try {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║     TEST: SpectrumMaximaFinder с синусоидами             ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

        // 1. Инициализация DrvGPU
        std::cout << "🔧 Инициализация DrvGPU...\n";
        DrvGPU gpu(BackendType::OPENCL, 0);
        gpu.Initialize();
        std::cout << "  ✅ GPU: " << gpu.GetDeviceName() << "\n\n";

        // 2. Параметры теста (по плану Pl1.md)
        SpectrumParams params;
        params.antenna_count = 5;
        params.n_point = 1000;
        params.repeat_count = 2;
        params.sample_rate = 1000.0f;

        // 3. Создать и инициализировать SpectrumMaximaFinder
        SpectrumMaximaFinder finder(params, &gpu.GetBackend());
        finder.Initialize();
        finder.PrintInfo();

        // Получить обновлённые параметры (с вычисленным nFFT)
        params = finder.GetParams();

        // 4. Сгенерировать тестовые данные
        auto input_data = GenerateTestData(params);

        // 5. Аналитический расчёт ожидаемых результатов
        auto expected = CalculateExpected(params);

        // 6. Обработка данных
        std::cout << "🚀 Запуск обработки...\n";
        auto results = finder.Process(input_data);
        std::cout << "  ✅ Обработка завершена!\n\n";

        // 7. Вывод профилирования
        PrintProfiling(finder.GetProfilingData());

        // 8. Проверка результатов
        bool passed = ValidateResults(results, expected, params);

        // 9. Финал
        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        if (passed) {
            std::cout << "║     ✅ ТЕСТ УСПЕШНО ПРОЙДЕН!                              ║\n";
        } else {
            std::cout << "║     ❌ ТЕСТ НЕ ПРОЙДЕН!                                   ║\n";
        }
        std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

        return passed ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ ОШИБКА: " << e.what() << "\n\n";
        return 1;
    }
}

} // namespace test_spectrum_maxima
