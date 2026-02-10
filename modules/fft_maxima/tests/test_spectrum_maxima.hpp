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
#include "DrvGPU/services/gpu_profiler.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <map>
#define _USE_MATH_DEFINES  // ✅ Windows: для M_PI
#include <cmath>
#include <string>
#include <thread>
#include <chrono>

#include "modules/fft_maxima/tests/cpu_fft_reference.hpp"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_spectrum_maxima {

using namespace antenna_fft;
using namespace drv_gpu_lib;

/**
 * @enum TestMode
 * @brief Режим тестирования: 1 или 2 пика
 */
enum class TestMode {
    ONE_PEAK,   ///< Тестировать OnePeak кернел (4 MaxValue на луч)
    TWO_PEAKS   ///< Тестировать TwoPeaks кернел (8 MaxValue на луч)
};

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
        float freq = 2.5f *(1.0f + (antenna + 1)/10.0f);
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
        result.frequency =  2.5f *(1.0f + (antenna + 1)/10.0f);
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
// Конвертация vector<SpectrumResult> → map<int, CPUSpectrumResult> для валидации
// Layout вектора: [left0, right0, left1, right1, ...] — 2 элемента на антену
// ════════════════════════════════════════════════════════════════════════════

inline std::map<int, CPUSpectrumResult> VectorToMapForValidation(
    const std::vector<SpectrumResult>& vec)
{
    std::map<int, CPUSpectrumResult> result;
    for (size_t i = 0; i + 1 < vec.size(); i += 2) {
        int ant_id = static_cast<int>(i / 2);
        result[ant_id].SpectrMax_left = vec[i];
        result[ant_id].SpectrMax_right = vec[i + 1];
    }
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Проверка результатов (используем SpectrMax_left — тестовые частоты в положительном диапазоне)
// ════════════════════════════════════════════════════════════════════════════

inline bool ValidateResults(
    const std::map<int, CPUSpectrumResult>& results,
    const std::vector<ExpectedResult>& expected,
    const SpectrumParams& /*params*/) {

    std::cout << "🔍 ПРОВЕРКА РЕЗУЛЬТАТОВ:\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    bool all_passed = true;

    for (size_t i = 0; i < expected.size(); ++i) {
        auto it = results.find(static_cast<int>(i));
        if (it == results.end()) {
            std::cout << "  Антена " << i << ": результат отсутствует ❌\n";
            all_passed = false;
            continue;
        }
        const SpectrumResult& result = it->second.SpectrMax_left;  // тестовые частоты в левом диапазоне
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
// Проверка CPU vs GPU (Ref02: 8 на 8 — left с left, right с right)
// ════════════════════════════════════════════════════════════════════════════

inline bool ValidateResultsCPUvsGPU(
    const std::map<int, CPUSpectrumResult>& gpu_results,
    const std::map<int, cpu_reference::CPUSpectrumResult>& cpu_results,
    float max_freq_tolerance = 0.5f,
    float max_bin_tolerance = 1.5f)
{
    std::cout << "🔍 ПРОВЕРКА CPU vs GPU (8 на 8):\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    bool all_passed = true;

    for (const auto& [ant_id, gpu_r] : gpu_results) {
        int i = ant_id;
        auto it = cpu_results.find(i);
        if (it == cpu_results.end()) {
            std::cout << "  Антена " << i << ": CPU результат отсутствует ❌\n";
            all_passed = false;
            continue;
        }

        const auto& cpu_r = it->second;

        // Left: GPU vs CPU
        float freq_err_L = std::abs(gpu_r.SpectrMax_left.interpolated.refined_frequency -
            cpu_r.SpectrMax_left.interpolated.refined_frequency);
        float bin_err_L = std::abs(static_cast<float>(gpu_r.SpectrMax_left.center_point.index) -
            static_cast<float>(cpu_r.SpectrMax_left.center_point.index));

        // Right: GPU vs CPU
        float freq_err_R = std::abs(gpu_r.SpectrMax_right.interpolated.refined_frequency -
            cpu_r.SpectrMax_right.interpolated.refined_frequency);
        float bin_err_R = std::abs(static_cast<float>(gpu_r.SpectrMax_right.center_point.index) -
            static_cast<float>(cpu_r.SpectrMax_right.center_point.index));

        bool ok_L = (freq_err_L < max_freq_tolerance && bin_err_L < max_bin_tolerance);
        bool ok_R = (freq_err_R < max_freq_tolerance && bin_err_R < max_bin_tolerance);
        bool passed = ok_L && ok_R;
        if (!passed) all_passed = false;

        std::cout << "\n  Антена " << i << ":\n";
        std::cout << "  ├─ Left:  GPU freq " << std::fixed << std::setprecision(4)
                  << gpu_r.SpectrMax_left.interpolated.refined_frequency << " vs CPU "
                  << cpu_r.SpectrMax_left.interpolated.refined_frequency << " Hz, "
                  << "ampl " << gpu_r.SpectrMax_left.center_point.magnitude << " vs "
                  << cpu_r.SpectrMax_left.center_point.magnitude
                  << (ok_L ? " ✅" : " ❌") << "\n";
        std::cout << "  ├─ Right: GPU freq " << gpu_r.SpectrMax_right.interpolated.refined_frequency
                  << " vs CPU " << cpu_r.SpectrMax_right.interpolated.refined_frequency << " Hz, "
                  << "ampl " << gpu_r.SpectrMax_right.center_point.magnitude << " vs "
                  << cpu_r.SpectrMax_right.center_point.magnitude
                  << (ok_R ? " ✅" : " ❌") << "\n";
        std::cout << "  └─ Статус: " << (passed ? "✅ PASS" : "❌ FAIL") << "\n";
    }

    std::cout << "\n════════════════════════════════════════════════════════════\n";
    std::cout << "  CPU vs GPU: " << (all_passed ? "✅ СОВПАДАЮТ!" : "❌ ЕСТЬ РАСХОЖДЕНИЯ!") << "\n";
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
// Конвертация для ONE_PEAK режима (1 результат на антену)
// ════════════════════════════════════════════════════════════════════════════

inline std::map<int, SpectrumResult> VectorToMapForOnePeak(
    const std::vector<SpectrumResult>& vec)
{
    std::map<int, SpectrumResult> result;
    for (size_t i = 0; i < vec.size(); ++i) {
        result[static_cast<int>(i)] = vec[i];
    }
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Валидация для ONE_PEAK: проверяем что найденный максимум корректен
// ════════════════════════════════════════════════════════════════════════════

inline bool ValidateOnePeakResults(
    const std::map<int, SpectrumResult>& results,
    const std::vector<ExpectedResult>& expected)
{
    std::cout << "🔍 ПРОВЕРКА РЕЗУЛЬТАТОВ (ONE_PEAK):\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    bool all_passed = true;

    for (size_t i = 0; i < expected.size(); ++i) {
        auto it = results.find(static_cast<int>(i));
        if (it == results.end()) {
            std::cout << "  Антена " << i << ": результат отсутствует ❌\n";
            all_passed = false;
            continue;
        }
        const SpectrumResult& result = it->second;
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
        std::cout << "  └─ Статус:             " << (passed ? "✅ PASS" : "❌ FAIL") << "\n";
    }

    std::cout << "\n════════════════════════════════════════════════════════════\n";
    std::cout << "  ИТОГО: " << (all_passed ? "✅ ВСЕ ТЕСТЫ ПРОШЛИ!" : "❌ ЕСТЬ ОШИБКИ!") << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n\n";

    return all_passed;
}

// ════════════════════════════════════════════════════════════════════════════
// Главная функция теста с переключателем режима
// ════════════════════════════════════════════════════════════════════════════

inline int run(TestMode mode = TestMode::TWO_PEAKS) {
    try {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        if (mode == TestMode::ONE_PEAK) {
            std::cout << "║     TEST: SpectrumMaximaFinder (ONE_PEAK mode)            ║\n";
        } else {
            std::cout << "║     TEST: SpectrumMaximaFinder (TWO_PEAKS mode)           ║\n";
        }
        std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

        // 1. Инициализация DrvGPU
        std::cout << "🔧 Инициализация DrvGPU...\n";
        DrvGPU gpu(BackendType::OPENCL, 0);
        gpu.Initialize();
        std::cout << "  ✅ GPU: " << gpu.GetDeviceName() << "\n\n";

        // 1a. Запуск GPUProfiler (если профилирование включено в конфиге)
        auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
        const bool is_prof = profiler.IsEnabled() && profiler.IsGPUEnabled(0);
        if (is_prof) {
            profiler.Start();
        }

        // 2. Параметры теста (по плану Pl1.md)
        SpectrumParams params;
        params.antenna_count = 5;
        params.n_point = 100000;
        params.repeat_count = 4;
        params.sample_rate = 1000.0f;

        // Установка режима поиска пиков
        params.peak_mode = (mode == TestMode::ONE_PEAK)
            ? PeakSearchMode::ONE_PEAK
            : PeakSearchMode::TWO_PEAKS;

        // 3. Создать и инициализировать SpectrumMaximaFinder
        SpectrumMaximaFinder finder(params, &gpu.GetBackend());
        finder.Initialize();
        finder.PrintInfo();

        // Получить обновлённые параметры (с вычисленным nFFT)
        params = finder.GetParams();

        // 4. Сгенерировать тестовые данные
        auto input_data = GenerateTestData(params);

        // 5. Обработка данных (GPU)
        std::cout << "🚀 Запуск обработки...\n";
        auto gpu_vector = finder.Process(input_data);
        std::cout << "  ✅ Обработка завершена!\n\n";

        // 6. Вывод профилирования
        if (is_prof) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            profiler.PrintReport();
        } else {
            PrintProfiling(finder.GetProfilingData());
        }

        // 7. Валидация результатов
        bool passed = false;

        if (mode == TestMode::ONE_PEAK) {
            // ONE_PEAK: один результат на антену
            auto gpu_map = VectorToMapForOnePeak(gpu_vector);
            auto expected = CalculateExpected(params);
            passed = ValidateOnePeakResults(gpu_map, expected);
        } else {
            // TWO_PEAKS: CPU референс + сравнение
            auto cpu_results = cpu_reference::NewProcessAllBeams_CPU(input_data, params);
            auto gpu_map = VectorToMapForValidation(gpu_vector);
            passed = ValidateResultsCPUvsGPU(gpu_map, cpu_results);
        }

        // 8. Финал
        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        if (passed) {
            std::cout << "║     ✅ ТЕСТ УСПЕШНО ПРОЙДЕН!                              ║\n";
        } else {
            std::cout << "║     ❌ ТЕСТ НЕ ПРОЙДЕН!                                   ║\n";
        }
        std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

        // 9. Остановка GPUProfiler
        if (is_prof) {
            profiler.Stop();
        }

        return passed ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ ОШИБКА: " << e.what() << "\n\n";
        return 1;
    }
}

// Перегрузка без аргументов — для обратной совместимости (TWO_PEAKS)
inline int run_two_peaks() {
    return run(TestMode::TWO_PEAKS);
}

// Запуск теста ONE_PEAK
inline int run_one_peak() {
    return run(TestMode::ONE_PEAK);
}

} // namespace test_spectrum_maxima
