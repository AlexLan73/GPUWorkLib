#pragma once

/**
 * @file test_fft_svm.hpp
 * @brief Тест FFT + поиск максимумов с CPU данными через SVM (пункт 6 плана)
 *
 * ============================================================================
 * СЦЕНАРИЙ ТЕСТА:
 *   1. Инициализация DrvGPU через GPUManager (Multi-GPU)
 *   2. Генерация тестового сигнала на CPU (синусоида с известной частотой + шум)
 *   3. Загрузка данных через SVM (SVMBuffer с zero-copy / coarse-grained)
 *   4. Обработка через AntennaFFTProcMax (FFT + clFFT callbacks + поиск максимумов)
 *   5. Проверка результатов (частота максимума совпадает с ожидаемой)
 *   6. Профилирование через GPUProfiler (async, отдельный поток)
 *
 * ПОТОК ДАННЫХ:
 *   CPU: generate_signal()
 *     |
 *     v
 *   SVMBuffer (SVM_COARSE_GRAIN): Write(signal_data) → Map/Unmap
 *     |
 *     v  (SVM → CPU → cl_mem через ProcessNew(vector))
 *   AntennaFFTProcMax::ProcessNew()
 *     |
 *     v
 *   pre-callback (padding) → clfftEnqueueTransform → post-callback (magnitude+select)
 *     |
 *     v
 *   AntennaFFTResult: max_values[0].index_point → сравнить с expected_bin
 *
 * FALLBACK:
 *   Если GPU не поддерживает SVM (OpenCL < 2.0), тест автоматически
 *   переключается на обычный cl_mem буфер (через ProcessNew(vector)).
 *
 * ЗАВИСИМОСТИ:
 *   - DrvGPU/gpu_manager.hpp       — инициализация Multi-GPU
 *   - DrvGPU/memory/svm_buffer.hpp — RAII обёртка SVM
 *   - DrvGPU/services/service_manager.hpp — управление сервисами
 *   - modules/fft_maxima/include/antenna_fft_release.h — FFT процессор
 *
 * ЗАПУСК:
 *   test_fft_svm::run()  — возвращает 0 при успехе, 1 при ошибке
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "DrvGPU/gpu_manager.hpp"
#include "DrvGPU/memory/svm_buffer.hpp"
#include "DrvGPU/services/service_manager.hpp"
#include "DrvGPU/services/gpu_profiler.hpp"
#include "DrvGPU/services/console_output.hpp"
#include "antenna_fft_release.h"

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>

namespace test_fft_svm {

// ============================================================================
// Константы теста
// ============================================================================

/// Количество лучей (beam_count)
static constexpr size_t TEST_BEAM_COUNT = 8;

/// Количество точек в каждом луче (count_points)
static constexpr size_t TEST_COUNT_POINTS = 512;

/// Количество выходных точек FFT (out_count_points_fft)
static constexpr size_t TEST_OUT_POINTS_FFT = 256;

/// Максимальное количество пиков для поиска
static constexpr size_t TEST_MAX_PEAKS = 3;

/// Амплитуда тестового сигнала
static constexpr float SIGNAL_AMPLITUDE = 10.0f;

/// Уровень шума (СКО гауссового распределения)
static constexpr float NOISE_LEVEL = 0.1f;

/// Допустимая ошибка при проверке частотного бина (±бинов)
static constexpr size_t FREQ_BIN_TOLERANCE = 2;

// ============================================================================
// Генерация тестового сигнала
// ============================================================================

/**
 * @brief Генерация тестового сигнала для одного луча
 *
 * Формула: signal[n] = A * exp(j * 2*pi * f_norm * n) + gaussian_noise
 *
 * f_norm = target_bin / count_points гарантирует, что пик FFT попадёт
 * точно в бин target_bin (без утечки спектра).
 *
 * @param count_points  Количество точек
 * @param target_bin    Целевой частотный бин (0 .. count_points-1)
 * @param amplitude     Амплитуда синусоиды
 * @param noise_std     СКО шума
 * @param seed          Random seed (для воспроизводимости)
 * @return Вектор complex<float> длиной count_points
 */
static std::vector<std::complex<float>> GenerateBeamSignal(
    size_t count_points,
    size_t target_bin,
    float amplitude = SIGNAL_AMPLITUDE,
    float noise_std = NOISE_LEVEL,
    unsigned seed = 42)
{
    std::vector<std::complex<float>> signal(count_points);

    // Нормализованная частота: точно попадёт в бин target_bin
    const float freq_norm = static_cast<float>(target_bin) / static_cast<float>(count_points);
    const float two_pi = 2.0f * 3.14159265358979f;

    // Генератор шума (Mersenne Twister + нормальное распределение)
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise_dist(0.0f, noise_std);

    for (size_t n = 0; n < count_points; ++n) {
        float phase = two_pi * freq_norm * static_cast<float>(n);
        float re = amplitude * std::cos(phase) + noise_dist(rng);
        float im = amplitude * std::sin(phase) + noise_dist(rng);
        signal[n] = std::complex<float>(re, im);
    }

    return signal;
}

/**
 * @brief Генерация данных для всех лучей
 *
 * Каждый луч получает синусоиду на уникальной частоте:
 *   Луч 0: бин 10,  Луч 1: бин 20,  Луч 2: бин 30, ...
 *   Формула: target_bin = beam_index * 10 + 10
 *
 * Данные выкладываются в один непрерывный массив:
 *   [beam0_pt0, beam0_pt1, ..., beam0_ptN, beam1_pt0, ...]
 *
 * @param beam_count    Количество лучей
 * @param count_points  Точек на один луч
 * @return Массив beam_count * count_points complex<float>
 */
static std::vector<std::complex<float>> GenerateAllBeamsData(
    size_t beam_count,
    size_t count_points)
{
    std::vector<std::complex<float>> all_data;
    all_data.reserve(beam_count * count_points);

    for (size_t beam = 0; beam < beam_count; ++beam) {
        size_t target_bin = beam * 10 + 10;

        auto signal = GenerateBeamSignal(
            count_points,
            target_bin,
            SIGNAL_AMPLITUDE,
            NOISE_LEVEL,
            static_cast<unsigned>(42 + beam)
        );

        all_data.insert(all_data.end(), signal.begin(), signal.end());
    }

    return all_data;
}

// ============================================================================
// Проверка результатов
// ============================================================================

/**
 * @brief Проверить результаты FFT: максимум должен быть в ожидаемом бине
 *
 * Для каждого луча:
 *   1. Ожидаемый бин = beam_index * 10 + 10
 *   2. Фактический бин = max_values[0].index_point
 *   3. |фактический - ожидаемый| <= tolerance → PASS
 *
 * @param result      Результат от AntennaFFTProcMax::ProcessNew()
 * @param beam_count  Количество лучей
 * @param tolerance   Допуск в бинах (по умолчанию ±2)
 * @return true если ВСЕ лучи прошли проверку
 */
static bool VerifyResults(
    const antenna_fft::AntennaFFTResult& result,
    size_t beam_count,
    size_t tolerance = FREQ_BIN_TOLERANCE)
{
    bool all_ok = true;

    // Проверка количества результатов
    if (result.results.size() != beam_count) {
        std::cerr << "  [FAIL] Expected " << beam_count
                  << " beam results, got " << result.results.size() << "\n";
        return false;
    }

    for (size_t beam = 0; beam < beam_count; ++beam) {
        const auto& br = result.results[beam];
        size_t expected_bin = beam * 10 + 10;

        // Проверка наличия максимумов
        if (br.max_values.empty()) {
            std::cerr << "  [FAIL] Beam " << beam << ": no max_values found\n";
            all_ok = false;
            continue;
        }

        // Главный максимум
        size_t actual_bin = br.max_values[0].index_point;
        float actual_amp = br.max_values[0].amplitude;
        int diff = static_cast<int>(actual_bin) - static_cast<int>(expected_bin);

        bool bin_ok = (std::abs(diff) <= static_cast<int>(tolerance));
        bool amp_ok = (actual_amp > 0.0f);

        if (bin_ok && amp_ok) {
            std::cout << "  [PASS] Beam " << std::setw(2) << beam
                      << ": expected_bin=" << std::setw(4) << expected_bin
                      << " actual_bin=" << std::setw(4) << actual_bin
                      << " amp=" << std::fixed << std::setprecision(2) << actual_amp
                      << " (diff=" << diff << ")\n";
        } else {
            std::cerr << "  [FAIL] Beam " << std::setw(2) << beam
                      << ": expected_bin=" << std::setw(4) << expected_bin
                      << " actual_bin=" << std::setw(4) << actual_bin
                      << " amp=" << actual_amp
                      << " (bin_ok=" << bin_ok << " amp_ok=" << amp_ok << ")\n";
            all_ok = false;
        }
    }

    return all_ok;
}

// ============================================================================
// ОСНОВНОЙ ТЕСТ
// ============================================================================

/**
 * @brief Запуск теста FFT + maxima с SVM данными
 * @return 0 при успехе, 1 при ошибке
 *
 * АЛГОРИТМ:
 *  ШАГ 1: GPUManager::InitializeAll() → обнаружение GPU
 *  ШАГ 2: ServiceManager → запуск Profiler, Console
 *  ШАГ 3: CPU → GenerateAllBeamsData()
 *  ШАГ 4: SVMBuffer(context, queue, data) → SVM загрузка
 *  ШАГ 5: AntennaFFTProcMax(params, backend) → FFT обработка
 *  ШАГ 6: VerifyResults() → проверка корректности
 *  ШАГ 7: GPUProfiler::PrintSummary() → профилирование
 */
inline int run() {
    std::cout << R"(
+====================================================================+
|                                                                    |
|     TEST: FFT + Maxima Search with CPU data via SVM                |
|                                                                    |
|     Pipeline:                                                      |
|     CPU -> SVMBuffer -> AntennaFFTProcMax -> Verify Results        |
|                                                                    |
|     Parameters:                                                    |
|       Beams:       )" << TEST_BEAM_COUNT << R"(                                            |
|       Points/beam: )" << TEST_COUNT_POINTS << R"(                                          |
|       Out FFT pts: )" << TEST_OUT_POINTS_FFT << R"(                                        |
|       Max peaks:   )" << TEST_MAX_PEAKS << R"(                                             |
|                                                                    |
+====================================================================+
)";

    try {
        // ══════════════════════════════════════════════════════════════
        // ШАГ 1: Инициализация GPU
        // ══════════════════════════════════════════════════════════════
        std::cout << "[STEP 1] Initializing GPU...\n";

        drv_gpu_lib::GPUManager manager;
        manager.InitializeAll(drv_gpu_lib::BackendType::OPENCL);

        std::cout << "   GPU count: " << manager.GetGPUCount() << "\n";
        manager.PrintAllDevices();

        // Берём первый GPU для теста
        auto& gpu = manager.GetGPU(0);
        auto& backend = gpu.GetBackend();
        std::cout << "   Using GPU 0: " << gpu.GetDeviceName() << "\n\n";

        // ══════════════════════════════════════════════════════════════
        // ШАГ 2: Запуск сервисов (Profiler, Console)
        // ══════════════════════════════════════════════════════════════
        std::cout << "[STEP 2] Starting services...\n";

        auto& sm = drv_gpu_lib::ServiceManager::GetInstance();
        sm.InitializeDefaults();
        sm.StartAll();
        std::cout << "   Services started (Console + Profiler)\n\n";

        // ══════════════════════════════════════════════════════════════
        // ШАГ 3: Генерация тестового сигнала на CPU
        // ══════════════════════════════════════════════════════════════
        std::cout << "[STEP 3] Generating test signal on CPU...\n";

        auto cpu_data = GenerateAllBeamsData(TEST_BEAM_COUNT, TEST_COUNT_POINTS);

        size_t data_bytes = cpu_data.size() * sizeof(std::complex<float>);
        std::cout << "   Total samples: " << cpu_data.size()
                  << " (" << (data_bytes / 1024.0) << " KB)\n";

        std::cout << "   Expected frequency bins: ";
        for (size_t b = 0; b < TEST_BEAM_COUNT; ++b) {
            std::cout << (b * 10 + 10);
            if (b < TEST_BEAM_COUNT - 1) std::cout << ", ";
        }
        std::cout << "\n\n";

        // ══════════════════════════════════════════════════════════════
        // ШАГ 4: Загрузка данных через SVM
        // ══════════════════════════════════════════════════════════════
        std::cout << "[STEP 4] Loading data via SVM...\n";

        // Получаем нативные OpenCL ресурсы
        cl_context context = static_cast<cl_context>(backend.GetNativeContext());
        cl_command_queue queue = static_cast<cl_command_queue>(backend.GetNativeQueue());

        auto t_svm_start = std::chrono::high_resolution_clock::now();

        // Проверяем поддержку SVM
        bool svm_supported = backend.SupportsSVM();
        std::cout << "   SVM supported: " << (svm_supported ? "YES" : "NO") << "\n";

        // Данные для обработки (из SVM или напрямую из CPU)
        std::vector<std::complex<float>> data_for_fft;

        if (svm_supported) {
            // ──────────────────────────────────────────────────────────
            // SVM PATH: CPU → SVMBuffer → Read → ProcessNew(vector)
            // ──────────────────────────────────────────────────────────
            std::cout << "   Creating SVMBuffer (COARSE_GRAIN)...\n";

            drv_gpu_lib::SVMBuffer svm_buffer(
                context,
                queue,
                cpu_data,  // Конструктор с данными: auto Write(data) внутри
                drv_gpu_lib::MemoryStrategy::SVM_COARSE_GRAIN,
                drv_gpu_lib::MemoryType::GPU_READ_WRITE
            );

            std::cout << "   SVM buffer: " << svm_buffer.GetNumElements()
                      << " elements (" << (svm_buffer.GetSizeBytes() / 1024.0) << " KB)\n";

            // Читаем из SVM обратно (демонстрация round-trip: CPU→SVM→CPU)
            data_for_fft = svm_buffer.Read();

            std::cout << "   SVM round-trip: CPU -> SVM -> CPU OK\n";
        } else {
            // ──────────────────────────────────────────────────────────
            // FALLBACK: Используем CPU данные напрямую
            // ──────────────────────────────────────────────────────────
            std::cout << "   FALLBACK: Using CPU data directly (no SVM)\n";
            data_for_fft = cpu_data;
        }

        auto t_svm_end = std::chrono::high_resolution_clock::now();
        double svm_time = std::chrono::duration<double, std::milli>(t_svm_end - t_svm_start).count();

        // Профилирование через async GPUProfiler (non-blocking Enqueue)
        drv_gpu_lib::GPUProfiler::GetInstance().Record(
            0, "TestSVM", "DataUpload", svm_time);

        std::cout << "   Upload time: " << std::fixed << std::setprecision(2)
                  << svm_time << " ms\n\n";

        // ══════════════════════════════════════════════════════════════
        // ШАГ 5: Обработка FFT через AntennaFFTProcMax
        // ══════════════════════════════════════════════════════════════
        std::cout << "[STEP 5] Running FFT processing...\n";

        // Параметры FFT
        antenna_fft::AntennaFFTParams params(
            TEST_BEAM_COUNT,
            TEST_COUNT_POINTS,
            TEST_OUT_POINTS_FFT,
            TEST_MAX_PEAKS,
            "test_svm",      // task_id
            "TestFFT_SVM"    // module_name
        );

        // Создаём процессор (внутри: создание FFT плана с callbacks)
        antenna_fft::AntennaFFTProcMax processor(params, &backend);

        std::cout << "   nFFT = " << processor.GetNFFT() << "\n";

        // Засекаем время и запускаем FFT
        auto t_fft_start = std::chrono::high_resolution_clock::now();

        // ProcessNew(vector) создаёт cl_mem буфер внутри, копирует данные,
        // выполняет FFT с callbacks, и возвращает результат
        auto result = processor.ProcessNew(data_for_fft);

        auto t_fft_end = std::chrono::high_resolution_clock::now();
        double fft_time = std::chrono::duration<double, std::milli>(t_fft_end - t_fft_start).count();

        // Профилирование
        drv_gpu_lib::GPUProfiler::GetInstance().Record(
            0, "TestSVM", "FFT_Total", fft_time);

        std::cout << "   FFT processing time: " << fft_time << " ms\n";
        std::cout << "   Batch mode used: "
                  << (processor.WasBatchModeUsed() ? "YES" : "NO") << "\n";
        std::cout << "   Total beams processed: " << result.total_beams << "\n\n";

        // ══════════════════════════════════════════════════════════════
        // ШАГ 6: Проверка результатов
        // ══════════════════════════════════════════════════════════════
        std::cout << "[STEP 6] Verifying results...\n";

        bool test_passed = VerifyResults(result, TEST_BEAM_COUNT);

        // ══════════════════════════════════════════════════════════════
        // ШАГ 7: Профилирование и вывод
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[STEP 7] Profiling summary...\n";

        sm.PrintProfilingSummary();

        // Останавливаем сервисы (drain queues, join threads)
        sm.StopAll();

        // ══════════════════════════════════════════════════════════════
        // Итоговый результат
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n" << std::string(60, '=') << "\n";
        if (test_passed) {
            std::cout << "  RESULT: PASSED - FFT + Maxima + SVM\n";
        } else {
            std::cout << "  RESULT: FAILED - check details above\n";
        }
        std::cout << std::string(60, '=') << "\n\n";

        return test_passed ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: " << e.what() << "\n";
        try { drv_gpu_lib::ServiceManager::GetInstance().StopAll(); } catch (...) {}
        return 1;
    }
}

} // namespace test_fft_svm
