#pragma once
/**
 * @file test_large_batch.hpp
 * @brief Тест Batch Processing для больших данных (256 x 1300000)
 *
 * Тестирует работу SpectrumMaximaFinder с batch processing:
 * - 256 антенн x 1300000 точек (не помещаются на GPU целиком)
 * - Автоматическое разбиение на батчи через BatchManager
 * - Проверка корректности результатов (первые 10 лучей CPU vs GPU)
 * - Профилирование (GPUProfiler)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-11
 */

#include "spectrum_maxima_finder.h"
#include "drv_gpu.hpp"
#include "common/backend_type.hpp"
#include "DrvGPU/services/gpu_profiler.hpp"
#include "DrvGPU/services/batch_manager.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <map>
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <thread>
#include <chrono>

#include "modules/fft_maxima/tests/cpu_fft_reference.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_large_batch {

using namespace antenna_fft;
using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════════════════
// Параметры теста
// ════════════════════════════════════════════════════════════════════════════

constexpr uint32_t TEST_ANTENNA_COUNT = 256;
constexpr uint32_t TEST_N_POINT = 1300000;
constexpr uint32_t TEST_REPEAT_COUNT = 2;
constexpr float TEST_SAMPLE_RATE = 1000.0f;
constexpr uint32_t VALIDATION_ANTENNAS = 10;  // Первые 10 лучей для валидации

// ════════════════════════════════════════════════════════════════════════════
// Генерация тестовых данных (256 x 1300000)
// ════════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateLargeTestData(
    uint32_t antenna_count,
    uint32_t n_point,
    float sample_rate)
{
    std::cout << "\n[test_large_batch] Генерация тестовых данных:\n";
    std::cout << "  Антенн: " << antenna_count << "\n";
    std::cout << "  Точек на антенну: " << n_point << "\n";
    std::cout << "  Всего точек: " << (static_cast<size_t>(antenna_count) * n_point) << "\n";
    std::cout << "  Размер данных: " << (antenna_count * n_point * 8 / 1024 / 1024) << " MB\n";

    std::vector<std::complex<float>> data(
        static_cast<size_t>(antenna_count) * n_point);

    // Генерация синусоид с разными частотами для каждой антенны
    for (uint32_t antenna = 0; antenna < antenna_count; ++antenna) {
        // Частота = 2.5 * (1 + (antenna+1)/10) Hz
        float freq = 2.5f * (1.0f + (antenna + 1) / 10.0f);

        for (uint32_t t = 0; t < n_point; ++t) {
            float phase = 2.0f * static_cast<float>(M_PI) * freq * t / sample_rate;
            float value = std::sin(phase);
            size_t idx = static_cast<size_t>(antenna) * n_point + t;
            data[idx] = std::complex<float>(value, 0.0f);
        }

        // Прогресс каждые 32 антенны
        if ((antenna + 1) % 32 == 0) {
            std::cout << "  Сгенерировано: " << (antenna + 1) << "/" << antenna_count << " антенн\n";
        }
    }

    std::cout << "  Генерация завершена!\n\n";
    return data;
}

// ════════════════════════════════════════════════════════════════════════════
// Валидация первых N антенн (CPU vs GPU)
// ════════════════════════════════════════════════════════════════════════════

inline bool ValidateFirstAntennas(
    const std::vector<SpectrumResult>& gpu_results,
    const std::vector<std::complex<float>>& input_data,
    const SpectrumParams& params,
    uint32_t num_antennas_to_validate)
{
    std::cout << "[test_large_batch] Валидация первых " << num_antennas_to_validate << " антенн (CPU vs GPU):\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    bool all_passed = true;

    // Извлечь срез данных для валидации
    size_t slice_size = static_cast<size_t>(num_antennas_to_validate) * params.n_point;
    std::vector<std::complex<float>> validation_data(
        input_data.begin(),
        input_data.begin() + slice_size);

    // Создать params для CPU reference (только num_antennas_to_validate антенн)
    SpectrumParams cpu_params = params;
    cpu_params.antenna_count = num_antennas_to_validate;

    // Обработать на CPU
    std::cout << "  Обработка на CPU...\n";
    auto cpu_results = cpu_reference::NewProcessAllBeams_CPU(validation_data, cpu_params);
    std::cout << "  CPU обработка завершена\n";

    // Сравнить результаты
    for (uint32_t i = 0; i < num_antennas_to_validate; ++i) {
        // GPU результат для антенны i
        if (i >= gpu_results.size()) {
            std::cout << "  Антена " << i << ": GPU результат отсутствует\n";
            all_passed = false;
            continue;
        }
        const SpectrumResult& gpu_r = gpu_results[i];

        // CPU результат (ONE_PEAK режим — берём left, т.к. тестовые частоты в левом диапазоне)
        auto it = cpu_results.find(static_cast<int>(i));
        if (it == cpu_results.end()) {
            std::cout << "  Антена " << i << ": CPU результат отсутствует\n";
            all_passed = false;
            continue;
        }
        const auto& cpu_r = it->second.SpectrMax_left;

        // Сравнение частоты (основной критерий для ONE_PEAK)
        float freq_err = std::abs(gpu_r.interpolated.refined_frequency -
                                  cpu_r.interpolated.refined_frequency);

        // Сравнение bin (может отличаться из-за симметрии FFT, но частота должна совпадать)
        // Для ONE_PEAK режима bin может быть в "зеркальной" позиции nFFT - bin
        uint32_t gpu_bin = gpu_r.center_point.index;
        uint32_t cpu_bin = cpu_r.center_point.index;
        uint32_t nFFT = static_cast<uint32_t>(params.nFFT);

        // Проверяем либо совпадение, либо зеркальную позицию
        uint32_t mirror_gpu = (gpu_bin > nFFT/2) ? (nFFT - gpu_bin) : gpu_bin;
        float bin_err = std::abs(static_cast<float>(mirror_gpu) -
                                 static_cast<float>(cpu_bin));

        // Допуски
        bool freq_ok = (freq_err < 0.5f);
        bool bin_ok = (bin_err < 5.0f);  // увеличили допуск для зеркальных случаев
        bool passed = freq_ok && bin_ok;

        if (!passed) all_passed = false;

        std::cout << "  Антена " << std::setw(3) << i << ": "
                  << "freq GPU=" << std::fixed << std::setprecision(2)
                  << gpu_r.interpolated.refined_frequency << " Hz, "
                  << "CPU=" << cpu_r.interpolated.refined_frequency << " Hz "
                  << (freq_ok ? "✓" : "✗")
                  << " | bin mirror=" << mirror_gpu
                  << " CPU=" << cpu_bin
                  << " " << (bin_ok ? "✓" : "✗")
                  << " → " << (passed ? "OK" : "FAIL") << "\n";
    }

    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  Валидация: " << (all_passed ? "PASSED" : "FAILED") << "\n\n";

    return all_passed;
}

// ════════════════════════════════════════════════════════════════════════════
// Главная функция теста
// ════════════════════════════════════════════════════════════════════════════

inline int run() {
    try {
        std::cout << "\n";
        std::cout << "+==============================================================+\n";
        std::cout << "|   TEST: Large Batch Processing (256 x 1300000)               |\n";
        std::cout << "|   ВЕЗДЕ ищем ОДИН пик (ONE_PEAK)                             |\n";
        std::cout << "+==============================================================+\n\n";

        // ═══════════════════════════════════════════════════════════════════
        // 1. Инициализация DrvGPU
        // ═══════════════════════════════════════════════════════════════════
        std::cout << "[1] Инициализация DrvGPU...\n";
        DrvGPU gpu(BackendType::OPENCL, 0);
        gpu.Initialize();
        std::cout << "    GPU: " << gpu.GetDeviceName() << "\n";

        // Показать доступную память
        size_t total_mem = gpu.GetBackend().GetGlobalMemorySize();
        std::cout << "    Память GPU: " << (total_mem / 1024 / 1024) << " MB\n\n";

        // ═══════════════════════════════════════════════════════════════════
        // 2. Запуск GPUProfiler + передача информации о GPU
        // ═══════════════════════════════════════════════════════════════════
        auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
        const bool is_prof = profiler.IsEnabled() && profiler.IsGPUEnabled(0);
        if (is_prof) {
            std::cout << "[2] GPUProfiler включен — запускаем...\n\n";

            // Передаём информацию о GPU в профайлер для отчёта
            auto device_info = gpu.GetDeviceInfo();
            drv_gpu_lib::GPUReportInfo gpu_info;
            gpu_info.gpu_name = device_info.name;
            gpu_info.backend_type = BackendType::OPENCL;
            gpu_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);

            // Информация о драйвере OpenCL
            std::map<std::string, std::string> opencl_driver;
            opencl_driver["driver_type"] = "OpenCL";
            opencl_driver["version"] = device_info.opencl_version;
            opencl_driver["driver_version"] = device_info.driver_version;
            opencl_driver["vendor"] = device_info.vendor;
            gpu_info.drivers.push_back(opencl_driver);

            profiler.SetGPUInfo(0, gpu_info);
            profiler.Start();
        } else {
            std::cout << "[2] GPUProfiler выключен (можно включить в configGPU.json)\n\n";
        }

        // ═══════════════════════════════════════════════════════════════════
        // 3. Настройка параметров (НОВЫЙ API!)
        // ═══════════════════════════════════════════════════════════════════
        std::cout << "[3] Настройка параметров (новый API):\n";
        std::cout << "    antenna_count: " << TEST_ANTENNA_COUNT << "\n";
        std::cout << "    n_point: " << TEST_N_POINT << "\n";
        std::cout << "    repeat_count: " << TEST_REPEAT_COUNT << "\n";
        std::cout << "    sample_rate: " << TEST_SAMPLE_RATE << " Hz\n";
        std::cout << "    peak_mode: ONE_PEAK\n";
        std::cout << "    memory_limit: 80%\n\n";

        // ═══════════════════════════════════════════════════════════════════
        // 4. Создать SpectrumMaximaFinder (НОВЫЙ API — только backend!)
        // ═══════════════════════════════════════════════════════════════════
        std::cout << "[4] Создание SpectrumMaximaFinder (новый API)...\n";
        SpectrumMaximaFinder finder(&gpu.GetBackend());

        // Показать оценку памяти (приблизительно)
        uint32_t nFFT_estimate = 1;
        while (nFFT_estimate < TEST_N_POINT * TEST_REPEAT_COUNT) nFFT_estimate *= 2;
        size_t bytes_per_antenna = TEST_N_POINT * 8 + 2 * nFFT_estimate * 8 + 128;
        size_t total_required = static_cast<size_t>(TEST_ANTENNA_COUNT) * bytes_per_antenna;
        std::cout << "    nFFT (оценка): " << nFFT_estimate << "\n";
        std::cout << "    Память на антенну: " << (bytes_per_antenna / 1024 / 1024) << " MB\n";
        std::cout << "    Требуется всего: " << (total_required / 1024 / 1024 / 1024) << " GB\n";
        std::cout << "    Batch processing: ОБЯЗАТЕЛЕН!\n\n";

        // ═══════════════════════════════════════════════════════════════════
        // 5. Генерация тестовых данных
        // ═══════════════════════════════════════════════════════════════════
        std::cout << "[5] Генерация тестовых данных...\n";
        auto raw_data = GenerateLargeTestData(
            TEST_ANTENNA_COUNT,
            TEST_N_POINT,
            TEST_SAMPLE_RATE);

        // ═══════════════════════════════════════════════════════════════════
        // 6. Обработка данных (НОВЫЙ API с InputData!)
        // ═══════════════════════════════════════════════════════════════════
        std::cout << "[6] Запуск обработки (новый API, batch processing)...\n";

        // Создаём InputData с данными + параметрами
        InputData<std::vector<std::complex<float>>> input{
            .antenna_count = TEST_ANTENNA_COUNT,
            .n_point = TEST_N_POINT,
            .data = std::move(raw_data),
            .gpu_memory_bytes = 0,
            .repeat_count = TEST_REPEAT_COUNT,
            .sample_rate = TEST_SAMPLE_RATE,
            .search_range = 0,
            .memory_limit = 0.80f
        };

        auto start_time = std::chrono::high_resolution_clock::now();

        // НОВЫЙ API: Process(InputData, PeakSearchMode, DriverType)
        auto gpu_results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "    Обработка завершена!\n";
        std::cout << "    Результатов: " << gpu_results.size() << "\n";
        std::cout << "    Время: " << duration.count() << " ms\n\n";

        // ═══════════════════════════════════════════════════════════════════
        // 7. Профилирование
        // ═══════════════════════════════════════════════════════════════════
        std::cout << "[7] Профилирование:\n";
        if (is_prof) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            profiler.PrintReport();
        } else {
            const auto& prof = finder.GetProfilingData();
            std::cout << "    Upload:      " << std::fixed << std::setprecision(2)
                      << prof.upload_time_ms << " ms\n";
            std::cout << "    FFT:         " << prof.fft_time_ms << " ms\n";
            std::cout << "    Post-kernel: " << prof.post_kernel_time_ms << " ms\n";
            std::cout << "    Download:    " << prof.download_time_ms << " ms\n";
            std::cout << "    Total:       " << prof.total_time_ms << " ms\n\n";
        }

        // ═══════════════════════════════════════════════════════════════════
        // 8. Валидация (первые 10 антенн)
        // ═══════════════════════════════════════════════════════════════════
        std::cout << "[8] Валидация результатов:\n";
        // Создаём временный SpectrumParams для валидации
        SpectrumParams validation_params;
        validation_params.antenna_count = input.antenna_count;
        validation_params.n_point = input.n_point;
        validation_params.repeat_count = input.repeat_count;
        validation_params.sample_rate = input.sample_rate;
        validation_params.nFFT = finder.GetParams().nFFT;  // Получить вычисленный nFFT
        bool validation_passed = ValidateFirstAntennas(
            gpu_results, input.data, validation_params, VALIDATION_ANTENNAS);

        // ═══════════════════════════════════════════════════════════════════
        // 9. Проверка корректности antenna_id
        // ═══════════════════════════════════════════════════════════════════
        std::cout << "[9] Проверка antenna_id:\n";
        bool antenna_id_ok = true;
        for (size_t i = 0; i < gpu_results.size(); ++i) {
            if (gpu_results[i].antenna_id != static_cast<uint32_t>(i)) {
                std::cout << "    ОШИБКА: результат " << i
                          << " имеет antenna_id=" << gpu_results[i].antenna_id << "\n";
                antenna_id_ok = false;
            }
        }
        if (antenna_id_ok) {
            std::cout << "    Все antenna_id корректны!\n\n";
        }

        // ═══════════════════════════════════════════════════════════════════
        // 10. Финал
        // ═══════════════════════════════════════════════════════════════════
        bool all_passed = validation_passed && antenna_id_ok;

        std::cout << "+==============================================================+\n";
        if (all_passed) {
            std::cout << "|   ТЕСТ УСПЕШНО ПРОЙДЕН!                                      |\n";
        } else {
            std::cout << "|   ТЕСТ НЕ ПРОЙДЕН!                                           |\n";
        }
        std::cout << "+==============================================================+\n\n";

        // Остановка GPUProfiler
        if (is_prof) {
            profiler.Stop();
        }

        return all_passed ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "\nОШИБКА: " << e.what() << "\n\n";
        return 1;
    }
}

} // namespace test_large_batch
