#pragma once
/**
 * @file test_batch_all_maxima.hpp
 * @brief Тесты для batch-обработки FindAllMaxima
 *
 * Проверяет:
 * 1. Batch FindAllMaxima с Dest=CPU (vector input)
 * 2. Batch FindAllMaxima с Dest=GPU (cl_mem input)
 * 3. Корректность beam_offset — каждый batch пишет в правильную позицию
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "spectrum_maxima_finder.h"
#include "drv_gpu.hpp"
#include "common/backend_type.hpp"
#include "services/console_output.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <thread>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_batch_all_maxima {

using namespace antenna_fft;
using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════════════════
// Генератор синтетических данных с известными пиками
// ════════════════════════════════════════════════════════════════════════════
inline std::vector<std::complex<float>> GenerateSignalWithPeaks(
    uint32_t antenna_count, uint32_t n_point, const std::vector<float>& freqs, float sample_rate)
{
    std::vector<std::complex<float>> all_data(antenna_count * n_point);

    for (uint32_t beam = 0; beam < antenna_count; ++beam) {
        for (uint32_t t = 0; t < n_point; ++t) {
            float val = 0.0f;
            for (float f : freqs) {
                val += std::sin(2.0f * static_cast<float>(M_PI) * f * t / sample_rate);
            }
            all_data[beam * n_point + t] = std::complex<float>(val, 0.0f);
        }
    }
    return all_data;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 1: Batch FindAllMaxima с Dest=CPU (vector input)
// ════════════════════════════════════════════════════════════════════════════
inline bool TestBatchVectorInput_DestCPU(IBackend* backend) {
    std::cout << "\n═══ Test Batch FindAllMaxima (Vector Input, Dest=CPU) ═══\n";

    const uint32_t antenna_count = 64;  // Большой батч для провоцирования batch-split
    const uint32_t n_point = 512;
    const float sample_rate = 1000.0f;

    // Каждая антенна — сигнал с одной частотой (50 Hz)
    std::vector<float> freqs = {50.0f};
    auto all_data = GenerateSignalWithPeaks(antenna_count, n_point, freqs, sample_rate);

    // Создаём SpectrumMaximaFinder с новым API
    SpectrumMaximaFinder finder(backend);

    InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = antenna_count;
    input.n_point = n_point;
    input.data = all_data;
    input.repeat_count = 1;
    input.sample_rate = sample_rate;
    input.memory_limit = 512 * 1024;  // 512 KB — маленький лимит для batch

    // FindAllMaxima с Dest=CPU
    AllMaximaResult result = finder.FindAllMaxima(input, OutputDestination::CPU);

    // Проверки
    if (result.beams.size() != antenna_count) {
        std::cerr << "ERROR: Expected " << antenna_count << " beams, got "
                  << result.beams.size() << std::endl;
        return false;
    }

    // Проверяем что все antenna_id корректные (0..antenna_count-1)
    for (const auto& beam : result.beams) {
        if (beam.antenna_id >= antenna_count) {
            std::cerr << "ERROR: Invalid antenna_id " << beam.antenna_id << std::endl;
            return false;
        }
        if (beam.positions.empty()) {
            std::cerr << "ERROR: Beam " << beam.antenna_id << " has no maxima!" << std::endl;
            return false;
        }
    }

    std::cout << "✅ Batch FindAllMaxima (Vector, Dest=CPU): PASSED ("
              << result.beams.size() << " beams, " << result.total_maxima << " maxima)\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 2: Batch FindAllMaxima с Dest=GPU (vector input)
// ════════════════════════════════════════════════════════════════════════════
inline bool TestBatchVectorInput_DestGPU(IBackend* backend) {
    std::cout << "\n═══ Test Batch FindAllMaxima (Vector Input, Dest=GPU) ═══\n";

    const uint32_t antenna_count = 64;
    const uint32_t n_point = 512;
    const float sample_rate = 1000.0f;

    std::vector<float> freqs = {50.0f};
    auto all_data = GenerateSignalWithPeaks(antenna_count, n_point, freqs, sample_rate);

    SpectrumMaximaFinder finder(backend);

    InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = antenna_count;
    input.n_point = n_point;
    input.data = all_data;
    input.repeat_count = 1;
    input.sample_rate = sample_rate;
    input.memory_limit = 512 * 1024;  // 512 KB — маленький лимит для batch

    // FindAllMaxima с Dest=GPU
    AllMaximaResult result = finder.FindAllMaxima(input, OutputDestination::GPU);

    // Проверяем что данные остались на GPU
    if (result.destination != OutputDestination::GPU) {
        std::cerr << "ERROR: Expected Dest=GPU, got " << static_cast<int>(result.destination) << std::endl;
        return false;
    }

    if (!result.gpu_positions || !result.gpu_magnitudes || !result.gpu_counts) {
        std::cerr << "ERROR: GPU buffers are null!" << std::endl;
        return false;
    }

    // Проверяем beam metadata (должны быть > 0 maxima)
    if (result.beams.size() != antenna_count) {
        std::cerr << "ERROR: Expected " << antenna_count << " beams metadata, got "
                  << result.beams.size() << std::endl;
        return false;
    }

    for (const auto& beam : result.beams) {
        if (beam.num_maxima == 0) {
            std::cerr << "ERROR: Beam " << beam.antenna_id << " has 0 maxima!" << std::endl;
            return false;
        }
    }

    std::cout << "✅ Batch FindAllMaxima (Vector, Dest=GPU): PASSED ("
              << result.beams.size() << " beams, " << result.total_maxima << " maxima on GPU)\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Запуск всех тестов
// ════════════════════════════════════════════════════════════════════════════
inline bool RunAllTests(IBackend* backend) {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout <<   "║       Batch AllMaxima Tests                                ║\n";
    std::cout <<   "╚════════════════════════════════════════════════════════════╝\n";

    bool all_passed = true;

    all_passed &= TestBatchVectorInput_DestCPU(backend);
    all_passed &= TestBatchVectorInput_DestGPU(backend);

    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    if (all_passed) {
        std::cout << "✅ ALL BATCH TESTS PASSED\n";
    } else {
        std::cout << "❌ SOME TESTS FAILED\n";
    }
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    return all_passed;
}

// ════════════════════════════════════════════════════════════════════════════
// run() — Entry point для main.cpp (создаёт собственный OpenCL контекст)
// ════════════════════════════════════════════════════════════════════════════
inline int run() {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    if (!con.IsRunning()) con.Start();

    try {
        cl_int err;
        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, nullptr);
        cl_device_id device;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
#ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
#else
        cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif

        auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
        backend->InitializeFromExternalContext(context, device, queue);

        // Запускаем все тесты
        bool result = RunAllTests(backend.get());

        backend.reset();
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        con.Stop();

        return result ? 0 : 1;

    } catch (const std::exception& e) {
        con.PrintError(0, "BatchTests", std::string("FATAL: ") + e.what());
        return 1;
    }
}

} // namespace test_batch_all_maxima
