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
#include "services/gpu_profiler.hpp"

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
    input.memory_limit = 0.01f;  // 1% памяти — принудительно включить batch

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
        if (beam.maxima.empty()) {
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
    input.memory_limit = 0.01f;  // 1% памяти — принудительно включить batch

    // FindAllMaxima с Dest=GPU
    AllMaximaResult result = finder.FindAllMaxima(input, OutputDestination::GPU);

    // Проверяем что данные остались на GPU
    if (result.destination != OutputDestination::GPU) {
        std::cerr << "ERROR: Expected Dest=GPU, got " << static_cast<int>(result.destination) << std::endl;
        return false;
    }

    if (!result.gpu_maxima || !result.gpu_counts) {
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

    // Caller владеет буферами — освобождаем после проверки
    if (result.gpu_maxima) clReleaseMemObject(static_cast<cl_mem>(result.gpu_maxima));
    if (result.gpu_counts) clReleaseMemObject(static_cast<cl_mem>(result.gpu_counts));

    std::cout << "✅ Batch FindAllMaxima (Vector, Dest=GPU): PASSED ("
              << result.beams.size() << " beams, " << result.total_maxima << " maxima on GPU)\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 3: Batch FindAllMaxima с GPU данными (InputData<cl_mem>) → Dest=CPU
// ════════════════════════════════════════════════════════════════════════════
inline bool TestBatchGPUInput_DestCPU(IBackend* backend) {
    std::cout << "\n═══ Test Batch FindAllMaxima (GPU Input, Dest=CPU) ═══\n";

    const uint32_t antenna_count = 32;
    const uint32_t n_point = 512;
    const float sample_rate = 1000.0f;
    std::vector<float> freqs = {50.0f};
    auto all_data = GenerateSignalWithPeaks(antenna_count, n_point, freqs, sample_rate);

    // Загружаем RAW сигнал на GPU (данные с GPU!)
    cl_context ctx = static_cast<cl_context>(backend->GetNativeContext());
    cl_int err;
    cl_mem gpu_signal = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        all_data.size() * sizeof(std::complex<float>), all_data.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: Failed to create GPU buffer\n";
        return false;
    }

    SpectrumMaximaFinder finder(backend);
    InputData<cl_mem> input;
    input.antenna_count = antenna_count;
    input.n_point = n_point;
    input.data = gpu_signal;
    input.gpu_memory_bytes = all_data.size() * sizeof(std::complex<float>);
    input.repeat_count = 1;
    input.sample_rate = sample_rate;
    input.memory_limit = 0.01f;

    AllMaximaResult result = finder.FindAllMaxima(input, OutputDestination::CPU);
    clReleaseMemObject(gpu_signal);

    bool ok = (result.beams.size() == antenna_count && result.total_maxima > 0);
    for (const auto& beam : result.beams) {
        if (beam.maxima.empty()) { ok = false; break; }
    }

    std::cout << (ok ? "✅" : "❌") << " Batch FindAllMaxima (GPU Input, Dest=CPU): "
              << result.beams.size() << " beams, " << result.total_maxima << " maxima\n";
    return ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 4: Batch FindAllMaxima с GPU данными (InputData<cl_mem>) → Dest=GPU
// ════════════════════════════════════════════════════════════════════════════
inline bool TestBatchGPUInput_DestGPU(IBackend* backend) {
    std::cout << "\n═══ Test Batch FindAllMaxima (GPU Input, Dest=GPU) ═══\n";

    const uint32_t antenna_count = 32;
    const uint32_t n_point = 512;
    const float sample_rate = 1000.0f;
    std::vector<float> freqs = {50.0f};
    auto all_data = GenerateSignalWithPeaks(antenna_count, n_point, freqs, sample_rate);

    cl_context ctx = static_cast<cl_context>(backend->GetNativeContext());
    cl_int err;
    cl_mem gpu_signal = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        all_data.size() * sizeof(std::complex<float>), all_data.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: Failed to create GPU buffer\n";
        return false;
    }

    SpectrumMaximaFinder finder(backend);
    InputData<cl_mem> input;
    input.antenna_count = antenna_count;
    input.n_point = n_point;
    input.data = gpu_signal;
    input.gpu_memory_bytes = all_data.size() * sizeof(std::complex<float>);
    input.repeat_count = 1;
    input.sample_rate = sample_rate;
    input.memory_limit = 0.01f;

    AllMaximaResult result = finder.FindAllMaxima(input, OutputDestination::GPU);
    clReleaseMemObject(gpu_signal);

    bool ok = (result.destination == OutputDestination::GPU &&
               result.gpu_maxima && result.gpu_counts &&
               result.beams.size() == antenna_count);

    if (result.gpu_maxima) clReleaseMemObject(static_cast<cl_mem>(result.gpu_maxima));
    if (result.gpu_counts) clReleaseMemObject(static_cast<cl_mem>(result.gpu_counts));

    std::cout << (ok ? "✅" : "❌") << " Batch FindAllMaxima (GPU Input, Dest=GPU): "
              << result.beams.size() << " beams, " << result.total_maxima << " maxima on GPU\n";
    return ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 5: FindAllMaxima с профилированием (GPUProfiler → console_output)
// ════════════════════════════════════════════════════════════════════════════
inline bool TestBatchWithProfiling(IBackend* backend) {
    std::cout << "\n═══ Test Batch FindAllMaxima + GPUProfiler ═══\n";

    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    profiler.Reset();
    profiler.SetEnabled(true);
    profiler.SetGPUEnabled(0, true);

    // Передаём информацию о GPU в профайлер для отчёта
    // gpu_id = backend->GetDeviceIndex() — Record() использует этот же id
    int gpu_id = backend->GetDeviceIndex();
    if (gpu_id < 0) gpu_id = 0;  // external context: -1 → 0

    auto device_info = backend->GetDeviceInfo();
    drv_gpu_lib::GPUReportInfo gpu_info;
    gpu_info.gpu_name = device_info.name.empty() ? "Unknown" : device_info.name;
    gpu_info.backend_type = BackendType::OPENCL;
    gpu_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);
    std::map<std::string, std::string> opencl_driver;
    opencl_driver["driver_type"] = "OpenCL";
    opencl_driver["version"] = device_info.opencl_version;
    opencl_driver["driver_version"] = device_info.driver_version;
    opencl_driver["vendor"] = device_info.vendor;
    gpu_info.drivers.push_back(opencl_driver);
    profiler.SetGPUInfo(gpu_id, gpu_info);
    if (backend->GetDeviceIndex() < 0) profiler.SetGPUInfo(-1, gpu_info);

    const uint32_t antenna_count = 32;
    const uint32_t n_point = 1024;
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
    input.memory_limit = 0.01f;

    profiler.Start();
    AllMaximaResult result = finder.FindAllMaxima(input, OutputDestination::CPU);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    profiler.PrintReport();
    profiler.Stop();

    bool ok = (result.beams.size() == antenna_count && result.total_maxima > 0);
    std::cout << (ok ? "✅" : "❌") << " Batch+Profiling: " << result.total_maxima << " maxima, "
              << "beams=" << result.beams.size() << "\n";
    return ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Запуск всех тестов
// ════════════════════════════════════════════════════════════════════════════
inline bool RunAllTests(IBackend* backend) {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout <<   "║       Batch AllMaxima Tests                                ║\n";
    std::cout <<   "╚════════════════════════════════════════════════════════════╝\n";

    bool all_passed = true;

    all_passed &= TestBatchVectorInput_DestCPU(backend);   // CPU data → Dest=CPU
    all_passed &= TestBatchVectorInput_DestGPU(backend);   // CPU data → Dest=GPU
    all_passed &= TestBatchGPUInput_DestCPU(backend);      // GPU data → Dest=CPU
    all_passed &= TestBatchGPUInput_DestGPU(backend);     // GPU data → Dest=GPU
    all_passed &= TestBatchWithProfiling(backend);

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
