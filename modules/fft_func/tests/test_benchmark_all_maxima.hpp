#pragma once
/**
 * @file test_benchmark_all_maxima.hpp
 * @brief Бенчмарк: 10 лучей × 500k точек — профилирование FindAllMaxima
 *
 * Формат вывода как в test_gpu_generator_integration (PrintReport с задержками).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-14
 */

#include "spectrum_maxima_finder.h"
#include "drv_gpu.hpp"
#include "common/backend_type.hpp"
#include "services/gpu_profiler.hpp"
#include "services/console_output.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <memory>
#include <thread>
#include <sstream>
#include <iomanip>
#include <map>
#include <string>
#include <filesystem>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_benchmark_all_maxima {

using namespace antenna_fft;
using namespace drv_gpu_lib;
using Clock = std::chrono::high_resolution_clock;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline std::vector<std::complex<float>> GenerateSignal(
    uint32_t beam_count, uint32_t n_point, float sample_rate)
{
    std::vector<std::complex<float>> data(
        static_cast<size_t>(beam_count) * n_point);
    for (uint32_t b = 0; b < beam_count; ++b) {
        float freq = 50.0f + b * 30.0f;
        for (uint32_t t = 0; t < n_point; ++t) {
            float val = std::sin(2.0f * static_cast<float>(M_PI) * freq * t / sample_rate);
            data[static_cast<size_t>(b) * n_point + t] = {val, 0.0f};
        }
    }
    return data;
}

inline int run() {
    const uint32_t BEAM_COUNT = 10;
    const uint32_t N_POINT = 500000;
    const float SAMPLE_RATE = 100000.0f;

    auto& con = ConsoleOutput::GetInstance();
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
        char dev_name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);

        // OpenCL версия и драйвер
        char ocl_ver[256], drv_ver[256], vendor[256];
        clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(ocl_ver), ocl_ver, nullptr);
        clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(drv_ver), drv_ver, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);

        auto backend = std::make_unique<OpenCLBackend>();
        backend->InitializeFromExternalContext(context, device, queue);

        // Настройка GPUProfiler (как в test_gpu_generator_integration)
        auto& profiler = GPUProfiler::GetInstance();
        profiler.Reset();
        profiler.SetEnabled(true);
        profiler.SetGPUEnabled(0, true);

        // GPU info для PrintReport (полная шапка)
        GPUReportInfo gpu_info;
        gpu_info.gpu_name = std::string(dev_name);
        gpu_info.global_mem_mb = static_cast<uint32_t>(backend->GetGlobalMemorySize() / (1024 * 1024));
        gpu_info.backend_type = BackendType::OPENCL;

        std::map<std::string, std::string> opencl_driver;
        opencl_driver["driver_type"] = "OpenCL";
        opencl_driver["version"] = std::string(ocl_ver);
        opencl_driver["driver_version"] = std::string(drv_ver);
        opencl_driver["vendor"] = std::string(vendor);
        gpu_info.drivers.push_back(opencl_driver);

        profiler.SetGPUInfo(0, gpu_info);

        con.Print(0, "Benchmark", "GPU: " + std::string(dev_name));
        con.Print(0, "Benchmark", std::to_string(BEAM_COUNT) + " beams x " +
            std::to_string(N_POINT) + " points, Fs=" + std::to_string(static_cast<int>(SAMPLE_RATE)) + " Hz");

        // Генерация данных
        auto signal = GenerateSignal(BEAM_COUNT, N_POINT, SAMPLE_RATE);

        // ═══════════════════════════════════════════════════
        // FindAllMaxima<T> (FFT → ALL maxima) — полный профиль
        // ═══════════════════════════════════════════════════
        profiler.Start();
        {
            SpectrumMaximaFinder finder(backend.get());
            InputData<std::vector<std::complex<float>>> input;
            input.antenna_count = BEAM_COUNT;
            input.n_point = N_POINT;
            input.data = signal;
            input.repeat_count = 2;
            input.sample_rate = SAMPLE_RATE;

            auto t0 = Clock::now();
            auto result = finder.FindAllMaxima(input, OutputDestination::CPU, DriverType::OPENCL);
            clFinish(queue);
            double wall = ms_since(t0);

            con.Print(0, "Benchmark", "FindAllMaxima: nFFT=" + std::to_string(finder.GetParams().nFFT) +
                " maxima=" + std::to_string(result.total_maxima) +
                " WALL=" + std::to_string(static_cast<int>(wall)) + " ms");
        }

        // Ждём обработку + вывод через PrintReport (как в test_gpu_generator_integration)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        profiler.PrintReport();

        // Экспорт результатов профилирования в файлы
        {
            auto now = std::chrono::system_clock::now();
            auto time_t_val = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf;
#ifdef _WIN32
            localtime_s(&tm_buf, &time_t_val);
#else
            localtime_r(&time_t_val, &tm_buf);
#endif
            std::ostringstream time_str;
            time_str << std::put_time(&tm_buf, "%H-%M-%S");

            std::string profiler_dir = "Results/Profiler/GPU_00_Profiler";
            std::filesystem::create_directories(profiler_dir);

            std::string base_path = profiler_dir + "/benchmark_all_maxima_" + time_str.str();
            profiler.ExportMarkdown(base_path + ".md");
            profiler.ExportJSON(base_path + ".json");

            con.Print(0, "Benchmark", "Profiler exported: " + base_path + ".{md,json}");
        }

        profiler.Stop();

        // Дождаться ConsoleOutput
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        con.Stop();

        backend.reset();
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;

    } catch (const std::exception& e) {
        con.PrintError(0, "Benchmark", std::string("FATAL: ") + e.what());
        return 1;
    }
}

} // namespace test_benchmark_all_maxima
