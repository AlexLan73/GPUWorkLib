#pragma once

/**
 * @file test_gpu_generator_integration.hpp
 * @brief Интеграционный тест: signal_gen::CwGenerator + SpectrumMaximaFinder
 *
 * Тестирует полный pipeline:
 * 1. ExternalOpenCLContext создаёт контекст
 * 2. CwGenerator (signal_gen) генерирует данные на GPU (cl_mem)
 * 3. SpectrumMaximaFinder.Process(InputData<cl_mem>) обрабатывает БЕЗ upload
 * 4. Валидация результатов
 * 5. Benchmark
 *
 * Использует signal_gen::CwGenerator вместо устаревшего TestSignalGenerator.
 * CwGenerator доступен и из C++, и из Python (через gpuworklib bindings).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "signal_generator_factory.hpp"
#include "params/system_sampling.hpp"
#include "spectrum_maxima_finder.h"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include "DrvGPU/services/gpu_profiler.hpp"
#include "DrvGPU/logger/logger.hpp"

#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cmath>
#include <thread>
#include <sstream>
#include <filesystem>
#include <complex>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_gpu_generator_integration {

// ════════════════════════════════════════════════════════════════════════════
// Константы теста
// ════════════════════════════════════════════════════════════════════════════

static constexpr size_t TEST_ANTENNA_COUNT = 256;
static constexpr size_t TEST_N_POINT = 1'300'000;
static constexpr uint32_t TEST_REPEAT_COUNT = 2;
static constexpr float TEST_SAMPLE_RATE = 1000.0f;
static constexpr float TEST_BASE_FREQ = 2.5f;
static constexpr float TEST_FREQ_STEP = 0.25f;

// ════════════════════════════════════════════════════════════════════════════
// ExternalOpenCLContext — эмуляция внешнего OpenCL кода
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class ExternalOpenCLContext
 * @brief Создаёт и владеет OpenCL контекстом (как внешнее приложение)
 */
class ExternalOpenCLContext {
public:
    ExternalOpenCLContext() {
        cl_int err;

        // Platform
        err = clGetPlatformIDs(1, &platform_, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clGetPlatformIDs failed: " + std::to_string(err));

        // Device (GPU)
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clGetDeviceIDs failed: " + std::to_string(err));

        // Context
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateContext failed: " + std::to_string(err));

        // Queue с профилированием
#ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
#else
        queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateCommandQueue failed: " + std::to_string(err));

        // Имя устройства
        char name[256];
        clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        device_name_ = std::string(name);

        std::cout << "[ExternalOpenCLContext] Initialized on: " << device_name_ << "\n";
    }

    ~ExternalOpenCLContext() {
        if (queue_) { clReleaseCommandQueue(queue_); queue_ = nullptr; }
        if (context_) { clReleaseContext(context_); context_ = nullptr; }
        std::cout << "[ExternalOpenCLContext] Cleaned up\n";
    }

    cl_context GetContext() const { return context_; }
    cl_device_id GetDevice() const { return device_; }
    cl_command_queue GetQueue() const { return queue_; }
    const std::string& GetDeviceName() const { return device_name_; }

    std::string GetOpenCLVersion() const {
        char buf[256];
        clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(buf), buf, nullptr);
        std::string ver(buf);
        size_t pos = ver.find("OpenCL ");
        if (pos != std::string::npos) {
            ver = ver.substr(pos + 7);
            size_t space = ver.find(' ');
            if (space != std::string::npos) ver = ver.substr(0, space);
        }
        return ver;
    }

    std::string GetDriverVersion() const {
        char buf[256];
        clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(buf), buf, nullptr);
        return std::string(buf);
    }

    std::string GetVendor() const {
        char buf[256];
        clGetDeviceInfo(device_, CL_DEVICE_VENDOR, sizeof(buf), buf, nullptr);
        return std::string(buf);
    }

private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    std::string device_name_;
};

// ════════════════════════════════════════════════════════════════════════════
// Валидация сгенерированных данных (GPU vs CPU reference)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Читает данные с GPU и сравнивает с CPU reference
 *
 * @param queue        OpenCL queue для чтения
 * @param buffer       GPU буфер с данными
 * @param antenna_count Количество антенн
 * @param n_point      Точек на антенну
 * @param sample_rate  Частота дискретизации
 * @param base_freq    Базовая частота
 * @param freq_step    Шаг частоты
 * @param check_points Сколько точек проверить
 * @return true если данные совпадают
 */
static bool ValidateGeneratedData(
    cl_command_queue queue,
    cl_mem buffer,
    size_t antenna_count,
    size_t n_point,
    float sample_rate,
    float base_freq,
    float freq_step,
    size_t check_points = 10)
{
    size_t points_to_check = std::min(check_points, n_point);
    size_t antennas_to_check = std::min(antenna_count, size_t(5));

    bool all_ok = true;
    const float tolerance = 1e-4f;

    std::cout << "[ValidateGeneratedData] Checking first " << points_to_check
              << " points of " << antennas_to_check << " antennas...\n";

    std::vector<std::complex<float>> gpu_data(points_to_check);

    for (size_t ant = 0; ant < antennas_to_check; ++ant) {
        size_t offset = ant * n_point * sizeof(std::complex<float>);
        size_t size = points_to_check * sizeof(std::complex<float>);

        cl_int err = clEnqueueReadBuffer(queue, buffer, CL_TRUE,
                                          offset, size,
                                          gpu_data.data(),
                                          0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "  Read failed for antenna " << ant << "\n";
            return false;
        }

        // CPU reference: freq = base_freq + ant * freq_step
        float freq = base_freq + static_cast<float>(ant) * freq_step;
        bool antenna_ok = true;

        for (size_t i = 0; i < points_to_check && antenna_ok; ++i) {
            float t = static_cast<float>(i) / sample_rate;
            float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
            std::complex<float> expected(std::cos(phase), std::sin(phase));
            std::complex<float> actual = gpu_data[i];

            float diff_re = std::abs(actual.real() - expected.real());
            float diff_im = std::abs(actual.imag() - expected.imag());

            if (diff_re > tolerance || diff_im > tolerance) {
                std::cerr << "  Antenna " << ant << " point " << i
                          << ": expected (" << expected.real() << ", " << expected.imag()
                          << "), got (" << actual.real() << ", " << actual.imag() << ")\n";
                antenna_ok = false;
                all_ok = false;
            }
        }

        if (antenna_ok) {
            std::cout << "  Antenna " << ant << " (freq=" << freq << " Hz): OK\n";
        }
    }

    return all_ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Валидация результатов спектра
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Проверяет что найденные частоты совпадают с ожидаемыми
 */
static bool ValidateResults(
    const std::vector<antenna_fft::SpectrumResult>& results,
    float base_freq,
    float freq_step,
    size_t check_count = 10)
{
    std::cout << "\n[Validation] Checking first " << check_count << " results...\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    bool all_ok = true;
    const float tolerance = 0.1f;  // 0.1 Hz допуск

    size_t to_check = std::min(check_count, results.size());

    for (size_t i = 0; i < to_check; ++i) {
        const auto& r = results[i];
        float expected_freq = base_freq + i * freq_step;
        float actual_freq = r.interpolated.refined_frequency;

        float diff = std::abs(actual_freq - expected_freq);
        bool ok = (diff < tolerance);

        std::cout << "  Antenna " << std::setw(3) << r.antenna_id
                  << ": freq GPU=" << std::fixed << std::setprecision(2) << actual_freq
                  << " Hz, expected=" << expected_freq << " Hz";

        if (ok) {
            std::cout << " OK\n";
        } else {
            std::cout << " FAIL (diff=" << diff << ")\n";
            all_ok = false;
        }
    }

    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  Validation: " << (all_ok ? "PASSED" : "FAILED") << "\n";

    return all_ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Основной тест
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Запуск интеграционного теста
 * @return 0 при успехе, 1 при ошибке
 */
inline int run() {
    std::cout << R"(
+====================================================================+
|                                                                    |
|     TEST: CwGenerator (signal_gen) + SpectrumMaximaFinder          |
|                                                                    |
|     Pipeline:                                                      |
|       1. ExternalOpenCLContext creates context                     |
|       2. CwGenerator generates data on GPU (cl_mem)                |
|       3. SpectrumMaximaFinder.Process(InputData<cl_mem>)           |
|       4. Validate results                                          |
|       5. Benchmark                                                 |
|                                                                    |
+====================================================================+
)";

    try {
        // ══════════════════════════════════════════════════════════════
        // 1. Создаём внешний OpenCL контекст
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[1] Creating external OpenCL context...\n";
        ExternalOpenCLContext ext_ctx;

        // ══════════════════════════════════════════════════════════════
        // 2. Создаём DrvGPU backend с внешним контекстом
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[2] Initializing DrvGPU backend with external context...\n";
        auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
        backend->InitializeFromExternalContext(
            ext_ctx.GetContext(),
            ext_ctx.GetDevice(),
            ext_ctx.GetQueue()
        );
        std::cout << "  Backend: owns_resources = "
                  << (backend->OwnsResources() ? "true" : "false") << "\n";

        // ══════════════════════════════════════════════════════════════
        // 3. Создаём CwGenerator через фабрику signal_gen
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[3] Creating CwGenerator (signal_gen module)...\n";

        signal_gen::CwParams cw_params;
        cw_params.f0 = TEST_BASE_FREQ;
        cw_params.freq_step = TEST_FREQ_STEP;
        cw_params.amplitude = 1.0;
        cw_params.phase = 0.0;
        cw_params.complex_iq = true;

        auto generator = signal_gen::SignalGeneratorFactory::CreateCw(
            backend.get(), cw_params);

        std::cout << "  Generator: CW (f0=" << cw_params.f0
                  << " Hz, freq_step=" << cw_params.freq_step
                  << " Hz, complex_iq=true)\n";

        // ══════════════════════════════════════════════════════════════
        // 4. Генерируем данные на GPU
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[4] Generating test data on GPU via CwGenerator...\n";
        std::cout << "  " << TEST_ANTENNA_COUNT << " beams x "
                  << TEST_N_POINT << " points, fs=" << TEST_SAMPLE_RATE << " Hz\n";
        std::cout << "  Frequencies: " << TEST_BASE_FREQ << " Hz to "
                  << (TEST_BASE_FREQ + (TEST_ANTENNA_COUNT - 1) * TEST_FREQ_STEP)
                  << " Hz (step " << TEST_FREQ_STEP << " Hz)\n";

        signal_gen::SystemSampling sampling;
        sampling.fs = TEST_SAMPLE_RATE;
        sampling.length = TEST_N_POINT;

        auto gen_start = std::chrono::high_resolution_clock::now();

        cl_mem gpu_data = generator->GenerateToGpu(sampling, TEST_ANTENNA_COUNT);

        auto gen_end = std::chrono::high_resolution_clock::now();
        double gen_time_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();

        size_t buffer_size_mb = TEST_ANTENNA_COUNT * TEST_N_POINT * sizeof(std::complex<float>) / (1024 * 1024);
        std::cout << "  Generated: " << buffer_size_mb << " MB in " << gen_time_ms << " ms\n";

        // Валидация сгенерированных данных
        std::cout << "\n[4.1] Validating generated data (GPU vs CPU reference)...\n";
        bool gen_valid = ValidateGeneratedData(
            ext_ctx.GetQueue(), gpu_data,
            TEST_ANTENNA_COUNT, TEST_N_POINT,
            TEST_SAMPLE_RATE, TEST_BASE_FREQ, TEST_FREQ_STEP
        );
        if (!gen_valid) {
            std::cerr << "  Generated data validation FAILED!\n";
            clReleaseMemObject(gpu_data);
            return 1;
        }
        std::cout << "  Generated data: OK\n";

        // ══════════════════════════════════════════════════════════════
        // 5. Настройка GPUProfiler
        // ══════════════════════════════════════════════════════════════
        auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
        profiler.Reset();
        const bool is_prof = profiler.IsEnabled() && profiler.IsGPUEnabled(0);
        if (is_prof) {
            std::cout << "\n[5] GPUProfiler включен — запускаем...\n";

            drv_gpu_lib::GPUReportInfo gpu_info;
            gpu_info.gpu_name = ext_ctx.GetDeviceName();
            gpu_info.global_mem_mb = backend->GetGlobalMemorySize() / (1024 * 1024);
            gpu_info.backend_type = drv_gpu_lib::BackendType::OPENCL;

            std::map<std::string, std::string> opencl_driver;
            opencl_driver["driver_type"] = "OpenCL";
            opencl_driver["version"] = ext_ctx.GetOpenCLVersion();
            opencl_driver["driver_version"] = ext_ctx.GetDriverVersion();
            opencl_driver["vendor"] = ext_ctx.GetVendor();
            gpu_info.drivers.push_back(opencl_driver);

            profiler.SetGPUInfo(0, gpu_info);
            profiler.Start();
        } else {
            std::cout << "\n[5] GPUProfiler выключен\n";
        }

        // ══════════════════════════════════════════════════════════════
        // 6. Создаём SpectrumMaximaFinder с НОВЫМ API
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[6] Creating SpectrumMaximaFinder (NEW API)...\n";
        antenna_fft::SpectrumMaximaFinder finder(backend.get());

        // ══════════════════════════════════════════════════════════════
        // 7. Обработка данных с GPU (InputData<cl_mem>)
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[7] Processing data from GPU (cl_mem)...\n";

        size_t gpu_buffer_size = TEST_ANTENNA_COUNT * TEST_N_POINT * sizeof(std::complex<float>);

        antenna_fft::InputData<cl_mem> input{
            .antenna_count = static_cast<uint32_t>(TEST_ANTENNA_COUNT),
            .n_point = static_cast<uint32_t>(TEST_N_POINT),
            .data = gpu_data,
            .gpu_memory_bytes = gpu_buffer_size,
            .repeat_count = TEST_REPEAT_COUNT,
            .sample_rate = TEST_SAMPLE_RATE,
            .search_range = 0,      // auto
            .memory_limit = 0.80f
        };

        auto proc_start = std::chrono::high_resolution_clock::now();

        DRVGPU_LOG_INFO("SpectrumMaxima", ">>> Starting Process(): " +
            std::to_string(TEST_ANTENNA_COUNT) + " antennas x " +
            std::to_string(TEST_N_POINT) + " points, mode=ONE_PEAK");

        auto results = finder.Process(input, antenna_fft::PeakSearchMode::ONE_PEAK, antenna_fft::DriverType::OPENCL);

        auto proc_end = std::chrono::high_resolution_clock::now();
        double proc_time_ms = std::chrono::duration<double, std::milli>(proc_end - proc_start).count();

        DRVGPU_LOG_INFO("SpectrumMaxima", "<<< Process() completed: " +
            std::to_string(results.size()) + " results in " +
            std::to_string(static_cast<int>(proc_time_ms)) + " ms");

        std::cout << "\n  Processing completed!\n";
        std::cout << "  Results: " << results.size() << "\n";
        std::cout << "  Processing time: " << proc_time_ms << " ms\n";

        // ══════════════════════════════════════════════════════════════
        // 8. Профилирование через GPUProfiler
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[8] Профилирование:\n";
        if (is_prof) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            profiler.PrintReport();

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

            std::string base_path = profiler_dir + "/gpu_generator_" + time_str.str();
            profiler.ExportMarkdown(base_path + ".md");
            profiler.ExportJSON(base_path + ".json");
        } else {
            std::cout << "  GPUProfiler выключен\n";
        }

        // ══════════════════════════════════════════════════════════════
        // 9. Валидация результатов
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[9] Validating results...\n";
        bool results_valid = ValidateResults(results, TEST_BASE_FREQ, TEST_FREQ_STEP, 10);

        // ══════════════════════════════════════════════════════════════
        // 10. Проверяем antenna_id
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[10] Checking antenna_id...\n";
        bool ids_ok = true;
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].antenna_id != i) {
                std::cerr << "  antenna_id mismatch at " << i
                          << ": expected " << i << ", got " << results[i].antenna_id << "\n";
                ids_ok = false;
            }
        }
        std::cout << "  " << (ids_ok ? "All antenna_id correct!" : "antenna_id ERRORS found!") << "\n";

        // ══════════════════════════════════════════════════════════════
        // 11. Итоги
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "  BENCHMARK RESULTS:\n";
        std::cout << "    Data generation on GPU: " << gen_time_ms << " ms\n";
        std::cout << "    Processing (GPU->GPU):  " << proc_time_ms << " ms\n";
        std::cout << "    Total:                  " << (gen_time_ms + proc_time_ms) << " ms\n";
        std::cout << std::string(60, '=') << "\n";

        // Остановка GPUProfiler
        if (is_prof) {
            profiler.Stop();
        }

        // Cleanup — CwGenerator через ISignalGenerator*, cl_mem освобождаем сами
        clReleaseMemObject(gpu_data);
        generator.reset();
        backend.reset();

        // Результат
        bool success = results_valid && ids_ok && (results.size() == TEST_ANTENNA_COUNT);

        if (success) {
            std::cout << "\n+==============================================================+\n";
            std::cout << "|   TEST PASSED!                                               |\n";
            std::cout << "+==============================================================+\n\n";
            return 0;
        } else {
            std::cout << "\n+==============================================================+\n";
            std::cout << "|   TEST FAILED!                                               |\n";
            std::cout << "+==============================================================+\n\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\nFATAL EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}

} // namespace test_gpu_generator_integration
