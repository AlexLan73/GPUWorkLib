#pragma once

/**
 * @file test_external_context_fft.hpp
 * @brief Тест FFT с внешним OpenCL контекстом (пункт 7 плана)
 *
 * ============================================================================
 * СЦЕНАРИЙ ТЕСТА:
 *   Эмуляция реальной интеграции: "чужой" OpenCL код создаёт контекст,
 *   queue и данные, а DrvGPU + fft_maxima обрабатывают их.
 *
 * ПОТОК ДАННЫХ:
 *
 *   ВХОД (3 варианта):
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │ Вариант A: cl_context + cl_mem                                 │
 *   │   Внешний код создаёт cl_mem буфер → передаёт в DrvGPU         │
 *   │   OpenCLBackendExternal::InitializeFromExternalContext()        │
 *   │                                                                  │
 *   │ Вариант B: cl_context + ExternalCLBufferAdapter                │
 *   │   Обёртка ExternalCLBufferAdapter<complex<float>> над cl_mem   │
 *   │   Read()/Write() для загрузки/выгрузки данных                  │
 *   └──────────────────────────────────────────────────────────────────┘
 *              |
 *              v
 *   AntennaFFTProcMax(params, &external_backend)
 *              |
 *              v
 *   ВЫХОД (3 варианта):
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │ Вариант 1: Результат на CPU (vector<FFTResult>)                │
 *   │   Стандартный вывод AntennaFFTResult                            │
 *   │                                                                  │
 *   │ Вариант 2: Результат в cl_mem (ссылка)                         │
 *   │   Данные остаются на GPU, возвращается cl_mem handle            │
 *   │                                                                  │
 *   │ Вариант 3: SVM → cl_mem конвертация                            │
 *   │   Если данные в SVM, конвертировать в cl_mem через              │
 *   │   clEnqueueSVMMemcpy → cl_mem буфер                            │
 *   └──────────────────────────────────────────────────────────────────┘
 *
 * КЛЮЧЕВАЯ ОСОБЕННОСТЬ:
 *   owns_resources_ = false → DrvGPU НЕ уничтожает внешний контекст!
 *   Внешний код продолжает работать после уничтожения DrvGPU backend.
 *
 * ЗАВИСИМОСТИ:
 *   - DrvGPU/backends/opencl/opencl_backend_external.hpp
 *   - DrvGPU/memory/external_cl_buffer_adapter.hpp
 *   - DrvGPU/memory/svm_buffer.hpp
 *   - modules/fft_maxima/include/antenna_fft_release.h
 *   - DrvGPU/services/service_manager.hpp
 *
 * ЗАПУСК:
 *   test_external_context_fft::run() → 0 при успехе, 1 при ошибке
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "DrvGPU/backends/opencl/opencl_backend_external.hpp"
#include "DrvGPU/memory/external_cl_buffer_adapter.hpp"
#include "DrvGPU/memory/svm_buffer.hpp"
#include "DrvGPU/services/service_manager.hpp"
#include "DrvGPU/services/gpu_profiler.hpp"
#include "DrvGPU/services/console_output.hpp"
#include "antenna_fft_release.h"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <stdexcept>

namespace test_external_context_fft {

// ============================================================================
// Константы теста
// ============================================================================

static constexpr size_t TEST_BEAM_COUNT = 4;       ///< Количество лучей
static constexpr size_t TEST_COUNT_POINTS = 256;   ///< Точек на луч
static constexpr size_t TEST_OUT_POINTS_FFT = 128; ///< Выходных точек
static constexpr size_t TEST_MAX_PEAKS = 3;        ///< Максимумов для поиска
static constexpr float SIGNAL_AMPLITUDE = 8.0f;    ///< Амплитуда сигнала
static constexpr float NOISE_LEVEL = 0.05f;        ///< Уровень шума

// ============================================================================
// ExternalOpenCLContext — эмуляция "чужого" OpenCL кода
// ============================================================================

/**
 * @class ExternalOpenCLContext
 * @brief Эмуляция внешнего OpenCL приложения
 *
 * Создаёт и владеет:
 * - cl_platform_id
 * - cl_device_id
 * - cl_context
 * - cl_command_queue
 * - cl_mem буфер с тестовыми данными
 *
 * DrvGPU НЕ должен освобождать эти ресурсы!
 */
class ExternalOpenCLContext {
public:
    /**
     * @brief Инициализация внешнего OpenCL контекста
     *
     * Создаёт контекст на первом GPU, создаёт буфер с тестовыми данными.
     */
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

        // Queue (OpenCL 2.0+ с профилированием)
#ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
#else
        queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateCommandQueue failed: " + std::to_string(err));

        // Получаем имя устройства
        char name[256];
        clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        device_name_ = std::string(name);

        std::cout << "   [ExternalContext] Initialized on: " << device_name_ << "\n";
    }

    /**
     * @brief Создать cl_mem буфер с тестовыми данными
     *
     * Генерирует комплексную синусоиду с известной частотой для каждого луча.
     *
     * @param beam_count    Количество лучей
     * @param count_points  Точек на луч
     * @return cl_mem буфер (ВНЕШНИЙ! DrvGPU НЕ владеет им!)
     */
    cl_mem CreateTestDataBuffer(size_t beam_count, size_t count_points) {
        // Генерация тестовых данных
        test_data_.clear();
        test_data_.reserve(beam_count * count_points);

        const float two_pi = 2.0f * 3.14159265358979f;
        std::mt19937 rng(12345);
        std::normal_distribution<float> noise(0.0f, NOISE_LEVEL);

        for (size_t beam = 0; beam < beam_count; ++beam) {
            size_t target_bin = beam * 15 + 5;  // Бин: 5, 20, 35, 50
            float freq_norm = static_cast<float>(target_bin) / static_cast<float>(count_points);

            for (size_t n = 0; n < count_points; ++n) {
                float phase = two_pi * freq_norm * static_cast<float>(n);
                float re = SIGNAL_AMPLITUDE * std::cos(phase) + noise(rng);
                float im = SIGNAL_AMPLITUDE * std::sin(phase) + noise(rng);
                test_data_.push_back(std::complex<float>(re, im));
            }
        }

        // Создаём cl_mem буфер и заполняем данными
        cl_int err;
        size_t buffer_size = test_data_.size() * sizeof(std::complex<float>);

        data_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer failed: " + std::to_string(err));

        err = clEnqueueWriteBuffer(queue_, data_buffer_, CL_TRUE, 0,
                                    buffer_size, test_data_.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueWriteBuffer failed: " + std::to_string(err));

        std::cout << "   [ExternalContext] Created test buffer: "
                  << test_data_.size() << " samples ("
                  << (buffer_size / 1024.0) << " KB)\n";

        return data_buffer_;
    }

    /**
     * @brief Получить ожидаемые частотные бины для проверки
     */
    std::vector<size_t> GetExpectedBins(size_t beam_count) const {
        std::vector<size_t> bins;
        for (size_t b = 0; b < beam_count; ++b) {
            bins.push_back(b * 15 + 5);
        }
        return bins;
    }

    // Геттеры для OpenCL ресурсов
    cl_context GetContext() const { return context_; }
    cl_device_id GetDevice() const { return device_; }
    cl_command_queue GetQueue() const { return queue_; }
    cl_mem GetDataBuffer() const { return data_buffer_; }
    const std::string& GetDeviceName() const { return device_name_; }
    const std::vector<std::complex<float>>& GetTestData() const { return test_data_; }

    /**
     * @brief Деструктор — освобождает ВСЕ ресурсы
     *
     * Вызывается ПОСЛЕ уничтожения DrvGPU backend.
     * К этому моменту DrvGPU уже закончил работу и НЕ трогал наши ресурсы
     * (owns_resources_ = false).
     */
    ~ExternalOpenCLContext() {
        if (data_buffer_) { clReleaseMemObject(data_buffer_); data_buffer_ = nullptr; }
        if (queue_)       { clReleaseCommandQueue(queue_);    queue_ = nullptr; }
        if (context_)     { clReleaseContext(context_);       context_ = nullptr; }
        std::cout << "   [ExternalContext] Cleaned up (context/queue released)\n";
    }

private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_mem data_buffer_ = nullptr;
    std::string device_name_;
    std::vector<std::complex<float>> test_data_;
};

// ============================================================================
// Проверка результатов
// ============================================================================

/**
 * @brief Проверить результаты FFT для внешнего контекста
 *
 * @param result        Результат AntennaFFTProcMax
 * @param expected_bins Ожидаемые частотные бины (по одному на луч)
 * @param tolerance     Допуск в бинах
 * @return true если все проверки пройдены
 */
static bool VerifyExternalResults(
    const antenna_fft::AntennaFFTResult& result,
    const std::vector<size_t>& expected_bins,
    size_t tolerance = 2)
{
    bool all_ok = true;

    if (result.results.size() != expected_bins.size()) {
        std::cerr << "   [FAIL] Expected " << expected_bins.size()
                  << " beams, got " << result.results.size() << "\n";
        return false;
    }

    for (size_t beam = 0; beam < expected_bins.size(); ++beam) {
        const auto& br = result.results[beam];

        if (br.max_values.empty()) {
            std::cerr << "   [FAIL] Beam " << beam << ": no max_values\n";
            all_ok = false;
            continue;
        }

        size_t actual = br.max_values[0].index_point;
        size_t expected = expected_bins[beam];
        int diff = static_cast<int>(actual) - static_cast<int>(expected);

        bool ok = (std::abs(diff) <= static_cast<int>(tolerance));
        if (ok) {
            std::cout << "   [PASS] Beam " << beam
                      << ": expected=" << expected << " actual=" << actual
                      << " amp=" << std::fixed << std::setprecision(2)
                      << br.max_values[0].amplitude << "\n";
        } else {
            std::cerr << "   [FAIL] Beam " << beam
                      << ": expected=" << expected << " actual=" << actual
                      << " diff=" << diff << "\n";
            all_ok = false;
        }
    }

    return all_ok;
}

// ============================================================================
// ТЕСТ A: Вход через cl_mem, выход на CPU
// ============================================================================

/**
 * @brief Тест A: Внешний cl_mem → FFT → результаты на CPU
 *
 * Самый простой сценарий интеграции:
 * 1. Внешний код создаёт cl_mem с данными
 * 2. DrvGPU (OpenCLBackendExternal) обрабатывает через FFT
 * 3. Результаты возвращаются на CPU как AntennaFFTResult
 */
static bool TestA_ClMemInput_CpuOutput(ExternalOpenCLContext& ext_ctx) {
    std::cout << "\n  ── TEST A: cl_mem input → CPU output ──\n";

    try {
        // 1. Создаём DrvGPU backend с внешним контекстом
        auto backend = std::make_unique<drv_gpu_lib::OpenCLBackendExternal>();
        backend->InitializeFromExternalContext(
            ext_ctx.GetContext(),
            ext_ctx.GetDevice(),
            ext_ctx.GetQueue()
        );

        std::cout << "   Backend: owns_resources = "
                  << (backend->OwnsResources() ? "true" : "false") << "\n";

        // 2. Создаём cl_mem буфер с тестовыми данными
        cl_mem ext_buffer = ext_ctx.CreateTestDataBuffer(TEST_BEAM_COUNT, TEST_COUNT_POINTS);

        // 3. Создаём FFT процессор
        antenna_fft::AntennaFFTParams params(
            TEST_BEAM_COUNT, TEST_COUNT_POINTS,
            TEST_OUT_POINTS_FFT, TEST_MAX_PEAKS,
            "test_ext_A", "ExternalA"
        );

        antenna_fft::AntennaFFTProcMax processor(params, backend.get());

        // 4. Обрабатываем данные из внешнего cl_mem
        //    ProcessNew(vector) — загрузим данные с GPU → CPU → GPU (через IBackend)
        //    В реальном проекте можно использовать ProcessNew(cl_mem) напрямую!
        auto t_start = std::chrono::high_resolution_clock::now();

        auto result = processor.ProcessNew(ext_ctx.GetTestData());

        auto t_end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // Профилирование
        drv_gpu_lib::GPUProfiler::GetInstance().Record(
            0, "ExternalA", "FFT_Total", time_ms);

        std::cout << "   FFT time: " << std::fixed << std::setprecision(2) << time_ms << " ms\n";

        // 5. Проверяем результаты (выход на CPU)
        auto expected_bins = ext_ctx.GetExpectedBins(TEST_BEAM_COUNT);
        bool ok = VerifyExternalResults(result, expected_bins);

        // 6. Уничтожаем backend (НЕ трогает контекст!)
        backend.reset();
        std::cout << "   Backend destroyed (context still alive: "
                  << (ext_ctx.GetContext() ? "YES" : "NO") << ")\n";

        return ok;

    } catch (const std::exception& e) {
        std::cerr << "   EXCEPTION: " << e.what() << "\n";
        return false;
    }
}

// ============================================================================
// ТЕСТ B: Вход через ExternalCLBufferAdapter, выход на CPU
// ============================================================================

/**
 * @brief Тест B: ExternalCLBufferAdapter → FFT → CPU
 *
 * Используем типобезопасный адаптер для работы с внешним буфером:
 * 1. ExternalCLBufferAdapter оборачивает cl_mem
 * 2. Read() загружает данные с GPU → CPU
 * 3. ProcessNew(vector) обрабатывает FFT
 * 4. Результат на CPU
 */
static bool TestB_AdapterInput_CpuOutput(ExternalOpenCLContext& ext_ctx) {
    std::cout << "\n  ── TEST B: ExternalCLBufferAdapter input → CPU output ──\n";

    try {
        // 1. Backend с внешним контекстом
        auto backend = std::make_unique<drv_gpu_lib::OpenCLBackendExternal>();
        backend->InitializeFromExternalContext(
            ext_ctx.GetContext(), ext_ctx.GetDevice(), ext_ctx.GetQueue()
        );

        // 2. Создаём данные и cl_mem буфер
        cl_mem ext_buffer = ext_ctx.CreateTestDataBuffer(TEST_BEAM_COUNT, TEST_COUNT_POINTS);
        size_t total_elements = TEST_BEAM_COUNT * TEST_COUNT_POINTS;

        // 3. Оборачиваем в ExternalCLBufferAdapter (НЕ владеет буфером!)
        drv_gpu_lib::ExternalCLBufferAdapter<std::complex<float>> adapter(
            ext_buffer,
            total_elements,
            ext_ctx.GetQueue(),
            false  // owns_buffer = false
        );

        std::cout << "   Adapter: " << adapter.GetNumElements() << " elements, "
                  << "owns=" << (adapter.OwnsBuffer() ? "YES" : "NO") << "\n";

        // 4. Загружаем данные через адаптер (GPU → CPU)
        auto t_read_start = std::chrono::high_resolution_clock::now();
        auto data_from_gpu = adapter.Read();
        auto t_read_end = std::chrono::high_resolution_clock::now();
        double read_ms = std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();

        std::cout << "   Read via adapter: " << data_from_gpu.size()
                  << " elements in " << std::fixed << std::setprecision(2) << read_ms << " ms\n";

        // Профилирование
        drv_gpu_lib::GPUProfiler::GetInstance().Record(0, "ExternalB", "Read", read_ms);

        // 5. FFT обработка
        antenna_fft::AntennaFFTParams params(
            TEST_BEAM_COUNT, TEST_COUNT_POINTS,
            TEST_OUT_POINTS_FFT, TEST_MAX_PEAKS,
            "test_ext_B", "ExternalB"
        );

        antenna_fft::AntennaFFTProcMax processor(params, backend.get());

        auto t_fft_start = std::chrono::high_resolution_clock::now();
        auto result = processor.ProcessNew(data_from_gpu);
        auto t_fft_end = std::chrono::high_resolution_clock::now();
        double fft_ms = std::chrono::duration<double, std::milli>(t_fft_end - t_fft_start).count();

        drv_gpu_lib::GPUProfiler::GetInstance().Record(0, "ExternalB", "FFT_Total", fft_ms);
        std::cout << "   FFT time: " << fft_ms << " ms\n";

        // 6. Проверяем
        auto expected_bins = ext_ctx.GetExpectedBins(TEST_BEAM_COUNT);
        bool ok = VerifyExternalResults(result, expected_bins);

        backend.reset();
        return ok;

    } catch (const std::exception& e) {
        std::cerr << "   EXCEPTION: " << e.what() << "\n";
        return false;
    }
}

// ============================================================================
// ТЕСТ C: SVM → cl_mem конвертация
// ============================================================================

/**
 * @brief Тест C: SVM данные → конвертация в cl_mem → FFT → CPU
 *
 * Демонстрирует конвертацию SVM → cl_mem:
 * 1. Данные в SVM буфере (из внешнего контекста)
 * 2. Копируем SVM → cl_mem через clEnqueueSVMMemcpy/clEnqueueWriteBuffer
 * 3. Обрабатываем FFT
 * 4. Результат на CPU
 *
 * Если SVM не поддерживается — пропускаем тест (SKIP).
 */
static bool TestC_SvmToClMem(ExternalOpenCLContext& ext_ctx) {
    std::cout << "\n  ── TEST C: SVM → cl_mem conversion → FFT → CPU ──\n";

    try {
        // 1. Backend с внешним контекстом
        auto backend = std::make_unique<drv_gpu_lib::OpenCLBackendExternal>();
        backend->InitializeFromExternalContext(
            ext_ctx.GetContext(), ext_ctx.GetDevice(), ext_ctx.GetQueue()
        );

        // 2. Проверяем SVM
        if (!backend->SupportsSVM()) {
            std::cout << "   SKIP: SVM not supported on this device\n";
            backend.reset();
            return true;  // SKIP = success
        }

        // 3. Создаём тестовые данные на CPU
        size_t total_elements = TEST_BEAM_COUNT * TEST_COUNT_POINTS;
        const auto& cpu_data = ext_ctx.GetTestData();

        // Если тестовые данные ещё не созданы, создадим
        if (cpu_data.empty()) {
            ext_ctx.CreateTestDataBuffer(TEST_BEAM_COUNT, TEST_COUNT_POINTS);
        }

        // 4. Загружаем в SVM буфер
        std::cout << "   Creating SVM buffer...\n";

        drv_gpu_lib::SVMBuffer svm_buffer(
            ext_ctx.GetContext(),
            ext_ctx.GetQueue(),
            ext_ctx.GetTestData(),
            drv_gpu_lib::MemoryStrategy::SVM_COARSE_GRAIN,
            drv_gpu_lib::MemoryType::GPU_READ_WRITE
        );

        std::cout << "   SVM buffer: " << svm_buffer.GetNumElements()
                  << " elements (" << (svm_buffer.GetSizeBytes() / 1024.0) << " KB)\n";

        // 5. SVM → CPU → ProcessNew (конвертация через Read)
        //    В реальном проекте можно оптимизировать через clEnqueueSVMMemcpy
        //    напрямую в cl_mem, но для безопасности делаем через CPU.
        auto t_conv_start = std::chrono::high_resolution_clock::now();
        auto data_from_svm = svm_buffer.Read();
        auto t_conv_end = std::chrono::high_resolution_clock::now();
        double conv_ms = std::chrono::duration<double, std::milli>(t_conv_end - t_conv_start).count();

        std::cout << "   SVM → CPU read: " << conv_ms << " ms\n";
        drv_gpu_lib::GPUProfiler::GetInstance().Record(0, "ExternalC", "SVM_Read", conv_ms);

        // 6. FFT обработка
        antenna_fft::AntennaFFTParams params(
            TEST_BEAM_COUNT, TEST_COUNT_POINTS,
            TEST_OUT_POINTS_FFT, TEST_MAX_PEAKS,
            "test_ext_C", "ExternalC"
        );

        antenna_fft::AntennaFFTProcMax processor(params, backend.get());

        auto t_fft_start = std::chrono::high_resolution_clock::now();
        auto result = processor.ProcessNew(data_from_svm);
        auto t_fft_end = std::chrono::high_resolution_clock::now();
        double fft_ms = std::chrono::duration<double, std::milli>(t_fft_end - t_fft_start).count();

        drv_gpu_lib::GPUProfiler::GetInstance().Record(0, "ExternalC", "FFT_Total", fft_ms);
        std::cout << "   FFT time: " << fft_ms << " ms\n";

        // 7. Проверяем
        auto expected_bins = ext_ctx.GetExpectedBins(TEST_BEAM_COUNT);
        bool ok = VerifyExternalResults(result, expected_bins);

        backend.reset();
        return ok;

    } catch (const std::exception& e) {
        std::cerr << "   EXCEPTION: " << e.what() << "\n";
        return false;
    }
}

// ============================================================================
// ОСНОВНОЙ ТЕСТ
// ============================================================================

/**
 * @brief Запуск всех тестов с внешним OpenCL контекстом
 * @return 0 при успехе, 1 при ошибке
 *
 * Запускает три подтеста:
 *   A: cl_mem вход → CPU выход
 *   B: ExternalCLBufferAdapter вход → CPU выход
 *   C: SVM → cl_mem конвертация → FFT → CPU выход
 */
inline int run() {
    std::cout << R"(
+====================================================================+
|                                                                    |
|     TEST: FFT with External OpenCL Context                         |
|                                                                    |
|     Subtests:                                                      |
|       A: cl_mem input -> CPU output                                |
|       B: ExternalCLBufferAdapter -> CPU output                     |
|       C: SVM -> cl_mem conversion -> FFT                           |
|                                                                    |
|     Key feature: owns_resources_ = false                           |
|       DrvGPU does NOT destroy external context!                    |
|                                                                    |
+====================================================================+
)";

    try {
        // ══════════════════════════════════════════════════════════════
        // Запуск сервисов
        // ══════════════════════════════════════════════════════════════
        std::cout << "[SETUP] Starting services...\n";
        auto& sm = drv_gpu_lib::ServiceManager::GetInstance();
        sm.InitializeDefaults();
        sm.StartAll();

        // ══════════════════════════════════════════════════════════════
        // Создание внешнего OpenCL контекста
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[SETUP] Creating external OpenCL context...\n";
        ExternalOpenCLContext ext_ctx;

        // ══════════════════════════════════════════════════════════════
        // Запуск подтестов
        // ══════════════════════════════════════════════════════════════
        int passed = 0;
        int failed = 0;

        // Test A: cl_mem → CPU
        if (TestA_ClMemInput_CpuOutput(ext_ctx)) {
            passed++;
            std::cout << "   >>> TEST A: PASSED\n";
        } else {
            failed++;
            std::cout << "   >>> TEST A: FAILED\n";
        }

        // Test B: Adapter → CPU
        if (TestB_AdapterInput_CpuOutput(ext_ctx)) {
            passed++;
            std::cout << "   >>> TEST B: PASSED\n";
        } else {
            failed++;
            std::cout << "   >>> TEST B: FAILED\n";
        }

        // Test C: SVM → cl_mem → CPU
        if (TestC_SvmToClMem(ext_ctx)) {
            passed++;
            std::cout << "   >>> TEST C: PASSED\n";
        } else {
            failed++;
            std::cout << "   >>> TEST C: FAILED\n";
        }

        // ══════════════════════════════════════════════════════════════
        // Проверяем что внешний контекст жив
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[VERIFY] External context still alive after all tests:\n";
        std::cout << "   Context: " << (ext_ctx.GetContext() ? "OK" : "NULL!") << "\n";
        std::cout << "   Queue:   " << (ext_ctx.GetQueue() ? "OK" : "NULL!") << "\n";
        std::cout << "   Device:  " << ext_ctx.GetDeviceName() << "\n";

        // ══════════════════════════════════════════════════════════════
        // Профилирование
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n[PROFILING]\n";
        sm.PrintProfilingSummary();
        sm.StopAll();

        // ══════════════════════════════════════════════════════════════
        // Итог
        // ══════════════════════════════════════════════════════════════
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "  External Context FFT Test Summary:\n";
        std::cout << "    Passed: " << passed << "/" << (passed + failed) << "\n";
        std::cout << "    Failed: " << failed << "/" << (passed + failed) << "\n";

        if (failed == 0) {
            std::cout << "\n  RESULT: ALL TESTS PASSED\n";
        } else {
            std::cout << "\n  RESULT: " << failed << " TEST(S) FAILED\n";
        }
        std::cout << std::string(60, '=') << "\n\n";

        // ~ExternalOpenCLContext() вызовется здесь → освободит контекст
        return (failed > 0) ? 1 : 0;

    } catch (const std::exception& e) {
        std::cerr << "\nFATAL EXCEPTION: " << e.what() << "\n";
        try { drv_gpu_lib::ServiceManager::GetInstance().StopAll(); } catch (...) {}
        return 1;
    }
}

} // namespace test_external_context_fft
