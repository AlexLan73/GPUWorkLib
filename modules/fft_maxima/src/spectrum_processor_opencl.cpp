/**
 * @file spectrum_maxima_finder.cpp
 * @brief Реализация SpectrumProcessorOpenCL — поиск максимумов спектра FFT на GPU
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ОГЛАВЛЕНИЕ (~1090 строк)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * ЧАСТЬ 1: Конструкторы, деструктор, move
 * ЧАСТЬ 2: Публичный API — Initialize(), PrintInfo()
 * ЧАСТЬ 3: Batch Processing — ProcessBatch(), ReallocateBuffersForBatch()
 * ЧАСТЬ 4: Утилиты — CalculateFFTSize(), NextPowerOf2(), CalculateBytesPerAntenna()
 * ЧАСТЬ 5: GPU буферы — AllocateBuffers(), CreateFFTPlanWithCallback(), CompilePostKernel()
 * ЧАСТЬ 6: GPU операции — UploadData(), ExecuteFFT(), ExecutePostKernel(), ReadResults()
 * ЧАСТЬ 7: Вспомогательные — WritePreCallbackHeader(), ReleaseResources()
 *
 * СВЯЗАННЫЕ ФАЙЛЫ:
 *   spectrum_maxima_finder_process.cpp     — Process<T> dispatch (ProcessFromCPU/GPU)
 *   spectrum_maxima_finder_all_maxima.cpp  — FindAllMaxima pipeline (detect→scan→compact)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-06
 */

#include "processors/spectrum_processor_opencl.hpp"
#include "interface/i_backend.hpp"
#include "backends/opencl/opencl_profiling.hpp"
#include "services/gpu_profiler.hpp"
#include "services/console_output.hpp"
#include "services/batch_manager.hpp"
#include "kernels/all_maxima_kernel_sources.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 1: КОНСТРУКТОРЫ И ДЕСТРУКТОРЫ
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Конструктор — получает OpenCL ресурсы из backend, создаёт AllMaxima pipeline
 *
 * Аналог конструктора SpectrumMaximaFinder, но дополнительно создаёт
 * pipeline_ = AllMaximaPipelineOpenCL(context, queue, device) для AllMaxima операций.
 * pipeline_ содержит detect/scan/compact kernels (lazy compiled при первом Execute).
 *
 * backend_ не владеет — lifetime backend > processor.
 *
 * @param backend Инициализированный IBackend (не nullptr, не владеем)
 * @throws std::invalid_argument если backend == nullptr
 * @throws std::runtime_error   если backend не инициализирован или нет OpenCL ресурсов
 */
SpectrumProcessorOpenCL::SpectrumProcessorOpenCL(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

    if (!backend_) {
        throw std::invalid_argument("SpectrumProcessorOpenCL: backend cannot be null");
    }

    if (!backend_->IsInitialized()) {
        throw std::runtime_error("SpectrumProcessorOpenCL: backend is not initialized");
    }

    // Получаем OpenCL ресурсы из backend
    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    queue_ = static_cast<cl_command_queue>(backend_->GetNativeQueue());
    device_ = static_cast<cl_device_id>(backend_->GetNativeDevice());

    if (!context_ || !queue_ || !device_) {
        throw std::runtime_error("SpectrumProcessorOpenCL: failed to get OpenCL resources from backend");
    }

    pipeline_ = std::make_unique<AllMaximaPipelineOpenCL>(context_, queue_, device_,
        backend_ ? backend_->GetDeviceIndex() : 0);
}

/**
 * @brief Деструктор — ReleaseResources() + pipeline_ уничтожается автоматически (unique_ptr)
 */
SpectrumProcessorOpenCL::~SpectrumProcessorOpenCL() {
    ReleaseResources();
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 2: ПУБЛИЧНЫЙ API — Initialize()
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Принять params и инициализировать GPU ресурсы: буферы, clFFT план, post-kernel
 *
 * Аналог SpectrumMaximaFinder::Initialize(), но принимает SpectrumParams явно (ISpectrumProcessor API).
 * params копируется в params_. Если Initialize уже был вызван — выходим немедленно (guard).
 *
 * Стратегия выделения:
 * - Если все антенны помещаются → AllocateBuffers() + CreateFFTPlanWithCallback() + CompilePostKernel()
 * - Иначе batch mode → ReallocateBuffersForBatch(max_batch_size) + CompilePostKernel()
 *
 * @param params SpectrumParams с конфигурацией (antenna_count, n_point, repeat_count, etc.)
 * @throws std::runtime_error если любой шаг инициализации провалился
 */
void SpectrumProcessorOpenCL::Initialize(const SpectrumParams& params) {
    params_ = params;
    if (initialized_) {
        return;
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "SpectrumMaxima", "Инициализация...");

    // 1. Вычислить размеры FFT
    CalculateFFTSize();

    con.Print(0, "SpectrumMaxima",
        "antenna_count=" + std::to_string(params_.antenna_count) +
        " n_point=" + std::to_string(params_.n_point) +
        " repeat_count=" + std::to_string(params_.repeat_count) +
        " nFFT=" + std::to_string(params_.nFFT) +
        " search_range=" + std::to_string(params_.search_range) +
        " sample_rate=" + std::to_string(static_cast<int>(params_.sample_rate)) + " Hz");

    // 2. Проверяем, нужен ли batch processing
    //    Используем ту же формулу что и CalculateBytesPerAntenna()
    size_t bytes_per_antenna = params_.n_point * sizeof(std::complex<float>) +
                               3 * params_.nFFT * sizeof(std::complex<float>) +  // input+output+temp
                               ((params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8) * sizeof(MaxValue);

    bool need_batch = !drv_gpu_lib::BatchManager::AllItemsFit(
        backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

    if (need_batch) {
        // ═══════════════════════════════════════════════════════════════════
        // Оптимизация: сразу резервируем буферы под максимальный batch
        // Это экономит время на перевыделение буферов между batch-ами
        // ═══════════════════════════════════════════════════════════════════
        size_t max_batch_size = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
            backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

        con.Print(0, "SpectrumMaxima", "[Batch mode] max batch=" + std::to_string(max_batch_size));
        ReallocateBuffersForBatch(max_batch_size);
        CompilePostKernel();
        con.Print(0, "SpectrumMaxima", "Буферы + PostKernel готовы (batch mode)");
    } else {
        AllocateBuffers();
        CreateFFTPlanWithCallback();
        CompilePostKernel();
        con.Print(0, "SpectrumMaxima", "Буферы + FFT план + PostKernel готовы");

        current_batch_size_ = params_.antenna_count;
        actual_batch_size_ = params_.antenna_count;
    }

    initialized_ = true;
    con.Print(0, "SpectrumMaxima", "Инициализация завершена");
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 3: BATCH PROCESSING — ProcessBatch(), ReallocateBuffersForBatch()
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Обработать один batch: Upload → FFT → PostKernel → ReadResults, с profiling
 *
 * Отличие от SpectrumMaximaFinder::ProcessBatch: использует RecordProfilingEvent (не CollectOrRelease).
 * Это потому что SpectrumProcessorOpenCL работает без ProfEvents* параметра — профилирование
 * через GPUProfiler::Record внутри, не через аккумуляцию событий в вектор.
 *
 * @param input_data          Весь массив (не только batch)
 * @param start_antenna       Абсолютный индекс первой антенны
 * @param batch_antenna_count Число антенн в batch
 * @return vector<SpectrumResult> с корректными antenna_id
 */
std::vector<SpectrumResult> SpectrumProcessorOpenCL::ProcessBatch(
    const std::vector<std::complex<float>>& input_data,
    size_t start_antenna,
    size_t batch_antenna_count) {

    // Перевыделить буферы под текущий batch (если нужно)
    // current_batch_size_ — размер буферов, actual_batch_size_ — реальный batch
    if (batch_antenna_count > current_batch_size_ || !plan_created_) {
        ReallocateBuffersForBatch(batch_antenna_count);
    }
    actual_batch_size_ = batch_antenna_count;  // Запоминаем реальный размер batch

    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;

    // Извлечь срез данных для текущего batch
    size_t offset = start_antenna * params_.n_point;
    size_t count = batch_antenna_count * params_.n_point;
    std::vector<std::complex<float>> batch_data(
        input_data.begin() + offset,
        input_data.begin() + offset + count);

    // 1. Загрузить данные batch на GPU
    cl_event upload_event = UploadData(batch_data);
    drv_gpu_lib::RecordProfilingEvent(upload_event, gpu_id, "SpectrumMaxima", "Upload");

    // 2. Выполнить FFT
    cl_event fft_event = ExecuteFFT(upload_event);
    clReleaseEvent(upload_event);
    drv_gpu_lib::RecordProfilingEvent(fft_event, gpu_id, "SpectrumMaxima", "FFT");

    // 3. Выполнить post-kernel
    cl_event post_event = ExecutePostKernel(fft_event);
    clReleaseEvent(fft_event);
    drv_gpu_lib::RecordProfilingEvent(post_event, gpu_id, "SpectrumMaxima", "PostKernel");

    // 4. Прочитать результаты batch
    std::vector<SpectrumResult> batch_results = ReadResults(post_event);
    clReleaseEvent(post_event);

    // Скорректировать antenna_id для результатов (добавить start_antenna)
    for (auto& result : batch_results) {
        result.antenna_id += start_antenna;
    }

    return batch_results;
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessFromCPU — обработка CPU-данных (vector<complex<float>>)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief ISpectrumProcessor::ProcessFromCPU — сырой сигнал с CPU → GPU → результат
 *
 * Реализация интерфейса ISpectrumProcessor. Аналог ProcessFromCPU в SpectrumMaximaFinder,
 * но без ProfEvents* — профилирование через RecordProfilingEvent внутри ProcessBatch.
 *
 * @param data Плоский массив [antenna_count × n_point] complex<float>
 * @return vector<SpectrumResult>[antenna_count]
 * @throws std::runtime_error если не инициализирован или размер не совпадает
 */
std::vector<SpectrumResult> SpectrumProcessorOpenCL::ProcessFromCPU(
    const std::vector<std::complex<float>>& data)
{
    if (!initialized_) {
        throw std::runtime_error("SpectrumProcessorOpenCL::ProcessFromCPU: not initialized");
    }

    size_t expected_size = params_.antenna_count * params_.n_point;
    if (data.size() != expected_size) {
        throw std::invalid_argument(
            "SpectrumProcessorOpenCL::ProcessFromCPU: input size mismatch. "
            "Expected " + std::to_string(expected_size) +
            ", got " + std::to_string(data.size()));
    }

    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    if (drv_gpu_lib::BatchManager::AllItemsFit(backend_, params_.antenna_count,
                                                bytes_per_antenna, params_.memory_limit)) {
        auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
        con.Print(0, "SpectrumMaxima", "Все " + std::to_string(params_.antenna_count) +
            " антенн помещаются в память — batch не нужен");
        return ProcessBatch(data, 0, params_.antenna_count);
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

    size_t batch_size = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params_.antenna_count, batch_size, 3, true);

    con.Print(0, "SpectrumMaxima", "Batch Processing: " +
        std::to_string(bytes_per_antenna / 1024 / 1024) + " MB/antenna, limit=" +
        std::to_string(static_cast<int>(params_.memory_limit * 100)) + "%");
    drv_gpu_lib::BatchManager::PrintBatchInfo(batches, params_.antenna_count);

    std::vector<SpectrumResult> all_results;
    all_results.reserve(params_.antenna_count);

    for (const auto& batch : batches) {
        con.Print(0, "SpectrumMaxima", "Batch " + std::to_string(batch.batch_idx) +
            ": antennas [" + std::to_string(batch.start) + ".." +
            std::to_string(batch.start + batch.count - 1) + "]");

        auto batch_results = ProcessBatch(data, batch.start, batch.count);
        all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
    }

    con.Print(0, "SpectrumMaxima", "Batch completed: " + std::to_string(all_results.size()) + " results");

    return all_results;
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessFromGPU — обработка GPU-данных (void* → cl_mem)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief ISpectrumProcessor::ProcessFromGPU — данные уже на GPU (void* = cl_mem)
 *
 * gpu_data: void* для соответствия интерфейсу ISpectrumProcessor (backend-agnostic),
 * внутри приводится к cl_mem. Использует G2G CopyBuffer вместо Upload.
 *
 * external_memory учитывается при расчёте batch (gpu_data уже занимает VRAM).
 *
 * @param gpu_data        Указатель-cl_mem [antenna_count × n_point × complex<float>]
 * @param antenna_count   Число антенн
 * @param n_point         Точек на антенну
 * @param gpu_memory_bytes Размер gpu_data (0 = авто = antenna_count × n_point × 8)
 * @return vector<SpectrumResult>[antenna_count]
 * @throws std::invalid_argument если gpu_data == nullptr
 */
std::vector<SpectrumResult> SpectrumProcessorOpenCL::ProcessFromGPU(
    void* gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes)
{
    cl_mem cl_data = static_cast<cl_mem>(gpu_data);
    if (!cl_data) {
        throw std::invalid_argument("ProcessFromGPU: gpu_data cannot be null");
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "SpectrumMaxima", "ProcessFromGPU: " +
        std::to_string(antenna_count) + " antennas, n_point=" + std::to_string(n_point));

    if (params_.nFFT == 0) {
        CalculateFFTSize();
        con.Print(0, "SpectrumMaxima", "nFFT=" + std::to_string(params_.nFFT) +
            " search_range=" + std::to_string(params_.search_range));
    }

    if (!post_kernel_) {
        CompilePostKernel();
    }

    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : antenna_count * n_point * sizeof(std::complex<float>);

    con.Print(0, "SpectrumMaxima", "External data: " +
        std::to_string(external_memory / 1024 / 1024) + " MB on GPU");

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, antenna_count, bytes_per_antenna,
        params_.memory_limit, external_memory);

    bool need_batch = (optimal_batch < antenna_count);

    if (!need_batch) {
        con.Print(0, "SpectrumMaxima", "Single batch — все данные помещаются");

        ReallocateBuffersForBatch(antenna_count);
        actual_batch_size_ = antenna_count;

        size_t data_size = antenna_count * n_point * sizeof(std::complex<float>);
        cl_event copy_event = nullptr;
        cl_int err = clEnqueueCopyBuffer(queue_,
            cl_data, pre_callback_userdata_,
            0, PRE_CALLBACK_HEADER_SIZE,
            data_size,
            0, nullptr, &copy_event);

        if (err != CL_SUCCESS) {
            throw std::runtime_error("ProcessFromGPU: clEnqueueCopyBuffer failed: " + std::to_string(err));
        }

        cl_event fft_event = ExecuteFFT(copy_event);
        cl_event post_event = ExecutePostKernel(fft_event);
        auto results = ReadResults(post_event);

        clReleaseEvent(copy_event);
        clReleaseEvent(fft_event);
        clReleaseEvent(post_event);

        return results;
    }

    con.Print(0, "SpectrumMaxima", "Batch mode: batch_size=" + std::to_string(optimal_batch));

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(antenna_count, optimal_batch);
    drv_gpu_lib::BatchManager::PrintBatchInfo(batches, antenna_count);

    ReallocateBuffersForBatch(optimal_batch);

    std::vector<SpectrumResult> all_results;
    all_results.reserve(antenna_count);

    for (const auto& batch : batches) {
        con.Print(0, "SpectrumMaxima", "Batch [" + std::to_string(batch.start) +
            ".." + std::to_string(batch.start + batch.count - 1) + "]");

        size_t src_offset = batch.start * n_point * sizeof(std::complex<float>);
        auto batch_results = ProcessBatchFromGPU(gpu_data, src_offset,
                                                  batch.start, batch.count);

        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    con.Print(0, "SpectrumMaxima", "GPU Batch completed: " + std::to_string(all_results.size()) + " results");
    return all_results;
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessBatchFromGPU — обработка одного batch с GPU-данными
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Обработать один batch из внешнего GPU буфера (void* = cl_mem)
 * G2G CopyBuffer(src_offset_bytes) → ExecuteFFT → ExecutePostKernel → ReadResults.
 * antenna_id корректируется на start_antenna. Использует RecordProfilingEvent (не ProfEvents*).
 */
std::vector<SpectrumResult> SpectrumProcessorOpenCL::ProcessBatchFromGPU(
    void* gpu_data, size_t src_offset_bytes,
    size_t start_antenna, size_t batch_antenna_count)
{
    cl_mem cl_data = static_cast<cl_mem>(gpu_data);

    if (batch_antenna_count > current_batch_size_ || !plan_created_) {
        ReallocateBuffersForBatch(batch_antenna_count);
    }
    actual_batch_size_ = batch_antenna_count;

    WritePreCallbackHeader(batch_antenna_count);

    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;

    size_t data_size = batch_antenna_count * params_.n_point * sizeof(std::complex<float>);
    cl_event copy_event = nullptr;
    cl_int err = clEnqueueCopyBuffer(queue_,
        cl_data, pre_callback_userdata_,
        src_offset_bytes, PRE_CALLBACK_HEADER_SIZE,
        data_size,
        0, nullptr, &copy_event);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("ProcessBatchFromGPU: clEnqueueCopyBuffer failed: " + std::to_string(err));
    }

    drv_gpu_lib::RecordProfilingEvent(copy_event, gpu_id, "SpectrumMaxima", "GPU→GPU Copy");

    cl_event fft_event = ExecuteFFT(copy_event);
    clReleaseEvent(copy_event);
    drv_gpu_lib::RecordProfilingEvent(fft_event, gpu_id, "SpectrumMaxima", "FFT");

    cl_event post_event = ExecutePostKernel(fft_event);
    clReleaseEvent(fft_event);
    drv_gpu_lib::RecordProfilingEvent(post_event, gpu_id, "SpectrumMaxima", "PostKernel");

    auto results = ReadResults(post_event);
    clReleaseEvent(post_event);

    for (size_t i = 0; i < results.size(); ++i) {
        results[i].antenna_id = static_cast<uint32_t>(start_antenna + i);
    }

    return results;
}

/**
 * @brief Выделить или переиспользовать буферы и FFT план для batch_antenna_count антенн
 * Идентичен SpectrumMaximaFinder::ReallocateBuffersForBatch — см. документацию там.
 * FAST PATH / PARTIAL PATH / FULL PATH — те же три пути оптимизации.
 */
void SpectrumProcessorOpenCL::ReallocateBuffersForBatch(size_t batch_antenna_count) {
    cl_int err;

    // ═══════════════════════════════════════════════════════════════════════
    // 0. Инициализация clFFT (один раз на экземпляр, НЕ static!)
    // ═══════════════════════════════════════════════════════════════════════
    if (!clfft_initialized_) {
        clfftSetupData setup;
        setup.major = clfftVersionMajor;
        setup.minor = clfftVersionMinor;
        setup.patch = clfftVersionPatch;
        setup.debugFlags = 0;
        clfftSetup(&setup);
        clfft_initialized_ = true;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Оптимизация: переиспользование FFT плана и буферов
    // Буферы можно переиспользовать если current_batch_size_ >= batch_antenna_count
    // План можно переиспользовать ТОЛЬКО если batch size точно совпадает
    // ═══════════════════════════════════════════════════════════════════════
    bool need_new_plan = (plan_batch_size_ != batch_antenna_count) || !plan_created_;
    bool need_new_buffers = (current_batch_size_ < batch_antenna_count) || !pre_callback_userdata_;

    // ═══════════════════════════════════════════════════════════════════════
    // FAST PATH: буферы достаточно большие, план совпадает — только заголовок
    // ═══════════════════════════════════════════════════════════════════════
    if (!need_new_buffers && !need_new_plan) {
        try {
            WritePreCallbackHeader(batch_antenna_count);
            return;  // Fast path success!
        } catch (...) {
            // Если failed — идём в полное перевыделение
            need_new_buffers = true;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. Освободить и создать буферы только если нужны новые
    // ═══════════════════════════════════════════════════════════════════════
    if (need_new_buffers) {
        // Освобождаем старые буферы
        if (pre_callback_userdata_) {
            clReleaseMemObject(pre_callback_userdata_);
            pre_callback_userdata_ = nullptr;
        }
        if (fft_input_) {
            clReleaseMemObject(fft_input_);
            fft_input_ = nullptr;
        }
        if (fft_output_) {
            clReleaseMemObject(fft_output_);
            fft_output_ = nullptr;
        }
        if (maxima_output_) {
            clReleaseMemObject(maxima_output_);
            maxima_output_ = nullptr;
        }
        // fft_temp_buffer_, pinned_staging_buffer_ НЕ освобождаем — переиспользуем

        // ═══════════════════════════════════════════════════════════════════
        // 2. Создать новые буферы под batch_antenna_count антенн
        // ═══════════════════════════════════════════════════════════════════

        // 2.1 Pre-callback userdata
        size_t input_data_size = batch_antenna_count * params_.n_point * sizeof(std::complex<float>);
        size_t userdata_size = PRE_CALLBACK_HEADER_SIZE + input_data_size;

        pre_callback_userdata_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                                 userdata_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("ReallocateBuffersForBatch: pre_callback_userdata failed: " + std::to_string(err));
        }

        // Записать заголовок (beam_count = batch_antenna_count!)
        WritePreCallbackHeader(batch_antenna_count);

        // 2.2 FFT буферы
        size_t fft_buffer_size = batch_antenna_count * params_.nFFT * sizeof(std::complex<float>);

        fft_input_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, fft_buffer_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("ReallocateBuffersForBatch: fft_input failed: " + std::to_string(err));
        }

        fft_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, fft_buffer_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("ReallocateBuffersForBatch: fft_output failed: " + std::to_string(err));
        }

        // 2.3 Maxima output
        size_t max_values_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
        size_t maxima_size = batch_antenna_count * max_values_per_beam * sizeof(MaxValue);

        maxima_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, maxima_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("ReallocateBuffersForBatch: maxima_output failed: " + std::to_string(err));
        }

        current_batch_size_ = batch_antenna_count;
    }

    // Уничтожить старый FFT план только если нужен новый
    if (need_new_plan && plan_created_ && plan_handle_) {
        clfftDestroyPlan(&plan_handle_);
        plan_handle_ = 0;
        plan_created_ = false;
        plan_batch_size_ = 0;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Создать новый FFT план под batch_antenna_count (если нужен)
    //    Экономим ~50-200ms на bake когда batch size не меняется!
    // ═══════════════════════════════════════════════════════════════════════
    if (need_new_plan) {
        size_t dim = params_.nFFT;

        clfftStatus status = clfftCreateDefaultPlan(&plan_handle_, context_, CLFFT_1D, &dim);
        if (status != CLFFT_SUCCESS) {
            throw std::runtime_error("ReallocateBuffersForBatch: clfftCreateDefaultPlan failed: " + std::to_string(status));
        }

        clfftSetPlanPrecision(plan_handle_, CLFFT_SINGLE);
        clfftSetLayout(plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
        clfftSetResultLocation(plan_handle_, CLFFT_OUTOFPLACE);
        clfftSetPlanBatchSize(plan_handle_, batch_antenna_count);  // ← batch size!

        size_t strides[1] = {1};
        size_t dist = params_.nFFT;
        clfftSetPlanInStride(plan_handle_, CLFFT_1D, strides);
        clfftSetPlanOutStride(plan_handle_, CLFFT_1D, strides);
        clfftSetPlanDistance(plan_handle_, dist, dist);

        // Регистрировать pre-callback
        const char* pre_callback_source = kernels::GetPreCallbackSource32_opencl();
        status = clfftSetPlanCallback(plan_handle_, "prepareDataPre", pre_callback_source, 0,
                                       PRECALLBACK, &pre_callback_userdata_, 1);
        if (status != CLFFT_SUCCESS) {
            clfftDestroyPlan(&plan_handle_);
            throw std::runtime_error("ReallocateBuffersForBatch: clfftSetPlanCallback failed: " + std::to_string(status));
        }

        // Bake план
        status = clfftBakePlan(plan_handle_, 1, &queue_, nullptr, nullptr);
        if (status != CLFFT_SUCCESS) {
            clfftDestroyPlan(&plan_handle_);
            throw std::runtime_error("ReallocateBuffersForBatch: clfftBakePlan failed: " + std::to_string(status));
        }

        plan_batch_size_ = batch_antenna_count;
        plan_created_ = true;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. Создать/переиспользовать временный буфер для FFT
    //    Переиспользуем существующий буфер если он достаточного размера!
    // ═══════════════════════════════════════════════════════════════════════
    size_t tmp_buf_size = 0;
    clfftStatus tmp_status = clfftGetTmpBufSize(plan_handle_, &tmp_buf_size);
    if (tmp_status == CLFFT_SUCCESS && tmp_buf_size > 0) {
        // Нужен ли новый буфер? Только если старый меньше требуемого
        if (tmp_buf_size > fft_temp_buffer_size_) {
            // Освобождаем старый если есть
            if (fft_temp_buffer_) {
                clReleaseMemObject(fft_temp_buffer_);
                fft_temp_buffer_ = nullptr;
            }
            // Создаём новый
            fft_temp_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, tmp_buf_size, nullptr, &err);
            if (err != CL_SUCCESS) {
                clfftDestroyPlan(&plan_handle_);
                throw std::runtime_error("ReallocateBuffersForBatch: fft_temp_buffer failed: " + std::to_string(err));
            }
            fft_temp_buffer_size_ = tmp_buf_size;
        }
        // Иначе переиспользуем существующий fft_temp_buffer_
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. Создать/переиспользовать Pinned Buffer для быстрого upload (DMA)
    //    CL_MEM_ALLOC_HOST_PTR создаёт page-locked память на хосте
    // ═══════════════════════════════════════════════════════════════════════
    size_t required_pinned_size = batch_antenna_count * params_.n_point * sizeof(std::complex<float>);
    if (required_pinned_size > pinned_buffer_size_) {
        // Освобождаем старый если есть
        if (pinned_staging_buffer_) {
            clReleaseMemObject(pinned_staging_buffer_);
            pinned_staging_buffer_ = nullptr;
        }
        // Создаём новый pinned buffer
        pinned_staging_buffer_ = clCreateBuffer(
            context_,
            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,  // Pinned memory на хосте
            required_pinned_size,
            nullptr,
            &err
        );
        if (err != CL_SUCCESS) {
            // Pinned memory не критична — продолжаем без неё
            drv_gpu_lib::ConsoleOutput::GetInstance().PrintWarning(0, "SpectrumMaxima",
                "Failed to create pinned buffer: " + std::to_string(err) + ", using regular upload");
            pinned_staging_buffer_ = nullptr;
            pinned_buffer_size_ = 0;
        } else {
            pinned_buffer_size_ = required_pinned_size;
        }
    }

    // current_batch_size_ уже обновлён внутри if (need_new_buffers)
    // Не обновляем здесь, чтобы сохранить реальный размер выделенных буферов
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 4: УТИЛИТЫ — CalculateFFTSize(), NextPowerOf2(), CalculateBytesPerAntenna()
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Вычислить base_fft, nFFT, search_range в params_
 * base_fft = nextPow2(n_point), nFFT = base_fft × repeat_count, search_range=0→nFFT/4.
 */
void SpectrumProcessorOpenCL::CalculateFFTSize() {
    // base_fft = следующая степень двойки от n_point
    params_.base_fft = NextPowerOf2(params_.n_point);

    // nFFT = base_fft * repeat_count
    params_.nFFT = params_.base_fft * params_.repeat_count;

    // search_range по умолчанию = nFFT / 4
    if (params_.search_range == 0) {
        params_.search_range = params_.nFFT / 4;
    }
}

/**
 * @brief Следующая степень двойки ≥ n (bit-twiddling, O(1))
 * hipFFT/clFFT оптимально работают со степенями двойки.
 */
uint32_t SpectrumProcessorOpenCL::NextPowerOf2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

/**
 * @brief Оценить потребление GPU памяти на одну антенну (для BatchManager)
 * Упрощённая формула (без ALL_MAXIMA branch) т.к. SpectrumProcessorOpenCL не поддерживает
 * peak_mode=ALL_MAXIMA напрямую — только через FindAllMaxima() и AllMaximaFromCPU().
 */
size_t SpectrumProcessorOpenCL::CalculateBytesPerAntenna() const {
    // Формула памяти на одну антенну для BatchManager:
    // 1. Input data: n_point * sizeof(complex<float>)
    size_t input_bytes = params_.n_point * sizeof(std::complex<float>);  // n_point * 8

    // 2. FFT buffers (input + output + TEMP): 3 * nFFT * sizeof(complex<float>)
    //    clFFT требует temp buffer размером примерно batch * nFFT * 8!
    size_t fft_bytes = 3 * params_.nFFT * sizeof(std::complex<float>);  // 3 * nFFT * 8

    // 3. Maxima output: (4 or 8) * sizeof(MaxValue)
    size_t maxima_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
    size_t maxima_bytes = maxima_per_beam * sizeof(MaxValue);  // 4*32=128 or 8*32=256

    return input_bytes + fft_bytes + maxima_bytes;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 5: GPU БУФЕРЫ (single-batch режим)
// AllocateBuffers(), CreateFFTPlanWithCallback(), CompilePostKernel()
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Выделить GPU буферы для single-batch (все помещаются в VRAM)
 * Аналог SpectrumMaximaFinder::AllocateBuffers() — см. документацию там.
 */
void SpectrumProcessorOpenCL::AllocateBuffers() {
    cl_int err;

    // 1. Pre-callback userdata: [32 bytes header][input data]
    // Header: {beam_count, count_points, nFFT, pad, pad, pad, pad, pad}
    size_t input_data_size = params_.antenna_count * params_.n_point * sizeof(std::complex<float>);
    size_t userdata_size = PRE_CALLBACK_HEADER_SIZE + input_data_size;

    pre_callback_userdata_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                             userdata_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create pre_callback_userdata buffer: " + std::to_string(err));
    }

    // Записать заголовок с параметрами для GPU
    WritePreCallbackHeader(params_.antenna_count);

    // 2. FFT буферы
    size_t fft_buffer_size = params_.antenna_count * params_.nFFT * sizeof(std::complex<float>);

    fft_input_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                 fft_buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create fft_input buffer: " + std::to_string(err));
    }

    fft_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                  fft_buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create fft_output buffer: " + std::to_string(err));
    }

    // 3. Maxima output: размер зависит от режима поиска пиков
    //    ONE_PEAK: 4 MaxValue на луч
    //    TWO_PEAKS: 8 MaxValue на луч
    size_t max_values_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
    size_t maxima_size = params_.antenna_count * max_values_per_beam * sizeof(MaxValue);
    static_assert(sizeof(MaxValue) == 32, "MaxValue must be 32 bytes");

    maxima_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                     maxima_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create maxima_output buffer: " + std::to_string(err));
    }
}

/**
 * @brief Создать clFFT план с pre-callback (zero-padding) для single-batch режима
 * Аналог SpectrumMaximaFinder::CreateFFTPlanWithCallback() — см. документацию там.
 */
void SpectrumProcessorOpenCL::CreateFFTPlanWithCallback() {
    // Инициализация clFFT (один раз на экземпляр, НЕ static для multi-GPU!)
    if (!clfft_initialized_) {
        clfftSetupData setup;
        setup.major = clfftVersionMajor;
        setup.minor = clfftVersionMinor;
        setup.patch = clfftVersionPatch;
        setup.debugFlags = 0;
        clfftSetup(&setup);
        clfft_initialized_ = true;
    }

    // Создать план
    size_t dim = params_.nFFT;
    clfftStatus status = clfftCreateDefaultPlan(&plan_handle_, context_, CLFFT_1D, &dim);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed: " + std::to_string(status));
    }

    // Настроить план
    clfftSetPlanPrecision(plan_handle_, CLFFT_SINGLE);
    clfftSetLayout(plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle_, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle_, params_.antenna_count);

    size_t strides[1] = {1};
    size_t dist = params_.nFFT;
    clfftSetPlanInStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle_, dist, dist);

    // Регистрировать pre-callback
    const char* pre_callback_source = kernels::GetPreCallbackSource32_opencl();
    status = clfftSetPlanCallback(plan_handle_, "prepareDataPre", pre_callback_source, 0,
                                   PRECALLBACK, &pre_callback_userdata_, 1);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftSetPlanCallback (pre) failed: " + std::to_string(status));
    }

    // Bake план
    status = clfftBakePlan(plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftBakePlan failed: " + std::to_string(status));
    }

    plan_created_ = true;
}

/**
 * @brief Скомпилировать post-kernel (peak search: ONE_PEAK или TWO_PEAKS)
 * Аналог SpectrumMaximaFinder::CompilePostKernel() — см. документацию там.
 */
void SpectrumProcessorOpenCL::CompilePostKernel() {
    cl_int err;

    // Выбор кернела в зависимости от режима поиска пиков
    const char* source;
    const char* kernel_name;

    if (params_.peak_mode == PeakSearchMode::ONE_PEAK) {
        source = kernels::GetPostKernelSource_OnePeak_opencl();
        kernel_name = "post_kernel_one_peak";
        drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "SpectrumMaxima", "Peak mode: ONE_PEAK (4 MaxValue/beam)");
    } else {
        source = kernels::GetPostKernelSource_TwoPeaks_opencl();
        kernel_name = "post_kernel";
        drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "SpectrumMaxima", "Peak mode: TWO_PEAKS (8 MaxValue/beam)");
    }

    size_t source_len = strlen(source);

    // Создать программу
    post_program_ = clCreateProgramWithSource(context_, 1, &source, &source_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clCreateProgramWithSource failed: " + std::to_string(err));
    }

    // Скомпилировать
    err = clBuildProgram(post_program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Получить лог ошибок
        size_t log_size;
        clGetProgramBuildInfo(post_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(post_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        drv_gpu_lib::ConsoleOutput::GetInstance().PrintError(0, "SpectrumMaxima", "Build log: " + std::string(log.data()));
        clReleaseProgram(post_program_);
        post_program_ = nullptr;
        throw std::runtime_error("clBuildProgram failed: " + std::to_string(err));
    }

    // Создать kernel
    post_kernel_ = clCreateKernel(post_program_, kernel_name, &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(post_program_);
        post_program_ = nullptr;
        throw std::runtime_error("clCreateKernel failed: " + std::to_string(err));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 6: GPU ОПЕРАЦИИ — UploadData(), ExecuteFFT(), ExecutePostKernel(), ReadResults()
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief CPU→GPU upload через Pinned Memory (DMA) или fallback WriteBuffer
 * Аналог SpectrumMaximaFinder::UploadData() — см. документацию там.
 */
cl_event SpectrumProcessorOpenCL::UploadData(const std::vector<std::complex<float>>& input_data) {
    cl_event event = nullptr;
    size_t data_size = input_data.size() * sizeof(std::complex<float>);
    cl_int err;

    // ═══════════════════════════════════════════════════════════════════════
    // Оптимизация: Pinned Memory для быстрого DMA transfer
    // 1. Map pinned buffer → получаем page-locked указатель
    // 2. memcpy данные туда (быстрее чем pageable)
    // 3. Unmap → запускает DMA transfer
    // 4. Copy из pinned staging в целевой буфер
    // ═══════════════════════════════════════════════════════════════════════
    if (pinned_staging_buffer_ && data_size <= pinned_buffer_size_) {
        // Используем pinned memory для DMA
        void* mapped_ptr = clEnqueueMapBuffer(
            queue_,
            pinned_staging_buffer_,
            CL_TRUE,  // Blocking map
            CL_MAP_WRITE_INVALIDATE_REGION,
            0,
            data_size,
            0, nullptr, nullptr,
            &err
        );

        if (err == CL_SUCCESS && mapped_ptr) {
            // Копируем данные в pinned память (быстрее чем в pageable)
            std::memcpy(mapped_ptr, input_data.data(), data_size);

            // Unmap → это НЕ запускает transfer, просто разблокирует буфер
            clEnqueueUnmapMemObject(queue_, pinned_staging_buffer_, mapped_ptr, 0, nullptr, nullptr);

            // Копируем из pinned staging в целевой буфер (DMA transfer)
            err = clEnqueueCopyBuffer(
                queue_,
                pinned_staging_buffer_,
                pre_callback_userdata_,
                0,                              // src offset
                PRE_CALLBACK_HEADER_SIZE,       // dst offset (после заголовка)
                data_size,
                0, nullptr,
                &event
            );

            if (err == CL_SUCCESS) {
                return event;  // Успех с pinned memory
            }
            // Fallback: если copy failed, используем обычный путь
        }
        // Fallback: если map failed, используем обычный WriteBuffer
    }

    // Обычный путь: clEnqueueWriteBuffer (без pinned memory)
    err = clEnqueueWriteBuffer(
        queue_,
        pre_callback_userdata_,
        CL_FALSE,  // Non-blocking
        PRE_CALLBACK_HEADER_SIZE,  // Offset после заголовка
        data_size,
        input_data.data(),
        0, nullptr,
        &event
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("UploadData failed: " + std::to_string(err));
    }

    return event;
}

/**
 * @brief Запустить clFFT с plan_handle_ (pre-callback встроен)
 * Аналог SpectrumMaximaFinder::ExecuteFFT() — см. документацию там.
 */
cl_event SpectrumProcessorOpenCL::ExecuteFFT(cl_event wait_event) {
    cl_event event = nullptr;

    // Выполнить FFT с pre-callback
    // Используем временный буфер, если он был создан
    clfftStatus status = clfftEnqueueTransform(
        plan_handle_,
        CLFFT_FORWARD,
        1, &queue_,
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &event,
        &fft_input_,       // Input (pre-callback читает из userdata)
        &fft_output_,      // Output
        fft_temp_buffer_   // Temp buffer (может быть nullptr)
    );

    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftEnqueueTransform failed: " + std::to_string(status));
    }

    return event;
}

/**
 * @brief Запустить post-kernel (peak search) после FFT
 * Аналог SpectrumMaximaFinder::ExecutePostKernel() — см. документацию там.
 */
cl_event SpectrumProcessorOpenCL::ExecutePostKernel(cl_event wait_event) {
    cl_int err;
    cl_event event = nullptr;

    // Используем actual_batch_size_ — реальный размер текущего batch
    uint32_t antenna_count_for_kernel = (actual_batch_size_ > 0)
        ? static_cast<uint32_t>(actual_batch_size_)
        : params_.antenna_count;

    // Установить аргументы kernel
    err = clSetKernelArg(post_kernel_, 0, sizeof(cl_mem), &fft_output_);
    err |= clSetKernelArg(post_kernel_, 1, sizeof(cl_mem), &maxima_output_);
    err |= clSetKernelArg(post_kernel_, 2, sizeof(uint32_t), &antenna_count_for_kernel);
    err |= clSetKernelArg(post_kernel_, 3, sizeof(uint32_t), &params_.nFFT);
    err |= clSetKernelArg(post_kernel_, 4, sizeof(uint32_t), &params_.search_range);
    err |= clSetKernelArg(post_kernel_, 5, sizeof(float), &params_.sample_rate);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("clSetKernelArg failed: " + std::to_string(err));
    }

    // NDRange: каждая work-group = одна антена
    size_t global_size = antenna_count_for_kernel * LOCAL_SIZE;
    size_t local_size = LOCAL_SIZE;

    err = clEnqueueNDRangeKernel(
        queue_,
        post_kernel_,
        1,
        nullptr,
        &global_size,
        &local_size,
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &event
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("clEnqueueNDRangeKernel failed: " + std::to_string(err));
    }

    return event;
}

/**
 * @brief Читать maxima_output_ с GPU → vector<SpectrumResult>
 * Отличие от SpectrumMaximaFinder::ReadResults: профилирование через RecordProfilingEvent
 * (не CollectOrRelease) — read_event записывается в GPUProfiler["Download"].
 */
std::vector<SpectrumResult> SpectrumProcessorOpenCL::ReadResults(cl_event wait_event) {
    // Количество MaxValue зависит от режима: ONE_PEAK=4, TWO_PEAKS=8
    // Используем actual_batch_size_ — реальный размер текущего batch (не размер буферов!)
    size_t antenna_count_to_read = (actual_batch_size_ > 0) ? actual_batch_size_ : params_.antenna_count;
    size_t max_values_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
    size_t num_results = antenna_count_to_read * max_values_per_beam;
    std::vector<MaxValue> raw_results(num_results);

    cl_event read_event = nullptr;
    cl_int err = clEnqueueReadBuffer(
        queue_,
        maxima_output_,
        CL_FALSE,  // Non-blocking
        0,
        num_results * sizeof(MaxValue),
        raw_results.data(),
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &read_event
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("ReadResults failed: " + std::to_string(err));
    }

    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;
    drv_gpu_lib::RecordProfilingEvent(read_event, gpu_id, "SpectrumMaxima", "Download");
    clReleaseEvent(read_event);

    // Преобразуем в vector<SpectrumResult>
    std::vector<SpectrumResult> results;

    if (params_.peak_mode == PeakSearchMode::ONE_PEAK) {
        // ONE_PEAK: 1 результат на луч (4 MaxValue)
        results.reserve(antenna_count_to_read);

        for (uint32_t i = 0; i < antenna_count_to_read; ++i) {
            size_t base = i * 4;

            SpectrumResult result{};
            result.antenna_id = i;  // Будет скорректирован в ProcessBatch()
            result.interpolated = raw_results[base + 0];
            result.left_point = raw_results[base + 1];
            result.center_point = raw_results[base + 2];
            result.right_point = raw_results[base + 3];
            results.push_back(result);
        }
    } else {
        // TWO_PEAKS: 2 результата на луч (8 MaxValue) — [left, right]
        results.reserve(antenna_count_to_read * 2);

        for (uint32_t i = 0; i < antenna_count_to_read; ++i) {
            size_t base = i * 8;

            SpectrumResult left{};
            left.antenna_id = i;  // Будет скорректирован в ProcessBatch()
            left.interpolated = raw_results[base + 0];
            left.left_point = raw_results[base + 1];
            left.center_point = raw_results[base + 2];
            left.right_point = raw_results[base + 3];
            results.push_back(left);

            SpectrumResult right{};
            right.antenna_id = i;  // Будет скорректирован в ProcessBatch()
            right.interpolated = raw_results[base + 4];
            right.left_point = raw_results[base + 5];
            right.center_point = raw_results[base + 6];
            right.right_point = raw_results[base + 7];
            results.push_back(right);
        }
    }

    return results;
}

ProfilingData SpectrumProcessorOpenCL::GetProfilingData() const {
    ProfilingData out{};
    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;
    auto stats = drv_gpu_lib::GPUProfiler::GetInstance().GetStats(gpu_id);
    auto it = stats.find("SpectrumMaxima");
    if (it == stats.end()) return out;

    const auto& mod = it->second;
    auto ev = [&mod](const char* name) -> double {
        auto e = mod.events.find(name);
        return (e != mod.events.end()) ? e->second.GetAvgTimeMs() : 0.0;
    };
    out.upload_time_ms = ev("Upload") + ev("GPU→GPU Copy");
    out.fft_time_ms = ev("FFT");
    out.post_kernel_time_ms = ev("PostKernel");
    out.download_time_ms = ev("Download");
    out.total_time_ms = mod.GetTotalTimeMs();
    return out;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 7: AllMaxima — EnsureMagnitudesBuffer, CreateAllMaximaFFTPlan, etc.
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Lazy alloc/resize magnitudes_buffer_ (float[beam_count × nFFT])
 * Аналог SpectrumMaximaFinder::EnsureMagnitudesBuffer() — см. документацию там.
 */
void SpectrumProcessorOpenCL::EnsureMagnitudesBuffer(size_t total_elements) {
    if (magnitudes_buffer_ && magnitudes_buffer_size_ >= total_elements) {
        return;
    }

    if (magnitudes_buffer_) {
        clReleaseMemObject(magnitudes_buffer_);
        magnitudes_buffer_ = nullptr;
    }

    cl_int err;
    magnitudes_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
        total_elements * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("EnsureMagnitudesBuffer: alloc failed: " + std::to_string(err));
    }
    magnitudes_buffer_size_ = total_elements;
}

/**
 * @brief Создать clFFT план с pre+post callbacks для AllMaxima (lazy, с переиспользованием)
 * Аналог SpectrumMaximaFinder::CreateAllMaximaFFTPlan() — см. документацию там.
 * Post-callback "computeMagnitudePost" → magnitudes_buffer_ (float[batch×nFFT]).
 */
void SpectrumProcessorOpenCL::CreateAllMaximaFFTPlan(size_t batch_count) {
    if (allmax_plan_created_ && allmax_plan_batch_size_ == batch_count) {
        return;
    }

    if (allmax_plan_created_ && allmax_plan_handle_) {
        clfftDestroyPlan(&allmax_plan_handle_);
        allmax_plan_handle_ = 0;
        allmax_plan_created_ = false;
    }

    if (!clfft_initialized_) {
        clfftSetupData setup;
        setup.major = clfftVersionMajor;
        setup.minor = clfftVersionMinor;
        setup.patch = clfftVersionPatch;
        setup.debugFlags = 0;
        clfftSetup(&setup);
        clfft_initialized_ = true;
    }

    size_t total_fft_elements = batch_count * params_.nFFT;
    EnsureMagnitudesBuffer(total_fft_elements);

    size_t dim = params_.nFFT;
    clfftStatus status = clfftCreateDefaultPlan(&allmax_plan_handle_, context_, CLFFT_1D, &dim);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("CreateAllMaximaFFTPlan: clfftCreateDefaultPlan failed: " +
                                  std::to_string(status));
    }

    clfftSetPlanPrecision(allmax_plan_handle_, CLFFT_SINGLE);
    clfftSetLayout(allmax_plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(allmax_plan_handle_, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(allmax_plan_handle_, batch_count);

    size_t strides[1] = {1};
    size_t dist = params_.nFFT;
    clfftSetPlanInStride(allmax_plan_handle_, CLFFT_1D, strides);
    clfftSetPlanOutStride(allmax_plan_handle_, CLFFT_1D, strides);
    clfftSetPlanDistance(allmax_plan_handle_, dist, dist);

    const char* pre_source = kernels::GetPreCallbackSource32_opencl();
    status = clfftSetPlanCallback(allmax_plan_handle_, "prepareDataPre", pre_source, 0,
                                   PRECALLBACK, &pre_callback_userdata_, 1);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&allmax_plan_handle_);
        throw std::runtime_error("CreateAllMaximaFFTPlan: pre-callback failed: " +
                                  std::to_string(status));
    }

    const char* post_source = kernels::GetPostCallbackMagnitudeSource_opencl();
    status = clfftSetPlanCallback(allmax_plan_handle_, "computeMagnitudePost", post_source, 0,
                                   POSTCALLBACK, &magnitudes_buffer_, 1);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&allmax_plan_handle_);
        throw std::runtime_error("CreateAllMaximaFFTPlan: post-callback failed: " +
                                  std::to_string(status));
    }

    status = clfftBakePlan(allmax_plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&allmax_plan_handle_);
        throw std::runtime_error("CreateAllMaximaFFTPlan: bake failed: " +
                                  std::to_string(status));
    }

    allmax_plan_batch_size_ = batch_count;
    allmax_plan_created_ = true;

    drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "AllMaxima",
        "FFT plan created with pre+post callbacks (batch=" + std::to_string(batch_count) + ")");
}

/**
 * @brief Запустить FFT с allmax_plan_handle_ (pre+post callbacks встроены)
 * Аналог SpectrumMaximaFinder::ExecuteAllMaximaFFT() — magnitudes_buffer_ заполняется post-callback'ом.
 */
cl_event SpectrumProcessorOpenCL::ExecuteAllMaximaFFT(cl_event wait_event) {
    cl_event event = nullptr;

    clfftStatus status = clfftEnqueueTransform(
        allmax_plan_handle_,
        CLFFT_FORWARD,
        1, &queue_,
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &event,
        &fft_input_,
        &fft_output_,
        fft_temp_buffer_
    );

    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("ExecuteAllMaximaFFT: clfftEnqueueTransform failed: " +
                                  std::to_string(status));
    }

    return event;
}

/**
 * @brief IAllMaximaPipeline-compatible: CPU сырой сигнал → FFT → AllMaxima pipeline
 *
 * Аналог SpectrumMaximaFinder::FindAllMaximaFromCPU(), но:
 * - Профилирование через GPUProfiler::Record (не CollectOrRelease)
 * - Без batch mode — бросает runtime_error если данные не помещаются (упрощённая реализация)
 * - FindAllMaxima делегирует в pipeline_->Execute() (Strategy pattern!)
 *
 * @param data  Плоский массив [antenna_count × n_point] complex<float>
 * @param dest  CPU / GPU / ALL
 * @param search_start Начальный бин (0=авто=1)
 * @param search_end   Конечный бин (0=авто=nFFT/2)
 * @return AllMaximaResult; при dest=GPU/ALL — caller ОБЯЗАН освободить gpu_maxima/gpu_counts!
 */
AllMaximaResult SpectrumProcessorOpenCL::FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>& data,
    OutputDestination dest, uint32_t search_start, uint32_t search_end)
{
    if (!initialized_) {
        throw std::runtime_error("FindAllMaximaFromCPU: not initialized");
    }

    size_t expected_size = static_cast<size_t>(params_.antenna_count) * params_.n_point;
    if (data.size() != expected_size) {
        throw std::invalid_argument(
            "FindAllMaximaFromCPU: input size mismatch. Expected " +
            std::to_string(expected_size) + ", got " + std::to_string(data.size()));
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima", "FindAllMaximaFromCPU: " +
        std::to_string(params_.antenna_count) + " beams, nFFT=" +
        std::to_string(params_.nFFT));

    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    if (!drv_gpu_lib::BatchManager::AllItemsFit(backend_, params_.antenna_count,
                                                  bytes_per_antenna, params_.memory_limit)) {
        throw std::runtime_error(
            "FindAllMaximaFromCPU: data doesn't fit in GPU memory (" +
            std::to_string(params_.antenna_count) + " beams * " +
            std::to_string(bytes_per_antenna / 1024 / 1024) + " MB/beam).");
    }

    ReallocateBuffersForBatch(params_.antenna_count);
    actual_batch_size_ = params_.antenna_count;

    CreateAllMaximaFFTPlan(params_.antenna_count);

    const int gpu_id = 0;
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    const bool do_prof = profiler.IsEnabled() && profiler.IsGPUEnabled(gpu_id);

    cl_event upload_event = UploadData(data);

    if (do_prof && upload_event) {
        clWaitForEvents(1, &upload_event);
        drv_gpu_lib::OpenCLProfilingData pdata{};
        if (drv_gpu_lib::FillOpenCLProfilingData(upload_event, pdata))
            profiler.Record(gpu_id, "AllMaxima", "Upload", pdata);
    }

    cl_event fft_event = ExecuteAllMaximaFFT(upload_event);
    clReleaseEvent(upload_event);

    if (do_prof && fft_event) {
        clWaitForEvents(1, &fft_event);
        drv_gpu_lib::OpenCLProfilingData pdata{};
        if (drv_gpu_lib::FillOpenCLProfilingData(fft_event, pdata))
            profiler.Record(gpu_id, "AllMaxima", "FFT+PostCallback", pdata);
    }

    AllMaximaResult result = FindAllMaxima(
        static_cast<void*>(fft_output_), params_.antenna_count, params_.nFFT,
        params_.sample_rate, dest, search_start, search_end);

    clReleaseEvent(fft_event);
    return result;
}

/**
 * @brief IAllMaximaPipeline-compatible: GPU сырой сигнал → G2G Copy → FFT → AllMaxima
 *
 * Аналог SpectrumMaximaFinder::FindAllMaximaFromGPUPipeline() — без batch mode.
 * Бросает runtime_error если данные + рабочие буферы не помещаются в VRAM.
 * FindAllMaxima делегирует в pipeline_->Execute() (Strategy pattern!)
 *
 * @param gpu_data        void* = cl_mem [antenna_count × n_point × complex<float>]
 * @param gpu_memory_bytes Размер gpu_data (0 = авто)
 */
AllMaximaResult SpectrumProcessorOpenCL::FindAllMaximaFromGPUPipeline(
    void* gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes,
    OutputDestination dest, uint32_t search_start, uint32_t search_end)
{
    cl_mem cl_data = static_cast<cl_mem>(gpu_data);
    if (!cl_data) {
        throw std::invalid_argument("FindAllMaximaFromGPUPipeline: gpu_data cannot be null");
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima", "FindAllMaximaFromGPUPipeline: " +
        std::to_string(antenna_count) + " beams, n_point=" + std::to_string(n_point));

    if (params_.nFFT == 0) {
        CalculateFFTSize();
    }

    size_t bytes_per_antenna = CalculateBytesPerAntenna();
    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : antenna_count * n_point * sizeof(std::complex<float>);

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, antenna_count, bytes_per_antenna,
        params_.memory_limit, external_memory);

    if (optimal_batch < antenna_count) {
        throw std::runtime_error(
            "FindAllMaximaFromGPUPipeline: data doesn't fit in GPU memory (" +
            std::to_string(antenna_count) + " beams, batch=" +
            std::to_string(optimal_batch) + ").");
    }

    ReallocateBuffersForBatch(antenna_count);
    actual_batch_size_ = antenna_count;

    CreateAllMaximaFFTPlan(antenna_count);

    const int gpu_id = 0;
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    const bool do_prof = profiler.IsEnabled() && profiler.IsGPUEnabled(gpu_id);

    size_t data_size = antenna_count * n_point * sizeof(std::complex<float>);
    cl_event copy_event = nullptr;
    cl_int err = clEnqueueCopyBuffer(queue_,
        cl_data, pre_callback_userdata_,
        0, PRE_CALLBACK_HEADER_SIZE,
        data_size,
        0, nullptr, &copy_event);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("FindAllMaximaFromGPUPipeline: CopyBuffer failed: " +
                                  std::to_string(err));
    }

    if (do_prof && copy_event) {
        clWaitForEvents(1, &copy_event);
        drv_gpu_lib::OpenCLProfilingData pdata{};
        if (drv_gpu_lib::FillOpenCLProfilingData(copy_event, pdata))
            profiler.Record(gpu_id, "AllMaxima", "GPU→GPU Copy", pdata);
    }

    cl_event fft_event = ExecuteAllMaximaFFT(copy_event);
    clReleaseEvent(copy_event);

    if (do_prof && fft_event) {
        clWaitForEvents(1, &fft_event);
        drv_gpu_lib::OpenCLProfilingData pdata{};
        if (drv_gpu_lib::FillOpenCLProfilingData(fft_event, pdata))
            profiler.Record(gpu_id, "AllMaxima", "FFT+PostCallback", pdata);
    }

    AllMaximaResult result = FindAllMaxima(
        static_cast<void*>(fft_output_), static_cast<uint32_t>(antenna_count), params_.nFFT,
        params_.sample_rate, dest, search_start, search_end);

    clReleaseEvent(fft_event);
    return result;
}

/**
 * @brief CPU FFT данные → Upload → AllMaxima (без FFT — данные уже FFT'ые)
 * Аналог SpectrumMaximaFinder::AllMaximaFromCPU() — делегирует в FindAllMaxima() → pipeline_->Execute().
 */
AllMaximaResult SpectrumProcessorOpenCL::AllMaximaFromCPU(
    const std::vector<std::complex<float>>& fft_data,
    uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest, uint32_t search_start, uint32_t search_end)
{
    size_t expected_size = static_cast<size_t>(beam_count) * nFFT;
    if (fft_data.size() != expected_size) {
        throw std::invalid_argument(
            "AllMaximaFromCPU: size mismatch. Expected " +
            std::to_string(expected_size) + ", got " + std::to_string(fft_data.size()));
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima", "AllMaximaFromCPU: " +
        std::to_string(beam_count) + " beams, nFFT=" + std::to_string(nFFT));

    const int gpu_id = 0;
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    const bool do_prof = profiler.IsEnabled() && profiler.IsGPUEnabled(gpu_id);

    cl_int err;
    size_t data_bytes = expected_size * sizeof(std::complex<float>);

    cl_mem gpu_fft = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        data_bytes, const_cast<std::complex<float>*>(fft_data.data()), &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("AllMaximaFromCPU: clCreateBuffer failed: " + std::to_string(err));

    AllMaximaResult result = FindAllMaxima(static_cast<void*>(gpu_fft), beam_count, nFFT,
                                            sample_rate, dest, search_start, search_end);

    clReleaseMemObject(gpu_fft);
    return result;
}

/**
 * @brief Скомпилировать "compute_magnitudes" kernel (lazy, guard флаг)
 *
 * Используется только в AllMaxima raw FFT API (FindAllMaxima с fft_data != fft_output_).
 * Когда данные приходят через FindAllMaximaFromCPU/GPU — magnitudes уже заполнены post-callback'ом,
 * этот kernel не нужен. При прямом вызове FindAllMaxima с внешними FFT данными — нужен.
 */
void SpectrumProcessorOpenCL::CompileComputeMagnitudesKernel() {
    if (compute_magnitudes_kernel_) return;

    cl_int err;
    const char* compute_mag_src = kernels::GetComputeMagnitudesKernelSource_opencl();
    size_t src_len = strlen(compute_mag_src);

    all_maxima_program_ = clCreateProgramWithSource(context_, 1, &compute_mag_src, &src_len, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("CompileComputeMagnitudesKernel: program create failed: " + std::to_string(err));

    err = clBuildProgram(all_maxima_program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(all_maxima_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(all_maxima_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
        con.PrintError(0, "AllMaxima", std::string("ComputeMagnitudes kernel build log:\n") + log.data());
        clReleaseProgram(all_maxima_program_);
        all_maxima_program_ = nullptr;
        throw std::runtime_error("CompileComputeMagnitudesKernel: build failed: " + std::to_string(err));
    }

    compute_magnitudes_kernel_ = clCreateKernel(all_maxima_program_, "compute_magnitudes", &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("CompileComputeMagnitudesKernel: kernel create failed: " + std::to_string(err));
}

/**
 * @brief Core AllMaxima: Magnitudes → pipeline_->Execute() (Strategy pattern!)
 *
 * Ключевое отличие от SpectrumMaximaFinder::FindAllMaxima:
 * здесь Detect+Scan+Compact делегируется в pipeline_->Execute() (AllMaximaPipelineOpenCL),
 * а не выполняется inline через clSetKernelArg + clEnqueueNDRangeKernel.
 *
 * Это чище архитектурно: pipeline_ — отдельный объект с lazy compilation kernels.
 * magnitudes_buffer_ передаётся в pipeline_ (который не владеет им).
 *
 * @param fft_data     void* = cl_mem float2[beam_count × nFFT] — FFT результаты
 * @param search_start Начальный бин (0=авто=1)
 * @param search_end   Конечный бин (0=авто=nFFT/2)
 * @return AllMaximaResult; при dest=GPU/ALL — caller ОБЯЗАН освободить gpu_maxima/gpu_counts!
 */
AllMaximaResult SpectrumProcessorOpenCL::FindAllMaxima(
    void* fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest, uint32_t search_start, uint32_t search_end)
{
    cl_mem fft_mem = static_cast<cl_mem>(fft_data);
    if (!fft_mem)
        throw std::invalid_argument("FindAllMaxima: fft_data cannot be null");

    if (search_start == 0) search_start = 1;
    if (search_end == 0) search_end = nFFT / 2;
    if (search_end >= nFFT) search_end = nFFT - 1;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima", "FindAllMaxima: beam_count=" + std::to_string(beam_count)
        + " nFFT=" + std::to_string(nFFT)
        + " range=[" + std::to_string(search_start) + "," + std::to_string(search_end) + ")"
        + " Fs=" + std::to_string(static_cast<int>(sample_rate)) + "Hz");

    const size_t total_elements = static_cast<size_t>(beam_count) * nFFT;
    EnsureMagnitudesBuffer(total_elements);

    bool need_compute_magnitudes = (fft_mem != fft_output_ || !allmax_plan_created_);

    if (need_compute_magnitudes) {
        CompileComputeMagnitudesKernel();

        cl_int err;
        uint32_t total_size = static_cast<uint32_t>(total_elements);

        err = clSetKernelArg(compute_magnitudes_kernel_, 0, sizeof(cl_mem), &fft_mem);
        err |= clSetKernelArg(compute_magnitudes_kernel_, 1, sizeof(cl_mem), &magnitudes_buffer_);
        err |= clSetKernelArg(compute_magnitudes_kernel_, 2, sizeof(uint32_t), &total_size);
        if (err != CL_SUCCESS)
            throw std::runtime_error("FindAllMaxima: compute_magnitudes setKernelArg failed: " + std::to_string(err));

        size_t global_size = ((total_elements + 255) / 256) * 256;
        size_t local_size = 256;

        cl_event mag_event = nullptr;
        err = clEnqueueNDRangeKernel(queue_, compute_magnitudes_kernel_, 1, nullptr,
            &global_size, &local_size, 0, nullptr, &mag_event);
        if (err != CL_SUCCESS)
            throw std::runtime_error("FindAllMaxima: compute_magnitudes NDRange failed: " + std::to_string(err));

        const int gpu_id = 0;
        auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
        const bool do_prof = profiler.IsEnabled() && profiler.IsGPUEnabled(gpu_id);
        if (do_prof) {
            drv_gpu_lib::OpenCLProfilingData data{};
            if (drv_gpu_lib::FillOpenCLProfilingData(mag_event, data))
                profiler.Record(gpu_id, "AllMaxima", "ComputeMagnitudes", data);
        }
        clReleaseEvent(mag_event);
    }

    return pipeline_->Execute(magnitudes_buffer_, fft_mem, beam_count, nFFT, sample_rate,
                             dest, search_start, search_end);
}

/**
 * @brief Освободить AllMaxima ресурсы: compute_magnitudes kernel + program + pipeline_
 *
 * Отличие от SpectrumMaximaFinder::ReleaseAllMaximaResources:
 * pipeline_.reset() освобождает AllMaximaPipelineOpenCL (detect/scan/compact kernels) —
 * нам не нужно их освобождать по одному.
 */
void SpectrumProcessorOpenCL::ReleaseAllMaximaResources() {
    if (compute_magnitudes_kernel_) { clReleaseKernel(compute_magnitudes_kernel_); compute_magnitudes_kernel_ = nullptr; }
    if (all_maxima_program_) { clReleaseProgram(all_maxima_program_); all_maxima_program_ = nullptr; }
    pipeline_.reset();  // pipeline owns detect/scan/compact kernels
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 8: ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ — WritePreCallbackHeader(), ReleaseResources()
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Обновить PreCallbackHeader в pre_callback_userdata_ (32 байта, blocking write)
 *
 * Ключевое отличие от SpectrumMaximaFinder::WritePreCallbackHeader:
 * здесь вычисляется nFFT_log2 (TASK-3) — pre-callback использует bitwise операции
 * вместо div/mod для вычисления beam_idx и sample_idx:
 *   beam_idx   = element_idx >> nFFT_log2      (== element_idx / nFFT)
 *   sample_idx = element_idx & (nFFT-1)        (== element_idx % nFFT)
 * Это значительно быстрее на GPU где целочисленное деление дорогое.
 *
 * @param batch_count Реальное число лучей в текущем batch
 */
void SpectrumProcessorOpenCL::WritePreCallbackHeader(size_t batch_count) {
    // TASK-3: вычисляем log2(nFFT) для bitwise div/mod в pre-callback
    // nFFT гарантированно pow2 (CalculateFFTSize)
    uint32_t nFFT_log2 = 0;
    uint32_t tmp = params_.nFFT;
    while (tmp > 1) { tmp >>= 1; ++nFFT_log2; }

    // Записать заголовок pre-callback с параметрами для GPU
    PreCallbackHeader header = {
        static_cast<uint32_t>(batch_count),
        params_.n_point,
        params_.nFFT,
        nFFT_log2,    // TASK-3: log2(nFFT)
        0, 0, 0, 0
    };

    cl_int err = clEnqueueWriteBuffer(queue_, pre_callback_userdata_, CL_TRUE,
                                       0, sizeof(header), &header, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("WritePreCallbackHeader failed: " + std::to_string(err));
    }
}

/**
 * @brief Освободить все GPU ресурсы (вызывается из деструктора)
 * Аналог SpectrumMaximaFinder::ReleaseResources() — тот же порядок освобождения.
 * pipeline_ уже освобождён в ReleaseAllMaximaResources() через pipeline_.reset().
 */
void SpectrumProcessorOpenCL::ReleaseResources() {
    // AllMaxima ресурсы
    ReleaseAllMaximaResources();

    // AllMaxima FFT план
    if (allmax_plan_created_ && allmax_plan_handle_) {
        clfftDestroyPlan(&allmax_plan_handle_);
        allmax_plan_handle_ = 0;
        allmax_plan_created_ = false;
    }
    if (magnitudes_buffer_) {
        clReleaseMemObject(magnitudes_buffer_);
        magnitudes_buffer_ = nullptr;
        magnitudes_buffer_size_ = 0;
    }

    // Post-kernel
    if (post_kernel_) {
        clReleaseKernel(post_kernel_);
        post_kernel_ = nullptr;
    }
    if (post_program_) {
        clReleaseProgram(post_program_);
        post_program_ = nullptr;
    }

    // FFT план
    if (plan_created_ && plan_handle_) {
        clfftDestroyPlan(&plan_handle_);
        plan_handle_ = 0;
        plan_created_ = false;
    }

    // Буферы
    if (pre_callback_userdata_) {
        clReleaseMemObject(pre_callback_userdata_);
        pre_callback_userdata_ = nullptr;
    }
    if (fft_input_) {
        clReleaseMemObject(fft_input_);
        fft_input_ = nullptr;
    }
    if (fft_output_) {
        clReleaseMemObject(fft_output_);
        fft_output_ = nullptr;
    }
    if (maxima_output_) {
        clReleaseMemObject(maxima_output_);
        maxima_output_ = nullptr;
    }
    if (fft_temp_buffer_) {
        clReleaseMemObject(fft_temp_buffer_);
        fft_temp_buffer_ = nullptr;
    }
    if (pinned_staging_buffer_) {
        clReleaseMemObject(pinned_staging_buffer_);
        pinned_staging_buffer_ = nullptr;
    }

    initialized_ = false;
}

} // namespace antenna_fft

