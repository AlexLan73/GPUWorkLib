/**
 * @file spectrum_maxima_finder.cpp
 * @brief Реализация SpectrumMaximaFinder — поиск максимумов спектра FFT на GPU
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ОГЛАВЛЕНИЕ (Line numbers may change after edits)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * ЧАСТЬ 1: КОНСТРУКТОРЫ И ДЕСТРУКТОРЫ
 *   - SpectrumMaximaFinder()       — конструктор
 *   - ~SpectrumMaximaFinder()      — деструктор
 *   - Move constructor/operator    — перемещение
 *
 * ЧАСТЬ 2: ПУБЛИЧНЫЙ API
 *   - Initialize()                 — инициализация GPU ресурсов
 *   - Process()                    — обработка (single batch или multi-batch)
 *   - PrintInfo()                  — вывод информации о конфигурации
 *
 * ЧАСТЬ 3: BATCH PROCESSING
 *   - ProcessBatch()               — обработка одного batch антенн
 *   - ReallocateBuffersForBatch()  — выделение/переиспользование буферов и FFT плана
 *
 * ЧАСТЬ 4: УТИЛИТЫ И НАСТРОЙКИ
 *   - CalculateFFTSize()           — вычисление nFFT (степень двойки)
 *   - NextPowerOf2()               — вспомогательная функция
 *   - CalculateBytesPerAntenna()   — память на одну антенну
 *
 * ЧАСТЬ 5: GPU БУФЕРЫ (single-batch режим)
 *   - AllocateBuffers()            — создание буферов
 *   - CreateFFTPlanWithCallback()  — создание FFT плана с pre-callback
 *   - CompilePostKernel()          — компиляция post-kernel
 *
 * ЧАСТЬ 6: GPU ОПЕРАЦИИ
 *   - UploadData()                 — загрузка данных на GPU (Pinned Memory)
 *   - ExecuteFFT()                 — выполнение FFT
 *   - ExecutePostKernel()          — поиск максимумов + интерполяция
 *   - ReadResults()                — чтение результатов
 *
 * ЧАСТЬ 7: ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
 *   - WritePreCallbackHeader()     — запись заголовка в pre-callback userdata
 *   - ReleaseResources()           — освобождение GPU ресурсов
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-06
 */

#include "spectrum_maxima_finder.h"
#include "backends/opencl/opencl_profiling.hpp"
#include "services/gpu_profiler.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 1: КОНСТРУКТОРЫ И ДЕСТРУКТОРЫ
// ════════════════════════════════════════════════════════════════════════════

SpectrumMaximaFinder::SpectrumMaximaFinder(
    const SpectrumParams& params,
    drv_gpu_lib::IBackend* backend)
    : params_(params)
    , backend_(backend) {

    if (!backend_) {
        throw std::invalid_argument("SpectrumMaximaFinder: backend cannot be null");
    }

    if (!backend_->IsInitialized()) {
        throw std::runtime_error("SpectrumMaximaFinder: backend is not initialized");
    }

    // Получаем OpenCL ресурсы из backend
    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    queue_ = static_cast<cl_command_queue>(backend_->GetNativeQueue());
    device_ = static_cast<cl_device_id>(backend_->GetNativeDevice());

    if (!context_ || !queue_ || !device_) {
        throw std::runtime_error("SpectrumMaximaFinder: failed to get OpenCL resources from backend");
    }
}

SpectrumMaximaFinder::~SpectrumMaximaFinder() {
    ReleaseResources();
}

SpectrumMaximaFinder::SpectrumMaximaFinder(SpectrumMaximaFinder&& other) noexcept
    : params_(other.params_)
    , initialized_(other.initialized_)
    , backend_(other.backend_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , plan_handle_(other.plan_handle_)
    , plan_created_(other.plan_created_)
    , pre_callback_userdata_(other.pre_callback_userdata_)
    , fft_input_(other.fft_input_)
    , fft_output_(other.fft_output_)
    , maxima_output_(other.maxima_output_)
    , fft_temp_buffer_(other.fft_temp_buffer_)
    , pinned_staging_buffer_(other.pinned_staging_buffer_)
    , post_program_(other.post_program_)
    , post_kernel_(other.post_kernel_)
    , profiling_(other.profiling_)
    , current_batch_size_(other.current_batch_size_)
    , actual_batch_size_(other.actual_batch_size_)
    , plan_batch_size_(other.plan_batch_size_)
    , fft_temp_buffer_size_(other.fft_temp_buffer_size_)
    , pinned_buffer_size_(other.pinned_buffer_size_)
    , clfft_initialized_(other.clfft_initialized_) {

    // Invalidate source
    other.initialized_ = false;
    other.plan_handle_ = 0;
    other.plan_created_ = false;
    other.pre_callback_userdata_ = nullptr;
    other.fft_input_ = nullptr;
    other.fft_output_ = nullptr;
    other.maxima_output_ = nullptr;
    other.fft_temp_buffer_ = nullptr;
    other.pinned_staging_buffer_ = nullptr;
    other.post_program_ = nullptr;
    other.post_kernel_ = nullptr;
    other.current_batch_size_ = 0;
    other.actual_batch_size_ = 0;
    other.plan_batch_size_ = 0;
    other.fft_temp_buffer_size_ = 0;
    other.pinned_buffer_size_ = 0;
    other.clfft_initialized_ = false;
}

SpectrumMaximaFinder& SpectrumMaximaFinder::operator=(SpectrumMaximaFinder&& other) noexcept {
    if (this != &other) {
        ReleaseResources();

        params_ = other.params_;
        initialized_ = other.initialized_;
        backend_ = other.backend_;
        context_ = other.context_;
        queue_ = other.queue_;
        device_ = other.device_;
        plan_handle_ = other.plan_handle_;
        plan_created_ = other.plan_created_;
        pre_callback_userdata_ = other.pre_callback_userdata_;
        fft_input_ = other.fft_input_;
        fft_output_ = other.fft_output_;
        maxima_output_ = other.maxima_output_;
        fft_temp_buffer_ = other.fft_temp_buffer_;
        pinned_staging_buffer_ = other.pinned_staging_buffer_;
        post_program_ = other.post_program_;
        post_kernel_ = other.post_kernel_;
        profiling_ = other.profiling_;
        current_batch_size_ = other.current_batch_size_;
        actual_batch_size_ = other.actual_batch_size_;
        plan_batch_size_ = other.plan_batch_size_;
        fft_temp_buffer_size_ = other.fft_temp_buffer_size_;
        pinned_buffer_size_ = other.pinned_buffer_size_;
        clfft_initialized_ = other.clfft_initialized_;

        other.initialized_ = false;
        other.plan_handle_ = 0;
        other.plan_created_ = false;
        other.pre_callback_userdata_ = nullptr;
        other.fft_input_ = nullptr;
        other.fft_output_ = nullptr;
        other.maxima_output_ = nullptr;
        other.fft_temp_buffer_ = nullptr;
        other.pinned_staging_buffer_ = nullptr;
        other.post_program_ = nullptr;
        other.post_kernel_ = nullptr;
        other.current_batch_size_ = 0;
        other.actual_batch_size_ = 0;
        other.plan_batch_size_ = 0;
        other.fft_temp_buffer_size_ = 0;
        other.pinned_buffer_size_ = 0;
        other.clfft_initialized_ = false;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 2: ПУБЛИЧНЫЙ API — Initialize(), Process(), PrintInfo()
// ════════════════════════════════════════════════════════════════════════════

void SpectrumMaximaFinder::Initialize() {
    if (initialized_) {
        return;
    }

    std::cout << "\n[SpectrumMaximaFinder] Инициализация...\n";

    // 1. Вычислить размеры FFT
    CalculateFFTSize();

    std::cout << "  antenna_count: " << params_.antenna_count << "\n";
    std::cout << "  n_point: " << params_.n_point << "\n";
    std::cout << "  repeat_count: " << params_.repeat_count << "\n";
    std::cout << "  base_fft: " << params_.base_fft << "\n";
    std::cout << "  nFFT: " << params_.nFFT << "\n";
    std::cout << "  search_range: " << params_.search_range << "\n";
    std::cout << "  sample_rate: " << params_.sample_rate << " Hz\n";

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

        std::cout << "  [Batch mode] Резервируем буферы под max batch = " << max_batch_size << "\n";

        // Создаём буферы под максимальный batch size
        ReallocateBuffersForBatch(max_batch_size);
        std::cout << "  Буферы созданы (max batch = " << max_batch_size << ")\n";

        // Компилируем post-kernel сразу (он не зависит от размера batch)
        CompilePostKernel();
        std::cout << "  Post-kernel скомпилирован\n";
    } else {
        // Для маленьких данных — выделяем буферы сразу (старое поведение)
        AllocateBuffers();
        std::cout << "  Буферы созданы\n";

        CreateFFTPlanWithCallback();
        std::cout << "  FFT план создан с pre-callback\n";

        CompilePostKernel();
        std::cout << "  Post-kernel скомпилирован\n";

        // Устанавливаем размеры batch для не-batch режима
        current_batch_size_ = params_.antenna_count;
        actual_batch_size_ = params_.antenna_count;
    }

    initialized_ = true;
    std::cout << "[SpectrumMaximaFinder] Инициализация завершена!\n\n";
}

std::vector<SpectrumResult> SpectrumMaximaFinder::Process(
    const std::vector<std::complex<float>>& input_data) {

    if (!initialized_) {
        throw std::runtime_error("SpectrumMaximaFinder::Process: not initialized");
    }

    size_t expected_size = params_.antenna_count * params_.n_point;
    if (input_data.size() != expected_size) {
        throw std::invalid_argument(
            "SpectrumMaximaFinder::Process: input size mismatch. "
            "Expected " + std::to_string(expected_size) +
            ", got " + std::to_string(input_data.size()));
    }

    // Сбросить профилирование
    profiling_ = ProfilingData{};

    // ═══════════════════════════════════════════════════════════════════════
    // BATCH PROCESSING (использует BatchManager)
    // ═══════════════════════════════════════════════════════════════════════

    // Рассчитать память на одну антенну
    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    // Проверить, помещаются ли все данные в память
    if (drv_gpu_lib::BatchManager::AllItemsFit(backend_, params_.antenna_count,
                                                bytes_per_antenna, params_.memory_limit)) {
        // Все помещаются — обработать за один раз (без batch)
        std::cout << "[SpectrumMaximaFinder] Все " << params_.antenna_count
                  << " антенн помещаются в память — batch не нужен\n";
        return ProcessBatch(input_data, 0, params_.antenna_count);
    }

    // Рассчитать оптимальный размер batch
    size_t batch_size = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

    // Создать список батчей с умным слиянием хвоста [1..3]
    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params_.antenna_count, batch_size, 3, true);

    // Вывести информацию о батчах
    std::cout << "\n[SpectrumMaximaFinder] Batch Processing:\n";
    std::cout << "  📊 Память на антенну: " << (bytes_per_antenna / 1024 / 1024) << " MB\n";
    std::cout << "  📊 memory_limit: " << (params_.memory_limit * 100) << "%\n";
    drv_gpu_lib::BatchManager::PrintBatchInfo(batches, params_.antenna_count);

    // Обработать каждый batch
    std::vector<SpectrumResult> all_results;
    all_results.reserve(params_.antenna_count);  // ONE_PEAK: 1 результат на антенну

    for (const auto& batch : batches) {
        std::cout << "  🔄 Processing batch " << batch.batch_idx
                  << ": antennas [" << batch.start << ".."
                  << (batch.start + batch.count - 1) << "]\n";

        auto batch_results = ProcessBatch(input_data, batch.start, batch.count);
        all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
    }

    std::cout << "  ✅ Batch processing completed: " << all_results.size() << " results\n\n";

    return all_results;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 3: BATCH PROCESSING — ProcessBatch(), ReallocateBuffersForBatch()
// ════════════════════════════════════════════════════════════════════════════

std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessBatch(
    const std::vector<std::complex<float>>& input_data,
    size_t start_antenna,
    size_t batch_antenna_count) {

    // Перевыделить буферы под текущий batch (если нужно)
    // current_batch_size_ — размер буферов, actual_batch_size_ — реальный batch
    if (batch_antenna_count > current_batch_size_ || !plan_created_) {
        ReallocateBuffersForBatch(batch_antenna_count);
    }
    actual_batch_size_ = batch_antenna_count;  // Запоминаем реальный размер batch

    const int gpu_id = 0;
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    const bool do_prof = profiler.IsEnabled() && profiler.IsGPUEnabled(gpu_id);

    // Извлечь срез данных для текущего batch
    size_t offset = start_antenna * params_.n_point;
    size_t count = batch_antenna_count * params_.n_point;
    std::vector<std::complex<float>> batch_data(
        input_data.begin() + offset,
        input_data.begin() + offset + count);

    // 1. Загрузить данные batch на GPU
    cl_event upload_event = UploadData(batch_data);
    profiling_.upload_time_ms += ProfileEvent(upload_event, "Upload");
    if (do_prof) {
        drv_gpu_lib::OpenCLProfilingData data{};
        if (drv_gpu_lib::FillOpenCLProfilingData(upload_event, data)) {
            profiler.Record(gpu_id, "SpectrumMaxima", "Upload", data);
        }
    }

    // 2. Выполнить FFT
    cl_event fft_event = ExecuteFFT(upload_event);
    clReleaseEvent(upload_event);
    profiling_.fft_time_ms += ProfileEvent(fft_event, "FFT");
    if (do_prof) {
        drv_gpu_lib::OpenCLProfilingData data{};
        if (drv_gpu_lib::FillOpenCLProfilingData(fft_event, data)) {
            profiler.Record(gpu_id, "SpectrumMaxima", "FFT", data);
        }
    }

    // 3. Выполнить post-kernel
    cl_event post_event = ExecutePostKernel(fft_event);
    clReleaseEvent(fft_event);
    profiling_.post_kernel_time_ms += ProfileEvent(post_event, "PostKernel");
    if (do_prof) {
        drv_gpu_lib::OpenCLProfilingData data{};
        if (drv_gpu_lib::FillOpenCLProfilingData(post_event, data)) {
            profiler.Record(gpu_id, "SpectrumMaxima", "PostKernel", data);
        }
    }

    // 4. Прочитать результаты batch
    std::vector<SpectrumResult> batch_results = ReadResults(post_event, do_prof);
    clReleaseEvent(post_event);

    // Скорректировать antenna_id для результатов (добавить start_antenna)
    for (auto& result : batch_results) {
        result.antenna_id += start_antenna;
    }

    return batch_results;
}

void SpectrumMaximaFinder::ReallocateBuffersForBatch(size_t batch_antenna_count) {
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
            std::cerr << "[SpectrumMaximaFinder] WARNING: Failed to create pinned buffer: "
                      << err << ", using regular upload\n";
            pinned_staging_buffer_ = nullptr;
            pinned_buffer_size_ = 0;
        } else {
            pinned_buffer_size_ = required_pinned_size;
        }
    }

    // current_batch_size_ уже обновлён внутри if (need_new_buffers)
    // Не обновляем здесь, чтобы сохранить реальный размер выделенных буферов
}

void SpectrumMaximaFinder::PrintInfo() const {
    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  SpectrumMaximaFinder Configuration\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << std::left;
    std::cout << std::setw(25) << "  Antenna count:" << params_.antenna_count << "\n";
    std::cout << std::setw(25) << "  N points:" << params_.n_point << "\n";
    std::cout << std::setw(25) << "  Repeat count:" << params_.repeat_count << "\n";
    std::cout << std::setw(25) << "  Base FFT:" << params_.base_fft << "\n";
    std::cout << std::setw(25) << "  nFFT:" << params_.nFFT << "\n";
    std::cout << std::setw(25) << "  Search range:" << params_.search_range << "\n";
    std::cout << std::setw(25) << "  Sample rate:" << params_.sample_rate << " Hz\n";
    std::cout << std::setw(25) << "  Initialized:" << (initialized_ ? "Yes" : "No") << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n\n";
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 4: УТИЛИТЫ — CalculateFFTSize(), NextPowerOf2(), CalculateBytesPerAntenna()
// ════════════════════════════════════════════════════════════════════════════

void SpectrumMaximaFinder::CalculateFFTSize() {
    // base_fft = следующая степень двойки от n_point
    params_.base_fft = NextPowerOf2(params_.n_point);

    // nFFT = base_fft * repeat_count
    params_.nFFT = params_.base_fft * params_.repeat_count;

    // search_range по умолчанию = nFFT / 4
    if (params_.search_range == 0) {
        params_.search_range = params_.nFFT / 4;
    }
}

uint32_t SpectrumMaximaFinder::NextPowerOf2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

size_t SpectrumMaximaFinder::CalculateBytesPerAntenna() const {
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

void SpectrumMaximaFinder::AllocateBuffers() {
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

void SpectrumMaximaFinder::CreateFFTPlanWithCallback() {
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

void SpectrumMaximaFinder::CompilePostKernel() {
    cl_int err;

    // Выбор кернела в зависимости от режима поиска пиков
    const char* source;
    const char* kernel_name;

    if (params_.peak_mode == PeakSearchMode::ONE_PEAK) {
        source = kernels::GetPostKernelSource_OnePeak_opencl();
        kernel_name = "post_kernel_one_peak";
        std::cout << "  📊 Peak mode: ONE_PEAK (4 MaxValue per beam)\n";
    } else {
        source = kernels::GetPostKernelSource_TwoPeaks_opencl();
        kernel_name = "post_kernel";
        std::cout << "  📊 Peak mode: TWO_PEAKS (8 MaxValue per beam)\n";
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
        std::cerr << "Build log:\n" << log.data() << "\n";
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

cl_event SpectrumMaximaFinder::UploadData(const std::vector<std::complex<float>>& input_data) {
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

cl_event SpectrumMaximaFinder::ExecuteFFT(cl_event wait_event) {
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

cl_event SpectrumMaximaFinder::ExecutePostKernel(cl_event wait_event) {
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

std::vector<SpectrumResult> SpectrumMaximaFinder::ReadResults(cl_event wait_event, bool send_to_profiler) {
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

    // Ждём завершения
    clWaitForEvents(1, &read_event);
    profiling_.download_time_ms = ProfileEvent(read_event, "Download");
    if (send_to_profiler) {
        const int gpu_id = 0;
        drv_gpu_lib::OpenCLProfilingData data{};
        if (drv_gpu_lib::FillOpenCLProfilingData(read_event, data)) {
            drv_gpu_lib::GPUProfiler::GetInstance().Record(gpu_id, "SpectrumMaxima", "Download", data);
        }
    }
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

double SpectrumMaximaFinder::ProfileEvent(cl_event event, const char* name) {
    if (!event) return 0.0;

    // Ждём завершения события
    clWaitForEvents(1, &event);

    cl_ulong start = 0, end = 0;
    cl_int err1 = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                           sizeof(cl_ulong), &start, nullptr);
    cl_int err2 = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                           sizeof(cl_ulong), &end, nullptr);

    if (err1 != CL_SUCCESS || err2 != CL_SUCCESS) {
        std::cerr << "[ProfileEvent] " << name << " failed: " << err1 << "," << err2 << "\n";
        return 0.0;
    }

    double time_ms = (end - start) / 1e6;
    return time_ms;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 7: ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ — WritePreCallbackHeader(), ReleaseResources()
// ════════════════════════════════════════════════════════════════════════════

void SpectrumMaximaFinder::WritePreCallbackHeader(size_t batch_count) {
    // Записать заголовок pre-callback с параметрами для GPU
    PreCallbackHeader header = {
        static_cast<uint32_t>(batch_count),
        params_.n_point,
        params_.nFFT,
        0, 0, 0, 0, 0
    };

    cl_int err = clEnqueueWriteBuffer(queue_, pre_callback_userdata_, CL_TRUE,
                                       0, sizeof(header), &header, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("WritePreCallbackHeader failed: " + std::to_string(err));
    }
}

void SpectrumMaximaFinder::ReleaseResources() {
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
