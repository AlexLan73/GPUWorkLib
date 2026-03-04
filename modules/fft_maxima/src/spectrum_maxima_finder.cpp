/**
 * @file spectrum_maxima_finder.cpp
 * @brief Реализация SpectrumMaximaFinder — поиск максимумов спектра FFT на GPU
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

#include "spectrum_maxima_finder.h"
#include "services/console_output.hpp"
#include "services/gpu_profiler.hpp"
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 1: КОНСТРУКТОРЫ И ДЕСТРУКТОРЫ
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Конструктор фасада — принимает готовый backend, не владеет им
 *
 * Извлекает OpenCL ресурсы (context, queue, device) из backend через GetNative*().
 * Все GPU операции выполняются в этих context/queue — не создаём свои.
 * Backend должен оставаться живым дольше этого объекта.
 *
 * Инициализация (AllocateBuffers, CreateFFTPlanWithCallback) происходит ЛЕНИВО в Initialize()
 * или при первом вызове Process<T>. Конструктор только проверяет валидность backend'а.
 *
 * @param backend Указатель на инициализированный IBackend (не nullptr, не владеем)
 * @throws std::invalid_argument если backend == nullptr
 * @throws std::runtime_error   если backend не инициализирован или нет OpenCL ресурсов
 */
SpectrumMaximaFinder::SpectrumMaximaFinder(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

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

/**
 * @brief Деструктор — освобождает все GPU ресурсы через ReleaseResources()
 * ReleaseResources освобождает: clFFT планы, OpenCL kernels/programs, все cl_mem буферы.
 * backend_ НЕ освобождается — мы не владеем им.
 */
SpectrumMaximaFinder::~SpectrumMaximaFinder() {
    ReleaseResources();
}

/**
 * @brief Move constructor — передаёт владение всеми ресурсами, обнуляет источник
 * После перемещения source.initialized_=false, все его cl_mem/планы=nullptr.
 * Нужен для хранения в std::vector и возврата из фабрики без копирования.
 */
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
    , current_batch_size_(other.current_batch_size_)
    , actual_batch_size_(other.actual_batch_size_)
    , plan_batch_size_(other.plan_batch_size_)
    , fft_temp_buffer_size_(other.fft_temp_buffer_size_)
    , pinned_buffer_size_(other.pinned_buffer_size_)
    , clfft_initialized_(other.clfft_initialized_)
    , all_maxima_program_(other.all_maxima_program_)
    , detect_kernel_(other.detect_kernel_)
    , compute_magnitudes_kernel_(other.compute_magnitudes_kernel_)
    , prefix_sum_program_(other.prefix_sum_program_)
    , block_scan_kernel_(other.block_scan_kernel_)
    , block_add_kernel_(other.block_add_kernel_)
    , compact_program_(other.compact_program_)
    , compact_kernel_(other.compact_kernel_)
    , all_maxima_kernels_compiled_(other.all_maxima_kernels_compiled_)
    , allmax_plan_handle_(other.allmax_plan_handle_)
    , allmax_plan_created_(other.allmax_plan_created_)
    , allmax_plan_batch_size_(other.allmax_plan_batch_size_)
    , magnitudes_buffer_(other.magnitudes_buffer_)
    , magnitudes_buffer_size_(other.magnitudes_buffer_size_) {

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
    other.all_maxima_program_ = nullptr;
    other.detect_kernel_ = nullptr;
    other.compute_magnitudes_kernel_ = nullptr;
    other.prefix_sum_program_ = nullptr;
    other.block_scan_kernel_ = nullptr;
    other.block_add_kernel_ = nullptr;
    other.compact_program_ = nullptr;
    other.compact_kernel_ = nullptr;
    other.all_maxima_kernels_compiled_ = false;
    other.allmax_plan_handle_ = 0;
    other.allmax_plan_created_ = false;
    other.allmax_plan_batch_size_ = 0;
    other.magnitudes_buffer_ = nullptr;
    other.magnitudes_buffer_size_ = 0;
}

/**
 * @brief Move assignment — освобождает текущие ресурсы, принимает из other
 * ReleaseResources() вызывается ПЕРЕД присвоением — иначе потеряем текущие буферы.
 */
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
        current_batch_size_ = other.current_batch_size_;
        actual_batch_size_ = other.actual_batch_size_;
        plan_batch_size_ = other.plan_batch_size_;
        fft_temp_buffer_size_ = other.fft_temp_buffer_size_;
        pinned_buffer_size_ = other.pinned_buffer_size_;
        clfft_initialized_ = other.clfft_initialized_;
        all_maxima_program_ = other.all_maxima_program_;
        detect_kernel_ = other.detect_kernel_;
        compute_magnitudes_kernel_ = other.compute_magnitudes_kernel_;
        prefix_sum_program_ = other.prefix_sum_program_;
        block_scan_kernel_ = other.block_scan_kernel_;
        block_add_kernel_ = other.block_add_kernel_;
        compact_program_ = other.compact_program_;
        compact_kernel_ = other.compact_kernel_;
        all_maxima_kernels_compiled_ = other.all_maxima_kernels_compiled_;
        allmax_plan_handle_ = other.allmax_plan_handle_;
        allmax_plan_created_ = other.allmax_plan_created_;
        allmax_plan_batch_size_ = other.allmax_plan_batch_size_;
        magnitudes_buffer_ = other.magnitudes_buffer_;
        magnitudes_buffer_size_ = other.magnitudes_buffer_size_;

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
        other.all_maxima_program_ = nullptr;
        other.detect_kernel_ = nullptr;
        other.compute_magnitudes_kernel_ = nullptr;
        other.prefix_sum_program_ = nullptr;
        other.block_scan_kernel_ = nullptr;
        other.block_add_kernel_ = nullptr;
        other.compact_program_ = nullptr;
        other.compact_kernel_ = nullptr;
        other.all_maxima_kernels_compiled_ = false;
        other.allmax_plan_handle_ = 0;
        other.allmax_plan_created_ = false;
        other.allmax_plan_batch_size_ = 0;
        other.magnitudes_buffer_ = nullptr;
        other.magnitudes_buffer_size_ = 0;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 2: ПУБЛИЧНЫЙ API — Initialize(), Process(), PrintInfo()
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализировать GPU ресурсы: буферы, clFFT план, post-kernel
 *
 * Вызывается один раз перед первым Process<T>. Повторные вызовы — нет-оп (guard флаг).
 *
 * Стратегия выделения:
 * - Если все антенны помещаются в VRAM (AllItemsFit) → обычный путь:
 *   AllocateBuffers() + CreateFFTPlanWithCallback() + CompilePostKernel()
 * - Иначе batch mode: ReallocateBuffersForBatch(max_batch_size) + CompilePostKernel()
 *   (буферы на max_batch, FFT план создаётся при первом batch'е)
 *
 * Почему batch mode не создаёт FFT план здесь: план зависит от batch_count, который
 * определяется в ReallocateBuffersForBatch → нет смысла создавать его дважды.
 *
 * @throws std::runtime_error если любой шаг инициализации провалился
 */
void SpectrumMaximaFinder::Initialize() {
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
 * @brief Обработать один batch антенн: Upload → FFT → PostKernel → ReadResults
 *
 * Внутренний метод, вызывается из ProcessFromCPU() в цикле по batch'ам.
 * Извлекает срез [start_antenna, start_antenna+batch_antenna_count) из input_data,
 * выполняет полный GPU pipeline и корректирует antenna_id на start_antenna.
 *
 * Переиспользование буферов: если буферы достаточны (current_batch_size_ >= batch_antenna_count)
 * и план совпадает — только обновляем заголовок (WritePreCallbackHeader). Иначе ReallocateBuffersForBatch.
 *
 * @param input_data          Весь массив всех антенн (не только batch)
 * @param start_antenna       Абсолютный индекс первой антенны batch
 * @param batch_antenna_count Число антенн в batch
 * @param prof_events         Список cl_event для профилирования (nullptr = не профилируем)
 * @return vector<SpectrumResult>[batch_antenna_count] с корректными antenna_id
 */
std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessBatch(
    const std::vector<std::complex<float>>& input_data,
    size_t start_antenna,
    size_t batch_antenna_count,
    ProfEvents* prof_events) {

    // Перевыделить буферы под текущий batch (если нужно)
    // current_batch_size_ — размер буферов, actual_batch_size_ — реальный batch
    if (batch_antenna_count > current_batch_size_ || !plan_created_) {
        ReallocateBuffersForBatch(batch_antenna_count);
    }
    actual_batch_size_ = batch_antenna_count;  // Запоминаем реальный размер batch

    // Извлечь срез данных для текущего batch
    size_t offset = start_antenna * params_.n_point;
    size_t count = batch_antenna_count * params_.n_point;
    std::vector<std::complex<float>> batch_data(
        input_data.begin() + offset,
        input_data.begin() + offset + count);

    // 1. Загрузить данные batch на GPU
    cl_event upload_event = UploadData(batch_data);

    // 2. Выполнить FFT (upload_event — wait, потом освободить/собрать)
    cl_event fft_event = ExecuteFFT(upload_event);
    CollectOrRelease(upload_event, "Upload", prof_events);

    // 3. Выполнить post-kernel (fft_event — wait, потом освободить/собрать)
    cl_event post_event = ExecutePostKernel(fft_event);
    CollectOrRelease(fft_event, "FFT", prof_events);

    // 4. Прочитать результаты batch (post_event — wait; Download записывается внутри)
    std::vector<SpectrumResult> batch_results = ReadResults(post_event, prof_events);
    CollectOrRelease(post_event, "PostKernel", prof_events);

    // Скорректировать antenna_id для результатов (добавить start_antenna)
    for (auto& result : batch_results) {
        result.antenna_id += start_antenna;
    }

    return batch_results;
}

/**
 * @brief Выделить или переиспользовать буферы и FFT план для batch_antenna_count антенн
 *
 * Центральный метод управления GPU памятью. Реализует три пути:
 *
 * FAST PATH: буферы >= batch_antenna_count && план совпадает по размеру →
 *   только WritePreCallbackHeader (обновить beam_count в заголовке). O(1).
 *
 * PARTIAL PATH: нужен новый план (batch size изменился), но буферы достаточны →
 *   только пересоздать FFT план + bake. Экономим ~50-200ms на переаллокации буферов.
 *
 * FULL PATH: буферы недостаточны → освободить и создать всё заново:
 *   pre_callback_userdata_, fft_input_, fft_output_, maxima_output_, FFT план.
 *   fft_temp_buffer_ и pinned_staging_buffer_ переиспользуются если достаточно большие.
 *
 * Инварианты после успешного завершения:
 *   - current_batch_size_ >= batch_antenna_count (реальный размер буферов)
 *   - plan_batch_size_ == batch_antenna_count (план точно совпадает)
 *   - plan_created_ == true
 *
 * @param batch_antenna_count Число антенн в текущем batch
 * @throws std::runtime_error если любое выделение или bake провалилось
 */
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

/**
 * @brief Вывести конфигурацию (params_) в консоль через ConsoleOutput
 * Полезно для диагностики: проверить nFFT, search_range, инициализацию.
 */
void SpectrumMaximaFinder::PrintInfo() const {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "SpectrumMaxima",
        "Config: antennas=" + std::to_string(params_.antenna_count) +
        " n_point=" + std::to_string(params_.n_point) +
        " repeat=" + std::to_string(params_.repeat_count) +
        " nFFT=" + std::to_string(params_.nFFT) +
        " search_range=" + std::to_string(params_.search_range) +
        " Fs=" + std::to_string(static_cast<int>(params_.sample_rate)) + "Hz" +
        " init=" + (initialized_ ? "Yes" : "No"));
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 4: УТИЛИТЫ — CalculateFFTSize(), NextPowerOf2(), CalculateBytesPerAntenna()
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Вычислить и сохранить base_fft, nFFT, search_range в params_
 * base_fft = nextPow2(n_point), nFFT = base_fft × repeat_count.
 * search_range=0 → авто = nFFT/4 (первая четверть: положительные частоты).
 * Вызывается из Initialize() и лениво из ProcessFromGPU при params_.nFFT==0.
 */
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

/**
 * @brief Следующая степень двойки ≥ n (bit-twiddling, O(1))
 * Используется для выбора размера FFT: hipFFT/clFFT оптимально работают со степенями двойки.
 * @return степень двойки ≥ n; при n=0 возвращает 1
 */
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

/**
 * @brief Оценить потребление GPU памяти на одну антенну (для BatchManager)
 *
 * Используется в двух местах:
 * 1. Initialize() — решить нужен ли batch mode
 * 2. ProcessFromCPU/GPU — вычислить optimal_batch_size
 *
 * Формула зависит от params_.peak_mode:
 * - ONE_PEAK/TWO_PEAKS: input + 3×fft + maxima_output
 * - ALL_MAXIMA: input + 3×fft + pipeline_temp (magnitudes+flags+scan) + compact_output
 *
 * @return Байт GPU памяти на одну антенну (без учёта уже выделенных буферов)
 */
size_t SpectrumMaximaFinder::CalculateBytesPerAntenna() const {
    // Формула памяти на одну антенну для BatchManager:
    // 1. Input data: n_point * sizeof(complex<float>)
    size_t input_bytes = params_.n_point * sizeof(std::complex<float>);  // n_point * 8

    // 2. FFT buffers (input + output + TEMP): 3 * nFFT * sizeof(complex<float>)
    //    clFFT требует temp buffer размером примерно batch * nFFT * 8!
    size_t fft_bytes = 3 * params_.nFFT * sizeof(std::complex<float>);  // 3 * nFFT * 8

    // 3. Output buffers (зависит от режима поиска)
    if (params_.peak_mode == PeakSearchMode::ALL_MAXIMA) {
        // ALL_MAXIMA: временные буферы (magnitudes + flags + scan) + compact output
        // Временные: magnitudes (float) + flags (uint32) + scan (uint32)
        size_t pipeline_bytes = 3 * params_.nFFT * sizeof(uint32_t);  // 3 * nFFT * 4

        // Выходные буферы: positions + magnitudes + counts (на max_maxima_per_beam максимумов)
        size_t output_compact = params_.max_maxima_per_beam *
                               (sizeof(uint32_t) + sizeof(float)) +  // positions + magnitudes
                               sizeof(uint32_t);                     // counts

        return input_bytes + fft_bytes + pipeline_bytes + output_compact;
    }

    // ONE_PEAK / TWO_PEAKS: Maxima output (4 or 8) * sizeof(MaxValue)
    size_t maxima_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
    size_t maxima_bytes = maxima_per_beam * sizeof(MaxValue);  // 4*32=128 or 8*32=256

    return input_bytes + fft_bytes + maxima_bytes;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 5: GPU БУФЕРЫ (single-batch режим)
// AllocateBuffers(), CreateFFTPlanWithCallback(), CompilePostKernel()
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Выделить GPU буферы для single-batch режима (все антенны помещаются в VRAM)
 *
 * Используется только если AllItemsFit() вернул true в Initialize().
 * Для batch mode используется ReallocateBuffersForBatch().
 *
 * Создаваемые буферы:
 *   - pre_callback_userdata_: [32-байт header] + [input data] (все антенны)
 *   - fft_input_, fft_output_: complex<float>[antenna_count × nFFT]
 *   - maxima_output_: MaxValue[antenna_count × (4 или 8)]
 *   (fft_temp_buffer_ и pinned_staging_buffer_ создаются позже в ReallocateBuffersForBatch)
 *
 * @throws std::runtime_error если любой clCreateBuffer провалился
 */
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

/**
 * @brief Создать clFFT план с pre-callback (zero-padding) для single-batch режима
 *
 * Pre-callback "prepareDataPre": читает из pre_callback_userdata_ (offset 32),
 * выполняет zero-padding n_point → nFFT на лету внутри FFT без отдельного kernel'а.
 * Это значительно быстрее чем явный memset + writeBuffer.
 *
 * Почему НЕ static clfft_initialized_: у каждого GPU свой cl_context,
 * clfftSetup() не привязан к контексту (глобальная lib init), но мы
 * отслеживаем флаг per-instance для корректного teardown при деструкторе.
 *
 * @throws std::runtime_error если clfftCreateDefaultPlan, clfftSetPlanCallback или clfftBakePlan провалились
 */
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

/**
 * @brief Скомпилировать post-kernel (peak search) в зависимости от params_.peak_mode
 *
 * Два варианта:
 *   - ONE_PEAK:  "post_kernel_one_peak" — выдаёт 4 MaxValue/луч (interpolated+L+C+R)
 *   - TWO_PEAKS: "post_kernel"          — выдаёт 8 MaxValue/луч (2 пика × 4 MaxValue)
 *
 * Режим ALL_MAXIMA не использует post-kernel — у него свой pipeline (detect+scan+compact).
 * Компилируется один раз, повторные вызовы не проверяются (вызывается из Initialize единожды).
 *
 * При ошибке: выводит build log через ConsoleOutput, освобождает program, бросает runtime_error.
 */
void SpectrumMaximaFinder::CompilePostKernel() {
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
 * @brief Загрузить входные данные на GPU в pre_callback_userdata_ (после заголовка)
 *
 * Два пути (автоматический fallback):
 *
 * PINNED PATH (быстрый, если pinned_staging_buffer_ создан):
 *   1. MapBuffer(CL_MAP_WRITE_INVALIDATE_REGION) → page-locked указатель
 *   2. memcpy данных в pinned память (быстрее чем pageable → UMA)
 *   3. UnmapMemObject → разблокировать буфер (не запускает transfer!)
 *   4. CopyBuffer(pinned → pre_callback_userdata_, offset=PRE_CALLBACK_HEADER_SIZE)
 *   Пропускная способность DMA выше из-за page-locked памяти (нет page faults).
 *
 * REGULAR PATH (fallback если pinned недоступен или переполнен):
 *   clEnqueueWriteBuffer(CL_FALSE, offset=PRE_CALLBACK_HEADER_SIZE)
 *
 * Данные пишутся с offset=PRE_CALLBACK_HEADER_SIZE (32 байта) — не затираем заголовок!
 *
 * @param input_data CPU данные [batch_count × n_point] complex<float>
 * @return cl_event Upload события (caller ОБЯЗАН CollectOrRelease или clReleaseEvent)
 * @throws std::runtime_error если clEnqueueWriteBuffer провалился (только в regular path)
 */
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

/**
 * @brief Запустить clFFT с plan_handle_ (pre-callback встроен в план)
 *
 * fft_input_ — технический буфер clFFT, реальные данные читает pre-callback из pre_callback_userdata_.
 * fft_temp_buffer_ может быть nullptr (clFFT выделит internal temp в этом случае, медленнее).
 *
 * @param wait_event cl_event предыдущей операции (Upload); nullptr — без ожидания
 * @return cl_event FFT операции (caller ОБЯЗАН CollectOrRelease или clReleaseEvent)
 * @throws std::runtime_error если clfftEnqueueTransform провалился
 */
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

/**
 * @brief Запустить post-kernel (peak search) после FFT
 *
 * Читает fft_output_ (complex<float>[batch×nFFT]), пишет в maxima_output_ (MaxValue[batch×4/8]).
 * Использует actual_batch_size_ (реальный batch), а не current_batch_size_ (размер буферов) —
 * это позволяет безопасно обработать последний неполный batch.
 *
 * NDRange: global = actual_batch × LOCAL_SIZE, local = LOCAL_SIZE.
 * Каждая work-group = одна антенна → нет конкуренции за maxima_output_.
 *
 * @param wait_event cl_event FFT операции; nullptr — без ожидания
 * @return cl_event post-kernel операции (caller ОБЯЗАН CollectOrRelease или clReleaseEvent)
 * @throws std::runtime_error если clSetKernelArg или clEnqueueNDRangeKernel провалился
 */
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

/**
 * @brief Читать maxima_output_ с GPU и преобразовать в vector<SpectrumResult>
 *
 * clEnqueueReadBuffer(CL_FALSE) + clWaitForEvents — non-blocking запрос + явное ожидание,
 * чтобы read_event мог быть записан в prof_events (CollectOrRelease).
 *
 * Преобразование MaxValue[] → SpectrumResult[]:
 * - ONE_PEAK:  4 MaxValue/луч → [interpolated, L, C, R] → 1 SpectrumResult/луч
 * - TWO_PEAKS: 8 MaxValue/луч → [L_interp, L_L, L_C, L_R, R_interp, R_L, R_C, R_R]
 *              → 2 SpectrumResult/луч (левый и правый пики)
 *
 * antenna_id заполняется относительным индексом (0..actual_batch-1),
 * корректируется в ProcessBatch()/ProcessBatchFromGPU() на абсолютный.
 *
 * @param wait_event cl_event post-kernel; nullptr — без ожидания
 * @param pe         Список событий для профилирования; "Download" записывается если pe != nullptr
 * @return vector<SpectrumResult> размером actual_batch_size_ (ONE_PEAK) или 2×actual_batch (TWO_PEAKS)
 */
std::vector<SpectrumResult> SpectrumMaximaFinder::ReadResults(cl_event wait_event, ProfEvents* pe) {
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

    clWaitForEvents(1, &read_event);  // ожидаем завершения D2H (CL_FALSE выше)
    CollectOrRelease(read_event, "Download", pe);

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

/**
 * @brief Получить накопленные данные профилирования из GPUProfiler
 * Читает средние времена из GPUProfiler::GetStats(gpu_id)["SpectrumMaxima"].
 * upload_time_ms включает "Upload" и "GPU→GPU Copy" — объединённый transfer time.
 */
ProfilingData SpectrumMaximaFinder::GetProfilingData() const {
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
// ЧАСТЬ 7: ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ — WritePreCallbackHeader(), ReleaseResources()
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Обновить 32-байтный заголовок PreCallbackHeader в начале pre_callback_userdata_
 *
 * Заголовок содержит параметры, которые pre-callback kernel читает на GPU для zero-padding:
 * {beam_count, count_points=n_point, nFFT, pad×5}. Всё — uint32_t.
 *
 * Вызывается при каждом batch (beam_count может меняться у последнего batch).
 * CL_TRUE (blocking write) — гарантирует что заголовок записан до следующего UploadData.
 *
 * @param batch_count Реальное число лучей в текущем batch (не current_batch_size_!)
 */
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

/**
 * @brief Освободить все GPU ресурсы объекта (вызывается из деструктора и move operator=)
 *
 * Порядок освобождения критичен — нарушение может привести к use-after-free:
 * 1. AllMaxima pipeline (kernels + programs) через ReleaseAllMaximaResources()
 * 2. AllMaxima FFT план (allmax_plan_handle_) + magnitudes_buffer_
 * 3. Post-kernel + post-program
 * 4. Основной FFT план (plan_handle_)
 * 5. GPU буферы: pre_callback_userdata_, fft_input/output, maxima_output, temp, pinned
 *
 * После вызова initialized_=false — объект может быть безопасно уничтожен или переиспользован.
 * backend_ НЕ освобождается — мы не владеем им.
 */
void SpectrumMaximaFinder::ReleaseResources() {
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
