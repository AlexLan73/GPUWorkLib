/**
 * @file fft_processor.cpp
 * @brief Реализация FFTProcessor — FFT 1/n лучей с вариантами вывода
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ОГЛАВЛЕНИЕ
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * ЧАСТЬ 1: Конструктор, деструктор, перемещение
 * ЧАСТЬ 2: Публичный API (ProcessComplex, ProcessMagPhase)
 * ЧАСТЬ 3: GPU ресурсы (Allocate, CreatePlan, Compile)
 * ЧАСТЬ 4: GPU операции (Upload, FFT, PostKernel, Read)
 * ЧАСТЬ 5: Утилиты (NextPowerOf2, CalculateNFFT, Profiling)
 *
 * @note Production-код — ЧИСТЫЙ. Нет RecordProfilingEvent внутри.
 *       Профилирование — опционально через prof_events (benchmark-класс).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "fft_processor.hpp"
#include "services/gpu_profiler.hpp"
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace fft_processor {

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 1: Конструктор / Деструктор / Перемещение
// ════════════════════════════════════════════════════════════════════════════

FFTProcessor::FFTProcessor(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

    if (!backend_ || !backend_->IsInitialized()) {
        throw std::runtime_error("FFTProcessor: backend is null or not initialized");
    }

    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
    device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

    if (!context_ || !queue_ || !device_) {
        throw std::runtime_error("FFTProcessor: failed to get OpenCL resources from backend");
    }
}

FFTProcessor::~FFTProcessor() {
    ReleaseResources();
}

FFTProcessor::FFTProcessor(FFTProcessor&& other) noexcept
    : backend_(other.backend_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , plan_handle_(other.plan_handle_)
    , plan_created_(other.plan_created_)
    , clfft_initialized_(other.clfft_initialized_)
    , pre_callback_userdata_(other.pre_callback_userdata_)
    , fft_input_(other.fft_input_)
    , fft_output_(other.fft_output_)
    , fft_temp_buffer_(other.fft_temp_buffer_)
    , mag_output_(other.mag_output_)
    , phase_output_(other.phase_output_)
    , mag_phase_program_(other.mag_phase_program_)
    , mag_phase_kernel_(other.mag_phase_kernel_)
    , nFFT_(other.nFFT_)
    , n_point_(other.n_point_)
    , current_buffer_beams_(other.current_buffer_beams_)
    , plan_batch_size_(other.plan_batch_size_)
    , fft_temp_buffer_size_(other.fft_temp_buffer_size_)
    , has_mag_phase_buffers_(other.has_mag_phase_buffers_) {

    other.plan_handle_ = 0;
    other.plan_created_ = false;
    other.clfft_initialized_ = false;
    other.pre_callback_userdata_ = nullptr;
    other.fft_input_ = nullptr;
    other.fft_output_ = nullptr;
    other.fft_temp_buffer_ = nullptr;
    other.mag_output_ = nullptr;
    other.phase_output_ = nullptr;
    other.mag_phase_program_ = nullptr;
    other.mag_phase_kernel_ = nullptr;
    other.current_buffer_beams_ = 0;
    other.plan_batch_size_ = 0;
    other.fft_temp_buffer_size_ = 0;
    other.has_mag_phase_buffers_ = false;
}

FFTProcessor& FFTProcessor::operator=(FFTProcessor&& other) noexcept {
    if (this != &other) {
        ReleaseResources();

        backend_ = other.backend_;
        context_ = other.context_;
        queue_ = other.queue_;
        device_ = other.device_;
        plan_handle_ = other.plan_handle_;
        plan_created_ = other.plan_created_;
        clfft_initialized_ = other.clfft_initialized_;
        pre_callback_userdata_ = other.pre_callback_userdata_;
        fft_input_ = other.fft_input_;
        fft_output_ = other.fft_output_;
        fft_temp_buffer_ = other.fft_temp_buffer_;
        mag_output_ = other.mag_output_;
        phase_output_ = other.phase_output_;
        mag_phase_program_ = other.mag_phase_program_;
        mag_phase_kernel_ = other.mag_phase_kernel_;
        nFFT_ = other.nFFT_;
        n_point_ = other.n_point_;
        current_buffer_beams_ = other.current_buffer_beams_;
        plan_batch_size_ = other.plan_batch_size_;
        fft_temp_buffer_size_ = other.fft_temp_buffer_size_;
        has_mag_phase_buffers_ = other.has_mag_phase_buffers_;

        other.plan_handle_ = 0;
        other.plan_created_ = false;
        other.clfft_initialized_ = false;
        other.pre_callback_userdata_ = nullptr;
        other.fft_input_ = nullptr;
        other.fft_output_ = nullptr;
        other.fft_temp_buffer_ = nullptr;
        other.mag_output_ = nullptr;
        other.phase_output_ = nullptr;
        other.mag_phase_program_ = nullptr;
        other.mag_phase_kernel_ = nullptr;
        other.current_buffer_beams_ = 0;
        other.plan_batch_size_ = 0;
        other.fft_temp_buffer_size_ = 0;
        other.has_mag_phase_buffers_ = false;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 2: Публичный API
// ════════════════════════════════════════════════════════════════════════════

std::vector<FFTComplexResult> FFTProcessor::ProcessComplex(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    std::vector<std::pair<const char*, cl_event>>* prof_events)
{
    // Валидация
    size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
    if (data.size() != expected) {
        throw std::invalid_argument("ProcessComplex: input size " + std::to_string(data.size()) +
                                     " != expected " + std::to_string(expected));
    }

    // Вычислить nFFT
    CalculateNFFT(params);

    // Рассчитать batch
    size_t bytes_per_beam = CalculateBytesPerBeam(FFTOutputMode::COMPLEX);
    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params.beam_count, bytes_per_beam, params.memory_limit);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params.beam_count, optimal_batch, 3, true);

    std::vector<FFTComplexResult> all_results;
    all_results.reserve(params.beam_count);

    for (const auto& batch : batches) {
        // Подготовить буферы
        AllocateBuffers(batch.count, FFTOutputMode::COMPLEX);
        CreateFFTPlan(batch.count);

        // Upload batch
        const auto* batch_data = data.data() + batch.start * params.n_point;
        cl_event upload_event = UploadData(batch_data, batch.count * params.n_point);
        clFinish(queue_);

        // FFT
        cl_event fft_event = ExecuteFFT(upload_event);
        CollectOrRelease(upload_event, "Upload", prof_events);
        clFinish(queue_);

        // Read complex results
        auto batch_results = ReadComplexResults(fft_event, batch.count, batch.start,
                                                 params.sample_rate, prof_events);
        CollectOrRelease(fft_event, "FFT", prof_events);

        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

std::vector<FFTComplexResult> FFTProcessor::ProcessComplex(
    cl_mem gpu_data,
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes,
    std::vector<std::pair<const char*, cl_event>>* prof_events)
{
    if (!gpu_data) {
        throw std::invalid_argument("ProcessComplex: gpu_data is null");
    }

    CalculateNFFT(params);

    size_t bytes_per_beam = CalculateBytesPerBeam(FFTOutputMode::COMPLEX);
    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : static_cast<size_t>(params.beam_count) * params.n_point * sizeof(std::complex<float>);

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params.beam_count, bytes_per_beam, params.memory_limit, external_memory);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params.beam_count, optimal_batch, 3, true);

    std::vector<FFTComplexResult> all_results;
    all_results.reserve(params.beam_count);

    for (const auto& batch : batches) {
        AllocateBuffers(batch.count, FFTOutputMode::COMPLEX);
        CreateFFTPlan(batch.count);

        size_t src_offset = batch.start * params.n_point * sizeof(std::complex<float>);
        cl_event copy_event = CopyGpuData(gpu_data, src_offset, batch.count * params.n_point);
        clFinish(queue_);

        cl_event fft_event = ExecuteFFT(copy_event);
        CollectOrRelease(copy_event, "GPU_Copy", prof_events);
        clFinish(queue_);

        auto batch_results = ReadComplexResults(fft_event, batch.count, batch.start,
                                                 params.sample_rate, prof_events);
        CollectOrRelease(fft_event, "FFT", prof_events);

        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

std::vector<FFTMagPhaseResult> FFTProcessor::ProcessMagPhase(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    std::vector<std::pair<const char*, cl_event>>* prof_events)
{
    size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
    if (data.size() != expected) {
        throw std::invalid_argument("ProcessMagPhase: input size mismatch");
    }

    CalculateNFFT(params);

    if (!mag_phase_kernel_) {
        CompileMagPhaseKernel();
    }

    size_t bytes_per_beam = CalculateBytesPerBeam(params.output_mode);
    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params.beam_count, bytes_per_beam, params.memory_limit);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params.beam_count, optimal_batch, 3, true);

    bool include_freq = (params.output_mode == FFTOutputMode::MAGNITUDE_PHASE_FREQ);

    std::vector<FFTMagPhaseResult> all_results;
    all_results.reserve(params.beam_count);

    for (const auto& batch : batches) {
        AllocateBuffers(batch.count, params.output_mode);
        CreateFFTPlan(batch.count);

        const auto* batch_data = data.data() + batch.start * params.n_point;
        cl_event upload_event = UploadData(batch_data, batch.count * params.n_point);
        clFinish(queue_);

        cl_event fft_event = ExecuteFFT(upload_event);
        CollectOrRelease(upload_event, "Upload", prof_events);
        clFinish(queue_);

        cl_event post_event = ExecuteMagPhaseKernel(fft_event, batch.count);
        CollectOrRelease(fft_event, "FFT", prof_events);
        clFinish(queue_);

        auto batch_results = ReadMagPhaseResults(post_event, batch.count, batch.start,
                                                  params.sample_rate, include_freq, prof_events);
        CollectOrRelease(post_event, "PostProcessing", prof_events);

        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

std::vector<FFTMagPhaseResult> FFTProcessor::ProcessMagPhase(
    cl_mem gpu_data,
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes,
    std::vector<std::pair<const char*, cl_event>>* prof_events)
{
    if (!gpu_data) {
        throw std::invalid_argument("ProcessMagPhase: gpu_data is null");
    }

    CalculateNFFT(params);

    if (!mag_phase_kernel_) {
        CompileMagPhaseKernel();
    }

    size_t bytes_per_beam = CalculateBytesPerBeam(params.output_mode);
    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : static_cast<size_t>(params.beam_count) * params.n_point * sizeof(std::complex<float>);

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params.beam_count, bytes_per_beam, params.memory_limit, external_memory);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params.beam_count, optimal_batch, 3, true);

    bool include_freq = (params.output_mode == FFTOutputMode::MAGNITUDE_PHASE_FREQ);

    std::vector<FFTMagPhaseResult> all_results;
    all_results.reserve(params.beam_count);

    for (const auto& batch : batches) {
        AllocateBuffers(batch.count, params.output_mode);
        CreateFFTPlan(batch.count);

        size_t src_offset = batch.start * params.n_point * sizeof(std::complex<float>);
        cl_event copy_event = CopyGpuData(gpu_data, src_offset, batch.count * params.n_point);
        clFinish(queue_);

        cl_event fft_event = ExecuteFFT(copy_event);
        CollectOrRelease(copy_event, "GPU_Copy", prof_events);
        clFinish(queue_);

        cl_event post_event = ExecuteMagPhaseKernel(fft_event, batch.count);
        CollectOrRelease(fft_event, "FFT", prof_events);
        clFinish(queue_);

        auto batch_results = ReadMagPhaseResults(post_event, batch.count, batch.start,
                                                  params.sample_rate, include_freq, prof_events);
        CollectOrRelease(post_event, "PostProcessing", prof_events);

        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 3: GPU Resources Management
// ════════════════════════════════════════════════════════════════════════════

void FFTProcessor::AllocateBuffers(size_t batch_beam_count, FFTOutputMode mode) {
    cl_int err;

    bool need_realloc = (batch_beam_count > current_buffer_beams_) || !pre_callback_userdata_;

    if (!need_realloc) {
        // Буферы достаточного размера — обновляем только заголовок (batch_beam_count мог уменьшиться)
        WritePreCallbackHeader(batch_beam_count);

        bool need_mag_phase = (mode != FFTOutputMode::COMPLEX);
        if (need_mag_phase && !has_mag_phase_buffers_) {
            // mag/phase буферы нужны, но не выделены (первый вызов ProcessMagPhase после ProcessComplex).
            // Пустой if-блок — намеренно! Падаем ниже на полный realloc, который
            // пересоздаст ВСЕ буферы включая mag/phase. Это проще и надёжнее, чем
            // частичный realloc только mag/phase при уже живых fft_input_/fft_output_.
        } else {
            return;
        }
    }

    // Освободить старые буферы
    if (pre_callback_userdata_) { clReleaseMemObject(pre_callback_userdata_); pre_callback_userdata_ = nullptr; }
    if (fft_input_) { clReleaseMemObject(fft_input_); fft_input_ = nullptr; }
    if (fft_output_) { clReleaseMemObject(fft_output_); fft_output_ = nullptr; }
    if (mag_output_) { clReleaseMemObject(mag_output_); mag_output_ = nullptr; }
    if (phase_output_) { clReleaseMemObject(phase_output_); phase_output_ = nullptr; }
    has_mag_phase_buffers_ = false;

    // 1. Pre-callback userdata: [32B header][input data]
    size_t input_size = batch_beam_count * n_point_ * sizeof(std::complex<float>);
    size_t userdata_size = PRE_CALLBACK_HEADER_SIZE + input_size;

    pre_callback_userdata_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, userdata_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("AllocateBuffers: pre_callback_userdata failed: " + std::to_string(err));

    WritePreCallbackHeader(batch_beam_count);

    // 2. FFT буферы
    size_t fft_size = batch_beam_count * nFFT_ * sizeof(std::complex<float>);

    fft_input_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, fft_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("AllocateBuffers: fft_input failed: " + std::to_string(err));

    fft_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, fft_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("AllocateBuffers: fft_output failed: " + std::to_string(err));

    // 3. Mag/phase буферы (если нужны)
    if (mode != FFTOutputMode::COMPLEX) {
        size_t scalar_size = batch_beam_count * nFFT_ * sizeof(float);

        mag_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, scalar_size, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("AllocateBuffers: mag_output failed: " + std::to_string(err));

        phase_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, scalar_size, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("AllocateBuffers: phase_output failed: " + std::to_string(err));

        has_mag_phase_buffers_ = true;
    }

    current_buffer_beams_ = batch_beam_count;
}

void FFTProcessor::CreateFFTPlan(size_t batch_beam_count) {
    if (plan_created_ && plan_batch_size_ == batch_beam_count) {
        return;  // План уже создан для этого batch size — переиспользуем
    }

    // clfftSetup() — глобальная инициализация clFFT; вызывается один раз на весь объект.
    // (clFFT использует ref-count: несколько объектов безопасны, clfftTeardown не нужен явно)
    if (!clfft_initialized_) {
        clfftSetupData setup;
        setup.major = clfftVersionMajor;
        setup.minor = clfftVersionMinor;
        setup.patch = clfftVersionPatch;
        setup.debugFlags = 0;
        clfftSetup(&setup);
        clfft_initialized_ = true;
    }

    // OpenCL не имеет two-plan cache (в отличие от ROCm): при смене batch size план пересоздаётся.
    if (plan_created_ && plan_handle_) {
        clfftDestroyPlan(&plan_handle_);
        plan_handle_ = 0;
        plan_created_ = false;
    }

    // Создать 1D batch-план: nFFT_ точек × batch_beam_count лучей
    size_t dim = nFFT_;
    clfftStatus status = clfftCreateDefaultPlan(&plan_handle_, context_, CLFFT_1D, &dim);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("CreateFFTPlan: clfftCreateDefaultPlan failed: " + std::to_string(status));
    }

    clfftSetPlanPrecision(plan_handle_, CLFFT_SINGLE);
    clfftSetLayout(plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    // OUTOFPLACE: fft_input_ → fft_output_ (fft_input_ не трогается после FFT — может переиспользоваться)
    clfftSetResultLocation(plan_handle_, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle_, batch_beam_count);

    // Stride=1, dist=nFFT: лучи лежат непрерывно в памяти [beam0: 0..nFFT-1][beam1: nFFT..2*nFFT-1]...
    size_t strides[1] = {1};
    size_t dist = nFFT_;
    clfftSetPlanInStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle_, dist, dist);

    // Регистрируем pre-callback: clFFT будет вызывать prepareDataPre() вместо чтения из fft_input_.
    // pre_callback_userdata_ содержит заголовок + реальные данные (см. PreCallbackHeader).
    const char* pre_source = kernels::GetPreCallbackSource_opencl();
    status = clfftSetPlanCallback(plan_handle_, "prepareDataPre", pre_source, 0,
                                   PRECALLBACK, &pre_callback_userdata_, 1);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("CreateFFTPlan: clfftSetPlanCallback failed: " + std::to_string(status));
    }

    // clfftBakePlan — финализирует план: выбирает оптимальные размеры work-group,
    // компилирует OpenCL-код под конкретный device. Обязательно перед clfftEnqueueTransform!
    status = clfftBakePlan(plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("CreateFFTPlan: clfftBakePlan failed: " + std::to_string(status));
    }

    // Некоторые размеры FFT требуют scratch-буфера. Выделяем только если нужен больший размер.
    // tmp_size=0 для степеней 2 при малых batch — обычный случай.
    size_t tmp_size = 0;
    clfftGetTmpBufSize(plan_handle_, &tmp_size);
    if (tmp_size > fft_temp_buffer_size_) {
        if (fft_temp_buffer_) clReleaseMemObject(fft_temp_buffer_);
        cl_int err;
        fft_temp_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, tmp_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            clfftDestroyPlan(&plan_handle_);
            throw std::runtime_error("CreateFFTPlan: temp buffer failed: " + std::to_string(err));
        }
        fft_temp_buffer_size_ = tmp_size;
    }

    plan_batch_size_ = batch_beam_count;
    plan_created_ = true;
}

void FFTProcessor::CompileMagPhaseKernel() {
    cl_int err;

    const char* source = kernels::GetMagPhaseKernelSource_opencl();
    size_t source_len = strlen(source);

    mag_phase_program_ = clCreateProgramWithSource(context_, 1, &source, &source_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("CompileMagPhaseKernel: clCreateProgramWithSource failed: " + std::to_string(err));
    }

    err = clBuildProgram(mag_phase_program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(mag_phase_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(mag_phase_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(mag_phase_program_);
        mag_phase_program_ = nullptr;
        throw std::runtime_error("CompileMagPhaseKernel: build failed: " + std::to_string(err));
    }

    mag_phase_kernel_ = clCreateKernel(mag_phase_program_, "complex_to_mag_phase", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(mag_phase_program_);
        mag_phase_program_ = nullptr;
        throw std::runtime_error("CompileMagPhaseKernel: clCreateKernel failed: " + std::to_string(err));
    }
}

void FFTProcessor::WritePreCallbackHeader(size_t batch_beam_count) {
    PreCallbackHeader header = {
        static_cast<uint32_t>(batch_beam_count),
        n_point_,
        nFFT_,
        0, 0, 0, 0, 0
    };

    cl_int err = clEnqueueWriteBuffer(queue_, pre_callback_userdata_, CL_TRUE,
                                       0, sizeof(header), &header, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("WritePreCallbackHeader failed: " + std::to_string(err));
    }
}

void FFTProcessor::ReleaseResources() {
    if (mag_phase_kernel_) { clReleaseKernel(mag_phase_kernel_); mag_phase_kernel_ = nullptr; }
    if (mag_phase_program_) { clReleaseProgram(mag_phase_program_); mag_phase_program_ = nullptr; }

    if (plan_created_ && plan_handle_) {
        clfftDestroyPlan(&plan_handle_);
        plan_handle_ = 0;
        plan_created_ = false;
    }

    if (pre_callback_userdata_) { clReleaseMemObject(pre_callback_userdata_); pre_callback_userdata_ = nullptr; }
    if (fft_input_) { clReleaseMemObject(fft_input_); fft_input_ = nullptr; }
    if (fft_output_) { clReleaseMemObject(fft_output_); fft_output_ = nullptr; }
    if (fft_temp_buffer_) { clReleaseMemObject(fft_temp_buffer_); fft_temp_buffer_ = nullptr; }
    if (mag_output_) { clReleaseMemObject(mag_output_); mag_output_ = nullptr; }
    if (phase_output_) { clReleaseMemObject(phase_output_); phase_output_ = nullptr; }

    current_buffer_beams_ = 0;
    plan_batch_size_ = 0;
    fft_temp_buffer_size_ = 0;
    has_mag_phase_buffers_ = false;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 4: GPU Operations
// ════════════════════════════════════════════════════════════════════════════

cl_event FFTProcessor::UploadData(const std::complex<float>* data, size_t count) {
    cl_event event = nullptr;
    size_t data_size = count * sizeof(std::complex<float>);

    // Данные записываются ПОСЛЕ заголовка в pre_callback_userdata_!
    // Не в fft_input_ — потому что pre-callback читает из userdata (не из fft_input_).
    // PRE_CALLBACK_HEADER_SIZE = 32 байта = смещение сразу за PreCallbackHeader.
    cl_int err = clEnqueueWriteBuffer(
        queue_, pre_callback_userdata_,
        CL_FALSE,
        PRE_CALLBACK_HEADER_SIZE,   // dst_offset: пропустить заголовок
        data_size,
        data,
        0, nullptr, &event);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("UploadData failed: " + std::to_string(err));
    }
    return event;
}

cl_event FFTProcessor::CopyGpuData(cl_mem src, size_t src_offset, size_t count) {
    cl_event event = nullptr;
    size_t data_size = count * sizeof(std::complex<float>);

    cl_int err = clEnqueueCopyBuffer(
        queue_, src, pre_callback_userdata_,
        src_offset, PRE_CALLBACK_HEADER_SIZE,
        data_size,
        0, nullptr, &event);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("CopyGpuData failed: " + std::to_string(err));
    }
    return event;
}

cl_event FFTProcessor::ExecuteFFT(cl_event wait_event) {
    cl_event event = nullptr;

    clfftStatus status = clfftEnqueueTransform(
        plan_handle_,
        CLFFT_FORWARD,
        1, &queue_,
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &event,
        &fft_input_,
        &fft_output_,
        fft_temp_buffer_);

    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("ExecuteFFT: clfftEnqueueTransform failed: " + std::to_string(status));
    }
    return event;
}

cl_event FFTProcessor::ExecuteMagPhaseKernel(cl_event wait_event, size_t beam_count) {
    cl_int err;
    cl_event event = nullptr;

    uint32_t bc = static_cast<uint32_t>(beam_count);
    uint32_t nfft = nFFT_;

    err  = clSetKernelArg(mag_phase_kernel_, 0, sizeof(cl_mem), &fft_output_);
    err |= clSetKernelArg(mag_phase_kernel_, 1, sizeof(cl_mem), &mag_output_);
    err |= clSetKernelArg(mag_phase_kernel_, 2, sizeof(cl_mem), &phase_output_);
    err |= clSetKernelArg(mag_phase_kernel_, 3, sizeof(uint32_t), &bc);
    err |= clSetKernelArg(mag_phase_kernel_, 4, sizeof(uint32_t), &nfft);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("ExecuteMagPhaseKernel: clSetKernelArg failed: " + std::to_string(err));
    }

    size_t total = beam_count * nFFT_;
    size_t local_size = 256;
    size_t global_size = ((total + local_size - 1) / local_size) * local_size;

    err = clEnqueueNDRangeKernel(
        queue_, mag_phase_kernel_, 1, nullptr,
        &global_size, &local_size,
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &event);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("ExecuteMagPhaseKernel: clEnqueueNDRangeKernel failed: " + std::to_string(err));
    }
    return event;
}

std::vector<FFTComplexResult> FFTProcessor::ReadComplexResults(
    cl_event wait_event, size_t beam_count, size_t start_beam,
    float sample_rate,
    std::vector<std::pair<const char*, cl_event>>* prof_events)
{
    size_t total = beam_count * nFFT_;
    std::vector<std::complex<float>> raw(total);

    cl_event read_event = nullptr;
    cl_int err = clEnqueueReadBuffer(
        queue_, fft_output_, CL_FALSE,
        0, total * sizeof(std::complex<float>),
        raw.data(),
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &read_event);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("ReadComplexResults failed: " + std::to_string(err));
    }

    clFinish(queue_);
    CollectOrRelease(read_event, "Download", prof_events);

    // Разбить на лучи
    std::vector<FFTComplexResult> results;
    results.reserve(beam_count);

    for (size_t i = 0; i < beam_count; ++i) {
        FFTComplexResult result;
        result.beam_id = static_cast<uint32_t>(start_beam + i);
        result.nFFT = nFFT_;
        result.sample_rate = sample_rate;
        result.spectrum.assign(
            raw.begin() + i * nFFT_,
            raw.begin() + (i + 1) * nFFT_);
        results.push_back(std::move(result));
    }

    return results;
}

std::vector<FFTMagPhaseResult> FFTProcessor::ReadMagPhaseResults(
    cl_event wait_event, size_t beam_count, size_t start_beam,
    float sample_rate, bool include_freq,
    std::vector<std::pair<const char*, cl_event>>* prof_events)
{
    size_t total = beam_count * nFFT_;

    std::vector<float> raw_mag(total);
    std::vector<float> raw_phase(total);

    // Читаем mag и phase параллельно (non-blocking)
    cl_event mag_event = nullptr;
    cl_int err = clEnqueueReadBuffer(
        queue_, mag_output_, CL_FALSE,
        0, total * sizeof(float), raw_mag.data(),
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &mag_event);
    if (err != CL_SUCCESS) throw std::runtime_error("ReadMagPhaseResults: mag read failed");

    cl_event phase_event = nullptr;
    err = clEnqueueReadBuffer(
        queue_, phase_output_, CL_FALSE,
        0, total * sizeof(float), raw_phase.data(),
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &phase_event);
    if (err != CL_SUCCESS) throw std::runtime_error("ReadMagPhaseResults: phase read failed");

    // Ждём оба
    clFinish(queue_);
    CollectOrRelease(mag_event, "Download", prof_events);
    CollectOrRelease(phase_event, "Download", prof_events);

    // Разбить на лучи
    std::vector<FFTMagPhaseResult> results;
    results.reserve(beam_count);

    for (size_t i = 0; i < beam_count; ++i) {
        FFTMagPhaseResult result;
        result.beam_id = static_cast<uint32_t>(start_beam + i);
        result.nFFT = nFFT_;
        result.sample_rate = sample_rate;

        result.magnitude.assign(
            raw_mag.begin() + i * nFFT_,
            raw_mag.begin() + (i + 1) * nFFT_);

        result.phase.assign(
            raw_phase.begin() + i * nFFT_,
            raw_phase.begin() + (i + 1) * nFFT_);

        // frequency[] вычисляется на CPU: запуск отдельного kernel для nFFT умножений
        // не имеет смысла — overhead kernel launch >> само вычисление.
        if (include_freq) {
            result.frequency.resize(nFFT_);
            float freq_step = sample_rate / static_cast<float>(nFFT_);
            for (uint32_t k = 0; k < nFFT_; ++k) {
                result.frequency[k] = static_cast<float>(k) * freq_step;
            }
        }

        results.push_back(std::move(result));
    }

    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// ЧАСТЬ 5: Утилиты
// ════════════════════════════════════════════════════════════════════════════

// Bit-twiddling: заполняем все биты ниже старшего → n+1 = следующая степень двойки.
// Пример: 1000 (8) → 0111 (7) → fill → 1111 (15) → +1 = 10000 (16).
// Особый случай: n=0 → 1 (hipFFT/clFFT требуют nFFT >= 1)
uint32_t FFTProcessor::NextPowerOf2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void FFTProcessor::CalculateNFFT(const FFTProcessorParams& params) {
    n_point_ = params.n_point;
    uint32_t base_fft = NextPowerOf2(params.n_point);
    nFFT_ = base_fft * params.repeat_count;
}

size_t FFTProcessor::CalculateBytesPerBeam(FFTOutputMode mode) const {
    // Input data
    size_t input_bytes = n_point_ * sizeof(std::complex<float>);

    // FFT buffers (input + output + temp): 3 * nFFT * sizeof(complex<float>)
    size_t fft_bytes = 3 * nFFT_ * sizeof(std::complex<float>);

    // Mag/phase buffers
    size_t post_bytes = 0;
    if (mode != FFTOutputMode::COMPLEX) {
        post_bytes = 2 * nFFT_ * sizeof(float);  // mag + phase
    }

    return input_bytes + fft_bytes + post_bytes;
}

FFTProfilingData FFTProcessor::GetProfilingData() const {
    FFTProfilingData out{};
    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;
    auto stats = drv_gpu_lib::GPUProfiler::GetInstance().GetStats(gpu_id);
    auto it = stats.find("FFTProcessor");
    if (it == stats.end()) return out;

    const auto& mod = it->second;
    auto ev = [&mod](const char* name) -> double {
        auto e = mod.events.find(name);
        return (e != mod.events.end()) ? e->second.GetAvgTimeMs() : 0.0;
    };
    out.upload_time_ms = ev("Upload") + ev("GPU_Copy");
    out.fft_time_ms = ev("FFT");
    out.post_processing_time_ms = ev("PostProcessing");
    out.download_time_ms = ev("Download");
    out.total_time_ms = mod.GetTotalTimeMs();
    return out;
}

} // namespace fft_processor
