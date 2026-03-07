/**
 * @file fft_processor_rocm.cpp
 * @brief Implementation of FFTProcessorROCm -- FFT using hipFFT + hiprtc kernels
 *
 * =========================================================================
 * CONTENTS
 * =========================================================================
 *
 * PART 1: Constructor, Destructor, Move Semantics
 * PART 2: Public API (ProcessComplex, ProcessMagPhase)
 * PART 3: GPU Resources (Allocate, CreatePlan, CompileKernels)
 * PART 4: GPU Operations (Upload, Pad, FFT, MagPhase, Read)
 * PART 5: Utilities (NextPowerOf2, CalculateNFFT, Profiling)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "fft_processor_rocm.hpp"
#include "kernels/fft_processor_kernels_rocm.hpp"
#include "services/gpu_profiler.hpp"
#include "config/gpu_config.hpp"
#include "logger/logger.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"
#include "services/kernel_cache_service.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <iostream>
#include <chrono>

namespace fft_processor {

// =========================================================================
// Helper: create ROCmProfilingData from hipEvent timing
// =========================================================================

/**
 * @brief Конвертирует пару hipEvent → ROCmProfilingData и уничтожает events.
 *
 * hipEventElapsedTime возвращает относительное время (мс) между двумя event'ами.
 * ROCmProfilingData ожидает абсолютные метки в нс (start_ns, end_ns).
 * Решение: синтетические метки start=0, end=elapsed_ns — разница корректна,
 * GetAvgTimeMs() использует (end_ns - start_ns)/1e6, результат верный.
 * Абсолютные метки (queued/submit) не важны для бенчмаркинга — ставим 0.
 */
static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromEvents(
    hipEvent_t ev_start, hipEvent_t ev_end, uint32_t kind,
    const char* op_string = "")
{
    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, ev_start, ev_end);
    hipEventDestroy(ev_start);
    hipEventDestroy(ev_end);

    drv_gpu_lib::ROCmProfilingData d;
    uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_ms * 1e6f);
    d.queued_ns   = 0;
    d.submit_ns   = 0;
    d.start_ns    = 0;
    d.end_ns      = elapsed_ns;
    d.complete_ns = elapsed_ns;
    d.kind        = kind;
    d.op_string   = op_string;
    return d;
}

/**
 * @brief Конвертирует пару wall-clock временных точек → ROCmProfilingData.
 *
 * Используется для синхронных операций (hipMemcpyDtoH), которые блокируют CPU —
 * hipEvent там нельзя использовать (нет async-записи). std::chrono измеряет
 * реальное время включая ожидание device-sync, что корректно отражает overhead DtoH.
 */
static drv_gpu_lib::ROCmProfilingData MakeROCmDataFromClock(
    std::chrono::high_resolution_clock::time_point t_start,
    std::chrono::high_resolution_clock::time_point t_end,
    uint32_t kind, const char* op_string = "")
{
    uint64_t elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());

    drv_gpu_lib::ROCmProfilingData d;
    d.queued_ns   = 0;
    d.submit_ns   = 0;
    d.start_ns    = 0;
    d.end_ns      = elapsed_ns;
    d.complete_ns = elapsed_ns;
    d.kind        = kind;
    d.op_string   = op_string;
    return d;
}

// =========================================================================
// PART 1: Constructor / Destructor / Move Semantics
// =========================================================================

FFTProcessorROCm::FFTProcessorROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

    if (!backend_ || !backend_->IsInitialized()) {
        throw std::runtime_error("FFTProcessorROCm: backend is null or not initialized");
    }

    if (backend_->GetType() != drv_gpu_lib::BackendType::ROCm) {
        throw std::runtime_error("FFTProcessorROCm: requires ROCm backend, got different type");
    }

    stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
    if (!stream_) {
        throw std::runtime_error("FFTProcessorROCm: failed to get HIP stream from backend");
    }

    // TASK-1: Initialize disk cache for compiled HSACO kernels
    kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
        "modules/fft_processor/kernels", drv_gpu_lib::BackendType::ROCm);
}

FFTProcessorROCm::~FFTProcessorROCm() {
    ReleaseResources();
}

FFTProcessorROCm::FFTProcessorROCm(FFTProcessorROCm&& other) noexcept
    : backend_(other.backend_)
    , stream_(other.stream_)
    , plan_(other.plan_)
    , plan_created_(other.plan_created_)
    , plan_last_(other.plan_last_)
    , plan_last_batch_(other.plan_last_batch_)
    , input_buffer_(other.input_buffer_)
    , fft_input_(other.fft_input_)
    , fft_output_(other.fft_output_)
    , mag_phase_interleaved_(other.mag_phase_interleaved_)
    , kernel_cache_(std::move(other.kernel_cache_))
    , module_(other.module_)
    , pad_kernel_(other.pad_kernel_)
    , mag_phase_kernel_(other.mag_phase_kernel_)
    , kernels_compiled_(other.kernels_compiled_)
    , nFFT_(other.nFFT_)
    , n_point_(other.n_point_)
    , current_buffer_beams_(other.current_buffer_beams_)
    , plan_batch_size_(other.plan_batch_size_)
    , has_mag_phase_buffers_(other.has_mag_phase_buffers_) {

    other.backend_ = nullptr;
    other.stream_ = nullptr;
    other.plan_ = 0;
    other.plan_created_ = false;
    other.plan_last_ = 0;
    other.plan_last_batch_ = 0;
    other.input_buffer_ = nullptr;
    other.fft_input_ = nullptr;
    other.fft_output_ = nullptr;
    other.mag_phase_interleaved_ = nullptr;
    other.module_ = nullptr;
    other.pad_kernel_ = nullptr;
    other.mag_phase_kernel_ = nullptr;
    other.kernels_compiled_ = false;
    other.current_buffer_beams_ = 0;
    other.plan_batch_size_ = 0;
    other.has_mag_phase_buffers_ = false;
}

FFTProcessorROCm& FFTProcessorROCm::operator=(FFTProcessorROCm&& other) noexcept {
    if (this != &other) {
        ReleaseResources();

        backend_ = other.backend_;
        stream_ = other.stream_;
        plan_ = other.plan_;
        plan_created_ = other.plan_created_;
        plan_last_ = other.plan_last_;
        plan_last_batch_ = other.plan_last_batch_;
        input_buffer_ = other.input_buffer_;
        fft_input_ = other.fft_input_;
        fft_output_ = other.fft_output_;
        mag_phase_interleaved_ = other.mag_phase_interleaved_;
        kernel_cache_ = std::move(other.kernel_cache_);
        module_ = other.module_;
        pad_kernel_ = other.pad_kernel_;
        mag_phase_kernel_ = other.mag_phase_kernel_;
        kernels_compiled_ = other.kernels_compiled_;
        nFFT_ = other.nFFT_;
        n_point_ = other.n_point_;
        current_buffer_beams_ = other.current_buffer_beams_;
        plan_batch_size_ = other.plan_batch_size_;
        has_mag_phase_buffers_ = other.has_mag_phase_buffers_;

        other.backend_ = nullptr;
        other.stream_ = nullptr;
        other.plan_ = 0;
        other.plan_created_ = false;
        other.plan_last_ = 0;
        other.plan_last_batch_ = 0;
        other.input_buffer_ = nullptr;
        other.fft_input_ = nullptr;
        other.fft_output_ = nullptr;
        other.mag_phase_interleaved_ = nullptr;
        other.module_ = nullptr;
        other.pad_kernel_ = nullptr;
        other.mag_phase_kernel_ = nullptr;
        other.kernels_compiled_ = false;
        other.current_buffer_beams_ = 0;
        other.plan_batch_size_ = 0;
        other.has_mag_phase_buffers_ = false;
    }
    return *this;
}

// =========================================================================
// PART 2: Public API
// =========================================================================

std::vector<FFTComplexResult> FFTProcessorROCm::ProcessComplex(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    ROCmProfEvents* prof_events)
{
    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;

    // Validate
    size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
    if (data.size() != expected) {
        throw std::invalid_argument("ProcessComplex: input size " + std::to_string(data.size()) +
                                     " != expected " + std::to_string(expected));
    }

    CalculateNFFT(params);

    if (!kernels_compiled_) {
        CompileKernels();
    }

    // Calculate batch
    size_t bytes_per_beam = CalculateBytesPerBeam(FFTOutputMode::COMPLEX);
    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params.beam_count, bytes_per_beam, params.memory_limit);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params.beam_count, optimal_batch, 3, true);

    std::vector<FFTComplexResult> all_results;
    all_results.reserve(params.beam_count);

    for (const auto& batch : batches) {
        AllocateBuffers(batch.count, FFTOutputMode::COMPLEX);
        CreateFFTPlan(batch.count);

        // ── HIP events для async-операций (Upload, Pad, FFT) ────────────
        hipEvent_t ev_upload_start = nullptr, ev_upload_end = nullptr;
        hipEvent_t ev_pad_start    = nullptr, ev_pad_end    = nullptr;
        hipEvent_t ev_fft_start    = nullptr, ev_fft_end    = nullptr;
        if (prof_events) {
            hipEventCreate(&ev_upload_start); hipEventCreate(&ev_upload_end);
            hipEventCreate(&ev_pad_start);    hipEventCreate(&ev_pad_end);
            hipEventCreate(&ev_fft_start);    hipEventCreate(&ev_fft_end);
        }

        // Upload
        const auto* batch_data = data.data() + batch.start * params.n_point;
        if (prof_events) hipEventRecord(ev_upload_start, stream_);
        UploadData(batch_data, batch.count * params.n_point);
        if (prof_events) hipEventRecord(ev_upload_end, stream_);

        // Pad: input_buffer_ -> fft_input_
        if (prof_events) hipEventRecord(ev_pad_start, stream_);
        ExecutePadKernel(batch.count);
        if (prof_events) hipEventRecord(ev_pad_end, stream_);

        // FFT: fft_input_ -> fft_output_
        if (prof_events) hipEventRecord(ev_fft_start, stream_);
        ExecuteFFT();
        if (prof_events) hipEventRecord(ev_fft_end, stream_);

        // Sync — ждём завершения всех GPU-операций (и events)
        hipStreamSynchronize(stream_);

        // Извлечь GPU-тайминги после sync
        if (prof_events) {
            prof_events->push_back({"Upload", MakeROCmDataFromEvents(
                ev_upload_start, ev_upload_end, 1, "H2D copy")});
            prof_events->push_back({"Pad", MakeROCmDataFromEvents(
                ev_pad_start, ev_pad_end, 0, "pad kernel")});
            prof_events->push_back({"FFT", MakeROCmDataFromEvents(
                ev_fft_start, ev_fft_end, 0, "hipfftExecC2C")});
        }

        // Download (синхронный DtoH — измеряем wall-clock)
        auto t_dl_start = std::chrono::high_resolution_clock::now();
        auto batch_results = ReadComplexResults(batch.count, batch.start, params.sample_rate);
        auto t_dl_end = std::chrono::high_resolution_clock::now();
        if (prof_events) {
            prof_events->push_back({"Download", MakeROCmDataFromClock(
                t_dl_start, t_dl_end, 1, "D2H copy")});
        }

        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

std::vector<FFTComplexResult> FFTProcessorROCm::ProcessComplex(
    void* gpu_data,
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes)
{
    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;

    if (!gpu_data) {
        throw std::invalid_argument("ProcessComplex: gpu_data is null");
    }

    CalculateNFFT(params);

    if (!kernels_compiled_) {
        CompileKernels();
    }

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

        // D2D copy
        size_t src_offset = batch.start * params.n_point * sizeof(std::complex<float>);
        CopyGpuData(gpu_data, src_offset, batch.count * params.n_point);

        // Pad + FFT
        ExecutePadKernel(batch.count);
        ExecuteFFT();
        hipStreamSynchronize(stream_);

        auto batch_results = ReadComplexResults(batch.count, batch.start, params.sample_rate);
        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

std::vector<FFTMagPhaseResult> FFTProcessorROCm::ProcessMagPhase(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    ROCmProfEvents* prof_events)
{
    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;

    size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
    if (data.size() != expected) {
        throw std::invalid_argument("ProcessMagPhase: input size mismatch");
    }

    CalculateNFFT(params);

    if (!kernels_compiled_) {
        CompileKernels();
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

        // ── HIP events ───────────────────────────────────────────────────
        hipEvent_t ev_upload_start = nullptr, ev_upload_end   = nullptr;
        hipEvent_t ev_pad_start    = nullptr, ev_pad_end      = nullptr;
        hipEvent_t ev_fft_start    = nullptr, ev_fft_end      = nullptr;
        hipEvent_t ev_mag_start    = nullptr, ev_mag_end      = nullptr;
        if (prof_events) {
            hipEventCreate(&ev_upload_start); hipEventCreate(&ev_upload_end);
            hipEventCreate(&ev_pad_start);    hipEventCreate(&ev_pad_end);
            hipEventCreate(&ev_fft_start);    hipEventCreate(&ev_fft_end);
            hipEventCreate(&ev_mag_start);    hipEventCreate(&ev_mag_end);
        }

        // Upload
        const auto* batch_data = data.data() + batch.start * params.n_point;
        if (prof_events) hipEventRecord(ev_upload_start, stream_);
        UploadData(batch_data, batch.count * params.n_point);
        if (prof_events) hipEventRecord(ev_upload_end, stream_);

        // Pad
        if (prof_events) hipEventRecord(ev_pad_start, stream_);
        ExecutePadKernel(batch.count);
        if (prof_events) hipEventRecord(ev_pad_end, stream_);

        // FFT
        if (prof_events) hipEventRecord(ev_fft_start, stream_);
        ExecuteFFT();
        if (prof_events) hipEventRecord(ev_fft_end, stream_);

        // MagPhase kernel
        if (prof_events) hipEventRecord(ev_mag_start, stream_);
        ExecuteMagPhaseKernel(batch.count);
        if (prof_events) hipEventRecord(ev_mag_end, stream_);

        hipStreamSynchronize(stream_);

        if (prof_events) {
            prof_events->push_back({"Upload", MakeROCmDataFromEvents(
                ev_upload_start, ev_upload_end, 1, "H2D copy")});
            prof_events->push_back({"Pad", MakeROCmDataFromEvents(
                ev_pad_start, ev_pad_end, 0, "pad kernel")});
            prof_events->push_back({"FFT", MakeROCmDataFromEvents(
                ev_fft_start, ev_fft_end, 0, "hipfftExecC2C")});
            prof_events->push_back({"MagPhase", MakeROCmDataFromEvents(
                ev_mag_start, ev_mag_end, 0, "mag_phase kernel")});
        }

        // Download (синхронный — wall-clock)
        auto t_dl_start = std::chrono::high_resolution_clock::now();
        auto batch_results = ReadMagPhaseResults(batch.count, batch.start,
                                                  params.sample_rate, include_freq);
        auto t_dl_end = std::chrono::high_resolution_clock::now();
        if (prof_events) {
            prof_events->push_back({"Download", MakeROCmDataFromClock(
                t_dl_start, t_dl_end, 1, "D2H copy")});
        }

        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

std::vector<FFTMagPhaseResult> FFTProcessorROCm::ProcessMagPhase(
    void* gpu_data,
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes)
{
    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;

    if (!gpu_data) {
        throw std::invalid_argument("ProcessMagPhase: gpu_data is null");
    }

    CalculateNFFT(params);

    if (!kernels_compiled_) {
        CompileKernels();
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
        CopyGpuData(gpu_data, src_offset, batch.count * params.n_point);

        ExecutePadKernel(batch.count);
        ExecuteFFT();
        ExecuteMagPhaseKernel(batch.count);
        hipStreamSynchronize(stream_);

        auto batch_results = ReadMagPhaseResults(batch.count, batch.start,
                                                  params.sample_rate, include_freq);
        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

// =========================================================================
// PART 3: GPU Resources Management
// =========================================================================

void FFTProcessorROCm::AllocateBuffers(size_t batch_beam_count, FFTOutputMode mode) {
    bool need_realloc = (batch_beam_count > current_buffer_beams_) || !input_buffer_;

    if (!need_realloc) {
        bool need_mag_phase = (mode != FFTOutputMode::COMPLEX);
        if (need_mag_phase && !has_mag_phase_buffers_) {
            // Need to allocate mag/phase buffers
        } else {
            return;
        }
    }

    // Free old buffers
    if (input_buffer_)         { (void)hipFree(input_buffer_);         input_buffer_ = nullptr; }
    if (fft_input_)            { (void)hipFree(fft_input_);            fft_input_ = nullptr; }
    if (fft_output_)           { (void)hipFree(fft_output_);           fft_output_ = nullptr; }
    if (mag_phase_interleaved_){ (void)hipFree(mag_phase_interleaved_);mag_phase_interleaved_ = nullptr; }
    has_mag_phase_buffers_ = false;

    hipError_t err;

    // 1. Input buffer (raw data, not padded)
    size_t input_size = batch_beam_count * n_point_ * sizeof(std::complex<float>);
    err = hipMalloc(&input_buffer_, input_size);
    if (err != hipSuccess) {
        throw std::runtime_error("AllocateBuffers: input_buffer hipMalloc failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // 2. FFT buffers (padded)
    size_t fft_size = batch_beam_count * nFFT_ * sizeof(std::complex<float>);
    err = hipMalloc(&fft_input_, fft_size);
    if (err != hipSuccess) {
        throw std::runtime_error("AllocateBuffers: fft_input hipMalloc failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    err = hipMalloc(&fft_output_, fft_size);
    if (err != hipSuccess) {
        throw std::runtime_error("AllocateBuffers: fft_output hipMalloc failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // 3. Interleaved mag+phase buffer (TASK-4A: single buffer instead of two)
    if (mode != FFTOutputMode::COMPLEX) {
        // float2_t per element: .x=mag, .y=phase  (2 floats = 8 bytes total)
        size_t interleaved_size = batch_beam_count * nFFT_ * 2 * sizeof(float);

        err = hipMalloc(&mag_phase_interleaved_, interleaved_size);
        if (err != hipSuccess) {
            throw std::runtime_error("AllocateBuffers: mag_phase_interleaved hipMalloc failed: " +
                                      std::string(hipGetErrorString(err)));
        }

        has_mag_phase_buffers_ = true;
    }

    current_buffer_beams_ = batch_beam_count;
}

void FFTProcessorROCm::CreateFFTPlan(size_t batch_beam_count) {
    // Активный план подходит — переиспользуем
    if (plan_created_ && plan_batch_size_ == batch_beam_count) {
        return;
    }

    // Two-plan LRU-2 cache: позволяет чередовать два разных batch size без пересоздания планов.
    // Типичный кейс: BatchManager разбивает N лучей на батчи, последний меньше остальных.
    // → первые батчи используют plan_ (большой), последний — plan_last_ (меньший), оба готовы.
    if (plan_last_ != 0 && plan_last_batch_ == batch_beam_count) {
        // Размер совпадает с вторичным кешем — swap: plan_last_ становится активным
        std::swap(plan_, plan_last_);
        std::swap(plan_batch_size_, plan_last_batch_);
        plan_created_ = true;
        return;
    }

    // Нужен план нового размера: вытесняем вторичный кеш (он больше не нужен),
    // перемещаем текущий активный план в кеш (может пригодиться для чередования).
    // ВНИМАНИЕ: если используются 3+ разных размеров подряд — старый plan_ теряется,
    // при следующем обращении к нему план пересоздаётся. Для матричного бенчмарка
    // с N разными nFFT → создавать отдельный экземпляр FFTProcessorROCm на каждый размер!
    if (plan_last_ != 0) {
        hipfftDestroy(plan_last_);
        plan_last_ = 0;
        plan_last_batch_ = 0;
    }
    if (plan_created_) {
        plan_last_ = plan_;
        plan_last_batch_ = plan_batch_size_;
        plan_ = 0;
        plan_created_ = false;
    }

    // Create 1D batch FFT plan
    hipfftResult result = hipfftPlan1d(&plan_, static_cast<int>(nFFT_),
                                        HIPFFT_C2C,
                                        static_cast<int>(batch_beam_count));
    if (result != HIPFFT_SUCCESS) {
        throw std::runtime_error("CreateFFTPlan: hipfftPlan1d failed: " +
                                  std::to_string(static_cast<int>(result)));
    }

    result = hipfftSetStream(plan_, stream_);
    if (result != HIPFFT_SUCCESS) {
        hipfftDestroy(plan_);
        plan_ = 0;
        throw std::runtime_error("CreateFFTPlan: hipfftSetStream failed: " +
                                  std::to_string(static_cast<int>(result)));
    }

    plan_batch_size_ = batch_beam_count;
    plan_created_ = true;
}

void FFTProcessorROCm::CompileKernels() {
    if (kernels_compiled_) return;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;
    constexpr const char* kCacheName = "fft_processor_kernels";

    // ─── Helper: load kernel functions from loaded module ──────────────────
    auto getKernel = [this](hipFunction_t* func, const char* name) {
        hipError_t err = hipModuleGetFunction(func, module_, name);
        if (err != hipSuccess) {
            throw std::runtime_error(std::string("CompileKernels: hipModuleGetFunction(") +
                                      name + ") failed: " + hipGetErrorString(err));
        }
    };

    // ─── Try loading from HSACO disk cache ─────────────────────────────────
    if (kernel_cache_) {
        try {
            auto entry = kernel_cache_->Load(kCacheName);
            if (entry.has_binary()) {
                hipError_t hip_err = hipModuleLoadData(&module_, entry.binary.data());
                if (hip_err == hipSuccess) {
                    getKernel(&pad_kernel_,       "pad_data");
                    getKernel(&mag_phase_kernel_, "complex_to_mag_phase");
                    kernels_compiled_ = true;
                    con.Print(gpu_id, "FFTProc", "kernels loaded from HSACO cache");
                    DRVGPU_LOG_INFO_GPU(gpu_id, "FFTProcessorROCm", "HIP kernels loaded from cache");
                    return;
                }
                // Stale cache: .hsaco скомпилирован для другой архитектуры GPU (напр., gfx1100 → gfx1201).
                // hipModuleLoadData вернула не hipSuccess → модуль не загружен, нужна перекомпиляция.
                (void)hipModuleUnload(module_);
                module_ = nullptr;
            }
        } catch (...) {
            // Cache miss or corrupt — fall through to compile
        }
    }

    // ─── Get target architecture for --offload-arch ────────────────────────
    std::string arch_name;
    try {
        auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
        arch_name = rocm_backend->GetCore().GetArchName();
    } catch (...) {
        arch_name = "";
    }

    // ─── Compile via hiprtc with optimization flags ─────────────────────────
    const char* source = kernels::GetHIPKernelSource();

    hiprtcProgram prog;
    hiprtcResult rtcResult = hiprtcCreateProgram(&prog, source, "fft_kernels.hip",
                                                  0, nullptr, nullptr);
    if (rtcResult != HIPRTC_SUCCESS) {
        throw std::runtime_error("CompileKernels: hiprtcCreateProgram failed: " +
                                  std::to_string(static_cast<int>(rtcResult)));
    }

    // P4: Determine WARP_SIZE by GPU architecture (CDNA=64, RDNA=32)
    int warp_size = 32;  // default: RDNA (gfx10xx, gfx11xx, gfx12xx)
    if (arch_name.find("gfx9") == 0) {
        warp_size = 64;  // CDNA / Vega (gfx900, gfx906, gfx908, gfx90a, gfx940, gfx942)
    }
    std::string warp_define = "-DWARP_SIZE=" + std::to_string(warp_size);
    std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);
    std::vector<const char*> opts = {"-O3", "-std=c++17", warp_define.c_str()};
    if (!arch_flag.empty()) {
        opts.push_back(arch_flag.c_str());
    }

    rtcResult = hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    if (rtcResult != HIPRTC_SUCCESS) {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        (void)hiprtcDestroyProgram(&prog);
        throw std::runtime_error("CompileKernels: compilation failed:\n" + log);
    }

    size_t codeSize = 0;
    hiprtcGetCodeSize(prog, &codeSize);
    std::vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());
    (void)hiprtcDestroyProgram(&prog);

    hipError_t hipErr = hipModuleLoadData(&module_, code.data());
    if (hipErr != hipSuccess) {
        throw std::runtime_error("CompileKernels: hipModuleLoadData failed: " +
                                  std::string(hipGetErrorString(hipErr)));
    }

    getKernel(&pad_kernel_,       "pad_data");
    getKernel(&mag_phase_kernel_, "complex_to_mag_phase");

    kernels_compiled_ = true;
    con.Print(gpu_id, "FFTProc",
              "kernels compiled (" + std::to_string(codeSize) + " bytes HSACO" +
              (arch_name.empty() ? "" : ", " + arch_name) + ")");
    DRVGPU_LOG_INFO_GPU(gpu_id, "FFTProcessorROCm", "HIP kernels compiled successfully");

    // ─── Save compiled binary to disk cache ────────────────────────────────
    if (kernel_cache_) {
        try {
            std::vector<uint8_t> binary(code.begin(), code.end());
            kernel_cache_->Save(kCacheName, source, binary, "", "fft_processor hiprtc kernels");
            con.Print(gpu_id, "FFTProc", "kernels saved to HSACO cache");
        } catch (const std::exception& e) {
            con.Print(gpu_id, "FFTProc", "warning: cache save failed: " + std::string(e.what()));
        }
    }
}

void FFTProcessorROCm::ReleaseResources() {
    // Destroy hipFFT plans (active + secondary cache)
    if (plan_created_) {
        hipfftDestroy(plan_);
        plan_ = 0;
        plan_created_ = false;
    }
    if (plan_last_ != 0) {
        hipfftDestroy(plan_last_);
        plan_last_ = 0;
        plan_last_batch_ = 0;
    }

    // Free GPU buffers
    if (input_buffer_)         { (void)hipFree(input_buffer_);         input_buffer_ = nullptr; }
    if (fft_input_)            { (void)hipFree(fft_input_);            fft_input_ = nullptr; }
    if (fft_output_)           { (void)hipFree(fft_output_);           fft_output_ = nullptr; }
    if (mag_phase_interleaved_){ (void)hipFree(mag_phase_interleaved_);mag_phase_interleaved_ = nullptr; }

    // Unload hiprtc module
    if (module_) {
        (void)hipModuleUnload(module_);
        module_ = nullptr;
        pad_kernel_ = nullptr;
        mag_phase_kernel_ = nullptr;
        kernels_compiled_ = false;
    }

    current_buffer_beams_ = 0;
    plan_batch_size_ = 0;
    has_mag_phase_buffers_ = false;
}

// =========================================================================
// PART 4: GPU Operations
// =========================================================================

void FFTProcessorROCm::UploadData(const std::complex<float>* data, size_t count) {
    size_t data_size = count * sizeof(std::complex<float>);

    hipError_t err = hipMemcpyHtoDAsync(input_buffer_,
                                         const_cast<std::complex<float>*>(data),
                                         data_size, stream_);
    if (err != hipSuccess) {
        throw std::runtime_error("UploadData: hipMemcpyHtoDAsync failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

void FFTProcessorROCm::CopyGpuData(void* src, size_t src_offset_bytes, size_t count) {
    size_t data_size = count * sizeof(std::complex<float>);
    char* src_ptr = static_cast<char*>(src) + src_offset_bytes;

    hipError_t err = hipMemcpyDtoDAsync(input_buffer_, src_ptr, data_size, stream_);
    if (err != hipSuccess) {
        throw std::runtime_error("CopyGpuData: hipMemcpyDtoDAsync failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

void FFTProcessorROCm::ExecutePadKernel(size_t beam_count) {
    unsigned int np   = n_point_;
    unsigned int nfft = nFFT_;

    // Обнуляем весь padded-буфер перед запуском kernel (асинхронно на stream_).
    // ЗАЧЕМ: kernel копирует только n_point точек (idx < n_point), остаток [n_point..nFFT-1]
    // уже нулевой после hipMemsetAsync — нет else-ветки в kernel, нет warp divergence.
    hipError_t err = hipMemsetAsync(fft_input_,
                                     0,
                                     beam_count * nFFT_ * sizeof(std::complex<float>),
                                     stream_);
    if (err != hipSuccess) {
        throw std::runtime_error("ExecutePadKernel: hipMemsetAsync failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // 2D grid: X покрывает n_point точек на луч, Y = beam_id.
    // blockIdx.y = beam_id — избегаем деления и остатка для вычисления номера луча.
    // grid_x = ceil(n_point/256) — обрабатываем только реальные точки, нули уже в буфере.
    unsigned int block_x = 256;
    unsigned int grid_x  = static_cast<unsigned int>((n_point_ + block_x - 1) / block_x);
    unsigned int grid_y  = static_cast<unsigned int>(beam_count);

    void* args[] = {
        &input_buffer_,
        &fft_input_,
        &np,
        &nfft
    };

    err = hipModuleLaunchKernel(
        pad_kernel_,
        grid_x, grid_y, 1,     // grid: X covers n_point, Y = beam index
        block_x, 1, 1,          // block
        0,                       // shared memory
        stream_,                 // stream
        args,                    // kernel arguments
        nullptr);                // extra

    if (err != hipSuccess) {
        throw std::runtime_error("ExecutePadKernel: hipModuleLaunchKernel failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

void FFTProcessorROCm::ExecuteFFT() {
    hipfftResult result = hipfftExecC2C(
        plan_,
        static_cast<hipfftComplex*>(fft_input_),
        static_cast<hipfftComplex*>(fft_output_),
        HIPFFT_FORWARD);

    if (result != HIPFFT_SUCCESS) {
        throw std::runtime_error("ExecuteFFT: hipfftExecC2C failed: " +
                                  std::to_string(static_cast<int>(result)));
    }
}

void FFTProcessorROCm::ExecuteMagPhaseKernel(size_t beam_count) {
    // TASK-4A: single interleaved output buffer {mag, phase} per element
    unsigned int total_u = static_cast<unsigned int>(beam_count * nFFT_);

    unsigned int block_size = 256;
    unsigned int grid_size  = (total_u + block_size - 1) / block_size;

    void* args[] = {
        &fft_output_,
        &mag_phase_interleaved_,
        &total_u
    };

    hipError_t err = hipModuleLaunchKernel(
        mag_phase_kernel_,
        grid_size, 1, 1,
        block_size, 1, 1,
        0,
        stream_,
        args,
        nullptr);

    if (err != hipSuccess) {
        throw std::runtime_error("ExecuteMagPhaseKernel: hipModuleLaunchKernel failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

std::vector<FFTComplexResult> FFTProcessorROCm::ReadComplexResults(
    size_t beam_count, size_t start_beam, float sample_rate)
{
    size_t total = beam_count * nFFT_;
    std::vector<std::complex<float>> raw(total);

    hipError_t err = hipMemcpyDtoH(raw.data(), fft_output_,
                                     total * sizeof(std::complex<float>));
    if (err != hipSuccess) {
        throw std::runtime_error("ReadComplexResults: hipMemcpyDtoH failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // Split into per-beam results
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

std::vector<FFTMagPhaseResult> FFTProcessorROCm::ReadMagPhaseResults(
    size_t beam_count, size_t start_beam,
    float sample_rate, bool include_freq)
{
    size_t total = beam_count * nFFT_;

    // TASK-4A: Single DtoH of interleaved {mag, phase} buffer (2 floats per element)
    // Layout: [mag0, phase0, mag1, phase1, ...] — matches float2_t kernel output
    std::vector<float> raw(total * 2);
    hipError_t err = hipMemcpyDtoH(raw.data(), mag_phase_interleaved_,
                                    total * 2 * sizeof(float));
    if (err != hipSuccess) {
        throw std::runtime_error("ReadMagPhaseResults: hipMemcpyDtoH failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // Split interleaved data into per-beam results
    std::vector<FFTMagPhaseResult> results;
    results.reserve(beam_count);

    for (size_t i = 0; i < beam_count; ++i) {
        FFTMagPhaseResult result;
        result.beam_id = static_cast<uint32_t>(start_beam + i);
        result.nFFT = nFFT_;
        result.sample_rate = sample_rate;
        result.magnitude.resize(nFFT_);
        result.phase.resize(nFFT_);

        for (uint32_t k = 0; k < nFFT_; ++k) {
            size_t idx = (i * nFFT_ + k) * 2;
            result.magnitude[k] = raw[idx    ];  // .x = mag
            result.phase[k]     = raw[idx + 1];  // .y = phase
        }

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

// =========================================================================
// PART 5: Utilities
// =========================================================================

uint32_t FFTProcessorROCm::NextPowerOf2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void FFTProcessorROCm::CalculateNFFT(const FFTProcessorParams& params) {
    n_point_ = params.n_point;
    uint32_t base_fft = NextPowerOf2(params.n_point);
    nFFT_ = base_fft * params.repeat_count;
}

size_t FFTProcessorROCm::CalculateBytesPerBeam(FFTOutputMode mode) const {
    // Input data
    size_t input_bytes = n_point_ * sizeof(std::complex<float>);

    // FFT buffers (input_buffer + fft_input + fft_output)
    size_t fft_bytes = nFFT_ * sizeof(std::complex<float>) * 2 + input_bytes;

    // Mag/phase buffers
    size_t post_bytes = 0;
    if (mode != FFTOutputMode::COMPLEX) {
        post_bytes = 2 * nFFT_ * sizeof(float);  // mag + phase
    }

    return input_bytes + fft_bytes + post_bytes;
}

FFTProfilingData FFTProcessorROCm::GetProfilingData() const {
    FFTProfilingData out{};
    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;
    auto stats = drv_gpu_lib::GPUProfiler::GetInstance().GetStats(gpu_id);
    auto it = stats.find("FFTProcessorROCm");
    if (it == stats.end()) return out;

    const auto& mod = it->second;
    auto ev = [&mod](const char* name) -> double {
        auto e = mod.events.find(name);
        return (e != mod.events.end()) ? e->second.GetAvgTimeMs() : 0.0;
    };
    out.upload_time_ms = ev("Upload");
    out.fft_time_ms = ev("FFT");
    out.post_processing_time_ms = ev("PostProcessing");
    out.download_time_ms = ev("Download");
    out.total_time_ms = mod.GetTotalTimeMs();
    return out;
}

}  // namespace fft_processor

#endif  // ENABLE_ROCM
