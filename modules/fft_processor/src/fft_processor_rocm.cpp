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

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <iostream>

namespace fft_processor {

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
}

FFTProcessorROCm::~FFTProcessorROCm() {
    ReleaseResources();
}

FFTProcessorROCm::FFTProcessorROCm(FFTProcessorROCm&& other) noexcept
    : backend_(other.backend_)
    , stream_(other.stream_)
    , plan_(other.plan_)
    , plan_created_(other.plan_created_)
    , input_buffer_(other.input_buffer_)
    , fft_input_(other.fft_input_)
    , fft_output_(other.fft_output_)
    , mag_output_(other.mag_output_)
    , phase_output_(other.phase_output_)
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
    other.input_buffer_ = nullptr;
    other.fft_input_ = nullptr;
    other.fft_output_ = nullptr;
    other.mag_output_ = nullptr;
    other.phase_output_ = nullptr;
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
        input_buffer_ = other.input_buffer_;
        fft_input_ = other.fft_input_;
        fft_output_ = other.fft_output_;
        mag_output_ = other.mag_output_;
        phase_output_ = other.phase_output_;
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
        other.input_buffer_ = nullptr;
        other.fft_input_ = nullptr;
        other.fft_output_ = nullptr;
        other.mag_output_ = nullptr;
        other.phase_output_ = nullptr;
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
    const FFTProcessorParams& params)
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

        // Upload batch data
        const auto* batch_data = data.data() + batch.start * params.n_point;
        UploadData(batch_data, batch.count * params.n_point);

        // Pad: input_buffer_ -> fft_input_
        ExecutePadKernel(batch.count);

        // FFT: fft_input_ -> fft_output_
        ExecuteFFT();

        // Synchronize
        hipStreamSynchronize(stream_);

        // Read results
        auto batch_results = ReadComplexResults(batch.count, batch.start, params.sample_rate);
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
    const FFTProcessorParams& params)
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

        // Upload
        const auto* batch_data = data.data() + batch.start * params.n_point;
        UploadData(batch_data, batch.count * params.n_point);

        // Pad -> FFT -> MagPhase
        ExecutePadKernel(batch.count);
        ExecuteFFT();
        ExecuteMagPhaseKernel(batch.count);
        hipStreamSynchronize(stream_);

        // Read
        auto batch_results = ReadMagPhaseResults(batch.count, batch.start,
                                                  params.sample_rate, include_freq);
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
    if (input_buffer_)  { (void)hipFree(input_buffer_);  input_buffer_ = nullptr; }
    if (fft_input_)     { (void)hipFree(fft_input_);     fft_input_ = nullptr; }
    if (fft_output_)    { (void)hipFree(fft_output_);    fft_output_ = nullptr; }
    if (mag_output_)    { (void)hipFree(mag_output_);    mag_output_ = nullptr; }
    if (phase_output_)  { (void)hipFree(phase_output_);  phase_output_ = nullptr; }
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

    // 3. Mag/phase buffers (if needed)
    if (mode != FFTOutputMode::COMPLEX) {
        size_t scalar_size = batch_beam_count * nFFT_ * sizeof(float);

        err = hipMalloc(&mag_output_, scalar_size);
        if (err != hipSuccess) {
            throw std::runtime_error("AllocateBuffers: mag_output hipMalloc failed: " +
                                      std::string(hipGetErrorString(err)));
        }

        err = hipMalloc(&phase_output_, scalar_size);
        if (err != hipSuccess) {
            throw std::runtime_error("AllocateBuffers: phase_output hipMalloc failed: " +
                                      std::string(hipGetErrorString(err)));
        }

        has_mag_phase_buffers_ = true;
    }

    current_buffer_beams_ = batch_beam_count;
}

void FFTProcessorROCm::CreateFFTPlan(size_t batch_beam_count) {
    if (plan_created_ && plan_batch_size_ == batch_beam_count) {
        return;  // Plan already created for this batch size
    }

    // Destroy old plan
    if (plan_created_) {
        hipfftDestroy(plan_);
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

    // Set stream
    result = hipfftSetStream(plan_, stream_);
    if (result != HIPFFT_SUCCESS) {
        hipfftDestroy(plan_);
        throw std::runtime_error("CreateFFTPlan: hipfftSetStream failed: " +
                                  std::to_string(static_cast<int>(result)));
    }

    plan_batch_size_ = batch_beam_count;
    plan_created_ = true;
}

void FFTProcessorROCm::CompileKernels() {
    if (kernels_compiled_) return;

    const char* source = kernels::GetHIPKernelSource();

    // Create hiprtc program
    hiprtcProgram prog;
    hiprtcResult rtcResult = hiprtcCreateProgram(&prog, source, "fft_kernels.hip",
                                                  0, nullptr, nullptr);
    if (rtcResult != HIPRTC_SUCCESS) {
        throw std::runtime_error("CompileKernels: hiprtcCreateProgram failed: " +
                                  std::to_string(static_cast<int>(rtcResult)));
    }

    // Compile
    rtcResult = hiprtcCompileProgram(prog, 0, nullptr);
    if (rtcResult != HIPRTC_SUCCESS) {
        // Get build log
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        (void)hiprtcDestroyProgram(&prog);
        throw std::runtime_error("CompileKernels: compilation failed:\n" + log);
    }

    // Get compiled code
    size_t codeSize = 0;
    hiprtcGetCodeSize(prog, &codeSize);
    std::vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());
    (void)hiprtcDestroyProgram(&prog);

    // Load module
    hipError_t hipErr = hipModuleLoadData(&module_, code.data());
    if (hipErr != hipSuccess) {
        throw std::runtime_error("CompileKernels: hipModuleLoadData failed: " +
                                  std::string(hipGetErrorString(hipErr)));
    }

    // Get kernel functions
    hipErr = hipModuleGetFunction(&pad_kernel_, module_, "pad_data");
    if (hipErr != hipSuccess) {
        (void)hipModuleUnload(module_);
        module_ = nullptr;
        throw std::runtime_error("CompileKernels: hipModuleGetFunction(pad_data) failed: " +
                                  std::string(hipGetErrorString(hipErr)));
    }

    hipErr = hipModuleGetFunction(&mag_phase_kernel_, module_, "complex_to_mag_phase");
    if (hipErr != hipSuccess) {
        (void)hipModuleUnload(module_);
        module_ = nullptr;
        pad_kernel_ = nullptr;
        throw std::runtime_error("CompileKernels: hipModuleGetFunction(complex_to_mag_phase) failed: " +
                                  std::string(hipGetErrorString(hipErr)));
    }

    kernels_compiled_ = true;

    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;
    DRVGPU_LOG_INFO_GPU(gpu_id, "FFTProcessorROCm", "HIP kernels compiled successfully");
}

void FFTProcessorROCm::ReleaseResources() {
    // Destroy hipFFT plan
    if (plan_created_) {
        hipfftDestroy(plan_);
        plan_ = 0;
        plan_created_ = false;
    }

    // Free GPU buffers
    if (input_buffer_)  { (void)hipFree(input_buffer_);  input_buffer_ = nullptr; }
    if (fft_input_)     { (void)hipFree(fft_input_);     fft_input_ = nullptr; }
    if (fft_output_)    { (void)hipFree(fft_output_);    fft_output_ = nullptr; }
    if (mag_output_)    { (void)hipFree(mag_output_);    mag_output_ = nullptr; }
    if (phase_output_)  { (void)hipFree(phase_output_);  phase_output_ = nullptr; }

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
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = n_point_;
    unsigned int nfft = nFFT_;

    size_t total = beam_count * nFFT_;
    unsigned int block_size = 256;
    unsigned int grid_size = static_cast<unsigned int>((total + block_size - 1) / block_size);

    void* args[] = {
        &input_buffer_,
        &fft_input_,
        &bc,
        &np,
        &nfft
    };

    hipError_t err = hipModuleLaunchKernel(
        pad_kernel_,
        grid_size, 1, 1,       // grid dimensions
        block_size, 1, 1,      // block dimensions
        0,                      // shared memory
        stream_,                // stream
        args,                   // kernel arguments
        nullptr);               // extra

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
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int nfft = nFFT_;

    size_t total = beam_count * nFFT_;
    unsigned int block_size = 256;
    unsigned int grid_size = static_cast<unsigned int>((total + block_size - 1) / block_size);

    void* args[] = {
        &fft_output_,
        &mag_output_,
        &phase_output_,
        &bc,
        &nfft
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

    std::vector<float> raw_mag(total);
    std::vector<float> raw_phase(total);

    // Read mag and phase
    hipError_t err = hipMemcpyDtoH(raw_mag.data(), mag_output_,
                                     total * sizeof(float));
    if (err != hipSuccess) {
        throw std::runtime_error("ReadMagPhaseResults: mag hipMemcpyDtoH failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    err = hipMemcpyDtoH(raw_phase.data(), phase_output_,
                          total * sizeof(float));
    if (err != hipSuccess) {
        throw std::runtime_error("ReadMagPhaseResults: phase hipMemcpyDtoH failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // Split into per-beam results
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

        // Calculate frequencies on CPU
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
