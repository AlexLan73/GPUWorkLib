/**
 * @file spectrum_processor_rocm.cpp
 * @brief ROCm/HIP implementation of ISpectrumProcessor (hipFFT + hiprtc kernels)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * CONTENTS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * PART 1: Constructor, Destructor
 * PART 2: Initialize
 * PART 3: Process (ProcessFromCPU, ProcessFromGPU, ProcessBatch, ProcessBatchFromGPU)
 * PART 4: FindAllMaxima pipeline (FindAllMaximaFromCPU, FromGPUPipeline, AllMaximaFromCPU, FindAllMaxima)
 * PART 5: GPU Resources (AllocateBuffers, CreateFFTPlan, CompileKernels, CompilePostKernel)
 * PART 6: GPU Operations (UploadData, CopyGpuData, ExecutePadKernel, ExecuteFFT, ExecutePostKernel, etc.)
 * PART 7: Utilities (CalculateFFTSize, ReallocateBuffersForBatch, ReleaseResources, etc.)
 *
 * Key differences from OpenCL version:
 * - No pre-callback → separate pad_data kernel before hipFFT
 * - No post-callback → separate compute_magnitudes kernel after hipFFT
 * - Stream-ordered execution (no explicit events)
 * - void* device pointers instead of cl_mem
 * - hiprtc for runtime kernel compilation
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "processors/spectrum_processor_rocm.hpp"
#include "kernels/fft_kernel_sources_rocm.hpp"
#include "kernels/all_maxima_kernel_sources_rocm.hpp"
#include "services/console_output.hpp"
#include "services/gpu_profiler.hpp"
#include "services/batch_manager.hpp"
#include "interface/i_backend.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// PART 1: Constructor, Destructor
// ════════════════════════════════════════════════════════════════════════════

SpectrumProcessorROCm::SpectrumProcessorROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

    if (!backend_) {
        throw std::invalid_argument("SpectrumProcessorROCm: backend cannot be null");
    }

    if (!backend_->IsInitialized()) {
        throw std::runtime_error("SpectrumProcessorROCm: backend is not initialized");
    }

    stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
    if (!stream_) {
        throw std::runtime_error("SpectrumProcessorROCm: failed to get HIP stream from backend");
    }

    pipeline_ = std::make_unique<AllMaximaPipelineROCm>(stream_, backend_);
}

SpectrumProcessorROCm::~SpectrumProcessorROCm() {
    ReleaseResources();
}

// ════════════════════════════════════════════════════════════════════════════
// PART 2: Initialize
// ════════════════════════════════════════════════════════════════════════════

void SpectrumProcessorROCm::Initialize(const SpectrumParams& params) {
    params_ = params;
    if (initialized_) return;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "SpectrumMaxima[ROCm]", "Initializing...");

    CalculateFFTSize();

    con.Print(0, "SpectrumMaxima[ROCm]",
        "antenna_count=" + std::to_string(params_.antenna_count) +
        " n_point=" + std::to_string(params_.n_point) +
        " repeat_count=" + std::to_string(params_.repeat_count) +
        " nFFT=" + std::to_string(params_.nFFT) +
        " search_range=" + std::to_string(params_.search_range) +
        " sample_rate=" + std::to_string(static_cast<int>(params_.sample_rate)) + " Hz");

    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    bool need_batch = !drv_gpu_lib::BatchManager::AllItemsFit(
        backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

    if (need_batch) {
        size_t max_batch_size = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
            backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

        con.Print(0, "SpectrumMaxima[ROCm]", "[Batch mode] max batch=" + std::to_string(max_batch_size));
        ReallocateBuffersForBatch(max_batch_size);
        CompilePostKernel();
    } else {
        AllocateBuffers();
        CreateFFTPlan(params_.antenna_count);
        CompileKernels();
        CompilePostKernel();

        current_batch_size_ = params_.antenna_count;
        actual_batch_size_ = params_.antenna_count;
    }

    initialized_ = true;
    con.Print(0, "SpectrumMaxima[ROCm]", "Initialization complete");
}

// ════════════════════════════════════════════════════════════════════════════
// PART 3: Process (ONE_PEAK / TWO_PEAKS mode)
// ════════════════════════════════════════════════════════════════════════════

std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatch(
    const std::vector<std::complex<float>>& input_data,
    size_t start_antenna,
    size_t batch_antenna_count)
{
    if (batch_antenna_count > current_batch_size_ || !plan_created_) {
        ReallocateBuffersForBatch(batch_antenna_count);
    }
    actual_batch_size_ = batch_antenna_count;

    // Extract batch slice
    size_t offset = start_antenna * params_.n_point;
    size_t count = batch_antenna_count * params_.n_point;

    // Upload → Pad → FFT → PostKernel → Read
    UploadData(input_data.data() + offset, count);
    ExecutePadKernel(batch_antenna_count);
    ExecuteFFT();
    ExecutePostKernel(batch_antenna_count);
    hipStreamSynchronize(stream_);

    auto results = ReadResults(batch_antenna_count);

    for (auto& result : results) {
        result.antenna_id += static_cast<uint32_t>(start_antenna);
    }

    return results;
}

std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data)
{
    if (!initialized_) {
        throw std::runtime_error("SpectrumProcessorROCm::ProcessFromCPU: not initialized");
    }

    size_t expected_size = params_.antenna_count * params_.n_point;
    if (data.size() != expected_size) {
        throw std::invalid_argument(
            "SpectrumProcessorROCm::ProcessFromCPU: input size mismatch. "
            "Expected " + std::to_string(expected_size) +
            ", got " + std::to_string(data.size()));
    }

    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    if (drv_gpu_lib::BatchManager::AllItemsFit(backend_, params_.antenna_count,
                                                bytes_per_antenna, params_.memory_limit)) {
        return ProcessBatch(data, 0, params_.antenna_count);
    }

    // Batch processing
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

    size_t batch_size = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params_.antenna_count, batch_size, 3, true);

    con.Print(0, "SpectrumMaxima[ROCm]", "Batch Processing: " +
        std::to_string(batches.size()) + " batches");

    std::vector<SpectrumResult> all_results;
    all_results.reserve(params_.antenna_count);

    for (const auto& batch : batches) {
        auto batch_results = ProcessBatch(data, batch.start, batch.count);
        all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
    }

    return all_results;
}

std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessFromGPU(
    void* gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes)
{
    if (!gpu_data) {
        throw std::invalid_argument("SpectrumProcessorROCm::ProcessFromGPU: gpu_data cannot be null");
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "SpectrumMaxima[ROCm]", "ProcessFromGPU: " +
        std::to_string(antenna_count) + " antennas, n_point=" + std::to_string(n_point));

    if (params_.nFFT == 0) {
        CalculateFFTSize();
    }
    if (!kernels_compiled_) {
        CompileKernels();
    }
    if (!post_kernel_) {
        CompilePostKernel();
    }

    size_t bytes_per_antenna = CalculateBytesPerAntenna();
    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : antenna_count * n_point * sizeof(std::complex<float>);

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, antenna_count, bytes_per_antenna,
        params_.memory_limit, external_memory);

    bool need_batch = (optimal_batch < antenna_count);

    if (!need_batch) {
        ReallocateBuffersForBatch(antenna_count);
        actual_batch_size_ = antenna_count;

        // D2D copy → Pad → FFT → PostKernel → Read
        CopyGpuData(gpu_data, 0, antenna_count * n_point);
        ExecutePadKernel(antenna_count);
        ExecuteFFT();
        ExecutePostKernel(antenna_count);
        hipStreamSynchronize(stream_);

        return ReadResults(antenna_count);
    }

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(antenna_count, optimal_batch);
    ReallocateBuffersForBatch(optimal_batch);

    std::vector<SpectrumResult> all_results;
    all_results.reserve(antenna_count);

    for (const auto& batch : batches) {
        auto batch_results = ProcessBatchFromGPU(gpu_data,
            batch.start * n_point * sizeof(std::complex<float>),
            batch.start, batch.count);
        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatchFromGPU(
    void* gpu_data, size_t src_offset_bytes,
    size_t start_antenna, size_t batch_antenna_count)
{
    if (batch_antenna_count > current_batch_size_ || !plan_created_) {
        ReallocateBuffersForBatch(batch_antenna_count);
    }
    actual_batch_size_ = batch_antenna_count;

    CopyGpuData(gpu_data, src_offset_bytes, batch_antenna_count * params_.n_point);
    ExecutePadKernel(batch_antenna_count);
    ExecuteFFT();
    ExecutePostKernel(batch_antenna_count);
    hipStreamSynchronize(stream_);

    auto results = ReadResults(batch_antenna_count);

    for (size_t i = 0; i < results.size(); ++i) {
        results[i].antenna_id = static_cast<uint32_t>(start_antenna + i);
    }

    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 4: FindAllMaxima pipeline
// ════════════════════════════════════════════════════════════════════════════

AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>& data,
    OutputDestination dest, uint32_t search_start, uint32_t search_end)
{
    if (!initialized_) {
        throw std::runtime_error("SpectrumProcessorROCm::FindAllMaximaFromCPU: not initialized");
    }

    size_t expected_size = static_cast<size_t>(params_.antenna_count) * params_.n_point;
    if (data.size() != expected_size) {
        throw std::invalid_argument(
            "FindAllMaximaFromCPU: input size mismatch. Expected " +
            std::to_string(expected_size) + ", got " + std::to_string(data.size()));
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima[ROCm]", "FindAllMaximaFromCPU: " +
        std::to_string(params_.antenna_count) + " beams, nFFT=" +
        std::to_string(params_.nFFT));

    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    if (!drv_gpu_lib::BatchManager::AllItemsFit(backend_, params_.antenna_count,
                                                  bytes_per_antenna, params_.memory_limit)) {
        throw std::runtime_error(
            "FindAllMaximaFromCPU: data doesn't fit in GPU memory");
    }

    ReallocateBuffersForBatch(params_.antenna_count);
    actual_batch_size_ = params_.antenna_count;
    CreateAllMaximaFFTPlan(params_.antenna_count);

    if (!kernels_compiled_) CompileKernels();

    // Upload → Pad → FFT → ComputeMagnitudes → pipeline
    UploadData(data.data(), data.size());
    ExecutePadKernel(params_.antenna_count);
    ExecuteFFT();

    size_t total_elements = static_cast<size_t>(params_.antenna_count) * params_.nFFT;
    ExecuteComputeMagnitudes(total_elements);
    hipStreamSynchronize(stream_);

    return pipeline_->Execute(
        magnitudes_buffer_, fft_output_,
        params_.antenna_count, params_.nFFT, params_.sample_rate,
        dest, search_start, search_end);
}

AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromGPUPipeline(
    void* gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes,
    OutputDestination dest, uint32_t search_start, uint32_t search_end)
{
    if (!gpu_data) {
        throw std::invalid_argument("FindAllMaximaFromGPUPipeline: gpu_data cannot be null");
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima[ROCm]", "FindAllMaximaFromGPUPipeline: " +
        std::to_string(antenna_count) + " beams, n_point=" + std::to_string(n_point));

    if (params_.nFFT == 0) CalculateFFTSize();
    if (!kernels_compiled_) CompileKernels();

    size_t bytes_per_antenna = CalculateBytesPerAntenna();
    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : antenna_count * n_point * sizeof(std::complex<float>);

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, antenna_count, bytes_per_antenna,
        params_.memory_limit, external_memory);

    if (optimal_batch < antenna_count) {
        throw std::runtime_error("FindAllMaximaFromGPUPipeline: data doesn't fit in GPU memory");
    }

    ReallocateBuffersForBatch(antenna_count);
    actual_batch_size_ = antenna_count;
    CreateAllMaximaFFTPlan(antenna_count);

    // D2D copy → Pad → FFT → ComputeMagnitudes → pipeline
    CopyGpuData(gpu_data, 0, antenna_count * n_point);
    ExecutePadKernel(antenna_count);
    ExecuteFFT();

    size_t total_elements = antenna_count * params_.nFFT;
    ExecuteComputeMagnitudes(total_elements);
    hipStreamSynchronize(stream_);

    return pipeline_->Execute(
        magnitudes_buffer_, fft_output_,
        static_cast<uint32_t>(antenna_count), params_.nFFT, params_.sample_rate,
        dest, search_start, search_end);
}

AllMaximaResult SpectrumProcessorROCm::AllMaximaFromCPU(
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
    con.Print(0, "AllMaxima[ROCm]", "AllMaximaFromCPU: " +
        std::to_string(beam_count) + " beams, nFFT=" + std::to_string(nFFT));

    if (!kernels_compiled_) CompileKernels();

    // Upload FFT data to GPU
    size_t data_bytes = expected_size * sizeof(std::complex<float>);
    void* gpu_fft = nullptr;

    hipError_t err = hipMalloc(&gpu_fft, data_bytes);
    if (err != hipSuccess)
        throw std::runtime_error("AllMaximaFromCPU: hipMalloc failed");

    err = hipMemcpyHtoDAsync(gpu_fft, const_cast<std::complex<float>*>(fft_data.data()),
                              data_bytes, stream_);
    if (err != hipSuccess) {
        (void)hipFree(gpu_fft);
        throw std::runtime_error("AllMaximaFromCPU: hipMemcpyHtoDAsync failed");
    }

    hipStreamSynchronize(stream_);

    AllMaximaResult result = FindAllMaxima(gpu_fft, beam_count, nFFT,
                                            sample_rate, dest, search_start, search_end);

    (void)hipFree(gpu_fft);
    return result;
}

AllMaximaResult SpectrumProcessorROCm::FindAllMaxima(
    void* fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest, uint32_t search_start, uint32_t search_end)
{
    if (!fft_data)
        throw std::invalid_argument("FindAllMaxima: fft_data cannot be null");

    if (search_start == 0) search_start = 1;
    if (search_end == 0) search_end = nFFT / 2;
    if (search_end >= nFFT) search_end = nFFT - 1;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima[ROCm]", "FindAllMaxima: beam_count=" + std::to_string(beam_count)
        + " nFFT=" + std::to_string(nFFT)
        + " range=[" + std::to_string(search_start) + "," + std::to_string(search_end) + ")"
        + " Fs=" + std::to_string(static_cast<int>(sample_rate)) + "Hz");

    if (!kernels_compiled_) CompileKernels();

    const size_t total_elements = static_cast<size_t>(beam_count) * nFFT;
    EnsureMagnitudesBuffer(total_elements);

    // Compute magnitudes: fft_data → magnitudes_buffer_
    ExecuteComputeMagnitudes(total_elements);
    hipStreamSynchronize(stream_);

    // We need to pass fft_data to compute_magnitudes, but the function uses fft_output_
    // Actually, we need a local magnitude compute since fft_data might not be fft_output_.
    // Let me fix: compute magnitudes from arbitrary fft_data.
    // We'll launch the kernel with fft_data pointer explicitly.
    {
        unsigned int total_size = static_cast<unsigned int>(total_elements);

        void* args[] = { &fft_data, &magnitudes_buffer_, &total_size };

        unsigned int grid_x = static_cast<unsigned int>((total_elements + 255) / 256);
        unsigned int block_x = 256;

        hipError_t hip_err = hipModuleLaunchKernel(
            compute_mag_kernel_,
            grid_x, 1, 1,
            block_x, 1, 1,
            0, stream_,
            args, nullptr);
        if (hip_err != hipSuccess) {
            throw std::runtime_error("FindAllMaxima: compute_magnitudes launch failed: " +
                                      std::string(hipGetErrorString(hip_err)));
        }
        hipStreamSynchronize(stream_);
    }

    return pipeline_->Execute(
        magnitudes_buffer_, fft_data,
        beam_count, nFFT, sample_rate,
        dest, search_start, search_end);
}

// ════════════════════════════════════════════════════════════════════════════
// PART 5: GPU Resources
// ════════════════════════════════════════════════════════════════════════════

void SpectrumProcessorROCm::AllocateBuffers() {
    hipError_t err;

    // 1. Input buffer (raw data, not padded)
    size_t input_size = params_.antenna_count * params_.n_point * sizeof(std::complex<float>);
    err = hipMalloc(&input_buffer_, input_size);
    if (err != hipSuccess)
        throw std::runtime_error("AllocateBuffers: input_buffer failed: " +
                                  std::string(hipGetErrorString(err)));

    // 2. FFT buffers (padded)
    size_t fft_size = params_.antenna_count * params_.nFFT * sizeof(std::complex<float>);

    err = hipMalloc(&fft_input_, fft_size);
    if (err != hipSuccess)
        throw std::runtime_error("AllocateBuffers: fft_input failed");

    err = hipMalloc(&fft_output_, fft_size);
    if (err != hipSuccess)
        throw std::runtime_error("AllocateBuffers: fft_output failed");

    // 3. Maxima output (ONE_PEAK=4, TWO_PEAKS=8 per beam)
    size_t max_values_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
    size_t maxima_size = params_.antenna_count * max_values_per_beam * sizeof(MaxValue);

    err = hipMalloc(&maxima_output_, maxima_size);
    if (err != hipSuccess)
        throw std::runtime_error("AllocateBuffers: maxima_output failed");
}

void SpectrumProcessorROCm::CreateFFTPlan(size_t batch_count) {
    if (plan_created_ && plan_batch_size_ == batch_count) return;

    if (plan_created_) {
        hipfftDestroy(plan_);
        plan_ = 0;
        plan_created_ = false;
    }

    hipfftResult result = hipfftPlan1d(&plan_,
        static_cast<int>(params_.nFFT),
        HIPFFT_C2C,
        static_cast<int>(batch_count));
    if (result != HIPFFT_SUCCESS) {
        throw std::runtime_error("CreateFFTPlan: hipfftPlan1d failed: " +
                                  std::to_string(static_cast<int>(result)));
    }

    result = hipfftSetStream(plan_, stream_);
    if (result != HIPFFT_SUCCESS) {
        hipfftDestroy(plan_);
        throw std::runtime_error("CreateFFTPlan: hipfftSetStream failed");
    }

    plan_batch_size_ = batch_count;
    plan_created_ = true;
}

void SpectrumProcessorROCm::CreateAllMaximaFFTPlan(size_t batch_count) {
    if (allmax_plan_created_ && allmax_plan_batch_size_ == batch_count) return;

    if (allmax_plan_created_) {
        hipfftDestroy(allmax_plan_);
        allmax_plan_ = 0;
        allmax_plan_created_ = false;
    }

    size_t total_fft_elements = batch_count * params_.nFFT;
    EnsureMagnitudesBuffer(total_fft_elements);

    hipfftResult result = hipfftPlan1d(&allmax_plan_,
        static_cast<int>(params_.nFFT),
        HIPFFT_C2C,
        static_cast<int>(batch_count));
    if (result != HIPFFT_SUCCESS) {
        throw std::runtime_error("CreateAllMaximaFFTPlan: hipfftPlan1d failed");
    }

    result = hipfftSetStream(allmax_plan_, stream_);
    if (result != HIPFFT_SUCCESS) {
        hipfftDestroy(allmax_plan_);
        throw std::runtime_error("CreateAllMaximaFFTPlan: hipfftSetStream failed");
    }

    allmax_plan_batch_size_ = batch_count;
    allmax_plan_created_ = true;
}

void SpectrumProcessorROCm::CompileKernels() {
    if (kernels_compiled_) return;

    const char* src = kernels::GetSpectrumHIPKernelSource();

    hiprtcProgram prog;
    hiprtcResult rtc_err = hiprtcCreateProgram(&prog, src, "spectrum_kernels.hip",
                                                0, nullptr, nullptr);
    if (rtc_err != HIPRTC_SUCCESS) {
        throw std::runtime_error("CompileKernels: hiprtcCreateProgram failed: " +
                                  std::string(hiprtcGetErrorString(rtc_err)));
    }

    const char* options[] = { "-O3" };
    rtc_err = hiprtcCompileProgram(prog, 1, options);
    if (rtc_err != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        hiprtcGetProgramLog(prog, log.data());

        auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
        con.PrintError(0, "SpectrumMaxima[ROCm]", "Kernel compile log:\n" + log);

        (void)hiprtcDestroyProgram(&prog);
        throw std::runtime_error("CompileKernels: hiprtcCompileProgram failed");
    }

    size_t code_size = 0;
    hiprtcGetCodeSize(prog, &code_size);
    std::vector<char> code(code_size);
    hiprtcGetCode(prog, code.data());
    (void)hiprtcDestroyProgram(&prog);

    hipError_t hip_err = hipModuleLoadData(&module_, code.data());
    if (hip_err != hipSuccess) {
        throw std::runtime_error("CompileKernels: hipModuleLoadData failed: " +
                                  std::string(hipGetErrorString(hip_err)));
    }

    hip_err = hipModuleGetFunction(&pad_kernel_, module_, "pad_data");
    if (hip_err != hipSuccess)
        throw std::runtime_error("CompileKernels: get pad_data failed");

    hip_err = hipModuleGetFunction(&compute_mag_kernel_, module_, "compute_magnitudes");
    if (hip_err != hipSuccess)
        throw std::runtime_error("CompileKernels: get compute_magnitudes failed");

    kernels_compiled_ = true;

    drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "SpectrumMaxima[ROCm]",
        "HIP kernels compiled (pad + compute_magnitudes)");
}

void SpectrumProcessorROCm::CompilePostKernel() {
    if (post_kernel_) return;

    if (!kernels_compiled_) CompileKernels();

    const char* kernel_name;
    if (params_.peak_mode == PeakSearchMode::ONE_PEAK) {
        kernel_name = "post_kernel_one_peak";
        drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "SpectrumMaxima[ROCm]",
            "Peak mode: ONE_PEAK (4 MaxValue/beam)");
    } else {
        kernel_name = "post_kernel_two_peaks";
        drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "SpectrumMaxima[ROCm]",
            "Peak mode: TWO_PEAKS (8 MaxValue/beam)");
    }

    hipError_t hip_err = hipModuleGetFunction(&post_kernel_, module_, kernel_name);
    if (hip_err != hipSuccess) {
        throw std::runtime_error("CompilePostKernel: hipModuleGetFunction(" +
                                  std::string(kernel_name) + ") failed: " +
                                  std::string(hipGetErrorString(hip_err)));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PART 6: GPU Operations
// ════════════════════════════════════════════════════════════════════════════

void SpectrumProcessorROCm::UploadData(const std::complex<float>* data, size_t element_count) {
    size_t data_size = element_count * sizeof(std::complex<float>);

    hipError_t err = hipMemcpyHtoDAsync(input_buffer_,
                                         const_cast<std::complex<float>*>(data),
                                         data_size, stream_);
    if (err != hipSuccess) {
        throw std::runtime_error("UploadData: hipMemcpyHtoDAsync failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

void SpectrumProcessorROCm::CopyGpuData(void* src, size_t src_offset_bytes, size_t element_count) {
    size_t data_size = element_count * sizeof(std::complex<float>);
    char* src_ptr = static_cast<char*>(src) + src_offset_bytes;

    hipError_t err = hipMemcpyDtoDAsync(input_buffer_, src_ptr, data_size, stream_);
    if (err != hipSuccess) {
        throw std::runtime_error("CopyGpuData: hipMemcpyDtoDAsync failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

void SpectrumProcessorROCm::ExecutePadKernel(size_t beam_count, size_t beam_offset) {
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = params_.n_point;
    unsigned int nfft = params_.nFFT;
    unsigned int bo = static_cast<unsigned int>(beam_offset);

    size_t total = beam_count * params_.nFFT;
    unsigned int grid_x = static_cast<unsigned int>((total + 255) / 256);
    unsigned int block_x = 256;

    void* args[] = {
        &input_buffer_, &fft_input_,
        &bc, &np, &nfft, &bo
    };

    hipError_t err = hipModuleLaunchKernel(
        pad_kernel_,
        grid_x, 1, 1,
        block_x, 1, 1,
        0, stream_,
        args, nullptr);
    if (err != hipSuccess) {
        throw std::runtime_error("ExecutePadKernel: launch failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

void SpectrumProcessorROCm::ExecuteFFT() {
    hipfftHandle active_plan = allmax_plan_created_ ? allmax_plan_ : plan_;

    hipfftResult result = hipfftExecC2C(
        active_plan,
        static_cast<hipfftComplex*>(fft_input_),
        static_cast<hipfftComplex*>(fft_output_),
        HIPFFT_FORWARD);

    if (result != HIPFFT_SUCCESS) {
        throw std::runtime_error("ExecuteFFT: hipfftExecC2C failed: " +
                                  std::to_string(static_cast<int>(result)));
    }
}

void SpectrumProcessorROCm::ExecutePostKernel(size_t beam_count, size_t beam_offset) {
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int nfft = params_.nFFT;
    unsigned int search_range = params_.search_range;
    float sample_rate = params_.sample_rate;

    void* args[] = {
        &fft_output_, &maxima_output_,
        &bc, &nfft, &search_range, &sample_rate
    };

    // One work-group per beam (LOCAL_SIZE threads)
    unsigned int grid_x = static_cast<unsigned int>(beam_count);
    unsigned int block_x = static_cast<unsigned int>(LOCAL_SIZE);

    hipError_t err = hipModuleLaunchKernel(
        post_kernel_,
        grid_x, 1, 1,
        block_x, 1, 1,
        0, stream_,
        args, nullptr);
    if (err != hipSuccess) {
        throw std::runtime_error("ExecutePostKernel: launch failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

void SpectrumProcessorROCm::ExecuteComputeMagnitudes(size_t total_elements) {
    if (!compute_mag_kernel_) {
        if (!kernels_compiled_) CompileKernels();
    }

    unsigned int total_size = static_cast<unsigned int>(total_elements);

    void* args[] = { &fft_output_, &magnitudes_buffer_, &total_size };

    unsigned int grid_x = static_cast<unsigned int>((total_elements + 255) / 256);
    unsigned int block_x = 256;

    hipError_t err = hipModuleLaunchKernel(
        compute_mag_kernel_,
        grid_x, 1, 1,
        block_x, 1, 1,
        0, stream_,
        args, nullptr);
    if (err != hipSuccess) {
        throw std::runtime_error("ExecuteComputeMagnitudes: launch failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

std::vector<SpectrumResult> SpectrumProcessorROCm::ReadResults(size_t beam_count, size_t beam_offset) {
    size_t max_values_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
    size_t num_results = beam_count * max_values_per_beam;
    std::vector<MaxValue> raw_results(num_results);

    hipError_t err = hipMemcpyDtoH(raw_results.data(), maxima_output_,
                                     num_results * sizeof(MaxValue));
    if (err != hipSuccess) {
        throw std::runtime_error("ReadResults: hipMemcpyDtoH failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    std::vector<SpectrumResult> results;

    if (params_.peak_mode == PeakSearchMode::ONE_PEAK) {
        results.reserve(beam_count);
        for (uint32_t i = 0; i < beam_count; ++i) {
            size_t base = i * 4;
            SpectrumResult result{};
            result.antenna_id = static_cast<uint32_t>(beam_offset + i);
            result.interpolated = raw_results[base + 0];
            result.left_point   = raw_results[base + 1];
            result.center_point = raw_results[base + 2];
            result.right_point  = raw_results[base + 3];
            results.push_back(result);
        }
    } else {
        results.reserve(beam_count * 2);
        for (uint32_t i = 0; i < beam_count; ++i) {
            size_t base = i * 8;

            SpectrumResult left{};
            left.antenna_id = static_cast<uint32_t>(beam_offset + i);
            left.interpolated = raw_results[base + 0];
            left.left_point   = raw_results[base + 1];
            left.center_point = raw_results[base + 2];
            left.right_point  = raw_results[base + 3];
            results.push_back(left);

            SpectrumResult right{};
            right.antenna_id = static_cast<uint32_t>(beam_offset + i);
            right.interpolated = raw_results[base + 4];
            right.left_point   = raw_results[base + 5];
            right.center_point = raw_results[base + 6];
            right.right_point  = raw_results[base + 7];
            results.push_back(right);
        }
    }

    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 7: Utilities
// ════════════════════════════════════════════════════════════════════════════

void SpectrumProcessorROCm::CalculateFFTSize() {
    params_.base_fft = NextPowerOf2(params_.n_point);
    params_.nFFT = params_.base_fft * params_.repeat_count;
    if (params_.search_range == 0) {
        params_.search_range = params_.nFFT / 4;
    }
}

uint32_t SpectrumProcessorROCm::NextPowerOf2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

size_t SpectrumProcessorROCm::CalculateBytesPerAntenna() const {
    size_t input_bytes = params_.n_point * sizeof(std::complex<float>);
    size_t fft_bytes = 2 * params_.nFFT * sizeof(std::complex<float>);  // input + output
    size_t mag_bytes = params_.nFFT * sizeof(float);  // magnitudes

    size_t maxima_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
    size_t maxima_bytes = maxima_per_beam * sizeof(MaxValue);

    return input_bytes + fft_bytes + mag_bytes + maxima_bytes;
}

void SpectrumProcessorROCm::EnsureMagnitudesBuffer(size_t total_elements) {
    if (magnitudes_buffer_ && magnitudes_buffer_size_ >= total_elements) return;

    if (magnitudes_buffer_) {
        (void)hipFree(magnitudes_buffer_);
        magnitudes_buffer_ = nullptr;
    }

    hipError_t err = hipMalloc(&magnitudes_buffer_, total_elements * sizeof(float));
    if (err != hipSuccess) {
        throw std::runtime_error("EnsureMagnitudesBuffer: hipMalloc failed: " +
                                  std::string(hipGetErrorString(err)));
    }
    magnitudes_buffer_size_ = total_elements;
}

void SpectrumProcessorROCm::ReallocateBuffersForBatch(size_t batch_antenna_count) {
    bool need_new_plan = (plan_batch_size_ != batch_antenna_count) || !plan_created_;
    bool need_new_buffers = (current_batch_size_ < batch_antenna_count) || !input_buffer_;

    if (!need_new_buffers && !need_new_plan) return;

    if (need_new_buffers) {
        // Free old buffers
        if (input_buffer_)    { (void)hipFree(input_buffer_);    input_buffer_ = nullptr; }
        if (fft_input_)       { (void)hipFree(fft_input_);       fft_input_ = nullptr; }
        if (fft_output_)      { (void)hipFree(fft_output_);      fft_output_ = nullptr; }
        if (maxima_output_)   { (void)hipFree(maxima_output_);   maxima_output_ = nullptr; }

        hipError_t err;

        // Input buffer
        size_t input_size = batch_antenna_count * params_.n_point * sizeof(std::complex<float>);
        err = hipMalloc(&input_buffer_, input_size);
        if (err != hipSuccess)
            throw std::runtime_error("ReallocateBuffersForBatch: input_buffer failed");

        // FFT buffers
        size_t fft_size = batch_antenna_count * params_.nFFT * sizeof(std::complex<float>);
        err = hipMalloc(&fft_input_, fft_size);
        if (err != hipSuccess)
            throw std::runtime_error("ReallocateBuffersForBatch: fft_input failed");

        err = hipMalloc(&fft_output_, fft_size);
        if (err != hipSuccess)
            throw std::runtime_error("ReallocateBuffersForBatch: fft_output failed");

        // Maxima output
        size_t max_values_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
        size_t maxima_size = batch_antenna_count * max_values_per_beam * sizeof(MaxValue);
        err = hipMalloc(&maxima_output_, maxima_size);
        if (err != hipSuccess)
            throw std::runtime_error("ReallocateBuffersForBatch: maxima_output failed");

        current_batch_size_ = batch_antenna_count;
    }

    if (need_new_plan) {
        CreateFFTPlan(batch_antenna_count);
    }

    if (!kernels_compiled_) CompileKernels();
}

void SpectrumProcessorROCm::ReleaseAllMaximaResources() {
    pipeline_.reset();

    if (allmax_plan_created_) {
        hipfftDestroy(allmax_plan_);
        allmax_plan_ = 0;
        allmax_plan_created_ = false;
        allmax_plan_batch_size_ = 0;
    }

    if (magnitudes_buffer_) {
        (void)hipFree(magnitudes_buffer_);
        magnitudes_buffer_ = nullptr;
        magnitudes_buffer_size_ = 0;
    }
}

void SpectrumProcessorROCm::ReleaseResources() {
    ReleaseAllMaximaResources();

    // hipFFT plan (Process mode)
    if (plan_created_) {
        hipfftDestroy(plan_);
        plan_ = 0;
        plan_created_ = false;
        plan_batch_size_ = 0;
    }

    // GPU buffers
    if (input_buffer_)    { (void)hipFree(input_buffer_);    input_buffer_ = nullptr; }
    if (fft_input_)       { (void)hipFree(fft_input_);       fft_input_ = nullptr; }
    if (fft_output_)      { (void)hipFree(fft_output_);      fft_output_ = nullptr; }
    if (maxima_output_)   { (void)hipFree(maxima_output_);   maxima_output_ = nullptr; }

    // hiprtc module
    if (module_) {
        (void)hipModuleUnload(module_);
        module_ = nullptr;
        pad_kernel_ = nullptr;
        compute_mag_kernel_ = nullptr;
        post_kernel_ = nullptr;
        kernels_compiled_ = false;
    }

    current_batch_size_ = 0;
    actual_batch_size_ = 0;
    initialized_ = false;
}

ProfilingData SpectrumProcessorROCm::GetProfilingData() const {
    ProfilingData out{};
    // TODO: Add HIP profiling via hipEvent timing when running on AMD GPU
    return out;
}

}  // namespace antenna_fft

#else  // !ENABLE_ROCM

// ════════════════════════════════════════════════════════════════════════════
// Stub implementation for non-ROCm builds (Windows)
// ════════════════════════════════════════════════════════════════════════════

#include "processors/spectrum_processor_rocm.hpp"
#include <stdexcept>

namespace antenna_fft {

SpectrumProcessorROCm::SpectrumProcessorROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {}

void SpectrumProcessorROCm::Initialize(const SpectrumParams&) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled (compile with ENABLE_ROCM=1)");
}

std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessFromCPU(const std::vector<std::complex<float>>&) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessFromGPU(void*, size_t, size_t, size_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatch(const std::vector<std::complex<float>>&, size_t, size_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatchFromGPU(void*, size_t, size_t, size_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromCPU(const std::vector<std::complex<float>>&, OutputDestination, uint32_t, uint32_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromGPUPipeline(void*, size_t, size_t, size_t, OutputDestination, uint32_t, uint32_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
AllMaximaResult SpectrumProcessorROCm::AllMaximaFromCPU(const std::vector<std::complex<float>>&, uint32_t, uint32_t, float, OutputDestination, uint32_t, uint32_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
AllMaximaResult SpectrumProcessorROCm::FindAllMaxima(void*, uint32_t, uint32_t, float, OutputDestination, uint32_t, uint32_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
ProfilingData SpectrumProcessorROCm::GetProfilingData() const { return {}; }
void SpectrumProcessorROCm::ReallocateBuffersForBatch(size_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
size_t SpectrumProcessorROCm::CalculateBytesPerAntenna() const {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
void SpectrumProcessorROCm::CompilePostKernel() {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}

}  // namespace antenna_fft

#endif  // ENABLE_ROCM
