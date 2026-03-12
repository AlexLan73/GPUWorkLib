/**
 * @file complex_to_mag_phase_rocm.cpp
 * @brief Implementation of ComplexToMagPhaseROCm -- direct complex->mag+phase on GPU
 *
 * =========================================================================
 * CONTENTS
 * =========================================================================
 *
 * PART 1: Constructor, Destructor, Move Semantics
 * PART 2: Public API (Process, ProcessToGPU)
 * PART 3: GPU Resources (Allocate, CompileKernels, Release)
 * PART 4: GPU Operations (Upload, Copy, Execute, Read)
 * PART 5: Utilities
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include "complex_to_mag_phase_rocm.hpp"
#include "kernels/complex_to_mag_phase_kernels_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"
#include "services/kernel_cache_service.hpp"
#include "logger/logger.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>

namespace fft_processor {

// =========================================================================
// PART 1: Constructor / Destructor / Move Semantics
// =========================================================================

ComplexToMagPhaseROCm::ComplexToMagPhaseROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

    if (!backend_ || !backend_->IsInitialized()) {
        throw std::runtime_error("ComplexToMagPhaseROCm: backend is null or not initialized");
    }

    if (backend_->GetType() != drv_gpu_lib::BackendType::ROCm) {
        throw std::runtime_error("ComplexToMagPhaseROCm: requires ROCm backend, got different type");
    }

    stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
    if (!stream_) {
        throw std::runtime_error("ComplexToMagPhaseROCm: failed to get HIP stream from backend");
    }

    // Initialize disk cache for compiled HSACO kernels
    kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
        "modules/fft_processor/kernels", drv_gpu_lib::BackendType::ROCm);
}

ComplexToMagPhaseROCm::~ComplexToMagPhaseROCm() {
    ReleaseResources();
}

ComplexToMagPhaseROCm::ComplexToMagPhaseROCm(ComplexToMagPhaseROCm&& other) noexcept
    : backend_(other.backend_)
    , stream_(other.stream_)
    , module_(other.module_)
    , mag_module_(other.mag_module_)
    , kernel_(other.kernel_)
    , magnitude_kernel_(other.magnitude_kernel_)
    , kernels_compiled_(other.kernels_compiled_)
    , magnitude_kernel_compiled_(other.magnitude_kernel_compiled_)
    , kernel_cache_(std::move(other.kernel_cache_))
    , input_buffer_(other.input_buffer_)
    , output_buffer_(other.output_buffer_)
    , mag_only_buffer_(other.mag_only_buffer_)
    , n_point_(other.n_point_)
    , current_buffer_beams_(other.current_buffer_beams_) {

    other.backend_ = nullptr;
    other.stream_ = nullptr;
    other.module_ = nullptr;
    other.mag_module_ = nullptr;
    other.kernel_ = nullptr;
    other.magnitude_kernel_ = nullptr;
    other.kernels_compiled_ = false;
    other.magnitude_kernel_compiled_ = false;
    other.input_buffer_ = nullptr;
    other.output_buffer_ = nullptr;
    other.mag_only_buffer_ = nullptr;
    other.n_point_ = 0;
    other.current_buffer_beams_ = 0;
}

ComplexToMagPhaseROCm& ComplexToMagPhaseROCm::operator=(ComplexToMagPhaseROCm&& other) noexcept {
    if (this != &other) {
        ReleaseResources();

        backend_ = other.backend_;
        stream_ = other.stream_;
        module_ = other.module_;
        mag_module_ = other.mag_module_;
        kernel_ = other.kernel_;
        magnitude_kernel_ = other.magnitude_kernel_;
        kernels_compiled_ = other.kernels_compiled_;
        magnitude_kernel_compiled_ = other.magnitude_kernel_compiled_;
        kernel_cache_ = std::move(other.kernel_cache_);
        input_buffer_ = other.input_buffer_;
        output_buffer_ = other.output_buffer_;
        mag_only_buffer_ = other.mag_only_buffer_;
        n_point_ = other.n_point_;
        current_buffer_beams_ = other.current_buffer_beams_;

        other.backend_ = nullptr;
        other.stream_ = nullptr;
        other.module_ = nullptr;
        other.mag_module_ = nullptr;
        other.kernel_ = nullptr;
        other.magnitude_kernel_ = nullptr;
        other.kernels_compiled_ = false;
        other.magnitude_kernel_compiled_ = false;
        other.input_buffer_ = nullptr;
        other.output_buffer_ = nullptr;
        other.mag_only_buffer_ = nullptr;
        other.n_point_ = 0;
        other.current_buffer_beams_ = 0;
    }
    return *this;
}

// =========================================================================
// PART 2: Public API
// =========================================================================

std::vector<MagPhaseResult> ComplexToMagPhaseROCm::Process(
    const std::vector<std::complex<float>>& data,
    const MagPhaseParams& params)
{
    // Validate
    size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
    if (data.size() != expected) {
        throw std::invalid_argument(
            "ComplexToMagPhaseROCm::Process: input size " + std::to_string(data.size()) +
            " != expected " + std::to_string(expected) +
            " (beam_count=" + std::to_string(params.beam_count) +
            " * n_point=" + std::to_string(params.n_point) + ")");
    }

    n_point_ = params.n_point;

    if (!kernels_compiled_) {
        CompileKernels();
    }

    // Calculate batch
    size_t bytes_per_beam = CalculateBytesPerBeam();
    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params.beam_count, bytes_per_beam, params.memory_limit);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params.beam_count, optimal_batch, 3, true);

    std::vector<MagPhaseResult> all_results;
    all_results.reserve(params.beam_count);

    for (const auto& batch : batches) {
        AllocateBuffers(batch.count);

        // Upload batch data (H2D)
        const auto* batch_data = data.data() + batch.start * params.n_point;
        UploadData(batch_data, batch.count * params.n_point);

        // Execute conversion kernel
        ExecuteKernel(batch.count * n_point_);

        // Synchronize
        hipStreamSynchronize(stream_);

        // Read results
        auto batch_results = ReadResults(batch.count, batch.start);
        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

std::vector<MagPhaseResult> ComplexToMagPhaseROCm::Process(
    void* gpu_data,
    const MagPhaseParams& params,
    size_t gpu_memory_bytes)
{
    if (!gpu_data) {
        throw std::invalid_argument("ComplexToMagPhaseROCm::Process: gpu_data is null");
    }

    n_point_ = params.n_point;

    if (!kernels_compiled_) {
        CompileKernels();
    }

    size_t bytes_per_beam = CalculateBytesPerBeam();
    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : static_cast<size_t>(params.beam_count) * params.n_point * sizeof(std::complex<float>);

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params.beam_count, bytes_per_beam, params.memory_limit, external_memory);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params.beam_count, optimal_batch, 3, true);

    std::vector<MagPhaseResult> all_results;
    all_results.reserve(params.beam_count);

    for (const auto& batch : batches) {
        AllocateBuffers(batch.count);

        // D2D copy from external GPU buffer
        size_t src_offset = batch.start * params.n_point * sizeof(std::complex<float>);
        CopyGpuData(gpu_data, src_offset, batch.count * params.n_point);

        // Execute conversion kernel
        ExecuteKernel(batch.count * n_point_);
        hipStreamSynchronize(stream_);

        // Read results
        auto batch_results = ReadResults(batch.count, batch.start);
        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

void* ComplexToMagPhaseROCm::ProcessToGPU(
    const std::vector<std::complex<float>>& data,
    const MagPhaseParams& params)
{
    // Validate
    size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
    if (data.size() != expected) {
        throw std::invalid_argument(
            "ComplexToMagPhaseROCm::ProcessToGPU: input size mismatch");
    }

    n_point_ = params.n_point;

    if (!kernels_compiled_) {
        CompileKernels();
    }

    size_t total = static_cast<size_t>(params.beam_count) * params.n_point;

    // ProcessToGPU использует отдельные tmp_input/output_ptr вместо internal input_buffer_/output_buffer_.
    // ЗАЧЕМ: internal буферы принадлежат объекту и переиспользуются между вызовами.
    // output_ptr возвращается caller'у — caller владеет и обязан вызвать backend->Free(ptr).
    // Если бы мы вернули output_buffer_, объект потерял бы владение — двойной hipFree в деструкторе.
    void* tmp_input = nullptr;
    size_t input_size = total * sizeof(std::complex<float>);
    hipError_t err = hipMalloc(&tmp_input, input_size);
    if (err != hipSuccess) {
        throw std::runtime_error("ProcessToGPU: hipMalloc input failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    err = hipMemcpyHtoDAsync(tmp_input, const_cast<std::complex<float>*>(data.data()),
                              input_size, stream_);
    if (err != hipSuccess) {
        (void)hipFree(tmp_input);
        throw std::runtime_error("ProcessToGPU: hipMemcpyHtoDAsync failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // output_ptr — CALLER OWNS: размер = total × 2 × sizeof(float)
    // Layout: interleaved float2_t[total], .x=mag, .y=phase
    void* output_ptr = nullptr;
    size_t output_size = total * 2 * sizeof(float);
    err = hipMalloc(&output_ptr, output_size);
    if (err != hipSuccess) {
        (void)hipFree(tmp_input);
        throw std::runtime_error("ProcessToGPU: hipMalloc output failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    ExecuteKernelDirect(tmp_input, output_ptr, total);
    hipStreamSynchronize(stream_);

    (void)hipFree(tmp_input);  // Временный input больше не нужен — output_ptr остаётся у caller'а

    return output_ptr;
}

void* ComplexToMagPhaseROCm::ProcessToGPU(
    void* gpu_data,
    const MagPhaseParams& params,
    size_t /*gpu_memory_bytes*/)
{
    if (!gpu_data) {
        throw std::invalid_argument("ComplexToMagPhaseROCm::ProcessToGPU: gpu_data is null");
    }

    n_point_ = params.n_point;

    if (!kernels_compiled_) {
        CompileKernels();
    }

    size_t total = static_cast<size_t>(params.beam_count) * params.n_point;

    // Allocate output buffer -- CALLER WILL OWN THIS
    void* output_ptr = nullptr;
    size_t output_size = total * 2 * sizeof(float);
    hipError_t err = hipMalloc(&output_ptr, output_size);
    if (err != hipSuccess) {
        throw std::runtime_error("ProcessToGPU: hipMalloc output failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // Zero-copy: данные уже на GPU — kernel читает прямо из gpu_data, без D2D copy.
    // Это отличает GPU→GPU от CPU→GPU пути (там нужен tmp_input для H2D upload).
    ExecuteKernelDirect(gpu_data, output_ptr, total);

    hipStreamSynchronize(stream_);

    // Return output pointer -- caller owns!
    return output_ptr;
}

std::vector<MagnitudeResult> ComplexToMagPhaseROCm::ProcessMagnitude(
    void* gpu_data,
    const MagPhaseParams& params,
    size_t gpu_memory_bytes)
{
    if (!gpu_data) {
        throw std::invalid_argument("ComplexToMagPhaseROCm::ProcessMagnitude: gpu_data is null");
    }

    n_point_ = params.n_point;

    if (!magnitude_kernel_compiled_) {
        CompileMagnitudeKernel();
    }

    // Compute normalization factor on host (no division in kernel)
    float inv_n = 1.0f;
    if (params.norm_coeff < 0.0f) {
        inv_n = (params.n_point > 0) ? 1.0f / static_cast<float>(params.n_point) : 1.0f;
    } else if (params.norm_coeff > 0.0f) {
        inv_n = params.norm_coeff;
    }

    size_t bytes_per_beam = CalculateBytesPerBeam();
    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : static_cast<size_t>(params.beam_count) * params.n_point * sizeof(std::complex<float>);

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params.beam_count, bytes_per_beam, params.memory_limit, external_memory);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params.beam_count, optimal_batch, 3, true);

    std::vector<MagnitudeResult> all_results;
    all_results.reserve(params.beam_count);

    for (const auto& batch : batches) {
        AllocateBuffers(batch.count);

        // D2D copy from external GPU buffer
        size_t src_offset = batch.start * params.n_point * sizeof(std::complex<float>);
        CopyGpuData(gpu_data, src_offset, batch.count * params.n_point);

        // Execute magnitude-only kernel
        size_t total = batch.count * n_point_;
        ExecuteMagnitudeKernel(input_buffer_, mag_only_buffer_, total, inv_n);
        hipStreamSynchronize(stream_);

        // Read float results (no interleaving)
        std::vector<float> raw(total);
        hipError_t err = hipMemcpyDtoH(raw.data(), mag_only_buffer_, total * sizeof(float));
        if (err != hipSuccess) {
            throw std::runtime_error("ProcessMagnitude: hipMemcpyDtoH failed: " +
                                      std::string(hipGetErrorString(err)));
        }

        for (size_t i = 0; i < batch.count; ++i) {
            MagnitudeResult result;
            result.beam_id = static_cast<uint32_t>(batch.start + i);
            result.n_point = n_point_;
            result.magnitude.assign(raw.data() + i * n_point_,
                                     raw.data() + (i + 1) * n_point_);
            all_results.push_back(std::move(result));
        }
    }

    return all_results;
}

void* ComplexToMagPhaseROCm::ProcessMagnitudeToGPU(
    void* gpu_data,
    const MagPhaseParams& params,
    size_t /*gpu_memory_bytes*/)
{
    if (!gpu_data) {
        throw std::invalid_argument("ComplexToMagPhaseROCm::ProcessMagnitudeToGPU: gpu_data is null");
    }

    n_point_ = params.n_point;

    if (!magnitude_kernel_compiled_) {
        CompileMagnitudeKernel();
    }

    // Normalization factor (host-side, no division in kernel)
    float inv_n = 1.0f;
    if (params.norm_coeff < 0.0f) {
        inv_n = (params.n_point > 0) ? 1.0f / static_cast<float>(params.n_point) : 1.0f;
    } else if (params.norm_coeff > 0.0f) {
        inv_n = params.norm_coeff;
    }

    size_t total = static_cast<size_t>(params.beam_count) * params.n_point;

    // Allocate output -- CALLER OWNS THIS
    void* output_ptr = nullptr;
    size_t output_size = total * sizeof(float);
    hipError_t err = hipMalloc(&output_ptr, output_size);
    if (err != hipSuccess) {
        throw std::runtime_error("ProcessMagnitudeToGPU: hipMalloc output failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // Zero-copy: данные уже на GPU, kernel читает напрямую
    ExecuteMagnitudeKernel(gpu_data, output_ptr, total, inv_n);
    hipStreamSynchronize(stream_);

    return output_ptr;  // CALLER OWNS — must hipFree
}

void ComplexToMagPhaseROCm::ProcessMagnitudeToBuffer(
    void* gpu_complex_in, void* gpu_magnitude_out,
    const MagPhaseParams& params)
{
    if (!gpu_complex_in || !gpu_magnitude_out) {
        throw std::invalid_argument("ProcessMagnitudeToBuffer: null pointer");
    }

    n_point_ = params.n_point;

    if (!magnitude_kernel_compiled_) {
        CompileMagnitudeKernel();
    }

    float inv_n = 1.0f;
    if (params.norm_coeff < 0.0f)
        inv_n = (params.n_point > 0) ? 1.0f / static_cast<float>(params.n_point) : 1.0f;
    else if (params.norm_coeff > 0.0f)
        inv_n = params.norm_coeff;

    size_t total = static_cast<size_t>(params.beam_count) * params.n_point;
    ExecuteMagnitudeKernel(gpu_complex_in, gpu_magnitude_out, total, inv_n);
    hipStreamSynchronize(stream_);
}

// =========================================================================
// PART 3: GPU Resources Management
// =========================================================================

void ComplexToMagPhaseROCm::AllocateBuffers(size_t batch_beam_count) {
    // Reuse existing buffers if large enough
    if (batch_beam_count <= current_buffer_beams_ &&
        input_buffer_ && output_buffer_ && mag_only_buffer_) {
        return;
    }

    // Free old buffers
    if (input_buffer_)    { (void)hipFree(input_buffer_);    input_buffer_ = nullptr; }
    if (output_buffer_)   { (void)hipFree(output_buffer_);   output_buffer_ = nullptr; }
    if (mag_only_buffer_) { (void)hipFree(mag_only_buffer_); mag_only_buffer_ = nullptr; }

    hipError_t err;

    // 1. Input buffer: complex data
    size_t input_size = batch_beam_count * n_point_ * sizeof(std::complex<float>);
    err = hipMalloc(&input_buffer_, input_size);
    if (err != hipSuccess) {
        throw std::runtime_error("AllocateBuffers: input hipMalloc failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // 2. Output buffer: interleaved {mag, phase} (float2_t per element)
    size_t output_size = batch_beam_count * n_point_ * 2 * sizeof(float);
    err = hipMalloc(&output_buffer_, output_size);
    if (err != hipSuccess) {
        (void)hipFree(input_buffer_);
        input_buffer_ = nullptr;
        throw std::runtime_error("AllocateBuffers: output hipMalloc failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // 3. Magnitude-only output buffer: plain float[] (for ProcessMagnitude)
    size_t mag_only_size = batch_beam_count * n_point_ * sizeof(float);
    err = hipMalloc(&mag_only_buffer_, mag_only_size);
    if (err != hipSuccess) {
        (void)hipFree(input_buffer_);
        (void)hipFree(output_buffer_);
        input_buffer_ = nullptr;
        output_buffer_ = nullptr;
        throw std::runtime_error("AllocateBuffers: mag_only hipMalloc failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    current_buffer_beams_ = batch_beam_count;
}

void ComplexToMagPhaseROCm::CompileKernels() {
    if (kernels_compiled_) return;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;
    constexpr const char* kCacheName = "c2mp_kernels";

    // ─── Try loading from HSACO disk cache ─────────────────────────────────
    if (kernel_cache_) {
        try {
            auto entry = kernel_cache_->Load(kCacheName);
            if (entry.has_binary()) {
                hipError_t hip_err = hipModuleLoadData(&module_, entry.binary.data());
                if (hip_err == hipSuccess) {
                    hip_err = hipModuleGetFunction(&kernel_, module_, "complex_to_mag_phase");
                    if (hip_err == hipSuccess) {
                        kernels_compiled_ = true;
                        con.Print(gpu_id, "C2MP", "kernel loaded from HSACO cache");
                        DRVGPU_LOG_INFO_GPU(gpu_id, "C2MP", "HIP kernel loaded from cache");
                        return;
                    }
                }
                // Stale cache (different arch) -- fall through to recompile
                if (module_) {
                    (void)hipModuleUnload(module_);
                    module_ = nullptr;
                }
            }
        } catch (...) {
            // Cache miss or corrupt -- fall through to compile
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

    // ─── Compile via hiprtc ────────────────────────────────────────────────
    const char* source = kernels::GetComplexToMagPhaseKernelSource();

    hiprtcProgram prog;
    hiprtcResult rtcResult = hiprtcCreateProgram(&prog, source, "c2mp_kernel.hip",
                                                  0, nullptr, nullptr);
    if (rtcResult != HIPRTC_SUCCESS) {
        throw std::runtime_error("CompileKernels: hiprtcCreateProgram failed: " +
                                  std::to_string(static_cast<int>(rtcResult)));
    }

    // P4: Determine WARP_SIZE by GPU architecture (CDNA=64, RDNA=32)
    int warp_size = 32;  // default: RDNA (gfx10xx, gfx11xx, gfx12xx)
    if (arch_name.find("gfx9") == 0) {
        warp_size = 64;  // CDNA / Vega
    }
    std::string warp_define = "-DWARP_SIZE=" + std::to_string(warp_size);
    std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);
    std::vector<const char*> opts = {"-O3", "-std=c++17", "-DBLOCK_SIZE=256", warp_define.c_str()};
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

    hipErr = hipModuleGetFunction(&kernel_, module_, "complex_to_mag_phase");
    if (hipErr != hipSuccess) {
        throw std::runtime_error("CompileKernels: hipModuleGetFunction failed: " +
                                  std::string(hipGetErrorString(hipErr)));
    }

    kernels_compiled_ = true;
    con.Print(gpu_id, "C2MP",
              "kernel compiled (" + std::to_string(codeSize) + " bytes HSACO" +
              (arch_name.empty() ? "" : ", " + arch_name) + ")");
    DRVGPU_LOG_INFO_GPU(gpu_id, "C2MP", "HIP kernel compiled successfully");

    // ─── Save compiled binary to disk cache ────────────────────────────────
    if (kernel_cache_) {
        try {
            std::vector<uint8_t> binary(code.begin(), code.end());
            kernel_cache_->Save(kCacheName, source, binary, "",
                                "complex_to_mag_phase hiprtc kernel");
            con.Print(gpu_id, "C2MP", "kernel saved to HSACO cache");
        } catch (const std::exception& e) {
            con.Print(gpu_id, "C2MP", "warning: cache save failed: " + std::string(e.what()));
        }
    }
}

void ComplexToMagPhaseROCm::ReleaseResources() {
    // Free GPU buffers
    if (input_buffer_)    { (void)hipFree(input_buffer_);    input_buffer_ = nullptr; }
    if (output_buffer_)   { (void)hipFree(output_buffer_);   output_buffer_ = nullptr; }
    if (mag_only_buffer_) { (void)hipFree(mag_only_buffer_); mag_only_buffer_ = nullptr; }

    // Unload hiprtc modules
    if (mag_module_) {
        (void)hipModuleUnload(mag_module_);
        mag_module_ = nullptr;
        magnitude_kernel_ = nullptr;
        magnitude_kernel_compiled_ = false;
    }
    if (module_) {
        (void)hipModuleUnload(module_);
        module_ = nullptr;
        kernel_ = nullptr;
        kernels_compiled_ = false;
    }

    current_buffer_beams_ = 0;
}

// =========================================================================
// PART 4: GPU Operations
// =========================================================================

void ComplexToMagPhaseROCm::UploadData(const std::complex<float>* data, size_t count) {
    size_t data_size = count * sizeof(std::complex<float>);

    hipError_t err = hipMemcpyHtoDAsync(input_buffer_,
                                         const_cast<std::complex<float>*>(data),
                                         data_size, stream_);
    if (err != hipSuccess) {
        throw std::runtime_error("UploadData: hipMemcpyHtoDAsync failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

void ComplexToMagPhaseROCm::CopyGpuData(void* src, size_t offset_bytes, size_t count) {
    size_t data_size = count * sizeof(std::complex<float>);
    char* src_ptr = static_cast<char*>(src) + offset_bytes;

    hipError_t err = hipMemcpyDtoDAsync(input_buffer_, src_ptr, data_size, stream_);
    if (err != hipSuccess) {
        throw std::runtime_error("CopyGpuData: hipMemcpyDtoDAsync failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

void ComplexToMagPhaseROCm::ExecuteKernel(size_t total_elements) {
    ExecuteKernelDirect(input_buffer_, output_buffer_, total_elements);
}

void ComplexToMagPhaseROCm::ExecuteKernelDirect(
    void* input_ptr, void* output_ptr, size_t total_elements)
{
    unsigned int total_u = static_cast<unsigned int>(total_elements);

    unsigned int block_size = 256;
    unsigned int grid_size  = (total_u + block_size - 1) / block_size;

    void* args[] = {
        &input_ptr,
        &output_ptr,
        &total_u
    };

    hipError_t err = hipModuleLaunchKernel(
        kernel_,
        grid_size, 1, 1,     // grid
        block_size, 1, 1,    // block
        0,                    // shared memory
        stream_,              // stream
        args,                 // kernel arguments
        nullptr);             // extra

    if (err != hipSuccess) {
        throw std::runtime_error("ExecuteKernel: hipModuleLaunchKernel failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

std::vector<MagPhaseResult> ComplexToMagPhaseROCm::ReadResults(
    size_t beam_count, size_t start_beam)
{
    size_t total = beam_count * n_point_;

    // Single DtoH of interleaved {mag, phase} buffer
    // Layout: [mag0, phase0, mag1, phase1, ...] -- matches float2_t kernel output
    std::vector<float> raw(total * 2);
    hipError_t err = hipMemcpyDtoH(raw.data(), output_buffer_,
                                    total * 2 * sizeof(float));
    if (err != hipSuccess) {
        throw std::runtime_error("ReadResults: hipMemcpyDtoH failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    // Split interleaved data into per-beam results
    std::vector<MagPhaseResult> results;
    results.reserve(beam_count);

    for (size_t i = 0; i < beam_count; ++i) {
        MagPhaseResult result;
        result.beam_id = static_cast<uint32_t>(start_beam + i);
        result.n_point = n_point_;
        result.magnitude.resize(n_point_);
        result.phase.resize(n_point_);

        for (uint32_t k = 0; k < n_point_; ++k) {
            size_t idx = (i * n_point_ + k) * 2;
            result.magnitude[k] = raw[idx    ];  // .x = mag
            result.phase[k]     = raw[idx + 1];  // .y = phase
        }

        results.push_back(std::move(result));
    }

    return results;
}

void ComplexToMagPhaseROCm::CompileMagnitudeKernel() {
    if (magnitude_kernel_compiled_) return;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    const int gpu_id = backend_ ? backend_->GetDeviceIndex() : 0;

    // ─── Get target architecture ────────────────────────────────────────────
    std::string arch_name;
    try {
        auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
        arch_name = rocm_backend->GetCore().GetArchName();
    } catch (...) {
        arch_name = "";
    }

    // ─── Compile via hiprtc ────────────────────────────────────────────────
    const char* source = kernels::GetComplexToMagnitudeKernelSource();

    hiprtcProgram prog;
    hiprtcResult rtcResult = hiprtcCreateProgram(&prog, source, "c2mag_kernel.hip",
                                                  0, nullptr, nullptr);
    if (rtcResult != HIPRTC_SUCCESS) {
        throw std::runtime_error("CompileMagnitudeKernel: hiprtcCreateProgram failed");
    }

    int warp_size = 32;
    if (arch_name.find("gfx9") == 0) warp_size = 64;
    std::string warp_define = "-DWARP_SIZE=" + std::to_string(warp_size);
    std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);
    std::vector<const char*> opts = {"-O3", "-std=c++17", "-DBLOCK_SIZE=256", warp_define.c_str()};
    if (!arch_flag.empty()) opts.push_back(arch_flag.c_str());

    rtcResult = hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    if (rtcResult != HIPRTC_SUCCESS) {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        (void)hiprtcDestroyProgram(&prog);
        throw std::runtime_error("CompileMagnitudeKernel: compilation failed:\n" + log);
    }

    size_t codeSize = 0;
    hiprtcGetCodeSize(prog, &codeSize);
    std::vector<char> code(codeSize);
    hiprtcGetCode(prog, code.data());
    (void)hiprtcDestroyProgram(&prog);

    // Загружаем в отдельный модуль mag_module_ (hipFunction_t живёт пока жив модуль)
    hipError_t hipErr = hipModuleLoadData(&mag_module_, code.data());
    if (hipErr != hipSuccess) {
        throw std::runtime_error("CompileMagnitudeKernel: hipModuleLoadData failed: " +
                                  std::string(hipGetErrorString(hipErr)));
    }

    hipErr = hipModuleGetFunction(&magnitude_kernel_, mag_module_, "complex_to_magnitude");
    if (hipErr != hipSuccess) {
        (void)hipModuleUnload(mag_module_);
        mag_module_ = nullptr;
        throw std::runtime_error("CompileMagnitudeKernel: hipModuleGetFunction failed: " +
                                  std::string(hipGetErrorString(hipErr)));
    }

    magnitude_kernel_compiled_ = true;
    con.Print(gpu_id, "C2MP", "magnitude-only kernel compiled");
}

void ComplexToMagPhaseROCm::ExecuteMagnitudeKernel(
    void* input_ptr, void* output_ptr, size_t total_elements, float inv_n)
{
    unsigned int total_u = static_cast<unsigned int>(total_elements);
    unsigned int block_size = 256;
    unsigned int grid_size  = (total_u + block_size - 1) / block_size;

    void* args[] = {
        &input_ptr,
        &output_ptr,
        &inv_n,
        &total_u
    };

    hipError_t err = hipModuleLaunchKernel(
        magnitude_kernel_,
        grid_size, 1, 1,
        block_size, 1, 1,
        0,
        stream_,
        args,
        nullptr);

    if (err != hipSuccess) {
        throw std::runtime_error("ExecuteMagnitudeKernel: hipModuleLaunchKernel failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

// =========================================================================
// PART 5: Utilities
// =========================================================================

size_t ComplexToMagPhaseROCm::CalculateBytesPerBeam() const {
    // Input: complex<float> per point (8 bytes)
    size_t input_bytes = n_point_ * sizeof(std::complex<float>);

    // Output: interleaved {mag, phase} per point (8 bytes)
    size_t output_bytes = n_point_ * 2 * sizeof(float);

    return input_bytes + output_bytes;
}

}  // namespace fft_processor

#endif  // ENABLE_ROCM
