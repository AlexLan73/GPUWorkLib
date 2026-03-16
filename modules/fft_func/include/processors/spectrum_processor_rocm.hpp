#pragma once

/**
 * @file spectrum_processor_rocm.hpp
 * @brief ROCm/HIP implementation of ISpectrumProcessor (hipFFT + hiprtc kernels)
 *
 * Strategy pattern - ROCm backend for spectrum maxima finding.
 * HIP implementation:
 * - hipFFT (no pre-callback → separate pad kernel)
 * - hiprtc-compiled kernels for padding, magnitudes, post-processing
 * - AllMaximaPipelineROCm for FindAllMaxima
 *
 * Compiles ONLY with ENABLE_ROCM=1 (Linux + AMD GPU).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "interface/i_spectrum_processor.hpp"
#include "interface/spectrum_maxima_types.h"
#include "interface/i_backend.hpp"
#include "pipelines/all_maxima_pipeline_rocm.hpp"
#include "services/profiling_types.hpp"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <hip/hiprtc.h>

#include <complex>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>

namespace antenna_fft {

/// ROCm profiling events: (name, ROCmProfilingData) pairs collected during processing
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class SpectrumProcessorROCm
 * @brief ROCm implementation: hipFFT + post-kernel + AllMaxima pipeline
 *
 * Lifecycle:
 *   1. Construct with IBackend* (ROCm backend)
 *   2. Initialize(params) — allocates buffers, creates FFT plan, compiles kernels
 *   3. ProcessFromCPU / ProcessFromGPU / FindAllMaxima...
 *   4. Destructor releases all resources
 */
class SpectrumProcessorROCm : public ISpectrumProcessor {
public:
    explicit SpectrumProcessorROCm(drv_gpu_lib::IBackend* backend);
    ~SpectrumProcessorROCm() override;

    SpectrumProcessorROCm(const SpectrumProcessorROCm&) = delete;
    SpectrumProcessorROCm& operator=(const SpectrumProcessorROCm&) = delete;

    // =========================================================================
    // ISpectrumProcessor interface
    // =========================================================================

    void Initialize(const SpectrumParams& params) override;
    bool IsInitialized() const override { return initialized_; }

    std::vector<SpectrumResult> ProcessFromCPU(
        const std::vector<std::complex<float>>& data) override {
        return ProcessFromCPU(data, nullptr);
    }
    std::vector<SpectrumResult> ProcessFromCPU(
        const std::vector<std::complex<float>>& data,
        ROCmProfEvents* prof_events);

    std::vector<SpectrumResult> ProcessFromGPU(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes = 0) override;

    std::vector<SpectrumResult> ProcessBatch(
        const std::vector<std::complex<float>>& batch_data,
        size_t start_antenna,
        size_t batch_antenna_count) override {
        return ProcessBatch(batch_data, start_antenna, batch_antenna_count, nullptr);
    }
    std::vector<SpectrumResult> ProcessBatch(
        const std::vector<std::complex<float>>& batch_data,
        size_t start_antenna,
        size_t batch_antenna_count,
        ROCmProfEvents* prof_events);

    std::vector<SpectrumResult> ProcessBatchFromGPU(
        void* gpu_data, size_t src_offset_bytes,
        size_t start_antenna, size_t batch_antenna_count) override;

    AllMaximaResult FindAllMaximaFromCPU(
        const std::vector<std::complex<float>>& data,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override {
        return FindAllMaximaFromCPU(data, dest, search_start, search_end, nullptr);
    }
    AllMaximaResult FindAllMaximaFromCPU(
        const std::vector<std::complex<float>>& data,
        OutputDestination dest, uint32_t search_start, uint32_t search_end,
        ROCmProfEvents* prof_events);

    AllMaximaResult FindAllMaximaFromGPUPipeline(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override;

    AllMaximaResult AllMaximaFromCPU(
        const std::vector<std::complex<float>>& fft_data,
        uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override;

    AllMaximaResult FindAllMaxima(
        void* fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 0, uint32_t search_end = 0) override;

    DriverType GetDriverType() const override { return DriverType::ROCm; }
    ProfilingData GetProfilingData() const override;
    void ReallocateBuffersForBatch(size_t batch_antenna_count) override;
    size_t CalculateBytesPerAntenna() const override;
    void CompilePostKernel() override;

    const SpectrumParams& GetParams() const { return params_; }

private:
    // =========================================================================
    // Internal methods
    // =========================================================================

    void CalculateFFTSize();
    static uint32_t NextPowerOf2(uint32_t n);

    /// Allocate GPU buffers for current_batch_size_ beams
    void AllocateBuffers();

    /// Create hipFFT plan for batch processing
    void CreateFFTPlan(size_t batch_count);

    /// Compile HIP kernels via hiprtc (pad + compute_magnitudes + post_kernel)
    void CompileKernels();

    /// Upload CPU data → input_buffer_ (H2D)
    void UploadData(const std::complex<float>* data, size_t element_count);

    /// Copy GPU data → input_buffer_ (D2D)
    void CopyGpuData(void* src, size_t src_offset_bytes, size_t element_count);

    /// Execute pad kernel: input_buffer_ → fft_input_ (zero-padding n_point → nFFT)
    void ExecutePadKernel(size_t beam_count, size_t beam_offset = 0);

    /// Execute hipFFT: fft_input_ → fft_output_
    void ExecuteFFT();

    /// Execute post-processing kernel (ONE_PEAK or TWO_PEAKS) → maxima_output_
    void ExecutePostKernel(size_t beam_count, size_t beam_offset = 0);

    /// Execute compute_magnitudes kernel: fft_output_ → magnitudes_buffer_
    void ExecuteComputeMagnitudes(size_t total_elements);

    /// Read results from GPU → SpectrumResult vector
    std::vector<SpectrumResult> ReadResults(size_t beam_count, size_t beam_offset = 0);

    /// Release all GPU resources
    void ReleaseResources();

    /// Ensure magnitudes buffer is large enough
    void EnsureMagnitudesBuffer(size_t total_elements);

    /// Create hipFFT plan for AllMaxima (no pre-callback, separate pad)
    void CreateAllMaximaFFTPlan(size_t batch_count);

    /// Release AllMaxima-specific resources
    void ReleaseAllMaximaResources();

    // =========================================================================
    // State
    // =========================================================================

    SpectrumParams params_;
    bool initialized_ = false;
    drv_gpu_lib::IBackend* backend_ = nullptr;

    // HIP stream (from backend)
    hipStream_t stream_ = nullptr;

    // =========================================================================
    // hipFFT (Process mode)
    // =========================================================================

    hipfftHandle plan_ = 0;
    bool plan_created_ = false;
    size_t plan_batch_size_ = 0;

    // hipFFT for AllMaxima (separate plan, no pre-callback)
    hipfftHandle allmax_plan_ = 0;
    bool allmax_plan_created_ = false;
    size_t allmax_plan_batch_size_ = 0;

    // =========================================================================
    // GPU buffers (device pointers, void*)
    // =========================================================================

    void* input_buffer_ = nullptr;       ///< Raw input: batch * n_point * sizeof(complex)
    void* fft_input_ = nullptr;          ///< Padded:    batch * nFFT * sizeof(complex)
    void* fft_output_ = nullptr;         ///< FFT out:   batch * nFFT * sizeof(complex)
    void* maxima_output_ = nullptr;      ///< MaxValue[]: batch * maxima_per_beam * sizeof(MaxValue)
    void* magnitudes_buffer_ = nullptr;  ///< |FFT|:     batch * nFFT * sizeof(float)

    // =========================================================================
    // hiprtc compiled kernels
    // =========================================================================

    hipModule_t module_ = nullptr;
    hipFunction_t pad_kernel_ = nullptr;
    hipFunction_t compute_mag_kernel_ = nullptr;
    hipFunction_t post_kernel_ = nullptr;       ///< ONE_PEAK or TWO_PEAKS
    bool kernels_compiled_ = false;

    // =========================================================================
    // Sizes and batch tracking
    // =========================================================================

    size_t current_batch_size_ = 0;      ///< Currently allocated buffer size (beams)
    size_t actual_batch_size_ = 0;       ///< Actual beams in current batch
    size_t magnitudes_buffer_size_ = 0;  ///< Current magnitudes buffer size (elements)

    static constexpr size_t LOCAL_SIZE = 256;

    // =========================================================================
    // AllMaxima pipeline
    // =========================================================================

    std::unique_ptr<AllMaximaPipelineROCm> pipeline_;
};

}  // namespace antenna_fft

#else  // !ENABLE_ROCM

// ============================================================================
// Stub for non-ROCm builds (Windows)
// ============================================================================

#include "interface/i_spectrum_processor.hpp"
#include "interface/spectrum_maxima_types.h"
#include "interface/i_backend.hpp"

namespace antenna_fft {

class SpectrumProcessorROCm : public ISpectrumProcessor {
public:
    explicit SpectrumProcessorROCm(drv_gpu_lib::IBackend* backend);
    ~SpectrumProcessorROCm() override = default;

    void Initialize(const SpectrumParams& params) override;
    bool IsInitialized() const override { return false; }

    std::vector<SpectrumResult> ProcessFromCPU(
        const std::vector<std::complex<float>>& data) override;

    std::vector<SpectrumResult> ProcessFromGPU(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes = 0) override;

    std::vector<SpectrumResult> ProcessBatch(
        const std::vector<std::complex<float>>& batch_data,
        size_t start_antenna,
        size_t batch_antenna_count) override;

    std::vector<SpectrumResult> ProcessBatchFromGPU(
        void* gpu_data, size_t src_offset_bytes,
        size_t start_antenna, size_t batch_antenna_count) override;

    AllMaximaResult FindAllMaximaFromCPU(
        const std::vector<std::complex<float>>& data,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override;

    AllMaximaResult FindAllMaximaFromGPUPipeline(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override;

    AllMaximaResult AllMaximaFromCPU(
        const std::vector<std::complex<float>>& fft_data,
        uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override;

    AllMaximaResult FindAllMaxima(
        void* fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 0, uint32_t search_end = 0) override;

    DriverType GetDriverType() const override { return DriverType::ROCm; }
    ProfilingData GetProfilingData() const override;
    void ReallocateBuffersForBatch(size_t batch_antenna_count) override;
    size_t CalculateBytesPerAntenna() const override;
    void CompilePostKernel() override;

private:
    drv_gpu_lib::IBackend* backend_ = nullptr;
};

}  // namespace antenna_fft

#endif  // ENABLE_ROCM
