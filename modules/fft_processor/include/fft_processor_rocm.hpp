#pragma once

/**
 * @file fft_processor_rocm.hpp
 * @brief FFTProcessorROCm -- FFT 1/n beams using hipFFT (ROCm backend)
 *
 * ROCm port of FFTProcessor. Same public API, different GPU backend:
 * - hipFFT instead of clFFT
 * - hiprtc-compiled kernels for padding and mag/phase
 * - Device pointers (void*) instead of cl_mem
 *
 * Key differences from OpenCL version:
 * - No pre-callback (hipFFT doesn't support it) -> separate pad kernel
 * - hiprtc for runtime kernel compilation
 * - hipStream_t for async operations
 *
 * IMPORTANT: This file compiles ONLY with ENABLE_ROCM=1.
 * On Windows (no ROCm) this file is completely skipped.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "fft_processor_types.hpp"
#include "interface/i_backend.hpp"
#include "services/batch_manager.hpp"
#include "services/gpu_profiler.hpp"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <hip/hiprtc.h>

#include <complex>
#include <memory>
#include <vector>
#include <utility>
#include <cstdint>
#include <string>
#include <mutex>

// Forward declaration — full header included in .cpp only
namespace drv_gpu_lib { class KernelCacheService; }

namespace fft_processor {
/// Тип для сбора ROCm-тайминга из методов Process*
/// name → ROCmProfilingData (Upload, Pad, FFT, Download)
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
}  // namespace fft_processor

namespace fft_processor {

class FFTProcessorROCm {
public:
    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    /**
     * @brief Constructor
     * @param backend Pointer to IBackend (non-owning, must be ROCm backend)
     */
    explicit FFTProcessorROCm(drv_gpu_lib::IBackend* backend);

    ~FFTProcessorROCm();

    // No copying
    FFTProcessorROCm(const FFTProcessorROCm&) = delete;
    FFTProcessorROCm& operator=(const FFTProcessorROCm&) = delete;

    // Move semantics
    FFTProcessorROCm(FFTProcessorROCm&& other) noexcept;
    FFTProcessorROCm& operator=(FFTProcessorROCm&& other) noexcept;

    // =========================================================================
    // Public API -- Complex output
    // =========================================================================

    /**
     * @brief FFT with complex output (CPU data)
     * @param data Input: beam_count * n_point complex<float>
     * @param params FFT parameters
     * @param prof_events Optional: collect per-stage ROCm timing (nullptr = production, no overhead)
     *   Stages recorded: "Upload", "Pad", "FFT", "Download"
     * @return Vector of FFTComplexResult (one per beam)
     */
    std::vector<FFTComplexResult> ProcessComplex(
        const std::vector<std::complex<float>>& data,
        const FFTProcessorParams& params,
        ROCmProfEvents* prof_events = nullptr);

    /**
     * @brief FFT with complex output (GPU data)
     * @param gpu_data Device pointer to input data
     * @param params FFT parameters
     * @param gpu_memory_bytes Size of GPU buffer (0 = auto)
     * @return Vector of FFTComplexResult (one per beam)
     */
    std::vector<FFTComplexResult> ProcessComplex(
        void* gpu_data,
        const FFTProcessorParams& params,
        size_t gpu_memory_bytes = 0);

    // =========================================================================
    // Public API -- Magnitude + Phase output
    // =========================================================================

    /**
     * @brief FFT with magnitude + phase output (CPU data)
     * @param prof_events Optional: collect per-stage ROCm timing (nullptr = no overhead)
     *   Stages recorded: "Upload", "Pad", "FFT", "MagPhase", "Download"
     */
    std::vector<FFTMagPhaseResult> ProcessMagPhase(
        const std::vector<std::complex<float>>& data,
        const FFTProcessorParams& params,
        ROCmProfEvents* prof_events = nullptr);

    /**
     * @brief FFT with magnitude + phase output (GPU data)
     */
    std::vector<FFTMagPhaseResult> ProcessMagPhase(
        void* gpu_data,
        const FFTProcessorParams& params,
        size_t gpu_memory_bytes = 0);

    // =========================================================================
    // Information
    // =========================================================================

    FFTProfilingData GetProfilingData() const;
    uint32_t GetNFFT() const { return nFFT_; }

private:
    // =========================================================================
    // Utilities
    // =========================================================================

    static uint32_t NextPowerOf2(uint32_t n);
    void CalculateNFFT(const FFTProcessorParams& params);
    size_t CalculateBytesPerBeam(FFTOutputMode mode) const;

    // =========================================================================
    // GPU Resources management
    // =========================================================================

    /// Allocate/reuse buffers for batch_beam_count beams
    void AllocateBuffers(size_t batch_beam_count, FFTOutputMode mode);

    /// Create hipFFT plan for batch processing
    void CreateFFTPlan(size_t batch_beam_count);

    /// Compile HIP kernels via hiprtc (pad + mag_phase)
    void CompileKernels();

    /// Release all GPU resources
    void ReleaseResources();

    // =========================================================================
    // GPU Operations
    // =========================================================================

    /// Upload CPU data to input_buffer_
    void UploadData(const std::complex<float>* data, size_t count);

    /// Copy GPU data to input_buffer_ (D2D)
    void CopyGpuData(void* src, size_t src_offset_bytes, size_t count);

    /// Execute pad kernel: input_buffer_ -> fft_input_
    void ExecutePadKernel(size_t beam_count);

    /// Execute hipFFT: fft_input_ -> fft_output_
    void ExecuteFFT();

    /// Execute mag/phase kernel: fft_output_ -> mag_output_ + phase_output_
    void ExecuteMagPhaseKernel(size_t beam_count);

    /// Read complex results from GPU
    std::vector<FFTComplexResult> ReadComplexResults(
        size_t beam_count, size_t start_beam, float sample_rate);

    /// Read magnitude + phase results from GPU
    std::vector<FFTMagPhaseResult> ReadMagPhaseResults(
        size_t beam_count, size_t start_beam,
        float sample_rate, bool include_freq);

    // =========================================================================
    // Members
    // =========================================================================

    // Backend
    drv_gpu_lib::IBackend* backend_ = nullptr;
    hipStream_t stream_ = nullptr;

    // hipFFT — two-plan cache (avoids Destroy+Create on last smaller batch)
    hipfftHandle plan_ = 0;
    bool plan_created_ = false;
    hipfftHandle plan_last_ = 0;      ///< Cached secondary plan (different batch size)
    size_t plan_last_batch_ = 0;

    // GPU buffers (device pointers)
    void* input_buffer_ = nullptr;         ///< Raw input data: batch * n_point * sizeof(complex)
    void* fft_input_ = nullptr;            ///< Padded input:   batch * nFFT * sizeof(complex)
    void* fft_output_ = nullptr;           ///< FFT output:     batch * nFFT * sizeof(complex)
    void* mag_phase_interleaved_ = nullptr;///< Interleaved {mag, phase}: batch * nFFT * 2*sizeof(float)

    // hiprtc compiled kernels
    hipModule_t module_ = nullptr;
    hipFunction_t pad_kernel_ = nullptr;
    hipFunction_t mag_phase_kernel_ = nullptr;
    bool kernels_compiled_ = false;

    // HSACO kernel cache (disk)
    std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;

    // State
    uint32_t nFFT_ = 0;
    uint32_t n_point_ = 0;
    size_t current_buffer_beams_ = 0;
    size_t plan_batch_size_ = 0;
    bool has_mag_phase_buffers_ = false;
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
