#pragma once

/**
 * @file complex_to_mag_phase_rocm.hpp
 * @brief ComplexToMagPhaseROCm -- direct complex-to-magnitude+phase conversion on GPU
 *
 * Separate class for converting complex IQ data to magnitude+phase pairs.
 * Does NOT perform FFT -- pure mathematical conversion:
 *   magnitude = sqrt(re^2 + im^2)
 *   phase     = atan2(im, re)
 *
 * Supports:
 * - CPU data input (std::vector<complex<float>>)
 * - GPU data input (void* device pointer)
 * - CPU output (std::vector<MagPhaseResult>)
 * - GPU output (void* -- caller owns, must Free)
 * - Batch processing for large datasets (BatchManager)
 *
 * Key differences from FFTProcessorROCm:
 * - No hipFFT dependency (no FFT plan, no padding)
 * - Only 2 GPU buffers (input + output, no fft_input/fft_output)
 * - ProcessToGPU() methods for "stay on GPU" use case
 * - Lighter MagPhaseResult types (no nFFT, sample_rate, frequency)
 *
 * IMPORTANT: This file compiles ONLY with ENABLE_ROCM=1.
 * On Windows (no ROCm) this file is completely skipped.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include "types/mag_phase_types.hpp"
#include "interface/i_backend.hpp"
#include "services/batch_manager.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <complex>
#include <memory>
#include <vector>
#include <cstdint>
#include <string>

// Forward declaration -- full header included in .cpp only
namespace drv_gpu_lib { class KernelCacheService; }

namespace fft_processor {

class ComplexToMagPhaseROCm {
public:
    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    /**
     * @brief Constructor
     * @param backend Pointer to IBackend (non-owning, must be ROCm backend)
     */
    explicit ComplexToMagPhaseROCm(drv_gpu_lib::IBackend* backend);

    ~ComplexToMagPhaseROCm();

    // No copying
    ComplexToMagPhaseROCm(const ComplexToMagPhaseROCm&) = delete;
    ComplexToMagPhaseROCm& operator=(const ComplexToMagPhaseROCm&) = delete;

    // Move semantics
    ComplexToMagPhaseROCm(ComplexToMagPhaseROCm&& other) noexcept;
    ComplexToMagPhaseROCm& operator=(ComplexToMagPhaseROCm&& other) noexcept;

    // =========================================================================
    // Public API -- CPU input -> CPU output
    // =========================================================================

    /**
     * @brief Convert complex data to magnitude+phase (CPU to CPU)
     * @param data Input: beam_count * n_point complex<float> values
     * @param params Conversion parameters (beam_count, n_point)
     * @return Vector of MagPhaseResult (one per beam)
     */
    std::vector<MagPhaseResult> Process(
        const std::vector<std::complex<float>>& data,
        const MagPhaseParams& params);

    // =========================================================================
    // Public API -- GPU input -> CPU output
    // =========================================================================

    /**
     * @brief Convert complex data to magnitude+phase (GPU to CPU)
     * @param gpu_data Device pointer to complex data (beam_count * n_point)
     * @param params Conversion parameters
     * @param gpu_memory_bytes Size of GPU buffer in bytes (0 = auto-calculate)
     * @return Vector of MagPhaseResult (one per beam)
     */
    std::vector<MagPhaseResult> Process(
        void* gpu_data,
        const MagPhaseParams& params,
        size_t gpu_memory_bytes = 0);

    // =========================================================================
    // Public API -- CPU input -> GPU output ("stay on GPU")
    // =========================================================================

    /**
     * @brief Convert and keep result on GPU
     * @param data Input: beam_count * n_point complex<float> values
     * @param params Conversion parameters
     * @return Device pointer to interleaved {mag, phase} pairs (float2_t layout)
     *         CALLER OWNS this pointer -- must call backend->Free() when done!
     *         Size = beam_count * n_point * 2 * sizeof(float)
     * @throws std::runtime_error if data does not fit in GPU memory
     */
    void* ProcessToGPU(
        const std::vector<std::complex<float>>& data,
        const MagPhaseParams& params);

    // =========================================================================
    // Public API -- GPU input -> GPU output (full GPU pipeline)
    // =========================================================================

    /**
     * @brief Convert GPU data, output stays on GPU (zero-copy path)
     * @param gpu_data Device pointer to complex data
     * @param params Conversion parameters
     * @param gpu_memory_bytes Size of GPU buffer in bytes (0 = auto-calculate)
     * @return Device pointer to interleaved {mag, phase} pairs (float2_t layout)
     *         CALLER OWNS this pointer -- must call backend->Free() when done!
     * @throws std::runtime_error if output buffer allocation fails
     */
    void* ProcessToGPU(
        void* gpu_data,
        const MagPhaseParams& params,
        size_t gpu_memory_bytes = 0);

private:
    // =========================================================================
    // Internal methods
    // =========================================================================

    /// Compile HIP kernel via hiprtc
    void CompileKernels();

    /// Allocate/reuse internal GPU buffers for batch_beam_count beams
    void AllocateBuffers(size_t batch_beam_count);

    /// Release all GPU resources
    void ReleaseResources();

    /// Upload CPU data to input_buffer_ (H2D async)
    void UploadData(const std::complex<float>* data, size_t count);

    /// Copy GPU data to input_buffer_ (D2D async)
    void CopyGpuData(void* src, size_t offset_bytes, size_t count);

    /// Launch complex_to_mag_phase kernel
    void ExecuteKernel(size_t total_elements);

    /// Execute kernel with custom input/output pointers (for ProcessToGPU)
    void ExecuteKernelDirect(void* input_ptr, void* output_ptr, size_t total_elements);

    /// Read interleaved results from GPU and split into per-beam MagPhaseResult
    std::vector<MagPhaseResult> ReadResults(size_t beam_count, size_t start_beam);

    /// Calculate GPU memory required per beam (input + output)
    size_t CalculateBytesPerBeam() const;

    // =========================================================================
    // Members
    // =========================================================================

    // Backend
    drv_gpu_lib::IBackend* backend_ = nullptr;
    hipStream_t stream_ = nullptr;

    // hiprtc compiled kernel
    hipModule_t module_ = nullptr;
    hipFunction_t kernel_ = nullptr;
    bool kernels_compiled_ = false;

    // HSACO kernel cache (disk)
    std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;

    // GPU buffers (internal, for batched CPU->CPU / GPU->CPU processing)
    void* input_buffer_ = nullptr;    ///< Complex input: batch * n_point * sizeof(complex<float>)
    void* output_buffer_ = nullptr;   ///< Interleaved {mag,phase}: batch * n_point * 2 * sizeof(float)

    // State
    uint32_t n_point_ = 0;
    size_t current_buffer_beams_ = 0;
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
