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
    drv_gpu_lib::IBackend* backend_ = nullptr;  ///< Не владеет; должен жить дольше FFTProcessorROCm
    hipStream_t stream_ = nullptr;              ///< Получен из backend_->GetNativeQueue()

    // hipFFT — two-plan LRU-2 cache (избегаем Destroy+Create при чередовании двух размеров batch).
    // Схема: plan_ = активный план, plan_last_ = вторичный кеш.
    // При запросе нового batch size: если совпадает с plan_last_ → swap (план уже готов).
    // При третьем размере: plan_last_ уничтожается, plan_ перемещается в plan_last_, создаётся новый plan_.
    // Для матричного бенчмарка с N разными nFFT: создавать отдельный FFTProcessorROCm на каждый размер.
    hipfftHandle plan_ = 0;
    bool plan_created_ = false;
    hipfftHandle plan_last_ = 0;   ///< Вторичный кеш: план для альтернативного batch size
    size_t plan_last_batch_ = 0;   ///< batch size, под который создан plan_last_

    // GPU buffers (device pointers, hipMalloc)
    // Стадии pipeline: input_buffer_ → [pad_data kernel] → fft_input_ → [hipFFT] → fft_output_ → [c2mp kernel] → mag_phase_interleaved_
    void* input_buffer_ = nullptr;          ///< Сырые данные без padding: batch × n_point × complex<float>
    void* fft_input_ = nullptr;             ///< После zero-padding: batch × nFFT × complex<float>; заполняет pad_data kernel
    void* fft_output_ = nullptr;            ///< Результат hipFFT: batch × nFFT × complex<float>
    void* mag_phase_interleaved_ = nullptr; ///< Интерливованный вывод: batch × nFFT × float2_t {mag, phase}.
                                            ///< Один D2H вместо двух раздельных (оптимизация). nullptr если режим COMPLEX.

    // hiprtc compiled kernels — загружаются из HSACO-кеша или компилируются lazy при первом вызове
    hipModule_t module_ = nullptr;
    hipFunction_t pad_kernel_ = nullptr;       ///< pad_data: input_buffer_ → fft_input_ (zero-padding)
    hipFunction_t mag_phase_kernel_ = nullptr; ///< complex_to_mag_phase: fft_output_ → mag_phase_interleaved_
    bool kernels_compiled_ = false;            ///< Guard lazy-init: CompileKernels() вызывается один раз

    // HSACO kernel cache (disk) — кеш по ключу "fft_processor_kernels"
    std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;

    // State
    uint32_t nFFT_ = 0;                    ///< Текущий размер FFT = nextPow2(n_point_) × repeat_count
    uint32_t n_point_ = 0;                 ///< Реальный размер входа на луч (до padding)
    size_t current_buffer_beams_ = 0;      ///< Batch size, под который выделены input/fft/output буферы
    size_t plan_batch_size_ = 0;           ///< Batch size активного плана plan_ (может != current_buffer_beams_)
    bool has_mag_phase_buffers_ = false;   ///< true если mag_phase_interleaved_ выделен
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
