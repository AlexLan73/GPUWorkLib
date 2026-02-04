#pragma once

/**
 * @file antenna_fft_debug.h
 * @brief Debug implementation of FFT processing with step-by-step kernels
 *
 * Debug/Test implementation that executes each step separately:
 * 1. Padding kernel
 * 2. FFT (without callbacks)
 * 3. Post-processing kernel
 * 4. Maxima search kernel
 *
 * Allows inspection of intermediate results at each step.
 *
 * @author DrvGPU Team
 * @date 2026-02-04
 */

#include "antenna_fft_core.h"
#include "kernels/fft_kernel_sources.hpp"

namespace antenna_fft {

/**
 * @class AntennaFFTDebug
 * @brief Debug implementation - step-by-step kernels for testing
 *
 * This class is for DEBUG and TESTING purposes.
 * Each processing step is executed separately, allowing:
 * - Inspection of intermediate results
 * - Step-by-step debugging
 * - Comparison with Release implementation
 *
 * Pipeline (step-by-step):
 * 1. ExecutePaddingKernel(): input data -> padded FFT input
 * 2. ExecuteFFTOnly(): FFT without callbacks
 * 3. ExecutePostKernel(): fftshift + magnitude + select
 * 4. FindMaximaOnGPU(): maxima search
 *
 * Usage:
 * ```cpp
 * AntennaFFTDebug fft_debug(params, backend);
 *
 * // Full processing (automatic)
 * auto result = fft_debug.ProcessNew(input_data);
 *
 * // OR step-by-step for debugging:
 * fft_debug.SetInputData(input_data);
 * fft_debug.ExecutePaddingKernel();
 * // inspect buffer_fft_input_...
 * fft_debug.ExecuteFFTOnly();
 * // inspect buffer_fft_output_...
 * fft_debug.ExecutePostKernel();
 * // inspect selected buffers...
 * auto maxima = fft_debug.FindMaximaOnGPU();
 * ```
 */
class AntennaFFTDebug : public AntennaFFTCore {
public:
    /**
     * @brief Constructor
     * @param params Processing parameters
     * @param backend Pointer to IBackend (Multi-GPU support)
     */
    explicit AntennaFFTDebug(const AntennaFFTParams& params, drv_gpu_lib::IBackend* backend);

    /**
     * @brief Destructor
     */
    ~AntennaFFTDebug() override;

    // Delete copy, allow move
    AntennaFFTDebug(const AntennaFFTDebug&) = delete;
    AntennaFFTDebug& operator=(const AntennaFFTDebug&) = delete;
    AntennaFFTDebug(AntennaFFTDebug&&) noexcept = default;
    AntennaFFTDebug& operator=(AntennaFFTDebug&&) noexcept = default;

    // ═══════════════════════════════════════════════════════════════════════════
    // Debug methods - step-by-step execution
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Set input data from CPU (copies to GPU)
     * @param input_data Input complex data
     */
    void SetInputData(const std::vector<std::complex<float>>& input_data);

    /**
     * @brief Set input data from GPU buffer
     * @param input_signal GPU buffer with input data
     * @param num_beams Number of beams in buffer
     */
    void SetInputData(cl_mem input_signal, size_t num_beams);

    /**
     * @brief Execute padding kernel only
     * Reads from input buffer, writes to buffer_fft_input_
     * @param wait_event Optional event to wait for
     * @param out_event Optional output event
     */
    void ExecutePaddingKernel(cl_event wait_event = nullptr, cl_event* out_event = nullptr);

    /**
     * @brief Execute FFT only (no callbacks)
     * Reads from buffer_fft_input_, writes to buffer_fft_output_
     * @param wait_event Optional event to wait for
     * @param out_event Optional output event
     */
    void ExecuteFFTOnly(cl_event wait_event = nullptr, cl_event* out_event = nullptr);

    /**
     * @brief Execute post-processing kernel only
     * Reads from buffer_fft_output_, writes to selected buffers
     * @param wait_event Optional event to wait for
     * @param out_event Optional output event
     */
    void ExecutePostKernel(cl_event wait_event = nullptr, cl_event* out_event = nullptr);

    /**
     * @brief Execute maxima search on GPU
     * @param wait_event Optional event to wait for
     * @return Maxima results for all beams
     */
    std::vector<std::vector<FFTMaxResult>> FindMaximaOnGPU(cl_event wait_event = nullptr);

    // ═══════════════════════════════════════════════════════════════════════════
    // Buffer access for debugging
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Read FFT input buffer (after padding)
     * @return Vector of complex values
     */
    std::vector<std::complex<float>> ReadFFTInputBuffer();

    /**
     * @brief Read FFT output buffer (after FFT)
     * @return Vector of complex values
     */
    std::vector<std::complex<float>> ReadFFTOutputBuffer();

    /**
     * @brief Read selected complex buffer (after post-processing)
     * @return Vector of complex values
     */
    std::vector<std::complex<float>> ReadSelectedComplexBuffer();

    /**
     * @brief Read selected magnitude buffer (after post-processing)
     * @return Vector of magnitudes
     */
    std::vector<float> ReadSelectedMagnitudeBuffer();

    /**
     * @brief Get current number of beams set
     */
    size_t GetCurrentBeams() const { return debug_num_beams_; }

protected:
    // ═══════════════════════════════════════════════════════════════════════════
    // Virtual method implementations
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Initialize kernels and FFT plan (no callbacks)
     */
    void Initialize() override;

    /**
     * @brief Process all beams in single batch (step-by-step)
     */
    AntennaFFTResult ProcessSingleBatch(cl_mem input_signal) override;

    /**
     * @brief Process one batch (step-by-step)
     */
    std::vector<FFTResult> ProcessBatch(
        cl_mem input_signal,
        size_t start_beam,
        size_t num_beams,
        BatchProfilingData* out_profiling = nullptr) override;

    /**
     * @brief Allocate buffers for debug processing
     */
    void AllocateBuffers(size_t num_beams) override;

    /**
     * @brief Release allocated buffers
     */
    void ReleaseBuffers() override;

private:
    // ═══════════════════════════════════════════════════════════════════════════
    // Private methods
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Create FFT plan without callbacks
     */
    void CreateFFTPlanNoCallbacks(size_t num_beams);

    /**
     * @brief Create padding kernel
     */
    void CreatePaddingKernel();

    /**
     * @brief Create post-processing kernel
     */
    void CreatePostKernel();

    /**
     * @brief Create maxima reduction kernel
     */
    void CreateMaximaKernel();

    /**
     * @brief Convert maxima results to FFTResult vector
     */
    std::vector<FFTResult> ConvertMaximaToResults(
        const std::vector<std::vector<FFTMaxResult>>& maxima,
        size_t start_beam);

    // ═══════════════════════════════════════════════════════════════════════════
    // Private fields
    // ═══════════════════════════════════════════════════════════════════════════

    // Input buffer (copied from user data)
    cl_mem buffer_input_;

    // Buffers for selected spectrum (output of post kernel)
    cl_mem buffer_selected_complex_;
    cl_mem buffer_selected_magnitude_;

    // Kernels
    cl_kernel padding_kernel_;
    cl_kernel post_kernel_;
    cl_kernel maxima_kernel_;

    // Programs (for kernel compilation)
    cl_program padding_program_;
    cl_program post_program_;
    cl_program maxima_program_;

    // Cached plan parameters
    size_t plan_num_beams_;

    // Current debug state
    size_t debug_num_beams_;
    bool input_data_set_;
};

} // namespace antenna_fft
