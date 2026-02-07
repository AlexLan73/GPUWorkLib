#pragma once

/**
 * @file antenna_fft_release.h
 * @brief Release implementation of FFT processing with clFFT callbacks
 *
 * High-performance implementation using clFFT pre/post callbacks
 * for zero-copy GPU processing.
 *
 * Pipeline: pre-callback (padding) -> FFT -> post-callback (magnitude + select)
 *
 * @author DrvGPU Team
 * @date 2026-02-04
 */

#include "antenna_fft_core.h"
#include "kernels/fft_kernel_sources.hpp"
#include "fft_plan_cache.hpp"

#include <memory>

namespace antenna_fft {

/**
 * @class AntennaFFTProcMax
 * @brief Release implementation - uses clFFT callbacks for maximum performance
 *
 * This is the PRODUCTION class for FFT processing.
 * All processing happens in one clFFT call with callbacks.
 *
 * Pipeline:
 * 1. Pre-callback: reads input data + padding to nFFT
 * 2. clFFT: forward FFT
 * 3. Post-callback: fftshift + magnitude calculation + select out_count_points_fft
 *
 * Usage:
 * ```cpp
 * AntennaFFTProcMax fft(params, backend);
 * auto result = fft.ProcessNew(input_data);
 * ```
 */
class AntennaFFTProcMax : public AntennaFFTCore {
public:
    /**
     * @brief Constructor
     * @param params Processing parameters
     * @param backend Pointer to IBackend (Multi-GPU support)
     */
    explicit AntennaFFTProcMax(const AntennaFFTParams& params, drv_gpu_lib::IBackend* backend);

    /**
     * @brief Destructor
     */
    ~AntennaFFTProcMax() override;

    // Delete copy, allow move
    AntennaFFTProcMax(const AntennaFFTProcMax&) = delete;
    AntennaFFTProcMax& operator=(const AntennaFFTProcMax&) = delete;
    AntennaFFTProcMax(AntennaFFTProcMax&&) noexcept = default;
    AntennaFFTProcMax& operator=(AntennaFFTProcMax&&) noexcept = default;

protected:
    // ═══════════════════════════════════════════════════════════════════════════
    // Virtual method implementations
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Initialize FFT plan with callbacks
     */
    void Initialize() override;

    /**
     * @brief Process all beams in single batch (through callbacks)
     */
    AntennaFFTResult ProcessSingleBatch(cl_mem input_signal) override;

    /**
     * @brief Process one batch (through callbacks)
     */
    std::vector<FFTResult> ProcessBatch(
        cl_mem input_signal,
        size_t start_beam,
        size_t num_beams,
        BatchProfilingData* out_profiling = nullptr) override;

    /**
     * @brief Allocate buffers for callback processing
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
     * @brief Create FFT plan with pre and post callbacks
     * @param num_beams Number of beams (batch size)
     */
    void CreateFFTPlanWithCallbacks(size_t num_beams);

    /**
     * @brief Execute FFT with callbacks
     * @param input_signal Input data buffer
     * @param num_beams Number of beams to process
     * @param start_beam Starting beam index (for batching)
     * @param out_fft_event Output FFT completion event
     * @return true if successful
     */
    bool ExecuteFFTWithCallbacks(
        cl_mem input_signal,
        size_t num_beams,
        size_t start_beam,
        cl_event* out_fft_event);

    /**
     * @brief Read results from GPU after FFT
     * @param num_beams Number of beams
     * @param start_beam Starting beam index
     * @return Results for processed beams
     */
    std::vector<FFTResult> ReadResults(size_t num_beams, size_t start_beam);

    // ═══════════════════════════════════════════════════════════════════════════
    // Private fields
    // ═══════════════════════════════════════════════════════════════════════════

    // Buffers for selected spectrum (output of post-callback)
    cl_mem buffer_selected_complex_;       // Complex values of selected points
    cl_mem buffer_selected_magnitude_;     // Magnitudes of selected points

    // Cached plan parameters
    size_t plan_num_beams_;                // Number of beams plan was created for

    // FFT Plan Cache (avoids expensive plan recreation)
    std::unique_ptr<FFTPlanCache> plan_cache_;
};

} // namespace antenna_fft
