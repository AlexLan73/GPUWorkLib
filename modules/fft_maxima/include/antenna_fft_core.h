#pragma once

/**
 * @file antenna_fft_core.h
 * @brief Base abstract class for FFT processing with maxima search
 *
 * Core functionality shared between Release and Debug implementations.
 * Contains common batching logic, buffer management, and utilities.
 *
 * @author DrvGPU Team
 * @date 2026-02-04
 */

#include "interface/antenna_fft_params.h"
#include "interface/i_backend.hpp"

#include <CL/cl.h>
#include <clFFT.h>
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <chrono>

namespace antenna_fft {

/**
 * @class AntennaFFTCore
 * @brief Abstract base class for FFT processing
 *
 * Provides common functionality:
 * - Batching logic (ProcessWithBatching)
 * - Buffer allocation
 * - FFT plan management
 * - Profiling
 *
 * Derived classes implement:
 * - ProcessSingleBatch() - how to process one batch
 * - Initialize() - specific initialization (callbacks vs kernels)
 */
class AntennaFFTCore {
public:
    // ═══════════════════════════════════════════════════════════════════════════
    // Public types
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Profiling data for one batch
     */
    struct BatchProfilingData {
        size_t batch_index = 0;
        size_t start_beam = 0;
        size_t num_beams = 0;
        double padding_time_ms = 0.0;
        double fft_time_ms = 0.0;
        double post_time_ms = 0.0;
        double gpu_time_ms = 0.0;
    };

    /**
     * @brief Batch processing configuration
     */
    struct BatchConfig {
        double memory_usage_limit = 0.65;    // 65% of available memory
        double batch_size_ratio = 0.22;      // 22% of beams per batch
        size_t min_beams_for_batch = 10;     // Minimum beams for batch mode
        size_t beams_per_batch = 0;          // Computed beams per batch
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Constructor / Destructor
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Constructor with IBackend for Multi-GPU support
     * @param params Processing parameters
     * @param backend Pointer to IBackend (NOT SINGLETON!)
     */
    explicit AntennaFFTCore(const AntennaFFTParams& params, drv_gpu_lib::IBackend* backend);

    /**
     * @brief Virtual destructor
     */
    virtual ~AntennaFFTCore();

    // Delete copy, allow move
    AntennaFFTCore(const AntennaFFTCore&) = delete;
    AntennaFFTCore& operator=(const AntennaFFTCore&) = delete;
    AntennaFFTCore(AntennaFFTCore&&) noexcept;
    AntennaFFTCore& operator=(AntennaFFTCore&&) noexcept;

    // ═══════════════════════════════════════════════════════════════════════════
    // Public interface (common for all implementations)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Main entry point - process data from CPU
     * @param input_data Vector of complex numbers (beam_count * count_points)
     * @return Processing results for all beams
     */
    AntennaFFTResult ProcessNew(const std::vector<std::complex<float>>& input_data);

    /**
     * @brief Main entry point - process data from GPU
     * @param input_signal GPU buffer with input data
     * @return Processing results for all beams
     */
    AntennaFFTResult ProcessNew(cl_mem input_signal);

    /**
     * @brief Process using batching (common loop, virtual ProcessBatch)
     * @param input_signal GPU buffer with all input data
     * @return Processing results for all beams
     */
    AntennaFFTResult ProcessWithBatching(cl_mem input_signal);

    /**
     * @brief Get last profiling results
     */
    const FFTProfilingResults& GetLastProfilingResults() const { return last_profiling_results_; }

    /**
     * @brief Get computed nFFT size
     */
    size_t GetNFFT() const { return nFFT_; }

    /**
     * @brief Get processing parameters
     */
    const AntennaFFTParams& GetParams() const { return params_; }

    /**
     * @brief Get batch profiling data
     */
    const std::vector<BatchProfilingData>& GetBatchProfiling() const { return batch_profiling_; }

    /**
     * @brief Check if batch mode was used in last processing
     */
    bool WasBatchModeUsed() const { return last_used_batch_mode_; }

protected:
    // ═══════════════════════════════════════════════════════════════════════════
    // Virtual methods (implemented by derived classes)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Initialize specific resources (FFT plans, kernels)
     * Called from constructor after base initialization
     */
    virtual void Initialize() = 0;

    /**
     * @brief Process single batch (all beams fit in memory)
     * @param input_signal GPU buffer with input data
     * @return Processing results
     */
    virtual AntennaFFTResult ProcessSingleBatch(cl_mem input_signal) = 0;

    /**
     * @brief Process one batch in batching mode
     * @param input_signal Full input buffer (all beams)
     * @param start_beam Starting beam index
     * @param num_beams Number of beams in this batch
     * @param out_profiling Optional profiling output
     * @return Results for beams in this batch
     */
    virtual std::vector<FFTResult> ProcessBatch(
        cl_mem input_signal,
        size_t start_beam,
        size_t num_beams,
        BatchProfilingData* out_profiling = nullptr) = 0;

    /**
     * @brief Allocate GPU buffers for processing
     * @param num_beams Number of beams to allocate for
     */
    virtual void AllocateBuffers(size_t num_beams) = 0;

    /**
     * @brief Release allocated buffers
     */
    virtual void ReleaseBuffers() = 0;

    // ═══════════════════════════════════════════════════════════════════════════
    // Protected utilities (common implementation)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Calculate nFFT from count_points
     */
    size_t CalculateNFFT(size_t count_points) const;

    /**
     * @brief Check if number is power of two
     */
    bool IsPowerOf2(size_t n) const;

    /**
     * @brief Find next larger power of two
     */
    size_t NextPowerOf2(size_t n) const;

    /**
     * @brief Estimate required memory for given number of beams
     */
    size_t EstimateRequiredMemory(size_t num_beams) const;

    /**
     * @brief Check if available memory is sufficient
     */
    bool CheckAvailableMemory(size_t required_memory, double threshold = 0.4) const;

    /**
     * @brief Calculate batch configuration
     */
    void CalculateBatchConfig();

    /**
     * @brief Check if batching is needed
     */
    bool NeedsBatching() const;

    /**
     * @brief Create input buffer from CPU data
     */
    cl_mem CreateInputBuffer(const std::vector<std::complex<float>>& input_data);

    /**
     * @brief Create pre-callback userdata buffer
     */
    void CreatePreCallbackUserData(size_t num_beams);

    /**
     * @brief Create post-callback userdata buffer
     */
    void CreatePostCallbackUserData(size_t num_beams);

    /**
     * @brief Profile OpenCL event
     */
    double ProfileEvent(cl_event event, const std::string& operation_name);

    /**
     * @brief Release FFT plan
     */
    void ReleaseFFTPlan();

    // ═══════════════════════════════════════════════════════════════════════════
    // Protected fields (accessible by derived classes)
    // ═══════════════════════════════════════════════════════════════════════════

    AntennaFFTParams params_;              // Processing parameters
    size_t nFFT_;                          // Computed FFT size

    // DrvGPU backend (Multi-GPU support)
    drv_gpu_lib::IBackend* backend_;       // Pointer to backend (NOT owned)

    // OpenCL resources (obtained from backend)
    cl_context context_;                   // OpenCL context
    cl_command_queue queue_;               // Command queue
    cl_device_id device_;                  // OpenCL device

    // clFFT resources
    clfftPlanHandle plan_handle_;          // FFT plan handle
    bool plan_created_;                    // Plan creation flag

    // Common GPU buffers
    cl_mem buffer_fft_input_;              // FFT input buffer (nFFT * beam_count)
    cl_mem buffer_fft_output_;             // FFT output buffer
    cl_mem buffer_maxima_;                 // Maxima buffer

    // Userdata buffers for callbacks
    cl_mem pre_callback_userdata_;         // Userdata for pre-callback
    cl_mem post_callback_userdata_;        // Userdata for post-callback

    // Profiling
    FFTProfilingResults last_profiling_results_;
    std::vector<BatchProfilingData> batch_profiling_;
    double batch_total_cpu_time_ms_;
    bool last_used_batch_mode_;

    // Batch configuration
    BatchConfig batch_config_;
    size_t current_buffer_beams_;          // Current allocated buffer size
};

} // namespace antenna_fft
