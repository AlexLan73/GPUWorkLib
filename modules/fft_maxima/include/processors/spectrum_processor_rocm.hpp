#pragma once

/**
 * @file spectrum_processor_rocm.hpp
 * @brief ROCm/HIP stub for ISpectrumProcessor
 *
 * Placeholder for future AMD GPU support (hipFFT).
 * All methods throw std::runtime_error("ROCm not implemented").
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "interface/i_spectrum_processor.hpp"
#include "interface/spectrum_maxima_types.h"
#include "interface/i_backend.hpp"

namespace antenna_fft {

/**
 * @class SpectrumProcessorROCm
 * @brief ROCm stub — throws on any call
 *
 * To implement: replace throws with hipFFT + HIP kernel logic.
 */
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

} // namespace antenna_fft
