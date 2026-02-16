#pragma once

/**
 * @file spectrum_processor_opencl.hpp
 * @brief OpenCL implementation of ISpectrumProcessor (clFFT, cl_mem)
 *
 * Strategy pattern - OpenCL backend for spectrum maxima finding.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "interface/i_spectrum_processor.hpp"
#include "interface/spectrum_maxima_types.h"
#include "kernels/fft_kernel_sources.hpp"
#include "kernels/all_maxima_kernel_sources.hpp"
#include "interface/i_backend.hpp"
#include "pipelines/all_maxima_pipeline_opencl.hpp"

#include <CL/cl.h>
#include <clFFT.h>
#include <complex>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>

namespace antenna_fft {

/**
 * @class SpectrumProcessorOpenCL
 * @brief OpenCL implementation: clFFT + post-kernel + AllMaxima pipeline
 */
class SpectrumProcessorOpenCL : public ISpectrumProcessor {
public:
    explicit SpectrumProcessorOpenCL(drv_gpu_lib::IBackend* backend);
    ~SpectrumProcessorOpenCL() override;

    SpectrumProcessorOpenCL(const SpectrumProcessorOpenCL&) = delete;
    SpectrumProcessorOpenCL& operator=(const SpectrumProcessorOpenCL&) = delete;

    void Initialize(const SpectrumParams& params) override;
    bool IsInitialized() const override { return initialized_; }

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

    DriverType GetDriverType() const override { return DriverType::OPENCL; }
    ProfilingData GetProfilingData() const override;

    void ReallocateBuffersForBatch(size_t batch_antenna_count) override;
    size_t CalculateBytesPerAntenna() const override;
    void CompilePostKernel() override;

    const SpectrumParams& GetParams() const { return params_; }

private:
    void CalculateFFTSize();
    static uint32_t NextPowerOf2(uint32_t n);
    void AllocateBuffers();
    void CreateFFTPlanWithCallback();
    cl_event UploadData(const std::vector<std::complex<float>>& input_data);
    cl_event ExecuteFFT(cl_event wait_event);
    cl_event ExecutePostKernel(cl_event wait_event);
    std::vector<SpectrumResult> ReadResults(cl_event wait_event);
    void WritePreCallbackHeader(size_t batch_count);
    void ReleaseResources();

    void EnsureMagnitudesBuffer(size_t total_elements);
    void CreateAllMaximaFFTPlan(size_t batch_count);
    cl_event ExecuteAllMaximaFFT(cl_event wait_event);
    void CompileComputeMagnitudesKernel();
    void ReleaseAllMaximaResources();

    struct PreCallbackHeader {
        uint32_t beam_count;
        uint32_t count_points;
        uint32_t nFFT;
        uint32_t padding1;
        uint32_t padding2;
        uint32_t padding3;
        uint32_t padding4;
        uint32_t padding5;
    };
    static_assert(sizeof(PreCallbackHeader) == 32, "PreCallbackHeader must be 32 bytes");

    SpectrumParams params_;
    bool initialized_ = false;
    drv_gpu_lib::IBackend* backend_ = nullptr;

    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;

    clfftPlanHandle plan_handle_ = 0;
    bool plan_created_ = false;

    cl_mem pre_callback_userdata_ = nullptr;
    cl_mem fft_input_ = nullptr;
    cl_mem fft_output_ = nullptr;
    cl_mem maxima_output_ = nullptr;
    cl_mem fft_temp_buffer_ = nullptr;
    cl_mem pinned_staging_buffer_ = nullptr;

    cl_program post_program_ = nullptr;
    cl_kernel post_kernel_ = nullptr;

    size_t current_batch_size_ = 0;
    size_t actual_batch_size_ = 0;
    size_t plan_batch_size_ = 0;
    size_t fft_temp_buffer_size_ = 0;
    size_t pinned_buffer_size_ = 0;
    bool clfft_initialized_ = false;

    static constexpr size_t PRE_CALLBACK_HEADER_SIZE = 32;
    static constexpr size_t LOCAL_SIZE = 256;

    cl_kernel compute_magnitudes_kernel_ = nullptr;
    cl_program all_maxima_program_ = nullptr;  // for compute_magnitudes only

    std::unique_ptr<AllMaximaPipelineOpenCL> pipeline_;

    clfftPlanHandle allmax_plan_handle_ = 0;
    bool allmax_plan_created_ = false;
    size_t allmax_plan_batch_size_ = 0;
    cl_mem magnitudes_buffer_ = nullptr;
    size_t magnitudes_buffer_size_ = 0;

};

} // namespace antenna_fft
