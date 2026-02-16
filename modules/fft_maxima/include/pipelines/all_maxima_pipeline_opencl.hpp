#pragma once

/**
 * @file all_maxima_pipeline_opencl.hpp
 * @brief OpenCL implementation: Detect -> Scan -> Compact
 *
 * Extracted from SpectrumProcessorOpenCL (Phase 3 refactoring).
 * Pipeline pattern: each step produces input for the next.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "interface/i_all_maxima_pipeline.hpp"
#include "interface/spectrum_maxima_types.h"

#include <CL/cl.h>
#include <cstdint>
#include <cstddef>

namespace antenna_fft {

/**
 * @class AllMaximaPipelineOpenCL
 * @brief OpenCL pipeline: detect_all_maxima -> prefix_sum -> compact_maxima
 */
class AllMaximaPipelineOpenCL : public IAllMaximaPipeline {
public:
    /**
     * @brief Create pipeline with OpenCL context
     * @param context OpenCL context
     * @param queue OpenCL command queue
     * @param device OpenCL device
     */
    AllMaximaPipelineOpenCL(cl_context context, cl_command_queue queue, cl_device_id device);
    ~AllMaximaPipelineOpenCL() override;

    AllMaximaPipelineOpenCL(const AllMaximaPipelineOpenCL&) = delete;
    AllMaximaPipelineOpenCL& operator=(const AllMaximaPipelineOpenCL&) = delete;

    AllMaximaResult Execute(
        void* magnitudes_gpu,
        void* fft_data_gpu,
        uint32_t beam_count,
        uint32_t nFFT,
        float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 1,
        uint32_t search_end = 0,
        size_t max_maxima_per_beam = 1000) override;

private:
    void CompileKernels();
    cl_event ExecutePrefixSum(cl_mem input, cl_mem output,
                             size_t n_per_beam, size_t beam_count,
                             cl_event wait_event);

    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_device_id device_ = nullptr;

    cl_program all_maxima_program_ = nullptr;
    cl_kernel detect_kernel_ = nullptr;

    cl_program prefix_sum_program_ = nullptr;
    cl_kernel block_scan_kernel_ = nullptr;
    cl_kernel block_add_kernel_ = nullptr;

    cl_program compact_program_ = nullptr;
    cl_kernel compact_kernel_ = nullptr;

    bool kernels_compiled_ = false;

    static constexpr size_t SCAN_LOCAL_SIZE = 256;
    static constexpr size_t SCAN_BLOCK_SIZE = SCAN_LOCAL_SIZE * 2;
};

} // namespace antenna_fft
