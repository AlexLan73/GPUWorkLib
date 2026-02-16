/**
 * @file all_maxima_pipeline_opencl.cpp
 * @brief OpenCL pipeline: Detect -> Scan -> Compact
 *
 * Extracted from SpectrumProcessorOpenCL (Phase 3).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "pipelines/all_maxima_pipeline_opencl.hpp"
#include "backends/opencl/opencl_profiling.hpp"
#include "services/gpu_profiler.hpp"
#include "services/console_output.hpp"
#include "kernels/all_maxima_kernel_sources.hpp"

#include <cstring>
#include <stdexcept>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// Constructor, destructor
// ════════════════════════════════════════════════════════════════════════════

AllMaximaPipelineOpenCL::AllMaximaPipelineOpenCL(cl_context context,
                                                 cl_command_queue queue,
                                                 cl_device_id device)
    : context_(context), queue_(queue), device_(device)
{
    if (!context_ || !queue_ || !device_) {
        throw std::invalid_argument("AllMaximaPipelineOpenCL: context, queue, device required");
    }
}

AllMaximaPipelineOpenCL::~AllMaximaPipelineOpenCL() {
    if (detect_kernel_) { clReleaseKernel(detect_kernel_); detect_kernel_ = nullptr; }
    if (all_maxima_program_) { clReleaseProgram(all_maxima_program_); all_maxima_program_ = nullptr; }
    if (block_scan_kernel_) { clReleaseKernel(block_scan_kernel_); block_scan_kernel_ = nullptr; }
    if (block_add_kernel_) { clReleaseKernel(block_add_kernel_); block_add_kernel_ = nullptr; }
    if (prefix_sum_program_) { clReleaseProgram(prefix_sum_program_); prefix_sum_program_ = nullptr; }
    if (compact_kernel_) { clReleaseKernel(compact_kernel_); compact_kernel_ = nullptr; }
    if (compact_program_) { clReleaseProgram(compact_program_); compact_program_ = nullptr; }
}

// ════════════════════════════════════════════════════════════════════════════
// CompileKernels
// ════════════════════════════════════════════════════════════════════════════

void AllMaximaPipelineOpenCL::CompileKernels() {
    if (kernels_compiled_) return;

    cl_int err;

    {
        const char* detect_src = kernels::GetDetectAllMaximaKernelSource_opencl();
        size_t src_len = strlen(detect_src);
        all_maxima_program_ = clCreateProgramWithSource(context_, 1, &detect_src, &src_len, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("AllMaximaPipeline: detect program create failed: " + std::to_string(err));

        err = clBuildProgram(all_maxima_program_, 1, &device_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(all_maxima_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(all_maxima_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
            con.PrintError(0, "AllMaxima", std::string("Detect kernel build log:\n") + log.data());
            clReleaseProgram(all_maxima_program_);
            all_maxima_program_ = nullptr;
            throw std::runtime_error("AllMaximaPipeline: detect build failed: " + std::to_string(err));
        }

        detect_kernel_ = clCreateKernel(all_maxima_program_, "detect_all_maxima", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("AllMaximaPipeline: detect kernel create failed: " + std::to_string(err));
    }

    {
        const char* src = kernels::GetPrefixSumKernelSource_opencl();
        size_t src_len = strlen(src);
        prefix_sum_program_ = clCreateProgramWithSource(context_, 1, &src, &src_len, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("AllMaximaPipeline: scan program create failed: " + std::to_string(err));

        err = clBuildProgram(prefix_sum_program_, 1, &device_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(prefix_sum_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(prefix_sum_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
            con.PrintError(0, "AllMaxima", std::string("Scan kernel build log:\n") + log.data());
            clReleaseProgram(prefix_sum_program_);
            prefix_sum_program_ = nullptr;
            throw std::runtime_error("AllMaximaPipeline: scan build failed: " + std::to_string(err));
        }

        block_scan_kernel_ = clCreateKernel(prefix_sum_program_, "block_scan", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("AllMaximaPipeline: block_scan kernel create failed: " + std::to_string(err));

        block_add_kernel_ = clCreateKernel(prefix_sum_program_, "block_add", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("AllMaximaPipeline: block_add kernel create failed: " + std::to_string(err));
    }

    {
        const char* src = kernels::GetCompactMaximaKernelSource_opencl();
        size_t src_len = strlen(src);
        compact_program_ = clCreateProgramWithSource(context_, 1, &src, &src_len, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("AllMaximaPipeline: compact program create failed: " + std::to_string(err));

        err = clBuildProgram(compact_program_, 1, &device_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(compact_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(compact_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
            con.PrintError(0, "AllMaxima", std::string("Compact kernel build log:\n") + log.data());
            clReleaseProgram(compact_program_);
            compact_program_ = nullptr;
            throw std::runtime_error("AllMaximaPipeline: compact build failed: " + std::to_string(err));
        }

        compact_kernel_ = clCreateKernel(compact_program_, "compact_maxima", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("AllMaximaPipeline: compact kernel create failed: " + std::to_string(err));
    }

    kernels_compiled_ = true;
    drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "AllMaxima",
        "Pipeline kernels compiled (detect + scan + compact)");
}

// ════════════════════════════════════════════════════════════════════════════
// ExecutePrefixSum (beam-aware)
// ════════════════════════════════════════════════════════════════════════════

cl_event AllMaximaPipelineOpenCL::ExecutePrefixSum(
    cl_mem input, cl_mem output, size_t n_per_beam, size_t beam_count,
    cl_event wait_event)
{
    cl_int err;

    size_t blocks_per_beam = (n_per_beam + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
    size_t total_blocks = beam_count * blocks_per_beam;

    uint32_t npb = static_cast<uint32_t>(n_per_beam);
    uint32_t bc = static_cast<uint32_t>(beam_count);
    uint32_t bpb = static_cast<uint32_t>(blocks_per_beam);

    if (blocks_per_beam == 1) {
        size_t local_mem_size = SCAN_BLOCK_SIZE * sizeof(uint32_t);
        cl_mem null_mem = nullptr;

        err = clSetKernelArg(block_scan_kernel_, 0, sizeof(cl_mem), &input);
        err |= clSetKernelArg(block_scan_kernel_, 1, sizeof(cl_mem), &output);
        err |= clSetKernelArg(block_scan_kernel_, 2, sizeof(cl_mem), &null_mem);
        err |= clSetKernelArg(block_scan_kernel_, 3, sizeof(uint32_t), &npb);
        err |= clSetKernelArg(block_scan_kernel_, 4, sizeof(uint32_t), &bc);
        err |= clSetKernelArg(block_scan_kernel_, 5, sizeof(uint32_t), &bpb);
        err |= clSetKernelArg(block_scan_kernel_, 6, local_mem_size, nullptr);
        if (err != CL_SUCCESS)
            throw std::runtime_error("ExecutePrefixSum: setKernelArg failed: " + std::to_string(err));

        size_t global_size = total_blocks * SCAN_LOCAL_SIZE;
        size_t local_size = SCAN_LOCAL_SIZE;

        cl_event event = nullptr;
        err = clEnqueueNDRangeKernel(queue_, block_scan_kernel_, 1, nullptr,
            &global_size, &local_size,
            (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr), &event);
        if (err != CL_SUCCESS)
            throw std::runtime_error("ExecutePrefixSum: single block NDRange failed: " + std::to_string(err));

        return event;
    }

    cl_mem block_sums = clCreateBuffer(context_, CL_MEM_READ_WRITE,
        total_blocks * sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("ExecutePrefixSum: block_sums alloc failed: " + std::to_string(err));

    cl_mem block_sums_scanned = clCreateBuffer(context_, CL_MEM_READ_WRITE,
        total_blocks * sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(block_sums);
        throw std::runtime_error("ExecutePrefixSum: block_sums_scanned alloc failed: " + std::to_string(err));
    }

    {
        size_t local_mem_size = SCAN_BLOCK_SIZE * sizeof(uint32_t);

        err = clSetKernelArg(block_scan_kernel_, 0, sizeof(cl_mem), &input);
        err |= clSetKernelArg(block_scan_kernel_, 1, sizeof(cl_mem), &output);
        err |= clSetKernelArg(block_scan_kernel_, 2, sizeof(cl_mem), &block_sums);
        err |= clSetKernelArg(block_scan_kernel_, 3, sizeof(uint32_t), &npb);
        err |= clSetKernelArg(block_scan_kernel_, 4, sizeof(uint32_t), &bc);
        err |= clSetKernelArg(block_scan_kernel_, 5, sizeof(uint32_t), &bpb);
        err |= clSetKernelArg(block_scan_kernel_, 6, local_mem_size, nullptr);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(block_sums);
            clReleaseMemObject(block_sums_scanned);
            throw std::runtime_error("ExecutePrefixSum: L1 setKernelArg failed: " + std::to_string(err));
        }

        size_t global_size = total_blocks * SCAN_LOCAL_SIZE;
        size_t local_size = SCAN_LOCAL_SIZE;

        cl_event scan_event = nullptr;
        err = clEnqueueNDRangeKernel(queue_, block_scan_kernel_, 1, nullptr,
            &global_size, &local_size,
            (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr), &scan_event);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(block_sums);
            clReleaseMemObject(block_sums_scanned);
            throw std::runtime_error("ExecutePrefixSum: L1 NDRange failed: " + std::to_string(err));
        }

        cl_event sums_scan_event = ExecutePrefixSum(
            block_sums, block_sums_scanned, blocks_per_beam, beam_count, scan_event);
        clReleaseEvent(scan_event);

        uint32_t block_size_uint = static_cast<uint32_t>(SCAN_BLOCK_SIZE);

        err = clSetKernelArg(block_add_kernel_, 0, sizeof(cl_mem), &output);
        err |= clSetKernelArg(block_add_kernel_, 1, sizeof(cl_mem), &block_sums_scanned);
        err |= clSetKernelArg(block_add_kernel_, 2, sizeof(uint32_t), &npb);
        err |= clSetKernelArg(block_add_kernel_, 3, sizeof(uint32_t), &bc);
        err |= clSetKernelArg(block_add_kernel_, 4, sizeof(uint32_t), &bpb);
        err |= clSetKernelArg(block_add_kernel_, 5, sizeof(uint32_t), &block_size_uint);
        if (err != CL_SUCCESS) {
            clReleaseEvent(sums_scan_event);
            clReleaseMemObject(block_sums);
            clReleaseMemObject(block_sums_scanned);
            throw std::runtime_error("ExecutePrefixSum: block_add setKernelArg failed: " + std::to_string(err));
        }

        size_t total_n = beam_count * n_per_beam;
        size_t add_global = ((total_n + 255) / 256) * 256;
        size_t add_local = 256;

        cl_event add_event = nullptr;
        err = clEnqueueNDRangeKernel(queue_, block_add_kernel_, 1, nullptr,
            &add_global, &add_local,
            1, &sums_scan_event, &add_event);
        clReleaseEvent(sums_scan_event);

        if (err != CL_SUCCESS) {
            clReleaseMemObject(block_sums);
            clReleaseMemObject(block_sums_scanned);
            throw std::runtime_error("ExecutePrefixSum: block_add NDRange failed: " + std::to_string(err));
        }

        clWaitForEvents(1, &add_event);
        clReleaseMemObject(block_sums);
        clReleaseMemObject(block_sums_scanned);

        return add_event;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Execute — main pipeline
// ════════════════════════════════════════════════════════════════════════════

static double GetEventMs(cl_event event) {
    if (!event) return 0.0;
    clWaitForEvents(1, &event);
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    return (end - start) / 1e6;
}

AllMaximaResult AllMaximaPipelineOpenCL::Execute(
    void* magnitudes_gpu,
    void* fft_data_gpu,
    uint32_t beam_count,
    uint32_t nFFT,
    float sample_rate,
    OutputDestination dest,
    uint32_t search_start,
    uint32_t search_end,
    size_t max_maxima_per_beam)
{
    cl_mem mag_mem = static_cast<cl_mem>(magnitudes_gpu);
    cl_mem fft_mem = static_cast<cl_mem>(fft_data_gpu);
    if (!mag_mem)
        throw std::invalid_argument("AllMaximaPipeline::Execute: magnitudes_gpu cannot be null");

    if (search_start == 0) search_start = 1;
    if (search_end == 0) search_end = nFFT / 2;
    if (search_end >= nFFT) search_end = nFFT - 1;

    CompileKernels();

    cl_int err;
    const size_t total_elements = static_cast<size_t>(beam_count) * nFFT;

    const int gpu_id = 0;
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    const bool do_prof = profiler.IsEnabled() && profiler.IsGPUEnabled(gpu_id);
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

    cl_mem flags_buf = clCreateBuffer(context_, CL_MEM_READ_WRITE,
        total_elements * sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("AllMaximaPipeline: flags_buf alloc failed: " + std::to_string(err));

    cl_mem scan_buf = clCreateBuffer(context_, CL_MEM_READ_WRITE,
        total_elements * sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(flags_buf);
        throw std::runtime_error("AllMaximaPipeline: scan_buf alloc failed: " + std::to_string(err));
    }

    uint32_t max_output_per_beam = std::min(
        (search_end - search_start) / 2,
        static_cast<uint32_t>(max_maxima_per_beam)
    );

    cl_mem out_maxima = clCreateBuffer(context_, CL_MEM_READ_WRITE,
        static_cast<size_t>(beam_count) * max_output_per_beam * sizeof(MaxValue), nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(flags_buf);
        clReleaseMemObject(scan_buf);
        throw std::runtime_error("AllMaximaPipeline: out_maxima alloc failed: " + std::to_string(err));
    }

    cl_mem out_beam_counts = clCreateBuffer(context_, CL_MEM_READ_WRITE,
        beam_count * sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(flags_buf);
        clReleaseMemObject(scan_buf);
        clReleaseMemObject(out_maxima);
        throw std::runtime_error("AllMaximaPipeline: out_beam_counts alloc failed: " + std::to_string(err));
    }

    err = clSetKernelArg(detect_kernel_, 0, sizeof(cl_mem), &mag_mem);
    err |= clSetKernelArg(detect_kernel_, 1, sizeof(cl_mem), &flags_buf);
    err |= clSetKernelArg(detect_kernel_, 2, sizeof(uint32_t), &beam_count);
    err |= clSetKernelArg(detect_kernel_, 3, sizeof(uint32_t), &nFFT);
    err |= clSetKernelArg(detect_kernel_, 4, sizeof(uint32_t), &search_start);
    err |= clSetKernelArg(detect_kernel_, 5, sizeof(uint32_t), &search_end);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(flags_buf);
        clReleaseMemObject(scan_buf);
        clReleaseMemObject(out_maxima);
        clReleaseMemObject(out_beam_counts);
        throw std::runtime_error("AllMaximaPipeline: detect setKernelArg failed: " + std::to_string(err));
    }

    size_t global_size = ((total_elements + 255) / 256) * 256;
    size_t local_size = 256;

    cl_event detect_event = nullptr;
    err = clEnqueueNDRangeKernel(queue_, detect_kernel_, 1, nullptr,
        &global_size, &local_size, 0, nullptr, &detect_event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(flags_buf);
        clReleaseMemObject(scan_buf);
        clReleaseMemObject(out_maxima);
        clReleaseMemObject(out_beam_counts);
        throw std::runtime_error("AllMaximaPipeline: detect NDRange failed: " + std::to_string(err));
    }

    cl_event scan_all_event = ExecutePrefixSum(flags_buf, scan_buf, nFFT, beam_count, detect_event);

    double detect_ms = GetEventMs(detect_event);
    if (do_prof) {
        drv_gpu_lib::OpenCLProfilingData data{};
        if (drv_gpu_lib::FillOpenCLProfilingData(detect_event, data))
            profiler.Record(gpu_id, "AllMaxima", "Detect", data);
    }
    clReleaseEvent(detect_event);

    double scan_ms = GetEventMs(scan_all_event);
    if (do_prof) {
        drv_gpu_lib::OpenCLProfilingData data{};
        if (drv_gpu_lib::FillOpenCLProfilingData(scan_all_event, data))
            profiler.Record(gpu_id, "AllMaxima", "Scan", data);
    }

    uint32_t beam_offset = 0;  // Для single-batch режима offset = 0

    if (!fft_mem)
        throw std::invalid_argument("AllMaximaPipeline::Execute: fft_data_gpu required for MaxValue output");

    err = clSetKernelArg(compact_kernel_, 0, sizeof(cl_mem), &fft_mem);
    err |= clSetKernelArg(compact_kernel_, 1, sizeof(cl_mem), &mag_mem);
    err |= clSetKernelArg(compact_kernel_, 2, sizeof(cl_mem), &flags_buf);
    err |= clSetKernelArg(compact_kernel_, 3, sizeof(cl_mem), &scan_buf);
    err |= clSetKernelArg(compact_kernel_, 4, sizeof(cl_mem), &out_maxima);
    err |= clSetKernelArg(compact_kernel_, 5, sizeof(cl_mem), &out_beam_counts);
    err |= clSetKernelArg(compact_kernel_, 6, sizeof(uint32_t), &beam_count);
    err |= clSetKernelArg(compact_kernel_, 7, sizeof(uint32_t), &nFFT);
    err |= clSetKernelArg(compact_kernel_, 8, sizeof(float), &sample_rate);
    err |= clSetKernelArg(compact_kernel_, 9, sizeof(uint32_t), &max_output_per_beam);
    err |= clSetKernelArg(compact_kernel_, 10, sizeof(uint32_t), &beam_offset);

    cl_event compact_event = nullptr;
    size_t compact_global = ((total_elements + 255) / 256) * 256;
    size_t compact_local = 256;

    err = clEnqueueNDRangeKernel(queue_, compact_kernel_, 1, nullptr,
        &compact_global, &compact_local,
        1, &scan_all_event, &compact_event);
    clReleaseEvent(scan_all_event);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(flags_buf);
        clReleaseMemObject(scan_buf);
        clReleaseMemObject(out_maxima);
        clReleaseMemObject(out_beam_counts);
        throw std::runtime_error("AllMaximaPipeline: compact NDRange failed: " + std::to_string(err));
    }

    double compact_ms = GetEventMs(compact_event);
    if (do_prof) {
        drv_gpu_lib::OpenCLProfilingData data{};
        if (drv_gpu_lib::FillOpenCLProfilingData(compact_event, data))
            profiler.Record(gpu_id, "AllMaxima", "Compact", data);
    }

    std::vector<uint32_t> beam_counts(beam_count);
    cl_event read_counts_event = nullptr;
    clEnqueueReadBuffer(queue_, out_beam_counts, CL_FALSE, 0,
        beam_count * sizeof(uint32_t), beam_counts.data(),
        1, &compact_event, &read_counts_event);
    clWaitForEvents(1, &read_counts_event);
    clReleaseEvent(read_counts_event);
    clReleaseEvent(compact_event);

    AllMaximaResult result;
    result.destination = dest;
    result.total_maxima = 0;

    if (dest == OutputDestination::CPU || dest == OutputDestination::ALL) {
        result.beams.resize(beam_count);

        for (uint32_t b = 0; b < beam_count; ++b) {
            uint32_t count = beam_counts[b];
            if (count > max_output_per_beam) {
                con.Print(gpu_id,
                    "WARNING: Beam {} reached max_maxima limit ({}/{}), results truncated",
                    b, count, max_output_per_beam);
                count = max_output_per_beam;
            }

            result.beams[b].antenna_id = b;
            result.beams[b].num_maxima = count;
            result.total_maxima += count;

            if (count > 0) {
                result.beams[b].maxima.resize(count);

                size_t out_offset = static_cast<size_t>(b) * max_output_per_beam;

                clEnqueueReadBuffer(queue_, out_maxima, CL_TRUE,
                    out_offset * sizeof(MaxValue),
                    count * sizeof(MaxValue),
                    result.beams[b].maxima.data(),
                    0, nullptr, nullptr);
            }
        }
    } else {
        for (uint32_t b = 0; b < beam_count; ++b) {
            result.total_maxima += beam_counts[b];
        }
    }

    if (dest == OutputDestination::GPU || dest == OutputDestination::ALL) {
        result.gpu_maxima = out_maxima;
        result.gpu_counts = out_beam_counts;
        result.gpu_bytes = static_cast<size_t>(beam_count) * max_output_per_beam * sizeof(MaxValue);
        out_maxima = nullptr;
        out_beam_counts = nullptr;
    }

    con.Print(0, "AllMaxima", "Found " + std::to_string(result.total_maxima) + " maxima"
        + " | detect=" + std::to_string(detect_ms) + "ms"
        + " scan=" + std::to_string(scan_ms) + "ms"
        + " compact=" + std::to_string(compact_ms) + "ms");

    clReleaseMemObject(flags_buf);
    clReleaseMemObject(scan_buf);
    if (out_maxima) clReleaseMemObject(out_maxima);
    if (out_beam_counts) clReleaseMemObject(out_beam_counts);

    return result;
}

} // namespace antenna_fft
