/**
 * @file cw_generator.cpp
 * @brief Реализация CW генератора (синусоида на GPU/CPU)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "generators/cw_generator.hpp"
#include "kernel_loader.hpp"
#include "prof_utils.hpp"
#include <stdexcept>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace signal_gen {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

CwGenerator::CwGenerator(drv_gpu_lib::IBackend* backend, const CwParams& params)
    : backend_(backend), params_(params) {

    if (!backend_ || !backend_->IsInitialized()) {
        throw std::runtime_error("CwGenerator: backend is null or not initialized");
    }

    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
    device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

    CompileKernel();
}

CwGenerator::~CwGenerator() {
    ReleaseGpuResources();
}

CwGenerator::CwGenerator(CwGenerator&& other) noexcept
    : backend_(other.backend_)
    , params_(other.params_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , program_(other.program_) {
    other.program_ = nullptr;
}

CwGenerator& CwGenerator::operator=(CwGenerator&& other) noexcept {
    if (this != &other) {
        ReleaseGpuResources();
        backend_ = other.backend_;
        params_ = other.params_;
        context_ = other.context_;
        queue_ = other.queue_;
        device_ = other.device_;
        program_ = other.program_;
        other.program_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU генерация (reference)
// ════════════════════════════════════════════════════════════════════════════

void CwGenerator::GenerateToCpu(
    const SystemSampling& system,
    std::complex<float>* out,
    size_t out_size)
{
    if (out_size < system.length) {
        throw std::invalid_argument("CwGenerator::GenerateToCpu: out_size < length");
    }

    float amp = static_cast<float>(params_.amplitude);
    float f0 = static_cast<float>(params_.f0);
    float fs = static_cast<float>(system.fs);
    float init_phase = static_cast<float>(params_.phase);

    for (size_t i = 0; i < system.length; ++i) {
        float t = static_cast<float>(i) / fs;
        float phase = 2.0f * static_cast<float>(M_PI) * f0 * t + init_phase;

        if (params_.complex_iq) {
            out[i] = std::complex<float>(amp * std::cos(phase), amp * std::sin(phase));
        } else {
            out[i] = std::complex<float>(amp * std::cos(phase), 0.0f);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// GPU генерация
// ════════════════════════════════════════════════════════════════════════════

cl_mem CwGenerator::GenerateToGpu(const SystemSampling& system, size_t beam_count) {
    return GenerateToGpu(system, beam_count, nullptr);
}

cl_mem CwGenerator::GenerateToGpu(const SystemSampling& system, size_t beam_count,
                                    ProfEvents* prof_events) {
    size_t total_points = beam_count * system.length;
    size_t buffer_size = total_points * sizeof(std::complex<float>);

    cl_int err;
    cl_mem output = clCreateBuffer(context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("CwGenerator::GenerateToGpu: clCreateBuffer failed: " + std::to_string(err));
    }

    // Выбираем kernel (complex или real)
    const char* kernel_name = params_.complex_iq ? "generate_cw" : "generate_cw_real";

    cl_kernel k = clCreateKernel(program_, kernel_name, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("CwGenerator::GenerateToGpu: clCreateKernel failed: " + std::to_string(err));
    }

    uint32_t bc = static_cast<uint32_t>(beam_count);
    uint32_t np = static_cast<uint32_t>(system.length);
    float fs = static_cast<float>(system.fs);
    float f0 = static_cast<float>(params_.f0);
    float fstep = static_cast<float>(params_.freq_step);
    float amp = static_cast<float>(params_.amplitude);
    float phase = static_cast<float>(params_.phase);

    err  = clSetKernelArg(k, 0, sizeof(cl_mem), &output);
    err |= clSetKernelArg(k, 1, sizeof(uint32_t), &bc);
    err |= clSetKernelArg(k, 2, sizeof(uint32_t), &np);
    err |= clSetKernelArg(k, 3, sizeof(float), &fs);
    err |= clSetKernelArg(k, 4, sizeof(float), &f0);
    err |= clSetKernelArg(k, 5, sizeof(float), &fstep);
    err |= clSetKernelArg(k, 6, sizeof(float), &amp);
    err |= clSetKernelArg(k, 7, sizeof(float), &phase);

    if (err != CL_SUCCESS) {
        clReleaseKernel(k);
        clReleaseMemObject(output);
        throw std::runtime_error("CwGenerator::GenerateToGpu: clSetKernelArg failed");
    }

    // 2D grid: dim0 = samples, dim1 = beams (eliminates div/mod in kernel)
    size_t local_size[2]  = { 256, 1 };
    size_t global_size[2] = {
        ((system.length + 255) / 256) * 256,
        beam_count
    };

    cl_event ev_kernel = nullptr;
    err = clEnqueueNDRangeKernel(queue_, k, 2, nullptr,
                                  global_size, local_size,
                                  0, nullptr, prof_events ? &ev_kernel : nullptr);
    clReleaseKernel(k);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("CwGenerator::GenerateToGpu: clEnqueueNDRangeKernel failed: " + std::to_string(err));
    }

    CollectOrRelease(ev_kernel, "Kernel", prof_events);

    clFinish(queue_);
    return output;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU internals
// ════════════════════════════════════════════════════════════════════════════

void CwGenerator::CompileKernel() {
    // Load kernel from cw_kernel.cl
    std::string source = LoadKernelFile("cw_kernel.cl");
    const char* src_ptr = source.c_str();
    size_t source_len = source.size();

    cl_int err;
    program_ = clCreateProgramWithSource(context_, 1, &src_ptr, &source_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("CwGenerator::CompileKernel: clCreateProgramWithSource failed");
    }

    err = clBuildProgram(program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error("CwGenerator kernel build failed:\n" + std::string(log.data()));
    }

}

void CwGenerator::ReleaseGpuResources() {
    if (program_) { clReleaseProgram(program_); program_ = nullptr; }
}

} // namespace signal_gen
