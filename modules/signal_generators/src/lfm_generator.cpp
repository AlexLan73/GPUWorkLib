/**
 * @file lfm_generator.cpp
 * @brief Реализация LFM генератора (chirp на GPU/CPU)
 *
 * s(t) = A * exp(j * (pi * k * t^2 + 2*pi*f_start*t))
 * k = (f_end - f_start) / duration
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "generators/lfm_generator.hpp"
#include "kernel_loader.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace signal_gen {
namespace {

// Сохранить cl_event для профилирования или освободить (production path).
// Ключевое правило: вызывать ПОСЛЕ того как event использован как wait-dependency.
void CollectOrRelease(cl_event ev, const char* name, LfmGenerator::ProfEvents* prof_events) {
    if (!ev) return;
    if (prof_events) prof_events->push_back({name, ev});
    else clReleaseEvent(ev);
}

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

LfmGenerator::LfmGenerator(drv_gpu_lib::IBackend* backend, const LfmParams& params)
    : backend_(backend), params_(params) {

    if (!backend_ || !backend_->IsInitialized()) {
        throw std::runtime_error("LfmGenerator: backend is null or not initialized");
    }

    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
    device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

    CompileKernel();
}

LfmGenerator::~LfmGenerator() {
    ReleaseGpuResources();
}

LfmGenerator::LfmGenerator(LfmGenerator&& other) noexcept
    : backend_(other.backend_)
    , params_(other.params_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , program_(other.program_)
    , kernel_(other.kernel_) {
    other.program_ = nullptr;
    other.kernel_ = nullptr;
}

LfmGenerator& LfmGenerator::operator=(LfmGenerator&& other) noexcept {
    if (this != &other) {
        ReleaseGpuResources();
        backend_ = other.backend_;
        params_ = other.params_;
        context_ = other.context_;
        queue_ = other.queue_;
        device_ = other.device_;
        program_ = other.program_;
        kernel_ = other.kernel_;
        other.program_ = nullptr;
        other.kernel_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU генерация (reference)
// ════════════════════════════════════════════════════════════════════════════

void LfmGenerator::GenerateToCpu(
    const SystemSampling& system,
    std::complex<float>* out,
    size_t out_size)
{
    if (out_size < system.length) {
        throw std::invalid_argument("LfmGenerator::GenerateToCpu: out_size < length");
    }

    float amp = static_cast<float>(params_.amplitude);
    float f_start = static_cast<float>(params_.f_start);
    float fs = static_cast<float>(system.fs);
    double duration = static_cast<double>(system.length) / system.fs;
    float chirp_rate = static_cast<float>(params_.GetChirpRate(duration));

    for (size_t i = 0; i < system.length; ++i) {
        float t = static_cast<float>(i) / fs;
        float phase = static_cast<float>(M_PI) * chirp_rate * t * t
                    + 2.0f * static_cast<float>(M_PI) * f_start * t;

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

cl_mem LfmGenerator::GenerateToGpu(const SystemSampling& system, size_t beam_count) {
    return GenerateToGpu(system, beam_count, nullptr);
}

cl_mem LfmGenerator::GenerateToGpu(const SystemSampling& system, size_t beam_count,
                                     ProfEvents* prof_events) {
    size_t total_points = beam_count * system.length;
    size_t buffer_size = total_points * sizeof(std::complex<float>);

    cl_int err;
    cl_mem output = clCreateBuffer(context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("LfmGenerator::GenerateToGpu: clCreateBuffer failed");
    }

    const char* kernel_name = params_.complex_iq ? "generate_lfm" : "generate_lfm_real";
    cl_kernel k = clCreateKernel(program_, kernel_name, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("LfmGenerator::GenerateToGpu: clCreateKernel failed");
    }

    uint32_t bc = static_cast<uint32_t>(beam_count);
    uint32_t np = static_cast<uint32_t>(system.length);
    float fs = static_cast<float>(system.fs);
    float f_start = static_cast<float>(params_.f_start);
    double duration = static_cast<double>(system.length) / system.fs;
    float chirp_rate = static_cast<float>(params_.GetChirpRate(duration));
    float amp = static_cast<float>(params_.amplitude);

    err  = clSetKernelArg(k, 0, sizeof(cl_mem), &output);
    err |= clSetKernelArg(k, 1, sizeof(uint32_t), &bc);
    err |= clSetKernelArg(k, 2, sizeof(uint32_t), &np);
    err |= clSetKernelArg(k, 3, sizeof(float), &fs);
    err |= clSetKernelArg(k, 4, sizeof(float), &f_start);
    err |= clSetKernelArg(k, 5, sizeof(float), &chirp_rate);
    err |= clSetKernelArg(k, 6, sizeof(float), &amp);

    if (err != CL_SUCCESS) {
        clReleaseKernel(k);
        clReleaseMemObject(output);
        throw std::runtime_error("LfmGenerator::GenerateToGpu: clSetKernelArg failed");
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
        throw std::runtime_error("LfmGenerator::GenerateToGpu: enqueue failed: " + std::to_string(err));
    }

    CollectOrRelease(ev_kernel, "Kernel", prof_events);

    clFinish(queue_);
    return output;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU internals
// ════════════════════════════════════════════════════════════════════════════

void LfmGenerator::CompileKernel() {
    // Load kernel from lfm_kernel.cl
    std::string source = LoadKernelFile("lfm_kernel.cl");
    const char* src_ptr = source.c_str();
    size_t source_len = source.size();

    cl_int err;
    program_ = clCreateProgramWithSource(context_, 1, &src_ptr, &source_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("LfmGenerator: clCreateProgramWithSource failed");
    }

    err = clBuildProgram(program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error("LfmGenerator kernel build failed:\n" + std::string(log.data()));
    }

    kernel_ = clCreateKernel(program_, "generate_lfm", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error("LfmGenerator: clCreateKernel failed");
    }
}

void LfmGenerator::ReleaseGpuResources() {
    if (kernel_) { clReleaseKernel(kernel_); kernel_ = nullptr; }
    if (program_) { clReleaseProgram(program_); program_ = nullptr; }
}

} // namespace signal_gen
