/**
 * @file noise_generator.cpp
 * @brief Реализация Noise генератора (Philox PRNG + Box-Muller на GPU)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "generators/noise_generator.hpp"
#include "kernel_loader.hpp"
#include <stdexcept>
#include <cmath>
#include <random>
#include <chrono>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace signal_gen {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

NoiseGenerator::NoiseGenerator(drv_gpu_lib::IBackend* backend, const NoiseParams& params)
    : backend_(backend), params_(params) {

    if (!backend_ || !backend_->IsInitialized()) {
        throw std::runtime_error("NoiseGenerator: backend is null or not initialized");
    }

    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    queue_   = static_cast<cl_command_queue>(backend_->GetNativeQueue());
    device_  = static_cast<cl_device_id>(backend_->GetNativeDevice());

    CompileKernel();
}

NoiseGenerator::~NoiseGenerator() {
    ReleaseGpuResources();
}

NoiseGenerator::NoiseGenerator(NoiseGenerator&& other) noexcept
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

NoiseGenerator& NoiseGenerator::operator=(NoiseGenerator&& other) noexcept {
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

void NoiseGenerator::GenerateToCpu(
    const SystemSampling& system,
    std::complex<float>* out,
    size_t out_size)
{
    if (out_size < system.length) {
        throw std::invalid_argument("NoiseGenerator::GenerateToCpu: out_size < length");
    }

    // Seed
    uint64_t seed = params_.seed;
    if (seed == 0) {
        seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    std::mt19937_64 rng(seed);

    if (params_.type == NoiseType::GAUSSIAN) {
        float std_dev = static_cast<float>(std::sqrt(params_.power));
        std::normal_distribution<float> dist(0.0f, std_dev);

        for (size_t i = 0; i < system.length; ++i) {
            out[i] = std::complex<float>(dist(rng), dist(rng));
        }
    } else {
        // White noise: uniform [-amplitude, +amplitude]
        float amp = static_cast<float>(std::sqrt(params_.power));
        std::uniform_real_distribution<float> dist(-amp, amp);

        for (size_t i = 0; i < system.length; ++i) {
            out[i] = std::complex<float>(dist(rng), dist(rng));
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// GPU генерация
// ════════════════════════════════════════════════════════════════════════════

cl_mem NoiseGenerator::GenerateToGpu(const SystemSampling& system, size_t beam_count) {
    size_t total_points = beam_count * system.length;
    size_t buffer_size = total_points * sizeof(std::complex<float>);

    cl_int err;
    cl_mem output = clCreateBuffer(context_, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("NoiseGenerator::GenerateToGpu: clCreateBuffer failed");
    }

    // Seed
    uint32_t seed = static_cast<uint32_t>(params_.seed);
    if (seed == 0) {
        seed = static_cast<uint32_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count() & 0xFFFFFFFF);
    }

    const char* kernel_name = (params_.type == NoiseType::GAUSSIAN)
        ? "generate_noise_gaussian"
        : "generate_noise_white";

    cl_kernel k = clCreateKernel(program_, kernel_name, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("NoiseGenerator::GenerateToGpu: clCreateKernel failed");
    }

    uint32_t tp = static_cast<uint32_t>(total_points);
    float param_val = (params_.type == NoiseType::GAUSSIAN)
        ? static_cast<float>(std::sqrt(params_.power))   // std_dev
        : static_cast<float>(std::sqrt(params_.power));   // amplitude

    err  = clSetKernelArg(k, 0, sizeof(cl_mem), &output);
    err |= clSetKernelArg(k, 1, sizeof(uint32_t), &tp);
    err |= clSetKernelArg(k, 2, sizeof(float), &param_val);
    err |= clSetKernelArg(k, 3, sizeof(uint32_t), &seed);

    if (err != CL_SUCCESS) {
        clReleaseKernel(k);
        clReleaseMemObject(output);
        throw std::runtime_error("NoiseGenerator::GenerateToGpu: clSetKernelArg failed");
    }

    size_t local_size = 256;
    size_t global_size = ((total_points + local_size - 1) / local_size) * local_size;

    err = clEnqueueNDRangeKernel(queue_, k, 1, nullptr,
                                  &global_size, &local_size, 0, nullptr, nullptr);
    clReleaseKernel(k);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("NoiseGenerator::GenerateToGpu: enqueue failed");
    }

    clFinish(queue_);
    return output;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU internals
// ════════════════════════════════════════════════════════════════════════════

void NoiseGenerator::CompileKernel() {
    // Load kernel from .cl files: prng.cl + noise_kernel.cl
    std::string source = LoadKernelWithPrng("noise_kernel.cl");
    const char* src_ptr = source.c_str();
    size_t source_len = source.size();

    cl_int err;
    program_ = clCreateProgramWithSource(context_, 1, &src_ptr, &source_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("NoiseGenerator: clCreateProgramWithSource failed");
    }

    err = clBuildProgram(program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error("NoiseGenerator kernel build failed:\n" + std::string(log.data()));
    }

    kernel_ = clCreateKernel(program_, "generate_noise_gaussian", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error("NoiseGenerator: clCreateKernel failed");
    }
}

void NoiseGenerator::ReleaseGpuResources() {
    if (kernel_) { clReleaseKernel(kernel_); kernel_ = nullptr; }
    if (program_) { clReleaseProgram(program_); program_ = nullptr; }
}

} // namespace signal_gen
