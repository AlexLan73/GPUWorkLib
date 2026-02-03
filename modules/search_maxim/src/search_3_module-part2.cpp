/**
 * @file search_3_module.cpp - ЧАСТЬ 2/4
 * @brief Создание kernels и FFT планов
 */

#include "search_3_module.hpp"
#include "common/logger.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef SEARCH_3_KERNELS_PATH
#define SEARCH_3_KERNELS_PATH "kernels"
#endif

namespace drv_gpu_lib {
namespace search_3_ {

// ════════════════════════════════════════════════════════════════════════════
// LoadKernelSource
// ════════════════════════════════════════════════════════════════════════════

std::string Search3Module::LoadKernelSource(const std::string& filename) {
    std::string full_path = std::string(SEARCH_3_KERNELS_PATH) + "/" + filename;
    
    std::ifstream file(full_path);
    if (!file.is_open()) {
        throw std::runtime_error("Search3Module: Cannot open kernel file: " + full_path);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// ════════════════════════════════════════════════════════════════════════════
// CreateKernels
// ════════════════════════════════════════════════════════════════════════════

void Search3Module::CreateKernels() {
    DRVGPU_LOG_INFO("Search3Module", "Creating kernels...");
    
    // Загрузить исходный код
    std::string source = LoadKernelSource("search_3_fft.cl");
    const char* source_ptr = source.c_str();
    size_t source_size = source.length();
    
    // Создать программу
    cl_int err;
    program_ = clCreateProgramWithSource(context_, 1, &source_ptr, &source_size, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Search3Module: clCreateProgramWithSource failed");
    }
    
    // Скомпилировать
    err = clBuildProgram(program_, 1, &device_, "-cl-std=CL2.0", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Получить лог компиляции
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        
        std::string error_msg = "Search3Module: Kernel compilation failed:\n" + std::string(log.data());
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error(error_msg);
    }
    
    // Создать kernels
    padding_kernel_ = clCreateKernel(program_, "padding_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Search3Module: Failed to create padding_kernel");
    }
    
    post_kernel_ = clCreateKernel(program_, "post_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Search3Module: Failed to create post_kernel");
    }
    
    reduction_kernel_ = clCreateKernel(program_, "reduction_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Search3Module: Failed to create reduction_kernel");
    }
    
    DRVGPU_LOG_INFO("Search3Module", "Kernels created ✅");
}

// ════════════════════════════════════════════════════════════════════════════
// ReleaseKernels
// ════════════════════════════════════════════════════════════════════════════

void Search3Module::ReleaseKernels() {
    if (padding_kernel_) {
        clReleaseKernel(padding_kernel_);
        padding_kernel_ = nullptr;
    }
    
    if (post_kernel_) {
        clReleaseKernel(post_kernel_);
        post_kernel_ = nullptr;
    }
    
    if (reduction_kernel_) {
        clReleaseKernel(reduction_kernel_);
        reduction_kernel_ = nullptr;
    }
    
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// CreateOrReuseFFTPlan - для полной обработки
// ════════════════════════════════════════════════════════════════════════════

void Search3Module::CreateOrReuseFFTPlan() {
    // Если план уже существует, переиспользовать
    if (main_plan_handle_ != 0) {
        DRVGPU_LOG_INFO("Search3Module", "Reusing existing FFT plan ♻️");
        return;
    }
    
    DRVGPU_LOG_INFO("Search3Module", "Creating FFT plan...");
    
    clfftStatus status;
    size_t clLengths[1] = { nFFT_ };
    
    // Создать план
    status = clfftCreateDefaultPlan(&main_plan_handle_, context_, CLFFT_1D, clLengths);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("Search3Module: clfftCreateDefaultPlan failed");
    }
    
    // Настроить параметры
    status = clfftSetPlanPrecision(main_plan_handle_, CLFFT_SINGLE);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&main_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetPlanPrecision failed");
    }
    
    status = clfftSetLayout(main_plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&main_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetLayout failed");
    }
    
    status = clfftSetResultLocation(main_plan_handle_, CLFFT_OUTOFPLACE);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&main_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetResultLocation failed");
    }
    
    // Batch: количество лучей
    size_t batch_size = params_.beam_count;
    size_t stride_in = nFFT_;
    size_t stride_out = nFFT_;
    size_t dist_in = nFFT_;
    size_t dist_out = nFFT_;
    
    status = clfftSetPlanBatchSize(main_plan_handle_, batch_size);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&main_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetPlanBatchSize failed");
    }
    
    status = clfftSetPlanInStride(main_plan_handle_, CLFFT_1D, &stride_in);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&main_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetPlanInStride failed");
    }
    
    status = clfftSetPlanOutStride(main_plan_handle_, CLFFT_1D, &stride_out);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&main_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetPlanOutStride failed");
    }
    
    status = clfftSetPlanDistance(main_plan_handle_, dist_in, dist_out);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&main_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetPlanDistance failed");
    }
    
    // Bake план
    status = clfftBakePlan(main_plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&main_plan_handle_);
        throw std::runtime_error("Search3Module: clfftBakePlan failed");
    }
    
    DRVGPU_LOG_INFO("Search3Module", "FFT plan created ✅");
}

// ════════════════════════════════════════════════════════════════════════════
// CreateBatchFFTPlan - для batch обработки
// ════════════════════════════════════════════════════════════════════════════

void Search3Module::CreateBatchFFTPlan(size_t batch_size) {
    // Если план существует и размер совпадает, переиспользовать
    if (batch_plan_handle_ != 0 && batch_plan_beams_ == batch_size) {
        DRVGPU_LOG_INFO("Search3Module", "Reusing existing batch FFT plan ♻️");
        return;
    }
    
    // Освободить старый план если есть
    if (batch_plan_handle_ != 0) {
        clfftDestroyPlan(&batch_plan_handle_);
        batch_plan_handle_ = 0;
    }
    
    DRVGPU_LOG_INFO("Search3Module", "Creating batch FFT plan (batch_size=" + 
                    std::to_string(batch_size) + ")...");
    
    clfftStatus status;
    size_t clLengths[1] = { nFFT_ };
    
    status = clfftCreateDefaultPlan(&batch_plan_handle_, context_, CLFFT_1D, clLengths);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("Search3Module: clfftCreateDefaultPlan (batch) failed");
    }
    
    status = clfftSetPlanPrecision(batch_plan_handle_, CLFFT_SINGLE);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&batch_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetPlanPrecision (batch) failed");
    }
    
    status = clfftSetLayout(batch_plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&batch_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetLayout (batch) failed");
    }
    
    status = clfftSetResultLocation(batch_plan_handle_, CLFFT_OUTOFPLACE);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&batch_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetResultLocation (batch) failed");
    }
    
    size_t stride_in = nFFT_;
    size_t stride_out = nFFT_;
    size_t dist_in = nFFT_;
    size_t dist_out = nFFT_;
    
    status = clfftSetPlanBatchSize(batch_plan_handle_, batch_size);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&batch_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetPlanBatchSize (batch) failed");
    }
    
    status = clfftSetPlanInStride(batch_plan_handle_, CLFFT_1D, &stride_in);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&batch_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetPlanInStride (batch) failed");
    }
    
    status = clfftSetPlanOutStride(batch_plan_handle_, CLFFT_1D, &stride_out);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&batch_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetPlanOutStride (batch) failed");
    }
    
    status = clfftSetPlanDistance(batch_plan_handle_, dist_in, dist_out);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&batch_plan_handle_);
        throw std::runtime_error("Search3Module: clfftSetPlanDistance (batch) failed");
    }
    
    status = clfftBakePlan(batch_plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&batch_plan_handle_);
        throw std::runtime_error("Search3Module: clfftBakePlan (batch) failed");
    }
    
    batch_plan_beams_ = batch_size;
    DRVGPU_LOG_INFO("Search3Module", "Batch FFT plan created ✅");
}

// ════════════════════════════════════════════════════════════════════════════
// ReleaseFFTPlan
// ════════════════════════════════════════════════════════════════════════════

void Search3Module::ReleaseFFTPlan() {
    if (main_plan_handle_ != 0) {
        clfftDestroyPlan(&main_plan_handle_);
        main_plan_handle_ = 0;
    }
    
    if (batch_plan_handle_ != 0) {
        clfftDestroyPlan(&batch_plan_handle_);
        batch_plan_handle_ = 0;
        batch_plan_beams_ = 0;
    }
}

} // namespace search_3_
} // namespace drv_gpu_lib

// КОНЕЦ ЧАСТИ 2/4
// Следующая часть: ProcessNew, ProcessSingleBatch, ProcessMultiBatch
