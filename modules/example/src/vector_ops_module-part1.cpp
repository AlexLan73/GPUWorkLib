#include "vector_ops_module.hpp"
#include "logger/logger.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

// Определено в CMakeLists.txt
#ifndef VECTOR_OPS_KERNELS_PATH
#define VECTOR_OPS_KERNELS_PATH "kernels"
#endif

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор и деструктор
// ════════════════════════════════════════════════════════════════════════════

VectorOpsModule::VectorOpsModule(IBackend* backend)
    : backend_(backend)
    , initialized_(false)
    , program_(nullptr)
    , kernel_add_one_out_(nullptr)
    , kernel_add_one_inplace_(nullptr)
    , kernel_sub_one_out_(nullptr)
    , kernel_sub_one_inplace_(nullptr)
    , kernel_add_vectors_out_(nullptr)
    , kernel_add_vectors_inplace_(nullptr)
    , context_(nullptr)
    , device_(nullptr)
    , queue_(nullptr)
{
    if (!backend_) {
        throw std::invalid_argument("VectorOpsModule: backend cannot be null");
    }
    
    DRVGPU_LOG_INFO("VectorOpsModule", "Created (not initialized)");
}

VectorOpsModule::~VectorOpsModule() {
    Cleanup();
}

// ════════════════════════════════════════════════════════════════════════════
// Жизненный цикл: Initialize
// ════════════════════════════════════════════════════════════════════════════

void VectorOpsModule::Initialize() {
    if (initialized_) {
        DRVGPU_LOG_WARNING("VectorOpsModule", "Already initialized");
        return;
    }
    
    DRVGPU_LOG_INFO("VectorOpsModule", "Initializing...");
    
    // Получаем нативные хэндлы из бэкенда
    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    device_ = static_cast<cl_device_id>(backend_->GetNativeDevice());
    queue_ = static_cast<cl_command_queue>(backend_->GetNativeQueue());
    
    if (!context_ || !device_ || !queue_) {
        throw std::runtime_error("VectorOpsModule: Invalid OpenCL handles from backend");
    }
    
    // Компилируем kernels
    CompileKernels();
    
    // Создаём kernel объекты
    CreateKernelObjects();
    
    initialized_ = true;
    DRVGPU_LOG_INFO("VectorOpsModule", "Initialized successfully ✅");
}

// ════════════════════════════════════════════════════════════════════════════
// Жизненный цикл: Cleanup
// ════════════════════════════════════════════════════════════════════════════

void VectorOpsModule::Cleanup() {
    if (!initialized_) {
        return;
    }
    
    DRVGPU_LOG_INFO("VectorOpsModule", "Cleanup...");
    
    ReleaseKernels();
    
    initialized_ = false;
    DRVGPU_LOG_INFO("VectorOpsModule", "Cleanup complete");
}

// ════════════════════════════════════════════════════════════════════════════
// Операции: Добавление скаляра
// ════════════════════════════════════════════════════════════════════════════

void VectorOpsModule::AddOneOut(
    std::shared_ptr<GPUBuffer<float>> input,
    std::shared_ptr<GPUBuffer<float>> output,
    size_t size)
{
    if (!initialized_) {
        throw std::runtime_error("VectorOpsModule: not initialized");
    }
    
    cl_mem input_mem = static_cast<cl_mem>(input->GetPtr());
    cl_mem output_mem = static_cast<cl_mem>(output->GetPtr());
    int n = static_cast<int>(size);
    
    // Устанавливаем аргументы kernel
    cl_int err;
    err = clSetKernelArg(kernel_add_one_out_, 0, sizeof(cl_mem), &input_mem);
    err |= clSetKernelArg(kernel_add_one_out_, 1, sizeof(cl_mem), &output_mem);
    err |= clSetKernelArg(kernel_add_one_out_, 2, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::AddOneOut - Failed to set kernel args");
    }
    
    // Запускаем kernel
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue_, kernel_add_one_out_, 1, nullptr,
                                  &global_size, nullptr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::AddOneOut - Failed to enqueue kernel");
    }
    
    clFinish(queue_); // Ждём завершения
}

void VectorOpsModule::AddOneInPlace(
    std::shared_ptr<GPUBuffer<float>> data,
    size_t size)
{
    if (!initialized_) {
        throw std::runtime_error("VectorOpsModule: not initialized");
    }
    
    cl_mem data_mem = static_cast<cl_mem>(data->GetPtr());
    int n = static_cast<int>(size);
    
    cl_int err;
    err = clSetKernelArg(kernel_add_one_inplace_, 0, sizeof(cl_mem), &data_mem);
    err |= clSetKernelArg(kernel_add_one_inplace_, 1, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::AddOneInPlace - Failed to set kernel args");
    }
    
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue_, kernel_add_one_inplace_, 1, nullptr,
                                  &global_size, nullptr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::AddOneInPlace - Failed to enqueue kernel");
    }
    
    clFinish(queue_);
}

// Продолжение следует в части 2...
} // namespace drv_gpu_lib
