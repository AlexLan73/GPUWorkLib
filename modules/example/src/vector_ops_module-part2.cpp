// ЧАСТЬ 2: Остальные операции (SubOne, AddVectors) и компиляция kernels

#include "vector_ops_module.hpp"
#include "common/logger.hpp"
#include <memory>
#include <cstddef>
#include <string>
#include <stdexcept>
#include <vector>

namespace drv_gpu_lib {

// Операции: Вычитание скаляра
void VectorOpsModule::SubOneOut(
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
    
    cl_int err;
    err = clSetKernelArg(kernel_sub_one_out_, 0, sizeof(cl_mem), &input_mem);
    err |= clSetKernelArg(kernel_sub_one_out_, 1, sizeof(cl_mem), &output_mem);
    err |= clSetKernelArg(kernel_sub_one_out_, 2, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::SubOneOut - Failed to set kernel args");
    }
    
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue_, kernel_sub_one_out_, 1, nullptr,
                                  &global_size, nullptr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::SubOneOut - Failed to enqueue kernel");
    }
    
    clFinish(queue_);
}

void VectorOpsModule::SubOneInPlace(
    std::shared_ptr<GPUBuffer<float>> data,
    size_t size)
{
    if (!initialized_) {
        throw std::runtime_error("VectorOpsModule: not initialized");
    }
    
    cl_mem data_mem = static_cast<cl_mem>(data->GetPtr());
    int n = static_cast<int>(size);
    
    cl_int err;
    err = clSetKernelArg(kernel_sub_one_inplace_, 0, sizeof(cl_mem), &data_mem);
    err |= clSetKernelArg(kernel_sub_one_inplace_, 1, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::SubOneInPlace - Failed to set kernel args");
    }
    
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue_, kernel_sub_one_inplace_, 1, nullptr,
                                  &global_size, nullptr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::SubOneInPlace - Failed to enqueue kernel");
    }
    
    clFinish(queue_);
}

// Операции: Сложение векторов
void VectorOpsModule::AddVectorsOut(
    std::shared_ptr<GPUBuffer<float>> input_a,
    std::shared_ptr<GPUBuffer<float>> input_b,
    std::shared_ptr<GPUBuffer<float>> output,
    size_t size)
{
    if (!initialized_) {
        throw std::runtime_error("VectorOpsModule: not initialized");
    }
    
    cl_mem a_mem = static_cast<cl_mem>(input_a->GetPtr());
    cl_mem b_mem = static_cast<cl_mem>(input_b->GetPtr());
    cl_mem out_mem = static_cast<cl_mem>(output->GetPtr());
    int n = static_cast<int>(size);
    
    cl_int err;
    err = clSetKernelArg(kernel_add_vectors_out_, 0, sizeof(cl_mem), &a_mem);
    err |= clSetKernelArg(kernel_add_vectors_out_, 1, sizeof(cl_mem), &b_mem);
    err |= clSetKernelArg(kernel_add_vectors_out_, 2, sizeof(cl_mem), &out_mem);
    err |= clSetKernelArg(kernel_add_vectors_out_, 3, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::AddVectorsOut - Failed to set kernel args");
    }
    
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue_, kernel_add_vectors_out_, 1, nullptr,
                                  &global_size, nullptr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::AddVectorsOut - Failed to enqueue kernel");
    }
    
    clFinish(queue_);
}

void VectorOpsModule::AddVectorsInPlace(
    std::shared_ptr<GPUBuffer<float>> data_a,
    std::shared_ptr<GPUBuffer<float>> input_b,
    size_t size)
{
    if (!initialized_) {
        throw std::runtime_error("VectorOpsModule: not initialized");
    }
    
    cl_mem a_mem = static_cast<cl_mem>(data_a->GetPtr());
    cl_mem b_mem = static_cast<cl_mem>(input_b->GetPtr());
    int n = static_cast<int>(size);
    
    cl_int err;
    err = clSetKernelArg(kernel_add_vectors_inplace_, 0, sizeof(cl_mem), &a_mem);
    err |= clSetKernelArg(kernel_add_vectors_inplace_, 1, sizeof(cl_mem), &b_mem);
    err |= clSetKernelArg(kernel_add_vectors_inplace_, 2, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::AddVectorsInPlace - Failed to set kernel args");
    }
    
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue_, kernel_add_vectors_inplace_, 1, nullptr,
                                  &global_size, nullptr, 0, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("VectorOpsModule::AddVectorsInPlace - Failed to enqueue kernel");
    }
    
    clFinish(queue_);
}

// Компиляция kernels
void VectorOpsModule::CompileKernels() {
    DRVGPU_LOG_INFO("VectorOpsModule", "Loading kernel source...");
    
    std::string kernel_source = LoadKernelSource("vector_ops.cl");
    
    DRVGPU_LOG_DEBUG("VectorOpsModule", "Kernel source loaded (" + 
                     std::to_string(kernel_source.size()) + " bytes)");
    
    const char* source_ptr = kernel_source.c_str();
    size_t source_size = kernel_source.size();
    
    cl_int err;
    program_ = clCreateProgramWithSource(context_, 1, &source_ptr, &source_size, &err);
    
    if (err != CL_SUCCESS || !program_) {
        throw std::runtime_error("VectorOpsModule: Failed to create program");
    }
    
    DRVGPU_LOG_INFO("VectorOpsModule", "Compiling kernels...");
    
    err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        
        DRVGPU_LOG_ERROR("VectorOpsModule", "Kernel compilation failed:");
        DRVGPU_LOG_ERROR("VectorOpsModule", std::string(log.data()));
        
        clReleaseProgram(program_);
        program_ = nullptr;
        
        throw std::runtime_error("VectorOpsModule: Kernel compilation failed");
    }
    
    DRVGPU_LOG_INFO("VectorOpsModule", "Kernels compiled successfully ✅");
}

} // namespace drv_gpu_lib

// См. часть 3 для CreateKernelObjects, ReleaseKernels, LoadKernelSource
