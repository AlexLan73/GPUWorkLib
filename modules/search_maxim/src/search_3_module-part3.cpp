/**
 * @file search_3_module.cpp - ЧАСТЬ 3/4
 * @brief ProcessNew, ProcessSingleBatch
 */

#include "search_3_module.hpp"
#include "memory/memory_manager.hpp"
#include "common/logger.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace drv_gpu_lib {
namespace search_3_ {

// ════════════════════════════════════════════════════════════════════════════
// ProcessNew - обёртки для разных типов буферов
// ════════════════════════════════════════════════════════════════════════════

Search3FFTResult Search3Module::ProcessNew(
    std::shared_ptr<GPUBuffer<std::complex<float>>> input_signal) 
{
    if (!input_signal) {
        throw std::invalid_argument("Search3Module::ProcessNew: null GPU buffer");
    }
    
    cl_mem native_mem = static_cast<cl_mem>(input_signal->GetPtr());
    return ProcessNew(native_mem);
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessNew - ГЛАВНАЯ ФУНКЦИЯ
// ════════════════════════════════════════════════════════════════════════════

Search3FFTResult Search3Module::ProcessNew(cl_mem input_signal) {
    if (!initialized_) {
        throw std::runtime_error("Search3Module::ProcessNew: not initialized");
    }
    
    if (!input_signal) {
        throw std::invalid_argument("Search3Module::ProcessNew: null input signal");
    }
    
    // Проверка: буфер должен принадлежать нашему OpenCL контексту
    cl_context buf_context = nullptr;
    cl_int err = clGetMemObjectInfo(input_signal, CL_MEM_CONTEXT, sizeof(cl_context), &buf_context, nullptr);
    if (err != CL_SUCCESS || buf_context != context_) {
        throw std::invalid_argument("Search3Module::ProcessNew: input buffer from different OpenCL context");
    }
    
    // Проверка размера: нужно beam_count * count_points * sizeof(complex<float>)
    size_t required_size = params_.beam_count * params_.count_points * sizeof(std::complex<float>);
    size_t buf_size = 0;
    err = clGetMemObjectInfo(input_signal, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
    if (err != CL_SUCCESS || buf_size < required_size) {
        throw std::invalid_argument("Search3Module::ProcessNew: input buffer too small (need " +
            std::to_string(required_size) + " bytes, got " + std::to_string(buf_size) + ")");
    }
    
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Search3Module::ProcessNew() - Automatic Strategy Selection\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";
    
    // Шаг 1: Оценить требуемую память
    size_t required_memory = EstimateRequiredMemory();
    
    // Шаг 2: Проверить доступную память
    bool memory_ok = CheckAvailableMemory(required_memory);
    
    // Шаг 3: Выбрать стратегию
    Search3FFTResult result;
    
    if (memory_ok) {
        std::cout << "  ✅ STRATEGY: SINGLE-BATCH (full processing)\n";
        std::cout << "  All beams will be processed in one pass.\n\n";
        result = ProcessSingleBatch(input_signal);
    } else {
        std::cout << "  ⚠️  STRATEGY: MULTI-BATCH (batch processing)\n";
        std::cout << "  Beams will be split into batches.\n\n";
        result = ProcessMultiBatch(input_signal);
    }
    
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  ProcessNew() complete ✅\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";
    
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessSingleBatch
// ════════════════════════════════════════════════════════════════════════════

Search3FFTResult Search3Module::ProcessSingleBatch(cl_mem input_signal) {
    DRVGPU_LOG_INFO("Search3Module", "ProcessSingleBatch: start");
    
    size_t num_beams = params_.beam_count;
    
    // ═══ 1. Создать/переиспользовать FFT план ═══
    CreateOrReuseFFTPlan();
    
    // ═══ 2. Создать/переиспользовать буферы ═══
    size_t fft_buffer_size = num_beams * nFFT_;
    
    if (!buffer_fft_input_ || buffer_fft_input_->GetNumElements() != fft_buffer_size) {
        auto* mem_mgr = backend_->GetMemoryManager();
        if (!mem_mgr) throw std::runtime_error("Search3Module: backend has no MemoryManager");
        auto& mem_mgr_ref = *mem_mgr;
        buffer_fft_input_ = mem_mgr_ref.CreateBuffer<std::complex<float>>(fft_buffer_size);
        DRVGPU_LOG_INFO("Search3Module", "Created fft_input buffer");
    }
    
    if (!buffer_fft_output_ || buffer_fft_output_->GetNumElements() != fft_buffer_size) {
        auto* mem_mgr = backend_->GetMemoryManager();
        if (!mem_mgr) throw std::runtime_error("Search3Module: backend has no MemoryManager");
        auto& mem_mgr_ref = *mem_mgr;
        buffer_fft_output_ = mem_mgr_ref.CreateBuffer<std::complex<float>>(fft_buffer_size);
        DRVGPU_LOG_INFO("Search3Module", "Created fft_output buffer");
    }
    
    // ═══ 3. Padding kernel ═══
    cl_mem fft_input_mem = static_cast<cl_mem>(buffer_fft_input_->GetPtr());
    cl_uint beam_offset = 0; // Полная обработка - offset = 0
    
    clSetKernelArg(padding_kernel_, 0, sizeof(cl_mem), &input_signal);
    clSetKernelArg(padding_kernel_, 1, sizeof(cl_mem), &fft_input_mem);
    clSetKernelArg(padding_kernel_, 2, sizeof(cl_uint), &beam_offset);
    clSetKernelArg(padding_kernel_, 3, sizeof(cl_uint), &params_.count_points);
    clSetKernelArg(padding_kernel_, 4, sizeof(cl_uint), &nFFT_);
    clSetKernelArg(padding_kernel_, 5, sizeof(cl_uint), &num_beams);
    
    size_t global_work[2] = { nFFT_, num_beams };
    cl_int err = clEnqueueNDRangeKernel(queue_, padding_kernel_, 2, nullptr, global_work, 
                                        nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Search3Module: padding_kernel failed");
    }
    
    // ═══ 4. FFT transform ═══
    cl_mem fft_output_mem = static_cast<cl_mem>(buffer_fft_output_->GetPtr());
    clfftStatus status = clfftEnqueueTransform(
        main_plan_handle_,
        CLFFT_FORWARD,
        1, &queue_,
        0, nullptr, nullptr,
        &fft_input_mem, &fft_output_mem,
        nullptr);
    
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("Search3Module: clfftEnqueueTransform failed");
    }
    
    // ═══ 5. Post kernel (magnitude + select) ═══
    // Создать временные буферы для selected данных
    size_t selected_size = num_beams * params_.out_count_points_fft;
    auto* mem_mgr = backend_->GetMemoryManager();
        if (!mem_mgr) throw std::runtime_error("Search3Module: backend has no MemoryManager");
        auto& mem_mgr_ref = *mem_mgr;
    auto buffer_selected_complex = mem_mgr_ref.CreateBuffer<std::complex<float>>(selected_size);
    auto buffer_selected_magnitude = mem_mgr_ref.CreateBuffer<float>(selected_size);
    
    cl_mem selected_complex_mem = static_cast<cl_mem>(buffer_selected_complex->GetPtr());
    cl_mem selected_magnitude_mem = static_cast<cl_mem>(buffer_selected_magnitude->GetPtr());
    
    cl_uint out_count = static_cast<cl_uint>(params_.out_count_points_fft);
    cl_uint fft_size = static_cast<cl_uint>(nFFT_);
    
    clSetKernelArg(post_kernel_, 0, sizeof(cl_mem), &fft_output_mem);
    clSetKernelArg(post_kernel_, 1, sizeof(cl_mem), &selected_complex_mem);
    clSetKernelArg(post_kernel_, 2, sizeof(cl_mem), &selected_magnitude_mem);
    clSetKernelArg(post_kernel_, 3, sizeof(cl_uint), &out_count);
    clSetKernelArg(post_kernel_, 4, sizeof(cl_uint), &fft_size);
    clSetKernelArg(post_kernel_, 5, sizeof(cl_uint), &num_beams);
    
    size_t global_post[2] = { params_.out_count_points_fft, num_beams };
    err = clEnqueueNDRangeKernel(queue_, post_kernel_, 2, nullptr, global_post,
                                 nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Search3Module: post_kernel failed");
    }
    
    // ═══ 6. Reduction kernel (поиск максимумов) ═══
    std::vector<BeamFFTResult> results = FindMaximaOnGPU(num_beams);
    
    // ═══ 7. Собрать результат ═══
    Search3FFTResult final_result(num_beams, nFFT_, params_.task_id, params_.module_name);
    final_result.results = std::move(results);
    
    DRVGPU_LOG_INFO("Search3Module", "ProcessSingleBatch: complete");
    return final_result;
}

} // namespace search_3_
} // namespace drv_gpu_lib

// КОНЕЦ ЧАСТИ 3/4
// Следующая часть: ProcessMultiBatch, ProcessBatch, FindMaximaOnGPU
