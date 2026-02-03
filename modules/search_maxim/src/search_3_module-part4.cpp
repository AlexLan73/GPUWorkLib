/**
 * @file search_3_module.cpp - ЧАСТЬ 4/4
 * @brief ProcessMultiBatch, ProcessBatch, FindMaximaOnGPU
 */

#include "search_3_module.hpp"
#include "memory/memory_manager.hpp"
#include "common/logger.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace drv_gpu_lib {
namespace search_3_ {

// ════════════════════════════════════════════════════════════════════════════
// ProcessMultiBatch
// ════════════════════════════════════════════════════════════════════════════

Search3FFTResult Search3Module::ProcessMultiBatch(cl_mem input_signal) {
    DRVGPU_LOG_INFO("Search3Module", "ProcessMultiBatch: start");
    
    size_t total_beams = params_.beam_count;
    size_t batch_size = CalculateBatchSize(total_beams);
    
    std::cout << "  Batch Configuration:\n";
    printf("    - Total beams: %zu\n", total_beams);
    printf("    - Batch size: %zu (%.1f%%)\n", batch_size, 
           (batch_size * 100.0 / total_beams));
    printf("    - Estimated batches: %zu\n", (total_beams + batch_size - 1) / batch_size);
    std::cout << "\n";
    
    // Оптимизация: если последний батч слишком маленький (<3 лучей), 
    // добавить его в предпоследний
    size_t num_full_batches = total_beams / batch_size;
    size_t last_batch_size = total_beams % batch_size;
    
    if (last_batch_size > 0 && last_batch_size < 3 && num_full_batches > 0) {
        // Убрать один full batch и добавить его в last batch
        num_full_batches--;
        last_batch_size += batch_size;
        std::cout << "  Optimization: merged small last batch\n";
        printf("    - Adjusted last batch size: %zu\n\n", last_batch_size);
    }
    
    // Создать буферы для максимального размера батча
    size_t max_batch = std::max(batch_size, last_batch_size);
    
    if (!batch_fft_input_ || batch_buffers_size_ != max_batch) {
        size_t buffer_size = max_batch * nFFT_;
        auto* mem_mgr = backend_->GetMemoryManager();
        if (!mem_mgr) throw std::runtime_error("Search3Module: backend has no MemoryManager");
        auto& mem_mgr_ref = *mem_mgr;
        batch_fft_input_ = mem_mgr_ref.CreateBuffer<std::complex<float>>(buffer_size);
        batch_fft_output_ = mem_mgr_ref.CreateBuffer<std::complex<float>>(buffer_size);
        batch_buffers_size_ = max_batch;
        DRVGPU_LOG_INFO("Search3Module", "Created batch buffers (size=" + 
                        std::to_string(max_batch) + ")");
    }
    
    // Создать FFT план для батча
    CreateBatchFFTPlan(max_batch);
    
    // Обработать батчи
    std::vector<BeamFFTResult> all_results;
    all_results.reserve(total_beams);
    
    size_t current_beam = 0;
    size_t batch_index = 0;
    
    // Полные батчи
    for (size_t i = 0; i < num_full_batches; ++i) {
        std::cout << "  Processing batch " << (batch_index + 1) << "/" 
                  << (num_full_batches + (last_batch_size > 0 ? 1 : 0)) 
                  << " (beams " << current_beam << "-" << (current_beam + batch_size - 1) << ")...\n";
        
        auto batch_results = ProcessBatch(input_signal, current_beam, batch_size);
        all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
        
        current_beam += batch_size;
        batch_index++;
    }
    
    // Последний батч (если есть)
    if (last_batch_size > 0) {
        std::cout << "  Processing batch " << (batch_index + 1) << "/" 
                  << (num_full_batches + 1) 
                  << " (beams " << current_beam << "-" << (current_beam + last_batch_size - 1) << ")...\n";
        
        auto batch_results = ProcessBatch(input_signal, current_beam, last_batch_size);
        all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
    }
    
    std::cout << "\n";
    
    // Собрать результат
    Search3FFTResult final_result(total_beams, nFFT_, params_.task_id, params_.module_name);
    final_result.results = std::move(all_results);
    
    DRVGPU_LOG_INFO("Search3Module", "ProcessMultiBatch: complete");
    return final_result;
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessBatch
// ════════════════════════════════════════════════════════════════════════════

std::vector<BeamFFTResult> Search3Module::ProcessBatch(
    cl_mem input_signal, size_t start_beam, size_t num_beams) 
{
    // Пересоздать план если размер батча изменился
    if (batch_plan_beams_ != num_beams) {
        CreateBatchFFTPlan(num_beams);
    }
    
    // ═══ 1. Padding kernel (с offset!) ═══
    cl_mem fft_input_mem = static_cast<cl_mem>(batch_fft_input_->GetPtr());
    cl_uint beam_offset = static_cast<cl_uint>(start_beam);
    
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
        throw std::runtime_error("Search3Module::ProcessBatch: padding_kernel failed");
    }
    
    // ═══ 2. FFT transform ═══
    cl_mem fft_output_mem = static_cast<cl_mem>(batch_fft_output_->GetPtr());
    clfftStatus status = clfftEnqueueTransform(
        batch_plan_handle_,
        CLFFT_FORWARD,
        1, &queue_,
        0, nullptr, nullptr,
        &fft_input_mem, &fft_output_mem,
        nullptr);
    
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("Search3Module::ProcessBatch: clfftEnqueueTransform failed");
    }
    
    // ═══ 3. Post kernel ═══
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
    cl_uint beams_count = static_cast<cl_uint>(num_beams);
    
    clSetKernelArg(post_kernel_, 0, sizeof(cl_mem), &fft_output_mem);
    clSetKernelArg(post_kernel_, 1, sizeof(cl_mem), &selected_complex_mem);
    clSetKernelArg(post_kernel_, 2, sizeof(cl_mem), &selected_magnitude_mem);
    clSetKernelArg(post_kernel_, 3, sizeof(cl_uint), &out_count);
    clSetKernelArg(post_kernel_, 4, sizeof(cl_uint), &fft_size);
    clSetKernelArg(post_kernel_, 5, sizeof(cl_uint), &beams_count);
    
    size_t global_post[2] = { params_.out_count_points_fft, num_beams };
    err = clEnqueueNDRangeKernel(queue_, post_kernel_, 2, nullptr, global_post,
                                 nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Search3Module::ProcessBatch: post_kernel failed");
    }
    
    // ═══ 4. Reduction (поиск максимумов) ═══
    std::vector<BeamFFTResult> results = FindMaximaOnGPU(num_beams);
    
    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// FindMaximaOnGPU
// ════════════════════════════════════════════════════════════════════════════

std::vector<BeamFFTResult> Search3Module::FindMaximaOnGPU(size_t num_beams) {
    // Создать буфер для результатов (MaxValue структуры)
    // Структура: [beam0: max1, max2, max3] [beam1: max1, max2, max3] ...
    // Каждый MaxValue: index(uint), amplitude(float), phase(float), real(float), imag(float)
    // = 5 floats = 20 bytes (но alignment до 8, поэтому 32 bytes)
    
    size_t maxima_count = num_beams * params_.max_peaks_count;
    size_t result_buffer_floats = maxima_count * 8; // 8 floats per MaxValue (с padding)
    
    if (!buffer_maxima_ || buffer_maxima_->GetNumElements() != result_buffer_floats) {
        auto* mem_mgr = backend_->GetMemoryManager();
        if (!mem_mgr) throw std::runtime_error("Search3Module: backend has no MemoryManager");
        buffer_maxima_ = mem_mgr->CreateBuffer<float>(result_buffer_floats);
    }
    
    cl_mem maxima_mem = static_cast<cl_mem>(buffer_maxima_->GetPtr());
    
    // TODO: Здесь должен быть reduction kernel для поиска топ-N максимумов
    // Пока создаём заглушку - заполним нулями и вернём пустые результаты
    
    std::vector<float> zeros(result_buffer_floats, 0.0f);
    cl_int err = clEnqueueWriteBuffer(queue_, maxima_mem, CL_TRUE, 0,
                                      result_buffer_floats * sizeof(float),
                                      zeros.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Search3Module::FindMaximaOnGPU: failed to clear buffer");
    }
    
    // Читать результаты обратно
    std::vector<float> results_data(result_buffer_floats);
    err = clEnqueueReadBuffer(queue_, maxima_mem, CL_TRUE, 0,
                              result_buffer_floats * sizeof(float),
                              results_data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Search3Module::FindMaximaOnGPU: failed to read results");
    }
    
    // Парсить результаты
    std::vector<BeamFFTResult> beam_results(num_beams);
    
    for (size_t beam = 0; beam < num_beams; ++beam) {
        BeamFFTResult& beam_result = beam_results[beam];
        beam_result.max_values.reserve(params_.max_peaks_count);
        
        for (size_t peak = 0; peak < params_.max_peaks_count; ++peak) {
            size_t offset = (beam * params_.max_peaks_count + peak) * 8;
            
            FFTMaxValue max_val;
            max_val.index_point = static_cast<size_t>(results_data[offset + 0]);
            max_val.amplitude = results_data[offset + 1];
            max_val.phase = results_data[offset + 2];
            max_val.real = results_data[offset + 3];
            max_val.imag = results_data[offset + 4];
            
            beam_result.max_values.push_back(max_val);
        }
        
        // Параболическая интерполяция (упрощённая)
        if (!beam_result.max_values.empty()) {
            beam_result.freq_offset = 0.0f; // TODO: implement parabolic interpolation
            beam_result.refined_frequency = 0.0f; // TODO: calculate from offset
        }
    }
    
    return beam_results;
}

} // namespace search_3_
} // namespace drv_gpu_lib

// КОНЕЦ ЧАСТИ 4/4 - ФАЙЛ ЗАВЕРШЁН
