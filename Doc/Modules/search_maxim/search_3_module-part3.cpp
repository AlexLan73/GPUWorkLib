/**
 * @file antenna_module.cpp - ЧАСТЬ 3/4
 * @brief ProcessNew, ProcessSingleBatch
 */

// ════════════════════════════════════════════════════════════════════════════
// ProcessNew - обёртки для разных типов буферов
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTResult AntennaModule::ProcessNew(
    std::shared_ptr<SVMBuffer<std::complex<float>>> input_signal) 
{
    if (!input_signal) {
        throw std::invalid_argument("AntennaModule::ProcessNew: null SVM buffer");
    }
    
    cl_mem native_mem = input_signal->GetNativeHandle();
    return ProcessNew(native_mem);
}

AntennaFFTResult AntennaModule::ProcessNew(
    std::shared_ptr<GPUBuffer<std::complex<float>>> input_signal) 
{
    if (!input_signal) {
        throw std::invalid_argument("AntennaModule::ProcessNew: null GPU buffer");
    }
    
    cl_mem native_mem = input_signal->GetNativeHandle();
    return ProcessNew(native_mem);
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessNew - ГЛАВНАЯ ФУНКЦИЯ
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTResult AntennaModule::ProcessNew(cl_mem input_signal) {
    if (!initialized_) {
        throw std::runtime_error("AntennaModule::ProcessNew: not initialized");
    }
    
    if (!input_signal) {
        throw std::invalid_argument("AntennaModule::ProcessNew: null input signal");
    }
    
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "  AntennaModule::ProcessNew() - Automatic Strategy Selection\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";
    
    // Шаг 1: Оценить требуемую память
    size_t required_memory = EstimateRequiredMemory();
    
    // Шаг 2: Проверить доступную память
    bool memory_ok = CheckAvailableMemory(required_memory);
    
    // Шаг 3: Выбрать стратегию
    AntennaFFTResult result;
    
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

AntennaFFTResult AntennaModule::ProcessSingleBatch(cl_mem input_signal) {
    DRVGPU_LOG_INFO("AntennaModule", "ProcessSingleBatch: start");
    
    size_t num_beams = params_.beam_count;
    
    // ═══ 1. Создать/переиспользовать FFT план ═══
    CreateOrReuseFFTPlan();
    
    // ═══ 2. Создать/переиспользовать буферы ═══
    size_t fft_buffer_size = num_beams * nFFT_;
    
    if (!buffer_fft_input_ || buffer_fft_input_->GetSize() != fft_buffer_size) {
        auto& mem_mgr = backend_->GetMemoryManager();
        buffer_fft_input_ = mem_mgr.CreateGPUBuffer<std::complex<float>>(fft_buffer_size);
        DRVGPU_LOG_INFO("AntennaModule", "Created fft_input buffer");
    }
    
    if (!buffer_fft_output_ || buffer_fft_output_->GetSize() != fft_buffer_size) {
        auto& mem_mgr = backend_->GetMemoryManager();
        buffer_fft_output_ = mem_mgr.CreateGPUBuffer<std::complex<float>>(fft_buffer_size);
        DRVGPU_LOG_INFO("AntennaModule", "Created fft_output buffer");
    }
    
    // ═══ 3. Padding kernel ═══
    cl_mem fft_input_mem = buffer_fft_input_->GetNativeHandle();
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
        throw std::runtime_error("AntennaModule: padding_kernel failed");
    }
    
    // ═══ 4. FFT transform ═══
    cl_mem fft_output_mem = buffer_fft_output_->GetNativeHandle();
    clfftStatus status = clfftEnqueueTransform(
        main_plan_handle_,
        CLFFT_FORWARD,
        1, &queue_,
        0, nullptr, nullptr,
        &fft_input_mem, &fft_output_mem,
        nullptr);
    
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("AntennaModule: clfftEnqueueTransform failed");
    }
    
    // ═══ 5. Post kernel (magnitude + select) ═══
    // Создать временные буферы для selected данных
    size_t selected_size = num_beams * params_.out_count_points_fft;
    auto& mem_mgr = backend_->GetMemoryManager();
    auto buffer_selected_complex = mem_mgr.CreateGPUBuffer<std::complex<float>>(selected_size);
    auto buffer_selected_magnitude = mem_mgr.CreateGPUBuffer<float>(selected_size);
    
    cl_mem selected_complex_mem = buffer_selected_complex->GetNativeHandle();
    cl_mem selected_magnitude_mem = buffer_selected_magnitude->GetNativeHandle();
    
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
        throw std::runtime_error("AntennaModule: post_kernel failed");
    }
    
    // ═══ 6. Reduction kernel (поиск максимумов) ═══
    std::vector<BeamFFTResult> results = FindMaximaOnGPU(num_beams);
    
    // ═══ 7. Собрать результат ═══
    AntennaFFTResult final_result(num_beams, nFFT_, params_.task_id, params_.module_name);
    final_result.results = std::move(results);
    
    DRVGPU_LOG_INFO("AntennaModule", "ProcessSingleBatch: complete");
    return final_result;
}

// КОНЕЦ ЧАСТИ 3/4
// Следующая часть: ProcessMultiBatch, ProcessBatch, FindMaximaOnGPU
