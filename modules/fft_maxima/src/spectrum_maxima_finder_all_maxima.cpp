/**
 * @file spectrum_maxima_finder_all_maxima.cpp
 * @brief SpectrumMaximaFinder — поиск ВСЕХ локальных максимумов (AllMaxima pipeline)
 *
 * Отделено от spectrum_maxima_finder.cpp для читабельности.
 *
 * Содержит:
 *   - FindAllMaximaFromCPU()        — full pipeline: CPU data → FFT(pre+post) → AllMaxima
 *   - FindAllMaximaFromGPUPipeline() — full pipeline: GPU data → FFT(pre+post) → AllMaxima
 *   - CreateAllMaximaFFTPlan()      — FFT план с pre+post callbacks (magnitude)
 *   - ExecuteAllMaximaFFT()         — запуск FFT с AllMaxima планом
 *   - EnsureMagnitudesBuffer()      — создание/переиспользование magnitudes буфера
 *   - CompileAllMaximaKernels()     — компиляция OpenCL кернелов
 *   - ExecutePrefixSum()            — beam-aware parallel prefix sum (Blelloch)
 *   - FindAllMaxima(cl_mem)         — detect → scan → compact pipeline
 *   - ReleaseAllMaximaResources()   — освобождение ресурсов
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-14
 */

#include "spectrum_maxima_finder.h"
#include "services/console_output.hpp"
#include "kernels/all_maxima_kernel_sources.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// EnsureMagnitudesBuffer — создание/переиспользование буфера амплитуд
// ════════════════════════════════════════════════════════════════════════════

void SpectrumMaximaFinder::EnsureMagnitudesBuffer(size_t total_elements) {
    if (magnitudes_buffer_ && magnitudes_buffer_size_ >= total_elements) {
        return;  // Достаточно большой — переиспользуем
    }

    // Освобождаем старый
    if (magnitudes_buffer_) {
        clReleaseMemObject(magnitudes_buffer_);
        magnitudes_buffer_ = nullptr;
    }

    cl_int err;
    magnitudes_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
        total_elements * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("EnsureMagnitudesBuffer: alloc failed: " + std::to_string(err));
    }
    magnitudes_buffer_size_ = total_elements;
}

// ════════════════════════════════════════════════════════════════════════════
// CreateAllMaximaFFTPlan — FFT план с pre-callback + post-callback
// ════════════════════════════════════════════════════════════════════════════
//
// Pre-callback:  zero-padding (n_point → nFFT)
// Post-callback: вычисление |FFT[i]| → magnitudes_buffer_
//
// Это ОТДЕЛЬНЫЙ план от plan_handle_ (который используется для Process)

void SpectrumMaximaFinder::CreateAllMaximaFFTPlan(size_t batch_count) {
    // Если план уже создан с таким же batch size — переиспользуем
    if (allmax_plan_created_ && allmax_plan_batch_size_ == batch_count) {
        return;
    }

    // Уничтожить старый план
    if (allmax_plan_created_ && allmax_plan_handle_) {
        clfftDestroyPlan(&allmax_plan_handle_);
        allmax_plan_handle_ = 0;
        allmax_plan_created_ = false;
    }

    // Инициализация clFFT (если ещё не)
    if (!clfft_initialized_) {
        clfftSetupData setup;
        setup.major = clfftVersionMajor;
        setup.minor = clfftVersionMinor;
        setup.patch = clfftVersionPatch;
        setup.debugFlags = 0;
        clfftSetup(&setup);
        clfft_initialized_ = true;
    }

    // Буфер для амплитуд
    size_t total_fft_elements = batch_count * params_.nFFT;
    EnsureMagnitudesBuffer(total_fft_elements);

    // Создать план
    size_t dim = params_.nFFT;
    clfftStatus status = clfftCreateDefaultPlan(&allmax_plan_handle_, context_, CLFFT_1D, &dim);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("CreateAllMaximaFFTPlan: clfftCreateDefaultPlan failed: " +
                                  std::to_string(status));
    }

    clfftSetPlanPrecision(allmax_plan_handle_, CLFFT_SINGLE);
    clfftSetLayout(allmax_plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(allmax_plan_handle_, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(allmax_plan_handle_, batch_count);

    size_t strides[1] = {1};
    size_t dist = params_.nFFT;
    clfftSetPlanInStride(allmax_plan_handle_, CLFFT_1D, strides);
    clfftSetPlanOutStride(allmax_plan_handle_, CLFFT_1D, strides);
    clfftSetPlanDistance(allmax_plan_handle_, dist, dist);

    // 1. Регистрируем pre-callback (zero-padding)
    const char* pre_source = kernels::GetPreCallbackSource32_opencl();
    status = clfftSetPlanCallback(allmax_plan_handle_, "prepareDataPre", pre_source, 0,
                                   PRECALLBACK, &pre_callback_userdata_, 1);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&allmax_plan_handle_);
        throw std::runtime_error("CreateAllMaximaFFTPlan: pre-callback failed: " +
                                  std::to_string(status));
    }

    // 2. Регистрируем post-callback (magnitude → magnitudes_buffer_)
    const char* post_source = kernels::GetPostCallbackMagnitudeSource_opencl();
    status = clfftSetPlanCallback(allmax_plan_handle_, "computeMagnitudePost", post_source, 0,
                                   POSTCALLBACK, &magnitudes_buffer_, 1);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&allmax_plan_handle_);
        throw std::runtime_error("CreateAllMaximaFFTPlan: post-callback failed: " +
                                  std::to_string(status));
    }

    // 3. Bake план
    status = clfftBakePlan(allmax_plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&allmax_plan_handle_);
        throw std::runtime_error("CreateAllMaximaFFTPlan: bake failed: " +
                                  std::to_string(status));
    }

    allmax_plan_batch_size_ = batch_count;
    allmax_plan_created_ = true;

    drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "AllMaxima",
        "FFT plan created with pre+post callbacks (batch=" + std::to_string(batch_count) + ")");
}

// ════════════════════════════════════════════════════════════════════════════
// ExecuteAllMaximaFFT — запуск FFT с AllMaxima планом
// ════════════════════════════════════════════════════════════════════════════

cl_event SpectrumMaximaFinder::ExecuteAllMaximaFFT(cl_event wait_event) {
    cl_event event = nullptr;

    clfftStatus status = clfftEnqueueTransform(
        allmax_plan_handle_,
        CLFFT_FORWARD,
        1, &queue_,
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &event,
        &fft_input_,       // Input (pre-callback читает из userdata)
        &fft_output_,      // Output (post-callback пишет complex + magnitude)
        fft_temp_buffer_   // Temp buffer
    );

    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("ExecuteAllMaximaFFT: clfftEnqueueTransform failed: " +
                                  std::to_string(status));
    }

    return event;
}

// ════════════════════════════════════════════════════════════════════════════
// FindAllMaximaFromCPU — full pipeline: CPU data → FFT(pre+post) → AllMaxima
// ════════════════════════════════════════════════════════════════════════════

AllMaximaResult SpectrumMaximaFinder::FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>& data,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ProfEvents* prof_events)
{
    if (!initialized_) {
        throw std::runtime_error("FindAllMaximaFromCPU: not initialized");
    }

    size_t expected_size = static_cast<size_t>(params_.antenna_count) * params_.n_point;
    if (data.size() != expected_size) {
        throw std::invalid_argument(
            "FindAllMaximaFromCPU: input size mismatch. Expected " +
            std::to_string(expected_size) + ", got " + std::to_string(data.size()));
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima", "FindAllMaximaFromCPU: " +
        std::to_string(params_.antenna_count) + " beams, nFFT=" +
        std::to_string(params_.nFFT));

    // Проверка batch для FFT буферов
    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    // Проверить, помещаются ли все данные
    if (drv_gpu_lib::BatchManager::AllItemsFit(backend_, params_.antenna_count,
                                                 bytes_per_antenna, params_.memory_limit)) {
        // Все помещаются — обрабатываем за один раз
        con.Print(0, "AllMaxima", "All " + std::to_string(params_.antenna_count) +
            " beams fit in memory — no batch needed");

        ReallocateBuffersForBatch(params_.antenna_count);
        actual_batch_size_ = params_.antenna_count;
        CreateAllMaximaFFTPlan(params_.antenna_count);

        cl_event upload_event = UploadData(data);
        cl_event fft_event = ExecuteAllMaximaFFT(upload_event);
        CollectOrRelease(upload_event, "Upload", prof_events);

        AllMaximaResult result = FindAllMaxima(
            fft_output_, params_.antenna_count, params_.nFFT,
            params_.sample_rate, dest, search_start, search_end,
            0, nullptr, nullptr, prof_events);

        CollectOrRelease(fft_event, "FFT+PostCallback", prof_events);
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BATCH PROCESSING — данные не влезают целиком
    // ═══════════════════════════════════════════════════════════════════════

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

    con.Print(0, "AllMaxima", "FindAllMaximaFromCPU: Batch processing — " +
        std::to_string(params_.antenna_count) + " beams split into batches of " +
        std::to_string(optimal_batch));

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(params_.antenna_count, optimal_batch);

    AllMaximaResult combined;
    combined.destination = dest;
    combined.total_maxima = 0;

    // Аллоцируем выходные буферы на ВСЕ antenna_count (для dest=GPU)
    cl_mem combined_out_maxima = nullptr;
    cl_mem combined_out_counts = nullptr;

    if (dest == OutputDestination::GPU || dest == OutputDestination::ALL) {
        cl_int err;
        uint32_t max_output_per_beam = std::min(
            (search_end > search_start ? (search_end - search_start) / 2 : params_.nFFT / 4),
            static_cast<uint32_t>(params_.max_maxima_per_beam)
        );

        combined_out_maxima = clCreateBuffer(context_, CL_MEM_READ_WRITE,
            params_.antenna_count * max_output_per_beam * sizeof(MaxValue), nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("FindAllMaximaFromCPU(batch): combined_out_maxima alloc failed");

        combined_out_counts = clCreateBuffer(context_, CL_MEM_READ_WRITE,
            params_.antenna_count * sizeof(uint32_t), nullptr, &err);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(combined_out_maxima);
            throw std::runtime_error("FindAllMaximaFromCPU(batch): combined_out_counts alloc failed");
        }
    }

    // Обрабатываем каждый batch
    for (const auto& batch : batches) {
        con.Print(0, "AllMaxima", "  Processing batch: beams [" +
            std::to_string(batch.start) + ".." + std::to_string(batch.start + batch.count - 1) + "]");

        // Подготовить данные для batch'а
        std::vector<std::complex<float>> batch_data(batch.count * params_.n_point);
        std::copy_n(data.begin() + batch.start * params_.n_point,
                    batch.count * params_.n_point,
                    batch_data.begin());

        // Переконфигурировать буферы и FFT план
        ReallocateBuffersForBatch(batch.count);
        actual_batch_size_ = batch.count;
        CreateAllMaximaFFTPlan(batch.count);

        cl_event upload_event = UploadData(batch_data);
        cl_event fft_event = ExecuteAllMaximaFFT(upload_event);
        CollectOrRelease(upload_event, "Upload", prof_events);

        // FindAllMaxima с beam_offset и внешними буферами (для Dest=GPU)
        AllMaximaResult batch_result = FindAllMaxima(
            fft_output_, batch.count, params_.nFFT,
            params_.sample_rate, dest, search_start, search_end,
            static_cast<uint32_t>(batch.start),  // beam_offset
            combined_out_maxima,                  // external buffers (nullptr для CPU)
            combined_out_counts, prof_events);

        CollectOrRelease(fft_event, "FFT", prof_events);

        // Мерж результатов
        for (auto& beam : batch_result.beams) {
            beam.antenna_id += static_cast<uint32_t>(batch.start);  // Корректируем antenna_id
        }

        combined.beams.insert(combined.beams.end(),
            batch_result.beams.begin(), batch_result.beams.end());
        combined.total_maxima += batch_result.total_maxima;
    }

    // Передаём владение GPU-буферами caller'у (не освобождаем!)
    if (dest == OutputDestination::GPU || dest == OutputDestination::ALL) {
        combined.gpu_maxima = combined_out_maxima;
        combined.gpu_counts = combined_out_counts;
        uint32_t max_per_beam = std::min(
            (search_end > search_start ? (search_end - search_start) / 2 : params_.nFFT / 4),
            static_cast<uint32_t>(params_.max_maxima_per_beam));
        combined.gpu_bytes = params_.antenna_count * max_per_beam * sizeof(MaxValue);
    }

    return combined;
}

// ════════════════════════════════════════════════════════════════════════════
// FindAllMaximaFromGPUPipeline — full pipeline: GPU data → FFT(pre+post) → AllMaxima
// ════════════════════════════════════════════════════════════════════════════

AllMaximaResult SpectrumMaximaFinder::FindAllMaximaFromGPUPipeline(
    cl_mem gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ProfEvents* prof_events)
{
    if (!gpu_data) {
        throw std::invalid_argument("FindAllMaximaFromGPUPipeline: gpu_data cannot be null");
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima", "FindAllMaximaFromGPUPipeline: " +
        std::to_string(antenna_count) + " beams, n_point=" + std::to_string(n_point));

    // Lazy init FFT params
    if (params_.nFFT == 0) {
        CalculateFFTSize();
    }

    size_t bytes_per_antenna = CalculateBytesPerAntenna();
    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : antenna_count * n_point * sizeof(std::complex<float>);

    // Оптимальный batch size с учётом внешних данных
    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, antenna_count, bytes_per_antenna,
        params_.memory_limit, external_memory);

    if (optimal_batch >= antenna_count) {
        // Все помещаются — обрабатываем за один раз
        con.Print(0, "AllMaxima", "All " + std::to_string(antenna_count) +
            " beams fit in memory — no batch needed");

        ReallocateBuffersForBatch(antenna_count);
        actual_batch_size_ = antenna_count;
        CreateAllMaximaFFTPlan(antenna_count);

        size_t data_size = antenna_count * n_point * sizeof(std::complex<float>);
        cl_event copy_event = nullptr;
        cl_int err = clEnqueueCopyBuffer(queue_,
            gpu_data, pre_callback_userdata_,
            0, PRE_CALLBACK_HEADER_SIZE,
            data_size,
            0, nullptr, &copy_event);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("FindAllMaximaFromGPUPipeline: CopyBuffer failed");
        }

        cl_event fft_event = ExecuteAllMaximaFFT(copy_event);
        CollectOrRelease(copy_event, "GPU→GPU Copy", prof_events);

        AllMaximaResult result = FindAllMaxima(
            fft_output_, static_cast<uint32_t>(antenna_count), params_.nFFT,
            params_.sample_rate, dest, search_start, search_end,
            0, nullptr, nullptr, prof_events);

        CollectOrRelease(fft_event, "FFT+PostCallback", prof_events);
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BATCH PROCESSING — данные не влезают целиком
    // ═══════════════════════════════════════════════════════════════════════

    con.Print(0, "AllMaxima", "FindAllMaximaFromGPUPipeline: Batch processing — " +
        std::to_string(antenna_count) + " beams split into batches of " +
        std::to_string(optimal_batch));

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(antenna_count, optimal_batch);

    AllMaximaResult combined;
    combined.destination = dest;
    combined.total_maxima = 0;

    // Аллоцируем выходные буферы на ВСЕ antenna_count (для dest=GPU)
    cl_mem combined_out_maxima = nullptr;
    cl_mem combined_out_counts = nullptr;

    if (dest == OutputDestination::GPU || dest == OutputDestination::ALL) {
        cl_int err;
        uint32_t max_output_per_beam = std::min(
            (search_end > search_start ? (search_end - search_start) / 2 : params_.nFFT / 4),
            static_cast<uint32_t>(params_.max_maxima_per_beam)
        );

        combined_out_maxima = clCreateBuffer(context_, CL_MEM_READ_WRITE,
            antenna_count * max_output_per_beam * sizeof(MaxValue), nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("FindAllMaximaFromGPUPipeline(batch): combined_out_maxima alloc failed");

        combined_out_counts = clCreateBuffer(context_, CL_MEM_READ_WRITE,
            antenna_count * sizeof(uint32_t), nullptr, &err);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(combined_out_maxima);
            throw std::runtime_error("FindAllMaximaFromGPUPipeline(batch): combined_out_counts alloc failed");
        }
    }

    // Обрабатываем каждый batch
    for (const auto& batch : batches) {
        con.Print(0, "AllMaxima", "  Processing batch: beams [" +
            std::to_string(batch.start) + ".." + std::to_string(batch.start + batch.count - 1) + "]");

        ReallocateBuffersForBatch(batch.count);
        actual_batch_size_ = batch.count;
        CreateAllMaximaFFTPlan(batch.count);

        // GPU→GPU copy с offset
        size_t src_offset = batch.start * n_point * sizeof(std::complex<float>);
        size_t data_size = batch.count * n_point * sizeof(std::complex<float>);
        cl_event copy_event = nullptr;
        cl_int err = clEnqueueCopyBuffer(queue_,
            gpu_data, pre_callback_userdata_,
            src_offset, PRE_CALLBACK_HEADER_SIZE,
            data_size,
            0, nullptr, &copy_event);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("FindAllMaximaFromGPUPipeline(batch): CopyBuffer failed");
        }

        cl_event fft_event = ExecuteAllMaximaFFT(copy_event);
        CollectOrRelease(copy_event, "GPU→GPU Copy", prof_events);

        // Detect → Scan → Compact с beam_offset и внешними буферами
        AllMaximaResult batch_result = FindAllMaxima(
            fft_output_, static_cast<uint32_t>(batch.count), params_.nFFT,
            params_.sample_rate, dest, search_start, search_end,
            static_cast<uint32_t>(batch.start),  // beam_offset
            combined_out_maxima,                  // external buffers (nullptr для CPU)
            combined_out_counts, prof_events);

        CollectOrRelease(fft_event, "FFT", prof_events);

        // Мерж результатов
        for (auto& beam : batch_result.beams) {
            beam.antenna_id += static_cast<uint32_t>(batch.start);
        }

        combined.beams.insert(combined.beams.end(),
            batch_result.beams.begin(), batch_result.beams.end());
        combined.total_maxima += batch_result.total_maxima;
    }

    // Передаём владение GPU-буферами caller'у (не освобождаем!)
    if (dest == OutputDestination::GPU || dest == OutputDestination::ALL) {
        combined.gpu_maxima = combined_out_maxima;
        combined.gpu_counts = combined_out_counts;
        uint32_t max_per_beam = std::min(
            (search_end > search_start ? (search_end - search_start) / 2 : params_.nFFT / 4),
            static_cast<uint32_t>(params_.max_maxima_per_beam));
        combined.gpu_bytes = antenna_count * max_per_beam * sizeof(MaxValue);
    }

    return combined;
}

// ════════════════════════════════════════════════════════════════════════════
// AllMaximaFromCPU — upload готовых FFT данных → Detect → Scan → Compact
// (БЕЗ FFT! Данные уже FFT'd)
// ════════════════════════════════════════════════════════════════════════════

AllMaximaResult SpectrumMaximaFinder::AllMaximaFromCPU(
    const std::vector<std::complex<float>>& fft_data,
    uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ProfEvents* prof_events)
{
    size_t expected_size = static_cast<size_t>(beam_count) * nFFT;
    if (fft_data.size() != expected_size) {
        throw std::invalid_argument(
            "AllMaximaFromCPU: size mismatch. Expected " +
            std::to_string(expected_size) + ", got " + std::to_string(fft_data.size()));
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima", "AllMaximaFromCPU: " +
        std::to_string(beam_count) + " beams, nFFT=" + std::to_string(nFFT));

    // Upload FFT данных на GPU
    cl_int err;
    size_t data_bytes = expected_size * sizeof(std::complex<float>);

    cl_mem gpu_fft = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        data_bytes, const_cast<std::complex<float>*>(fft_data.data()), &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("AllMaximaFromCPU: clCreateBuffer failed: " + std::to_string(err));

    // Вызываем пайплайн для GPU FFT данных
    AllMaximaResult result = FindAllMaxima(gpu_fft, beam_count, nFFT,
                                            sample_rate, dest, search_start, search_end,
                                            0, nullptr, nullptr, prof_events);

    clReleaseMemObject(gpu_fft);
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// CompileAllMaximaKernels — компиляция OpenCL кернелов
// ════════════════════════════════════════════════════════════════════════════

void SpectrumMaximaFinder::CompileAllMaximaKernels() {
    if (all_maxima_kernels_compiled_) return;

    cl_int err;

    // 1. Detection + ComputeMagnitudes kernels (в одной программе)
    {
        // Объединяем два исходника в одну программу
        const char* detect_src = kernels::GetDetectAllMaximaKernelSource_opencl();
        const char* compute_mag_src = kernels::GetComputeMagnitudesKernelSource_opencl();

        // Конкатенация исходников
        std::string combined = std::string(detect_src) + "\n" + std::string(compute_mag_src);
        const char* combined_src = combined.c_str();
        size_t combined_len = combined.size();

        all_maxima_program_ = clCreateProgramWithSource(context_, 1, &combined_src, &combined_len, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("CompileAllMaximaKernels: detect program create failed: " + std::to_string(err));

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
            throw std::runtime_error("CompileAllMaximaKernels: detect build failed: " + std::to_string(err));
        }

        detect_kernel_ = clCreateKernel(all_maxima_program_, "detect_all_maxima", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("CompileAllMaximaKernels: detect kernel create failed: " + std::to_string(err));

        compute_magnitudes_kernel_ = clCreateKernel(all_maxima_program_, "compute_magnitudes", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("CompileAllMaximaKernels: compute_magnitudes kernel create failed: " + std::to_string(err));
    }

    // 2. Prefix sum kernels
    {
        const char* src = kernels::GetPrefixSumKernelSource_opencl();
        size_t src_len = strlen(src);
        prefix_sum_program_ = clCreateProgramWithSource(context_, 1, &src, &src_len, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("CompileAllMaximaKernels: scan program create failed: " + std::to_string(err));

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
            throw std::runtime_error("CompileAllMaximaKernels: scan build failed: " + std::to_string(err));
        }

        block_scan_kernel_ = clCreateKernel(prefix_sum_program_, "block_scan", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("CompileAllMaximaKernels: block_scan kernel create failed: " + std::to_string(err));

        block_add_kernel_ = clCreateKernel(prefix_sum_program_, "block_add", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("CompileAllMaximaKernels: block_add kernel create failed: " + std::to_string(err));
    }

    // 3. Compact kernel
    {
        const char* src = kernels::GetCompactMaximaKernelSource_opencl();
        size_t src_len = strlen(src);
        compact_program_ = clCreateProgramWithSource(context_, 1, &src, &src_len, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("CompileAllMaximaKernels: compact program create failed: " + std::to_string(err));

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
            throw std::runtime_error("CompileAllMaximaKernels: compact build failed: " + std::to_string(err));
        }

        compact_kernel_ = clCreateKernel(compact_program_, "compact_maxima", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("CompileAllMaximaKernels: compact kernel create failed: " + std::to_string(err));
    }

    all_maxima_kernels_compiled_ = true;
    drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "AllMaxima",
        "Kernels compiled (detect + compute_magnitudes + scan + compact)");
}

// ════════════════════════════════════════════════════════════════════════════
// ExecutePrefixSum — beam-aware parallel prefix sum (Blelloch)
// ════════════════════════════════════════════════════════════════════════════

cl_event SpectrumMaximaFinder::ExecutePrefixSum(
    cl_mem input, cl_mem output, size_t n_per_beam, size_t beam_count,
    cl_event wait_event)
{
    cl_int err;

    // Количество блоков на луч
    size_t blocks_per_beam = (n_per_beam + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
    size_t total_blocks = beam_count * blocks_per_beam;

    uint32_t npb = static_cast<uint32_t>(n_per_beam);
    uint32_t bc = static_cast<uint32_t>(beam_count);
    uint32_t bpb = static_cast<uint32_t>(blocks_per_beam);

    if (blocks_per_beam == 1) {
        // Один блок на луч — простой случай
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

    // Многоблочный beam-aware scan
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

    // Level 1: scan каждого блока
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

        // Level 2: scan сумм блоков (рекурсивно, beam-aware!)
        cl_event sums_scan_event = ExecutePrefixSum(
            block_sums, block_sums_scanned, blocks_per_beam, beam_count, scan_event);
        clReleaseEvent(scan_event);

        // Level 3: добавить scanned sums к результатам
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
// FindAllMaxima(cl_mem) — detect → scan → compact pipeline
// ════════════════════════════════════════════════════════════════════════════
//
// Принимает FFT данные (float2*) и magnitudes_buffer_ с pre-computed |FFT[i]|.
// Если magnitudes_buffer_ не заполнен (старый API) — запускает compute_magnitudes.

AllMaximaResult SpectrumMaximaFinder::FindAllMaxima(
    cl_mem fft_data,
    uint32_t beam_count,
    uint32_t nFFT,
    float sample_rate,
    OutputDestination dest,
    uint32_t search_start,
    uint32_t search_end,
    uint32_t beam_offset,
    cl_mem external_out_maxima,
    cl_mem external_out_counts,
    ProfEvents* prof_events)
{
    if (!fft_data)
        throw std::invalid_argument("FindAllMaxima: fft_data cannot be null");

    // Defaults
    if (search_start == 0) search_start = 1;
    if (search_end == 0) search_end = nFFT / 2;
    if (search_end >= nFFT) search_end = nFFT - 1;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima", "FindAllMaxima: beam_count=" + std::to_string(beam_count)
        + " nFFT=" + std::to_string(nFFT)
        + " range=[" + std::to_string(search_start) + "," + std::to_string(search_end) + ")"
        + " Fs=" + std::to_string(static_cast<int>(sample_rate)) + "Hz");

    // Lazy compilation
    CompileAllMaximaKernels();

    cl_int err;
    const size_t total_elements = static_cast<size_t>(beam_count) * nFFT;

    // ═══════════════════════════════════════════════════════════════════════
    // 0. Обеспечить magnitudes_buffer_ (pre-computed |FFT[i]|)
    // ═══════════════════════════════════════════════════════════════════════
    EnsureMagnitudesBuffer(total_elements);

    // Если magnitudes ещё не заполнены post-callback'ом → запускаем compute_magnitudes
    // (Для full pipeline magnitudes заполняются post-callback'ом — но мы не знаем,
    //  был ли вызван post-callback. Поэтому для raw FFT API всегда пересчитываем.)
    // Определяем по fft_data: если это наш fft_output_ — magnitudes уже заполнены
    bool need_compute_magnitudes = (fft_data != fft_output_ || !allmax_plan_created_);

    cl_event mag_event = nullptr;
    if (need_compute_magnitudes) {
        uint32_t total_size = static_cast<uint32_t>(total_elements);

        err = clSetKernelArg(compute_magnitudes_kernel_, 0, sizeof(cl_mem), &fft_data);
        err |= clSetKernelArg(compute_magnitudes_kernel_, 1, sizeof(cl_mem), &magnitudes_buffer_);
        err |= clSetKernelArg(compute_magnitudes_kernel_, 2, sizeof(uint32_t), &total_size);
        if (err != CL_SUCCESS)
            throw std::runtime_error("FindAllMaxima: compute_magnitudes setKernelArg failed: " + std::to_string(err));

        size_t global_size = ((total_elements + 255) / 256) * 256;
        size_t local_size = 256;

        err = clEnqueueNDRangeKernel(queue_, compute_magnitudes_kernel_, 1, nullptr,
            &global_size, &local_size, 0, nullptr, &mag_event);
        if (err != CL_SUCCESS)
            throw std::runtime_error("FindAllMaxima: compute_magnitudes NDRange failed: " + std::to_string(err));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. Выделить буферы
    // ═══════════════════════════════════════════════════════════════════════
    cl_mem flags_buf = clCreateBuffer(context_, CL_MEM_READ_WRITE,
        total_elements * sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error("FindAllMaxima: flags_buf alloc failed: " + std::to_string(err));

    cl_mem scan_buf = clCreateBuffer(context_, CL_MEM_READ_WRITE,
        total_elements * sizeof(uint32_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(flags_buf);
        throw std::runtime_error("FindAllMaxima: scan_buf alloc failed: " + std::to_string(err));
    }

    uint32_t max_output_per_beam = std::min(
        (search_end - search_start) / 2,
        static_cast<uint32_t>(params_.max_maxima_per_beam)
    );

    // Выходные буферы: используем внешние если переданы, иначе создаём свои
    bool use_external_buffers = (external_out_maxima != nullptr);
    cl_mem out_maxima = external_out_maxima;
    cl_mem out_beam_counts = external_out_counts;

    if (!use_external_buffers) {
        // Создаём локальные буферы на beam_count
        out_maxima = clCreateBuffer(context_, CL_MEM_READ_WRITE,
            static_cast<size_t>(beam_count) * max_output_per_beam * sizeof(MaxValue), nullptr, &err);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(flags_buf);
            clReleaseMemObject(scan_buf);
            throw std::runtime_error("FindAllMaxima: out_maxima alloc failed: " + std::to_string(err));
        }

        out_beam_counts = clCreateBuffer(context_, CL_MEM_READ_WRITE,
            beam_count * sizeof(uint32_t), nullptr, &err);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(flags_buf);
            clReleaseMemObject(scan_buf);
            clReleaseMemObject(out_maxima);
            throw std::runtime_error("FindAllMaxima: out_beam_counts alloc failed: " + std::to_string(err));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. Detection kernel (reads float* magnitudes_buffer_)
    // ═══════════════════════════════════════════════════════════════════════
    {
        err = clSetKernelArg(detect_kernel_, 0, sizeof(cl_mem), &magnitudes_buffer_);
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
            throw std::runtime_error("FindAllMaxima: detect setKernelArg failed: " + std::to_string(err));
        }

        size_t global_size = ((total_elements + 255) / 256) * 256;
        size_t local_size = 256;

        cl_event detect_event = nullptr;
        err = clEnqueueNDRangeKernel(queue_, detect_kernel_, 1, nullptr,
            &global_size, &local_size,
            (mag_event ? 1 : 0), (mag_event ? &mag_event : nullptr),
            &detect_event);
        if (err != CL_SUCCESS) {
            if (mag_event) clReleaseEvent(mag_event);
            clReleaseMemObject(flags_buf);
            clReleaseMemObject(scan_buf);
            clReleaseMemObject(out_maxima);
            clReleaseMemObject(out_beam_counts);
            throw std::runtime_error("FindAllMaxima: detect NDRange failed: " + std::to_string(err));
        }
        CollectOrRelease(mag_event, "ComputeMagnitudes", prof_events);

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Beam-aware Parallel Prefix Sum
    // ═══════════════════════════════════════════════════════════════════════
        cl_event scan_all_event = ExecutePrefixSum(
            flags_buf, scan_buf, nFFT, beam_count, detect_event);

    // ═══════════════════════════════════════════════════════════════════════
    // 4. Compaction kernel (reads float* magnitudes_buffer_)
    // ═══════════════════════════════════════════════════════════════════════
        // beam_offset передан как параметр функции

        err = clSetKernelArg(compact_kernel_, 0, sizeof(cl_mem), &fft_data);
        err |= clSetKernelArg(compact_kernel_, 1, sizeof(cl_mem), &magnitudes_buffer_);
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
        CollectOrRelease(scan_all_event, "Scan", prof_events);
        CollectOrRelease(detect_event, "Detect", prof_events);

        if (err != CL_SUCCESS) {
            clReleaseMemObject(flags_buf);
            clReleaseMemObject(scan_buf);
            clReleaseMemObject(out_maxima);
            clReleaseMemObject(out_beam_counts);
            throw std::runtime_error("FindAllMaxima: compact NDRange failed: " + std::to_string(err));
        }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. Читаем beam_counts
    // ═══════════════════════════════════════════════════════════════════════
        std::vector<uint32_t> beam_counts(beam_count);
        cl_event read_counts_event = nullptr;
        clEnqueueReadBuffer(queue_, out_beam_counts, CL_FALSE, 0,
            beam_count * sizeof(uint32_t), beam_counts.data(),
            1, &compact_event, &read_counts_event);
        clWaitForEvents(1, &read_counts_event);
        clReleaseEvent(read_counts_event);
        CollectOrRelease(compact_event, "Compact", prof_events);

    // ═══════════════════════════════════════════════════════════════════════
    // 6. Формируем результат
    // ═══════════════════════════════════════════════════════════════════════
        AllMaximaResult result;
        result.destination = dest;
        result.total_maxima = 0;

        if (dest == OutputDestination::CPU || dest == OutputDestination::ALL) {
            result.beams.resize(beam_count);

            for (uint32_t b = 0; b < beam_count; ++b) {
                uint32_t count = beam_counts[b];
                if (count > max_output_per_beam) {
                    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
                    con.Print(0, "AllMaxima", "WARNING: Beam " + std::to_string(b + beam_offset) +
                        " reached max_maxima limit (" + std::to_string(count) + "/" +
                        std::to_string(max_output_per_beam) + "), results truncated");
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
            // Dest=GPU: заполняем beams метаданными (antenna_id, num_maxima) для caller
            result.beams.resize(beam_count);
            for (uint32_t b = 0; b < beam_count; ++b) {
                uint32_t count = beam_counts[b];
                if (count > max_output_per_beam) count = max_output_per_beam;
                result.beams[b].antenna_id = b;
                result.beams[b].num_maxima = count;
                result.total_maxima += count;
            }
        }

        if (dest == OutputDestination::GPU || dest == OutputDestination::ALL) {
            result.gpu_maxima = out_maxima;
            result.gpu_counts = out_beam_counts;
            result.gpu_bytes = static_cast<size_t>(beam_count) * max_output_per_beam * sizeof(MaxValue);
            out_maxima = nullptr;
            out_beam_counts = nullptr;
        }

        con.Print(0, "AllMaxima", "Found " + std::to_string(result.total_maxima) + " maxima");

        // Освобождаем временные буферы
        clReleaseMemObject(flags_buf);
        clReleaseMemObject(scan_buf);

        // Освобождаем выходные буферы только если они НЕ внешние
        if (!use_external_buffers) {
            if (out_maxima) clReleaseMemObject(out_maxima);
            if (out_beam_counts) clReleaseMemObject(out_beam_counts);
        }

        return result;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ReleaseAllMaximaResources — освобождение ресурсов
// ════════════════════════════════════════════════════════════════════════════

void SpectrumMaximaFinder::ReleaseAllMaximaResources() {
    if (detect_kernel_) { clReleaseKernel(detect_kernel_); detect_kernel_ = nullptr; }
    if (compute_magnitudes_kernel_) { clReleaseKernel(compute_magnitudes_kernel_); compute_magnitudes_kernel_ = nullptr; }
    if (all_maxima_program_) { clReleaseProgram(all_maxima_program_); all_maxima_program_ = nullptr; }
    if (block_scan_kernel_) { clReleaseKernel(block_scan_kernel_); block_scan_kernel_ = nullptr; }
    if (block_add_kernel_) { clReleaseKernel(block_add_kernel_); block_add_kernel_ = nullptr; }
    if (prefix_sum_program_) { clReleaseProgram(prefix_sum_program_); prefix_sum_program_ = nullptr; }
    if (compact_kernel_) { clReleaseKernel(compact_kernel_); compact_kernel_ = nullptr; }
    if (compact_program_) { clReleaseProgram(compact_program_); compact_program_ = nullptr; }
    all_maxima_kernels_compiled_ = false;
}

} // namespace antenna_fft
