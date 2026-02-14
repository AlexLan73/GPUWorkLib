/**
 * @file spectrum_maxima_finder_process.cpp
 * @brief SpectrumMaximaFinder — методы для шаблонного Process<T>
 *
 * Отделено от spectrum_maxima_finder.cpp для читабельности.
 *
 * Содержит:
 *   - PrepareParams()           — подготовка параметров из InputData<T>
 *   - ProcessFromCPU()          — обработка CPU-данных (vector<complex<float>>)
 *   - ProcessFromGPU()          — обработка GPU-данных (cl_mem)
 *   - ProcessBatchFromGPU()     — обработка одного batch с GPU-данными
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-14
 */

#include "spectrum_maxima_finder.h"
#include "backends/opencl/opencl_profiling.hpp"
#include "services/gpu_profiler.hpp"
#include "services/console_output.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// PrepareParams — подготовка параметров из InputData<T>
// ════════════════════════════════════════════════════════════════════════════

void SpectrumMaximaFinder::PrepareParams(
    uint32_t antenna_count, uint32_t n_point,
    const ProcessingParams& proc_params, PeakSearchMode mode)
{
    // Заполняем params_ из входных данных
    params_.antenna_count = antenna_count;
    params_.n_point = n_point;
    params_.repeat_count = proc_params.repeat_count;
    params_.sample_rate = proc_params.sample_rate;
    params_.search_range = proc_params.search_range;
    params_.memory_limit = proc_params.memory_limit;
    params_.peak_mode = mode;

    // Вычисляем nFFT и base_fft
    params_.base_fft = NextPowerOf2(n_point);
    params_.nFFT = params_.base_fft * params_.repeat_count;

    if (params_.search_range == 0) {
        params_.search_range = params_.nFFT / 4;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessFromCPU — обработка CPU-данных (vector<complex<float>>)
// ════════════════════════════════════════════════════════════════════════════

std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessFromCPU(
    const std::vector<std::complex<float>>& data)
{
    // Проверка: инициализирован ли объект
    if (!initialized_) {
        throw std::runtime_error("SpectrumMaximaFinder::ProcessFromCPU: not initialized");
    }

    // Проверка размера входных данных
    size_t expected_size = params_.antenna_count * params_.n_point;
    if (data.size() != expected_size) {
        throw std::invalid_argument(
            "SpectrumMaximaFinder::ProcessFromCPU: input size mismatch. "
            "Expected " + std::to_string(expected_size) +
            ", got " + std::to_string(data.size()));
    }

    // Сбросить профилирование
    profiling_ = ProfilingData{};

    // ═══════════════════════════════════════════════════════════════════════
    // BATCH PROCESSING (использует BatchManager)
    // ═══════════════════════════════════════════════════════════════════════

    // Рассчитать память на одну антенну
    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    // Проверить, помещаются ли все данные в память
    if (drv_gpu_lib::BatchManager::AllItemsFit(backend_, params_.antenna_count,
                                                bytes_per_antenna, params_.memory_limit)) {
        auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
        con.Print(0, "SpectrumMaxima", "Все " + std::to_string(params_.antenna_count) +
            " антенн помещаются в память — batch не нужен");
        return ProcessBatch(data, 0, params_.antenna_count);
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

    size_t batch_size = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params_.antenna_count, batch_size, 3, true);

    con.Print(0, "SpectrumMaxima", "Batch Processing: " +
        std::to_string(bytes_per_antenna / 1024 / 1024) + " MB/antenna, limit=" +
        std::to_string(static_cast<int>(params_.memory_limit * 100)) + "%");
    drv_gpu_lib::BatchManager::PrintBatchInfo(batches, params_.antenna_count);

    std::vector<SpectrumResult> all_results;
    all_results.reserve(params_.antenna_count);

    for (const auto& batch : batches) {
        con.Print(0, "SpectrumMaxima", "Batch " + std::to_string(batch.batch_idx) +
            ": antennas [" + std::to_string(batch.start) + ".." +
            std::to_string(batch.start + batch.count - 1) + "]");

        auto batch_results = ProcessBatch(data, batch.start, batch.count);
        all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
    }

    con.Print(0, "SpectrumMaxima", "Batch completed: " + std::to_string(all_results.size()) + " results");

    return all_results;
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessFromGPU — обработка GPU-данных (cl_mem)
// ════════════════════════════════════════════════════════════════════════════

std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessFromGPU(
    cl_mem gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes)
{
    if (!gpu_data) {
        throw std::invalid_argument("ProcessFromGPU: gpu_data cannot be null");
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "SpectrumMaxima", "ProcessFromGPU: " +
        std::to_string(antenna_count) + " antennas, n_point=" + std::to_string(n_point));

    if (params_.nFFT == 0) {
        CalculateFFTSize();
        con.Print(0, "SpectrumMaxima", "nFFT=" + std::to_string(params_.nFFT) +
            " search_range=" + std::to_string(params_.search_range));
    }

    if (!post_kernel_) {
        CompilePostKernel();
    }

    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : antenna_count * n_point * sizeof(std::complex<float>);

    con.Print(0, "SpectrumMaxima", "External data: " +
        std::to_string(external_memory / 1024 / 1024) + " MB on GPU");

    // ═══════════════════════════════════════════════════════════════════════════
    // Расчёт batch size с учётом уже занятой памяти
    // ═══════════════════════════════════════════════════════════════════════════
    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, antenna_count, bytes_per_antenna,
        params_.memory_limit, external_memory);

    // Проверяем, нужен ли batch processing
    bool need_batch = (optimal_batch < antenna_count);

    if (!need_batch) {
        con.Print(0, "SpectrumMaxima", "Single batch — все данные помещаются");

        // Выделяем буферы под все антенны
        ReallocateBuffersForBatch(antenna_count);
        actual_batch_size_ = antenna_count;

        // Копируем данные GPU→GPU
        size_t data_size = antenna_count * n_point * sizeof(std::complex<float>);
        cl_event copy_event = nullptr;
        cl_int err = clEnqueueCopyBuffer(queue_,
            gpu_data, pre_callback_userdata_,
            0, PRE_CALLBACK_HEADER_SIZE,  // src_offset=0, dst_offset=32 (skip header)
            data_size,
            0, nullptr, &copy_event);

        if (err != CL_SUCCESS) {
            throw std::runtime_error("ProcessFromGPU: clEnqueueCopyBuffer failed: " + std::to_string(err));
        }

        // FFT + PostKernel + ReadResults
        cl_event fft_event = ExecuteFFT(copy_event);
        cl_event post_event = ExecutePostKernel(fft_event);
        auto results = ReadResults(post_event, true);

        // Cleanup events
        clReleaseEvent(copy_event);
        clReleaseEvent(fft_event);
        clReleaseEvent(post_event);

        return results;
    }

    con.Print(0, "SpectrumMaxima", "Batch mode: batch_size=" + std::to_string(optimal_batch));

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(antenna_count, optimal_batch);
    drv_gpu_lib::BatchManager::PrintBatchInfo(batches, antenna_count);

    ReallocateBuffersForBatch(optimal_batch);

    std::vector<SpectrumResult> all_results;
    all_results.reserve(antenna_count);

    for (const auto& batch : batches) {
        con.Print(0, "SpectrumMaxima", "Batch [" + std::to_string(batch.start) +
            ".." + std::to_string(batch.start + batch.count - 1) + "]");

        size_t src_offset = batch.start * n_point * sizeof(std::complex<float>);
        auto batch_results = ProcessBatchFromGPU(gpu_data, src_offset,
                                                  batch.start, batch.count);

        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    con.Print(0, "SpectrumMaxima", "GPU Batch completed: " + std::to_string(all_results.size()) + " results");
    return all_results;
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessBatchFromGPU — обработка одного batch с GPU-данными
// ════════════════════════════════════════════════════════════════════════════

std::vector<SpectrumResult> SpectrumMaximaFinder::ProcessBatchFromGPU(
    cl_mem gpu_data, size_t src_offset_bytes,
    size_t start_antenna, size_t batch_antenna_count)
{
    // Перевыделить буферы если нужно (с переиспользованием FFT плана)
    if (batch_antenna_count > current_batch_size_ || !plan_created_) {
        ReallocateBuffersForBatch(batch_antenna_count);
    }
    actual_batch_size_ = batch_antenna_count;

    // Обновить заголовок pre-callback
    WritePreCallbackHeader(batch_antenna_count);

    // Профилирование
    const int gpu_id = 0;
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
    const bool do_prof = profiler.IsEnabled() && profiler.IsGPUEnabled(gpu_id);

    // Копируем данные GPU→GPU (из внешнего буфера в наш pre_callback_userdata_)
    size_t data_size = batch_antenna_count * params_.n_point * sizeof(std::complex<float>);
    cl_event copy_event = nullptr;
    cl_int err = clEnqueueCopyBuffer(queue_,
        gpu_data, pre_callback_userdata_,
        src_offset_bytes, PRE_CALLBACK_HEADER_SIZE,
        data_size,
        0, nullptr, &copy_event);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("ProcessBatchFromGPU: clEnqueueCopyBuffer failed: " + std::to_string(err));
    }

    // Профилирование GPU→GPU Copy (аналог Upload для CPU данных)
    profiling_.upload_time_ms += ProfileEvent(copy_event, "GPU→GPU Copy");
    if (do_prof) {
        drv_gpu_lib::OpenCLProfilingData data{};
        if (drv_gpu_lib::FillOpenCLProfilingData(copy_event, data)) {
            profiler.Record(gpu_id, "SpectrumMaxima", "GPU→GPU Copy", data);
        }
    }

    // FFT
    cl_event fft_event = ExecuteFFT(copy_event);
    profiling_.fft_time_ms += ProfileEvent(fft_event, "FFT");
    if (do_prof) {
        drv_gpu_lib::OpenCLProfilingData data{};
        if (drv_gpu_lib::FillOpenCLProfilingData(fft_event, data)) {
            profiler.Record(gpu_id, "SpectrumMaxima", "FFT", data);
        }
    }

    // PostKernel
    cl_event post_event = ExecutePostKernel(fft_event);
    profiling_.post_kernel_time_ms += ProfileEvent(post_event, "PostKernel");
    if (do_prof) {
        drv_gpu_lib::OpenCLProfilingData data{};
        if (drv_gpu_lib::FillOpenCLProfilingData(post_event, data)) {
            profiler.Record(gpu_id, "SpectrumMaxima", "PostKernel", data);
        }
    }

    // ReadResults (Download профилируется внутри)
    auto results = ReadResults(post_event, do_prof);

    // Устанавливаем правильные antenna_id
    for (size_t i = 0; i < results.size(); ++i) {
        results[i].antenna_id = static_cast<uint32_t>(start_antenna + i);
    }

    // Cleanup events
    clReleaseEvent(copy_event);
    clReleaseEvent(fft_event);
    clReleaseEvent(post_event);

    return results;
}

} // namespace antenna_fft
