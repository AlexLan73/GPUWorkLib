#include "antenna_fft_release.h"
#include "fft_logger.h"
#include "services/gpu_profiler.hpp"
#include <cstring>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTProcMax::AntennaFFTProcMax(const AntennaFFTParams& params, drv_gpu_lib::IBackend* backend)
    : AntennaFFTCore(params, backend),
      buffer_selected_complex_(nullptr),
      buffer_selected_magnitude_(nullptr),
      plan_num_beams_(0) {

    // Вызов виртуального Initialize (создание плана с колбэками)
    Initialize();
}

AntennaFFTProcMax::~AntennaFFTProcMax() {
    ReleaseBuffers();
}

// ════════════════════════════════════════════════════════════════════════════
// Реализации виртуальных методов
// ════════════════════════════════════════════════════════════════════════════

void AntennaFFTProcMax::Initialize() {
    FFTLogger::Info("[AntennaFFTProcMax] Initializing with callbacks...");
    FFTLogger::Info("  beam_count: ", params_.beam_count);
    FFTLogger::Info("  count_points: ", params_.count_points);
    FFTLogger::Info("  nFFT: ", nFFT_);
    FFTLogger::Info("  out_count_points_fft: ", params_.out_count_points_fft);

    // Создание кэша FFT-планов для данного контекста
    plan_cache_ = std::make_unique<FFTPlanCache>(context_, queue_);

    // Выделение буферов для начального размера пакета
    size_t initial_beams = batch_config_.beams_per_batch;
    if (initial_beams == 0) initial_beams = params_.beam_count;

    AllocateBuffers(initial_beams);
    CreateFFTPlanWithCallbacks(initial_beams);

    FFTLogger::Info("[AntennaFFTProcMax] Initialized!");
}

AntennaFFTResult AntennaFFTProcMax::ProcessSingleBatch(cl_mem input_signal) {
    AntennaFFTResult result;
    result.total_beams = params_.beam_count;
    result.nFFT = nFFT_;
    result.task_id = params_.task_id;
    result.module_name = params_.module_name;

    // Убедиться, что буферы и план готовы для полного пакета
    if (current_buffer_beams_ < params_.beam_count) {
        ReleaseBuffers();
        ReleaseFFTPlan();
        AllocateBuffers(params_.beam_count);
        CreateFFTPlanWithCallbacks(params_.beam_count);
    }

    // Копирование входных данных в userdata pre-callback (после заголовка)
    size_t input_size = params_.beam_count * params_.count_points * sizeof(std::complex<float>);
    cl_int err = clEnqueueCopyBuffer(queue_, input_signal, pre_callback_userdata_,
                                      0, 32, // смещение заголовка 32 байта
                                      input_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to copy input to userdata: " + std::to_string(err));
    }

    // Выполнение FFT с колбэками
    cl_event fft_event;
    if (!ExecuteFFTWithCallbacks(input_signal, params_.beam_count, 0, &fft_event)) {
        throw std::runtime_error("FFT execution failed");
    }

    // Ожидание завершения
    clWaitForEvents(1, &fft_event);

    // Profile
    double fft_time_ms = ProfileEvent(fft_event, "FFT");
    last_profiling_results_.fft_time_ms = fft_time_ms;

    // Record to GPUProfiler (async, non-blocking)
    drv_gpu_lib::GPUProfiler::GetInstance().Record(
        backend_->GetDeviceIndex(),
        "AntennaFFT",
        "SingleBatchFFT",
        fft_time_ms
    );

    clReleaseEvent(fft_event);

    // Read results
    result.results = ReadResults(params_.beam_count, 0);

    return result;
}

std::vector<FFTResult> AntennaFFTProcMax::ProcessBatch(
    cl_mem input_signal,
    size_t start_beam,
    size_t num_beams,
    BatchProfilingData* out_profiling) {

    // Ensure buffers and plan are ready
    if (current_buffer_beams_ < num_beams || plan_num_beams_ != num_beams) {
        if (current_buffer_beams_ < num_beams) {
            ReleaseBuffers();
            AllocateBuffers(num_beams);
        }
        if (plan_num_beams_ != num_beams) {
            ReleaseFFTPlan();
            CreateFFTPlanWithCallbacks(num_beams);
        }
    }

    // Update userdata for this batch
    CreatePreCallbackUserData(num_beams);
    CreatePostCallbackUserData(num_beams);

    // Copy input data for this batch to userdata (after 32-byte header)
    size_t input_offset = start_beam * params_.count_points * sizeof(std::complex<float>);
    size_t input_size = num_beams * params_.count_points * sizeof(std::complex<float>);

    cl_int err = clEnqueueCopyBuffer(queue_, input_signal, pre_callback_userdata_,
                                      input_offset, 32, // 32 bytes header offset
                                      input_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to copy batch input: " + std::to_string(err));
    }

    // Execute FFT with callbacks
    cl_event fft_event;
    auto fft_start = std::chrono::high_resolution_clock::now();

    if (!ExecuteFFTWithCallbacks(input_signal, num_beams, start_beam, &fft_event)) {
        throw std::runtime_error("Batch FFT execution failed");
    }

    clWaitForEvents(1, &fft_event);

    auto fft_end = std::chrono::high_resolution_clock::now();

    // Profile
    double fft_time_ms = ProfileEvent(fft_event, "BatchFFT");
    if (out_profiling) {
        out_profiling->fft_time_ms = fft_time_ms;
        out_profiling->padding_time_ms = 0; // Included in pre-callback
        out_profiling->post_time_ms = 0;    // Included in post-callback
    }

    // Record to GPUProfiler (async, non-blocking)
    drv_gpu_lib::GPUProfiler::GetInstance().Record(
        backend_->GetDeviceIndex(),
        "AntennaFFT",
        "BatchFFT",
        fft_time_ms
    );

    clReleaseEvent(fft_event);

    // Read results
    return ReadResults(num_beams, start_beam);
}

void AntennaFFTProcMax::AllocateBuffers(size_t num_beams) {
    if (current_buffer_beams_ >= num_beams) return;

    ReleaseBuffers();

    cl_int err;

    // FFT buffers
    size_t fft_size = nFFT_ * num_beams * sizeof(std::complex<float>);
    buffer_fft_input_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, fft_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate fft_input buffer");

    buffer_fft_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, fft_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate fft_output buffer");

    // Selected spectrum buffers
    size_t selected_size = params_.out_count_points_fft * num_beams * sizeof(std::complex<float>);
    buffer_selected_complex_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, selected_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate selected_complex buffer");

    size_t magnitude_size = params_.out_count_points_fft * num_beams * sizeof(float);
    buffer_selected_magnitude_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, magnitude_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate selected_magnitude buffer");

    // Maxima buffer
    size_t maxima_size = params_.max_peaks_count * num_beams * 32; // MaxValue struct = 32 bytes
    buffer_maxima_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, maxima_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate maxima buffer");

    // Create userdata buffers
    CreatePreCallbackUserData(num_beams);
    CreatePostCallbackUserData(num_beams);

    current_buffer_beams_ = num_beams;

    FFTLogger::Info("  [Release] Allocated buffers for ", num_beams, " beams");
}

void AntennaFFTProcMax::ReleaseBuffers() {
    if (buffer_fft_input_) { clReleaseMemObject(buffer_fft_input_); buffer_fft_input_ = nullptr; }
    if (buffer_fft_output_) { clReleaseMemObject(buffer_fft_output_); buffer_fft_output_ = nullptr; }
    if (buffer_selected_complex_) { clReleaseMemObject(buffer_selected_complex_); buffer_selected_complex_ = nullptr; }
    if (buffer_selected_magnitude_) { clReleaseMemObject(buffer_selected_magnitude_); buffer_selected_magnitude_ = nullptr; }
    if (buffer_maxima_) { clReleaseMemObject(buffer_maxima_); buffer_maxima_ = nullptr; }
    current_buffer_beams_ = 0;
}

// ════════════════════════════════════════════════════════════════════════════
// Private methods
// ════════════════════════════════════════════════════════════════════════════

void AntennaFFTProcMax::CreateFFTPlanWithCallbacks(size_t num_beams) {
    if (plan_created_ && plan_num_beams_ == num_beams) return;

    // Check if plan is already in cache and baked
    if (plan_cache_ && plan_cache_->IsBaked(nFFT_, num_beams)) {
        // Cache HIT - reuse existing plan
        plan_handle_ = plan_cache_->GetOrCreate(nFFT_, num_beams);
        plan_created_ = true;
        plan_num_beams_ = num_beams;
        FFTLogger::Info("  [Release] FFT plan retrieved from cache (nFFT=", nFFT_, ", beams=", num_beams, ")");
        return;
    }

    // Need to release old plan (if not cached) and create new
    // Note: If we have cache, plans are managed by cache, not ReleaseFFTPlan
    if (!plan_cache_) {
        ReleaseFFTPlan();
    }

    FFTLogger::Info("  [Release] Creating FFT plan with callbacks for ", num_beams, " beams...");

    // Get or create plan from cache (returns unbaked plan)
    if (plan_cache_) {
        plan_handle_ = plan_cache_->GetOrCreate(nFFT_, num_beams);
    } else {
        // Fallback: create plan without cache
        size_t dim = nFFT_;
        clfftStatus status = clfftCreateDefaultPlan(&plan_handle_, context_, CLFFT_1D, &dim);
        if (status != CLFFT_SUCCESS) {
            throw std::runtime_error("clfftCreateDefaultPlan failed: " + std::to_string(status));
        }

        // Configure plan
        clfftSetPlanPrecision(plan_handle_, CLFFT_SINGLE);
        clfftSetLayout(plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
        clfftSetResultLocation(plan_handle_, CLFFT_OUTOFPLACE);
        clfftSetPlanBatchSize(plan_handle_, num_beams);

        size_t strides[1] = {1};
        size_t dist = nFFT_;
        clfftSetPlanInStride(plan_handle_, CLFFT_1D, strides);
        clfftSetPlanOutStride(plan_handle_, CLFFT_1D, strides);
        clfftSetPlanDistance(plan_handle_, dist, dist);
    }

    // Register pre-callback (32-byte struct version)
    const char* pre_callback_source = kernels::GetPreCallbackSource32();
    clfftStatus status = clfftSetPlanCallback(plan_handle_, "prepareDataPre", pre_callback_source, 0,
                                  PRECALLBACK, &pre_callback_userdata_, 1);
    if (status != CLFFT_SUCCESS) {
        if (!plan_cache_) clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftSetPlanCallback (pre) failed: " + std::to_string(status));
    }

    // Register post-callback
    const char* post_callback_source = kernels::GetPaddingKernelSource();

    status = clfftSetPlanCallback(plan_handle_, "processFFTPost", post_callback_source, 0,
                                  POSTCALLBACK, &post_callback_userdata_, 1);
    if (status != CLFFT_SUCCESS) {
        if (!plan_cache_) clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftSetPlanCallback (post) failed: " + std::to_string(status));
    }

    // Bake plan
    status = clfftBakePlan(plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        if (!plan_cache_) clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftBakePlan failed: " + std::to_string(status));
    }

    // Mark as baked in cache
    if (plan_cache_) {
        plan_cache_->MarkBaked(nFFT_, num_beams);
    }

    plan_created_ = true;
    plan_num_beams_ = num_beams;

    FFTLogger::Info("  [Release] FFT plan created and cached!");
}

bool AntennaFFTProcMax::ExecuteFFTWithCallbacks(
    cl_mem input_signal,
    size_t num_beams,
    size_t start_beam,
    cl_event* out_fft_event) {

    if (!plan_created_) {
        std::cerr << "FFT plan not created!\n";
        return false;
    }

    // Execute FFT (callbacks do padding and post-processing)
    clfftStatus status = clfftEnqueueTransform(
        plan_handle_,
        CLFFT_FORWARD,
        1, &queue_,
        0, nullptr,
        out_fft_event,
        &buffer_fft_input_,    // Input (pre-callback fills this)
        &buffer_fft_output_,   // Output (post-callback processes this)
        nullptr                // Temp buffer
    );

    if (status != CLFFT_SUCCESS) {
        std::cerr << "clfftEnqueueTransform failed: " << status << "\n";
        return false;
    }

    return true;
}

std::vector<FFTResult> AntennaFFTProcMax::ReadResults(size_t num_beams, size_t start_beam) {
    std::vector<FFTResult> results;
    results.reserve(num_beams);

    // Read maxima from GPU
    // MaxValue struct: {index, real, imag, magnitude, phase, freq_offset, refined_freq, pad} = 32 bytes
    struct MaxValue {
        cl_uint index;
        float real;
        float imag;
        float magnitude;
        float phase;
        float freq_offset;
        float refined_frequency;
        cl_uint pad;
    };

    size_t maxima_count = params_.max_peaks_count * num_beams;
    std::vector<MaxValue> maxima(maxima_count);

    cl_int err = clEnqueueReadBuffer(queue_, buffer_maxima_, CL_TRUE, 0,
                                      maxima_count * sizeof(MaxValue),
                                      maxima.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read maxima: " + std::to_string(err));
    }

    // Convert to FFTResult
    for (size_t beam = 0; beam < num_beams; ++beam) {
        FFTResult result(params_.out_count_points_fft, params_.task_id, params_.module_name);

        for (size_t peak = 0; peak < params_.max_peaks_count; ++peak) {
            size_t idx = beam * params_.max_peaks_count + peak;
            const MaxValue& mv = maxima[idx];

            FFTMaxResult max_result;
            max_result.index_point = mv.index;
            max_result.real = mv.real;
            max_result.imag = mv.imag;
            max_result.amplitude = mv.magnitude;
            max_result.phase = mv.phase;

            result.max_values.push_back(max_result);
        }

        // Set refined frequency from first maximum
        if (!result.max_values.empty()) {
            result.freq_offset = maxima[beam * params_.max_peaks_count].freq_offset;
            result.refined_frequency = maxima[beam * params_.max_peaks_count].refined_frequency;
        }

        results.push_back(std::move(result));
    }

    return results;
}

} // namespace antenna_fft
