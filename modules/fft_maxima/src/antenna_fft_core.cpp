#include "antenna_fft_core.h"
#include "fft_logger.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTCore::AntennaFFTCore(const AntennaFFTParams& params, drv_gpu_lib::IBackend* backend)
    : params_(params),
      nFFT_(0),
      backend_(backend),
      context_(nullptr),
      queue_(nullptr),
      device_(nullptr),
      plan_handle_(0),
      plan_created_(false),
      buffer_fft_input_(nullptr),
      buffer_fft_output_(nullptr),
      buffer_maxima_(nullptr),
      pre_callback_userdata_(nullptr),
      post_callback_userdata_(nullptr),
      batch_total_cpu_time_ms_(0.0),
      last_used_batch_mode_(false),
      current_buffer_beams_(0) {

    // Validate parameters
    if (!params_.IsValid()) {
        throw std::invalid_argument("AntennaFFTParams: invalid parameters");
    }

    // Check backend
    if (!backend_ || !backend_->IsInitialized()) {
        throw std::runtime_error("Backend not initialized. Call Initialize() first.");
    }

    // Get context, device and queue from backend (Multi-GPU!)
    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    device_ = static_cast<cl_device_id>(backend_->GetNativeDevice());
    queue_ = static_cast<cl_command_queue>(backend_->GetNativeQueue());

    // Calculate nFFT
    nFFT_ = CalculateNFFT(params_.count_points);

    // Initialize clFFT library
    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftStatus status = clfftSetup(&fftSetup);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftSetup failed with status: " + std::to_string(status));
    }

    // Initialize profiling
    last_profiling_results_ = {};

    // Calculate batch config
    CalculateBatchConfig();
}

AntennaFFTCore::~AntennaFFTCore() {
    ReleaseFFTPlan();

    if (pre_callback_userdata_) {
        clReleaseMemObject(pre_callback_userdata_);
        pre_callback_userdata_ = nullptr;
    }
    if (post_callback_userdata_) {
        clReleaseMemObject(post_callback_userdata_);
        post_callback_userdata_ = nullptr;
    }

    // Note: derived classes must release their own buffers
}

AntennaFFTCore::AntennaFFTCore(AntennaFFTCore&& other) noexcept
    : params_(other.params_),
      nFFT_(other.nFFT_),
      backend_(other.backend_),
      context_(other.context_),
      queue_(other.queue_),
      device_(other.device_),
      plan_handle_(other.plan_handle_),
      plan_created_(other.plan_created_),
      buffer_fft_input_(other.buffer_fft_input_),
      buffer_fft_output_(other.buffer_fft_output_),
      buffer_maxima_(other.buffer_maxima_),
      pre_callback_userdata_(other.pre_callback_userdata_),
      post_callback_userdata_(other.post_callback_userdata_),
      last_profiling_results_(other.last_profiling_results_),
      batch_profiling_(std::move(other.batch_profiling_)),
      batch_total_cpu_time_ms_(other.batch_total_cpu_time_ms_),
      last_used_batch_mode_(other.last_used_batch_mode_),
      batch_config_(other.batch_config_),
      current_buffer_beams_(other.current_buffer_beams_) {

    // Null out moved-from object
    other.plan_handle_ = 0;
    other.plan_created_ = false;
    other.buffer_fft_input_ = nullptr;
    other.buffer_fft_output_ = nullptr;
    other.buffer_maxima_ = nullptr;
    other.pre_callback_userdata_ = nullptr;
    other.post_callback_userdata_ = nullptr;
}

AntennaFFTCore& AntennaFFTCore::operator=(AntennaFFTCore&& other) noexcept {
    if (this != &other) {
        // Release current resources
        ReleaseFFTPlan();
        if (pre_callback_userdata_) clReleaseMemObject(pre_callback_userdata_);
        if (post_callback_userdata_) clReleaseMemObject(post_callback_userdata_);

        // Move from other
        params_ = other.params_;
        nFFT_ = other.nFFT_;
        backend_ = other.backend_;
        context_ = other.context_;
        queue_ = other.queue_;
        device_ = other.device_;
        plan_handle_ = other.plan_handle_;
        plan_created_ = other.plan_created_;
        buffer_fft_input_ = other.buffer_fft_input_;
        buffer_fft_output_ = other.buffer_fft_output_;
        buffer_maxima_ = other.buffer_maxima_;
        pre_callback_userdata_ = other.pre_callback_userdata_;
        post_callback_userdata_ = other.post_callback_userdata_;
        last_profiling_results_ = other.last_profiling_results_;
        batch_profiling_ = std::move(other.batch_profiling_);
        batch_total_cpu_time_ms_ = other.batch_total_cpu_time_ms_;
        last_used_batch_mode_ = other.last_used_batch_mode_;
        batch_config_ = other.batch_config_;
        current_buffer_beams_ = other.current_buffer_beams_;

        // Null out moved-from object
        other.plan_handle_ = 0;
        other.plan_created_ = false;
        other.buffer_fft_input_ = nullptr;
        other.buffer_fft_output_ = nullptr;
        other.buffer_maxima_ = nullptr;
        other.pre_callback_userdata_ = nullptr;
        other.post_callback_userdata_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Public interface
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTResult AntennaFFTCore::ProcessNew(const std::vector<std::complex<float>>& input_data) {
    // Create GPU buffer from CPU data
    cl_mem input_buffer = CreateInputBuffer(input_data);

    // Process
    AntennaFFTResult result = ProcessNew(input_buffer);

    // Release temporary buffer
    clReleaseMemObject(input_buffer);

    return result;
}

AntennaFFTResult AntennaFFTCore::ProcessNew(cl_mem input_signal) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Check if batching is needed
    if (NeedsBatching()) {
        last_used_batch_mode_ = true;
        return ProcessWithBatching(input_signal);
    } else {
        last_used_batch_mode_ = false;
        return ProcessSingleBatch(input_signal);
    }
}

AntennaFFTResult AntennaFFTCore::ProcessWithBatching(cl_mem input_signal) {
    auto total_start = std::chrono::high_resolution_clock::now();

    AntennaFFTResult final_result;
    final_result.total_beams = params_.beam_count;
    final_result.nFFT = nFFT_;
    final_result.task_id = params_.task_id;
    final_result.module_name = params_.module_name;
    final_result.results.reserve(params_.beam_count);

    // Clear previous profiling
    batch_profiling_.clear();
    batch_total_cpu_time_ms_ = 0.0;

    // Calculate batch size
    size_t beams_per_batch = batch_config_.beams_per_batch;
    if (beams_per_batch == 0 || beams_per_batch > params_.beam_count) {
        beams_per_batch = params_.beam_count;
    }

    size_t processed_beams = 0;
    size_t batch_index = 0;

    FFTLogger::Info("  [Batching] Total beams: ", params_.beam_count, ", beams per batch: ", beams_per_batch);

    // ═══════════════════════════════════════════════════════════════════════════
    // Main batching loop (COMMON for Release and Debug!)
    // ═══════════════════════════════════════════════════════════════════════════
    while (processed_beams < params_.beam_count) {
        size_t beams_in_batch = std::min(beams_per_batch, params_.beam_count - processed_beams);

        BatchProfilingData batch_prof;
        batch_prof.batch_index = batch_index;
        batch_prof.start_beam = processed_beams;
        batch_prof.num_beams = beams_in_batch;

        auto batch_start = std::chrono::high_resolution_clock::now();

        // ═══════════════════════════════════════════════════════════════════════
        // VIRTUAL CALL - how to process batch (Release vs Debug)
        // ═══════════════════════════════════════════════════════════════════════
        std::vector<FFTResult> batch_results = ProcessBatch(
            input_signal,
            processed_beams,
            beams_in_batch,
            &batch_prof
        );

        auto batch_end = std::chrono::high_resolution_clock::now();
        double batch_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
        batch_prof.gpu_time_ms = batch_time;

        // Collect results
        for (auto& r : batch_results) {
            final_result.results.push_back(std::move(r));
        }

        // Store profiling
        batch_profiling_.push_back(batch_prof);

        processed_beams += beams_in_batch;
        batch_index++;

        FFTLogger::Info("  [Batch ", batch_index, "] Processed ", processed_beams, "/", params_.beam_count, " beams, time: ", batch_time, " ms");
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    batch_total_cpu_time_ms_ = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    FFTLogger::Info("  [Batching] Complete! Total batches: ", batch_index, ", total time: ", batch_total_cpu_time_ms_, " ms");

    return final_result;
}

// ════════════════════════════════════════════════════════════════════════════
// Protected utilities
// ════════════════════════════════════════════════════════════════════════════

size_t AntennaFFTCore::CalculateNFFT(size_t count_points) const {
    // If power of 2, use same, otherwise next power of 2
    // Then multiply by 2 for padding
    if (IsPowerOf2(count_points)) {
        return count_points * 2;
    } else {
        return NextPowerOf2(count_points) * 2;
    }
}

bool AntennaFFTCore::IsPowerOf2(size_t n) const {
    return n > 0 && (n & (n - 1)) == 0;
}

size_t AntennaFFTCore::NextPowerOf2(size_t n) const {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

size_t AntennaFFTCore::EstimateRequiredMemory(size_t num_beams) const {
    // FFT input: nFFT * num_beams * sizeof(complex<float>)
    // FFT output: nFFT * num_beams * sizeof(complex<float>)
    // Selected complex: out_count_points_fft * num_beams * sizeof(complex<float>)
    // Maxima: max_peaks_count * num_beams * 32 bytes

    size_t fft_buffer_size = nFFT_ * num_beams * sizeof(std::complex<float>);
    size_t selected_size = params_.out_count_points_fft * num_beams * sizeof(std::complex<float>);
    size_t maxima_size = params_.max_peaks_count * num_beams * 32;

    return 2 * fft_buffer_size + 2 * selected_size + maxima_size;
}

bool AntennaFFTCore::CheckAvailableMemory(size_t required_memory, double threshold) const {
    // Get device memory info
    cl_ulong global_mem_size = 0;
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);

    size_t available = static_cast<size_t>(global_mem_size * threshold);
    return required_memory <= available;
}

void AntennaFFTCore::CalculateBatchConfig() {
    size_t required = EstimateRequiredMemory(params_.beam_count);

    if (CheckAvailableMemory(required, batch_config_.memory_usage_limit)) {
        // All beams fit in memory
        batch_config_.beams_per_batch = params_.beam_count;
    } else {
        // Calculate batch size
        size_t batch_beams = static_cast<size_t>(params_.beam_count * batch_config_.batch_size_ratio);
        batch_beams = std::max(batch_beams, batch_config_.min_beams_for_batch);
        batch_beams = std::min(batch_beams, params_.beam_count);
        batch_config_.beams_per_batch = batch_beams;
    }
}

bool AntennaFFTCore::NeedsBatching() const {
    return batch_config_.beams_per_batch < params_.beam_count;
}

cl_mem AntennaFFTCore::CreateInputBuffer(const std::vector<std::complex<float>>& input_data) {
    size_t buffer_size = input_data.size() * sizeof(std::complex<float>);

    cl_int err;
    cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   buffer_size, const_cast<std::complex<float>*>(input_data.data()), &err);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create input buffer: " + std::to_string(err));
    }

    return buffer;
}

void AntennaFFTCore::CreatePreCallbackUserData(size_t num_beams) {
    // Structure: {beam_count, count_points, nFFT, padding} + input data pointer space
    // 32 bytes header for alignment

    struct PreCallbackHeader {
        cl_uint beam_count;
        cl_uint count_points;
        cl_uint nFFT;
        cl_uint padding1;
        cl_uint padding2;
        cl_uint padding3;
        cl_uint padding4;
        cl_uint padding5;
    };

    PreCallbackHeader header;
    header.beam_count = static_cast<cl_uint>(num_beams);
    header.count_points = static_cast<cl_uint>(params_.count_points);
    header.nFFT = static_cast<cl_uint>(nFFT_);
    header.padding1 = 0;
    header.padding2 = 0;
    header.padding3 = 0;
    header.padding4 = 0;
    header.padding5 = 0;

    // Total size: header (32 bytes) + input data (beam_count * count_points * complex<float>)
    size_t input_data_size = num_beams * params_.count_points * sizeof(std::complex<float>);
    size_t total_size = sizeof(PreCallbackHeader) + input_data_size;

    cl_int err;
    if (pre_callback_userdata_) {
        clReleaseMemObject(pre_callback_userdata_);
    }

    pre_callback_userdata_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, total_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create pre-callback userdata: " + std::to_string(err));
    }

    // Write header
    err = clEnqueueWriteBuffer(queue_, pre_callback_userdata_, CL_TRUE, 0,
                               sizeof(PreCallbackHeader), &header, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to write pre-callback header: " + std::to_string(err));
    }
}

void AntennaFFTCore::CreatePostCallbackUserData(size_t num_beams) {
    // Structure for post-callback: {beam_count, nFFT, out_count_points_fft, max_peaks_count}

    struct PostCallbackHeader {
        cl_uint beam_count;
        cl_uint nFFT;
        cl_uint out_count_points_fft;
        cl_uint max_peaks_count;
    };

    PostCallbackHeader header;
    header.beam_count = static_cast<cl_uint>(num_beams);
    header.nFFT = static_cast<cl_uint>(nFFT_);
    header.out_count_points_fft = static_cast<cl_uint>(params_.out_count_points_fft);
    header.max_peaks_count = static_cast<cl_uint>(params_.max_peaks_count);

    // Allocate buffer for header + output data
    size_t output_size = num_beams * params_.out_count_points_fft * sizeof(std::complex<float>);
    size_t magnitude_size = num_beams * params_.out_count_points_fft * sizeof(float);
    size_t total_size = sizeof(PostCallbackHeader) + output_size + magnitude_size;

    cl_int err;
    if (post_callback_userdata_) {
        clReleaseMemObject(post_callback_userdata_);
    }

    post_callback_userdata_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, total_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create post-callback userdata: " + std::to_string(err));
    }

    // Write header
    err = clEnqueueWriteBuffer(queue_, post_callback_userdata_, CL_TRUE, 0,
                               sizeof(PostCallbackHeader), &header, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to write post-callback header: " + std::to_string(err));
    }
}

double AntennaFFTCore::ProfileEvent(cl_event event, const std::string& operation_name) {
    cl_ulong start_time, end_time;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);

    double time_ms = (end_time - start_time) / 1000000.0;
    return time_ms;
}

void AntennaFFTCore::ReleaseFFTPlan() {
    if (plan_created_ && plan_handle_) {
        clfftDestroyPlan(&plan_handle_);
        plan_handle_ = 0;
        plan_created_ = false;
    }
}

} // namespace antenna_fft
