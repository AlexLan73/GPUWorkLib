#include "antenna_fft_debug.h"
#include "fft_logger.h"
#include "kernels/fft_kernel_sources.hpp"
#include <cstring>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

AntennaFFTDebug::AntennaFFTDebug(const AntennaFFTParams& params, drv_gpu_lib::IBackend* backend)
    : AntennaFFTCore(params, backend),
      buffer_input_(nullptr),
      buffer_selected_complex_(nullptr),
      buffer_selected_magnitude_(nullptr),
      padding_kernel_(nullptr),
      post_kernel_(nullptr),
      maxima_kernel_(nullptr),
      padding_program_(nullptr),
      post_program_(nullptr),
      maxima_program_(nullptr),
      plan_num_beams_(0),
      debug_num_beams_(0),
      input_data_set_(false) {

    // Call virtual Initialize (create kernels and plan)
    Initialize();
}

AntennaFFTDebug::~AntennaFFTDebug() {
    ReleaseBuffers();

    // Release kernels
    if (padding_kernel_) clReleaseKernel(padding_kernel_);
    if (post_kernel_) clReleaseKernel(post_kernel_);
    if (maxima_kernel_) clReleaseKernel(maxima_kernel_);

    // Release programs
    if (padding_program_) clReleaseProgram(padding_program_);
    if (post_program_) clReleaseProgram(post_program_);
    if (maxima_program_) clReleaseProgram(maxima_program_);
}

// ════════════════════════════════════════════════════════════════════════════
// Virtual method implementations
// ════════════════════════════════════════════════════════════════════════════

void AntennaFFTDebug::Initialize() {
    FFTLogger::Info("[AntennaFFTDebug] Initializing with step-by-step kernels...");

    // Create kernels
    CreatePaddingKernel();
    CreatePostKernel();
    CreateMaximaKernel();

    // Allocate buffers for initial batch size
    size_t initial_beams = batch_config_.beams_per_batch;
    if (initial_beams == 0) initial_beams = params_.beam_count;

    AllocateBuffers(initial_beams);
    CreateFFTPlanNoCallbacks(initial_beams);

    FFTLogger::Info("[AntennaFFTDebug] Initialized!");
}

AntennaFFTResult AntennaFFTDebug::ProcessSingleBatch(cl_mem input_signal) {
    AntennaFFTResult result;
    result.total_beams = params_.beam_count;
    result.nFFT = nFFT_;
    result.task_id = params_.task_id;
    result.module_name = params_.module_name;

    // Set input data
    SetInputData(input_signal, params_.beam_count);

    // Step 1: Padding
    cl_event padding_event;
    ExecutePaddingKernel(nullptr, &padding_event);

    // Step 2: FFT
    cl_event fft_event;
    ExecuteFFTOnly(padding_event, &fft_event);

    // Step 3: Post-processing
    cl_event post_event;
    ExecutePostKernel(fft_event, &post_event);

    // Step 4: Find maxima
    clWaitForEvents(1, &post_event);
    auto maxima = FindMaximaOnGPU();

    // Profile
    last_profiling_results_.pre_callback_time_ms = ProfileEvent(padding_event, "Padding");
    last_profiling_results_.fft_time_ms = ProfileEvent(fft_event, "FFT");
    last_profiling_results_.post_callback_time_ms = ProfileEvent(post_event, "Post");

    clReleaseEvent(padding_event);
    clReleaseEvent(fft_event);
    clReleaseEvent(post_event);

    // Convert to results
    result.results = ConvertMaximaToResults(maxima, 0);

    return result;
}

std::vector<FFTResult> AntennaFFTDebug::ProcessBatch(
    cl_mem input_signal,
    size_t start_beam,
    size_t num_beams,
    BatchProfilingData* out_profiling) {

    // Ensure buffers are ready
    if (current_buffer_beams_ < num_beams) {
        ReleaseBuffers();
        AllocateBuffers(num_beams);
    }
    if (plan_num_beams_ != num_beams) {
        ReleaseFFTPlan();
        CreateFFTPlanNoCallbacks(num_beams);
    }

    // Copy batch input data
    size_t input_offset = start_beam * params_.count_points * sizeof(std::complex<float>);
    size_t input_size = num_beams * params_.count_points * sizeof(std::complex<float>);

    cl_int err = clEnqueueCopyBuffer(queue_, input_signal, buffer_input_,
                                      input_offset, 0, input_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to copy batch input: " + std::to_string(err));
    }

    debug_num_beams_ = num_beams;
    input_data_set_ = true;

    // Step 1: Padding
    cl_event padding_event;
    ExecutePaddingKernel(nullptr, &padding_event);

    // Step 2: FFT
    cl_event fft_event;
    ExecuteFFTOnly(padding_event, &fft_event);

    // Step 3: Post-processing
    cl_event post_event;
    ExecutePostKernel(fft_event, &post_event);

    // Step 4: Find maxima
    clWaitForEvents(1, &post_event);
    auto maxima = FindMaximaOnGPU();

    // Profile
    if (out_profiling) {
        out_profiling->padding_time_ms = ProfileEvent(padding_event, "Padding");
        out_profiling->fft_time_ms = ProfileEvent(fft_event, "FFT");
        out_profiling->post_time_ms = ProfileEvent(post_event, "Post");
    }

    clReleaseEvent(padding_event);
    clReleaseEvent(fft_event);
    clReleaseEvent(post_event);

    // Convert to results
    return ConvertMaximaToResults(maxima, start_beam);
}

void AntennaFFTDebug::AllocateBuffers(size_t num_beams) {
    if (current_buffer_beams_ >= num_beams) return;

    ReleaseBuffers();

    cl_int err;

    // Input buffer
    size_t input_size = params_.count_points * num_beams * sizeof(std::complex<float>);
    buffer_input_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, input_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate input buffer");

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
    size_t maxima_size = params_.max_peaks_count * num_beams * 32;
    buffer_maxima_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, maxima_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to allocate maxima buffer");

    current_buffer_beams_ = num_beams;
    FFTLogger::Info("  [Debug] Allocated buffers for ", num_beams, " beams");
}

void AntennaFFTDebug::ReleaseBuffers() {
    if (buffer_input_) { clReleaseMemObject(buffer_input_); buffer_input_ = nullptr; }
    if (buffer_fft_input_) { clReleaseMemObject(buffer_fft_input_); buffer_fft_input_ = nullptr; }
    if (buffer_fft_output_) { clReleaseMemObject(buffer_fft_output_); buffer_fft_output_ = nullptr; }
    if (buffer_selected_complex_) { clReleaseMemObject(buffer_selected_complex_); buffer_selected_complex_ = nullptr; }
    if (buffer_selected_magnitude_) { clReleaseMemObject(buffer_selected_magnitude_); buffer_selected_magnitude_ = nullptr; }
    if (buffer_maxima_) { clReleaseMemObject(buffer_maxima_); buffer_maxima_ = nullptr; }
    current_buffer_beams_ = 0;
    input_data_set_ = false;
}

// ════════════════════════════════════════════════════════════════════════════
// Debug methods - step-by-step execution
// ════════════════════════════════════════════════════════════════════════════

void AntennaFFTDebug::SetInputData(const std::vector<std::complex<float>>& input_data) {
    size_t expected_size = params_.beam_count * params_.count_points;
    if (input_data.size() != expected_size) {
        throw std::runtime_error("Input data size mismatch: expected " +
                                 std::to_string(expected_size) + ", got " +
                                 std::to_string(input_data.size()));
    }

    // Ensure buffers allocated
    if (current_buffer_beams_ < params_.beam_count) {
        AllocateBuffers(params_.beam_count);
    }

    // Copy to GPU
    size_t size = input_data.size() * sizeof(std::complex<float>);
    cl_int err = clEnqueueWriteBuffer(queue_, buffer_input_, CL_TRUE, 0, size,
                                       input_data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to write input data: " + std::to_string(err));
    }

    debug_num_beams_ = params_.beam_count;
    input_data_set_ = true;
}

void AntennaFFTDebug::SetInputData(cl_mem input_signal, size_t num_beams) {
    if (current_buffer_beams_ < num_beams) {
        AllocateBuffers(num_beams);
    }

    // Copy from input signal to our buffer
    size_t size = num_beams * params_.count_points * sizeof(std::complex<float>);
    cl_int err = clEnqueueCopyBuffer(queue_, input_signal, buffer_input_,
                                      0, 0, size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to copy input data: " + std::to_string(err));
    }

    debug_num_beams_ = num_beams;
    input_data_set_ = true;
}

void AntennaFFTDebug::ExecutePaddingKernel(cl_event wait_event, cl_event* out_event) {
    if (!input_data_set_) {
        throw std::runtime_error("Input data not set! Call SetInputData first.");
    }

    // Set kernel arguments
    cl_uint beam_count = static_cast<cl_uint>(debug_num_beams_);
    cl_uint count_points = static_cast<cl_uint>(params_.count_points);
    cl_uint nfft = static_cast<cl_uint>(nFFT_);
    cl_uint beam_offset = 0;

    clSetKernelArg(padding_kernel_, 0, sizeof(cl_mem), &buffer_input_);
    clSetKernelArg(padding_kernel_, 1, sizeof(cl_mem), &buffer_fft_input_);
    clSetKernelArg(padding_kernel_, 2, sizeof(cl_uint), &beam_count);
    clSetKernelArg(padding_kernel_, 3, sizeof(cl_uint), &count_points);
    clSetKernelArg(padding_kernel_, 4, sizeof(cl_uint), &nfft);
    clSetKernelArg(padding_kernel_, 5, sizeof(cl_uint), &beam_offset);

    // Execute kernel
    size_t global_size = debug_num_beams_ * nFFT_;
    size_t local_size = 256;
    global_size = ((global_size + local_size - 1) / local_size) * local_size;

    cl_uint num_wait = (wait_event != nullptr) ? 1 : 0;
    cl_event* wait_list = (wait_event != nullptr) ? &wait_event : nullptr;

    cl_int err = clEnqueueNDRangeKernel(queue_, padding_kernel_, 1, nullptr,
                                         &global_size, &local_size,
                                         num_wait, wait_list, out_event);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to execute padding kernel: " + std::to_string(err));
    }
}

void AntennaFFTDebug::ExecuteFFTOnly(cl_event wait_event, cl_event* out_event) {
    if (!plan_created_) {
        CreateFFTPlanNoCallbacks(debug_num_beams_);
    }

    cl_uint num_wait = (wait_event != nullptr) ? 1 : 0;
    cl_event* wait_list = (wait_event != nullptr) ? &wait_event : nullptr;

    clfftStatus status = clfftEnqueueTransform(
        plan_handle_,
        CLFFT_FORWARD,
        1, &queue_,
        num_wait, wait_list,
        out_event,
        &buffer_fft_input_,
        &buffer_fft_output_,
        nullptr
    );

    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftEnqueueTransform failed: " + std::to_string(status));
    }
}

void AntennaFFTDebug::ExecutePostKernel(cl_event wait_event, cl_event* out_event) {
    // Set kernel arguments
    cl_uint beam_count = static_cast<cl_uint>(debug_num_beams_);
    cl_uint nfft = static_cast<cl_uint>(nFFT_);
    cl_uint out_points = static_cast<cl_uint>(params_.out_count_points_fft);

    clSetKernelArg(post_kernel_, 0, sizeof(cl_mem), &buffer_fft_output_);
    clSetKernelArg(post_kernel_, 1, sizeof(cl_mem), &buffer_selected_complex_);
    clSetKernelArg(post_kernel_, 2, sizeof(cl_mem), &buffer_selected_magnitude_);
    clSetKernelArg(post_kernel_, 3, sizeof(cl_uint), &beam_count);
    clSetKernelArg(post_kernel_, 4, sizeof(cl_uint), &nfft);
    clSetKernelArg(post_kernel_, 5, sizeof(cl_uint), &out_points);

    // Execute kernel
    size_t global_size = debug_num_beams_ * params_.out_count_points_fft;
    size_t local_size = 256;
    global_size = ((global_size + local_size - 1) / local_size) * local_size;

    cl_uint num_wait = (wait_event != nullptr) ? 1 : 0;
    cl_event* wait_list = (wait_event != nullptr) ? &wait_event : nullptr;

    cl_int err = clEnqueueNDRangeKernel(queue_, post_kernel_, 1, nullptr,
                                         &global_size, &local_size,
                                         num_wait, wait_list, out_event);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to execute post kernel: " + std::to_string(err));
    }
}

std::vector<std::vector<FFTMaxResult>> AntennaFFTDebug::FindMaximaOnGPU(cl_event wait_event) {
    // Set kernel arguments
    cl_uint beam_count = static_cast<cl_uint>(debug_num_beams_);
    cl_uint search_range = static_cast<cl_uint>(params_.out_count_points_fft);
    cl_uint max_peaks = static_cast<cl_uint>(params_.max_peaks_count);

    clSetKernelArg(maxima_kernel_, 0, sizeof(cl_mem), &buffer_selected_magnitude_);
    clSetKernelArg(maxima_kernel_, 1, sizeof(cl_mem), &buffer_selected_complex_);
    clSetKernelArg(maxima_kernel_, 2, sizeof(cl_mem), &buffer_maxima_);
    clSetKernelArg(maxima_kernel_, 3, sizeof(cl_uint), &beam_count);
    clSetKernelArg(maxima_kernel_, 4, sizeof(cl_uint), &search_range);
    clSetKernelArg(maxima_kernel_, 5, sizeof(cl_uint), &max_peaks);

    // Execute kernel (one workgroup per beam)
    size_t global_size = debug_num_beams_ * 256;
    size_t local_size = 256;

    cl_uint num_wait = (wait_event != nullptr) ? 1 : 0;
    cl_event* wait_list = (wait_event != nullptr) ? &wait_event : nullptr;

    cl_event kernel_event;
    cl_int err = clEnqueueNDRangeKernel(queue_, maxima_kernel_, 1, nullptr,
                                         &global_size, &local_size,
                                         num_wait, wait_list, &kernel_event);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to execute maxima kernel: " + std::to_string(err));
    }

    clWaitForEvents(1, &kernel_event);
    clReleaseEvent(kernel_event);

    // Read results
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

    size_t total_maxima = debug_num_beams_ * params_.max_peaks_count;
    std::vector<MaxValue> raw_maxima(total_maxima);

    err = clEnqueueReadBuffer(queue_, buffer_maxima_, CL_TRUE, 0,
                               total_maxima * sizeof(MaxValue),
                               raw_maxima.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read maxima: " + std::to_string(err));
    }

    // Convert to result format
    std::vector<std::vector<FFTMaxResult>> results(debug_num_beams_);
    for (size_t beam = 0; beam < debug_num_beams_; ++beam) {
        results[beam].reserve(params_.max_peaks_count);
        for (size_t peak = 0; peak < params_.max_peaks_count; ++peak) {
            const MaxValue& mv = raw_maxima[beam * params_.max_peaks_count + peak];
            FFTMaxResult result;
            result.index_point = mv.index;
            result.real = mv.real;
            result.imag = mv.imag;
            result.amplitude = mv.magnitude;
            result.phase = mv.phase;
            results[beam].push_back(result);
        }
    }

    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// Buffer access for debugging
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> AntennaFFTDebug::ReadFFTInputBuffer() {
    size_t size = debug_num_beams_ * nFFT_;
    std::vector<std::complex<float>> data(size);

    cl_int err = clEnqueueReadBuffer(queue_, buffer_fft_input_, CL_TRUE, 0,
                                      size * sizeof(std::complex<float>),
                                      data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read FFT input buffer");
    }

    return data;
}

std::vector<std::complex<float>> AntennaFFTDebug::ReadFFTOutputBuffer() {
    size_t size = debug_num_beams_ * nFFT_;
    std::vector<std::complex<float>> data(size);

    cl_int err = clEnqueueReadBuffer(queue_, buffer_fft_output_, CL_TRUE, 0,
                                      size * sizeof(std::complex<float>),
                                      data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read FFT output buffer");
    }

    return data;
}

std::vector<std::complex<float>> AntennaFFTDebug::ReadSelectedComplexBuffer() {
    size_t size = debug_num_beams_ * params_.out_count_points_fft;
    std::vector<std::complex<float>> data(size);

    cl_int err = clEnqueueReadBuffer(queue_, buffer_selected_complex_, CL_TRUE, 0,
                                      size * sizeof(std::complex<float>),
                                      data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read selected complex buffer");
    }

    return data;
}

std::vector<float> AntennaFFTDebug::ReadSelectedMagnitudeBuffer() {
    size_t size = debug_num_beams_ * params_.out_count_points_fft;
    std::vector<float> data(size);

    cl_int err = clEnqueueReadBuffer(queue_, buffer_selected_magnitude_, CL_TRUE, 0,
                                      size * sizeof(float),
                                      data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read selected magnitude buffer");
    }

    return data;
}

// ════════════════════════════════════════════════════════════════════════════
// Private methods
// ════════════════════════════════════════════════════════════════════════════

void AntennaFFTDebug::CreateFFTPlanNoCallbacks(size_t num_beams) {
    if (plan_created_ && plan_num_beams_ == num_beams) return;

    ReleaseFFTPlan();

    FFTLogger::Info("  [Debug] Creating FFT plan (no callbacks) for ", num_beams, " beams...");

    size_t dim = nFFT_;
    clfftStatus status = clfftCreateDefaultPlan(&plan_handle_, context_, CLFFT_1D, &dim);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed: " + std::to_string(status));
    }

    clfftSetPlanPrecision(plan_handle_, CLFFT_SINGLE);
    clfftSetLayout(plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle_, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle_, num_beams);

    size_t strides[1] = {1};
    size_t dist = nFFT_;
    clfftSetPlanInStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle_, dist, dist);

    // NO callbacks!

    status = clfftBakePlan(plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftBakePlan failed: " + std::to_string(status));
    }

    plan_created_ = true;
    plan_num_beams_ = num_beams;

    FFTLogger::Info("  [Debug] FFT plan created!");
}

void AntennaFFTDebug::CreatePaddingKernel() {
    const char* source = kernels::GetPaddingKernelSource();

    cl_int err;
    padding_program_ = clCreateProgramWithSource(context_, 1, &source, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create padding program");
    }

    err = clBuildProgram(padding_program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(padding_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(padding_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        FFTLogger::Error("Padding kernel build error:\n", log.data());
        throw std::runtime_error("Failed to build padding program");
    }

    padding_kernel_ = clCreateKernel(padding_program_, "padding_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create padding kernel");
    }

    FFTLogger::Info("  [Debug] Padding kernel created");
}

void AntennaFFTDebug::CreatePostKernel() {
    // Используем специальное ядро для Debug версии (fftshift + magnitude без поиска максимумов)
    const char* source = kernels::GetDebugPostKernelSource();

    cl_int err;
    post_program_ = clCreateProgramWithSource(context_, 1, &source, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create post program");
    }

    err = clBuildProgram(post_program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(post_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(post_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        FFTLogger::Error("Post kernel build error:\n", log.data());
        throw std::runtime_error("Failed to build post program");
    }

    post_kernel_ = clCreateKernel(post_program_, "debug_post_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create post kernel");
    }

    FFTLogger::Info("  [Debug] Post kernel created");
}

void AntennaFFTDebug::CreateMaximaKernel() {
    // Simple maxima search kernel
    const char* source = R"(
        typedef struct {
            uint index;
            float real;
            float imag;
            float magnitude;
            float phase;
            float freq_offset;
            float refined_frequency;
            uint pad;
        } MaxValue;

        __kernel void maxima_kernel(
            __global const float* magnitudes,
            __global const float2* complex_values,
            __global MaxValue* maxima,
            uint beam_count,
            uint search_range,
            uint max_peaks)
        {
            uint beam_idx = get_group_id(0);
            if (beam_idx >= beam_count) return;

            uint local_id = get_local_id(0);
            uint local_size = get_local_size(0);

            // Each beam processes its own data
            __global const float* beam_mag = magnitudes + beam_idx * search_range;
            __global const float2* beam_complex = complex_values + beam_idx * search_range;
            __global MaxValue* beam_maxima = maxima + beam_idx * max_peaks;

            // Simple sequential search for max_peaks maxima
            // (This is a simplified version - real implementation uses parallel reduction)
            if (local_id == 0) {
                // Initialize with minimum values
                for (uint p = 0; p < max_peaks; ++p) {
                    beam_maxima[p].index = 0;
                    beam_maxima[p].magnitude = -1.0f;
                }

                // Find maxima
                for (uint i = 0; i < search_range; ++i) {
                    float mag = beam_mag[i];

                    // Check if this is larger than smallest maximum
                    uint min_idx = 0;
                    float min_mag = beam_maxima[0].magnitude;
                    for (uint p = 1; p < max_peaks; ++p) {
                        if (beam_maxima[p].magnitude < min_mag) {
                            min_mag = beam_maxima[p].magnitude;
                            min_idx = p;
                        }
                    }

                    if (mag > min_mag) {
                        beam_maxima[min_idx].index = i;
                        beam_maxima[min_idx].magnitude = mag;
                        beam_maxima[min_idx].real = beam_complex[i].x;
                        beam_maxima[min_idx].imag = beam_complex[i].y;
                        beam_maxima[min_idx].phase = atan2(beam_complex[i].y, beam_complex[i].x) * 57.2957795f;
                        beam_maxima[min_idx].freq_offset = 0.0f;
                        beam_maxima[min_idx].refined_frequency = 0.0f;
                    }
                }

                // Sort by magnitude (descending)
                for (uint i = 0; i < max_peaks - 1; ++i) {
                    for (uint j = i + 1; j < max_peaks; ++j) {
                        if (beam_maxima[j].magnitude > beam_maxima[i].magnitude) {
                            MaxValue tmp = beam_maxima[i];
                            beam_maxima[i] = beam_maxima[j];
                            beam_maxima[j] = tmp;
                        }
                    }
                }
            }
        }
    )";

    cl_int err;
    maxima_program_ = clCreateProgramWithSource(context_, 1, &source, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create maxima program");
    }

    err = clBuildProgram(maxima_program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(maxima_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(maxima_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        FFTLogger::Error("Maxima kernel build error:\n", log.data());
        throw std::runtime_error("Failed to build maxima program");
    }

    maxima_kernel_ = clCreateKernel(maxima_program_, "maxima_kernel", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create maxima kernel");
    }

    FFTLogger::Info("  [Debug] Maxima kernel created");
}

std::vector<FFTResult> AntennaFFTDebug::ConvertMaximaToResults(
    const std::vector<std::vector<FFTMaxResult>>& maxima,
    size_t start_beam) {

    std::vector<FFTResult> results;
    results.reserve(maxima.size());

    for (size_t i = 0; i < maxima.size(); ++i) {
        FFTResult result(params_.out_count_points_fft, params_.task_id, params_.module_name);
        result.max_values = maxima[i];

        if (!result.max_values.empty()) {
            result.freq_offset = 0.0f;  // Could add parabolic interpolation here
            result.refined_frequency = 0.0f;
        }

        results.push_back(std::move(result));
    }

    return results;
}

} // namespace antenna_fft
