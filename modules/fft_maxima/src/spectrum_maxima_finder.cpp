#include "spectrum_maxima_finder.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор / Деструктор
// ════════════════════════════════════════════════════════════════════════════

SpectrumMaximaFinder::SpectrumMaximaFinder(
    const SpectrumParams& params,
    drv_gpu_lib::IBackend* backend)
    : params_(params)
    , backend_(backend) {

    if (!backend_) {
        throw std::invalid_argument("SpectrumMaximaFinder: backend cannot be null");
    }

    if (!backend_->IsInitialized()) {
        throw std::runtime_error("SpectrumMaximaFinder: backend is not initialized");
    }

    // Получаем OpenCL ресурсы из backend
    context_ = static_cast<cl_context>(backend_->GetNativeContext());
    queue_ = static_cast<cl_command_queue>(backend_->GetNativeQueue());
    device_ = static_cast<cl_device_id>(backend_->GetNativeDevice());

    if (!context_ || !queue_ || !device_) {
        throw std::runtime_error("SpectrumMaximaFinder: failed to get OpenCL resources from backend");
    }
}

SpectrumMaximaFinder::~SpectrumMaximaFinder() {
    ReleaseResources();
}

SpectrumMaximaFinder::SpectrumMaximaFinder(SpectrumMaximaFinder&& other) noexcept
    : params_(other.params_)
    , initialized_(other.initialized_)
    , backend_(other.backend_)
    , context_(other.context_)
    , queue_(other.queue_)
    , device_(other.device_)
    , plan_handle_(other.plan_handle_)
    , plan_created_(other.plan_created_)
    , pre_callback_userdata_(other.pre_callback_userdata_)
    , fft_input_(other.fft_input_)
    , fft_output_(other.fft_output_)
    , maxima_output_(other.maxima_output_)
    , post_program_(other.post_program_)
    , post_kernel_(other.post_kernel_)
    , profiling_(other.profiling_) {

    // Invalidate source
    other.initialized_ = false;
    other.plan_handle_ = 0;
    other.plan_created_ = false;
    other.pre_callback_userdata_ = nullptr;
    other.fft_input_ = nullptr;
    other.fft_output_ = nullptr;
    other.maxima_output_ = nullptr;
    other.post_program_ = nullptr;
    other.post_kernel_ = nullptr;
}

SpectrumMaximaFinder& SpectrumMaximaFinder::operator=(SpectrumMaximaFinder&& other) noexcept {
    if (this != &other) {
        ReleaseResources();

        params_ = other.params_;
        initialized_ = other.initialized_;
        backend_ = other.backend_;
        context_ = other.context_;
        queue_ = other.queue_;
        device_ = other.device_;
        plan_handle_ = other.plan_handle_;
        plan_created_ = other.plan_created_;
        pre_callback_userdata_ = other.pre_callback_userdata_;
        fft_input_ = other.fft_input_;
        fft_output_ = other.fft_output_;
        maxima_output_ = other.maxima_output_;
        post_program_ = other.post_program_;
        post_kernel_ = other.post_kernel_;
        profiling_ = other.profiling_;

        other.initialized_ = false;
        other.plan_handle_ = 0;
        other.plan_created_ = false;
        other.pre_callback_userdata_ = nullptr;
        other.fft_input_ = nullptr;
        other.fft_output_ = nullptr;
        other.maxima_output_ = nullptr;
        other.post_program_ = nullptr;
        other.post_kernel_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Публичные методы
// ════════════════════════════════════════════════════════════════════════════

void SpectrumMaximaFinder::Initialize() {
    if (initialized_) {
        return;
    }

    std::cout << "\n[SpectrumMaximaFinder] Инициализация...\n";

    // 1. Вычислить размеры FFT
    CalculateFFTSize();

    std::cout << "  📊 antenna_count: " << params_.antenna_count << "\n";
    std::cout << "  📊 n_point: " << params_.n_point << "\n";
    std::cout << "  📊 repeat_count: " << params_.repeat_count << "\n";
    std::cout << "  📊 base_fft: " << params_.base_fft << "\n";
    std::cout << "  📊 nFFT: " << params_.nFFT << "\n";
    std::cout << "  📊 search_range: " << params_.search_range << "\n";
    std::cout << "  📊 sample_rate: " << params_.sample_rate << " Hz\n";

    // 2. Создать GPU буферы
    AllocateBuffers();
    std::cout << "  ✅ Буферы созданы\n";

    // 3. Создать FFT план с pre-callback
    CreateFFTPlanWithCallback();
    std::cout << "  ✅ FFT план создан с pre-callback\n";

    // 4. Скомпилировать post-kernel
    CompilePostKernel();
    std::cout << "  ✅ Post-kernel скомпилирован\n";

    initialized_ = true;
    std::cout << "[SpectrumMaximaFinder] Инициализация завершена!\n\n";
}

std::vector<SpectrumResult> SpectrumMaximaFinder::Process(
    const std::vector<std::complex<float>>& input_data) {

    if (!initialized_) {
        throw std::runtime_error("SpectrumMaximaFinder::Process: not initialized");
    }

    size_t expected_size = params_.antenna_count * params_.n_point;
    if (input_data.size() != expected_size) {
        throw std::invalid_argument(
            "SpectrumMaximaFinder::Process: input size mismatch. "
            "Expected " + std::to_string(expected_size) +
            ", got " + std::to_string(input_data.size()));
    }

    // Сбросить профилирование
    profiling_ = ProfilingData{};

    // 1. Загрузить данные на GPU
    cl_event upload_event = UploadData(input_data);
    profiling_.upload_time_ms = ProfileEvent(upload_event, "Upload");

    // 2. Выполнить FFT с pre-callback
    cl_event fft_event = ExecuteFFT(upload_event);
    profiling_.fft_time_ms = ProfileEvent(fft_event, "FFT");
    clReleaseEvent(upload_event);

    // 3. Выполнить post-kernel
    cl_event post_event = ExecutePostKernel(fft_event);
    profiling_.post_kernel_time_ms = ProfileEvent(post_event, "PostKernel");
    clReleaseEvent(fft_event);

    // 4. Прочитать результаты
    std::vector<SpectrumResult> results = ReadResults(post_event);
    clReleaseEvent(post_event);

    // Общее время
    profiling_.total_time_ms = profiling_.upload_time_ms +
                                profiling_.fft_time_ms +
                                profiling_.post_kernel_time_ms +
                                profiling_.download_time_ms;

    return results;
}

void SpectrumMaximaFinder::PrintInfo() const {
    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  SpectrumMaximaFinder Configuration\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << std::left;
    std::cout << std::setw(25) << "  Antenna count:" << params_.antenna_count << "\n";
    std::cout << std::setw(25) << "  N points:" << params_.n_point << "\n";
    std::cout << std::setw(25) << "  Repeat count:" << params_.repeat_count << "\n";
    std::cout << std::setw(25) << "  Base FFT:" << params_.base_fft << "\n";
    std::cout << std::setw(25) << "  nFFT:" << params_.nFFT << "\n";
    std::cout << std::setw(25) << "  Search range:" << params_.search_range << "\n";
    std::cout << std::setw(25) << "  Sample rate:" << params_.sample_rate << " Hz\n";
    std::cout << std::setw(25) << "  Initialized:" << (initialized_ ? "Yes" : "No") << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Приватные методы
// ════════════════════════════════════════════════════════════════════════════

void SpectrumMaximaFinder::CalculateFFTSize() {
    // base_fft = следующая степень двойки от n_point
    params_.base_fft = NextPowerOf2(params_.n_point);

    // nFFT = base_fft * repeat_count
    params_.nFFT = params_.base_fft * params_.repeat_count;

    // search_range по умолчанию = nFFT / 4
    if (params_.search_range == 0) {
        params_.search_range = params_.nFFT / 4;
    }
}

uint32_t SpectrumMaximaFinder::NextPowerOf2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void SpectrumMaximaFinder::AllocateBuffers() {
    cl_int err;

    // 1. Pre-callback userdata: [32 bytes header][input data]
    // Header: {beam_count, count_points, nFFT, pad, pad, pad, pad, pad}
    size_t input_data_size = params_.antenna_count * params_.n_point * sizeof(std::complex<float>);
    size_t userdata_size = PRE_CALLBACK_HEADER_SIZE + input_data_size;

    pre_callback_userdata_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                             userdata_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create pre_callback_userdata buffer: " + std::to_string(err));
    }

    // Записать заголовок (32 bytes)
    struct PreCallbackHeader {
        uint32_t beam_count;
        uint32_t count_points;
        uint32_t nFFT;
        uint32_t padding1;
        uint32_t padding2;
        uint32_t padding3;
        uint32_t padding4;
        uint32_t padding5;
    };
    static_assert(sizeof(PreCallbackHeader) == 32, "PreCallbackHeader must be 32 bytes");

    PreCallbackHeader header = {
        params_.antenna_count,
        params_.n_point,
        params_.nFFT,
        0, 0, 0, 0, 0
    };

    err = clEnqueueWriteBuffer(queue_, pre_callback_userdata_, CL_TRUE,
                               0, sizeof(header), &header,
                               0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to write pre_callback header: " + std::to_string(err));
    }

    // 2. FFT буферы
    size_t fft_buffer_size = params_.antenna_count * params_.nFFT * sizeof(std::complex<float>);

    fft_input_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                 fft_buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create fft_input buffer: " + std::to_string(err));
    }

    fft_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                  fft_buffer_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create fft_output buffer: " + std::to_string(err));
    }

    // 3. Maxima output: antenna_count * 8 * sizeof(MaxValue) (4 left + 4 right)
    size_t maxima_size = params_.antenna_count * 8 * sizeof(MaxValue);
    static_assert(sizeof(MaxValue) == 32, "MaxValue must be 32 bytes");

    maxima_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                     maxima_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create maxima_output buffer: " + std::to_string(err));
    }
}

void SpectrumMaximaFinder::CreateFFTPlanWithCallback() {
    // Инициализация clFFT (если ещё не сделано)
    static bool clfft_initialized = false;
    if (!clfft_initialized) {
        clfftSetupData setup;
        // Ручная инициализация (системный clFFT.h не имеет inline clfftInitSetupData)
        setup.major = clfftVersionMajor;
        setup.minor = clfftVersionMinor;
        setup.patch = clfftVersionPatch;
        setup.debugFlags = 0;
        clfftSetup(&setup);
        clfft_initialized = true;
    }

    // Создать план
    size_t dim = params_.nFFT;
    clfftStatus status = clfftCreateDefaultPlan(&plan_handle_, context_, CLFFT_1D, &dim);
    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftCreateDefaultPlan failed: " + std::to_string(status));
    }

    // Настроить план
    clfftSetPlanPrecision(plan_handle_, CLFFT_SINGLE);
    clfftSetLayout(plan_handle_, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan_handle_, CLFFT_OUTOFPLACE);
    clfftSetPlanBatchSize(plan_handle_, params_.antenna_count);

    size_t strides[1] = {1};
    size_t dist = params_.nFFT;
    clfftSetPlanInStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanOutStride(plan_handle_, CLFFT_1D, strides);
    clfftSetPlanDistance(plan_handle_, dist, dist);

    // Регистрировать pre-callback
    const char* pre_callback_source = kernels::GetPreCallbackSource32();
    status = clfftSetPlanCallback(plan_handle_, "prepareDataPre", pre_callback_source, 0,
                                   PRECALLBACK, &pre_callback_userdata_, 1);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftSetPlanCallback (pre) failed: " + std::to_string(status));
    }

    // Bake план
    status = clfftBakePlan(plan_handle_, 1, &queue_, nullptr, nullptr);
    if (status != CLFFT_SUCCESS) {
        clfftDestroyPlan(&plan_handle_);
        throw std::runtime_error("clfftBakePlan failed: " + std::to_string(status));
    }

    plan_created_ = true;
}

void SpectrumMaximaFinder::CompilePostKernel() {
    cl_int err;

    // Получить исходный код kernel
    const char* source = kernels::GetPostKernelSource();
    size_t source_len = strlen(source);

    // Создать программу
    post_program_ = clCreateProgramWithSource(context_, 1, &source, &source_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clCreateProgramWithSource failed: " + std::to_string(err));
    }

    // Скомпилировать
    err = clBuildProgram(post_program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Получить лог ошибок
        size_t log_size;
        clGetProgramBuildInfo(post_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(post_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << "\n";
        clReleaseProgram(post_program_);
        post_program_ = nullptr;
        throw std::runtime_error("clBuildProgram failed: " + std::to_string(err));
    }

    // Создать kernel
    post_kernel_ = clCreateKernel(post_program_, "post_kernel", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(post_program_);
        post_program_ = nullptr;
        throw std::runtime_error("clCreateKernel failed: " + std::to_string(err));
    }
}

cl_event SpectrumMaximaFinder::UploadData(const std::vector<std::complex<float>>& input_data) {
    cl_event event = nullptr;
    size_t data_size = input_data.size() * sizeof(std::complex<float>);

    // Записать данные в userdata после заголовка (offset = 32)
    cl_int err = clEnqueueWriteBuffer(
        queue_,
        pre_callback_userdata_,
        CL_FALSE,  // Non-blocking
        PRE_CALLBACK_HEADER_SIZE,  // Offset после заголовка
        data_size,
        input_data.data(),
        0, nullptr,
        &event
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("UploadData failed: " + std::to_string(err));
    }

    return event;
}

cl_event SpectrumMaximaFinder::ExecuteFFT(cl_event wait_event) {
    cl_event event = nullptr;

    // Выполнить FFT с pre-callback
    clfftStatus status = clfftEnqueueTransform(
        plan_handle_,
        CLFFT_FORWARD,
        1, &queue_,
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &event,
        &fft_input_,    // Input (pre-callback читает из userdata)
        &fft_output_,   // Output
        nullptr         // Temp buffer
    );

    if (status != CLFFT_SUCCESS) {
        throw std::runtime_error("clfftEnqueueTransform failed: " + std::to_string(status));
    }

    return event;
}

cl_event SpectrumMaximaFinder::ExecutePostKernel(cl_event wait_event) {
    cl_int err;
    cl_event event = nullptr;

    // Установить аргументы kernel
    err = clSetKernelArg(post_kernel_, 0, sizeof(cl_mem), &fft_output_);
    err |= clSetKernelArg(post_kernel_, 1, sizeof(cl_mem), &maxima_output_);
    err |= clSetKernelArg(post_kernel_, 2, sizeof(uint32_t), &params_.antenna_count);
    err |= clSetKernelArg(post_kernel_, 3, sizeof(uint32_t), &params_.nFFT);
    err |= clSetKernelArg(post_kernel_, 4, sizeof(uint32_t), &params_.search_range);
    err |= clSetKernelArg(post_kernel_, 5, sizeof(float), &params_.sample_rate);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("clSetKernelArg failed: " + std::to_string(err));
    }

    // NDRange: каждая work-group = одна антена
    size_t global_size = params_.antenna_count * LOCAL_SIZE;
    size_t local_size = LOCAL_SIZE;

    err = clEnqueueNDRangeKernel(
        queue_,
        post_kernel_,
        1,
        nullptr,
        &global_size,
        &local_size,
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &event
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("clEnqueueNDRangeKernel failed: " + std::to_string(err));
    }

    return event;
}

std::vector<SpectrumResult> SpectrumMaximaFinder::ReadResults(cl_event wait_event) {
    // Читаем 8 MaxValue на луч
    size_t num_results = params_.antenna_count * 8;
    std::vector<MaxValue> raw_results(num_results);

    cl_event read_event = nullptr;
    cl_int err = clEnqueueReadBuffer(
        queue_,
        maxima_output_,
        CL_FALSE,  // Non-blocking
        0,
        num_results * sizeof(MaxValue),
        raw_results.data(),
        (wait_event ? 1 : 0), (wait_event ? &wait_event : nullptr),
        &read_event
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("ReadResults failed: " + std::to_string(err));
    }

    // Ждём завершения
    clWaitForEvents(1, &read_event);
    profiling_.download_time_ms = ProfileEvent(read_event, "Download");
    clReleaseEvent(read_event);

    // Преобразуем в vector<SpectrumResult>: [left0, right0, left1, right1, ...]
    std::vector<SpectrumResult> results;
    results.reserve(params_.antenna_count * 2);

    for (uint32_t i = 0; i < params_.antenna_count; ++i) {
        size_t base = i * 8;

        SpectrumResult left{};
        left.antenna_id = i;
        left.interpolated = raw_results[base + 0];
        left.left_point = raw_results[base + 1];
        left.center_point = raw_results[base + 2];
        left.right_point = raw_results[base + 3];
        results.push_back(left);

        SpectrumResult right{};
        right.antenna_id = i;
        right.interpolated = raw_results[base + 4];
        right.left_point = raw_results[base + 5];
        right.center_point = raw_results[base + 6];
        right.right_point = raw_results[base + 7];
        results.push_back(right);
    }

    return results;
}

double SpectrumMaximaFinder::ProfileEvent(cl_event event, const char* name) {
    if (!event) return 0.0;

    // Ждём завершения события
    clWaitForEvents(1, &event);

    cl_ulong start = 0, end = 0;
    cl_int err1 = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                           sizeof(cl_ulong), &start, nullptr);
    cl_int err2 = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                           sizeof(cl_ulong), &end, nullptr);

    if (err1 != CL_SUCCESS || err2 != CL_SUCCESS) {
        // Профилирование недоступно (queue без CL_QUEUE_PROFILING_ENABLE)
        return 0.0;
    }

    double time_ms = (end - start) / 1e6;
    return time_ms;
}

void SpectrumMaximaFinder::ReleaseResources() {
    // Post-kernel
    if (post_kernel_) {
        clReleaseKernel(post_kernel_);
        post_kernel_ = nullptr;
    }
    if (post_program_) {
        clReleaseProgram(post_program_);
        post_program_ = nullptr;
    }

    // FFT план
    if (plan_created_ && plan_handle_) {
        clfftDestroyPlan(&plan_handle_);
        plan_handle_ = 0;
        plan_created_ = false;
    }

    // Буферы
    if (pre_callback_userdata_) {
        clReleaseMemObject(pre_callback_userdata_);
        pre_callback_userdata_ = nullptr;
    }
    if (fft_input_) {
        clReleaseMemObject(fft_input_);
        fft_input_ = nullptr;
    }
    if (fft_output_) {
        clReleaseMemObject(fft_output_);
        fft_output_ = nullptr;
    }
    if (maxima_output_) {
        clReleaseMemObject(maxima_output_);
        maxima_output_ = nullptr;
    }

    initialized_ = false;
}

} // namespace antenna_fft
