#include "opencl_compute_engine.hpp"
#include "gpu_memory_buffer.hpp"  // Определение класса
#include "command_queue_pool.hpp"  // Для получения command queue
// ПРИМЕЧАНИЕ: Реализация GPUMemoryBuffer находится в src/GPU/gpu_memory_manager.cpp
// Здесь оставлены только методы, специфичные для opencl_compute_engine
#include <iostream>
#include <iomanip>
#include <sstream>

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// GPUMemoryBuffer реализация ПЕРЕНЕСЕНА в src/GPU/gpu_memory_manager.cpp
// Основная реализация там, чтобы избежать дублирования
// ════════════════════════════════════════════════════════════════════════════

/*
// Реализация GPUMemoryBuffer закомментирована - используется из gpu_memory_manager.cpp
GPUMemoryBuffer::GPUMemoryBuffer(
    cl_context context,
    cl_command_queue queue,
    size_t num_elements,
    MemoryType type)
    : context_(context),
      queue_(queue),
      gpu_buffer_(nullptr),
      num_elements_(num_elements),
      type_(type),
      is_external_buffer_(false),
      gpu_dirty_(false) {
    AllocateGPUBuffer();
    AllocatePinnedHostBuffer();
}

GPUMemoryBuffer::GPUMemoryBuffer(
    cl_context context,
    cl_command_queue queue,
    cl_mem external_gpu_buffer,
    size_t num_elements,
    MemoryType type)
    : context_(context),
      queue_(queue),
      gpu_buffer_(external_gpu_buffer),
      num_elements_(num_elements),
      type_(type),
      is_external_buffer_(true),
      gpu_dirty_(false) {
    AllocatePinnedHostBuffer();
}

GPUMemoryBuffer::GPUMemoryBuffer(
    cl_context context,
    cl_command_queue queue,
    const void* host_data,
    size_t data_size_bytes,
    size_t num_elements,
    MemoryType type)
    : context_(context),
      queue_(queue),
      gpu_buffer_(nullptr),
      num_elements_(num_elements),
      type_(type),
      is_external_buffer_(false),
      gpu_dirty_(false) {
    if (!host_data) {
        throw std::invalid_argument("host_data cannot be nullptr");
    }

    cl_int err;
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;

    if (type_ == MemoryType::GPU_READ_ONLY) {
        flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    } else if (type_ == MemoryType::GPU_WRITE_ONLY) {
        flags = CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR;
    }

    gpu_buffer_ = clCreateBuffer(
        context_, flags, data_size_bytes, const_cast<void*>(host_data), &err);
    CheckCLError(err, "clCreateBuffer with CL_MEM_COPY_HOST_PTR");

    AllocatePinnedHostBuffer();
}

void GPUMemoryBuffer::AllocateGPUBuffer() {
    cl_int err;
    cl_mem_flags flags = CL_MEM_READ_WRITE;

    if (type_ == MemoryType::GPU_READ_ONLY) {
        flags = CL_MEM_READ_ONLY;
    } else if (type_ == MemoryType::GPU_WRITE_ONLY) {
        flags = CL_MEM_WRITE_ONLY;
    }

    gpu_buffer_ = clCreateBuffer(
        context_, flags, num_elements_ * sizeof(std::complex<float>), nullptr, &err);
    CheckCLError(err, "clCreateBuffer");
}

void GPUMemoryBuffer::AllocatePinnedHostBuffer() {
    pinned_host_buffer_.resize(num_elements_);
}

void GPUMemoryBuffer::ReleasePinnedHostBuffer() {
    pinned_host_buffer_.clear();
    pinned_host_buffer_.shrink_to_fit();
}

GPUMemoryBuffer::~GPUMemoryBuffer() {
    if (!is_external_buffer_ && gpu_buffer_) {
        clReleaseMemObject(gpu_buffer_);
        gpu_buffer_ = nullptr;
    }
    ReleasePinnedHostBuffer();
}

GPUMemoryBuffer::GPUMemoryBuffer(GPUMemoryBuffer&& other) noexcept
    : context_(other.context_),
      queue_(other.queue_),
      gpu_buffer_(other.gpu_buffer_),
      pinned_host_buffer_(std::move(other.pinned_host_buffer_)),
      num_elements_(other.num_elements_),
      type_(other.type_),
      is_external_buffer_(other.is_external_buffer_),
      gpu_dirty_(other.gpu_dirty_) {
    other.gpu_buffer_ = nullptr;
}

GPUMemoryBuffer& GPUMemoryBuffer::operator=(GPUMemoryBuffer&& other) noexcept {
    if (this != &other) {
        if (!is_external_buffer_ && gpu_buffer_) {
            clReleaseMemObject(gpu_buffer_);
        }
        ReleasePinnedHostBuffer();

        context_ = other.context_;
        queue_ = other.queue_;
        gpu_buffer_ = other.gpu_buffer_;
        pinned_host_buffer_ = std::move(other.pinned_host_buffer_);
        num_elements_ = other.num_elements_;
        type_ = other.type_;
        is_external_buffer_ = other.is_external_buffer_;
        gpu_dirty_ = other.gpu_dirty_;

        other.gpu_buffer_ = nullptr;
    }
    return *this;
}

std::vector<std::complex<float>> GPUMemoryBuffer::ReadFromGPU() {
    return ReadPartial(num_elements_);
}

std::vector<std::complex<float>> GPUMemoryBuffer::ReadPartial(size_t num_elements) {
    if (num_elements > num_elements_) {
        throw std::runtime_error("Reading more elements than buffer size");
    }

    std::vector<std::complex<float>> result(num_elements);

    cl_int err = clEnqueueReadBuffer(
        queue_, gpu_buffer_, CL_TRUE, 0,
        num_elements * sizeof(std::complex<float>), result.data(),
        0, nullptr, nullptr);

    CheckCLError(err, "clEnqueueReadBuffer");
    gpu_dirty_ = false;
    return result;
}

void GPUMemoryBuffer::WriteToGPU(const std::vector<std::complex<float>>& data) {
    if (data.size() > num_elements_) {
        throw std::runtime_error("Writing more data than buffer size");
    }

    cl_int err = clEnqueueWriteBuffer(
        queue_, gpu_buffer_, CL_TRUE, 0,
        data.size() * sizeof(std::complex<float>), data.data(),
        0, nullptr, nullptr);

    CheckCLError(err, "clEnqueueWriteBuffer");
    gpu_dirty_ = true;
}

std::pair<std::vector<std::complex<float>>, cl_event>
GPUMemoryBuffer::ReadFromGPUAsync() {
    std::vector<std::complex<float>> result(num_elements_);
    cl_event event;

    cl_int err = clEnqueueReadBuffer(
        queue_, gpu_buffer_, CL_FALSE, 0,
        num_elements_ * sizeof(std::complex<float>), result.data(),
        0, nullptr, &event);

    CheckCLError(err, "clEnqueueReadBuffer (async)");
    gpu_dirty_ = false;
    return {result, event};
}

cl_event GPUMemoryBuffer::WriteToGPUAsync(
    const std::vector<std::complex<float>>& data) {
    if (data.size() > num_elements_) {
        throw std::runtime_error("Writing more data than buffer size");
    }

    cl_event event;
    cl_int err = clEnqueueWriteBuffer(
        queue_, gpu_buffer_, CL_FALSE, 0,
        data.size() * sizeof(std::complex<float>), data.data(),
        0, nullptr, &event);

    CheckCLError(err, "clEnqueueWriteBuffer (async)");
    gpu_dirty_ = true;
    return event;
}

void GPUMemoryBuffer::PrintStats() const {
    std::cout << "\nGPUMemoryBuffer Stats:\n";
    std::cout << " Num Elements: " << num_elements_ << "\n";
    std::cout << " Memory (MB): " << std::fixed << std::setprecision(2)
              << (GetSizeBytes() / (1024.0 * 1024.0)) << "\n";
    std::cout << " External: " << (is_external_buffer_ ? "YES" : "NO") << "\n";
    std::cout << " GPU Dirty: " << (gpu_dirty_ ? "YES" : "NO") << "\n";
    std::cout << " Type: ";
    switch (type_) {
        case MemoryType::GPU_READ_ONLY:
            std::cout << "READ_ONLY\n";
            break;
        case MemoryType::GPU_WRITE_ONLY:
            std::cout << "WRITE_ONLY\n";
            break;
        case MemoryType::GPU_READ_WRITE:
            std::cout << "READ_WRITE\n";
            break;
    }
}
*/ // Конец закомментированной реализации GPUMemoryBuffer

// ════════════════════════════════════════════════════════════════════════════
// OpenCLComputeEngine реализация
// ════════════════════════════════════════════════════════════════════════════

std::unique_ptr<OpenCLComputeEngine> OpenCLComputeEngine::instance_ = nullptr;
bool OpenCLComputeEngine::initialized_ = false;
std::mutex OpenCLComputeEngine::initialization_mutex_;

OpenCLComputeEngine::OpenCLComputeEngine()
    : total_allocated_bytes_(0),
      num_buffers_(0),
      kernel_executions_(0) {
}

void OpenCLComputeEngine::Initialize(DeviceType device_type) {
    std::lock_guard<std::mutex> lock(initialization_mutex_);

    if (initialized_) {
        std::cerr << "[WARNING] OpenCLComputeEngine already initialized\n";
        return;
    }

    // Инициализировать OpenCLCore
    OpenCLCore::Initialize(device_type);

    // Инициализировать CommandQueuePool
    CommandQueuePool::Initialize();

    // Создать сам engine
    instance_ = std::unique_ptr<OpenCLComputeEngine>(new OpenCLComputeEngine());
    initialized_ = true;

    std::cout << "[OK] OpenCLComputeEngine initialized\n";
}

OpenCLComputeEngine& OpenCLComputeEngine::GetInstance() {
    if (!initialized_) {
        throw std::runtime_error(
            "OpenCLComputeEngine not initialized. Call Initialize() first.");
    }
    return *instance_;
}

bool OpenCLComputeEngine::IsInitialized() {
    return initialized_;
}

void OpenCLComputeEngine::Cleanup() {
    std::lock_guard<std::mutex> lock(initialization_mutex_);

    if (initialized_) {
        KernelProgramCache::Clear();
        instance_.reset();
        CommandQueuePool::Cleanup();
        OpenCLCore::Cleanup();
        initialized_ = false;
        std::cout << "[OK] OpenCLComputeEngine cleaned up\n";
    }
}

std::shared_ptr<KernelProgram> OpenCLComputeEngine::LoadProgram(
    const std::string& source) {
    return KernelProgramCache::GetOrCompile(source);
}

cl_kernel OpenCLComputeEngine::GetKernel(
    const std::shared_ptr<KernelProgram>& program,
    const std::string& kernel_name) {
    if (!program) {
        throw std::invalid_argument("program is nullptr");
    }
    return program->GetOrCreateKernel(kernel_name);
}

std::unique_ptr<GPUMemoryBuffer> OpenCLComputeEngine::CreateBuffer(
    size_t num_elements,
    MemoryType type) {
    auto& core = OpenCLCore::GetInstance();
    // Используем CommandQueuePool для получения очереди
    cl_command_queue queue = CommandQueuePool::GetNextQueue();

    auto buffer = std::make_unique<GPUMemoryBuffer>(
        core.GetContext(),
        queue,
        num_elements,
        type
    );

    total_allocated_bytes_ += buffer->GetSizeBytes();
    num_buffers_++;

    return buffer;
}

std::unique_ptr<GPUMemoryBuffer> OpenCLComputeEngine::CreateBufferWithData(
    const std::vector<std::complex<float>>& data,
    MemoryType type) {
    auto& core = OpenCLCore::GetInstance();
    // Используем CommandQueuePool для получения очереди
    cl_command_queue queue = CommandQueuePool::GetNextQueue();

    auto buffer = std::make_unique<GPUMemoryBuffer>(
        core.GetContext(),
        queue,
        data.data(),
        data.size() * sizeof(std::complex<float>),
        data.size(),
        type
    );

    total_allocated_bytes_ += buffer->GetSizeBytes();
    num_buffers_++;

    return buffer;
}

void OpenCLComputeEngine::ExecuteKernel(
    cl_kernel kernel,
    const std::vector<cl_mem>& buffers,
    const std::array<size_t, 3>& global_work_size,
    const std::array<size_t, 3>& local_work_size) {
    if (!kernel) {
        throw std::invalid_argument("kernel is nullptr");
    }

    auto& core = OpenCLCore::GetInstance();
    cl_int err;

    // Установить аргументы kernel
    for (size_t i = 0; i < buffers.size(); ++i) {
        err = clSetKernelArg(kernel, i, sizeof(cl_mem), &buffers[i]);
        CheckCLError(err, "clSetKernelArg");
    }

    // TODO: Нужна command queue для выполнения kernel!
    // Пока это заглушка, нужно добавить CommandQueuePool

    kernel_executions_++;
}

cl_event OpenCLComputeEngine::ExecuteKernelAsync(
    cl_kernel kernel,
    const std::vector<cl_mem>& buffers,
    const std::array<size_t, 3>& global_work_size,
    const std::array<size_t, 3>& local_work_size) {
    if (!kernel) {
        throw std::invalid_argument("kernel is nullptr");
    }

    auto& core = OpenCLCore::GetInstance();
    cl_int err;

    // Установить аргументы kernel
    for (size_t i = 0; i < buffers.size(); ++i) {
        err = clSetKernelArg(kernel, i, sizeof(cl_mem), &buffers[i]);
        CheckCLError(err, "clSetKernelArg");
    }

    // TODO: Асинхронное выполнение с command queue

    kernel_executions_++;
    return nullptr;  // Заглушка
}

void OpenCLComputeEngine::Flush() {
    // TODO: Flush queue
}

void OpenCLComputeEngine::Finish() {
    // TODO: Finish queue
}

void OpenCLComputeEngine::WaitForEvent(cl_event event) {
    if (event) {
        cl_int err = clWaitForEvents(1, &event);
        CheckCLError(err, "clWaitForEvents");
    }
}

void OpenCLComputeEngine::WaitForEvents(const std::vector<cl_event>& events) {
    if (!events.empty()) {
        cl_int err = clWaitForEvents(events.size(), events.data());
        CheckCLError(err, "clWaitForEvents");
    }
}

std::string OpenCLComputeEngine::GetStatistics() const {
    std::ostringstream oss;
    oss << "\n" << std::string(70, '=') << "\n";
    oss << "OpenCLComputeEngine Statistics\n";
    oss << std::string(70, '=') << "\n\n";
    oss << std::left << std::setw(30) << "Total Allocated Memory:"
        << std::fixed << std::setprecision(2)
        << (total_allocated_bytes_ / (1024.0 * 1024.0)) << " MB\n";
    oss << std::left << std::setw(30) << "Active Buffers:" << num_buffers_ << "\n";
    oss << std::left << std::setw(30) << "Kernel Executions:" << kernel_executions_ << "\n";
    oss << "\n" << GetCacheStatistics();
    oss << std::string(70, '=') << "\n\n";
    return oss.str();
}

std::string OpenCLComputeEngine::GetDeviceInfo() const {
    return OpenCLCore::GetInstance().GetDeviceInfo();
}

std::string OpenCLComputeEngine::GetCacheStatistics() const {
    return KernelProgramCache::GetCacheStatistics();
}

// ════════════════════════════════════════════════════════════════════════════
// Новая система памяти (SVM/Hybrid)
// ════════════════════════════════════════════════════════════════════════════

std::unique_ptr<BufferFactory> OpenCLComputeEngine::CreateBufferFactory(
    const BufferConfig& config) {
    
    auto& core = OpenCLCore::GetInstance();
    cl_command_queue queue = CommandQueuePool::GetNextQueue();
    
    return std::make_unique<BufferFactory>(
        core.GetContext(),
        queue,
        core.GetDevice(),
        config
    );
}

std::unique_ptr<IMemoryBuffer> OpenCLComputeEngine::CreateHybridBuffer(
    size_t num_elements,
    MemoryType mem_type) {
    
    auto factory = CreateBufferFactory();
    auto buffer = factory->Create(num_elements, mem_type);
    
    total_allocated_bytes_ += buffer->GetSizeBytes();
    num_buffers_++;
    
    return buffer;
}

std::unique_ptr<IMemoryBuffer> OpenCLComputeEngine::CreateBufferWithStrategy(
    size_t num_elements,
    MemoryStrategy strategy,
    MemoryType mem_type) {
    
    auto factory = CreateBufferFactory();
    auto buffer = factory->CreateWithStrategy(num_elements, strategy, mem_type);
    
    total_allocated_bytes_ += buffer->GetSizeBytes();
    num_buffers_++;
    
    return buffer;
}

SVMCapabilities OpenCLComputeEngine::GetSVMCapabilities() const {
    return OpenCLCore::GetInstance().GetSVMCapabilities();
}

bool OpenCLComputeEngine::IsSVMSupported() const {
    return OpenCLCore::GetInstance().IsSVMSupported();
}

std::string OpenCLComputeEngine::GetSVMInfo() const {
    return OpenCLCore::GetInstance().GetSVMInfo();
}

OpenCLComputeEngine::~OpenCLComputeEngine() {
    // Автоматическая очистка при удалении
}

}  // namespace ManagerOpenCL
