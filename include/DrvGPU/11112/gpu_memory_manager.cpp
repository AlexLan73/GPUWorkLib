#include "ManagerOpenCL/gpu_memory_manager.hpp"
#include "ManagerOpenCL/memory_type.hpp"  // Для полного определения MemoryType
#include <iostream>
#include <algorithm>
#include <iomanip>

namespace ManagerOpenCL {
using ManagerOpenCL::GPUMemoryManager;

// ════════════════════════════════════════════════════════════════════════════
// GPUMemoryBuffer - реализация конструкторов и деструктора
// ════════════════════════════════════════════════════════════════════════════

// Конструктор 1: OWNING (создаёт новый буфер)
GPUMemoryBuffer::GPUMemoryBuffer(
    cl_context context,
    cl_command_queue queue,
    size_t num_elements,
    MemoryType type)
    : context_(context),
      queue_(queue),
      gpu_buffer_(nullptr),
      num_elements_(num_elements),
      buffer_size_bytes_(num_elements * sizeof(std::complex<float>)),
      type_(type),
      is_external_buffer_(false),
      gpu_dirty_(false) {
    AllocateGPUBuffer();
    AllocatePinnedHostBuffer();
}

// Конструктор 2: NON-OWNING (использует готовый буфер)
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
      buffer_size_bytes_(num_elements * sizeof(std::complex<float>)),
      type_(type),
      is_external_buffer_(true),
      gpu_dirty_(false) {
    // НЕ вызываем AllocateGPUBuffer() - используем готовый буфер
    AllocatePinnedHostBuffer();
}

// Конструктор 3: OWNING с данными (создаёт буфер и копирует данные)
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
      buffer_size_bytes_(data_size_bytes),
      type_(type),
      is_external_buffer_(false),  // OWNING - буфер будет освобожден в деструкторе
      gpu_dirty_(false) {
    if (!host_data) {
        throw std::invalid_argument("host_data cannot be nullptr");
    }

    cl_int error = CL_SUCCESS;

    // Преобразовать MemoryType в флаги OpenCL
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    if (type_ == MemoryType::GPU_READ_ONLY) {
        flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    } else if (type_ == MemoryType::GPU_WRITE_ONLY) {
        flags = CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR;
    }

    // Создать буфер с копированием данных
    gpu_buffer_ = clCreateBuffer(
        context_,
        flags,
        data_size_bytes,
        const_cast<void*>(host_data),  // OpenCL копирует данные, исходный указатель не изменяется
        &error
    );

    CheckCLError(error, "clCreateBuffer with CL_MEM_COPY_HOST_PTR");
    AllocatePinnedHostBuffer();
}

// Деструктор
GPUMemoryBuffer::~GPUMemoryBuffer() {
    // Освобождаем только СОБСТВЕННЫЙ буфер
    if (!is_external_buffer_ && gpu_buffer_ != nullptr) {
        clReleaseMemObject(gpu_buffer_);
        gpu_buffer_ = nullptr;
    }
    ReleasePinnedHostBuffer();
}

// Move конструктор
GPUMemoryBuffer::GPUMemoryBuffer(GPUMemoryBuffer&& other) noexcept
    : context_(other.context_),
      queue_(other.queue_),
      gpu_buffer_(other.gpu_buffer_),
      pinned_host_buffer_(std::move(other.pinned_host_buffer_)),
      num_elements_(other.num_elements_),
      buffer_size_bytes_(other.buffer_size_bytes_),
      type_(other.type_),
      is_external_buffer_(other.is_external_buffer_),
      gpu_dirty_(other.gpu_dirty_) {
    other.gpu_buffer_ = nullptr;
}

// Move оператор присваивания
GPUMemoryBuffer& GPUMemoryBuffer::operator=(GPUMemoryBuffer&& other) noexcept {
    if (this != &other) {
        // Освободить старые ресурсы
        if (!is_external_buffer_ && gpu_buffer_ != nullptr) {
            clReleaseMemObject(gpu_buffer_);
        }
        ReleasePinnedHostBuffer();

        // Переместить новые ресурсы
        context_ = other.context_;
        queue_ = other.queue_;
        gpu_buffer_ = other.gpu_buffer_;
        pinned_host_buffer_ = std::move(other.pinned_host_buffer_);
        num_elements_ = other.num_elements_;
        buffer_size_bytes_ = other.buffer_size_bytes_;
        type_ = other.type_;
        is_external_buffer_ = other.is_external_buffer_;
        gpu_dirty_ = other.gpu_dirty_;

        other.gpu_buffer_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Приватные методы GPUMemoryBuffer
// ════════════════════════════════════════════════════════════════════════════

void GPUMemoryBuffer::AllocateGPUBuffer() {
    cl_int error = CL_SUCCESS;

    // Преобразовать MemoryType в флаги OpenCL
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    if (type_ == MemoryType::GPU_READ_ONLY) {
        flags = CL_MEM_READ_ONLY;
    } else if (type_ == MemoryType::GPU_WRITE_ONLY) {
        flags = CL_MEM_WRITE_ONLY;
    }

    // Выделить память на GPU
    gpu_buffer_ = clCreateBuffer(
        context_,
        flags,
        num_elements_ * sizeof(std::complex<float>),
        nullptr,
        &error
    );

    CheckCLError(error, "clCreateBuffer");
}

void GPUMemoryBuffer::AllocatePinnedHostBuffer() {
    pinned_host_buffer_.resize(num_elements_);
}

void GPUMemoryBuffer::ReleasePinnedHostBuffer() {
    pinned_host_buffer_.clear();
    pinned_host_buffer_.shrink_to_fit();
}

void GPUMemoryBuffer::CheckCLError(cl_int error, const std::string& operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(
            "OpenCL Error in " + operation + ": " + std::to_string(error)
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Операции чтения/записи GPUMemoryBuffer
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>> GPUMemoryBuffer::ReadFromGPU() {
    return ReadPartial(num_elements_);
}


std::vector<std::complex<float>> GPUMemoryBuffer::ReadPartial(size_t num_elements) {
    if (num_elements > num_elements_) {
        throw std::runtime_error("Reading more elements than buffer size");
    }

    std::vector<std::complex<float>> result(num_elements);

    // clEnqueueReadBuffer сигнатура:
    // cl_int clEnqueueReadBuffer(
    //     cl_command_queue command_queue,
    //     cl_mem buffer,
    //     cl_bool blocking_read,
    //     size_t offset,
    //     size_t cb,
    //     void *ptr,
    //     cl_uint num_events_in_wait_list,    <- ВСЕГДА 0 для простых случаев
    //     const cl_event *event_wait_list,    <- ВСЕГДА nullptr
    //     cl_event *event                     <- ВСЕГДА nullptr
    // )

    cl_int error = clEnqueueReadBuffer(
        queue_,                    // command_queue
        gpu_buffer_,               // buffer
        CL_TRUE,                   // blocking_read (ждём завершения)
        0,                         // offset (начинаем с 0)
        num_elements * sizeof(std::complex<float>),  // cb - размер в байтах
        result.data(),             // ptr - куда читать
        0,                         // num_events_in_wait_list
        nullptr,                   // event_wait_list
        nullptr                    // event
    );

    CheckCLError(error, "clEnqueueReadBuffer");
    gpu_dirty_ = false;

    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Записать данные на GPU
// ════════════════════════════════════════════════════════════════════════════

void GPUMemoryBuffer::WriteToGPU(const std::vector<std::complex<float>>& data) {
    if (data.size() > num_elements_) {
        throw std::runtime_error("Writing more data than buffer size");
    }

    // clEnqueueWriteBuffer сигнатура:
    // cl_int clEnqueueWriteBuffer(
    //     cl_command_queue command_queue,
    //     cl_mem buffer,
    //     cl_bool blocking_write,
    //     size_t offset,
    //     size_t cb,
    //     const void *ptr,
    //     cl_uint num_events_in_wait_list,
    //     const cl_event *event_wait_list,
    //     cl_event *event
    // )

    cl_int error = clEnqueueWriteBuffer(
        queue_,                    // command_queue
        gpu_buffer_,               // buffer
        CL_TRUE,                   // blocking_write (ждём завершения)
        0,                         // offset
        data.size() * sizeof(std::complex<float>),  // cb - размер в байтах
        data.data(),               // ptr - откуда писать
        0,                         // num_events_in_wait_list
        nullptr,                   // event_wait_list
        nullptr                    // event
    );

    CheckCLError(error, "clEnqueueWriteBuffer");
    gpu_dirty_ = true;
}

/*
std::vector<std::complex<float>> GPUMemoryBuffer::ReadPartial(size_t num_elements) {
    if (num_elements > num_elements_) {
        throw std::runtime_error("Reading more elements than buffer size");
    }

    std::vector<std::complex<float>> result(num_elements);

    cl_int error = clEnqueueReadBuffer(
        queue_,
        gpu_buffer_,
        CL_TRUE,  // blocking read
        0,
        num_elements * sizeof(std::complex<float>),
        result.data(),
        0,
        nullptr,
        nullptr
    );

    CheckCLError(error, "clEnqueueReadBuffer");
    gpu_dirty_ = false;

    return result;
}

void GPUMemoryBuffer::WriteToGPU(const std::vector<std::complex<float>>& data) {
    if (data.size() > num_elements_) {
        throw std::runtime_error("Writing more data than buffer size");
    }

    cl_int error = clEnqueueWriteBuffer(
        queue_,
        gpu_buffer_,
        CL_TRUE,  // blocking write
        0,
        data.size() * sizeof(std::complex<float>),
        data.data(),
        0,
        nullptr,
        nullptr
    );

    CheckCLError(error, "clEnqueueWriteBuffer");
    gpu_dirty_ = true;
}
*/
void GPUMemoryBuffer::PrintStats() const {
    std::cout << "GPUMemoryBuffer Stats:\n";
    std::cout << "  Num Elements:   " << num_elements_ << "\n";
    std::cout << "  Memory (MB):    " << std::fixed << std::setprecision(2)
              << (GetSizeBytes() / (1024.0 * 1024.0)) << "\n";
    std::cout << "  External:       " << (is_external_buffer_ ? "YES" : "NO") << "\n";
    std::cout << "  GPU Dirty:      " << (gpu_dirty_ ? "YES" : "NO") << "\n";
    std::cout << "  Type:           ";
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

// ════════════════════════════════════════════════════════════════════════════
// GPUMemoryManager - синглтон
// ════════════════════════════════════════════════════════════════════════════

// Статические члены
std::unique_ptr<GPUMemoryManager> GPUMemoryManager::instance_ = nullptr;
bool GPUMemoryManager::initialized_ = false;

// Конструктор
GPUMemoryManager::GPUMemoryManager()
    : context_(nullptr),
      queue_(nullptr),
      total_allocated_bytes_(0),
      num_buffers_(0) {
}

// Инициализация (один раз)
void GPUMemoryManager::Initialize() {
    if (initialized_) {
        std::cout << "[WARNING] GPUMemoryManager already initialized\n";
        return;
    }

//    instance_ = std::make_unique<GPUMemoryManager>();
    instance_.reset(new GPUMemoryManager());

    // Получить context и queue из OpenCLManager
    auto& opencl = OpenCLManager::GetInstance();
    instance_->context_ = opencl.GetContext();
    instance_->queue_ = opencl.GetQueue();

    initialized_ = true;
    std::cout << "[OK] GPUMemoryManager initialized\n";
}

// Создать новый буфер (OWNING)
std::unique_ptr<GPUMemoryBuffer> GPUMemoryManager::CreateBuffer(
    size_t num_elements,
    MemoryType type) {
    if (!initialized_) {
        throw std::runtime_error("GPUMemoryManager not initialized. Call Initialize() first.");
    }

    auto buffer = std::make_unique<GPUMemoryBuffer>(
        instance_->context_,
        instance_->queue_,
        num_elements,
        type
    );

    instance_->total_allocated_bytes_ += buffer->GetSizeBytes();
    instance_->num_buffers_++;

    return buffer;
}

// Обернуть готовый буфер (NON-OWNING)
std::unique_ptr<GPUMemoryBuffer> GPUMemoryManager::WrapExternalBuffer(
    cl_mem external_gpu_buffer,
    size_t num_elements,
    MemoryType type) {
    if (!initialized_) {
        throw std::runtime_error("GPUMemoryManager not initialized. Call Initialize() first.");
    }

    auto buffer = std::make_unique<GPUMemoryBuffer>(
        instance_->context_,
        instance_->queue_,
        external_gpu_buffer,
        num_elements,
        type
    );

    // Не считаем в total_allocated_bytes - это чужой буфер

    return buffer;
}

// Статистика
void GPUMemoryManager::PrintStatistics() {
    if (!initialized_) {
        std::cout << "[WARNING] GPUMemoryManager not initialized\n";
        return;
    }

    std::cout << "\nGPUMemoryManager Statistics:\n";
    std::cout << "  Total Allocated: " << std::fixed << std::setprecision(2)
              << (instance_->total_allocated_bytes_ / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Active Buffers:  " << instance_->num_buffers_ << "\n";
}

// Доступ к синглтону
GPUMemoryManager& GPUMemoryManager::GetInstance() {
    if (!initialized_) {
        throw std::runtime_error("GPUMemoryManager not initialized. Call Initialize() first.");
    }
    return *instance_;
}

} // namespace ManagerOpenCL
