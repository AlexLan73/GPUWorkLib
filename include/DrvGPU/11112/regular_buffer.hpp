#pragma once

/**
 * @file regular_buffer.hpp
 * @brief RAII обёртка для традиционных OpenCL буферов (cl_mem)
 * 
 * Реализует IMemoryBuffer интерфейс для традиционных буферов.
 * Используется как fallback когда SVM недоступен.
 * 
 * @author Codo (AI Assistant)
 * @date 2026-01-19
 */

#include "i_memory_buffer.hpp"
#include "svm_capabilities.hpp"
#include "memory_type.hpp"
#include <CL/cl.h>
#include <complex>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// Class: RegularBuffer - RAII обёртка для cl_mem
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class RegularBuffer
 * @brief Традиционный OpenCL буфер с RAII
 * 
 * Особенности:
 * - Использует clCreateBuffer/clEnqueueReadBuffer/clEnqueueWriteBuffer
 * - Поддерживает owning и non-owning режимы
 * - Полная совместимость с OpenCL 1.x
 * 
 * @code
 * RegularBuffer buffer(context, queue, 1024);
 * buffer.Write(data);
 * // ... kernel execution ...
 * auto result = buffer.Read();
 * @endcode
 */
class RegularBuffer : public IMemoryBuffer {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструкторы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать буфер (OWNING)
     */
    RegularBuffer(
        cl_context context,
        cl_command_queue queue,
        size_t num_elements,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    );
    
    /**
     * @brief Создать буфер с начальными данными (OWNING)
     */
    RegularBuffer(
        cl_context context,
        cl_command_queue queue,
        const ComplexVector& initial_data,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    );
    
    /**
     * @brief Обернуть внешний буфер (NON-OWNING)
     */
    RegularBuffer(
        cl_context context,
        cl_command_queue queue,
        cl_mem external_buffer,
        size_t num_elements,
        MemoryType mem_type = MemoryType::GPU_READ_WRITE
    );
    
    // ═══════════════════════════════════════════════════════════════
    // Деструктор (RAII)
    // ═══════════════════════════════════════════════════════════════
    
    ~RegularBuffer() override;
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение move
    // ═══════════════════════════════════════════════════════════════
    
    RegularBuffer(const RegularBuffer&) = delete;
    RegularBuffer& operator=(const RegularBuffer&) = delete;
    
    RegularBuffer(RegularBuffer&& other) noexcept;
    RegularBuffer& operator=(RegularBuffer&& other) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Реализация IMemoryBuffer интерфейса
    // ═══════════════════════════════════════════════════════════════
    
    // --- Чтение/Запись ---
    void Write(const ComplexVector& data) override;
    void WriteRaw(const void* data, size_t size_bytes) override;
    ComplexVector Read() override;
    ComplexVector ReadPartial(size_t num_elements) override;
    void ReadRaw(void* dest, size_t size_bytes) override;
    
    // --- Асинхронные операции ---
    cl_event WriteAsync(const ComplexVector& data) override;
    cl_event ReadAsync(ComplexVector& out_data) override;
    
    // --- OpenCL ресурсы ---
    cl_mem GetCLMem() const override { return buffer_; }
    void* GetSVMPointer() const override { return nullptr; }  // Не SVM
    void SetAsKernelArg(cl_kernel kernel, cl_uint arg_index) override;
    
    // --- Информация ---
    size_t GetNumElements() const override { return num_elements_; }
    size_t GetSizeBytes() const override { return size_bytes_; }
    MemoryType GetMemoryType() const override { return mem_type_; }
    MemoryStrategy GetStrategy() const override { return MemoryStrategy::REGULAR_BUFFER; }
    bool IsExternal() const override { return is_external_; }
    bool IsSVM() const override { return false; }
    BufferInfo GetInfo() const override;
    void PrintStats() const override;
    
    // --- SVM операции (no-op для regular buffer) ---
    void Map(bool write = true, bool read = true) override { /* no-op */ }
    void Unmap() override { /* no-op */ }
    bool IsMapped() const override { return false; }

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    cl_context       context_      = nullptr;
    cl_command_queue queue_        = nullptr;
    cl_mem           buffer_       = nullptr;
    size_t           num_elements_ = 0;
    size_t           size_bytes_   = 0;
    MemoryType       mem_type_     = MemoryType::GPU_READ_WRITE;
    bool             is_external_  = false;
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    void AllocateBuffer();
    void FreeBuffer();
    
    cl_mem_flags GetMemFlags() const;
    
    static void CheckCLError(cl_int err, const std::string& operation);
};

// ════════════════════════════════════════════════════════════════════════════
// Реализация (inline)
// ════════════════════════════════════════════════════════════════════════════

inline RegularBuffer::RegularBuffer(
    cl_context context,
    cl_command_queue queue,
    size_t num_elements,
    MemoryType mem_type)
    : context_(context),
      queue_(queue),
      num_elements_(num_elements),
      size_bytes_(num_elements * sizeof(ComplexFloat)),
      mem_type_(mem_type),
      is_external_(false) {
    
    if (!context_ || !queue_) {
        throw std::invalid_argument("RegularBuffer: context and queue must not be null");
    }
    
    if (num_elements_ == 0) {
        throw std::invalid_argument("RegularBuffer: num_elements must be > 0");
    }
    
    AllocateBuffer();
}

inline RegularBuffer::RegularBuffer(
    cl_context context,
    cl_command_queue queue,
    const ComplexVector& initial_data,
    MemoryType mem_type)
    : context_(context),
      queue_(queue),
      num_elements_(initial_data.size()),
      size_bytes_(initial_data.size() * sizeof(ComplexFloat)),
      mem_type_(mem_type),
      is_external_(false) {
    
    if (!context_ || !queue_) {
        throw std::invalid_argument("RegularBuffer: context and queue must not be null");
    }
    
    if (initial_data.empty()) {
        throw std::invalid_argument("RegularBuffer: initial_data must not be empty");
    }
    
    // Создать буфер с данными (CL_MEM_COPY_HOST_PTR)
    cl_int err;
    cl_mem_flags flags = GetMemFlags() | CL_MEM_COPY_HOST_PTR;
    
    buffer_ = clCreateBuffer(
        context_,
        flags,
        size_bytes_,
        const_cast<ComplexFloat*>(initial_data.data()),
        &err
    );
    
    CheckCLError(err, "clCreateBuffer with CL_MEM_COPY_HOST_PTR");
}

inline RegularBuffer::RegularBuffer(
    cl_context context,
    cl_command_queue queue,
    cl_mem external_buffer,
    size_t num_elements,
    MemoryType mem_type)
    : context_(context),
      queue_(queue),
      buffer_(external_buffer),
      num_elements_(num_elements),
      size_bytes_(num_elements * sizeof(ComplexFloat)),
      mem_type_(mem_type),
      is_external_(true) {
    
    if (!context_ || !queue_ || !buffer_) {
        throw std::invalid_argument("RegularBuffer: all parameters must not be null");
    }
}

inline RegularBuffer::~RegularBuffer() {
    FreeBuffer();
}

inline RegularBuffer::RegularBuffer(RegularBuffer&& other) noexcept
    : context_(other.context_),
      queue_(other.queue_),
      buffer_(other.buffer_),
      num_elements_(other.num_elements_),
      size_bytes_(other.size_bytes_),
      mem_type_(other.mem_type_),
      is_external_(other.is_external_) {
    
    other.buffer_ = nullptr;
}

inline RegularBuffer& RegularBuffer::operator=(RegularBuffer&& other) noexcept {
    if (this != &other) {
        FreeBuffer();
        
        context_      = other.context_;
        queue_        = other.queue_;
        buffer_       = other.buffer_;
        num_elements_ = other.num_elements_;
        size_bytes_   = other.size_bytes_;
        mem_type_     = other.mem_type_;
        is_external_  = other.is_external_;
        
        other.buffer_ = nullptr;
    }
    return *this;
}

inline void RegularBuffer::AllocateBuffer() {
    cl_int err;
    cl_mem_flags flags = GetMemFlags();
    
    buffer_ = clCreateBuffer(
        context_,
        flags,
        size_bytes_,
        nullptr,
        &err
    );
    
    CheckCLError(err, "clCreateBuffer");
}

inline void RegularBuffer::FreeBuffer() {
    if (buffer_ && !is_external_) {
        clReleaseMemObject(buffer_);
    }
    buffer_ = nullptr;
}

inline cl_mem_flags RegularBuffer::GetMemFlags() const {
    switch (mem_type_) {
        case MemoryType::GPU_READ_ONLY:
            return CL_MEM_READ_ONLY;
        case MemoryType::GPU_WRITE_ONLY:
            return CL_MEM_WRITE_ONLY;
        case MemoryType::GPU_READ_WRITE:
        default:
            return CL_MEM_READ_WRITE;
    }
}

inline void RegularBuffer::Write(const ComplexVector& data) {
    if (data.size() > num_elements_) {
        throw std::runtime_error(
            "RegularBuffer::Write: data size exceeds buffer capacity"
        );
    }
    
    WriteRaw(data.data(), data.size() * sizeof(ComplexFloat));
}

inline void RegularBuffer::WriteRaw(const void* data, size_t size_bytes) {
    if (size_bytes > size_bytes_) {
        throw std::runtime_error(
            "RegularBuffer::WriteRaw: size exceeds buffer capacity"
        );
    }
    
    cl_int err = clEnqueueWriteBuffer(
        queue_,
        buffer_,
        CL_TRUE,  // Blocking
        0,
        size_bytes,
        data,
        0, nullptr, nullptr
    );
    
    CheckCLError(err, "clEnqueueWriteBuffer");
}

inline ComplexVector RegularBuffer::Read() {
    return ReadPartial(num_elements_);
}

inline ComplexVector RegularBuffer::ReadPartial(size_t num_elements) {
    if (num_elements > num_elements_) {
        throw std::runtime_error(
            "RegularBuffer::ReadPartial: requested elements exceed buffer size"
        );
    }
    
    ComplexVector result(num_elements);
    ReadRaw(result.data(), num_elements * sizeof(ComplexFloat));
    return result;
}

inline void RegularBuffer::ReadRaw(void* dest, size_t size_bytes) {
    if (size_bytes > size_bytes_) {
        throw std::runtime_error(
            "RegularBuffer::ReadRaw: size exceeds buffer capacity"
        );
    }
    
    cl_int err = clEnqueueReadBuffer(
        queue_,
        buffer_,
        CL_TRUE,  // Blocking
        0,
        size_bytes,
        dest,
        0, nullptr, nullptr
    );
    
    CheckCLError(err, "clEnqueueReadBuffer");
}

inline cl_event RegularBuffer::WriteAsync(const ComplexVector& data) {
    if (data.size() > num_elements_) {
        throw std::runtime_error(
            "RegularBuffer::WriteAsync: data size exceeds buffer capacity"
        );
    }
    
    cl_event event = nullptr;
    
    cl_int err = clEnqueueWriteBuffer(
        queue_,
        buffer_,
        CL_FALSE,  // Non-blocking
        0,
        data.size() * sizeof(ComplexFloat),
        data.data(),
        0, nullptr,
        &event
    );
    
    CheckCLError(err, "clEnqueueWriteBuffer (async)");
    return event;
}

inline cl_event RegularBuffer::ReadAsync(ComplexVector& out_data) {
    if (out_data.size() < num_elements_) {
        out_data.resize(num_elements_);
    }
    
    cl_event event = nullptr;
    
    cl_int err = clEnqueueReadBuffer(
        queue_,
        buffer_,
        CL_FALSE,  // Non-blocking
        0,
        num_elements_ * sizeof(ComplexFloat),
        out_data.data(),
        0, nullptr,
        &event
    );
    
    CheckCLError(err, "clEnqueueReadBuffer (async)");
    return event;
}

inline void RegularBuffer::SetAsKernelArg(cl_kernel kernel, cl_uint arg_index) {
    cl_int err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &buffer_);
    CheckCLError(err, "clSetKernelArg");
}

inline BufferInfo RegularBuffer::GetInfo() const {
    BufferInfo info;
    info.num_elements = num_elements_;
    info.size_bytes   = size_bytes_;
    info.memory_type  = mem_type_;
    info.strategy     = MemoryStrategy::REGULAR_BUFFER;
    info.is_external  = is_external_;
    info.is_mapped    = false;
    return info;
}

inline void RegularBuffer::PrintStats() const {
    std::cout << "\n" << std::string(50, '─') << "\n";
    std::cout << "RegularBuffer Statistics\n";
    std::cout << std::string(50, '─') << "\n";
    std::cout << std::left << std::setw(20) << "Elements:" << num_elements_ << "\n";
    std::cout << std::left << std::setw(20) << "Size:" 
              << std::fixed << std::setprecision(2) 
              << (size_bytes_ / (1024.0 * 1024.0)) << " MB\n";
    std::cout << std::left << std::setw(20) << "External:" 
              << (is_external_ ? "YES" : "NO") << "\n";
    std::cout << std::left << std::setw(20) << "cl_mem:" 
              << buffer_ << "\n";
    std::cout << std::string(50, '─') << "\n";
}

inline void RegularBuffer::CheckCLError(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "OpenCL Error in " + operation + ": " + std::to_string(err)
        );
    }
}

} // namespace ManagerOpenCL

