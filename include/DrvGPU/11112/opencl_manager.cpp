
#include "ManagerOpenCL/opencl_manager.h"
#include "ManagerOpenCL/memory_type.hpp"
#include "ManagerOpenCL/gpu_memory_buffer.hpp"
#include <iostream>
#include <sstream>
#include <functional>
#include <vector>
#include <cstring>
#include <iomanip>

namespace ManagerOpenCL {

// ═══════════════════════════════════════════════════════════════════
// SINGLETON INSTANCE
// ═══════════════════════════════════════════════════════════════════

OpenCLManager& OpenCLManager::GetInstance() {
    static OpenCLManager instance;  // Thread-safe (C++11 static local init)
    return instance;
}

void OpenCLManager::Initialize(cl_device_type device_type) {
    auto& instance = GetInstance();
    if (!instance.initialized_) {
        instance.InitializeOpenCL(device_type);
        instance.initialized_ = true;
    }
}

void OpenCLManager::Cleanup() {
    auto& instance = GetInstance();
    if (instance.initialized_) {
        instance.ReleaseResources();
        instance.initialized_ = false;
    }
}

// ═══════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════

void OpenCLManager::InitializeOpenCL(cl_device_type device_type) {
    cl_int err;
    cl_uint num_platforms;

    // 1. Get platforms
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get platforms");
    }

    // 2. Get devices
    platform_ = platforms[0];
    cl_uint num_devices;
    err = clGetDeviceIDs(platform_, device_type, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        throw std::runtime_error("No OpenCL devices found for specified type");
    }

    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform_, device_type, num_devices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get devices");
    }

    device_ = devices[0];

    // 3. Create context
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context");
    }

    // 4. Create command queue
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context_);
        throw std::runtime_error("Failed to create command queue");
    }
}

// ═══════════════════════════════════════════════════════════════════
// PROGRAM COMPILATION & CACHING
// ═══════════════════════════════════════════════════════════════════

cl_program OpenCLManager::GetOrCompileProgram(const std::string& source) {
    if (!initialized_) {
        throw std::runtime_error("OpenCLManager not initialized");
    }

    // Generate hash of source
    std::hash<std::string> hasher;
    std::string hash_key = std::to_string(hasher(source));

    {
        std::unique_lock<std::mutex> lock(cache_mutex_);

        // Check cache
        auto it = program_cache_.find(hash_key);
        if (it != program_cache_.end()) {
            cache_hits_++;
            return it->second;
        }
    }

    // Compile (outside lock - compilation is expensive)
    cl_program program = CompileProgram(source);

    // Store in cache
    {
        std::unique_lock<std::mutex> lock(cache_mutex_);
        cache_misses_++;
        program_cache_[hash_key] = program;
    }

    return program;
}

cl_program OpenCLManager::CompileProgram(const std::string& source) {
    cl_int err;

    const char* source_str = source.c_str();
    size_t source_len = source.length();

    cl_program program = clCreateProgramWithSource(
        context_, 1, &source_str, &source_len, &err);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create program");
    }

    // Compile
    err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);

        std::string error_msg = "Compilation failed:\n";
        error_msg += std::string(log.begin(), log.end());

        clReleaseProgram(program);
        throw std::runtime_error(error_msg);
    }

    return program;
}

std::string OpenCLManager::GetCacheStatistics() const {
    std::unique_lock<std::mutex> lock(cache_mutex_);

    std::ostringstream oss;
    oss << "Program Cache Statistics:\n";
    oss << "  Cache size: " << program_cache_.size() << " programs\n";
    oss << "  Cache hits: " << cache_hits_ << "\n";
    oss << "  Cache misses: " << cache_misses_ << "\n";

    if (cache_hits_ + cache_misses_ > 0) {
        double hit_rate = 100.0 * cache_hits_ / (cache_hits_ + cache_misses_);
        oss << "  Hit rate: " << hit_rate << "%\n";
    }

    return oss.str();
}

// ═══════════════════════════════════════════════════════════════════
// DEVICE INFORMATION
// ═══════════════════════════════════════════════════════════════════

std::string OpenCLManager::GetDeviceInfo() const {
    if (!initialized_) {
        return "OpenCLManager not initialized\n";
    }

    std::ostringstream oss;
    oss << "OpenCL Device Information:\n";
    oss << std::string(50, '=') << "\n";

    // Device name
    char device_name[1024] = {0};
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    oss << "Device Name: " << device_name << "\n";

    // Vendor
    char vendor[1024] = {0};
    clGetDeviceInfo(device_, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
    oss << "Vendor: " << vendor << "\n";

    // Driver version
    char driver_version[1024] = {0};
    clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(driver_version), driver_version, nullptr);
    oss << "Driver Version: " << driver_version << "\n";

    // Device type
    cl_device_type device_type;
    clGetDeviceInfo(device_, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
    oss << "Device Type: ";
    if (device_type & CL_DEVICE_TYPE_GPU) oss << "GPU";
    else if (device_type & CL_DEVICE_TYPE_CPU) oss << "CPU";
    else oss << "Other";
    oss << "\n";

    // Global memory
    cl_ulong global_mem;
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);
    oss << "Global Memory: " << (global_mem / (1024 * 1024)) << " MB\n";

    // Local memory
    cl_ulong local_mem;
    clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, nullptr);
    oss << "Local Memory: " << (local_mem / 1024) << " KB\n";

    // Compute units
    cl_uint compute_units;
    clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
    oss << "Compute Units: " << compute_units << "\n";

    // Work group size
    size_t work_group_size;
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(work_group_size), &work_group_size, nullptr);
    oss << "Max Work Group Size: " << work_group_size << "\n";

    oss << std::string(50, '=') << "\n";

    return oss.str();
}

// ═══════════════════════════════════════════════════════════════════
// CLEANUP
// ═══════════════════════════════════════════════════════════════════

void OpenCLManager::ReleaseResources() {
    // Release cached kernels
    {
        std::unique_lock<std::mutex> lock(kernel_cache_mutex_);
        for (auto& [key, kernel] : kernel_cache_) {
            if (kernel) clReleaseKernel(kernel);
        }
        kernel_cache_.clear();
    }

    // Release cached programs
    {
        std::unique_lock<std::mutex> lock(cache_mutex_);
        for (auto& [key, program] : program_cache_) {
            if (program) clReleaseProgram(program);
        }
        program_cache_.clear();
    }

    // Release OpenCL resources
    if (queue_) {
        clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    if (device_) {
        clReleaseDevice(device_);
        device_ = nullptr;
    }
}

OpenCLManager::~OpenCLManager() {
    if (initialized_) {
        ReleaseResources();
    }
}

// ═══════════════════════════════════════════════════════════════════
// GPU MEMORY MANAGEMENT
// ═══════════════════════════════════════════════════════════════════

std::unique_ptr<GPUMemoryBuffer> OpenCLManager::CreateBuffer(
    size_t num_elements,
    MemoryType type) {
    if (!initialized_) {
        throw std::runtime_error("OpenCLManager not initialized");
    }

    auto buffer = std::make_unique<GPUMemoryBuffer>(
        context_,
        queue_,
        num_elements,
        type
    );

    {
        std::unique_lock<std::mutex> lock(registry_mutex_);
        total_allocated_bytes_ += buffer->GetSizeBytes();
        num_buffers_++;
    }

    return buffer;
}

std::unique_ptr<GPUMemoryBuffer> OpenCLManager::CreateBufferWithData(
    size_t num_elements,
    const void* host_data,
    size_t data_size_bytes,
    MemoryType type) {
    if (!initialized_) {
        throw std::runtime_error("OpenCLManager not initialized");
    }

    if (!host_data) {
        throw std::invalid_argument("host_data cannot be nullptr");
    }

    // Если размер не указан, вычисляем из num_elements
    if (data_size_bytes == 0) {
        data_size_bytes = num_elements * sizeof(std::complex<float>);
    }

    // Создать буфер через GPUMemoryBuffer конструктор с данными (owning)
    auto buffer = std::make_unique<GPUMemoryBuffer>(
        context_,
        queue_,
        host_data,
        data_size_bytes,
        num_elements,
        type
    );

    {
        std::unique_lock<std::mutex> lock(registry_mutex_);
        total_allocated_bytes_ += data_size_bytes;
        num_buffers_++;
    }

    return buffer;
}

std::unique_ptr<GPUMemoryBuffer> OpenCLManager::WrapExternalBuffer(
    cl_mem external_gpu_buffer,
    size_t num_elements,
    MemoryType type) {
    if (!initialized_) {
        throw std::runtime_error("OpenCLManager not initialized");
    }

    // Validate that external buffer belongs to correct context
    ValidateBufferContext(external_gpu_buffer);

    auto buffer = std::make_unique<GPUMemoryBuffer>(
        context_,
        queue_,
        external_gpu_buffer,
        num_elements,
        type
    );

    // Don't count external buffers in statistics

    return buffer;
}

void OpenCLManager::ValidateBufferContext(cl_mem external_buffer) const {
    if (!external_buffer) {
        throw std::runtime_error("Invalid external buffer: nullptr");
    }

    cl_context buffer_context = nullptr;
    cl_int err = clGetMemObjectInfo(
        external_buffer,
        CL_MEM_CONTEXT,
        sizeof(cl_context),
        &buffer_context,
        nullptr
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "Failed to get buffer context: " + std::to_string(err)
        );
    }

    if (buffer_context != context_) {
        throw std::runtime_error(
            "External buffer belongs to different context. "
            "All buffers must be created through OpenCLManager::CreateBuffer() "
            "or belong to the same context as OpenCLManager. "
            "Error code: CL_INVALID_CONTEXT (-34)"
        );
    }
}

void OpenCLManager::RegisterBuffer(
    const std::string& name,
    std::shared_ptr<GPUMemoryBuffer> buffer) {
    if (!initialized_) {
        throw std::runtime_error("OpenCLManager not initialized");
    }

    std::unique_lock<std::mutex> lock(registry_mutex_);
    buffer_registry_[name] = buffer;  // weak_ptr автоматически создается
}

std::shared_ptr<GPUMemoryBuffer> OpenCLManager::GetBuffer(const std::string& name) {
    if (!initialized_) {
        throw std::runtime_error("OpenCLManager not initialized");
    }

    std::unique_lock<std::mutex> lock(registry_mutex_);
    auto it = buffer_registry_.find(name);
    
    if (it == buffer_registry_.end()) {
        return nullptr;
    }

    // Попытаться получить shared_ptr из weak_ptr
    return it->second.lock();
}

std::shared_ptr<GPUMemoryBuffer> OpenCLManager::GetOrCreateBuffer(
    const std::string& name,
    size_t num_elements,
    MemoryType type) {
    // Попытаться получить существующий
    auto existing = GetBuffer(name);
    if (existing) {
        return existing;
    }

    // Создать новый
    auto new_buffer = CreateBuffer(num_elements, type);
    auto shared_buffer = std::shared_ptr<GPUMemoryBuffer>(new_buffer.release());
    
    // Зарегистрировать
    RegisterBuffer(name, shared_buffer);
    
    return shared_buffer;
}

void OpenCLManager::PrintMemoryStatistics() const {
    if (!initialized_) {
        std::cout << "[WARNING] OpenCLManager not initialized\n";
        return;
    }

    std::unique_lock<std::mutex> lock(registry_mutex_);

    std::cout << "\nGPU Memory Statistics:\n";
    std::cout << "  Total Allocated: " << std::fixed << std::setprecision(2)
              << (total_allocated_bytes_ / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Active Buffers:  " << num_buffers_ << "\n";
    std::cout << "  Registered Buffers: " << buffer_registry_.size() << "\n";
    
    // Подсчитать активные (не expired) буферы
    size_t active_registered = 0;
    for (const auto& [name, weak_buf] : buffer_registry_) {
        if (!weak_buf.expired()) {
            active_registered++;
        }
    }
    std::cout << "  Active Registered: " << active_registered << "\n";
}

void OpenCLManager::CleanupExpiredBuffers() {
    if (!initialized_) {
        return;
    }

    std::unique_lock<std::mutex> lock(registry_mutex_);
    
    size_t removed = 0;
    auto it = buffer_registry_.begin();
    while (it != buffer_registry_.end()) {
        if (it->second.expired()) {
            it = buffer_registry_.erase(it);
            removed++;
        } else {
            ++it;
        }
    }
    
    if (removed > 0) {
        std::cout << "[INFO] Cleaned up " << removed << " expired buffer(s) from registry\n";
    }
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL CACHING
// ═══════════════════════════════════════════════════════════════════

cl_kernel OpenCLManager::CreateKernel(cl_program program, const std::string& kernel_name) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error(
            "Failed to create kernel '" + kernel_name + "': " + std::to_string(err)
        );
    }
    
    return kernel;
}

cl_kernel OpenCLManager::GetOrCreateKernel(cl_program program, const std::string& kernel_name) {
    if (!initialized_) {
        throw std::runtime_error("OpenCLManager not initialized");
    }

    // Generate cache key: program pointer + kernel name
    // Используем адрес program как часть ключа (уникален для каждого program)
    std::string cache_key = std::to_string(reinterpret_cast<uintptr_t>(program)) + ":" + kernel_name;

    {
        std::unique_lock<std::mutex> lock(kernel_cache_mutex_);

        // Check cache
        auto it = kernel_cache_.find(cache_key);
        if (it != kernel_cache_.end()) {
            kernel_cache_hits_++;
            return it->second;
        }
    }

    // Create kernel (outside lock - creation is fast but still)
    cl_kernel kernel = CreateKernel(program, kernel_name);

    // Store in cache
    {
        std::unique_lock<std::mutex> lock(kernel_cache_mutex_);
        kernel_cache_misses_++;
        kernel_cache_[cache_key] = kernel;
    }

    return kernel;
}

std::string OpenCLManager::GetKernelCacheStatistics() const {
    std::unique_lock<std::mutex> lock(kernel_cache_mutex_);

    std::ostringstream oss;
    oss << "Kernel Cache Statistics:\n";
    oss << "  Cache size: " << kernel_cache_.size() << " kernels\n";
    oss << "  Cache hits: " << kernel_cache_hits_ << "\n";
    oss << "  Cache misses: " << kernel_cache_misses_ << "\n";

    if (kernel_cache_hits_ + kernel_cache_misses_ > 0) {
        double hit_rate = 100.0 * kernel_cache_hits_ / (kernel_cache_hits_ + kernel_cache_misses_);
        oss << "  Hit rate: " << hit_rate << "%\n";
    }

    return oss.str();
}

void OpenCLManager::ClearKernelCache() {
    if (!initialized_) {
        return;
    }

    std::unique_lock<std::mutex> lock(kernel_cache_mutex_);
    
    size_t count = kernel_cache_.size();
    for (auto& [key, kernel] : kernel_cache_) {
        if (kernel) {
            clReleaseKernel(kernel);
        }
    }
    kernel_cache_.clear();
    
    if (count > 0) {
        std::cout << "[INFO] Cleared " << count << " kernel(s) from cache\n";
    }
}

void OpenCLManager::ClearKernelsForProgram(cl_program program) {
    if (!initialized_ || !program) {
        return;
    }

    std::string program_prefix = std::to_string(reinterpret_cast<uintptr_t>(program)) + ":";
    
    std::unique_lock<std::mutex> lock(kernel_cache_mutex_);
    
    size_t removed = 0;
    auto it = kernel_cache_.begin();
    while (it != kernel_cache_.end()) {
        // Check if this kernel belongs to the specified program
        if (it->first.find(program_prefix) == 0) {
            if (it->second) {
                clReleaseKernel(it->second);
            }
            it = kernel_cache_.erase(it);
            removed++;
        } else {
            ++it;
        }
    }
    
    if (removed > 0) {
        std::cout << "[INFO] Cleared " << removed << " kernel(s) for program " 
                  << reinterpret_cast<uintptr_t>(program) << "\n";
    }
}

size_t OpenCLManager::GetKernelCacheSize() const {
    std::unique_lock<std::mutex> lock(kernel_cache_mutex_);
    return kernel_cache_.size();
}

} // namespace ManagerOpenCL
