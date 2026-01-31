#include "kernel_program.hpp"
#include <iostream>
#include <vector>
#include <functional>
#include <sstream>
#include <iomanip>

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// KernelProgram реализация
// ════════════════════════════════════════════════════════════════════════════

KernelProgram::KernelProgram(const std::string& source)
    : program_(nullptr),
      source_(source) {
    CompileProgram();
}

void KernelProgram::CompileProgram() {
    auto& core = OpenCLCore::GetInstance();
    cl_context context = core.GetContext();
    cl_device_id device = core.GetDevice();

    cl_int err;
    const char* source_str = source_.c_str();
    size_t source_len = source_.length();

    // Создать программу из исходника
    program_ = clCreateProgramWithSource(context, 1, &source_str, &source_len, &err);
    CheckCLError(err, "clCreateProgramWithSource");

    // Откомпилировать программу
    err = clBuildProgram(program_, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string error_msg = "Program compilation failed:\n" + GetBuildLog();
        clReleaseProgram(program_);
        program_ = nullptr;
        throw std::runtime_error(error_msg);
    }
}

std::string KernelProgram::GetBuildLog() const {
    auto& core = OpenCLCore::GetInstance();
    cl_device_id device = core.GetDevice();

    size_t log_size;
    cl_int err = clGetProgramBuildInfo(
        program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    CheckCLError(err, "clGetProgramBuildInfo (size)");

    std::vector<char> log(log_size);
    err = clGetProgramBuildInfo(
        program_, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    CheckCLError(err, "clGetProgramBuildInfo (get)");

    return std::string(log.data());
}

cl_kernel KernelProgram::GetOrCreateKernel(const std::string& kernel_name) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Проверить кэш
    auto it = kernel_cache_.find(kernel_name);
    if (it != kernel_cache_.end()) {
        return it->second;
    }

    // Создать новый kernel
    cl_int err;
    cl_kernel kernel = clCreateKernel(program_, kernel_name.c_str(), &err);
    CheckCLError(err, "clCreateKernel: " + kernel_name);

    // Добавить в кэш
    kernel_cache_[kernel_name] = kernel;

    std::cout << "[OK] Kernel '" << kernel_name << "' created\n";

    return kernel;
}

bool KernelProgram::HasKernel(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return kernel_cache_.find(kernel_name) != kernel_cache_.end();
}

KernelProgram::~KernelProgram() {
    // Освободить все kernels
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (auto& [name, kernel] : kernel_cache_) {
            if (kernel) clReleaseKernel(kernel);
        }
        kernel_cache_.clear();
    }

    // Освободить программу
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
}

KernelProgram::KernelProgram(KernelProgram&& other) noexcept
    : program_(other.program_),
      source_(std::move(other.source_)),
      kernel_cache_(std::move(other.kernel_cache_)) {
    other.program_ = nullptr;
}

KernelProgram& KernelProgram::operator=(KernelProgram&& other) noexcept {
    if (this != &other) {
        // Очистить текущие ресурсы
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            for (auto& [name, kernel] : kernel_cache_) {
                if (kernel) clReleaseKernel(kernel);
            }
        }
        if (program_) clReleaseProgram(program_);

        // Переместить ресурсы
        program_ = other.program_;
        source_ = std::move(other.source_);
        kernel_cache_ = std::move(other.kernel_cache_);

        other.program_ = nullptr;
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// KernelProgramCache реализация
// ════════════════════════════════════════════════════════════════════════════

std::unordered_map<std::string, std::shared_ptr<KernelProgram>> KernelProgramCache::cache_;
std::mutex KernelProgramCache::cache_mutex_;
size_t KernelProgramCache::cache_hits_ = 0;
size_t KernelProgramCache::cache_misses_ = 0;

std::shared_ptr<KernelProgram> KernelProgramCache::GetOrCompile(const std::string& source) {
    // Вычислить хеш исходника
    std::hash<std::string> hasher;
    std::string hash_key = std::to_string(hasher(source));

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Проверить кэш
        auto it = cache_.find(hash_key);
        if (it != cache_.end()) {
            cache_hits_++;
            return it->second;
        }
    }

    // Компилировать (вне блокировки, т.к. это дорогая операция)
    auto program = std::make_shared<KernelProgram>(source);

    // Добавить в кэш
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_misses_++;
        cache_[hash_key] = program;
    }

    return program;
}

std::string KernelProgramCache::GetCacheStatistics() {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    std::ostringstream oss;
    oss << "\nKernel Program Cache Statistics:\n";
    oss << " Cache size: " << cache_.size() << " programs\n";
    oss << " Cache hits: " << cache_hits_ << "\n";
    oss << " Cache misses: " << cache_misses_ << "\n";

    if (cache_hits_ + cache_misses_ > 0) {
        double hit_rate = 100.0 * cache_hits_ / (cache_hits_ + cache_misses_);
        oss << " Hit rate: " << std::fixed << std::setprecision(1) << hit_rate << "%\n";
    }

    return oss.str();
}

void KernelProgramCache::Clear() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
    std::cout << "[OK] KernelProgramCache cleared\n";
}

size_t KernelProgramCache::GetCacheSize() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.size();
}

}  // namespace ManagerOpenCL
