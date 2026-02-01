#include "opencl_core.hpp"
#include "../../memory/svm_capabilities.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <array>
#include <cstdio>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Static инициализация
// ════════════════════════════════════════════════════════════════════════════

std::unique_ptr<OpenCLCore> OpenCLCore::instance_ = nullptr;
bool OpenCLCore::initialized_ = false;
std::mutex OpenCLCore::initialization_mutex_;

// ════════════════════════════════════════════════════════════════════════════
// Публичные статические методы
// ════════════════════════════════════════════════════════════════════════════

void OpenCLCore::Initialize(DeviceType device_type) {
    std::lock_guard<std::mutex> lock(initialization_mutex_);

    if (initialized_) {
        std::cerr << "[WARNING] OpenCLCore already initialized\n";
        return;
    }

    instance_ = std::unique_ptr<OpenCLCore>(new OpenCLCore());
    instance_->InitializeOpenCL(device_type);
    initialized_ = true;

    std::cout << "[OK] OpenCLCore initialized\n";
    std::cout << instance_->GetDeviceInfo();
}

OpenCLCore& OpenCLCore::GetInstance() {
    if (!initialized_) {
        throw std::runtime_error(
            "OpenCLCore not initialized. Call Initialize() first.");
    }
    return *instance_;
}

bool OpenCLCore::IsInitialized() {
    return initialized_;
}

void OpenCLCore::Cleanup() {
    std::lock_guard<std::mutex> lock(initialization_mutex_);

    if (initialized_) {
        instance_->ReleaseResources();
        instance_.reset();
        initialized_ = false;
        std::cout << "[OK] OpenCLCore cleaned up\n";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Конструктор
// ════════════════════════════════════════════════════════════════════════════

OpenCLCore::OpenCLCore()
    : platform_(nullptr),
      device_(nullptr),
      context_(nullptr),
      device_type_(DeviceType::GPU) {
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация OpenCL
// ════════════════════════════════════════════════════════════════════════════

void OpenCLCore::InitializeOpenCL(DeviceType device_type) {
    device_type_ = device_type;

    cl_int err;
    cl_uint num_platforms = 0;

    // 1. Получить платформы
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    CheckCLError(err, "clGetPlatformIDs (count)");

    if (num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    CheckCLError(err, "clGetPlatformIDs (get)");

    platform_ = platforms[0];  // Используем первую платформу

    // 2. Получить девайсы
    cl_device_type cl_device_type =
        (device_type == DeviceType::GPU) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform_, cl_device_type, 0, nullptr, &num_devices);
    CheckCLError(err, "clGetDeviceIDs (count)");

    if (num_devices == 0) {
        throw std::runtime_error("No OpenCL devices found for specified type");
    }

    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform_, cl_device_type, num_devices, devices.data(), nullptr);
    CheckCLError(err, "clGetDeviceIDs (get)");

    device_ = devices[0];  // Используем первый девайс

    // 3. Создать контекст
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    CheckCLError(err, "clCreateContext");
}

// ════════════════════════════════════════════════════════════════════════════
// Деструктор и очистка
// ════════════════════════════════════════════════════════════════════════════

OpenCLCore::~OpenCLCore() {
    ReleaseResources();
}

void OpenCLCore::ReleaseResources() {
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    if (device_) {
        clReleaseDevice(device_);
        device_ = nullptr;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Информация о девайсе - приватные утилиты
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
T OpenCLCore::GetDeviceInfoValue(cl_device_info param) const {
    T value;
    cl_int err = clGetDeviceInfo(device_, param, sizeof(T), &value, nullptr);
    CheckCLError(err, "clGetDeviceInfo");
    return value;
}

std::string OpenCLCore::GetDeviceInfoString(cl_device_info param) const {
    size_t size = 0;
    cl_int err = clGetDeviceInfo(device_, param, 0, nullptr, &size);
    CheckCLError(err, "clGetDeviceInfo (size)");

    std::vector<char> buffer(size);
    err = clGetDeviceInfo(device_, param, size, buffer.data(), nullptr);
    CheckCLError(err, "clGetDeviceInfo (get)");

    return std::string(buffer.data());
}

// ════════════════════════════════════════════════════════════════════════════
// Публичные методы получения информации
// ════════════════════════════════════════════════════════════════════════════

std::string OpenCLCore::GetDeviceName() const {
    return GetDeviceInfoString(CL_DEVICE_NAME);
}

std::string OpenCLCore::GetVendor() const {
    return GetDeviceInfoString(CL_DEVICE_VENDOR);
}

std::string OpenCLCore::GetDriverVersion() const {
    return GetDeviceInfoString(CL_DRIVER_VERSION);
}

size_t OpenCLCore::GetGlobalMemorySize() const {
    return GetDeviceInfoValue<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE);
}

size_t OpenCLCore::GetLocalMemorySize() const {
    return GetDeviceInfoValue<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE);
}

cl_uint OpenCLCore::GetComputeUnits() const {
    return GetDeviceInfoValue<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
}

size_t OpenCLCore::GetMaxWorkGroupSize() const {
    return GetDeviceInfoValue<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
}

std::array<size_t, 3> OpenCLCore::GetMaxWorkItemSizes() const {
    std::array<size_t, 3> sizes;
    cl_int err = clGetDeviceInfo(
        device_,
        CL_DEVICE_MAX_WORK_ITEM_SIZES,
        sizeof(sizes),
        sizes.data(),
        nullptr
    );
    CheckCLError(err, "clGetDeviceInfo (MAX_WORK_ITEM_SIZES)");
    return sizes;
}

// ════════════════════════════════════════════════════════════════════════════
// GetDeviceInfo - красивый вывод
// ════════════════════════════════════════════════════════════════════════════

std::string OpenCLCore::GetDeviceInfo() const {
    std::ostringstream oss;

    oss << "\n" << std::string(70, '=') << "\n";
    oss << "OpenCL Device Information\n";
    oss << std::string(70, '=') << "\n\n";

    oss << std::left << std::setw(25) << "Device Name:" << GetDeviceName() << "\n";
    oss << std::left << std::setw(25) << "Vendor:" << GetVendor() << "\n";
    oss << std::left << std::setw(25) << "Driver Version:" << GetDriverVersion() << "\n";

    // Device type
    oss << std::left << std::setw(25) << "Device Type:";
    oss << (device_type_ == DeviceType::GPU ? "GPU" : "CPU") << "\n";

    // Memory
    size_t global_mem = GetGlobalMemorySize();
    size_t local_mem = GetLocalMemorySize();
    oss << std::left << std::setw(25) << "Global Memory:"
        << std::fixed << std::setprecision(2)
        << (global_mem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    oss << std::left << std::setw(25) << "Local Memory:"
        << (local_mem / 1024.0) << " KB\n";

    // Compute units
    oss << std::left << std::setw(25) << "Compute Units:"
        << GetComputeUnits() << "\n";

    // Work group size
    oss << std::left << std::setw(25) << "Max Work Group Size:"
        << GetMaxWorkGroupSize() << "\n";

    // Work item sizes
    auto sizes = GetMaxWorkItemSizes();
    oss << std::left << std::setw(25) << "Max Work Item Sizes:"
        << "[" << sizes[0] << ", " << sizes[1] << ", " << sizes[2] << "]\n";

    oss << "\n" << std::string(70, '=') << "\n\n";

    return oss.str();
}

// ════════════════════════════════════════════════════════════════════════════
// SVM методы
// ════════════════════════════════════════════════════════════════════════════

cl_uint OpenCLCore::GetOpenCLVersionMajor() const {
    char version_str[256] = {0};
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(version_str), version_str, nullptr);
    if (err != CL_SUCCESS) return 0;
    
    int major = 0, minor = 0;
    if (sscanf(version_str, "OpenCL %d.%d", &major, &minor) >= 1) {
        return static_cast<cl_uint>(major);
    }
    return 0;
}

cl_uint OpenCLCore::GetOpenCLVersionMinor() const {
    char version_str[256] = {0};
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(version_str), version_str, nullptr);
    if (err != CL_SUCCESS) return 0;
    
    int major = 0, minor = 0;
    if (sscanf(version_str, "OpenCL %d.%d", &major, &minor) == 2) {
        return static_cast<cl_uint>(minor);
    }
    return 0;
}

bool OpenCLCore::IsSVMSupported() const {
    if (GetOpenCLVersionMajor() < 2) {
        return false;
    }
    
    cl_device_svm_capabilities svm_caps = 0;
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);
    
    return (err == CL_SUCCESS && svm_caps != 0);
}

SVMCapabilities OpenCLCore::GetSVMCapabilities() const {
    return SVMCapabilities::Query(device_);
}

std::string OpenCLCore::GetSVMInfo() const {
    std::ostringstream oss;
    
    oss << "\n" << std::string(60, '═') << "\n";
    oss << "SVM Capabilities\n";
    oss << std::string(60, '═') << "\n\n";
    
    cl_uint major = GetOpenCLVersionMajor();
    cl_uint minor = GetOpenCLVersionMinor();
    
    oss << std::left << std::setw(25) << "OpenCL Version:" << major << "." << minor << "\n";
    
    if (major < 2) {
        oss << std::left << std::setw(25) << "SVM Supported:" << "NO (OpenCL < 2.0)\n";
        oss << std::string(60, '═') << "\n";
        return oss.str();
    }
    
    cl_device_svm_capabilities svm_caps = 0;
    cl_int err = clGetDeviceInfo(device_, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);
    
    if (err != CL_SUCCESS || svm_caps == 0) {
        oss << std::left << std::setw(25) << "SVM Supported:" << "NO\n";
        oss << std::string(60, '═') << "\n";
        return oss.str();
    }
    
    oss << std::left << std::setw(25) << "SVM Supported:" << "YES ✅\n\n";
    
    oss << "SVM Types:\n";
    oss << "  " << std::left << std::setw(23) << "Coarse-Grain Buffer:" 
        << ((svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) ? "YES ✅" : "NO ❌") << "\n";
    oss << "  " << std::left << std::setw(23) << "Fine-Grain Buffer:" 
        << ((svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) ? "YES ✅" : "NO ❌") << "\n";
    oss << "  " << std::left << std::setw(23) << "Fine-Grain System:" 
        << ((svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) ? "YES ✅" : "NO ❌") << "\n";
    oss << "  " << std::left << std::setw(23) << "Atomics:" 
        << ((svm_caps & CL_DEVICE_SVM_ATOMICS) ? "YES ✅" : "NO ❌") << "\n";
    
    oss << "\n" << std::string(60, '═') << "\n";

    return oss.str();
}

}  // namespace ManagerOpenCL
