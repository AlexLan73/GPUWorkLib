#if ENABLE_ROCM

#include "rocm_core.hpp"
#include "../../logger/logger.hpp"
#include <sstream>
#include <iomanip>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Конструктор
// ════════════════════════════════════════════════════════════════════════════

ROCmCore::ROCmCore(int device_index)
    : device_index_(device_index),
      initialized_(false),
      device_(0),
      stream_(nullptr),
      device_props_{} {
  DRVGPU_LOG_DEBUG_GPU(device_index, "ROCmCore",
                       "Created for device index " + std::to_string(device_index));
}

// ════════════════════════════════════════════════════════════════════════════
// Деструктор
// ════════════════════════════════════════════════════════════════════════════

ROCmCore::~ROCmCore() {
  if (initialized_) {
    ReleaseResources();
    initialized_ = false;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics
// ════════════════════════════════════════════════════════════════════════════

ROCmCore::ROCmCore(ROCmCore&& other) noexcept
    : device_index_(other.device_index_),
      initialized_(other.initialized_),
      device_(other.device_),
      stream_(other.stream_),
      device_props_(other.device_props_) {
  other.initialized_ = false;
  other.device_ = 0;
  other.stream_ = nullptr;
}

ROCmCore& ROCmCore::operator=(ROCmCore&& other) noexcept {
  if (this != &other) {
    Cleanup();
    device_index_ = other.device_index_;
    initialized_ = other.initialized_;
    device_ = other.device_;
    stream_ = other.stream_;
    device_props_ = other.device_props_;
    other.initialized_ = false;
    other.device_ = 0;
    other.stream_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Инициализация
// ════════════════════════════════════════════════════════════════════════════

void ROCmCore::Initialize() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    DRVGPU_LOG_WARNING_GPU(device_index_, "ROCmCore",
                           "Device " + std::to_string(device_index_) + " already initialized");
    return;
  }

  InitializeHIP();
  initialized_ = true;

  DRVGPU_LOG_INFO_GPU(device_index_, "ROCmCore",
                      "Device " + std::to_string(device_index_) + " initialized: " + GetDeviceName());
}

void ROCmCore::InitializeHIP() {
  // Шаг 1: Инициализация HIP runtime
  CheckHIPError(hipInit(0), "hipInit");

  // Шаг 2: Проверить количество устройств
  int device_count = 0;
  CheckHIPError(hipGetDeviceCount(&device_count), "hipGetDeviceCount");

  if (device_count == 0) {
    throw std::runtime_error("No HIP/ROCm devices found");
  }

  if (device_index_ < 0 || device_index_ >= device_count) {
    throw std::runtime_error(
        "Invalid device index: " + std::to_string(device_index_) +
        ". Available devices: " + std::to_string(device_count));
  }

  // Шаг 3: Выбрать устройство
  CheckHIPError(hipSetDevice(device_index_), "hipSetDevice");

  // Шаг 4: Получить device handle
  CheckHIPError(hipDeviceGet(&device_, device_index_), "hipDeviceGet");

  // Шаг 5: Получить свойства устройства
  CheckHIPError(hipGetDeviceProperties(&device_props_, device_index_),
                "hipGetDeviceProperties");

  // Шаг 6: Создать stream
  CheckHIPError(hipStreamCreate(&stream_), "hipStreamCreate");

  DRVGPU_LOG_DEBUG_GPU(device_index_, "ROCmCore",
                       "HIP initialized for device " + std::to_string(device_index_));
}

// ════════════════════════════════════════════════════════════════════════════
// Очистка ресурсов
// ════════════════════════════════════════════════════════════════════════════

void ROCmCore::Cleanup() {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    ReleaseResources();
    initialized_ = false;
    DRVGPU_LOG_DEBUG_GPU(device_index_, "ROCmCore",
                         "Device " + std::to_string(device_index_) + " cleaned up");
  }
}

void ROCmCore::ReleaseResources() {
  if (stream_) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
  device_ = 0;
}

// ════════════════════════════════════════════════════════════════════════════
// СТАТИЧЕСКИЕ МЕТОДЫ - Multi-GPU Discovery
// ════════════════════════════════════════════════════════════════════════════

int ROCmCore::GetAvailableDeviceCount() {
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess) {
    return 0;
  }
  return count;
}

std::string ROCmCore::GetAllDevicesInfo() {
  std::ostringstream oss;
  int count = GetAvailableDeviceCount();

  oss << "\n" << std::string(70, '=') << "\n";
  oss << "Available ROCm/HIP GPU Devices\n";
  oss << std::string(70, '=') << "\n\n";

  if (count == 0) {
    oss << "  No devices found!\n";
  } else {
    for (int i = 0; i < count; ++i) {
      hipDeviceProp_t props;
      hipError_t err = hipGetDeviceProperties(&props, i);
      if (err != hipSuccess) continue;

      oss << "  [" << i << "] " << props.name << "\n";
      oss << "      GCN Arch: " << props.gcnArchName << "\n";
      oss << "      Memory: " << std::fixed << std::setprecision(2)
          << (props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
      oss << "      Compute Units: " << props.multiProcessorCount << "\n";
      oss << "      Clock: " << props.clockRate / 1000 << " MHz\n\n";
    }
  }

  oss << std::string(70, '=') << "\n";
  return oss.str();
}

// ════════════════════════════════════════════════════════════════════════════
// Информация о девайсе
// ════════════════════════════════════════════════════════════════════════════

std::string ROCmCore::GetDeviceName() const {
  if (!initialized_) return "Unknown";
  return std::string(device_props_.name);
}

std::string ROCmCore::GetVendor() const {
  return "AMD";
}

std::string ROCmCore::GetArchName() const {
  if (!initialized_) return "Unknown";
  return std::string(device_props_.gcnArchName);
}

size_t ROCmCore::GetGlobalMemorySize() const {
  if (!initialized_) return 0;
  return device_props_.totalGlobalMem;
}

size_t ROCmCore::GetFreeMemorySize() const {
  if (!initialized_) return 0;

  size_t free_mem = 0;
  size_t total_mem = 0;
  hipError_t err = hipMemGetInfo(&free_mem, &total_mem);
  if (err == hipSuccess) {
    return free_mem;
  }

  // Fallback: 90% от общей
  return static_cast<size_t>(static_cast<double>(device_props_.totalGlobalMem) * 0.9);
}

size_t ROCmCore::GetLocalMemorySize() const {
  if (!initialized_) return 0;
  return device_props_.sharedMemPerBlock;
}

int ROCmCore::GetComputeUnits() const {
  if (!initialized_) return 0;
  return device_props_.multiProcessorCount;
}

size_t ROCmCore::GetMaxWorkGroupSize() const {
  if (!initialized_) return 0;
  return static_cast<size_t>(device_props_.maxThreadsPerBlock);
}

size_t ROCmCore::GetMaxClockFrequency() const {
  if (!initialized_) return 0;
  return static_cast<size_t>(device_props_.clockRate / 1000);  // kHz -> MHz
}

bool ROCmCore::SupportsDoublePrecision() const {
  if (!initialized_) return false;
  // HIP: check arch >= gfx900 basically always supports fp64
  return device_props_.arch.hasDoubles != 0;
}

std::string ROCmCore::GetDeviceInfo() const {
  std::ostringstream oss;

  oss << "\n" << std::string(70, '=') << "\n";
  oss << "ROCm/HIP Device [" << device_index_ << "] Information\n";
  oss << std::string(70, '=') << "\n\n";

  oss << std::left << std::setw(25) << "Device Index:" << device_index_ << "\n";
  oss << std::left << std::setw(25) << "Device Name:" << GetDeviceName() << "\n";
  oss << std::left << std::setw(25) << "Vendor:" << GetVendor() << "\n";
  oss << std::left << std::setw(25) << "GCN Arch:" << GetArchName() << "\n";

  size_t global_mem = GetGlobalMemorySize();
  size_t local_mem = GetLocalMemorySize();
  oss << std::left << std::setw(25) << "Global Memory:" << std::fixed << std::setprecision(2)
      << (global_mem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
  oss << std::left << std::setw(25) << "Local Memory:" << (local_mem / 1024.0) << " KB\n";
  oss << std::left << std::setw(25) << "Free Memory:" << std::fixed << std::setprecision(2)
      << (GetFreeMemorySize() / (1024.0 * 1024.0 * 1024.0)) << " GB\n";

  oss << std::left << std::setw(25) << "Compute Units:" << GetComputeUnits() << "\n";
  oss << std::left << std::setw(25) << "Max Work Group Size:" << GetMaxWorkGroupSize() << "\n";
  oss << std::left << std::setw(25) << "Clock:" << GetMaxClockFrequency() << " MHz\n";
  oss << std::left << std::setw(25) << "Double Precision:" << (SupportsDoublePrecision() ? "YES" : "NO") << "\n";

  oss << "\n" << std::string(70, '=') << "\n";
  return oss.str();
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
