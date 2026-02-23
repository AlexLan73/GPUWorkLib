#pragma once

/**
 * @file rocm_core.hpp
 * @brief Per-device HIP/ROCm контекст (аналог opencl_core.hpp)
 *
 * ROCmCore управляет HIP контекстом для КОНКРЕТНОГО устройства.
 * Каждый экземпляр владеет СВОИМ device по device_index.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <array>
#include <mutex>
#include <stdexcept>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Утилита: Проверка HIP ошибок
// ════════════════════════════════════════════════════════════════════════════

inline void CheckHIPError(hipError_t error, const std::string& operation) {
  if (error != hipSuccess) {
    std::string error_msg = "HIP Error [" + std::to_string(static_cast<int>(error)) +
                            ": " + hipGetErrorString(error) + "] in " + operation;
    throw std::runtime_error(error_msg);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// ROCmCore - Per-Device HIP контекст (Multi-GPU поддержка)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class ROCmCore
 * @brief Управляет HIP контекстом для КОНКРЕТНОГО устройства
 *
 * MULTI-GPU ARCHITECTURE:
 * Каждый экземпляр ROCmCore владеет СВОИМ устройством по device_index.
 *
 * Ответственность:
 * - Инициализация HIP runtime и выбор девайса
 * - Создание и владение stream
 * - Информация о девайсе (hipDeviceProp_t)
 * - Thread-safe доступ
 *
 * НЕ управляет:
 * - Буферами (это делает GPUBuffer через IBackend)
 * - Программами/кернелами (это модули)
 */
class ROCmCore {
public:
  // ═══════════════════════════════════════════════════════════════
  // Конструктор и деструктор
  // ═══════════════════════════════════════════════════════════════

  explicit ROCmCore(int device_index = 0);
  ~ROCmCore();

  // ═══════════════════════════════════════════════════════════════
  // Запрет копирования, разрешение перемещения
  // ═══════════════════════════════════════════════════════════════
  ROCmCore(const ROCmCore&) = delete;
  ROCmCore& operator=(const ROCmCore&) = delete;
  ROCmCore(ROCmCore&& other) noexcept;
  ROCmCore& operator=(ROCmCore&& other) noexcept;

  // ═══════════════════════════════════════════════════════════════
  // Инициализация
  // ═══════════════════════════════════════════════════════════════

  void Initialize();
  void Cleanup();
  bool IsInitialized() const { return initialized_; }

  // ═══════════════════════════════════════════════════════════════
  // Getters для HIP объектов
  // ═══════════════════════════════════════════════════════════════

  hipDevice_t GetDevice() const { return device_; }
  hipStream_t GetStream() const { return stream_; }
  int GetDeviceIndex() const { return device_index_; }

  // ═══════════════════════════════════════════════════════════════
  // Информация о девайсе
  // ═══════════════════════════════════════════════════════════════

  std::string GetDeviceInfo() const;
  std::string GetDeviceName() const;
  std::string GetVendor() const;
  std::string GetArchName() const;
  size_t GetGlobalMemorySize() const;
  size_t GetFreeMemorySize() const;
  size_t GetLocalMemorySize() const;
  int GetComputeUnits() const;
  size_t GetMaxWorkGroupSize() const;
  size_t GetMaxClockFrequency() const;
  bool SupportsDoublePrecision() const;

  // ═══════════════════════════════════════════════════════════════
  // СТАТИЧЕСКИЕ МЕТОДЫ для обнаружения GPU (Multi-GPU support)
  // ═══════════════════════════════════════════════════════════════

  static int GetAvailableDeviceCount();
  static std::string GetAllDevicesInfo();

private:
  int device_index_;
  bool initialized_;

  hipDevice_t device_;
  hipStream_t stream_;
  hipDeviceProp_t device_props_;

  mutable std::mutex mutex_;

  void InitializeHIP();
  void ReleaseResources();
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
