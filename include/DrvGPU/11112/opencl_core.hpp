#pragma once

#include <CL/cl.h>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <array>

// Forward declaration для SVMCapabilities
namespace ManagerOpenCL { struct SVMCapabilities; }

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// ENUM для типа девайса
// ════════════════════════════════════════════════════════════════════════════

enum class DeviceType {
    GPU,  // CL_DEVICE_TYPE_GPU
    CPU   // CL_DEVICE_TYPE_CPU
};

// ════════════════════════════════════════════════════════════════════════════
// OpenCLCore - Singleton контекст OpenCL
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class OpenCLCore
 * @brief Управляет единым OpenCL контекстом на приложение
 *
 * Ответственность:
 * - Инициализация платформы и девайса
 * - Создание и владение контекстом OpenCL
 * - Информация о девайсе
 * - Thread-safe доступ к контексту
 *
 * НЕ управляет:
 * - Command queues (это делает CommandQueuePool)
 * - Программы (это делает KernelProgram)
 * - Буферы (это делает GPUMemoryBuffer)
 */
class OpenCLCore {
public:
    // ═══════════════════════════════════════════════════════════════
    // Singleton интерфейс
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Инициализировать OpenCL контекст (один раз)
     * @param device_type GPU или CPU
     * @throws std::runtime_error если инициализация не удалась
     */
    static void Initialize(DeviceType device_type = DeviceType::GPU);

    /**
     * @brief Получить Singleton (thread-safe)
     * @throws std::runtime_error если не инициализирован
     */
    static OpenCLCore& GetInstance();

    /**
     * @brief Проверить инициализацию
     */
    static bool IsInitialized();

    /**
     * @brief Очистка ресурсов (опционально, вызывается в деструкторе)
     */
    static void Cleanup();

    // ═══════════════════════════════════════════════════════════════
    // Getters для OpenCL объектов
    // ═══════════════════════════════════════════════════════════════

    cl_context GetContext() const { return context_; }
    cl_device_id GetDevice() const { return device_; }
    cl_platform_id GetPlatform() const { return platform_; }

    // ═══════════════════════════════════════════════════════════════
    // Информация о девайсе
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Получить информацию о девайсе (красивый вывод)
     */
    std::string GetDeviceInfo() const;

    /**
     * @brief Получить имя девайса
     */
    std::string GetDeviceName() const;

    /**
     * @brief Получить вендора (NVIDIA, AMD, Intel)
     */
    std::string GetVendor() const;

    /**
     * @brief Получить версию драйвера
     */
    std::string GetDriverVersion() const;

    /**
     * @brief Получить размер глобальной памяти в байтах
     */
    size_t GetGlobalMemorySize() const;

    /**
     * @brief Получить размер локальной памяти в байтах
     */
    size_t GetLocalMemorySize() const;

    /**
     * @brief Получить количество compute units
     */
    cl_uint GetComputeUnits() const;

    /**
     * @brief Получить максимальный размер work group
     */
    size_t GetMaxWorkGroupSize() const;

    /**
     * @brief Получить максимальный размер работы для одного измерения
     */
    std::array<size_t, 3> GetMaxWorkItemSizes() const;

    // ═══════════════════════════════════════════════════════════════
    // SVM (Shared Virtual Memory) информация - OpenCL 2.0+
    // ═══════════════════════════════════════════════════════════════

    /**
     * @brief Получить версию OpenCL (major)
     */
    cl_uint GetOpenCLVersionMajor() const;

    /**
     * @brief Получить версию OpenCL (minor)
     */
    cl_uint GetOpenCLVersionMinor() const;

    /**
     * @brief Проверить поддержку SVM
     * @return true если OpenCL >= 2.0 и хотя бы один тип SVM поддерживается
     */
    bool IsSVMSupported() const;

    /**
     * @brief Получить SVM capabilities устройства
     * Включает header "svm_capabilities.hpp" для полного определения
     */
    SVMCapabilities GetSVMCapabilities() const;

    /**
     * @brief Получить информацию о SVM (красивый вывод)
     */
    std::string GetSVMInfo() const;

    // ═══════════════════════════════════════════════════════════════
    // Деструктор
    // ═══════════════════════════════════════════════════════════════

    ~OpenCLCore();

    // Запрет копирования
    OpenCLCore(const OpenCLCore&) = delete;
    OpenCLCore& operator=(const OpenCLCore&) = delete;

private:
    // ═══════════════════════════════════════════════════════════════
    // Singleton реализация
    // ═══════════════════════════════════════════════════════════════

    OpenCLCore();

    static std::unique_ptr<OpenCLCore> instance_;
    static bool initialized_;
    static std::mutex initialization_mutex_;

    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════

    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    DeviceType device_type_;

    // ═══════════════════════════════════════════════════════════════
    // Приватные методы инициализации
    // ═══════════════════════════════════════════════════════════════

    void InitializeOpenCL(DeviceType device_type);
    void ReleaseResources();

    // Утилиты для информации о девайсе
    template<typename T>
    T GetDeviceInfoValue(cl_device_info param) const;

    std::string GetDeviceInfoString(cl_device_info param) const;
};

// ════════════════════════════════════════════════════════════════════════════
// Утилита: Проверка OpenCL ошибок (inline для удобства)
// ════════════════════════════════════════════════════════════════════════════

inline void CheckCLError(cl_int error, const std::string& operation) {
    if (error != CL_SUCCESS) {
        std::string error_msg = "OpenCL Error [" + std::to_string(error) + "] in " + operation;
        throw std::runtime_error(error_msg);
    }
}

} // namespace ManagerOpenCL
