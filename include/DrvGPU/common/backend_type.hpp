#pragma once

/**
 * @file backend_type.hpp
 * @brief Перечисление типов бэкендов
 */

namespace drv_gpu_lib {

/**
 * @enum BackendType
 * @brief Типы поддерживаемых GPU бэкендов
 */
enum class BackendType {
    OPENCL,       ///< OpenCL backend (реализовано)
    CUDA,         ///< CUDA backend (запланировано)
    VULKAN,       ///< Vulkan backend (запланировано)
    ROCm,         ///< ROCm backend (будущее)
    OPENCLandROCm,   ///< OPENCLandROCm Compute backend
    AUTO          ///< Автоматический выбор
};

/**
 * @brief Конвертировать BackendType в строку
 */
inline const char* BackendTypeToString(BackendType type) {
    switch (type) {
        case BackendType::OPENCL:       return "OpenCL";
        case BackendType::CUDA:         return "CUDA";
        case BackendType::VULKAN:       return "Vulkan";
        case BackendType::ROCm:         return "ROCm";
        case BackendType::OPENCLandROCm: return "OpenCLandROCm";
        case BackendType::AUTO:         return "Auto";
        default:                        return "Unknown";
    }
}

} // namespace DrvGPU
