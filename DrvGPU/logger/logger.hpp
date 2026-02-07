#pragma once

/**
 * @file logger.hpp
 * @brief Logger - Главный фасад системы логирования DrvGPU
 * 
 * Предоставляет:
 * - Макросы DRVGPU_LOG_* для удобного логирования
 * - Фабрику Logger для установки своего логера
 * - Условную компиляцию (отключение в Release)
 * 
 * Уровни логирования:
 * - DEBUG: Подробная отладка (только в Debug сборке)
 * - INFO: Информационные сообщения
 * - WARNING: Предупреждения
 * - ERROR: Ошибки
 * 
 * Пример использования:
 * @code
 * #include "common/logger.hpp"
 * 
 * DRVGPU_LOG_INFO("DrvGPU", "Initialized successfully");
 * DRVGPU_LOG_WARNING("OpenCL", "Memory allocation warning");
 * DRVGPU_LOG_ERROR("Backend", "Failed to create context");
 * 
 * // В продакшене можно установить свой логер:
 * Logger::SetInstance(my_company_logger);
 * @endcode
 * 
 * @author DrvGPU Team
 * @date 2026-02-01
 */

// #include "../common/logger_interface.hpp"
 #include "../interface/i_logger.hpp"
#include "../logger/default_logger.hpp"
#include "../logger/config_logger.hpp"
#include <string>

namespace drv_gpu_lib {
 * - Получить текущий логер
 * - Установить свой логер (для продакшена)
 * - Быстрое логирование через статические методы
 */
class Logger {
public:
    /**
     * @brief Получить текущий логер
     * @return Ссылка на ILogger
     */
    static ILogger& GetInstance();

    /**
     * @brief Установить свой логер (для продакшена)
     * @param logger Умный указатель на ILogger
     */
    static void SetInstance(ILoggerPtr logger);

    /**
     * @brief Сбросить на стандартный логер
     */
    static void ResetToDefault();

    /**
     * @brief Логировать отладочное сообщение
     */
    static void Debug(const std::string& component, const std::string& message);

    /**
     * @brief Логировать информационное сообщение
     */
    static void Info(const std::string& component, const std::string& message);

    /**
     * @brief Логировать предупреждение
     */
    static void Warning(const std::string& component, const std::string& message);

    /**
     * @brief Логировать ошибку
     */
    static void Error(const std::string& component, const std::string& message);

    /**
     * @brief Проверить, включено ли логирование
     */
    static bool IsEnabled();

    /**
     * @brief Включить логирование
     */
    static void Enable();

    /**
     * @brief Выключить логирование (production mode)
     */
    static void Disable();

private:
    /// Текущий логер (по умолчанию DefaultLogger)
    static ILoggerPtr current_logger_;
};

// ════════════════════════════════════════════════════════════════════════════
// Макросы логирования
// ═══════════════════════════════════════════════════════════════════════════=

#ifdef NDEBUG
    // ═══════════════════════════════════════════════════════════════════════
    // Release сборка: DEBUG отключён, остальные уровни активны
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Логирование отладочного сообщения (только Debug)
     */
    #define DRVGPU_LOG_DEBUG(component, message) \
        ((void)0)

    /**
     * @brief Логирование информационного сообщения
     */
    #define DRVGPU_LOG_INFO(component, message) \
        do { \
            if (drv_gpu_lib::Logger::IsEnabled()) { \
                drv_gpu_lib::Logger::Info(component, message); \
            } \
        } while (0)

    /**
     * @brief Логирование предупреждения
     */
    #define DRVGPU_LOG_WARNING(component, message) \
        do { \
            if (drv_gpu_lib::Logger::IsEnabled()) { \
                drv_gpu_lib::Logger::Warning(component, message); \
            } \
        } while (0)

    /**
     * @brief Логирование ошибки
     */
    #define DRVGPU_LOG_ERROR(component, message) \
        do { \
            if (drv_gpu_lib::Logger::IsEnabled()) { \
                drv_gpu_lib::Logger::Error(component, message); \
            } \
        } while (0)

#else
    // ═══════════════════════════════════════════════════════════════════════
    // Debug сборка: все уровни активны
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Логирование отладочного сообщения (Debug only)
     */
    #define DRVGPU_LOG_DEBUG(component, message) \
        do { \
            if (drv_gpu_lib::Logger::IsEnabled()) { \
                drv_gpu_lib::Logger::Debug(component, message); \
            } \
        } while (0)

    /**
     * @brief Логирование информационного сообщения
     */
    #define DRVGPU_LOG_INFO(component, message) \
        do { \
            if (drv_gpu_lib::Logger::IsEnabled()) { \
                drv_gpu_lib::Logger::Info(component, message); \
            } \
        } while (0)

    /**
     * @brief Логирование предупреждения
     */
    #define DRVGPU_LOG_WARNING(component, message) \
        do { \
            if (drv_gpu_lib::Logger::IsEnabled()) { \
                drv_gpu_lib::Logger::Warning(component, message); \
            } \
        } while (0)

    /**
     * @brief Логирование ошибки
     */
    #define DRVGPU_LOG_ERROR(component, message) \
        do { \
            if (drv_gpu_lib::Logger::IsEnabled()) { \
                drv_gpu_lib::Logger::Error(component, message); \
            } \
        } while (0)

#endif // NDEBUG

// ════════════════════════════════════════════════════════════════════════════
// Устаревшие макросы (для совместимости)
// ═══════════════════════════════════════════════════════════════════════════=

/// Устаревший макрос для информации (используйте DRVGPU_LOG_INFO)
#define DRVGPU_LOG DRVGPU_LOG_INFO

} // namespace drv_gpu_lib
