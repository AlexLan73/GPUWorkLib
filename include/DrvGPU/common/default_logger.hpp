#pragma once

/**
 * @file default_logger.hpp
 * @brief DefaultLogger - Реализация ILogger на основе spdlog
 * 
 * Логирование ТОЛЬКО в файл с использованием spdlog.
 * Автоматически создаёт структуру папок для логов.
 * 
 * Поведение:
 * - ConfigLogger::IsEnabled() == true  -> пишем в файл
 * - ConfigLogger::IsEnabled() == false -> не логируем вообще
 * 
 * @author DrvGPU Team
 * @date 2026-02-01
 */

#include "logger_interface.hpp"
#include "config_logger.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <mutex>

namespace drv_gpu_lib {

/**
 * @class DefaultLogger
 * @brief Реализация ILogger на основе spdlog (файловое логирование)
 * 
 * Использует spdlog для:
 * - Логирования в файл с автоматическим созданием структуры папок
 * - Thread-safe логирования
 * 
 * Пример использования:
 * @code
 * // Включить логирование (по умолчанию включено)
 * ConfigLogger::GetInstance().Enable();
 * 
 * // Логировать сообщения (пишутся в файл)
 * DRVGPU_LOG_INFO("DrvGPU", "Initialized successfully");
 * DRVGPU_LOG_WARNING("OpenCL", "Memory low");
 * DRVGPU_LOG_ERROR("Backend", "Failed to allocate");
 * 
 * // Отключить логирование (ничего не пишется)
 * ConfigLogger::GetInstance().Disable();
 * @endcode
 */
class DefaultLogger : public ILogger {
public:
    /// Получить единственный экземпляр DefaultLogger
    static DefaultLogger& GetInstance();

    // ═══════════════════════════════════════════════════════════════════════
    // Реализация ILogger
    // ═══════════════════════════════════════════════════════════════════════

    void Debug(const std::string& component, const std::string& message) override;
    void Info(const std::string& component, const std::string& message) override;
    void Warning(const std::string& component, const std::string& message) override;
    void Error(const std::string& component, const std::string& message) override;

    bool IsDebugEnabled() const override;
    bool IsInfoEnabled() const override;
    bool IsWarningEnabled() const override;
    bool IsErrorEnabled() const override;

    void Reset() override;

    // ═══════════════════════════════════════════════════════════════════════
    // Дополнительные методы
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Форматировать сообщение с компонентом
     * @param component Имя компонента
     * @param message Текст сообщения
     * @return Отформатированное сообщение
     */
    static std::string FormatMessage(const std::string& component, 
                                      const std::string& message);

    /**
     * @brief Проверить, инициализирован ли логер
     * @return true если логер инициализирован
     */
    bool IsInitialized() const;

    /// Деструктор
    ~DefaultLogger();

    /// Конструктор
    DefaultLogger();

private:

    /// Инициализировать spdlog
    void Initialize();

    /// Очистить spdlog
    void Shutdown();

    /// Умный указатель на file sink
    std::shared_ptr<spdlog::sinks::basic_file_sink_st> file_sink_;

    /// Основной логер spdlog
    std::shared_ptr<spdlog::logger> logger_;

    /// Флаг инициализации
    bool initialized_;

    /// Мьютекс для thread-safety
    mutable std::mutex mutex_;

    /// Текущий уровень логирования
    spdlog::level::level_enum current_level_;
};

} // namespace drv_gpu_lib
