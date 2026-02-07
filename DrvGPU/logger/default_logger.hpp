#pragma once

/**
 * @file default_logger.hpp
 * @brief DefaultLogger - Реализация ILogger на основе plog
 *
 * Логирование ТОЛЬКО в файл с использованием plog (header-only).
 * Автоматически создаёт структуру папок для логов.
 *
 * Поведение:
 * - ConfigLogger::IsEnabled() == true  -> пишем в файл
 * - ConfigLogger::IsEnabled() == false -> не логируем вообще
 *
 * ЗАМЕНА spdlog → plog:
 * - plog — header-only, нет зависимостей (нет fmt)
 * - plog — стабильный, кросс-платформенный (Windows/Linux/macOS)
 * - plog — простой API, rolling файлы, thread-safe
 *
 * @author DrvGPU Team
 * @date 2026-02-01
 * @modified 2026-02-07 (spdlog → plog migration)
 */

#include "../interface/i_logger.hpp"
#include "../logger/config_logger.hpp"

// ═══════════════════════════════════════════════════════════════════════════
// plog — header-only logging library
// https://github.com/SergiusTheBest/plog
// ═══════════════════════════════════════════════════════════════════════════
#include <plog/Log.h>
#include <plog/Initializers/RollingFileInitializer.h>

#include <memory>
#include <mutex>
#include <string>

namespace drv_gpu_lib {

// ═══════════════════════════════════════════════════════════════════════════
// Уровни логирования plog:
//   plog::verbose  = 6 (самый детальный)
//   plog::debug    = 5
//   plog::info     = 4
//   plog::warning  = 3
//   plog::error    = 2
//   plog::fatal    = 1
//   plog::none     = 0 (отключено)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @class DefaultLogger
 * @brief Реализация ILogger на основе plog (файловое логирование)
 *
 * Использует plog для:
 * - Логирования в файл с автоматическим созданием структуры папок
 * - Thread-safe логирования
 * - Rolling файлов (автоматическая ротация по размеру)
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

    /// Инициализировать plog
    void Initialize();

    /// Очистить plog
    void Shutdown();

    /// Флаг инициализации
    bool initialized_;

    /// Мьютекс для thread-safety
    mutable std::mutex mutex_;

    /// Текущий уровень логирования (plog severity)
    plog::Severity current_level_;
};

} // namespace drv_gpu_lib
