#include "logger.hpp"

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Logger Implementation - Реализация фасада системы логирования
// ════════════════════════════════════════════════════════════════════════════

// Статический член класса - текущий логер (по умолчанию nullptr)
ILoggerPtr Logger::current_logger_ = nullptr;

/**
 * @brief Получить текущий логер (создаёт DefaultLogger при первом вызове)
 * @return Ссылка на ILogger
 * 
 * Если логер ещё не установлен, создаёт DefaultLogger.
 * 
 * @code
 * ILogger& logger = Logger::GetInstance();
 * logger.Info("Component", "Message");
 * @endcode
 */
ILogger& Logger::GetInstance() {
    if (!current_logger_) {
        current_logger_ = std::make_shared<DefaultLogger>();
    }
    return *current_logger_;
}

/**
 * @brief Установить свой логер (для интеграции с продакшен логерами)
 * @param logger Умный указатель на ILogger
 * 
 * Позволяет заменить DefaultLogger на любой другой логер,
 * например, на корпоративный логер компании.
 * 
 * @code
 * auto my_logger = std::make_shared<MyCompanyLogger>();
 * Logger::SetInstance(my_logger);
 * @endcode
 */
void Logger::SetInstance(ILoggerPtr logger) {
    current_logger_ = logger;
}

/**
 * @brief Сбросить на стандартный DefaultLogger
 * 
 * Заменяет текущий логер на новый экземпляр DefaultLogger.
 */
void Logger::ResetToDefault() {
    current_logger_ = std::make_shared<DefaultLogger>();
}

/**
 * @brief Логировать отладочное сообщение (статический метод)
 * @param component Имя компонента
 * @param message Текст сообщения
 */
void Logger::Debug(const std::string& component, const std::string& message) {
    GetInstance().Debug(component, message);
}

/**
 * @brief Логировать информационное сообщение (статический метод)
 * @param component Имя компонента
 * @param message Текст сообщения
 */
void Logger::Info(const std::string& component, const std::string& message) {
    GetInstance().Info(component, message);
}

/**
 * @brief Логировать предупреждение (статический метод)
 * @param component Имя компонента
 * @param message Текст сообщения
 */
void Logger::Warning(const std::string& component, const std::string& message) {
    GetInstance().Warning(component, message);
}

/**
 * @brief Логировать ошибку (статический метод)
 * @param component Имя компонента
 * @param message Текст сообщения
 */
void Logger::Error(const std::string& component, const std::string& message) {
    GetInstance().Error(component, message);
}

/**
 * @brief Проверить, включено ли логирование
 * @return true если логирование включено
 * 
 * Делегирует проверку ConfigLogger::IsEnabled().
 */
bool Logger::IsEnabled() {
    return ConfigLogger::GetInstance().IsEnabled();
}

/**
 * @brief Включить логирование
 * 
 * Делегирует ConfigLogger::Enable().
 */
void Logger::Enable() {
    ConfigLogger::GetInstance().Enable();
}

/**
 * @brief Выключить логирование (production mode)
 * 
 * Делегирует ConfigLogger::Disable().
 * Рекомендуется использовать в production для повышения производительности.
 */
void Logger::Disable() {
    ConfigLogger::GetInstance().Disable();
}

} // namespace drv_gpu_lib
