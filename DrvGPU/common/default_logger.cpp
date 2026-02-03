#include "default_logger.hpp"
#include <iostream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// DefaultLogger Implementation - Реализация файлового логирования на spdlog
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Получить единственный экземпляр DefaultLogger (Singleton)
 * @return Ссылка на статический экземпляр
 * 
 * Thread-safe инициализация через static local variable.
 */
DefaultLogger& DefaultLogger::GetInstance() {
    static DefaultLogger instance;
    return instance;
}

/**
 * @brief Конструктор DefaultLogger
 * 
 * Инициализирует:
 * - initialized_ = false
 * - current_level_ = debug
 * - Вызывает Initialize() для настройки spdlog
 */
DefaultLogger::DefaultLogger()
    : initialized_(false)
    , current_level_(spdlog::level::debug) {
    Initialize();
}

/**
 * @brief Деструктор DefaultLogger
 * 
 * Вызывает Shutdown() для корректного завершения работы spdlog.
 */
DefaultLogger::~DefaultLogger() {
    Shutdown();
}

/**
 * @brief Инициализировать spdlog file logger
 * 
 * Логика инициализации:
 * 1. Проверяем ConfigLogger::IsEnabled()
 * 2. Если отключено - помечаем как инициализированный без логера
 * 3. Если включено:
 *    - Создаём директорию для логов
 *    - Получаем путь к файлу лога
 *    - Создаём basic_file_sink_st
 *    - Настраиваем паттерн форматирования
 *    - Создаём logger с file sink
 *    - Устанавливаем как default logger spdlog
 * 
 * @note Если CreateLogDirectory() или file sink создать не удалось,
 *       логер помечается как инициализированный, но без функциональности.
 */
void DefaultLogger::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return;
    }

    // Проверяем, включено ли логирование
    // Только если IsEnabled() == true - создаём file sink и пишем в файл
    if (!ConfigLogger::GetInstance().IsEnabled()) {
        initialized_ = true;  // Помечаем как "инициализированный", но без логера
        return;
    }

    try {
        // Создаём директорию для логов
        ConfigLogger::GetInstance().CreateLogDirectory();

        // Получаем путь к файлу лога (с датой и временем)
        std::string log_file_path = ConfigLogger::GetInstance().GetLogFilePath();

        // Создаём только file sink (без console sink для производительности)
        // basic_file_sink_st - synchronous file sink (без буферизации)
        file_sink_ = std::make_shared<spdlog::sinks::basic_file_sink_st>(log_file_path, true);
        file_sink_->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] [%n] %v");
        file_sink_->set_level(spdlog::level::debug);

        // Создаём logger с именем "DRVGPU" и file sink
        logger_ = std::make_shared<spdlog::logger>("DRVGPU", file_sink_);
        logger_->set_level(spdlog::level::debug);
        logger_->flush_on(spdlog::level::err);  // Flush при ошибках

        // Устанавливаем как default logger для spdlog
        spdlog::set_default_logger(logger_);

        initialized_ = true;

    } catch (const spdlog::spdlog_ex& e) {
        // spdlog exception - не удалось создать file sink
        // Не логируем вообще (может быть бесконечный цикл)
        (void)e;
        initialized_ = true;  // Помечаем как инициализированный, но без логера
    } catch (const std::exception& e) {
        // Любое другое исключение
        (void)e;
        initialized_ = true;  // Помечаем как инициализированный, но без логера
    }
}

/**
 * @brief Очистить и завершить работу spdlog
 * 
 * Операции:
 * 1. Flush всех pending сообщений
 * 2. Удалить все loggers из spdlog
 * 3. Сбросить smart pointers
 * 4. Установить initialized_ = false
 * 
 * @note Thread-safe через mutex.
 */
void DefaultLogger::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (logger_) {
        logger_->flush();  // Flush всех pending сообщений
        spdlog::drop_all();  // Удалить все loggers
        logger_.reset();
    }
    
    file_sink_.reset();
    initialized_ = false;
}

/**
 * @brief Логировать отладочное сообщение
 * @param component Имя компонента (например: "OpenCL", "Memory")
 * @param message Текст сообщения
 * 
 * Форматирует сообщение и вызывает spdlog::debug().
 */
void DefaultLogger::Debug(const std::string& component, const std::string& message) {
    if (!initialized_ || !logger_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    logger_->debug(formatted);
}

/**
 * @brief Логировать информационное сообщение
 * @param component Имя компонента
 * @param message Текст сообщения
 */
void DefaultLogger::Info(const std::string& component, const std::string& message) {
    if (!initialized_ || !logger_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    logger_->info(formatted);
}

/**
 * @brief Логировать предупреждение
 * @param component Имя компонента
 * @param message Текст сообщения
 */
void DefaultLogger::Warning(const std::string& component, const std::string& message) {
    if (!initialized_ || !logger_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    logger_->warn(formatted);
}

/**
 * @brief Логировать ошибку
 * @param component Имя компонента
 * @param message Текст сообщения
 */
void DefaultLogger::Error(const std::string& component, const std::string& message) {
    if (!initialized_ || !logger_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    logger_->error(formatted);
}

/**
 * @brief Проверить, активен ли уровень DEBUG
 * @return true если DEBUG активен
 */
bool DefaultLogger::IsDebugEnabled() const {
    return initialized_ && current_level_ <= spdlog::level::debug;
}

/**
 * @brief Проверить, активен ли уровень INFO
 * @return true если INFO активен
 */
bool DefaultLogger::IsInfoEnabled() const {
    return initialized_ && current_level_ <= spdlog::level::info;
}

/**
 * @brief Проверить, активен ли уровень WARNING
 * @return true если WARNING активен
 */
bool DefaultLogger::IsWarningEnabled() const {
    return initialized_ && current_level_ <= spdlog::level::warn;
}

/**
 * @brief Проверить, активен ли уровень ERROR
 * @return true если ERROR активен
 */
bool DefaultLogger::IsErrorEnabled() const {
    return initialized_ && current_level_ <= spdlog::level::err;
}

/**
 * @brief Сбросить состояние логера
 * 
 * Вызывает Shutdown() + Initialize() для переинициализации.
 */
void DefaultLogger::Reset() {
    Shutdown();
    Initialize();
}

/**
 * @brief Форматировать сообщение с компонентом
 * @param component Имя компонента
 * @param message Текст сообщения
 * @return Отформатированное сообщение "[component] message"
 */
std::string DefaultLogger::FormatMessage(const std::string& component, 
                                          const std::string& message) {
    return "[" + component + "] " + message;
}

/**
 * @brief Проверить, инициализирован ли логер
 * @return true если логер инициализирован
 */
bool DefaultLogger::IsInitialized() const {
    return initialized_;
}

} // namespace drv_gpu_lib
