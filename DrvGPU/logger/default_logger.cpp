#include "default_logger.hpp"
#include <iostream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// DefaultLogger Implementation - Реализация файлового логирования на plog
// ════════════════════════════════════════════════════════════════════════════
// ЗАМЕНА: spdlog → plog (2026-02-07)
// Причина: spdlog был капризным при кросс-платформенной сборке (fmt зависимость,
//          LNK2019 errors, SPDLOG_HEADER_ONLY / FMT_HEADER_ONLY hacks).
// plog: header-only, zero dependencies, стабильный.
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
 * - Вызывает Initialize() для настройки plog
 */
DefaultLogger::DefaultLogger()
    : initialized_(false)
    , current_level_(plog::debug) {
    Initialize();
}

/**
 * @brief Деструктор DefaultLogger
 *
 * Вызывает Shutdown() для корректного завершения.
 */
DefaultLogger::~DefaultLogger() {
    Shutdown();
}

/**
 * @brief Инициализировать plog file logger
 *
 * Логика инициализации:
 * 1. Проверяем ConfigLogger::IsEnabled()
 * 2. Если отключено - помечаем как инициализированный без логера
 * 3. Если включено:
 *    - Создаём директорию для логов
 *    - Получаем путь к файлу лога
 *    - Инициализируем plog с RollingFileAppender
 *    - Устанавливаем уровень логирования
 *
 * plog::init() можно вызвать только один раз!
 * Для нескольких логеров (GPU) используются разные instance IDs.
 *
 * @note Если CreateLogDirectory() не удалось, логер помечается как
 *       инициализированный, но без функциональности.
 */
void DefaultLogger::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        return;
    }

    // Проверяем, включено ли логирование
    if (!ConfigLogger::GetInstance().IsEnabled()) {
        initialized_ = true;  // Помечаем как "инициализированный", но без логера
        return;
    }

    try {
        // Создаём директорию для логов
        ConfigLogger::GetInstance().CreateLogDirectory();

        // Получаем путь к файлу лога (с датой и временем)
        std::string log_file_path = ConfigLogger::GetInstance().GetLogFilePath();

        // ═══════════════════════════════════════════════════════════════
        // Инициализация plog
        // ═══════════════════════════════════════════════════════════════
        // plog::init(severity, filename, maxFileSize, maxFiles)
        //   - severity: минимальный уровень логирования
        //   - filename: путь к файлу лога
        //   - maxFileSize: максимальный размер файла (5 MB по умолчанию)
        //   - maxFiles: количество файлов для ротации (3 по умолчанию)
        //
        // ВАЖНО: plog::init() вызывается один раз на весь процесс.
        // Для multi-GPU используем разные instance IDs (будущее).
        // ═══════════════════════════════════════════════════════════════

        static const size_t kMaxFileSize = 5 * 1024 * 1024;  // 5 MB
        static const int    kMaxFiles    = 3;                  // 3 файла ротации

        plog::init(plog::debug, log_file_path.c_str(), kMaxFileSize, kMaxFiles);

        initialized_ = true;

    } catch (const std::exception& e) {
        // Любое исключение при инициализации
        (void)e;
        initialized_ = true;  // Помечаем как инициализированный, но без логера
    }
}

/**
 * @brief Очистить и завершить работу plog
 *
 * plog не требует явного shutdown — ресурсы освобождаются автоматически.
 * Метод оставлен для совместимости с интерфейсом.
 */
void DefaultLogger::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_ = false;
}

/**
 * @brief Логировать отладочное сообщение
 * @param component Имя компонента (например: "OpenCL", "Memory")
 * @param message Текст сообщения
 */
void DefaultLogger::Debug(const std::string& component, const std::string& message) {
    if (!initialized_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    PLOG_DEBUG << formatted;
}

/**
 * @brief Логировать информационное сообщение
 * @param component Имя компонента
 * @param message Текст сообщения
 */
void DefaultLogger::Info(const std::string& component, const std::string& message) {
    if (!initialized_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    PLOG_INFO << formatted;
}

/**
 * @brief Логировать предупреждение
 * @param component Имя компонента
 * @param message Текст сообщения
 */
void DefaultLogger::Warning(const std::string& component, const std::string& message) {
    if (!initialized_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    PLOG_WARNING << formatted;
}

/**
 * @brief Логировать ошибку
 * @param component Имя компонента
 * @param message Текст сообщения
 */
void DefaultLogger::Error(const std::string& component, const std::string& message) {
    if (!initialized_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    PLOG_ERROR << formatted;
}

/**
 * @brief Проверить, активен ли уровень DEBUG
 * @return true если DEBUG активен
 */
bool DefaultLogger::IsDebugEnabled() const {
    return initialized_ && current_level_ >= plog::debug;
}

/**
 * @brief Проверить, активен ли уровень INFO
 * @return true если INFO активен
 */
bool DefaultLogger::IsInfoEnabled() const {
    return initialized_ && current_level_ >= plog::info;
}

/**
 * @brief Проверить, активен ли уровень WARNING
 * @return true если WARNING активен
 */
bool DefaultLogger::IsWarningEnabled() const {
    return initialized_ && current_level_ >= plog::warning;
}

/**
 * @brief Проверить, активен ли уровень ERROR
 * @return true если ERROR активен
 */
bool DefaultLogger::IsErrorEnabled() const {
    return initialized_ && current_level_ >= plog::error;
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
