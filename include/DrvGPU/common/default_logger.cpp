#include "default_logger.hpp"
#include <iostream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// DefaultLogger Implementation
// ════════════════════════════════════════════════════════════════════════════

DefaultLogger& DefaultLogger::GetInstance() {
    static DefaultLogger instance;
    return instance;
}

DefaultLogger::DefaultLogger()
    : initialized_(false)
    , current_level_(spdlog::level::debug) {
    Initialize();
}

DefaultLogger::~DefaultLogger() {
    Shutdown();
}

void DefaultLogger::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return;
    }

    // Проверяем, включено ли логирование
    // Только если IsEnabled() == true - пишем в файл
    if (!ConfigLogger::GetInstance().IsEnabled()) {
        initialized_ = true;  // Помечаем как "инициализированный", но без логера
        return;
    }

    try {
        // Создаём директорию для логов
        ConfigLogger::GetInstance().CreateLogDirectory();

        // Получаем путь к файлу лога
        std::string log_file_path = ConfigLogger::GetInstance().GetLogFilePath();

        // Создаём только file sink (без console sink для Release)
        file_sink_ = std::make_shared<spdlog::sinks::basic_file_sink_st>(log_file_path, true);
        file_sink_->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] [%n] %v");
        file_sink_->set_level(spdlog::level::debug);

        // Создаём logger с file sink
        logger_ = std::make_shared<spdlog::logger>("DRVGPU", file_sink_);
        logger_->set_level(spdlog::level::debug);
        logger_->flush_on(spdlog::level::err);

        // Устанавливаем как default logger
        spdlog::set_default_logger(logger_);

        initialized_ = true;

    } catch (const spdlog::spdlog_ex& e) {
        // Не удалось создать file sink - не логируем вообще
        (void)e;
        initialized_ = true;  // Помечаем как инициализированный, но без логера
    } catch (const std::exception& e) {
        // Не удалось создать file sink - не логируем вообще
        (void)e;
        initialized_ = true;  // Помечаем как инициализированный, но без логера
    }
}

void DefaultLogger::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (logger_) {
        logger_->flush();
        spdlog::drop_all();
        logger_.reset();
    }
    
    file_sink_.reset();
    initialized_ = false;
}

void DefaultLogger::Debug(const std::string& component, const std::string& message) {
    if (!initialized_ || !logger_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    logger_->debug(formatted);
}

void DefaultLogger::Info(const std::string& component, const std::string& message) {
    if (!initialized_ || !logger_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    logger_->info(formatted);
}

void DefaultLogger::Warning(const std::string& component, const std::string& message) {
    if (!initialized_ || !logger_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    logger_->warn(formatted);
}

void DefaultLogger::Error(const std::string& component, const std::string& message) {
    if (!initialized_ || !logger_) {
        return;
    }
    std::string formatted = FormatMessage(component, message);
    logger_->error(formatted);
}

bool DefaultLogger::IsDebugEnabled() const {
    return initialized_ && current_level_ <= spdlog::level::debug;
}

bool DefaultLogger::IsInfoEnabled() const {
    return initialized_ && current_level_ <= spdlog::level::info;
}

bool DefaultLogger::IsWarningEnabled() const {
    return initialized_ && current_level_ <= spdlog::level::warn;
}

bool DefaultLogger::IsErrorEnabled() const {
    return initialized_ && current_level_ <= spdlog::level::err;
}

void DefaultLogger::Reset() {
    Shutdown();
    Initialize();
}

std::string DefaultLogger::FormatMessage(const std::string& component, 
                                          const std::string& message) {
    return "[" + component + "] " + message;
}

bool DefaultLogger::IsInitialized() const {
    return initialized_;
}

} // namespace drv_gpu_lib
