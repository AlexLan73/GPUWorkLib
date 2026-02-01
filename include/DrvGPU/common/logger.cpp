#include "logger.hpp"

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Logger Implementation
// ════════════════════════════════════════════════════════════════════════════

ILoggerPtr Logger::current_logger_ = nullptr;

ILogger& Logger::GetInstance() {
    if (!current_logger_) {
        current_logger_ = std::make_shared<DefaultLogger>();
    }
    return *current_logger_;
}

void Logger::SetInstance(ILoggerPtr logger) {
    current_logger_ = logger;
}

void Logger::ResetToDefault() {
    current_logger_ = std::make_shared<DefaultLogger>();
}

void Logger::Debug(const std::string& component, const std::string& message) {
    GetInstance().Debug(component, message);
}

void Logger::Info(const std::string& component, const std::string& message) {
    GetInstance().Info(component, message);
}

void Logger::Warning(const std::string& component, const std::string& message) {
    GetInstance().Warning(component, message);
}

void Logger::Error(const std::string& component, const std::string& message) {
    GetInstance().Error(component, message);
}

bool Logger::IsEnabled() {
    return ConfigLogger::GetInstance().IsEnabled();
}

void Logger::Enable() {
    ConfigLogger::GetInstance().Enable();
}

void Logger::Disable() {
    ConfigLogger::GetInstance().Disable();
}

} // namespace drv_gpu_lib
