#include "config_logger.hpp"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <iostream>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// ConfigLogger Implementation
// ════════════════════════════════════════════════════════════════════════════

ConfigLogger& ConfigLogger::GetInstance() {
    static ConfigLogger instance;
    return instance;
}

ConfigLogger::ConfigLogger()
    : log_path_("")
    , enabled_(true) {
}

void ConfigLogger::SetLogPath(const std::string& path) {
    log_path_ = path;
}

std::string ConfigLogger::GetLogPath() const {
    return log_path_;
}

std::string ConfigLogger::GetLogFilePath() const {
    // Получаем текущее время
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm;
    
#if defined(_WIN32)
    localtime_s(&now_tm, &now_time);
#else
    localtime_r(&now_time, &now_tm);
#endif

    // Формируем строку с датой: YYYY-MM-DD
    std::ostringstream date_ss;
    date_ss << std::put_time(&now_tm, "%Y-%m-%d");
    std::string date_str = date_ss.str();

    // Формируем строку со временем: HH-MM-SS
    std::ostringstream time_ss;
    time_ss << std::put_time(&now_tm, "%H-%M-%S");
    std::string time_str = time_ss.str();

    // Строим полный путь
    std::string base_path = log_path_;
    if (base_path.empty()) {
        base_path = std::filesystem::current_path().string();
    }

    // Формируем: {base_path}/Logs/DRVGPU/{YYYY-MM-DD}/{HH-MM-SS}.log
    std::ostringstream path_ss;
    path_ss << base_path;
    
    // Добавляем разделитель
    if (!base_path.empty() && base_path.back() != '/' && base_path.back() != '\\') {
        path_ss << std::filesystem::path::preferred_separator;
    }
    
    path_ss << kLogsDir << std::filesystem::path::preferred_separator;
    path_ss << kLogSubdir << std::filesystem::path::preferred_separator;
    path_ss << date_str << std::filesystem::path::preferred_separator;
    path_ss << time_str << ".log";

    return path_ss.str();
}

void ConfigLogger::SetEnabled(bool enabled) {
    enabled_ = enabled;
}

bool ConfigLogger::IsEnabled() const {
    return enabled_;
}

void ConfigLogger::Enable() {
    enabled_ = true;
}

void ConfigLogger::Disable() {
    enabled_ = false;
}

bool ConfigLogger::CreateLogDirectory() const {
    std::string file_path = GetLogFilePath();
    
    // Извлекаем директорию из полного пути к файлу
    std::filesystem::path log_file_path(file_path);
    std::filesystem::path log_dir = log_file_path.parent_path();

    try {
        // Создаём директорию (и все родительские), если не существует
        if (!std::filesystem::exists(log_dir)) {
            std::filesystem::create_directories(log_dir);
        }
        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[ConfigLogger] Failed to create log directory: " << e.what() << "\n";
        return false;
    }
}

void ConfigLogger::Reset() {
    log_path_ = "";
    enabled_ = true;
}

} // namespace drv_gpu_lib
