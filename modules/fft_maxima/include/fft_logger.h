#pragma once

/**
 * @file fft_logger.h
 * @brief Simple logger for FFT module
 *
 * Separates logging from business logic.
 * Can be disabled, redirected, or replaced.
 *
 * @author DrvGPU Team
 * @date 2026-02-04
 */

#include <iostream>
#include <string>
#include <sstream>
#include <functional>

namespace antenna_fft {

/**
 * @class FFTLogger
 * @brief Simple static logger for FFT module
 *
 * Usage:
 * ```cpp
 * FFTLogger::Info("Processing started");
 * FFTLogger::Debug("  nFFT = ", nfft);
 * FFTLogger::Error("Failed to allocate buffer");
 *
 * // Disable logging:
 * FFTLogger::SetEnabled(false);
 *
 * // Custom output:
 * FFTLogger::SetOutputStream(&myStream);
 * ```
 */
class FFTLogger {
public:
    enum class Level {
        Debug,
        Info,
        Warning,
        Error
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Enable or disable logging
     */
    static void SetEnabled(bool enabled) { enabled_ = enabled; }

    /**
     * @brief Check if logging is enabled
     */
    static bool IsEnabled() { return enabled_; }

    /**
     * @brief Set minimum log level
     */
    static void SetLevel(Level level) { min_level_ = level; }

    /**
     * @brief Set output stream (default: std::cout)
     */
    static void SetOutputStream(std::ostream* stream) { output_ = stream; }

    /**
     * @brief Set custom log callback
     */
    static void SetCallback(std::function<void(Level, const std::string&)> callback) {
        callback_ = callback;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Logging methods
    // ═══════════════════════════════════════════════════════════════════════════

    template<typename... Args>
    static void Debug(Args&&... args) {
        Log(Level::Debug, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void Info(Args&&... args) {
        Log(Level::Info, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void Warning(Args&&... args) {
        Log(Level::Warning, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void Error(Args&&... args) {
        Log(Level::Error, std::forward<Args>(args)...);
    }

private:
    template<typename... Args>
    static void Log(Level level, Args&&... args) {
        if (!enabled_ || level < min_level_) return;

        std::ostringstream ss;
        (ss << ... << std::forward<Args>(args));
        std::string message = ss.str();

        if (callback_) {
            callback_(level, message);
        } else if (output_) {
            *output_ << GetPrefix(level) << message << "\n";
        }
    }

    static const char* GetPrefix(Level level) {
        switch (level) {
            case Level::Debug:   return "[DEBUG] ";
            case Level::Info:    return "[INFO] ";
            case Level::Warning: return "[WARN] ";
            case Level::Error:   return "[ERROR] ";
            default:             return "";
        }
    }

    static inline bool enabled_ = true;
    static inline Level min_level_ = Level::Info;
    static inline std::ostream* output_ = &std::cout;
    static inline std::function<void(Level, const std::string&)> callback_ = nullptr;
};

} // namespace antenna_fft
