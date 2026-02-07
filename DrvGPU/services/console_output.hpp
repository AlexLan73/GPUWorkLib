#pragma once

/**
 * @file console_output.hpp
 * @brief ConsoleOutput - Thread-safe singleton for console output from multiple GPUs
 *
 * ============================================================================
 * PROBLEM:
 *   When 8 GPUs write to stdout simultaneously, output becomes garbled.
 *   Messages from different GPUs interleave unpredictably.
 *
 * SOLUTION:
 *   ConsoleOutput is a singleton service with:
 *   - Dedicated background thread for all console output
 *   - Message queue (GPU threads only do Enqueue - almost instant)
 *   - Formatted output: [HH:MM:SS.mmm] [GPU_XX] [Module] message
 *   - Per-GPU enable/disable via configGPU.json (is_console flag)
 *
 * ARCHITECTURE:
 *   GPU Thread 0 --> Print(0, "FFT", "Done") --> Enqueue() --+
 *   GPU Thread 1 --> Print(1, "FFT", "Done") --> Enqueue() --+--> [Queue] --> Worker --> stdout
 *   GPU Thread N --> Print(N, "FFT", "Done") --> Enqueue() --+
 *
 * USAGE:
 *   ConsoleOutput::GetInstance().Start();
 *   ConsoleOutput::GetInstance().Print(0, "FFT", "Processing 1024 beams...");
 *   ConsoleOutput::GetInstance().PrintError(0, "FFT", "Failed to allocate!");
 *   ConsoleOutput::GetInstance().Stop();
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "async_service_base.hpp"

#include <string>
#include <chrono>
#include <cstdint>
#include <unordered_set>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace drv_gpu_lib {

// ============================================================================
// ConsoleMessage - Message type for console output queue
// ============================================================================

/**
 * @struct ConsoleMessage
 * @brief Single message for console output
 */
struct ConsoleMessage {
    /// GPU device index (-1 = system message, no GPU prefix)
    int gpu_id = -1;

    /// Source module name (e.g., "FFT", "MemManager", "Backend")
    std::string module_name;

    /// Message severity level
    enum class Level : uint8_t {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3
    };
    Level level = Level::INFO;

    /// Message text
    std::string message;

    /// Timestamp (auto-set on creation)
    std::chrono::system_clock::time_point timestamp = std::chrono::system_clock::now();
};

// ============================================================================
// ConsoleOutput - Thread-safe console output service
// ============================================================================

/**
 * @class ConsoleOutput
 * @brief Singleton service for thread-safe console output
 *
 * Inherits from AsyncServiceBase<ConsoleMessage>:
 * - Background worker thread
 * - Non-blocking Enqueue() for GPU threads
 * - Ordered, formatted output to stdout
 */
class ConsoleOutput : public AsyncServiceBase<ConsoleMessage> {
public:
    // ========================================================================
    // Singleton
    // ========================================================================

    /**
     * @brief Get the singleton instance
     */
    static ConsoleOutput& GetInstance() {
        static ConsoleOutput instance;
        return instance;
    }

    // Delete copy (singleton)
    ConsoleOutput(const ConsoleOutput&) = delete;
    ConsoleOutput& operator=(const ConsoleOutput&) = delete;

    // ========================================================================
    // Convenience API (non-blocking)
    // ========================================================================

    /**
     * @brief Print info message to console
     * @param gpu_id GPU device index (-1 for system messages)
     * @param module Source module name
     * @param message Message text
     */
    void Print(int gpu_id, const std::string& module, const std::string& message) {
        ConsoleMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.level = ConsoleMessage::Level::INFO;
        msg.message = message;
        Enqueue(std::move(msg));
    }

    /**
     * @brief Print warning message to console
     */
    void PrintWarning(int gpu_id, const std::string& module, const std::string& message) {
        ConsoleMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.level = ConsoleMessage::Level::WARNING;
        msg.message = message;
        Enqueue(std::move(msg));
    }

    /**
     * @brief Print error message to console
     */
    void PrintError(int gpu_id, const std::string& module, const std::string& message) {
        ConsoleMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.level = ConsoleMessage::Level::ERROR;
        msg.message = message;
        Enqueue(std::move(msg));
    }

    /**
     * @brief Print debug message to console
     */
    void PrintDebug(int gpu_id, const std::string& module, const std::string& message) {
        ConsoleMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.level = ConsoleMessage::Level::DEBUG;
        msg.message = message;
        Enqueue(std::move(msg));
    }

    /**
     * @brief Print system message (no GPU prefix)
     */
    void PrintSystem(const std::string& module, const std::string& message) {
        Print(-1, module, message);
    }

    // ========================================================================
    // Per-GPU Enable/Disable
    // ========================================================================

    /**
     * @brief Enable or disable console output globally
     */
    void SetEnabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_release);
    }

    /**
     * @brief Check if console output is globally enabled
     */
    bool IsEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * @brief Enable or disable console output for specific GPU
     * @param gpu_id GPU device index
     * @param enabled true to enable, false to disable
     */
    void SetGPUEnabled(int gpu_id, bool enabled) {
        std::lock_guard<std::mutex> lock(gpu_filter_mutex_);
        if (enabled) {
            disabled_gpus_.erase(gpu_id);
        } else {
            disabled_gpus_.insert(gpu_id);
        }
    }

    /**
     * @brief Check if a specific GPU's console output is enabled
     */
    bool IsGPUEnabled(int gpu_id) const {
        std::lock_guard<std::mutex> lock(gpu_filter_mutex_);
        return disabled_gpus_.find(gpu_id) == disabled_gpus_.end();
    }

protected:
    // ========================================================================
    // AsyncServiceBase implementation
    // ========================================================================

    /**
     * @brief Process one console message (runs in worker thread)
     *
     * Formats and outputs the message to stdout.
     * Format: [HH:MM:SS.mmm] [GPU_XX] [Module] message
     */
    void ProcessMessage(const ConsoleMessage& msg) override {
        // Check global enable
        if (!enabled_.load(std::memory_order_acquire)) {
            return;
        }

        // Check per-GPU enable
        if (msg.gpu_id >= 0) {
            std::lock_guard<std::mutex> lock(gpu_filter_mutex_);
            if (disabled_gpus_.find(msg.gpu_id) != disabled_gpus_.end()) {
                return;
            }
        }

        // Format timestamp: HH:MM:SS.mmm
        auto time_t = std::chrono::system_clock::to_time_t(msg.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            msg.timestamp.time_since_epoch()) % 1000;

        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t);
#else
        localtime_r(&time_t, &tm_buf);
#endif

        std::ostringstream oss;

        // Timestamp
        oss << "[" << std::setfill('0')
            << std::setw(2) << tm_buf.tm_hour << ":"
            << std::setw(2) << tm_buf.tm_min << ":"
            << std::setw(2) << tm_buf.tm_sec << "."
            << std::setw(3) << ms.count() << "] ";

        // Level prefix
        switch (msg.level) {
            case ConsoleMessage::Level::DEBUG:
                oss << "[DBG] ";
                break;
            case ConsoleMessage::Level::INFO:
                oss << "[INF] ";
                break;
            case ConsoleMessage::Level::WARNING:
                oss << "[WRN] ";
                break;
            case ConsoleMessage::Level::ERROR:
                oss << "[ERR] ";
                break;
        }

        // GPU prefix
        if (msg.gpu_id >= 0) {
            oss << "[GPU_" << std::setfill('0') << std::setw(2) << msg.gpu_id << "] ";
        } else {
            oss << "[SYSTEM] ";
        }

        // Module
        if (!msg.module_name.empty()) {
            oss << "[" << msg.module_name << "] ";
        }

        // Message
        oss << msg.message;

        // Output to stdout (or stderr for errors)
        if (msg.level == ConsoleMessage::Level::ERROR) {
            std::cerr << oss.str() << "\n";
        } else {
            std::cout << oss.str() << "\n";
        }
    }

    /**
     * @brief Service name for diagnostics
     */
    std::string GetServiceName() const override {
        return "ConsoleOutput";
    }

private:
    // ========================================================================
    // Private constructor (singleton)
    // ========================================================================

    ConsoleOutput() : enabled_(true) {}

    // ========================================================================
    // Private members
    // ========================================================================

    /// Global enable flag
    std::atomic<bool> enabled_;

    /// Set of disabled GPU IDs
    std::unordered_set<int> disabled_gpus_;

    /// Mutex for disabled_gpus_ set
    mutable std::mutex gpu_filter_mutex_;
};

} // namespace drv_gpu_lib
