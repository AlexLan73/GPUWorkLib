#pragma once

/**
 * @file gpu_profiler.hpp
 * @brief GPUProfiler - Async singleton for GPU profiling data collection
 *
 * ============================================================================
 * PURPOSE:
 *   Centralized collection and aggregation of GPU profiling data.
 *   Modules send profiling records (kernel times, memory ops, etc.)
 *   via non-blocking Enqueue. Background thread aggregates statistics.
 *
 * ARCHITECTURE:
 *   GPU Module --> Profiler::Record(gpu_id, "FFT", 12.5ms) --> Enqueue() --+
 *                                                                           |
 *                                                                    [Worker Thread]
 *                                                                           |
 *                                                              Aggregation (min/max/avg)
 *                                                              JSON export
 *                                                              Observer notification
 *
 * USAGE:
 *   GPUProfiler::GetInstance().Start();
 *
 *   // From any GPU thread (non-blocking):
 *   GPUProfiler::GetInstance().Record(0, "AntennaFFT", "FFT_Execute", 12.5);
 *   GPUProfiler::GetInstance().Record(0, "AntennaFFT", "Padding_Kernel", 0.8);
 *   GPUProfiler::GetInstance().Record(1, "VectorOps", "VectorAdd", 3.2);
 *
 *   // Get aggregated stats:
 *   auto stats = GPUProfiler::GetInstance().GetStats(0);
 *   auto all_stats = GPUProfiler::GetInstance().GetAllStats();
 *
 *   // Export to JSON:
 *   GPUProfiler::GetInstance().ExportJSON("./Results/Profiler/2026-02-07_14-30-00.json");
 *
 *   GPUProfiler::GetInstance().Stop();
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "async_service_base.hpp"

#include <string>
#include <chrono>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>

namespace drv_gpu_lib {

// ============================================================================
// ProfilingMessage - Message type for profiling queue
// ============================================================================

/**
 * @struct ProfilingMessage
 * @brief Single profiling record from a GPU module
 */
struct ProfilingMessage {
    /// GPU device index
    int gpu_id = 0;

    /// Source module name (e.g., "AntennaFFT", "VectorOps")
    std::string module_name;

    /// Event name (e.g., "FFT_Execute", "Padding_Kernel", "MemAlloc")
    std::string event_name;

    /// Duration in milliseconds
    double duration_ms = 0.0;

    /// Timestamp (auto-set on creation)
    std::chrono::system_clock::time_point timestamp = std::chrono::system_clock::now();
};

// ============================================================================
// EventStats - Aggregated statistics for one event type
// ============================================================================

/**
 * @struct EventStats
 * @brief Aggregated statistics for a specific event
 *
 * Tracks min/max/avg/total for an event like "FFT_Execute"
 */
struct EventStats {
    /// Event name
    std::string event_name;

    /// Total number of calls
    uint64_t total_calls = 0;

    /// Total accumulated time (ms)
    double total_time_ms = 0.0;

    /// Minimum duration (ms)
    double min_time_ms = std::numeric_limits<double>::max();

    /// Maximum duration (ms)
    double max_time_ms = 0.0;

    /// Average duration (ms) - computed on request
    double GetAvgTimeMs() const {
        return total_calls > 0 ? total_time_ms / static_cast<double>(total_calls) : 0.0;
    }

    /// Update with new measurement
    void Update(double duration_ms) {
        total_calls++;
        total_time_ms += duration_ms;
        min_time_ms = std::min(min_time_ms, duration_ms);
        max_time_ms = std::max(max_time_ms, duration_ms);
    }
};

// ============================================================================
// ModuleStats - Statistics for one module on one GPU
// ============================================================================

/**
 * @struct ModuleStats
 * @brief Aggregated statistics for a module on a specific GPU
 *
 * Contains per-event statistics for one module (e.g., "AntennaFFT" on GPU 0)
 */
struct ModuleStats {
    /// Module name
    std::string module_name;

    /// Per-event statistics: event_name -> EventStats
    std::map<std::string, EventStats> events;

    /// Total time across all events
    double GetTotalTimeMs() const {
        double total = 0.0;
        for (const auto& [name, stats] : events) {
            total += stats.total_time_ms;
        }
        return total;
    }

    /// Total calls across all events
    uint64_t GetTotalCalls() const {
        uint64_t total = 0;
        for (const auto& [name, stats] : events) {
            total += stats.total_calls;
        }
        return total;
    }
};

// ============================================================================
// GPUProfiler - Async profiling service
// ============================================================================

/**
 * @class GPUProfiler
 * @brief Singleton service for GPU profiling data collection
 *
 * Inherits from AsyncServiceBase<ProfilingMessage>:
 * - Background worker thread for aggregation
 * - Non-blocking Record() for GPU threads
 * - Thread-safe stats access via GetStats()
 */
class GPUProfiler : public AsyncServiceBase<ProfilingMessage> {
public:
    // ========================================================================
    // Singleton
    // ========================================================================

    /**
     * @brief Get the singleton instance
     */
    static GPUProfiler& GetInstance() {
        static GPUProfiler instance;
        return instance;
    }

    // Delete copy (singleton)
    GPUProfiler(const GPUProfiler&) = delete;
    GPUProfiler& operator=(const GPUProfiler&) = delete;

    // ========================================================================
    // Recording API (non-blocking)
    // ========================================================================

    /**
     * @brief Record a profiling event
     * @param gpu_id GPU device index
     * @param module Module name (e.g., "AntennaFFT")
     * @param event Event name (e.g., "FFT_Execute")
     * @param duration_ms Duration in milliseconds
     *
     * This is the PRIMARY API for GPU modules.
     * Non-blocking: only enqueues message to background thread.
     */
    void Record(int gpu_id, const std::string& module,
                const std::string& event, double duration_ms) {
        if (!enabled_.load(std::memory_order_acquire)) {
            return;
        }

        ProfilingMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.event_name = event;
        msg.duration_ms = duration_ms;
        Enqueue(std::move(msg));
    }

    // ========================================================================
    // Statistics Access (thread-safe reads)
    // ========================================================================

    /**
     * @brief Get statistics for a specific GPU
     * @param gpu_id GPU device index
     * @return Map of module_name -> ModuleStats
     */
    std::map<std::string, ModuleStats> GetStats(int gpu_id) const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto it = stats_.find(gpu_id);
        if (it != stats_.end()) {
            return it->second;
        }
        return {};
    }

    /**
     * @brief Get statistics for all GPUs
     * @return Map of gpu_id -> (module_name -> ModuleStats)
     */
    std::map<int, std::map<std::string, ModuleStats>> GetAllStats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }

    /**
     * @brief Reset all collected statistics
     */
    void Reset() {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.clear();
    }

    // ========================================================================
    // Enable/Disable
    // ========================================================================

    /**
     * @brief Enable or disable profiling globally
     */
    void SetEnabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_release);
    }

    /**
     * @brief Check if profiling is enabled
     */
    bool IsEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    // ========================================================================
    // Export
    // ========================================================================

    /**
     * @brief Export profiling data to JSON file
     * @param file_path Path to output JSON file
     * @return true if exported successfully
     *
     * Creates file in format:
     *   ${path}/Results/Profiler/YYYY-MM-DD_HH-MM-SS.json
     *
     * JSON structure:
     *   {
     *     "timestamp": "2026-02-07T14:30:00",
     *     "gpus": {
     *       "0": {
     *         "AntennaFFT": {
     *           "FFT_Execute": { "calls": 100, "total_ms": 1250.0, ... },
     *           "Padding_Kernel": { "calls": 100, "total_ms": 80.0, ... }
     *         }
     *       }
     *     }
     *   }
     */
    bool ExportJSON(const std::string& file_path) const {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        try {
            std::ofstream file(file_path);
            if (!file.is_open()) {
                std::cerr << "[GPUProfiler] Cannot create file: " << file_path << "\n";
                return false;
            }

            // Manually build JSON string (avoid dependency on nlohmann/json here)
            file << "{\n";

            // Timestamp
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf;
#ifdef _WIN32
            localtime_s(&tm_buf, &time_t);
#else
            localtime_r(&time_t, &tm_buf);
#endif
            file << "  \"timestamp\": \""
                 << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S") << "\",\n";

            // GPUs
            file << "  \"gpus\": {\n";
            bool first_gpu = true;
            for (const auto& [gpu_id, modules] : stats_) {
                if (!first_gpu) file << ",\n";
                first_gpu = false;

                file << "    \"" << gpu_id << "\": {\n";

                bool first_module = true;
                for (const auto& [mod_name, mod_stats] : modules) {
                    if (!first_module) file << ",\n";
                    first_module = false;

                    file << "      \"" << mod_name << "\": {\n";

                    bool first_event = true;
                    for (const auto& [evt_name, evt_stats] : mod_stats.events) {
                        if (!first_event) file << ",\n";
                        first_event = false;

                        file << "        \"" << evt_name << "\": {\n";
                        file << "          \"calls\": " << evt_stats.total_calls << ",\n";
                        file << "          \"total_ms\": " << std::fixed << std::setprecision(3)
                             << evt_stats.total_time_ms << ",\n";
                        file << "          \"avg_ms\": " << evt_stats.GetAvgTimeMs() << ",\n";
                        file << "          \"min_ms\": "
                             << (evt_stats.min_time_ms == std::numeric_limits<double>::max()
                                 ? 0.0 : evt_stats.min_time_ms) << ",\n";
                        file << "          \"max_ms\": " << evt_stats.max_time_ms << "\n";
                        file << "        }";
                    }
                    file << "\n      }";
                }
                file << "\n    }";
            }
            file << "\n  }\n";
            file << "}\n";

            file.close();
            std::cout << "[GPUProfiler] Exported to: " << file_path << "\n";
            return true;

        } catch (const std::exception& e) {
            std::cerr << "[GPUProfiler] Export error: " << e.what() << "\n";
            return false;
        }
    }

    /**
     * @brief Print profiling summary to stdout
     */
    void PrintSummary() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════╗\n";
        std::cout << "║              GPU Profiling Summary                  ║\n";
        std::cout << "╚══════════════════════════════════════════════════════╝\n";

        for (const auto& [gpu_id, modules] : stats_) {
            std::cout << "\n  GPU " << gpu_id << ":\n";

            for (const auto& [mod_name, mod_stats] : modules) {
                std::cout << "    Module: " << mod_name
                          << " (total: " << std::fixed << std::setprecision(1)
                          << mod_stats.GetTotalTimeMs() << " ms, "
                          << mod_stats.GetTotalCalls() << " calls)\n";

                for (const auto& [evt_name, evt_stats] : mod_stats.events) {
                    std::cout << "      " << std::left << std::setw(25) << evt_name
                              << " calls=" << std::setw(6) << evt_stats.total_calls
                              << " avg=" << std::setw(8) << std::fixed << std::setprecision(2)
                              << evt_stats.GetAvgTimeMs() << "ms"
                              << " min=" << std::setw(8)
                              << (evt_stats.min_time_ms == std::numeric_limits<double>::max()
                                  ? 0.0 : evt_stats.min_time_ms) << "ms"
                              << " max=" << std::setw(8) << evt_stats.max_time_ms << "ms\n";
                }
            }
        }
        std::cout << "\n";
    }

protected:
    // ========================================================================
    // AsyncServiceBase implementation
    // ========================================================================

    /**
     * @brief Process one profiling message (runs in worker thread)
     *
     * Updates aggregated statistics for the GPU/module/event.
     */
    void ProcessMessage(const ProfilingMessage& msg) override {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        // Get or create module stats for this GPU
        auto& module_stats = stats_[msg.gpu_id][msg.module_name];
        module_stats.module_name = msg.module_name;

        // Get or create event stats
        auto& event_stats = module_stats.events[msg.event_name];
        event_stats.event_name = msg.event_name;

        // Update with new measurement
        event_stats.Update(msg.duration_ms);
    }

    /**
     * @brief Service name for diagnostics
     */
    std::string GetServiceName() const override {
        return "GPUProfiler";
    }

private:
    // ========================================================================
    // Private constructor (singleton)
    // ========================================================================

    GPUProfiler() : enabled_(true) {}

    // ========================================================================
    // Private members
    // ========================================================================

    /// Aggregated statistics: gpu_id -> module_name -> ModuleStats
    std::map<int, std::map<std::string, ModuleStats>> stats_;

    /// Mutex for stats access
    mutable std::mutex stats_mutex_;

    /// Global enable flag
    std::atomic<bool> enabled_;
};

} // namespace drv_gpu_lib
