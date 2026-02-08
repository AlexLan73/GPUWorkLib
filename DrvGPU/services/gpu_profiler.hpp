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
#include "profiling_types.hpp"
#include "profiling_stats.hpp"

#include <string>
#include <chrono>
#include <map>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace drv_gpu_lib {

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

    /**
     * @brief Получить текущую дату и время в формате "YYYY-MM-DD HH:MM:SS"
     */
    static std::string GetCurrentDateTimeString() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t);
#else
        localtime_r(&time_t, &tm_buf);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    // Delete copy (singleton)
    GPUProfiler(const GPUProfiler&) = delete;
    GPUProfiler& operator=(const GPUProfiler&) = delete;

    // ========================================================================
    // Recording API (non-blocking)
    // ========================================================================

    /**
     * @brief Record a profiling event (OpenCL: 5 параметров cl_profiling_info)
     * Конвертация в duration_ms выполняется в воркере.
     */
    void Record(int gpu_id, const std::string& module,
                const std::string& event, const OpenCLProfilingData& data) {
        if (!enabled_.load(std::memory_order_acquire)) return;
        ProfilingMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.event_name = event;
        msg.time_ = data;
        Enqueue(std::move(msg));
    }

    /**
     * @brief Record a profiling event (ROCm/HIP: база + доп. параметры)
     */
    void Record(int gpu_id, const std::string& module,
                const std::string& event, const ROCmProfilingData& data) {
        if (!enabled_.load(std::memory_order_acquire)) return;
        ProfilingMessage msg;
        msg.gpu_id = gpu_id;
        msg.module_name = module;
        msg.event_name = event;
        msg.time_ = data;
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
     * @brief Check if profiling is enabled globally
     */
    bool IsEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * @brief Enable or disable profiling for a specific GPU (from config: is_prof)
     * @param gpu_id GPU device index
     * @param enabled true — записывать профиль для этого GPU, false — отключить
     */
    void SetGPUEnabled(int gpu_id, bool enabled) {
        std::lock_guard<std::mutex> lock(profile_filter_mutex_);
        if (enabled) {
            disabled_gpus_.erase(gpu_id);
        } else {
            disabled_gpus_.insert(gpu_id);
        }
    }

    /**
     * @brief Check if profiling is enabled for this GPU
     */
    bool IsGPUEnabled(int gpu_id) const {
        std::lock_guard<std::mutex> lock(profile_filter_mutex_);
        return disabled_gpus_.find(gpu_id) == disabled_gpus_.end();
    }

    // ========================================================================
    // GPU Info (для шапки отчёта)
    // ========================================================================

    /**
     * @brief Установить информацию о GPU для отчёта
     * @param gpu_id GPU device index
     * @param info Структура с информацией о GPU
     */
    void SetGPUInfo(int gpu_id, const GPUReportInfo& info) {
        std::lock_guard<std::mutex> lock(gpu_info_mutex_);
        gpu_info_[gpu_id] = info;
    }

    /**
     * @brief Получить информацию о GPU
     * @param gpu_id GPU device index
     * @return GPUReportInfo или пустая структура если не задано
     */
    GPUReportInfo GetGPUInfo(int gpu_id) const {
        std::lock_guard<std::mutex> lock(gpu_info_mutex_);
        auto it = gpu_info_.find(gpu_id);
        if (it != gpu_info_.end()) {
            return it->second;
        }
        return GPUReportInfo{};
    }

    /**
     * @brief Получить информацию обо всех GPU
     */
    std::map<int, GPUReportInfo> GetAllGPUInfo() const {
        std::lock_guard<std::mutex> lock(gpu_info_mutex_);
        return gpu_info_;
    }

    // ========================================================================
    // ROCm Detection
    // ========================================================================

    /**
     * @brief Проверить, есть ли ROCm данные для указанной GPU
     */
    bool HasAnyROCmData(int gpu_id) const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto it = stats_.find(gpu_id);
        if (it == stats_.end()) return false;
        for (const auto& [mod, mstats] : it->second) {
            for (const auto& [evt, estats] : mstats.events) {
                if (estats.has_rocm_data) return true;
            }
        }
        return false;
    }

    /**
     * @brief Проверить, есть ли ROCm данные для ЛЮБОЙ GPU (внутренняя, без блокировки)
     * @note Вызывать ТОЛЬКО когда stats_mutex_ уже захвачен!
     */
    bool HasAnyROCmDataGlobal_NoLock() const {
        for (const auto& [gpu_id, modules] : stats_) {
            for (const auto& [mod, mstats] : modules) {
                for (const auto& [evt, estats] : mstats.events) {
                    if (estats.has_rocm_data) return true;
                }
            }
        }
        return false;
    }

    /**
     * @brief Проверить, есть ли ROCm данные в модуле (внутренняя, без блокировки)
     * @param mod_stats Статистика модуля
     * @return true если хотя бы одно событие имеет ROCm данные
     */
    static bool HasModuleROCmData(const ModuleStats& mod_stats) {
        for (const auto& [evt_name, evt_stats] : mod_stats.events) {
            if (evt_stats.has_rocm_data) return true;
        }
        return false;
    }

    /**
     * @brief Проверить, есть ли ROCm данные для ЛЮБОЙ GPU (публичная)
     */
    bool HasAnyROCmDataGlobal() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return HasAnyROCmDataGlobal_NoLock();
    }

    // ========================================================================
    // Export
    // ========================================================================

    /**
     * @brief Export profiling data to JSON file
     * @param file_path Path to output JSON file
     * @return true if exported successfully
     */
    bool ExportJSON(const std::string& file_path) const {
        std::lock_guard<std::mutex> lock(stats_mutex_);

        try {
            std::ofstream file(file_path);
            if (!file.is_open()) {
                std::cerr << "[GPUProfiler] Cannot create file: " << file_path << "\n";
                return false;
            }

            // Manually build JSON string
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

    /**
     * @brief Print beautiful profiling report with GPU info header
     *
     * Таблица с ВСЕМИ 5 полями времени ProfilingDataBase:
     * Очередь, Отправка, Старт, Конец, Готово
     */
    void PrintReport() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        std::lock_guard<std::mutex> lock2(gpu_info_mutex_);

        const int W = 110;

        auto pad = [](const std::string& s, int width) -> std::string {
            if (static_cast<int>(s.size()) >= width) return s.substr(0, width);
            return s + std::string(width - s.size(), ' ');
        };

        auto fmtD = [](double val, int prec = 3) -> std::string {
            if (val == 0.0) return "-";
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(prec) << val;
            return oss.str();
        };

        std::cout << "\n";
        std::cout << "+" << std::string(W - 2, '=') << "+\n";
        std::cout << "|" << pad("              ОТЧЁТ ПРОФИЛИРОВАНИЯ GPU", W - 2) << "|\n";
        std::cout << "|" << pad("  Дата: " + GetCurrentDateTimeString(), W - 2) << "|\n";
        std::cout << "+" << std::string(W - 2, '=') << "+\n";

        for (const auto& [gpu_id, modules] : stats_) {
            GPUReportInfo info;
            auto it = gpu_info_.find(gpu_id);
            if (it != gpu_info_.end()) {
                info = it->second;
            }

            // GPU Header
            std::cout << "|" << pad("  GPU " + std::to_string(gpu_id) + ": " +
                (info.gpu_name.empty() ? "Unknown" : info.gpu_name), W - 2) << "|\n";
            if (info.global_mem_mb > 0) {
                std::cout << "|" << pad("  Память: " + std::to_string(info.global_mem_mb) + " MB", W - 2) << "|\n";
            }

            // Драйверы из vector<map>
            std::cout << "|" << pad("  Драйверы:", W - 2) << "|\n";
            for (const auto& drv : info.drivers) {
                std::string drv_line = "    ";
                auto it_type = drv.find("driver_type");
                if (it_type != drv.end()) {
                    drv_line += "[" + it_type->second + "] ";
                    auto it_ver = drv.find("version");
                    if (it_ver != drv.end()) drv_line += "Версия: " + it_ver->second + " | ";
                    auto it_drv = drv.find("driver_version");
                    if (it_drv != drv.end()) drv_line += "Драйвер: " + it_drv->second;
                    if (it_type->second == "ROCm") {
                        auto it_hip = drv.find("hip_version");
                        if (it_hip != drv.end()) drv_line += " | HIP: " + it_hip->second;
                    } else {
                        auto it_plat = drv.find("platform_name");
                        if (it_plat != drv.end()) {
                            std::string plat = it_plat->second;
                            if (plat.length() > 25) plat = plat.substr(0, 25) + "...";
                            drv_line += " | Платформа: " + plat;
                        }
                    }
                }
                std::cout << "|" << pad(drv_line, W - 2) << "|\n";
            }
            if (info.drivers.empty()) {
                std::cout << "|" << pad("    [нет информации о драйверах]", W - 2) << "|\n";
            }
            std::cout << "+" << std::string(W - 2, '-') << "+\n";

            // Data rows - разделяем по типу таблицы (OpenCL vs ROCm)
            for (const auto& [mod_name, mod_stats] : modules) {
                bool is_rocm_module = HasModuleROCmData(mod_stats);

                if (is_rocm_module) {
                    // ========== ROCm ТАБЛИЦА (расширенная) ==========
                    std::cout << "| [ROCm] Модуль: " << pad(mod_name, W - 19) << "|\n";
                    std::cout << "+" << std::string(W - 2, '-') << "+\n";

                    // ROCm Header - расширенный
                    std::cout << "| " << std::left << std::setw(16) << "Событие"
                              << "| " << std::setw(5) << "N"
                              << "| " << std::setw(6) << "Домен"
                              << "| " << std::setw(5) << "Тип"
                              << "| " << std::setw(5) << "Опер"
                              << "| " << std::setw(7) << "УстрID"
                              << "| " << std::setw(8) << "ОчерID"
                              << "| " << std::setw(10) << "Байты(MB)"
                              << "| " << std::setw(10) << "Старт(мс)"
                              << "| " << std::setw(10) << "Конец(мс)"
                              << "| " << std::setw(10) << "Время(мс)"
                              << "|\n";
                    std::cout << "+" << std::string(17, '-')
                              << "+" << std::string(6, '-')
                              << "+" << std::string(7, '-')
                              << "+" << std::string(6, '-')
                              << "+" << std::string(6, '-')
                              << "+" << std::string(8, '-')
                              << "+" << std::string(9, '-')
                              << "+" << std::string(11, '-')
                              << "+" << std::string(11, '-')
                              << "+" << std::string(11, '-')
                              << "+" << std::string(11, '-')
                              << "+\n";

                    for (const auto& [evt_name, evt_stats] : mod_stats.events) {
                        // Основная строка ROCm
                        std::cout << "| " << std::left << std::setw(16) << evt_name.substr(0, 15)
                                  << "| " << std::right << std::setw(4) << evt_stats.total_calls << " "
                                  << "| " << std::setw(5) << evt_stats.last_domain << " "
                                  << "| " << std::setw(4) << evt_stats.last_kind << " "
                                  << "| " << std::setw(4) << evt_stats.last_op << " "
                                  << "| " << std::setw(6) << evt_stats.last_device_id << " "
                                  << "| " << std::setw(7) << evt_stats.last_queue_id << " "
                                  << "| " << std::setw(9) << (evt_stats.total_bytes / (1024*1024)) << " "
                                  << "| " << std::setw(9) << fmtD(evt_stats.start.GetAvgMs()) << " "
                                  << "| " << std::setw(9) << fmtD(evt_stats.end.GetAvgMs()) << " "
                                  << "| " << std::setw(9) << fmtD(evt_stats.exec_time.GetAvgMs()) << " "
                                  << "|\n";

                        // Имя кернела (если есть)
                        if (!evt_stats.last_kernel_name.empty()) {
                            std::cout << "|   Ядро: " << pad(evt_stats.last_kernel_name, W - 12) << "|\n";
                        }
                        // Строка операции (если есть)
                        if (!evt_stats.last_op_string.empty()) {
                            std::cout << "|   Опер: " << pad(evt_stats.last_op_string, W - 12) << "|\n";
                        }
                    }

                    // Module subtotal ROCm
                    std::cout << "| " << std::left << std::setw(16) << "--- ИТОГО ---"
                              << "| " << std::right << std::setw(4) << mod_stats.GetTotalCalls() << " "
                              << "| " << std::string(5, ' ') << " "
                              << "| " << std::string(4, ' ') << " "
                              << "| " << std::string(4, ' ') << " "
                              << "| " << std::string(6, ' ') << " "
                              << "| " << std::string(7, ' ') << " "
                              << "| " << std::string(9, ' ') << " "
                              << "| " << std::string(9, ' ') << " "
                              << "| " << std::string(9, ' ') << " "
                              << "| " << std::setw(9) << fmtD(mod_stats.GetTotalTimeMs(), 2) << " "
                              << "|\n";
                    std::cout << "+" << std::string(W - 2, '-') << "+\n";

                } else {
                    // ========== OpenCL ТАБЛИЦА (стандартная, 5 полей) ==========
                    std::cout << "| [OpenCL] Модуль: " << pad(mod_name, W - 21) << "|\n";
                    std::cout << "+" << std::string(W - 2, '-') << "+\n";

                    // OpenCL Header
                    std::cout << "| " << std::left << std::setw(16) << "Событие"
                              << "| " << std::setw(5) << "N"
                              << "| " << std::setw(12) << "Очередь"
                              << "| " << std::setw(12) << "Отправка"
                              << "| " << std::setw(12) << "Старт"
                              << "| " << std::setw(12) << "Конец"
                              << "| " << std::setw(12) << "Готово"
                              << "| " << std::setw(12) << "Время"
                              << "|\n";
                    std::cout << "+" << std::string(17, '-')
                              << "+" << std::string(6, '-')
                              << "+" << std::string(13, '-')
                              << "+" << std::string(13, '-')
                              << "+" << std::string(13, '-')
                              << "+" << std::string(13, '-')
                              << "+" << std::string(13, '-')
                              << "+" << std::string(13, '-')
                              << "+\n";

                    for (const auto& [evt_name, evt_stats] : mod_stats.events) {
                        // Основная строка OpenCL: 5 полей времени + время выполнения
                        std::cout << "| " << std::left << std::setw(16) << evt_name.substr(0, 15)
                                  << "| " << std::right << std::setw(4) << evt_stats.total_calls << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.queued.GetAvgMs()) << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.submit.GetAvgMs()) << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.start.GetAvgMs()) << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.end.GetAvgMs()) << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.complete.GetAvgMs()) << " "
                                  << "| " << std::setw(11) << fmtD(evt_stats.exec_time.GetAvgMs()) << " "
                                  << "|\n";
                    }

                    // Module subtotal OpenCL
                    std::cout << "| " << std::left << std::setw(16) << "--- ИТОГО ---"
                              << "| " << std::right << std::setw(4) << mod_stats.GetTotalCalls() << " "
                              << "| " << std::string(11, ' ') << " "
                              << "| " << std::string(11, ' ') << " "
                              << "| " << std::string(11, ' ') << " "
                              << "| " << std::string(11, ' ') << " "
                              << "| " << std::string(11, ' ') << " "
                              << "| " << std::setw(11) << fmtD(mod_stats.GetTotalTimeMs(), 2) << " "
                              << "|\n";
                    std::cout << "+" << std::string(W - 2, '-') << "+\n";
                }
            }

            std::cout << "+" << std::string(W - 2, '=') << "+\n";
        }

        PrintLegend();
    }

    /**
     * @brief Печать легенды (расшифровка колонок)
     */
    void PrintLegend() const {
        std::cout << "\n";
        std::cout << "+--- ЛЕГЕНДА ---+\n";
        std::cout << "| Время в миллисекундах (мс), усреднённое значение                           |\n";
        std::cout << "+---------------+------------------------------------------------------------+\n";
        std::cout << "| Очередь       | Команда попала в очередь хоста (queued_ns)                 |\n";
        std::cout << "| Отправка      | Команда отправлена на GPU (submit_ns)                      |\n";
        std::cout << "| Старт         | Кернел начал выполняться (start_ns)                        |\n";
        std::cout << "| Конец         | Кернел закончил выполняться (end_ns)                       |\n";
        std::cout << "| Готово        | Данные выгружены/доступны (complete_ns)                    |\n";
        std::cout << "+---------------+------------------------------------------------------------+\n";

        // ROCm легенда (показываем если есть ROCm данные)
        // Используем _NoLock версию, т.к. PrintReport() уже держит stats_mutex_
        if (HasAnyROCmDataGlobal_NoLock()) {
            std::cout << "| [ROCm поля]                                                                |\n";
            std::cout << "+---------------+------------------------------------------------------------+\n";
            std::cout << "| Домен         | Область профилирования (0=HIP API, 1=Activity, 2=HSA)      |\n";
            std::cout << "| Тип           | Тип операции (0=kernel, 1=copy, 2=barrier, 3=marker)       |\n";
            std::cout << "| Опер          | Код HIP операции                                           |\n";
            std::cout << "| КоррID        | Correlation ID - связь API вызова и выполнения             |\n";
            std::cout << "| УстрID        | ID устройства GPU                                          |\n";
            std::cout << "| ОчерID        | ID очереди/потока (stream)                                 |\n";
            std::cout << "| Байты         | Объём переданных данных                                    |\n";
            std::cout << "| Ядро          | Имя кернела                                                |\n";
            std::cout << "+---------------+------------------------------------------------------------+\n";
        }
    }

    /**
     * @brief Export profiling report to Markdown file
     * @param file_path Path to output .md file
     * @return true if exported successfully
     */
    bool ExportMarkdown(const std::string& file_path) const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        std::lock_guard<std::mutex> lock2(gpu_info_mutex_);

        auto fmtD = [](double val, int prec = 3) -> std::string {
            if (val == 0.0) return "-";
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(prec) << val;
            return oss.str();
        };

        try {
            std::ofstream file(file_path);
            if (!file.is_open()) {
                std::cerr << "[GPUProfiler] Cannot create MD file: " << file_path << "\n";
                return false;
            }

            file << "# Отчёт профилирования GPU\n\n";
            file << "**Дата:** " << GetCurrentDateTimeString() << "\n\n";

            for (const auto& [gpu_id, modules] : stats_) {
                GPUReportInfo info;
                auto it = gpu_info_.find(gpu_id);
                if (it != gpu_info_.end()) {
                    info = it->second;
                }

                file << "## GPU " << gpu_id << "\n\n";
                file << "### Информация о системе\n\n";
                file << "| Параметр | Значение |\n";
                file << "|----------|----------|\n";
                file << "| **GPU** | " << (info.gpu_name.empty() ? "Unknown" : info.gpu_name) << " |\n";
                if (info.global_mem_mb > 0)
                    file << "| **Память** | " << info.global_mem_mb << " MB |\n";

                // Драйверы
                for (size_t i = 0; i < info.drivers.size(); ++i) {
                    const auto& drv = info.drivers[i];
                    auto it_type = drv.find("driver_type");
                    if (it_type != drv.end()) {
                        std::string drv_info = "[" + it_type->second + "] ";
                        auto it_ver = drv.find("version");
                        if (it_ver != drv.end()) drv_info += "Версия: " + it_ver->second;
                        auto it_drv = drv.find("driver_version");
                        if (it_drv != drv.end()) drv_info += " \\| Драйвер: " + it_drv->second;
                        if (it_type->second == "ROCm") {
                            auto it_hip = drv.find("hip_version");
                            if (it_hip != drv.end()) drv_info += " \\| HIP: " + it_hip->second;
                        } else {
                            auto it_plat = drv.find("platform_name");
                            if (it_plat != drv.end()) drv_info += " \\| Платформа: " + it_plat->second;
                        }
                        file << "| **Драйвер " << (i + 1) << "** | " << drv_info << " |\n";
                    }
                }
                file << "\n";

                // Table
                file << "### Результаты профилирования\n\n";
                file << "| Модуль | Событие | N | Очередь | Отправка | Старт | Конец | Готово |\n";
                file << "|--------|---------|--:|--------:|---------:|------:|------:|-------:|\n";

                for (const auto& [mod_name, mod_stats] : modules) {
                    bool first_event = true;
                    for (const auto& [evt_name, evt_stats] : mod_stats.events) {
                        file << "| " << (first_event ? mod_name : "")
                             << " | " << evt_name
                             << " | " << evt_stats.total_calls
                             << " | " << fmtD(evt_stats.queued.GetAvgMs())
                             << " | " << fmtD(evt_stats.submit.GetAvgMs())
                             << " | " << fmtD(evt_stats.start.GetAvgMs())
                             << " | " << fmtD(evt_stats.end.GetAvgMs())
                             << " | " << fmtD(evt_stats.complete.GetAvgMs())
                             << " |\n";
                        first_event = false;
                    }
                    file << "| | **ИТОГО** | " << mod_stats.GetTotalCalls()
                         << " | " << std::fixed << std::setprecision(2) << mod_stats.GetTotalTimeMs()
                         << " | | | | |\n";
                }

                file << "\n";
            }

            // Legend
            file << "---\n\n";
            file << "## Легенда\n\n";
            file << "| Колонка | Описание |\n";
            file << "|---------|----------|\n";
            file << "| **N** | Количество вызовов |\n";
            file << "| **Очередь** | Команда попала в очередь хоста (queued_ns) |\n";
            file << "| **Отправка** | Команда отправлена на GPU (submit_ns) |\n";
            file << "| **Старт** | Кернел начал выполняться (start_ns) |\n";
            file << "| **Конец** | Кернел закончил выполняться (end_ns) |\n";
            file << "| **Готово** | Данные выгружены/доступны (complete_ns) |\n";
            file << "\n*Время в миллисекундах (мс), усреднённое значение*\n";

            file.close();
            std::cout << "[GPUProfiler] Exported MD to: " << file_path << "\n";
            return true;

        } catch (const std::exception& e) {
            std::cerr << "[GPUProfiler] MD Export error: " << e.what() << "\n";
            return false;
        }
    }

protected:
    // ========================================================================
    // AsyncServiceBase implementation
    // ========================================================================

    /**
     * @brief Process one profiling message (runs in worker thread)
     */
    void ProcessMessage(const ProfilingMessage& msg) override {
        {
            std::lock_guard<std::mutex> lock(profile_filter_mutex_);
            if (disabled_gpus_.find(msg.gpu_id) != disabled_gpus_.end()) return;
        }

        std::lock_guard<std::mutex> lock(stats_mutex_);
        auto& module_stats = stats_[msg.gpu_id][msg.module_name];
        module_stats.module_name = msg.module_name;
        auto& event_stats = module_stats.events[msg.event_name];
        event_stats.event_name = msg.event_name;

        std::visit([&event_stats](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ROCmProfilingData>) {
                event_stats.UpdateROCm(arg);
            } else {
                event_stats.UpdateFull(arg);
            }
        }, msg.time_);
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
    mutable std::mutex stats_mutex_;

    /// Per-GPU disable set
    std::unordered_set<int> disabled_gpus_;
    mutable std::mutex profile_filter_mutex_;

    /// GPU info for report headers
    std::map<int, GPUReportInfo> gpu_info_;
    mutable std::mutex gpu_info_mutex_;

    std::atomic<bool> enabled_;
};

} // namespace drv_gpu_lib
