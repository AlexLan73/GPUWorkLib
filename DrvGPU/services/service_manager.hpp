#pragma once

/**
 * @file service_manager.hpp
 * @brief ServiceManager - Centralised start/stop for all background services
 *
 * ============================================================================
 * PURPOSE:
 *   Single point for initialising, starting, and stopping all DrvGPU
 *   asynchronous services: Logger, Profiler, ConsoleOutput.
 *
 *   Reads configGPU.json and enables/disables services per GPU.
 *
 * LIFECYCLE:
 *   1. GPUManager creates GPUs
 *   2. ServiceManager::Initialize(config) - configure services from JSON
 *   3. ServiceManager::StartAll() - launch background threads
 *   4. ... GPU work, modules call Enqueue() ...
 *   5. ServiceManager::StopAll() - drain queues, join threads
 *
 * USAGE:
 *   // After GPUManager::InitializeAll():
 *   auto& sm = ServiceManager::GetInstance();
 *   sm.InitializeFromConfig("configGPU.json");
 *   sm.StartAll();
 *
 *   // ... GPU processing ...
 *
 *   // Before exit:
 *   sm.StopAll();
 *
 * THREAD SAFETY:
 *   - Initialize/Start/Stop are NOT meant to be called concurrently
 *   - They are called once from the main thread
 *   - Individual service APIs (Enqueue) are thread-safe
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "console_output.hpp"
#include "gpu_profiler.hpp"
#include "../config/gpu_config.hpp"
#include "../common/config_logger.hpp"

#include <string>
#include <iostream>
#include <vector>

namespace drv_gpu_lib {

// ============================================================================
// ServiceManager - Centralized service lifecycle manager
// ============================================================================

/**
 * @class ServiceManager
 * @brief Singleton that manages the lifecycle of all background services
 *
 * Responsibilities:
 * - Read configGPU.json and configure services
 * - Start/stop ConsoleOutput, GPUProfiler background threads
 * - Configure per-GPU Logger paths
 * - Provide convenience API for service status
 */
class ServiceManager {
public:
    // ========================================================================
    // Singleton
    // ========================================================================

    /**
     * @brief Get singleton instance
     */
    static ServiceManager& GetInstance() {
        static ServiceManager instance;
        return instance;
    }

    // Delete copy
    ServiceManager(const ServiceManager&) = delete;
    ServiceManager& operator=(const ServiceManager&) = delete;

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * @brief Initialize services from configGPU.json
     * @param config_file Path to configGPU.json
     * @return true if configuration loaded successfully
     *
     * Reads the JSON config and applies settings:
     * - is_console  -> ConsoleOutput per-GPU enable/disable
     * - is_prof     -> GPUProfiler enable flag
     * - is_logger   -> ConfigLogger per-GPU log paths
     * - log_level   -> Logger level configuration
     *
     * Does NOT start services (call StartAll() for that).
     */
    bool InitializeFromConfig(const std::string& config_file) {
        // Load config (or create default if missing)
        bool ok = GPUConfig::GetInstance().LoadOrCreate(config_file);
        if (!ok) {
            std::cerr << "[ServiceManager] WARNING: Failed to load config, using defaults\n";
        }

        const auto& data = GPUConfig::GetInstance().GetData();

        // Configure ConsoleOutput per-GPU
        for (const auto& gpu : data.gpus) {
            ConsoleOutput::GetInstance().SetGPUEnabled(gpu.id, gpu.is_console);
        }

        // Configure GPUProfiler
        bool any_profiling = false;
        for (const auto& gpu : data.gpus) {
            if (gpu.is_prof) {
                any_profiling = true;
                break;
            }
        }
        GPUProfiler::GetInstance().SetEnabled(any_profiling);

        // Configure Logger per-GPU paths
        for (const auto& gpu : data.gpus) {
            if (gpu.is_logger) {
                // Ensure log directory exists for this GPU
                ConfigLogger::GetInstance().CreateLogDirectoryForGPU(gpu.id);
            }
        }

        initialized_ = true;

        std::cout << "[ServiceManager] Configured "
                  << data.gpus.size() << " GPU(s) from: "
                  << config_file << "\n";

        return true;
    }

    /**
     * @brief Initialize with default settings (no config file)
     *
     * Creates default config for a single GPU with all services enabled.
     * Useful for testing and development.
     */
    void InitializeDefaults() {
        // GPUConfig already has defaults from constructor
        // Just enable everything
        ConsoleOutput::GetInstance().SetEnabled(true);
        GPUProfiler::GetInstance().SetEnabled(true);
        ConfigLogger::GetInstance().Enable();

        initialized_ = true;

        std::cout << "[ServiceManager] Initialized with defaults\n";
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /**
     * @brief Start all background service threads
     *
     * Starts:
     * - ConsoleOutput worker thread
     * - GPUProfiler worker thread
     *
     * Logger (spdlog) does not need a separate thread (it's already async).
     *
     * IMPORTANT: Call InitializeFromConfig() or InitializeDefaults() first!
     */
    void StartAll() {
        if (!initialized_) {
            std::cerr << "[ServiceManager] WARNING: Not initialized, calling InitializeDefaults()\n";
            InitializeDefaults();
        }

        // Start ConsoleOutput background thread
        ConsoleOutput::GetInstance().Start();

        // Start GPUProfiler background thread
        if (GPUProfiler::GetInstance().IsEnabled()) {
            GPUProfiler::GetInstance().Start();
        }

        running_ = true;

        ConsoleOutput::GetInstance().PrintSystem("ServiceManager", "All services started");
    }

    /**
     * @brief Stop all background service threads
     *
     * Drains all message queues, then joins worker threads.
     * After this call, no more messages will be processed.
     *
     * Safe to call multiple times.
     */
    void StopAll() {
        if (!running_) return;

        ConsoleOutput::GetInstance().PrintSystem("ServiceManager", "Stopping all services...");

        // Stop GPUProfiler first (it may still be recording during shutdown)
        GPUProfiler::GetInstance().Stop();

        // Stop ConsoleOutput last (so other services can log their shutdown)
        ConsoleOutput::GetInstance().Stop();

        running_ = false;

        // Print summary after console is stopped (goes directly to stdout)
        std::cout << "[ServiceManager] All services stopped.\n";
    }

    /**
     * @brief Check if services are running
     */
    bool IsRunning() const { return running_; }

    /**
     * @brief Check if services are initialized
     */
    bool IsInitialized() const { return initialized_; }

    // ========================================================================
    // Convenience API
    // ========================================================================

    /**
     * @brief Export profiling data to JSON file
     * @param file_path Output file path
     * @return true on success
     *
     * Convenience wrapper around GPUProfiler::ExportJSON().
     * Creates parent directories if needed.
     */
    bool ExportProfiling(const std::string& file_path) const {
        return GPUProfiler::GetInstance().ExportJSON(file_path);
    }

    /**
     * @brief Print profiling summary to console
     *
     * Convenience wrapper around GPUProfiler::PrintSummary().
     */
    void PrintProfilingSummary() const {
        GPUProfiler::GetInstance().PrintSummary();
    }

    /**
     * @brief Print GPU config to console
     *
     * Convenience wrapper around GPUConfig::Print().
     */
    void PrintConfig() const {
        GPUConfig::GetInstance().Print();
    }

    /**
     * @brief Get service statistics string
     * @return Human-readable service status
     */
    std::string GetStatus() const {
        std::ostringstream oss;
        oss << "ServiceManager Status:\n";
        oss << "  Initialized: " << (initialized_ ? "YES" : "NO") << "\n";
        oss << "  Running: " << (running_ ? "YES" : "NO") << "\n";
        oss << "  ConsoleOutput: "
            << (ConsoleOutput::GetInstance().IsRunning() ? "running" : "stopped")
            << " (processed: " << ConsoleOutput::GetInstance().GetProcessedCount()
            << ", queue: " << ConsoleOutput::GetInstance().GetQueueSize() << ")\n";
        oss << "  GPUProfiler: "
            << (GPUProfiler::GetInstance().IsRunning() ? "running" : "stopped")
            << " (enabled: " << (GPUProfiler::GetInstance().IsEnabled() ? "YES" : "NO")
            << ", processed: " << GPUProfiler::GetInstance().GetProcessedCount() << ")\n";
        return oss.str();
    }

private:
    // ========================================================================
    // Private constructor (singleton)
    // ========================================================================

    ServiceManager() : initialized_(false), running_(false) {}

    ~ServiceManager() {
        // Auto-stop on destruction
        if (running_) {
            StopAll();
        }
    }

    // ========================================================================
    // Private members
    // ========================================================================

    /// Initialization flag
    bool initialized_;

    /// Running flag
    bool running_;
};

} // namespace drv_gpu_lib
