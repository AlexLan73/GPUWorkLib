#pragma once

/**
 * @file gpu_config.hpp
 * @brief GPUConfig - Singleton for managing GPU configuration (configGPU.json)
 *
 * ============================================================================
 * PURPOSE:
 *   Centralized GPU configuration management:
 *   - Load/Save configGPU.json
 *   - Provide per-GPU configuration (is_prof, is_logger, etc.)
 *   - Auto-create default config if file doesn't exist
 *   - Thread-safe access
 *
 * JSON FORMAT (configGPU.json):
 *   {
 *     "version": "1.0",
 *     "description": "GPU Configuration for DrvGPU",
 *     "gpus": [
 *       {
 *         "id": 0,
 *         "name": "Alex",
 *         "is_prof": true,
 *         "is_logger": true,
 *         "is_console": true
 *       },
 *       {
 *         "id": 1,
 *         "name": "Evgeni"
 *       }
 *     ]
 *   }
 *
 * MISSING FIELDS:
 *   If a field is missing in JSON, the default value from GPUConfigEntry
 *   is used. Example: GPU id=1 above has is_prof=false (default).
 *
 * AUTO-CREATE:
 *   If configGPU.json doesn't exist, it is created with a single GPU:
 *   { id: 0, name: "TEST", is_prof: true, is_logger: true }
 *
 * USAGE:
 *   // Load configuration
 *   GPUConfig::GetInstance().Load("./configGPU.json");
 *
 *   // Get config for specific GPU
 *   auto& cfg = GPUConfig::GetInstance().GetConfig(0);
 *   if (cfg.is_prof) { ... }
 *
 *   // Get all configs
 *   auto& all = GPUConfig::GetInstance().GetAllConfigs();
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "config_types.hpp"

#include <string>
#include <vector>
#include <mutex>
#include <optional>

namespace drv_gpu_lib {

// ============================================================================
// GPUConfig - Singleton for GPU configuration
// ============================================================================

/**
 * @class GPUConfig
 * @brief Singleton manager for GPU configuration
 *
 * Provides centralized access to GPU configuration loaded from configGPU.json.
 * Thread-safe for concurrent reads from multiple GPU threads.
 *
 * Lifecycle:
 * 1. GPUConfig::GetInstance().Load(path) or LoadOrCreate(path)
 * 2. GPUConfig::GetInstance().GetConfig(gpu_id) - per-GPU access
 * 3. Optional: Save() to persist changes
 */
class GPUConfig {
public:
    // ========================================================================
    // Singleton access
    // ========================================================================

    /**
     * @brief Get the singleton instance
     * @return Reference to the global GPUConfig
     */
    static GPUConfig& GetInstance();

    // Delete copy/move (singleton)
    GPUConfig(const GPUConfig&) = delete;
    GPUConfig& operator=(const GPUConfig&) = delete;

    // ========================================================================
    // Loading and Saving
    // ========================================================================

    /**
     * @brief Load configuration from JSON file
     * @param file_path Path to configGPU.json
     * @return true if loaded successfully, false on error
     *
     * If loading fails (file not found, parse error), returns false
     * and keeps current configuration unchanged.
     */
    bool Load(const std::string& file_path);

    /**
     * @brief Load configuration from file, or create default if not found
     * @param file_path Path to configGPU.json
     * @return true if loaded or created successfully
     *
     * If the file doesn't exist:
     * 1. Creates default configuration (single GPU: id=0, name="TEST", is_prof=true, is_logger=true)
     * 2. Saves it to file_path
     * 3. Returns true
     *
     * If the file exists but has errors:
     * 1. Returns false
     * 2. Configuration unchanged
     */
    bool LoadOrCreate(const std::string& file_path);

    /**
     * @brief Save current configuration to JSON file
     * @param file_path Path to save (empty = last loaded path)
     * @return true if saved successfully
     */
    bool Save(const std::string& file_path = "");

    /**
     * @brief Check if configuration is loaded
     * @return true if Load() or LoadOrCreate() was called successfully
     */
    bool IsLoaded() const;

    // ========================================================================
    // Configuration Access (Thread-Safe)
    // ========================================================================

    /**
     * @brief Get configuration for specific GPU
     * @param gpu_id GPU device index
     * @return Reference to GPUConfigEntry for this GPU
     *
     * If gpu_id is not found in configuration, returns a default
     * GPUConfigEntry with the given id.
     */
    const GPUConfigEntry& GetConfig(int gpu_id) const;

    /**
     * @brief Get all GPU configurations
     * @return Reference to vector of all GPUConfigEntry
     */
    const std::vector<GPUConfigEntry>& GetAllConfigs() const;

    /**
     * @brief Get root configuration data (includes version, description)
     * @return Reference to GPUConfigData
     */
    const GPUConfigData& GetData() const;

    /**
     * @brief Get list of active GPU IDs (is_active == true)
     * @return Vector of GPU IDs that should be initialized
     */
    std::vector<int> GetActiveGPUIDs() const;

    /**
     * @brief Check if a specific GPU has profiling enabled
     * @param gpu_id GPU device index
     * @return true if is_prof == true for this GPU
     */
    bool IsProfilingEnabled(int gpu_id) const;

    /**
     * @brief Check if a specific GPU has logging enabled
     * @param gpu_id GPU device index
     * @return true if is_logger == true for this GPU
     */
    bool IsLoggingEnabled(int gpu_id) const;

    /**
     * @brief Check if a specific GPU has console output enabled
     * @param gpu_id GPU device index
     * @return true if is_console == true for this GPU
     */
    bool IsConsoleEnabled(int gpu_id) const;

    /**
     * @brief Get maximum memory percentage for a GPU
     * @param gpu_id GPU device index
     * @return Memory limit in percent (e.g., 70 means 70%)
     */
    size_t GetMaxMemoryPercent(int gpu_id) const;

    // ========================================================================
    // Modification
    // ========================================================================

    /**
     * @brief Set or update configuration for a GPU
     * @param entry New configuration entry
     *
     * If a GPU with the same id exists, it is replaced.
     * If not, a new entry is added.
     */
    void SetConfig(const GPUConfigEntry& entry);

    /**
     * @brief Reset to default configuration
     * Creates a single GPU: id=0, name="TEST", is_prof=true, is_logger=true
     */
    void ResetToDefault();

    // ========================================================================
    // Utilities
    // ========================================================================

    /**
     * @brief Get the file path that was last used for Load/Save
     * @return File path string (empty if not loaded)
     */
    std::string GetFilePath() const;

    /**
     * @brief Print configuration to stdout (for debugging)
     */
    void Print() const;

private:
    // ========================================================================
    // Private Constructor (Singleton)
    // ========================================================================

    GPUConfig();

    // ========================================================================
    // Private Methods
    // ========================================================================

    /// Create default configuration data
    GPUConfigData CreateDefaultConfig() const;

    /// Find config entry by GPU ID (returns nullptr if not found)
    const GPUConfigEntry* FindConfig(int gpu_id) const;

    // ========================================================================
    // Private Members
    // ========================================================================

    /// Root configuration data
    GPUConfigData data_;

    /// Default entry returned when GPU ID not found
    mutable GPUConfigEntry default_entry_;

    /// Path to configuration file
    std::string file_path_;

    /// Flag indicating config was loaded
    bool loaded_ = false;

    /// Mutex for thread-safe access
    mutable std::mutex mutex_;
};

} // namespace drv_gpu_lib
