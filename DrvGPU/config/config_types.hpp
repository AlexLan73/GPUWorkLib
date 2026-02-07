#pragma once

/**
 * @file config_types.hpp
 * @brief GPUConfigEntry - Configuration data types for GPU devices
 *
 * ============================================================================
 * PURPOSE:
 *   Defines the configuration structure for each GPU device.
 *   Used by GPUConfig to load/save configGPU.json.
 *
 * DESIGN PRINCIPLE:
 *   Every field has a DEFAULT value. If the field is missing in JSON,
 *   the default is used. This ensures forward/backward compatibility
 *   when new fields are added in the future.
 *
 * EXAMPLE JSON:
 *   {
 *     "id": 0,
 *     "name": "Alex",
 *     "is_prof": true,
 *     "is_logger": true
 *   }
 *   // is_console defaults to false, is_active defaults to true, etc.
 *
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include <string>
#include <cstddef>
#include <vector>


namespace drv_gpu_lib {

// ============================================================================
// GPUConfigEntry - Configuration for one GPU device
// ============================================================================

/**
 * @struct GPUConfigEntry
 * @brief Configuration parameters for a single GPU device
 *
 * ALL fields have default values. When loading from JSON,
 * any missing field will use its default.
 *
 * Field semantics:
 * - id: GPU device index (0-based, matches OpenCL device index)
 * - name: Human-readable name (for logs, console, debugging)
 * - is_prof: Enable GPU profiling for this device
 * - is_logger: Enable file logging for this device
 * - is_console: Enable console output for this device
 * - is_active: Whether this GPU should be initialized at startup
 * - is_db: Enable database output (future)
 * - max_memory_percent: Maximum GPU memory usage (% of total)
 * - log_level: Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR")
 */
struct GPUConfigEntry {
    // ========================================================================
    // Core identification
    // ========================================================================

    /// GPU device index (0-based, corresponds to OpenCL device index)
    int id = 0;

    /// Human-readable name for this GPU (used in logs and console)
    std::string name = "GPU";

    // ========================================================================
    // Feature flags (all default to false for safety)
    // ========================================================================

    /// Enable profiling data collection for this GPU
    /// When true, kernel execution times and memory stats are recorded
    bool is_prof = false;

    /// Enable file logging for this GPU
    /// When true, log files created at: ${path}/Logs/DRVGPU_XX/YYYY-MM-DD/HH-MM-SS.log
    bool is_logger = false;

    /// Enable console output for this GPU
    /// When true, messages from this GPU appear in stdout
    bool is_console = false;

    /// Whether this GPU is active and should be initialized
    /// Set to false to skip this GPU during GPUManager::InitializeAll()
    bool is_active = true;

    /// Enable database output (future feature)
    bool is_db = false;

    // ========================================================================
    // Resource limits
    // ========================================================================

    /// Maximum GPU memory to use (percentage of total global memory)
    /// Used by BatchManager to calculate batch sizes
    /// Default: 70% (conservative, leaves room for OS and other processes)
    size_t max_memory_percent = 70;

    // ========================================================================
    // Logging settings
    // ========================================================================

    /// Minimum log level for this GPU
    /// Options: "DEBUG", "INFO", "WARNING", "ERROR"
    /// Default: "INFO" (skip debug messages in production)
    std::string log_level = "INFO";
};

// ============================================================================
// GPUConfigData - Root configuration structure
// ============================================================================

/**
 * @struct GPUConfigData
 * @brief Root structure for configGPU.json
 *
 * Contains version info and list of GPU configurations.
 */
struct GPUConfigData {
    /// Configuration file format version
    std::string version = "1.0";

    /// Human-readable description
    std::string description = "GPU Configuration for DrvGPU";

    /// List of GPU configurations
    std::vector<GPUConfigEntry> gpus;
};

} // namespace drv_gpu_lib
