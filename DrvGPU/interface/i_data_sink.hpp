#pragma once

/**
 * @file i_data_sink.hpp
 * @brief IDataSink - Universal interface for data output destinations
 *
 * ============================================================================
 * PURPOSE:
 *   Provides a common interface for all data output destinations:
 *   - Console output (ConsoleOutput)
 *   - File logging (Logger / DefaultLogger)
 *   - Profiling data (GPUProfiler)
 *   - Database (future DBSink)
 *
 * PATTERN: Strategy + Observer
 *   Services can have multiple sinks attached.
 *   Each sink processes data independently.
 *
 * USAGE:
 *   class MyCustomSink : public IDataSink {
 *       void Write(const DataRecord& record) override {
 *           // Send to your monitoring system
 *           myMonitor.send(record.gpu_id, record.message);
 *       }
 *   };
 *
 *   // Attach to logger:
 *   Logger::AddSink(std::make_shared<MyCustomSink>());
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include <string>
#include <cstdint>
#include <chrono>
#include <memory>

namespace drv_gpu_lib {

// ============================================================================
// DataRecord - Universal data record for all sinks
// ============================================================================

/**
 * @struct DataRecord
 * @brief Universal data record passed to all IDataSink implementations
 *
 * Contains all information needed by any sink type:
 * - GPU identification (gpu_id)
 * - Source module name
 * - Log level
 * - Message content
 * - Timestamp
 * - Optional numeric data (for profiling)
 */
struct DataRecord {
    /// GPU device index (0-based, -1 = no specific GPU)
    int gpu_id = -1;

    /// Source module name (e.g., "AntennaFFT", "OpenCLBackend", "MemoryManager")
    std::string module_name;

    /// Log level / record type
    enum class Level : uint8_t {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3,
        PROFILING = 4,  // Special level for profiling data
        METRIC = 5      // Special level for numeric metrics
    };
    Level level = Level::INFO;

    /// Human-readable message
    std::string message;

    /// Timestamp (auto-set on creation)
    std::chrono::system_clock::time_point timestamp = std::chrono::system_clock::now();

    /// Optional numeric value (for profiling: duration_ms, memory_bytes, etc.)
    double value = 0.0;

    /// Optional event name (for profiling: "FFT", "MemAlloc", "KernelExec")
    std::string event_name;
};

// ============================================================================
// IDataSink - Abstract interface for data output
// ============================================================================

/**
 * @interface IDataSink
 * @brief Abstract interface for all data output destinations
 *
 * Implementations:
 * - ConsoleSink (ConsoleOutput) - formatted output to stdout
 * - FileSink (DefaultLogger) - plog file output
 * - ProfilingSink (GPUProfiler) - aggregation of profiling data
 * - DBSink (future) - database output
 *
 * Thread-safety:
 *   Implementations MUST be thread-safe as they can be called
 *   from multiple GPU worker threads simultaneously.
 */
class IDataSink {
public:
    /// Virtual destructor
    virtual ~IDataSink() = default;

    /**
     * @brief Write a data record to this sink
     * @param record The data record to process
     *
     * IMPORTANT: This method must be thread-safe!
     * It will be called from the async service worker thread,
     * but multiple sinks may be called concurrently.
     */
    virtual void Write(const DataRecord& record) = 0;

    /**
     * @brief Flush any buffered data
     *
     * Called when the service is shutting down or when
     * immediate output is required.
     */
    virtual void Flush() = 0;

    /**
     * @brief Get human-readable name of this sink
     * @return Sink name (e.g., "ConsoleSink", "FileSink_GPU_00")
     */
    virtual std::string GetName() const = 0;

    /**
     * @brief Check if this sink is enabled
     * @return true if sink is active and processing records
     */
    virtual bool IsEnabled() const = 0;

    /**
     * @brief Enable or disable this sink
     * @param enabled true to enable, false to disable
     */
    virtual void SetEnabled(bool enabled) = 0;
};

/// Shared pointer to IDataSink
using IDataSinkPtr = std::shared_ptr<IDataSink>;

} // namespace drv_gpu_lib
