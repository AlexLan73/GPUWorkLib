#pragma once

/**
 * @file batch_manager.hpp
 * @brief BatchManager - Universal batch processing manager for GPU modules
 *
 * ============================================================================
 * PURPOSE:
 *   Centralized batch size calculation and batch range generation.
 *   Moved out of fft_maxima to be reusable across ALL GPU modules.
 *
 * KEY FEATURES:
 *   - Analyzes REAL available GPU memory (not just total)
 *   - Takes configurable % of available memory (default 70%)
 *   - Smart tail merging: if last batch has 1-3 items, merge with previous
 *   - Works with any IBackend (not OpenCL-specific)
 *
 * USAGE:
 *   BatchManager manager;
 *
 *   // Calculate optimal batch size
 *   size_t per_item_memory = nFFT * sizeof(complex<float>) * 2 + maxima_size;
 *   size_t batch_size = manager.CalculateOptimalBatchSize(
 *       backend, total_beams, per_item_memory, 0.7);
 *
 *   // Generate batch ranges (with smart tail merging)
 *   auto batches = manager.CreateBatches(total_beams, batch_size, 3, true);
 *
 *   for (auto& batch : batches) {
 *       ProcessBatch(input, batch.start, batch.count);
 *   }
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include <vector>
#include <cstddef>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>

// Forward declaration (avoid circular dependency)
namespace drv_gpu_lib {
    class IBackend;
}

namespace drv_gpu_lib {

// ============================================================================
// BatchRange - Describes one batch of items to process
// ============================================================================

/**
 * @struct BatchRange
 * @brief Describes a range of items for one batch
 *
 * Used by modules to iterate over batches:
 *   for (auto& batch : batches) {
 *       process(input_data, batch.start, batch.count);
 *   }
 */
struct BatchRange {
    /// Starting index (0-based)
    size_t start = 0;

    /// Number of items in this batch
    size_t count = 0;

    /// Batch index (0-based, sequential)
    size_t batch_idx = 0;

    /// Flag: this batch was merged with a small tail
    bool is_merged = false;
};

// ============================================================================
// BatchManager - Universal batch processing manager
// ============================================================================

/**
 * @class BatchManager
 * @brief Calculates optimal batch sizes and generates batch ranges
 *
 * NOT a singleton - can be created per-module if needed.
 * No internal state between calls (pure computation).
 */
class BatchManager {
public:
    // ========================================================================
    // Batch Size Calculation
    // ========================================================================

    /**
     * @brief Calculate optimal batch size based on available GPU memory
     *
     * @param backend Pointer to IBackend (for memory queries)
     * @param total_items Total number of items to process (e.g., beams)
     * @param item_memory_bytes Memory required per item on GPU
     *        Example: nFFT * sizeof(complex<float>) * 2 + maxima_buffer
     * @param memory_limit Fraction of available memory to use (0.0 - 1.0)
     *        Default: 0.7 (use 70% of available GPU memory)
     * @return Optimal number of items per batch
     *
     * ALGORITHM:
     * 1. Query total GPU memory (GetGlobalMemorySize)
     * 2. Estimate allocated memory (heuristic: 10% overhead)
     * 3. available = total * memory_limit
     * 4. batch_size = available / item_memory_bytes
     * 5. Clamp to [1, total_items]
     *
     * If all items fit in memory, returns total_items (no batching needed).
     */
    static size_t CalculateOptimalBatchSize(
        IBackend* backend,
        size_t total_items,
        size_t item_memory_bytes,
        double memory_limit = 0.7);

    /**
     * @brief Calculate batch size from known available memory
     *
     * @param available_memory_bytes Available GPU memory in bytes
     * @param total_items Total number of items
     * @param item_memory_bytes Memory per item
     * @param memory_limit Fraction of available memory to use
     * @return Optimal batch size
     *
     * Use this when you already know the available memory
     * (e.g., from MemoryManager::GetFreeMemory()).
     */
    static size_t CalculateBatchSizeFromMemory(
        size_t available_memory_bytes,
        size_t total_items,
        size_t item_memory_bytes,
        double memory_limit = 0.7);

    // ========================================================================
    // Batch Range Generation
    // ========================================================================

    /**
     * @brief Create list of batch ranges with smart tail merging
     *
     * @param total_items Total number of items to process
     * @param items_per_batch Items per batch (from CalculateOptimalBatchSize)
     * @param min_tail Minimum items for the last batch to be standalone
     *        If last batch has fewer items, merge with previous batch.
     *        Default: 3 (if last batch has 1-3 items, merge)
     * @param merge_small_tail Enable tail merging optimization
     *        Default: true
     * @return Vector of BatchRange structs
     *
     * TAIL MERGING EXAMPLE:
     *   total=23, per_batch=10, min_tail=3
     *   WITHOUT merging: [0-9], [10-19], [20-22]  (3 batches, last has 3 items)
     *   WITH merging:    [0-9], [10-22]            (2 batches, last has 13 items)
     *
     *   total=22, per_batch=10, min_tail=3
     *   WITHOUT merging: [0-9], [10-19], [20-21]  (3 batches, last has 2 items)
     *   WITH merging:    [0-9], [10-21]            (2 batches, last has 12 items)
     *
     *   total=25, per_batch=10, min_tail=3
     *   WITHOUT merging: [0-9], [10-19], [20-24]  (3 batches, last has 5 items)
     *   WITH merging:    [0-9], [10-19], [20-24]  (3 batches, NO merge - tail >= min_tail+1)
     */
    static std::vector<BatchRange> CreateBatches(
        size_t total_items,
        size_t items_per_batch,
        size_t min_tail = 3,
        bool merge_small_tail = true);

    // ========================================================================
    // Memory Queries
    // ========================================================================

    /**
     * @brief Get estimated available GPU memory
     * @param backend Pointer to IBackend
     * @return Available memory in bytes (total * 0.9 - rough estimate)
     *
     * NOTE: This is an ESTIMATE. OpenCL doesn't provide exact free memory.
     * We use: total_memory * 0.9 (assume 10% is used by OS/driver).
     * For more precise control, use MemoryManager::GetAllocatedSize().
     */
    static size_t GetAvailableMemory(IBackend* backend);

    /**
     * @brief Check if all items fit in memory (no batching needed)
     * @param backend Pointer to IBackend
     * @param total_items Total number of items
     * @param item_memory_bytes Memory per item
     * @param memory_limit Fraction of memory to use
     * @return true if all items fit, false if batching is needed
     */
    static bool AllItemsFit(
        IBackend* backend,
        size_t total_items,
        size_t item_memory_bytes,
        double memory_limit = 0.7);

    // ========================================================================
    // Diagnostics
    // ========================================================================

    /**
     * @brief Print batch configuration to stdout
     * @param batches Vector of batch ranges
     * @param total_items Total items being processed
     */
    static void PrintBatchInfo(
        const std::vector<BatchRange>& batches,
        size_t total_items);
};

// ============================================================================
// Inline Implementation
// ============================================================================

inline size_t BatchManager::CalculateBatchSizeFromMemory(
    size_t available_memory_bytes,
    size_t total_items,
    size_t item_memory_bytes,
    double memory_limit)
{
    if (item_memory_bytes == 0 || total_items == 0) {
        return total_items;
    }

    // Usable memory
    size_t usable = static_cast<size_t>(
        static_cast<double>(available_memory_bytes) * memory_limit);

    // How many items fit?
    size_t fits = usable / item_memory_bytes;

    // Clamp to [1, total_items]
    fits = std::max(fits, static_cast<size_t>(1));
    fits = std::min(fits, total_items);

    return fits;
}

inline std::vector<BatchRange> BatchManager::CreateBatches(
    size_t total_items,
    size_t items_per_batch,
    size_t min_tail,
    bool merge_small_tail)
{
    std::vector<BatchRange> batches;

    if (total_items == 0 || items_per_batch == 0) {
        return batches;
    }

    // If all items fit in one batch
    if (items_per_batch >= total_items) {
        BatchRange single;
        single.start = 0;
        single.count = total_items;
        single.batch_idx = 0;
        batches.push_back(single);
        return batches;
    }

    // Calculate number of full batches and remainder
    size_t num_full = total_items / items_per_batch;
    size_t remainder = total_items % items_per_batch;

    // Tail merging: if remainder is small (1..min_tail), merge with previous
    if (merge_small_tail && remainder > 0 && remainder <= min_tail && num_full > 0) {
        num_full--;
        remainder += items_per_batch;
    }

    // Generate batch ranges
    size_t current = 0;
    size_t idx = 0;

    // Full batches
    for (size_t i = 0; i < num_full; ++i) {
        BatchRange batch;
        batch.start = current;
        batch.count = items_per_batch;
        batch.batch_idx = idx++;
        batches.push_back(batch);
        current += items_per_batch;
    }

    // Last batch (remainder)
    if (remainder > 0 && current < total_items) {
        BatchRange batch;
        batch.start = current;
        batch.count = remainder;
        batch.batch_idx = idx;
        batch.is_merged = (remainder > items_per_batch);  // Flag if merged
        batches.push_back(batch);
    }

    return batches;
}

inline void BatchManager::PrintBatchInfo(
    const std::vector<BatchRange>& batches,
    size_t total_items)
{
    std::cout << "  Batch Configuration:\n";
    std::cout << "    Total items: " << total_items << "\n";
    std::cout << "    Num batches: " << batches.size() << "\n";

    for (const auto& batch : batches) {
        std::cout << "    Batch " << batch.batch_idx
                  << ": items [" << batch.start
                  << " .. " << (batch.start + batch.count - 1) << "]"
                  << " count=" << batch.count;
        if (batch.is_merged) {
            std::cout << " (merged tail)";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

} // namespace drv_gpu_lib
