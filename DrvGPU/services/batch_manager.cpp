/**
 * @file batch_manager.cpp
 * @brief BatchManager implementation - methods that depend on IBackend
 *
 * ============================================================================
 * SEPARATION:
 *   Header (batch_manager.hpp): inline methods that don't need IBackend
 *   This file: methods that query IBackend for GPU memory info
 *
 * This avoids circular #include issues between services/ and common/.
 * ============================================================================
 *
 * @author Codo (AI Assistant)
 * @date 2026-02-07
 */

#include "batch_manager.hpp"
#include "../interface/i_backend.hpp"

#include <algorithm>
#include <iostream>

namespace drv_gpu_lib {

// ============================================================================
// Memory-dependent methods
// ============================================================================

size_t BatchManager::GetAvailableMemory(IBackend* backend) {
    if (!backend || !backend->IsInitialized()) {
        return 0;
    }

    // Get total global memory from device
    size_t total_memory = backend->GetGlobalMemorySize();

    // Estimate: assume 10% is used by OS/driver/other allocations
    // This is a rough heuristic. For precise control, modules should
    // track their own allocations via MemoryManager.
    size_t estimated_available = static_cast<size_t>(
        static_cast<double>(total_memory) * 0.9);

    return estimated_available;
}

size_t BatchManager::CalculateOptimalBatchSize(
    IBackend* backend,
    size_t total_items,
    size_t item_memory_bytes,
    double memory_limit)
{
    if (!backend || total_items == 0 || item_memory_bytes == 0) {
        return total_items;
    }

    // Get available memory
    size_t available = GetAvailableMemory(backend);

    if (available == 0) {
        // Fallback: use 22% of items (conservative guess)
        size_t fallback = std::max(
            static_cast<size_t>(total_items * 0.22),
            static_cast<size_t>(1));
        std::cerr << "[BatchManager] WARNING: Cannot query GPU memory, "
                  << "using fallback batch size: " << fallback << "\n";
        return fallback;
    }

    // Calculate using the inline helper
    size_t batch_size = CalculateBatchSizeFromMemory(
        available, total_items, item_memory_bytes, memory_limit);

    return batch_size;
}

bool BatchManager::AllItemsFit(
    IBackend* backend,
    size_t total_items,
    size_t item_memory_bytes,
    double memory_limit)
{
    if (!backend || total_items == 0) {
        return true;
    }

    size_t available = GetAvailableMemory(backend);
    size_t usable = static_cast<size_t>(
        static_cast<double>(available) * memory_limit);
    size_t required = total_items * item_memory_bytes;

    return required <= usable;
}

} // namespace drv_gpu_lib
