/**
 * @file batch_manager.cpp
 * @brief Реализация BatchManager — методы, зависящие от IBackend
 *
 * ============================================================================
 * РАЗДЕЛЕНИЕ:
 *   Заголовок (batch_manager.hpp): inline-методы без зависимости от IBackend
 *   Этот файл: методы, запрашивающие у IBackend информацию о памяти GPU
 *
 * Избегаем циклических #include между services/ и common/.
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
// Методы, зависящие от памяти
// ============================================================================

size_t BatchManager::GetAvailableMemory(IBackend* backend) {
    if (!backend || !backend->IsInitialized()) {
        return 0;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Получить РЕАЛЬНО СВОБОДНУЮ память GPU (через расширения вендоров)
    // NVIDIA: CL_DEVICE_GLOBAL_FREE_MEMORY_NV
    // AMD: CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
    // Fallback: GetGlobalMemorySize() * 0.9
    // ═══════════════════════════════════════════════════════════════════════
    return backend->GetFreeMemorySize();
}

size_t BatchManager::CalculateOptimalBatchSize(
    IBackend* backend,
    size_t total_items,
    size_t item_memory_bytes,
    double memory_limit,
    size_t external_memory_bytes)
{
    if (!backend || total_items == 0 || item_memory_bytes == 0) {
        return total_items;
    }

    // Получить доступную память
    size_t available = GetAvailableMemory(backend);

    if (available == 0) {
        // Запасной вариант: 22% элементов (консервативная оценка)
        size_t fallback = std::max(
            static_cast<size_t>(total_items * 0.22),
            static_cast<size_t>(1));
        std::cerr << "[BatchManager] WARNING: Cannot query GPU memory, "
                  << "using fallback batch size: " << fallback << "\n";
        return fallback;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Вычитаем память уже занятую внешними данными (напр., буфер от генератора)
    // ═══════════════════════════════════════════════════════════════════════════
    if (external_memory_bytes > 0) {
        if (external_memory_bytes >= available) {
            // Внешние данные занимают почти всю память — используем минимальный batch
            std::cerr << "[BatchManager] WARNING: External memory ("
                      << (external_memory_bytes / 1024 / 1024) << " MB) >= available ("
                      << (available / 1024 / 1024) << " MB), using batch=1\n";
            return 1;
        }
        available -= external_memory_bytes;
    }

    // Расчёт через inline-вспомогательную функцию
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
