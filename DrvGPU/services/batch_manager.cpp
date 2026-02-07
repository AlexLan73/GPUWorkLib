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

    // Получить общий объём глобальной памяти устройства
    size_t total_memory = backend->GetGlobalMemorySize();

    // Оценка: считаем, что 10% занято ОС/драйвером/другими аллокациями
    // Эвристика. Для точного контроля модули должны отслеживать
    // свои аллокации через MemoryManager.
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
