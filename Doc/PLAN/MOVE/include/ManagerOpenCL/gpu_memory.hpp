#pragma once

/**
 * @file gpu_memory.hpp
 * @brief Главный header для системы управления GPU памятью
 * 
 * Включает все компоненты:
 * - SVMCapabilities: определение возможностей SVM
 * - IMemoryBuffer: абстрактный интерфейс
 * - SVMBuffer: RAII для SVM памяти
 * - RegularBuffer: RAII для традиционных буферов
 * - BufferFactory: фабрика с автовыбором стратегии
 * 
 * @example Базовое использование:
 * @code
 * #include "GPU/gpu_memory.hpp"
 * 
 * // Инициализация
 * ManagerOpenCL::OpenCLComputeEngine::Initialize();
 * auto& engine = ManagerOpenCL::OpenCLComputeEngine::GetInstance();
 * 
 * // Создание фабрики
 * auto factory = engine.CreateBufferFactory();
 * 
 * // Создание буфера (автоматический выбор SVM/Regular)
 * auto buffer = factory->Create(1024 * 1024);  // 1M elements
 * 
 * // Запись данных
 * buffer->Write(my_data);
 * 
 * // Установка как аргумент kernel
 * buffer->SetAsKernelArg(kernel, 0);
 * 
 * // Чтение результатов
 * auto result = buffer->Read();
 * @endcode
 * 
 * @example Принудительное использование SVM:
 * @code
 * ManagerOpenCL::BufferConfig config = ManagerOpenCL::BufferConfig::SVMOnly();
 * auto factory = engine.CreateBufferFactory(config);
 * auto svm_buffer = factory->Create(1024);
 * @endcode
 * 
 * @example Работа с конкретной стратегией:
 * @code
 * auto buffer = factory->CreateWithStrategy(
 *     1024,
 *     ManagerOpenCL::MemoryStrategy::SVM_COARSE_GRAIN,
 *     ManagerOpenCL::MemoryType::GPU_READ_WRITE
 * );
 * @endcode
 * 
 * @author Codo (AI Assistant)
 * @date 2026-01-19
 */

// ════════════════════════════════════════════════════════════════════════════
// Базовые типы и enum-ы
// ════════════════════════════════════════════════════════════════════════════

#include "memory_type.hpp"
#include "svm_capabilities.hpp"

// ════════════════════════════════════════════════════════════════════════════
// Интерфейсы и реализации буферов
// ════════════════════════════════════════════════════════════════════════════

#include "i_memory_buffer.hpp"
#include "svm_buffer.hpp"
#include "regular_buffer.hpp"
#include "hybrid_buffer.hpp"

// ════════════════════════════════════════════════════════════════════════════
// Для совместимости со старым кодом
// ════════════════════════════════════════════════════════════════════════════

#include "gpu_memory_buffer.hpp"

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// Вспомогательные функции
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Проверить поддержку SVM на устройстве
 * @param device OpenCL device ID
 * @return true если хотя бы один тип SVM поддерживается
 */
inline bool IsSVMSupported(cl_device_id device) {
    auto caps = SVMCapabilities::Query(device);
    return caps.HasAnySVM();
}

/**
 * @brief Получить рекомендуемую стратегию для устройства
 * @param device OpenCL device ID
 * @return Лучшая доступная стратегия
 */
inline MemoryStrategy GetRecommendedStrategy(cl_device_id device) {
    auto caps = SVMCapabilities::Query(device);
    return caps.GetBestSVMStrategy();
}

/**
 * @brief Создать строку с информацией о памяти
 * @param buffer Указатель на буфер
 * @return Строка с информацией
 */
inline std::string GetBufferDescription(const IMemoryBuffer* buffer) {
    if (!buffer) return "null";
    
    auto info = buffer->GetInfo();
    std::ostringstream oss;
    oss << MemoryStrategyToString(info.strategy) << " buffer: "
        << info.num_elements << " elements ("
        << (info.size_bytes / (1024.0 * 1024.0)) << " MB)";
    return oss.str();
}

// ════════════════════════════════════════════════════════════════════════════
// Утилиты для миграции со старого GPUMemoryBuffer
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Адаптер для миграции старого GPUMemoryBuffer на новый интерфейс
 * 
 * Позволяет постепенно мигрировать код без полной переработки.
 * 
 * @deprecated Используйте IMemoryBuffer напрямую для нового кода
 */
class LegacyBufferAdapter {
public:
    /**
     * @brief Создать адаптер из старого GPUMemoryBuffer
     */
    static std::unique_ptr<IMemoryBuffer> FromLegacy(
        std::unique_ptr<GPUMemoryBuffer> legacy_buffer,
        cl_context context,
        cl_command_queue queue) {
        
        if (!legacy_buffer) {
            throw std::invalid_argument("legacy_buffer is null");
        }
        
        // Обернуть cl_mem из старого буфера
        return std::make_unique<RegularBuffer>(
            context,
            queue,
            legacy_buffer->Get(),
            legacy_buffer->GetNumElements(),
            legacy_buffer->GetMemoryType()
        );
    }
};

} // namespace ManagerOpenCL

// ════════════════════════════════════════════════════════════════════════════
// Документация архитектуры
// ════════════════════════════════════════════════════════════════════════════

/**
 * @page gpu_memory_architecture GPU Memory Architecture
 * 
 * @section overview Обзор
 * 
 * Система управления памятью GPU поддерживает два режима работы:
 * 
 * 1. **Regular Buffers (OpenCL 1.x+)**
 *    - Традиционные cl_mem буферы
 *    - clCreateBuffer / clEnqueueReadBuffer / clEnqueueWriteBuffer
 *    - Полная совместимость со всеми GPU
 * 
 * 2. **SVM Buffers (OpenCL 2.0+)**
 *    - Shared Virtual Memory
 *    - Zero-copy операции
 *    - Три типа: Coarse-Grain, Fine-Grain, Fine-Grain System
 * 
 * @section strategy_selection Выбор стратегии
 * 
 * BufferFactory автоматически выбирает стратегию на основе:
 * 
 * | Размер буфера | SVM доступен | Стратегия |
 * |---------------|--------------|-----------|
 * | < 1 MB        | -            | Regular   |
 * | 1-64 MB       | Да           | SVM Coarse |
 * | 1-64 MB       | Нет          | Regular   |
 * | > 64 MB       | Да           | SVM (best) |
 * | > 64 MB       | Нет          | Regular   |
 * 
 * @section class_diagram Диаграмма классов
 * 
 * @verbatim
 *                    ┌─────────────────────┐
 *                    │   IMemoryBuffer     │ (interface)
 *                    │   (abstract)        │
 *                    └─────────┬───────────┘
 *                              │
 *            ┌─────────────────┼─────────────────┐
 *            │                 │                 │
 *   ┌────────┴────────┐ ┌──────┴──────┐ ┌───────┴───────┐
 *   │  RegularBuffer  │ │  SVMBuffer  │ │ GPUMemoryBuffer│
 *   │  (cl_mem)       │ │  (SVM ptr)  │ │ (legacy)       │
 *   └─────────────────┘ └─────────────┘ └────────────────┘
 *            │                 │
 *            └────────┬────────┘
 *                     │
 *            ┌────────┴────────┐
 *            │  BufferFactory  │
 *            │  (auto-select)  │
 *            └─────────────────┘
 * @endverbatim
 * 
 * @section usage_patterns Паттерны использования
 * 
 * @subsection new_code Новый код
 * 
 * @code
 * // Рекомендуемый способ
 * auto factory = engine.CreateBufferFactory();
 * auto buffer = factory->Create(size);
 * buffer->Write(data);
 * // ...
 * auto result = buffer->Read();
 * @endcode
 * 
 * @subsection migration Миграция старого кода
 * 
 * @code
 * // Старый код:
 * auto old_buffer = GPUMemoryManager::CreateBuffer(size, type);
 * old_buffer->WriteToGPU(data);
 * 
 * // Новый код (минимальные изменения):
 * auto factory = engine.CreateBufferFactory();
 * auto new_buffer = factory->Create(size, type);
 * new_buffer->Write(data);
 * @endcode
 * 
 * @section performance Производительность
 * 
 * | Операция | Regular | SVM Coarse | SVM Fine |
 * |----------|---------|------------|----------|
 * | Создание | Fast    | Medium     | Medium   |
 * | Write    | Copy    | Zero-copy* | Zero-copy|
 * | Read     | Copy    | Zero-copy* | Zero-copy|
 * | Kernel   | Fast    | Fast       | Fast     |
 * 
 * * Требует map/unmap
 * 
 * @section thread_safety Thread Safety
 * 
 * - BufferFactory: Thread-safe (статистика защищена mutex)
 * - IMemoryBuffer: NOT thread-safe (один буфер = один поток)
 * - Для многопоточности: создавайте отдельные буферы для каждого потока
 */

