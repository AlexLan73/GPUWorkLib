#pragma once

/**
 * @file memory_manager.hpp
 * @brief Менеджер памяти для DrvGPU (обёртка над backend memory)
 * 
 * MemoryManager - слой абстракции над backend-специфичными
 * операциями с памятью. Предоставляет единый интерфейс независимо
 * от бэкенда (OpenCL, CUDA, Vulkan).
 * 
 * @author DrvGPU Team
 * @date 2026-01-31
 */

#include "i_backend.hpp"
#include "gpu_buffer.hpp"
#include <memory>
#include <vector>
#include <string>
#include <mutex>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: MemoryManager - Управление памятью GPU
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class MemoryManager
 * @brief Backend-агностичное управление памятью GPU
 * 
 * MemoryManager предоставляет высокоуровневый интерфейс для работы
 * с GPU памятью, скрывая детали конкретного бэкенда.
 * 
 * Основные возможности:
 * - Создание GPU буферов (GPUBuffer)
 * - Отслеживание аллокаций
 * - Статистика использования памяти
 * - RAII для автоматической очистки
 * 
 * Использование:
 * @code
 * MemoryManager& mem_mgr = gpu.GetMemoryManager();
 * 
 * // Создать буфер
 * auto buffer = mem_mgr.CreateBuffer<float>(1024);
 * 
 * // Записать данные
 * std::vector<float> data(1024, 1.0f);
 * buffer->Write(data.data(), 1024 * sizeof(float));
 * 
 * // Прочитать данные
 * std::vector<float> result(1024);
 * buffer->Read(result.data(), 1024 * sizeof(float));
 * @endcode
 */
class MemoryManager {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструктор и деструктор
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать MemoryManager привязанный к бэкенду
     * @param backend Указатель на IBackend
     */
    explicit MemoryManager(IBackend* backend);
    
    /**
     * @brief Деструктор (освобождает все буферы)
     */
    ~MemoryManager();
    
    // ═══════════════════════════════════════════════════════════════
    // Запрет копирования, разрешение перемещения
    // ═══════════════════════════════════════════════════════════════
    
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    MemoryManager(MemoryManager&& other) noexcept;
    MemoryManager& operator=(MemoryManager&& other) noexcept;
    
    // ═══════════════════════════════════════════════════════════════
    // Создание буферов
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать GPU буфер заданного размера
     * @tparam T Тип элементов
     * @param num_elements Количество элементов
     * @param flags Backend-специфичные флаги (0 по умолчанию)
     * @return Shared pointer на GPUBuffer
     */
    template<typename T>
    std::shared_ptr<GPUBuffer<T>> CreateBuffer(size_t num_elements, 
                                               unsigned int flags = 0);
    
    /**
     * @brief Создать GPU буфер с начальными данными
     * @tparam T Тип элементов
     * @param data Указатель на данные
     * @param num_elements Количество элементов
     * @param flags Backend-специфичные флаги
     */
    template<typename T>
    std::shared_ptr<GPUBuffer<T>> CreateBuffer(const T* data, 
                                               size_t num_elements,
                                               unsigned int flags = 0);
    
    // ═══════════════════════════════════════════════════════════════
    // Прямое выделение памяти (низкоуровневое)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Выделить память на GPU (возвращает void*)
     * @param size_bytes Размер в байтах
     * @param flags Backend-специфичные флаги
     * @return Указатель на выделенную память
     */
    void* Allocate(size_t size_bytes, unsigned int flags = 0);
    
    /**
     * @brief Освободить память
     * @param ptr Указатель на память
     */
    void Free(void* ptr);
    
    // ═══════════════════════════════════════════════════════════════
    // Статистика
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Получить количество активных аллокаций
     */
    size_t GetAllocationCount() const;
    
    /**
     * @brief Получить общий размер выделенной памяти (bytes)
     */
    size_t GetTotalAllocatedBytes() const;
    
    /**
     * @brief Вывести статистику памяти
     */
    void PrintStatistics() const;
    
    /**
     * @brief Получить строку со статистикой
     */
    std::string GetStatistics() const;
    
    /**
     * @brief Сбросить статистику
     */
    void ResetStatistics();
    
    // ═══════════════════════════════════════════════════════════════
    // Очистка
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Освободить все буферы
     */
    void Cleanup();

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    IBackend* backend_;  ///< Указатель на бэкенд (не владеет)
    
    // Статистика
    size_t total_allocations_;
    size_t total_frees_;
    size_t current_allocations_;
    size_t total_bytes_allocated_;
    size_t peak_bytes_allocated_;
    
    // Thread-safety
    mutable std::mutex mutex_;
    
    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    void TrackAllocation(size_t size_bytes);
    void TrackFree(size_t size_bytes);
};

// ════════════════════════════════════════════════════════════════════════════
// Template реализация
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
std::shared_ptr<GPUBuffer<T>> MemoryManager::CreateBuffer(
    size_t num_elements, 
    unsigned int flags) 
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t size_bytes = num_elements * sizeof(T);
    void* ptr = backend_->Allocate(size_bytes, flags);
    
    TrackAllocation(size_bytes);
    
    return std::make_shared<GPUBuffer<T>>(ptr, num_elements, backend_);
}

template<typename T>
std::shared_ptr<GPUBuffer<T>> MemoryManager::CreateBuffer(
    const T* data, 
    size_t num_elements,
    unsigned int flags) 
{
    auto buffer = CreateBuffer<T>(num_elements, flags);
    buffer->Write(data, num_elements * sizeof(T));
    return buffer;
}

} // namespace DrvGPU
