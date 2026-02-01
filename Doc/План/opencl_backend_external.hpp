#pragma once

/**
 * @file opencl_backend_external.hpp
 * @brief РАСШИРЕНИЕ OpenCLBackend для работы с ВНЕШНИМ OpenCL контекстом
 * 
 * НОВАЯ ФУНКЦИОНАЛЬНОСТЬ:
 * 1. Конструктор для инициализации с уже созданным cl_context/cl_device_id
 * 2. Адаптеры для работы с внешними cl_mem буферами
 * 3. Полная совместимость с вашим существующим OpenCL кодом
 * 
 * КРИТЕРИИ: Надежность + Простота
 * 
 * @author DrvGPU Team (Extended for External Context Support)
 * @date 2026-02-01
 */

#include "opencl_backend.hpp"
#include "../../memory/external_cl_buffer_adapter.hpp"
#include <CL/cl.h>
#include <memory>
#include <stdexcept>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Class: OpenCLBackendExternal - расширение для работы с внешним контекстом
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class OpenCLBackendExternal
 * @brief OpenCLBackend с поддержкой внешнего OpenCL контекста
 * 
 * USE CASE: Интеграция DrvGPU в существующий OpenCL проект
 * 
 * Пример использования:
 * @code
 * // У вас уже есть OpenCL контекст из другого класса
 * cl_context external_ctx = your_existing_opencl->GetContext();
 * cl_device_id external_dev = your_existing_opencl->GetDevice();
 * cl_command_queue external_queue = your_existing_opencl->GetQueue();
 * 
 * // Создаем DrvGPU backend, который использует ВАШ контекст
 * OpenCLBackendExternal backend(external_ctx, external_dev, external_queue);
 * backend.InitializeWithExternalContext();
 * 
 * // Теперь DrvGPU работает с вашими существующими данными на GPU!
 * // Адаптер для внешнего cl_mem буфера
 * auto adapter = backend.CreateExternalBufferAdapter<float>(your_cl_buffer, 1024);
 * 
 * // Читаем данные через DrvGPU API
 * std::vector<float> data = adapter->Read();
 * 
 * // Записываем данные через DrvGPU API
 * adapter->Write(new_data);
 * @endcode
 * 
 * ВАЖНО:
 * - Backend НЕ владеет внешними объектами (context, device, queue, buffers)
 * - Вы сами управляете жизненным циклом внешних OpenCL ресурсов
 * - При Cleanup() внешние объекты НЕ уничтожаются
 */
class OpenCLBackendExternal : public OpenCLBackend {
public:
    // ═══════════════════════════════════════════════════════════════
    // Конструкторы для внешнего контекста
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать backend с ВНЕШНИМ OpenCL контекстом
     * @param external_context Ваш существующий cl_context
     * @param external_device Ваш существующий cl_device_id
     * @param external_queue Ваш существующий cl_command_queue
     * @param owns_resources false (по умолчанию) - НЕ владеет ресурсами
     * 
     * ВАЖНО: DrvGPU НЕ уничтожит внешние ресурсы при Cleanup()
     */
    OpenCLBackendExternal(
        cl_context external_context,
        cl_device_id external_device,
        cl_command_queue external_queue,
        bool owns_resources = false
    );

    /**
     * @brief Деструктор (безопасный для внешних ресурсов)
     */
    ~OpenCLBackendExternal() override;

    // ═══════════════════════════════════════════════════════════════
    // Инициализация с внешним контекстом
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Инициализировать backend с внешним контекстом
     * @throws std::runtime_error если внешние объекты невалидны
     * 
     * Что делает:
     * 1. Проверяет валидность external_context, device, queue
     * 2. Инициализирует SVM capabilities для устройства
     * 3. Создает MemoryManager (работает с внешним контекстом)
     * 4. НЕ создает новый контекст - использует ваш
     */
    void InitializeWithExternalContext();

    // ═══════════════════════════════════════════════════════════════
    // Работа с ВНЕШНИМИ cl_mem буферами (КЛЮЧЕВАЯ ФУНКЦИЯ)
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Создать адаптер для работы с ВНЕШНИМ cl_mem буфером
     * @tparam T Тип элементов в буфере (float, int, double, etc.)
     * @param external_cl_mem Ваш существующий cl_mem буфер
     * @param num_elements Количество элементов типа T в буфере
     * @param owns_buffer false - НЕ владеет буфером (по умолчанию)
     * @return Shared pointer на адаптер
     * 
     * USE CASE: Обмен данными с внешними OpenCL буферами
     * 
     * @code
     * // У вас есть cl_mem буфер из другого класса
     * cl_mem your_buffer = external_class->GetBuffer();
     * 
     * // Создаем адаптер (тип float, 1024 элемента)
     * auto adapter = backend.CreateExternalBufferAdapter<float>(your_buffer, 1024);
     * 
     * // Загрузить данные с GPU в Host
     * std::vector<float> data = adapter->Read();
     * 
     * // Выгрузить данные с Host на GPU
     * std::vector<float> new_data(1024, 42.0f);
     * adapter->Write(new_data);
     * 
     * // Адаптер НЕ уничтожит your_buffer при удалении!
     * @endcode
     */
    template<typename T>
    std::shared_ptr<ExternalCLBufferAdapter<T>> CreateExternalBufferAdapter(
        cl_mem external_cl_mem,
        size_t num_elements,
        bool owns_buffer = false
    );

    /**
     * @brief Создать адаптер с явным размером в байтах
     * @tparam T Тип элементов
     * @param external_cl_mem Внешний cl_mem
     * @param size_bytes Размер буфера в байтах
     * @param owns_buffer false - НЕ владеет
     */
    template<typename T>
    std::shared_ptr<ExternalCLBufferAdapter<T>> CreateExternalBufferAdapterBytes(
        cl_mem external_cl_mem,
        size_t size_bytes,
        bool owns_buffer = false
    );

    // ═══════════════════════════════════════════════════════════════
    // Утилиты для работы с внешними буферами
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Скопировать данные Host -> External cl_mem (утилита)
     * @param external_cl_mem Целевой cl_mem буфер
     * @param host_data Данные с host
     * @param size_bytes Размер в байтах
     * @param blocking true - блокирующая операция (по умолчанию)
     */
    void WriteToExternalBuffer(
        cl_mem external_cl_mem,
        const void* host_data,
        size_t size_bytes,
        bool blocking = true
    );

    /**
     * @brief Скопировать данные External cl_mem -> Host (утилита)
     * @param external_cl_mem Источник cl_mem
     * @param host_dest Буфер на host для записи
     * @param size_bytes Размер в байтах
     * @param blocking true - блокирующая операция (по умолчанию)
     */
    void ReadFromExternalBuffer(
        cl_mem external_cl_mem,
        void* host_dest,
        size_t size_bytes,
        bool blocking = true
    );

    /**
     * @brief Скопировать данные между двумя cl_mem буферами
     * @param src_cl_mem Источник
     * @param dst_cl_mem Назначение
     * @param size_bytes Размер в байтах
     */
    void CopyExternalBuffers(
        cl_mem src_cl_mem,
        cl_mem dst_cl_mem,
        size_t size_bytes
    );

    // ═══════════════════════════════════════════════════════════════
    // Проверки
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Проверить, использует ли backend внешний контекст
     */
    bool IsExternalContext() const { return is_external_context_; }

    /**
     * @brief Проверить, владеет ли backend ресурсами
     */
    bool OwnsResources() const { return owns_resources_; }

    /**
     * @brief Override Cleanup для безопасной работы с внешними ресурсами
     */
    void Cleanup() override;

private:
    // ═══════════════════════════════════════════════════════════════
    // Члены класса
    // ═══════════════════════════════════════════════════════════════
    
    bool is_external_context_;  ///< true если используется внешний контекст
    bool owns_resources_;        ///< true если backend владеет ресурсами
    
    cl_context external_context_;
    cl_device_id external_device_;
    cl_command_queue external_queue_;

    // ═══════════════════════════════════════════════════════════════
    // Приватные методы
    // ═══════════════════════════════════════════════════════════════
    
    /**
     * @brief Валидация внешних OpenCL объектов
     */
    void ValidateExternalObjects();

    /**
     * @brief Получить информацию о внешнем cl_mem буфере
     */
    size_t GetBufferSize(cl_mem buffer) const;
};

// ════════════════════════════════════════════════════════════════════════════
// Template реализация
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
std::shared_ptr<ExternalCLBufferAdapter<T>> 
OpenCLBackendExternal::CreateExternalBufferAdapter(
    cl_mem external_cl_mem,
    size_t num_elements,
    bool owns_buffer)
{
    if (!external_cl_mem) {
        throw std::invalid_argument("CreateExternalBufferAdapter: external_cl_mem is null");
    }

    if (!IsInitialized()) {
        throw std::runtime_error("CreateExternalBufferAdapter: Backend not initialized");
    }

    // Создаем адаптер (передаем queue для операций)
    return std::make_shared<ExternalCLBufferAdapter<T>>(
        external_cl_mem,
        num_elements,
        external_queue_,
        owns_buffer
    );
}

template<typename T>
std::shared_ptr<ExternalCLBufferAdapter<T>> 
OpenCLBackendExternal::CreateExternalBufferAdapterBytes(
    cl_mem external_cl_mem,
    size_t size_bytes,
    bool owns_buffer)
{
    size_t num_elements = size_bytes / sizeof(T);
    return CreateExternalBufferAdapter<T>(external_cl_mem, num_elements, owns_buffer);
}

} // namespace drv_gpu_lib
