#pragma once

#include "memory_type.hpp"
#include <CL/cl.h>
#include <memory>
#include <vector>
#include <complex>
#include <string>

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// Класс GPUMemoryBuffer - обёртка над GPU памятью
// ════════════════════════════════════════════════════════════════════════════

class GPUMemoryBuffer {
public:
    // === Конструкторы ===

    // 1. OWNING конструктор - объект СОЗДАЁТ новый буфер
    GPUMemoryBuffer(
        cl_context context,
        cl_command_queue queue,
        size_t num_elements,
        MemoryType type = MemoryType::GPU_WRITE_ONLY
    );

    // 2. NON-OWNING конструктор - объект использует ГОТОВЫЙ буфер
    GPUMemoryBuffer(
        cl_context context,
        cl_command_queue queue,
        cl_mem external_gpu_buffer,
        size_t num_elements,
        MemoryType type = MemoryType::GPU_WRITE_ONLY
    );

    // 3. OWNING конструктор с данными - создаёт буфер и копирует данные (CL_MEM_COPY_HOST_PTR)
    GPUMemoryBuffer(
        cl_context context,
        cl_command_queue queue,
        const void* host_data,
        size_t data_size_bytes,
        size_t num_elements,
        MemoryType type = MemoryType::GPU_READ_ONLY
    );

    // Деструктор
    ~GPUMemoryBuffer();

    // === Запрет копирования (только move) ===
    GPUMemoryBuffer(const GPUMemoryBuffer&) = delete;
    GPUMemoryBuffer& operator=(const GPUMemoryBuffer&) = delete;

    GPUMemoryBuffer(GPUMemoryBuffer&& other) noexcept;
    GPUMemoryBuffer& operator=(GPUMemoryBuffer&& other) noexcept;

    // === Операции чтения/записи ===

    // Прочитать ВСЕ данные с GPU
    std::vector<std::complex<float>> ReadFromGPU();

    // Прочитать ЧАСТЬ данных с GPU (быстрее)
    std::vector<std::complex<float>> ReadPartial(size_t num_elements);

    // Записать данные на GPU
    void WriteToGPU(const std::vector<std::complex<float>>& data);

    // === Асинхронные операции (с cl_event) ===

    /**
     * @brief Асинхронно прочитать все данные с GPU
     * @return Пара (результат_вектор, event)
     */
    std::pair<std::vector<std::complex<float>>, cl_event>
    ReadFromGPUAsync();

    /**
     * @brief Асинхронно записать данные на GPU
     * @return cl_event для синхронизации
     */
    cl_event WriteToGPUAsync(const std::vector<std::complex<float>>& data);

    // === Информация о буфере ===

    // Количество элементов
    size_t GetNumElements() const { return num_elements_; }

    // Размер в байтах
    size_t GetSizeBytes() const {
        return buffer_size_bytes_;
    }

    // Является ли буфер внешним (non-owning)
    bool IsExternalBuffer() const { return is_external_buffer_; }

    // Грязный ли буфер (нужно читать)
    bool IsGPUDirty() const { return gpu_dirty_; }

    // === Тип памяти ===
    MemoryType GetMemoryType() const { return type_; }

    // === Получить cl_mem для использования в OpenCL API ===
    cl_mem Get() const { return gpu_buffer_; }

    // === Статистика ===
    void PrintStats() const;

private:
    // === Члены класса ===
    cl_context context_;
    cl_command_queue queue_;
    cl_mem gpu_buffer_;
    std::vector<std::complex<float>> pinned_host_buffer_;

    size_t num_elements_;
    size_t buffer_size_bytes_;
    MemoryType type_;

    // Флаг владения буфером (важно!)
    bool is_external_buffer_;

    // Флаг грязности GPU буфера
    bool gpu_dirty_;

    // === Приватные методы ===

    // Создать GPU буфер
    void AllocateGPUBuffer();

    // Создать pinned host буфер для быстрого DMA
    void AllocatePinnedHostBuffer();

    // Освободить pinned host буфер
    void ReleasePinnedHostBuffer();

    // Проверить ошибку OpenCL
    static void CheckCLError(cl_int error, const std::string& operation);
};

} // namespace ManagerOpenCL

