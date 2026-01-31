#pragma once

#include "ManagerOpenCL/opencl_manager.h"
#include "ManagerOpenCL/memory_type.hpp"
#include "ManagerOpenCL/gpu_memory_buffer.hpp"
#include <memory>
#include <vector>
#include <complex>
#include <stdexcept>
#include <iostream>
#include <CL/cl.h>

namespace ManagerOpenCL {

// ════════════════════════════════════════════════════════════════════════════
// Синглтон GPUMemoryManager - управляет всеми буферами
// ════════════════════════════════════════════════════════════════════════════

class GPUMemoryManager {
public:
    // === Инициализация (один раз) ===
    static void Initialize();

    // === Создание буферов ===

    // Создать НОВЫЙ буфер (объект ВЛАДЕЕТ памятью)
    static std::unique_ptr<GPUMemoryBuffer> CreateBuffer(
        size_t num_elements,
        MemoryType type = MemoryType::GPU_WRITE_ONLY
    );

    // Обернуть ГОТОВЫЙ буфер (объект НЕ владеет памятью)
    static std::unique_ptr<GPUMemoryBuffer> WrapExternalBuffer(
        cl_mem external_gpu_buffer,
        size_t num_elements,
        MemoryType type = MemoryType::GPU_WRITE_ONLY
    );

    // === Статистика ===
    static void PrintStatistics();

    // === Доступ к синглтону ===
    static GPUMemoryManager& GetInstance();
    ~GPUMemoryManager() = default;

private:
    // === Синглтон (приватный конструктор) ===
    GPUMemoryManager();

    // Запрет копирования
    GPUMemoryManager(const GPUMemoryManager&) = delete;
    GPUMemoryManager& operator=(const GPUMemoryManager&) = delete;

    // === Члены класса ===
    static std::unique_ptr<GPUMemoryManager> instance_;
    static bool initialized_;

    cl_context context_;
    cl_command_queue queue_;

    size_t total_allocated_bytes_;
    size_t num_buffers_;
};

} // namespace ManagerOpenCL
