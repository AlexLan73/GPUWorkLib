#pragma once

/**
 * @file all_test.hpp
 * @brief Перечень тестов и примеров DrvGPU
 *
 * main.cpp вызывает этот файл — НЕ отдельные тесты напрямую.
 * Включить/закомментировать нужные тесты здесь.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "single_gpu.hpp"
// #include "example_external_context_usage.hpp"
#include "test_services.hpp"
#include "test_gpu_profiler.hpp"
#include "test_storage_services.hpp"
#if ENABLE_ROCM
#include "test_rocm_backend.hpp"
#include "test_zero_copy.hpp"
#include "test_hybrid_backend.hpp"
#endif

namespace drvgpu_all_test {

inline void run() {
    // Пример: Single GPU — базовый GPU + память
    example_drv_gpu_singl::run();

    // Пример: внешний OpenCL контекст
    // external_context_example::run();

    // Services: многопоточные тесты (ConsoleOutput, AsyncService, ServiceManager)
    test_services::run();

    // GPUProfiler: Record, агрегация, PrintSummary
    test_gpu_profiler::run();

    // Storage Services: FileStorageBackend, KernelCacheService, FilterConfigService
    test_storage_services::run();

    // ROCm Backend: Initialize, Allocate, Memcpy, Synchronize
#if ENABLE_ROCM
    test_rocm_backend::run();
    test_zero_copy::run();
    test_hybrid_backend::run();
#endif
}

}  // namespace drvgpu_all_test
