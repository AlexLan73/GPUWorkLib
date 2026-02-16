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
// #include "test_services.hpp"
// #include "test_gpu_profiler.hpp"

namespace drvgpu_all_test {

inline void run() {
    // Пример: Single GPU
    // example_drv_gpu_singl::run();

    // Пример: внешний OpenCL контекст
    // external_context_example::run();

    // Services: многопоточные тесты
    // test_services::run();

    // GPUProfiler: Record, агрегация, PrintSummary
    // test_gpu_profiler::run();
}

}  // namespace drvgpu_all_test
