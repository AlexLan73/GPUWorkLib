# DrvGPU/tests — Тесты и примеры

Тесты DrvGPU. Вызов через `all_test.hpp` из `main.cpp`.

## Файлы

| Файл | Namespace | Описание |
|------|-----------|----------|
| `all_test.hpp` | `drvgpu_all_test` | Оркестратор — включить/закомментировать тесты |
| `single_gpu.hpp` | `example_drv_gpu_singl` | Single GPU: DrvGPU, MemoryManager, device info |
| `multi_gpu.hpp` | — | Multi-GPU сценарии |
| `test_services.hpp` | `test_services` | ConsoleOutput, ServiceManager (многопоточные) |
| `test_gpu_profiler.hpp` | `test_gpu_profiler` | GPUProfiler: Record, агрегация, PrintSummary |
| `example_external_context_usage.hpp` | `external_context_example` | Внешний OpenCL-контекст |

## Как запускать

```cpp
// src/main.cpp
#include "DrvGPU/tests/all_test.hpp"
drvgpu_all_test::run();
```

В `all_test.hpp` раскомментировать нужные тесты:
```cpp
#include "single_gpu.hpp"
// #include "test_services.hpp"
// #include "test_gpu_profiler.hpp"
```

## Покрытие по компонентам

| Компонент | Тест |
|-----------|------|
| DrvGPU, MemoryManager, GPUDeviceInfo | single_gpu.hpp |
| GPUProfiler | test_gpu_profiler.hpp |
| ConsoleOutput, ServiceManager | test_services.hpp |
| OpenCLBackend (внешний контекст) | example_external_context_usage.hpp |
