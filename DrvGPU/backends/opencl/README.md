# DrvGPU/backends/opencl — OpenCL бэкенд

Реализация OpenCL для DrvGPU.

## Классы

| Файл | Класс | Описание |
|------|-------|----------|
| `opencl_backend.hpp/cpp` | `OpenCLBackend` | Реализация IBackend. Multi-GPU (per-device OpenCLCore). `Initialize()`, `InitializeFromExternalContext()` |
| `opencl_core.hpp/cpp` | `OpenCLCore` | Per-device контекст (context, device, platform) |
| `command_queue_pool.hpp/cpp` | `CommandQueuePool` | Пул command queues для параллельного выполнения |
| `opencl_profiling.hpp/cpp` | — | Утилиты профилирования OpenCL (events) |

## Использование

```cpp
OpenCLBackend backend;
backend.Initialize(0);  // GPU 0

// или внешний контекст
backend.InitializeFromExternalContext(context, device, queue);
```

## Как тестировать

- **OpenCLBackend**: `tests/single_gpu.hpp` — `example_drv_gpu_singl::run()`
- **Внешний контекст**: `tests/example_external_context_usage.hpp` — `external_context_example::run()`
