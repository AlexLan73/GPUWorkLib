# DrvGPU/backends — Архитектура бэкендов

Абстракция GPU-бэкендов. `IBackend` (interface/) — единый интерфейс для OpenCL, ROCm, CUDA.

## Структура

| Папка | Описание |
|-------|----------|
| `opencl/` | OpenCL реализация (OpenCLBackend, OpenCLCore, CommandQueuePool) |
| `rocm/` | ROCm (планируется) |
| `cuda/` | CUDA (планируется) |

## IBackend (interface/i_backend.hpp)

- `Initialize()`, `Shutdown()`
- `CreateBuffer()`, `Allocate()`, `Free()`
- `GetDeviceInfo()`, `GetCommandQueue()`
- `InitializeFromExternalContext()` — внешний OpenCL-контекст

## Как тестировать

- **OpenCL**: `tests/single_gpu.hpp` — `example_drv_gpu_singl::run()`
- **Внешний контекст**: `tests/example_external_context_usage.hpp` — `external_context_example::run()`
