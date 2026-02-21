# DrvGPU — Краткий справочник

> Ядро библиотеки GPUWorkLib

---

## Основные классы

| Класс | Назначение |
|-------|------------|
| `DrvGPU` | Фасад: инициализация, бэкенд, память |
| `IBackend` | Интерфейс бэкенда (OpenCL, ROCm) |
| `MemoryManager` | Управление GPU-буферами |
| `GPUBuffer` | Стандартный буфер (cl_mem) |
| `SVMBuffer` | SVM (Shared Virtual Memory) |
| `GPUManager` | Multi-GPU, load balancing |

---

## Быстрый старт

```cpp
#include "DrvGPU/drv_gpu.hpp"

drv_gpu_lib::DrvGPU drv(BackendType::OPENCL, 0);
drv.Initialize();

auto* backend = drv.GetBackend();
auto* mem = drv.GetMemoryManager();

// Буфер
auto buf = mem->AllocateBuffer(4096 * sizeof(std::complex<float>));
// ... работа с buf ...
mem->ReleaseBuffer(buf);
```

---

## Ссылки

- [Architecture](Architecture.md) — слои, паттерны
- [Memory](Memory.md) — GPUBuffer, SVMBuffer
- [OpenCL](OpenCL.md) — OpenCL бэкенд
- [Classes](Classes.md) — полный справочник

---

*Обновлено: 2026-02-17*
