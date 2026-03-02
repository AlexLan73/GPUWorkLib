# DrvGPU — Краткий справочник

> Ядро библиотеки GPUWorkLib

---

## Основные классы

| Класс | Назначение |
|-------|------------|
| `DrvGPU` | Фасад: инициализация, бэкенд, память |
| `IBackend` | Интерфейс бэкенда (OpenCL, ROCm, Hybrid) |
| `MemoryManager` | Управление GPU-буферами |
| `GPUBuffer<T>` | Owning RAII буфер (cl_mem) |
| `HIPBuffer<T>` | Non-owning HIP буфер (ROCm only) |
| `SVMBuffer` | SVM (Shared Virtual Memory) |
| `GPUManager` | Multi-GPU, load balancing |
| `ServiceManager` | Init/Start/Stop всех сервисов |
| `ConsoleOutput` | Мультиgpu-безопасный stdout |
| `GPUProfiler` | Профилирование, экспорт JSON/MD |
| **Storage Services** | [KernelCacheService](Services/Quick.md), FilterConfigService |

---

## Быстрый старт

```cpp
#include "DrvGPU/drv_gpu.hpp"
#include "services/service_manager.hpp"

// Инициализация сервисов
auto& sm = drv_gpu_lib::ServiceManager::GetInstance();
sm.InitializeFromConfig("configGPU.json");
sm.StartAll();

drv_gpu_lib::DrvGPU drv(drv_gpu_lib::BackendType::OPENCL, 0);
drv.Initialize();

IBackend& backend = drv.GetBackend();   // ссылка, не указатель!
MemoryManager& mem = drv.GetMemoryManager(); // ссылка, не указатель!

// Буфер (CreateBuffer вместо AllocateBuffer)
auto buf = mem.CreateBuffer<std::complex<float>>(4096);
buf->Write(host_data);
auto result = buf->Read();  // buf — RAII, освобождается при выходе из scope

drv.Synchronize();
sm.StopAll();
```

---

## Ссылки

- [Architecture](Architecture.md) — слои, паттерны
- [Memory](Memory.md) — GPUBuffer, SVMBuffer
- [OpenCL](OpenCL.md) — OpenCL бэкенд
- [Classes](Classes.md) — полный справочник
- [Services](Services/Quick.md) — KernelCache, FilterConfig, Storage

---

*Обновлено: 2026-03-02*
