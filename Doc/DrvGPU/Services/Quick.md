# DrvGPU Services — Краткий справочник

> KernelCacheService, FilterConfigService, Storage Backend

---

## Сервисы

| Сервис | Назначение |
|--------|------------|
| **KernelCacheService** | On-disk кэш скомпилированных kernel (.cl + binary + manifest) |
| **FilterConfigService** | Сохранение/загрузка конфигов фильтров (FIR/IIR коэффициенты) |
| **IStorageBackend** | Абстракция хранения (FileStorageBackend → SQLite) |

---

## Быстрый старт

### KernelCacheService

```cpp
#include "services/kernel_cache_service.hpp"

drv_gpu_lib::KernelCacheService cache("modules/signal_generators/kernels");

// Save
cache.Save("my_kernel", cl_source, binary, "params", "comment");

// Load (binary fast path или source fallback)
auto entry = cache.Load("my_kernel");
if (entry.has_binary())
  LoadFromBinary(entry.binary);
else
  LoadFromSource(entry.source);

// List
auto names = cache.ListKernels();
```

### FilterConfigService

```cpp
#include "services/filter_config_service.hpp"

drv_gpu_lib::FilterConfigService svc("Results/FilterConfigs");

drv_gpu_lib::FilterConfigData data;
data.type = "fir";
data.coefficients = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};

svc.Save("lp_5000", data);
auto loaded = svc.Load("lp_5000");
```

---

## Структура на диске

```
base_dir/
├── name.cl              # KernelCacheService: source
├── bin/
│   └── name_opencl.bin  # binary (или _rocm.hsaco)
├── manifest.json        # индекс кернелов
└── filters/            # FilterConfigService
    └── lp_5000.json
```

---

## Ссылки

- [Full](Full.md) — полное описание, API, тесты
- [PLAN_KernelCacheService](../../../MemoryBank/tasks/PLAN_KernelCacheService_DrvGPU.md)

---

*Обновлено: 2026-02-23*
