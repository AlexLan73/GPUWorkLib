# План: KernelCacheService в DrvGPU

> **Статус:** Реализовано (2026-02-23)
> **Документация:** [Doc/DrvGPU/Services/](../../Doc/DrvGPU/Services/Quick.md)

---

## Реализовано

- **IStorageBackend + FileStorageBackend** — абстракция хранения
- **KernelCacheService** — on-disk кэш (.cl + binary + manifest.json), ROCm суффикс
- **FilterConfigService** — сохранение конфигов FIR/IIR в JSON
- **FormScriptGenerator** → KernelCacheService
- **FirFilter, IirFilter** → KernelCacheService

---

## Архитектура (итог)

```
DrvGPU/services/
├── storage/
│   ├── i_storage_backend.hpp
│   ├── file_storage_backend.hpp
│   └── file_storage_backend.cpp
├── kernel_cache_service.hpp
├── kernel_cache_service.cpp
├── filter_config_service.hpp
└── filter_config_service.cpp
```

**Раздельные base_dir:** `modules/signal_generators/kernels/`, `modules/filters/kernels/`

---

## Планы (на будущее)

### ROCm
- Суффикс `_rocm.hsaco` заложен в KernelCacheService
- hipModuleLoad / hiprtcCompileProgram — при порте на ROCm

### SQLite
- **Одна БД** с таблицами: tGPU, configGPU, tFIR, tIIR, tGenerators, profiling
- SqliteStorageBackend с тем же IStorageBackend
- Порядок: ROCm → SQLite

### TASK-006
- SaveFilterConfig/LoadFilterConfig в FirFilter/IirFilter через FilterConfigService

### Инвалидация кэша при смене драйвера
- См. [DiscussionPlan/~6. KernelCache/Driver_Invalidation_Note.md](../DiscussionPlan/~6. KernelCache/Driver_Invalidation_Note.md)
- Пока отложено

---

## Референсы

| Источник | Применение |
|----------|------------|
| [Intel COMPILER_CACHE](https://github.com/intel/compute-runtime/blob/master/programmers-guide/COMPILER_CACHE.md) | Hash, eviction LRU |
| [gnieto/cl-cache](https://github.com/gnieto/cl-cache) | Backend abstraction |
| ROCm hipModuleLoad | .hsaco загрузка |

---

*Обновлено: 2026-02-23*
