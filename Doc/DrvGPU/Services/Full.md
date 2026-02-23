# DrvGPU Services — Полная документация

> KernelCacheService, FilterConfigService, IStorageBackend

**Namespace**: `drv_gpu_lib`
**Каталог**: `DrvGPU/services/`
**Зависимости**: std::filesystem (C++17), nlohmann_json (FilterConfigService)

---

## Содержание

1. [Обзор](#1-обзор)
2. [IStorageBackend и FileStorageBackend](#2-istoragebackend-и-filestoragebackend)
3. [KernelCacheService](#3-kernelcacheservice)
4. [FilterConfigService](#4-filterconfigservice)
5. [Потребители](#5-потребители)
6. [Структура файлов](#6-структура-файлов)
7. [Тесты](#7-тесты)
8. [Ссылки](#8-ссылки)

---

## 1. Обзор

DrvGPU Services — инфраструктура для хранения данных модулей:

| Сервис | Что хранит | Ключи / формат |
|--------|------------|----------------|
| **KernelCacheService** | Скомпилированные OpenCL/ROCm kernel | `name.cl`, `bin/name_opencl.bin`, `manifest.json` |
| **FilterConfigService** | Конфигурации FIR/IIR фильтров | `filters/{name}.json` |
| **FileStorageBackend** | Универсальное key-value (бинарные данные) | `{key}` → файл в base_dir |

**Принцип:** Раздельные папки (base_dir) для каждого модуля. Один интерфейс IStorageBackend.

---

## 2. IStorageBackend и FileStorageBackend

### Интерфейс

```cpp
struct IStorageBackend {
  virtual void Save(const std::string& key, const std::vector<uint8_t>& data) = 0;
  virtual std::vector<uint8_t> Load(const std::string& key) const = 0;
  virtual std::vector<std::string> List(const std::string& prefix = "") const = 0;
  virtual bool Exists(const std::string& key) const = 0;
};
```

**Ключ** может содержать `/` — интерпретируется как поддиректория: `filters/lp_5000.json` → `base_dir/filters/lp_5000.json`.

### FileStorageBackend

- **Конструктор:** `FileStorageBackend(const std::string& base_dir)`
- **Save:** создаёт поддиректории при необходимости
- **Load:** выбрасывает `std::runtime_error` если ключ не найден
- Использует `std::filesystem` (C++17)

**Будущее:** SqliteStorageBackend с тем же интерфейсом.

---

## 3. KernelCacheService

### Назначение

On-disk кэш скомпилированных kernel. **Storage-agnostic** — не знает OpenCL, возвращает `{source, binary}`. Вызывающий создаёт `cl_program` через `clCreateProgramWithBinary` или `clCreateProgramWithSource`.

### API

```cpp
KernelCacheService(const std::string& base_dir,
                   BackendType backend_type = BackendType::OPENCL);

struct CacheEntry {
  std::string source;
  std::vector<uint8_t> binary;
  bool has_binary() const;
  bool has_source() const;
};

void Save(const std::string& name,
          const std::string& cl_source,
          const std::vector<uint8_t>& binary,
          const std::string& metadata = "",
          const std::string& comment = "");

CacheEntry Load(const std::string& name) const;  // throws if not found
std::vector<std::string> ListKernels() const;
std::string GetCacheDir() const;
std::string GetBinDir() const;
```

### Логика Save

1. `VersionOldFiles(name)` — если существуют `name.cl` и `name_*opencl.bin`, переименовать в `name_00.cl`, `name_opencl_00.bin`
2. Записать `base_dir/name.cl`
3. Записать `base_dir/bin/name_opencl.bin` (или `name_rocm.hsaco`)
4. Обновить `manifest.json`

### Логика Load

1. **Fast path:** `bin/name_opencl.bin` существует → прочитать binary + source из `name.cl`
2. **Fallback:** только `name.cl` → вернуть `{source, {}}` (вызывающий скомпилирует и вызовет Save)
3. Ничего нет → `std::runtime_error`

### ROCm

При `BackendType::ROCm` суффикс `_rocm.hsaco`. Логика идентична (байты в файл).

---

## 4. FilterConfigService

### Назначение

Сохранение/загрузка конфигураций фильтров (тип + коэффициенты). В отличие от kernel cache — хранит **данные фильтра**, не binary.

### FilterConfigData

```cpp
struct FilterConfigData {
  std::string name;
  std::string type;        // "fir" или "iir"
  std::string comment;
  std::string created;     // ISO 8601

  std::vector<float> coefficients;           // FIR: h[k]
  std::vector<std::array<float, 5>> sections; // IIR: [b0,b1,b2,a1,a2] per section
};
```

### API

```cpp
FilterConfigService(const std::string& base_dir);

void Save(const std::string& name, const FilterConfigData& data,
          const std::string& comment = "");
FilterConfigData Load(const std::string& name) const;
std::vector<std::string> ListFilters() const;
bool Exists(const std::string& name) const;
```

### Формат JSON

**FIR:**
```json
{
  "name": "lp_5000",
  "type": "fir",
  "comment": "Lowpass 5kHz",
  "created": "2026-02-21T12:00:00",
  "coefficients": [0.01, 0.02, ...]
}
```

**IIR:**
```json
{
  "name": "bp_1000",
  "type": "iir",
  "sections": [[b0,b1,b2,a1,a2], ...]
}
```

### Версионирование

При перезаписи того же имени → старый файл `lp_5000_00.json`, `lp_5000_01.json`, …

---

## 5. Потребители

| Модуль | Сервис | base_dir |
|--------|--------|----------|
| **FormScriptGenerator** | KernelCacheService | `modules/signal_generators/kernels` |
| **FirFilter** | KernelCacheService | `modules/filters/kernels` |
| **IirFilter** | KernelCacheService | `modules/filters/kernels` |
| **FilterConfigService** | FileStorageBackend | configurable |

**TASK-006 (планируется):** SaveFilterConfig/LoadFilterConfig в FirFilter/IirFilter через FilterConfigService.

---

## 6. Структура файлов

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

---

## 7. Тесты

**Файл:** `DrvGPU/tests/test_storage_services.hpp`

| Тест | Что проверяет |
|------|---------------|
| **TestFileStorageBackend** | Save/Load/List/Exists, поддиректории, перезапись |
| **TestKernelCacheService** | Save/Load/ListKernels, VersionOldFiles, GetBinDir |
| **TestFilterConfigService** | Save/Load FIR и IIR, ListFilters, VersionOldFiles, Exists |

**Вызов:** через `drvgpu_all_test::run()` в `main.cpp` (раскомментировать).

---

## 8. Ссылки

### План и таски

| Документ | Описание |
|----------|----------|
| [PLAN_KernelCacheService_DrvGPU.md](../../../MemoryBank/tasks/PLAN_KernelCacheService_DrvGPU.md) | План, архитектура, порядок |
| [DISCREPANCIES_KernelCacheService.md](../../../MemoryBank/tasks/DISCREPANCIES_KernelCacheService.md) | Разносласия, верификация |

### Референсы

| Источник | Применение |
|----------|------------|
| [Intel COMPILER_CACHE](https://github.com/intel/compute-runtime/blob/master/programmers-guide/COMPILER_CACHE.md) | Hash = source+options+device, eviction LRU |
| [gnieto/cl-cache](https://github.com/gnieto/cl-cache) | Backend abstraction |
| Khronos clCreateProgramWithBinary | Binary device-specific |

---

*Обновлено: 2026-02-23*
