# План: KernelCacheService в DrvGPU

> **Источник:** TASK_FormSignal_Kernels_OnDiskCache_Refactor.md, шаг 3
> **Статус:** Документирован, ожидает реализации

---

## 1. Текущее состояние

On-disk cache **уже реализован** в `FormScriptGenerator` (modules/signal_generators).
Код: `form_script_generator.cpp`, строки 428-766, 603-718.

| Метод | Назначение | Зависимость от signal_generators |
|-------|-----------|----------------------------------|
| SaveKernel(name, comment) | Save .cl + binary + manifest | ParamsToString() — метаданные |
| LoadKernel(name) | Load binary (fast) or source | Нет |
| ListKernels() | Список из manifest.json | Нет |
| GetProgramBinary() | Извлечь binary из cl_program | Нет (OpenCL API) |
| LoadFromBinary(binary) | Создать cl_program из binary | Нет (OpenCL API) |
| VersionOldFiles(name) | Переименовать _00, _01... | Нет |
| WriteManifestEntry() | Обновить manifest.json | ParamsToString() — метаданные |

**Вывод:** Ядро кэша generic. Только `ParamsToString()` специфичен для signal_generators (пишет метаданные в manifest). Путь к директории задаётся через CMake define.

---

## 2. Целевая архитектура

```
DrvGPU/
└── services/
    ├── kernel_cache_service.hpp   # API
    └── kernel_cache_service.cpp   # Реализация
```

### API KernelCacheService

```cpp
namespace drv_gpu_lib {

class KernelCacheService {
public:
  // base_dir — корневая директория кэша (e.g. "modules/signal_generators/kernels")
  KernelCacheService(const std::string& base_dir);

  // Save compiled kernel (source + binary + manifest)
  void Save(const std::string& name,
            const std::string& cl_source,
            const std::vector<unsigned char>& binary,
            const std::string& metadata = "",
            const std::string& comment = "");

  // Load binary (fast path) or source (compile + cache)
  // Returns: {source, binary} — one or both may be empty
  struct CacheEntry {
    std::string source;
    std::vector<unsigned char> binary;
  };
  CacheEntry Load(const std::string& name) const;

  // List cached kernel names
  std::vector<std::string> ListKernels() const;

  // Get paths
  std::string GetCacheDir() const;
  std::string GetBinDir() const;

private:
  std::string base_dir_;
  void VersionOldFiles(const std::string& name) const;
  void WriteManifestEntry(const std::string& name,
                          const std::string& metadata,
                          const std::string& comment) const;
};

} // namespace drv_gpu_lib
```

**API-уточнение:** KernelCacheService остаётся storage-agnostic (не знает OpenCL). Возвращает `{source, binary}`; вызывающий сам создаёт `cl_program` через `clCreateProgramWithBinary` или `clCreateProgramWithSource`.

### Потребители

- **signal_generators** — FormScriptGenerator (текущий, рефакторинг)
- **filters** — FirFilter, IirFilter (обязательная интеграция)
- lch_farrow — при необходимости

---

## 3. Порядок переноса

1. Создать `DrvGPU/services/kernel_cache_service.hpp` + `.cpp`
2. Перенести логику из FormScriptGenerator (save/load/list/version/manifest)
3. Обновить FormScriptGenerator — использовать KernelCacheService
4. **Интегрировать filters**: FirFilter, IirFilter — Load/Save через KernelCacheService
5. Добавить тесты (C++ unit test)
6. Обновить CMakeLists.txt DrvGPU

---

## 4. Интеграция с filters

### Текущее поведение

- FirFilter::CompileKernel() — компиляция из GetFirDirectSource_opencl() при каждом создании
- IirFilter::CompileKernel() — аналогично

### Целевое поведение

1. При создании FirFilter/IirFilter:
   - KernelCacheService::Load("fir_filter_cf32") — попытка binary
   - При отсутствии: компиляция из source → Save в cache
2. Cache key: имя kernel (fir_filter_cf32, iir_filter_cf32) — source не зависит от коэффициентов
3. base_dir: `modules/filters/kernels` (или общая `kernels_cache/`)

### Изменения в filters

- FirFilter: внедрить KernelCacheService; CompileKernel() → Load from cache или compile + Save
- IirFilter: аналогично
- Сохранить fallback: при отсутствии cache — compile from source (как сейчас)

### Структура filters после интеграции

```
modules/filters/
├── kernels/
│   ├── bin/                    # создаётся KernelCacheService
│   │   ├── fir_filter_cf32_opencl.bin
│   │   └── iir_filter_cf32_opencl.bin
│   └── manifest.json
```

---

## 5. Оценка сложности

| Этап | Оценка |
|------|--------|
| KernelCacheService в DrvGPU | ~1 день |
| Рефакторинг FormScriptGenerator | ~0.5 дня |
| Интеграция filters (FirFilter, IirFilter) | ~1 день |
| Тесты | ~0.5 дня |

---

*Создано: 2026-02-18*
*Обновлено: 2026-02-21 — добавлена интеграция с filters*

---

## 6. Добавки (обсуждение)

### 6.1. ROCm везде

Заложить вызов под ROCm во всех компонентах:
- KernelCacheService: `Save`/`Load` с суффиксом `_opencl.bin` / `_rocm.hsaco`
- Backend-выбор при создании сервиса: `BackendType::OPENCL` | `BackendType::ROCm`
- ROCm: `hipModuleLoad` / `hipModuleLoadData` для загрузки .hsaco; `hiprtcCompileProgram` → `hiprtcGetCode` для компиляции

### 6.2. Фильтры: тип + коэффициенты (отличие от генераторов)

**Текущий flow (2.1):**
- Python формирует фильтр (scipy) → передаёт коэффициенты в C++ → C++ запускает kernel → результат → анализ

**Фиксация (2.2):**
- Новый метод: `SaveFilter(name, comment, coefficients, filter_type)` — доп. параметры к Process или отдельный вызов
- Сохранение: JSON (позже SQLite) — `{name, comment, type, coefficients, created}`
- Повторный вызов: `LoadFilter(name)` → коэффициенты → при необходимости запуск kernel
- Версионирование: при перезаписи того же имени → `name_00`, `name_01`, …

### 6.3. Абстрактный механизм read/write в DrvGPU/services

Генераторы и фильтры хранят разные данные:
- **Генераторы**: kernel source + binary + manifest (metadata)
- **Фильтры**: тип фильтра + коэффициенты + metadata (name, comment)

**Предложение — двухуровневая абстракция:**

```
DrvGPU/services/
├── storage/
│   ├── i_storage_backend.hpp      # Save(key, payload), Load(key), List(prefix), Exists(key)
│   ├── file_storage_backend.hpp   # JSON + bin в файловой системе
│   └── sqlite_storage_backend.hpp # Будущее: SQLite
├── kernel_cache_service.hpp       # Использует IStorageBackend для kernel binary + manifest
└── filter_config_service.hpp      # Использует IStorageBackend для coefficients JSON
```

**IStorageBackend:**
```cpp
struct IStorageBackend {
  virtual void Save(const std::string& key, const std::vector<uint8_t>& data) = 0;
  virtual std::vector<uint8_t> Load(const std::string& key) const = 0;
  virtual std::vector<std::string> List(const std::string& prefix = "") const = 0;
  virtual bool Exists(const std::string& key) const = 0;
};
```

**Специализации:**
- `KernelCacheService` — ключи: `{name}.bin`, `{name}.cl`, `manifest.json`
- `FilterConfigService` — ключи: `filters/{name}.json`; payload: JSON с type, coefficients, comment

**Эволюция:** FileStorage → SqliteStorage; ROCm: `*.hsaco` вместо `*.bin`

### 6.4. Источники (Context7 / Web / GitHub)

| Источник | Применение |
|----------|------------|
| [Intel COMPILER_CACHE](https://github.com/intel/compute-runtime/blob/master/programmers-guide/COMPILER_CACHE.md) | Hash = source+options+device; eviction LRU; .cl_cache |
| [gnieto/cl-cache](https://github.com/gnieto/cl-cache) | Backend abstraction; get_with_tag/put_with_tag |
| [ROCm hipModuleLoad](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/module_management.html) | .hsaco загрузка |
| Khronos clCreateProgramWithBinary | Binary device-specific |

---

## 7. Вопросы для обсуждения

1. **Единая база vs раздельные:** Один StorageBackend на всё или отдельные экземпляры с разными base_dir?
2. **Ключи для filters:** `filters/{name}.json` или плоский `{name}_filter.json`?
3. **FilterConfigService:** Отдельный сервис или метод в FirFilter/IirFilter + общий StorageBackend?
4. **SQLite:** Когда планируется — до или после ROCm?

### Ответы
1. я незнаю объясни разницу где и в чем +- напиши обсудим и твои рекомендации почеу
2.  **Ключи для filters:** `filters/{name}.json` - дымаю это хороший вариант
3. **FilterConfigService:** Отдельный сервис или метод ...  незнаю распиши и твои рекомендации почеу
4. **SQLite:** -ROCm - начнем сегодня а SQLite - через пару недель

---

## 8. Ответы Кодо (рекомендации)

### 8.1. Вопрос 1: Единая база vs раздельные StorageBackend

**Единая база:** Один экземпляр `FileStorageBackend` с `base_dir = "cache/"` или `"Results/KernelCache/"`. Все ключи в одной иерархии: `kernels/fir_filter_cf32.bin`, `filters/lp_5000.json`, `signal_gen/my_signal.cl`.

**Раздельные:** Каждый модуль — свой backend со своим `base_dir`: `modules/filters/kernels/`, `modules/signal_generators/kernels/`.

| Критерий | Единая | Раздельные |
|----------|--------|------------|
| Миграция на SQLite | Одна таблица или namespace | Отдельные таблицы/схемы по модулю |
| Резервное копирование | Один каталог/файл | Разбросано по модулям |
| Конфликты имён | Возможны (fir_filter в kernels и filters) | Меньше — изоляция по модулю |
| Конфигурация | Один путь в config | Каждый модуль задаёт свой путь |
| Порядок внедрения | Проще начать — один сервис | Нужно прокидывать base_dir в каждый модуль |

**Рекомендация:** **Раздельные** — `base_dir` передаётся при создании KernelCacheService/FilterConfigService. Причины: (1) уже так в FormScriptGenerator — `SIGNAL_GENERATORS_KERNELS_DIR`; (2) при переходе на SQLite — `filters`, `kernels`, `signal_gen` как отдельные таблицы; (3) меньше риска пересечения имён. Реализация: один класс `FileStorageBackend`, разные экземпляры с разными `base_dir`.

### 8.2. Вопрос 2: Ключи filters

Принято: `filters/{name}.json`.

### 8.3. Вопрос 3: FilterConfigService — отдельный сервис или метод в FirFilter/IirFilter

**Вариант A — отдельный сервис:** `FilterConfigService` в DrvGPU/services. FirFilter/IirFilter получают его по ссылке или через DI. Сохранение: `filter_config.Save("lp_5000", json_data)`; загрузка: `filter_config.Load("lp_5000")` → JSON → парсинг в FirFilter/IirFilter.

**Вариант B — метод в FirFilter/IirFilter:** `fir.SaveFilterConfig("lp_5000", comment)` и `fir.LoadFilterConfig("lp_5000")` внутри класса. Внутри используется общий `IStorageBackend` (или `FilterConfigService` как утилита).

**Рекомендация:** **Вариант A — отдельный FilterConfigService.** Причины: (1) единообразие с KernelCacheService; (2) один сервис может обслуживать FirFilter и IirFilter; (3) при SQLite — одна точка доступа к данным фильтров; (4) FirFilter/IirFilter остаются «чистыми» — только обработка, без логики хранения. API: `FilterConfigService::Save(name, FilterConfigData)`, `Load(name) → FilterConfigData`.

### 8.4. Вопрос 4: ROCm сегодня, SQLite через пару недель

Порядок: **сначала ROCm**, потом SQLite. Сейчас — FileStorageBackend; через пару недель — SqliteStorageBackend с тем же IStorageBackend.

### Ответы 
8.1. Я планировал одна база db а в ней разные таблицы + в db добавить установки на каздую gpu в отдельности
- а ты предлогаешь создавать разные db? - не понял ((
  в db планирую ратлицы к примеру tGPU - id, name, ....  configGPU(tGPU) is_log, is_prof,..  может быть tFIR() - id, name, коеф; tIIR() - id, name, коеф; и прочее так же и с генераторами отдельная db (а может в месте  все в одной db обсудим) для профилирования как то так.
8.3 - согласен A

---

## 9. Уточнение Кодо (про DB)

**Про «раздельные» — уточнение:** Я имел в виду **разные папки** (base_dir) при **файловом** хранении: `modules/filters/kernels/`, `modules/signal_generators/kernels/`. Это не про разные SQLite-базы.

**По SQLite — согласен с твоим планом:**
- **Одна БД** с разными таблицами: tGPU, configGPU, tFIR, tIIR, tGenerators, profiling и т.д.
- tGPU — id, name, …
- configGPU — привязка к tGPU: is_log, is_prof, is_console
- tFIR, tIIR — id, name, coefficients (JSON или отдельные колонки)
- Генераторы — в той же БД (отдельная таблица tGenerators или tKernels)
- Профилирование — свои таблицы

**Одна БД лучше:** один файл, транзакции, бэкапы, связи между таблицами (например, configGPU → tGPU). Раздельные БД имели бы смысл только при очень разных жизненных циклах (например, profiling — отдельная БД с ротацией), но для старта — одна БД ок.

**Итог:** Раздельные — только для FileStorage (папки). SQLite — одна БД, много таблиц. Согласна.

**ПОДТВЕРЖДЕНО (Alex):** Согласен — разные папки (раздельные base_dir).

---

## 10. ДЕТАЛЬНЫЕ ТАСКИ (для исполнения и приёмки)

| № | Таска | Описание | Проверка |
|---|-------|----------|----------|
| 001 | [TASK_KernelCacheService_001_StorageBackend.md](TASK_KernelCacheService_001_StorageBackend.md) | IStorageBackend + FileStorageBackend | Кодо |
| 002 | [TASK_KernelCacheService_002_KernelCacheService.md](TASK_KernelCacheService_002_KernelCacheService.md) | KernelCacheService в DrvGPU | Кодо |
| 003 | [TASK_KernelCacheService_003_FormScriptRefactor.md](TASK_KernelCacheService_003_FormScriptRefactor.md) | FormScriptGenerator → KernelCacheService | Кодо |
| 004 | [TASK_KernelCacheService_004_Filters.md](TASK_KernelCacheService_004_Filters.md) | FirFilter, IirFilter → KernelCacheService | Кодо |
| 005 | [TASK_KernelCacheService_005_FilterConfigService.md](TASK_KernelCacheService_005_FilterConfigService.md) | FilterConfigService (SaveFilter/LoadFilter) | Кодо |

**Порядок:** 001 → 002 → (003 и 004 параллельно) → 005.