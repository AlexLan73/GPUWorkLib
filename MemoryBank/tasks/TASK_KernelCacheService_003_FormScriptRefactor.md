# TASK-003: Рефакторинг FormScriptGenerator — использование KernelCacheService

> **План:** [PLAN_KernelCacheService_DrvGPU.md](PLAN_KernelCacheService_DrvGPU.md)  
> **Зависимость:** TASK-002 (KernelCacheService)  
> **Связано:** [TASK_FormSignal_Kernels_OnDiskCache_Refactor.md](TASK_FormSignal_Kernels_OnDiskCache_Refactor.md) — шаги 1, 2, 4 остаются  
> **Проверка:** Кодо (старшая)

---

## 1. ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ

| № | Файл | Зачем |
|---|------|-------|
| 1 | `modules/signal_generators/src/form_script_generator.cpp` | Текущая реализация SaveKernel, LoadKernel |
| 2 | `modules/signal_generators/include/generators/form_script_generator.hpp` | Публичный API |
| 3 | `DrvGPU/services/kernel_cache_service.hpp` | API KernelCacheService (после TASK-002) |
| 4 | `modules/signal_generators/tests/test_form_script.hpp` | Тесты SaveKernel, LoadKernel |

---

## 2. ЦЕЛЬ

Заменить внутреннюю логику SaveKernel/LoadKernel в FormScriptGenerator на вызовы KernelCacheService. Удалить дублирующий код (VersionOldFiles, WriteManifestEntry, GetProgramBinary, LoadFromBinary — часть остаётся, т.к. работа с cl_program).

---

## 3. ИЗМЕНЕНИЯ В FormScriptGenerator

### 3.1. Добавить член

```cpp
std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;
```

Или `KernelCacheService kernel_cache_` (если не нужен optional).

Инициализация в конструкторе:
```cpp
kernel_cache_ = std::make_unique<drv_gpu_lib::KernelCacheService>(
    GetKernelsDir(), drv_gpu_lib::BackendType::OPENCL);
```

### 3.2. SaveKernel(name, comment)

**Было:** Ручная запись .cl, binary, manifest, VersionOldFiles.  
**Стало:**
1. `auto binary = GetProgramBinary();` — оставить (работа с cl_program)
2. `kernel_cache_->Save(name, kernel_source_, binary, ParamsToString(), comment);`
3. Удалить локальные VersionOldFiles, WriteManifestEntry, запись файлов.

### 3.3. LoadKernel(name)

**Было:** Ручное чтение bin_path, cl_path, LoadFromBinary, LoadFromSource.  
**Стало:**
1. `auto entry = kernel_cache_->Load(name);`
2. Если `entry.has_binary()`: `LoadFromBinary(entry.binary);`
3. Иначе если `entry.has_source()`: `LoadFromSource(entry.source);` затем `Save` binary (как fallback в текущей логике)
4. `kernel_source_ = entry.source;` (если есть)
5. `loaded_kernel_name_ = name;`
6. Удалить локальную логику чтения файлов.

### 3.4. ListKernels()

**Было:** Парсинг manifest.json вручную.  
**Стало:** `return kernel_cache_->ListKernels();`

### 3.5. Удалить из FormScriptGenerator

- `VersionOldFiles` (private)
- `WriteManifestEntry` (private)
- Логику записи/чтения в GetKernelsDir(), GetKernelsBinDir() — пути теперь внутри KernelCacheService
- `GetProgramBinary`, `LoadFromBinary`, `LoadFromSource` — **оставить** (работа с OpenCL)

---

## 4. ЗАВИСИМОСТИ

FormScriptGenerator должен линковаться с DrvGPU (уже линкуется). Добавить `#include "services/kernel_cache_service.hpp"`.

---

## 5. ТЕСТЫ

**Существующие тесты** в `test_form_script.hpp` должны пройти без изменений:
- TestSaveKernel — проверка файлов на диске
- TestLoadKernel — загрузка и генерация
- TestVersioning — повторный SaveKernel → _00
- TestListKernels

Если тесты проверяют структуру manifest или пути — убедиться, что KernelCacheService создаёт ту же структуру (manifest.json, bin/*.bin, *.cl).

---

## 6. КРИТЕРИИ ПРИЁМКИ

- [ ] `cmake -B build && cmake --build build` — успешно
- [ ] `test_form_script::run()` — все тесты проходят (SaveKernel, LoadKernel, Versioning, ListKernels)
- [ ] Python: `pytest Python_test/test_form_signal.py -v` — проходят
- [ ] Python: `pytest Python_test/test_form_script.py` (если есть) — проходят
- [ ] Дублирующий код (VersionOldFiles, WriteManifestEntry, ручная запись) удалён из FormScriptGenerator
- [ ] Поведение идентично до рефакторинга (обратная совместимость)

---

## 7. ОТЧЁТ

```
✅ TASK-003 выполнено:
- FormScriptGenerator использует KernelCacheService
- Удалено: [список методов/строк]
- Тесты: [результаты]

Проверь (Кодо): компиляция, test_form_script, Python тесты.
```
