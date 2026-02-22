# TASK-002: KernelCacheService в DrvGPU

> **План:** [PLAN_KernelCacheService_DrvGPU.md](PLAN_KernelCacheService_DrvGPU.md)  
> **Зависимость:** TASK-001 (IStorageBackend, FileStorageBackend)  
> **Проверка:** Кодо (старшая)

---

## 1. ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ

| № | Файл | Зачем |
|---|------|-------|
| 1 | `PLAN_KernelCacheService_DrvGPU.md` | Разделы 1, 2, 3 — API, порядок |
| 2 | `modules/signal_generators/src/form_script_generator.cpp` | Строки 428-766, 603-718 — логика для переноса |
| 3 | `DrvGPU/services/storage/i_storage_backend.hpp` | Интерфейс (после TASK-001) |

---

## 2. ЦЕЛЬ

Перенести on-disk cache из FormScriptGenerator в DrvGPU как `KernelCacheService`. Сервис **storage-agnostic** — не знает OpenCL, возвращает `{source, binary}`. Вызывающий сам создаёт `cl_program`.

**base_dir:** Передаётся в конструктор. Разные модули — разные папки (раздельные).

---

## 3. API KernelCacheService

**Файл:** `DrvGPU/services/kernel_cache_service.hpp`

```cpp
#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace drv_gpu_lib {

class KernelCacheService {
public:
  /**
   * @param base_dir Корневая директория кэша (e.g. "modules/signal_generators/kernels")
   * @param backend_type OPENCL или ROCm — суффикс файла: _opencl.bin / _rocm.hsaco
   */
  KernelCacheService(const std::string& base_dir, BackendType backend_type = BackendType::OPENCL);

  struct CacheEntry {
    std::string source;    // .cl содержимое
    std::vector<uint8_t> binary;  // скомпилированный binary
    bool has_binary() const { return !binary.empty(); }
    bool has_source() const { return !source.empty(); }
  };

  void Save(const std::string& name,
            const std::string& cl_source,
            const std::vector<uint8_t>& binary,
            const std::string& metadata = "",
            const std::string& comment = "");

  CacheEntry Load(const std::string& name) const;

  std::vector<std::string> ListKernels() const;

  std::string GetCacheDir() const;
  std::string GetBinDir() const;

private:
  std::string base_dir_;
  BackendType backend_type_;
  std::string GetBinarySuffix() const;  // "_opencl.bin" или "_rocm.hsaco"
  void VersionOldFiles(const std::string& name) const;
  void WriteManifestEntry(const std::string& name,
                          const std::string& metadata,
                          const std::string& comment) const;
};

} // namespace drv_gpu_lib
```

---

## 4. ЛОГИКА (перенос из FormScriptGenerator)

### 4.1. Save(name, cl_source, binary, metadata, comment)

1. `VersionOldFiles(name)` — если существуют `name.cl` и `name_*opencl.bin`, переименовать в `name_00.cl`, `name_opencl_00.bin`
2. Записать `base_dir/name.cl` — cl_source
3. Записать `base_dir/bin/name_opencl.bin` (или `name_rocm.hsaco`) — binary
4. `WriteManifestEntry` — обновить `base_dir/manifest.json`

**manifest.json формат:** как в FormScriptGenerator — `{version, kernels: [{name, comment, created, params: metadata, backend}]}`

### 4.2. Load(name)

1. **Fast path:** Проверить `base_dir/bin/name_opencl.bin`. Если есть — прочитать binary, прочитать source из `base_dir/name.cl` (если есть). Вернуть CacheEntry{binary, source}.
2. **Fallback:** Проверить `base_dir/name.cl`. Если есть — прочитать source. Вернуть CacheEntry{source} (binary пустой — вызывающий скомпилирует и вызовет Save).
3. Если ничего нет — выбросить `std::runtime_error`.

### 4.3. ListKernels()

Парсить `base_dir/manifest.json`, извлечь все `"name"` из массива `kernels`. Простой парсинг (без nlohmann_json — или использовать, если уже в проекте).

### 4.4. VersionOldFiles(name)

Как в form_script_generator.cpp строки 603-632: найти следующий свободный суффикс _00, _01, … переименовать существующие файлы.

---

## 5. РЕАЛИЗАЦИЯ

**Вариант A:** KernelCacheService использует `FileStorageBackend` внутри (композиция).  
**Вариант B:** KernelCacheService сам пишет файлы (как FormScriptGenerator сейчас).

**Рекомендация:** Вариант A — использовать FileStorageBackend. Ключи: `name.cl`, `bin/name_opencl.bin`, `manifest.json`. Но manifest — особый формат (JSON с массивом), возможно проще писать напрямую. Допустимо: KernelCacheService для manifest и .cl/.bin использовать низкоуровневую запись (как сейчас), а FileStorageBackend — для FilterConfigService. Или: KernelCacheService принимает `IStorageBackend*` в конструкторе — тогда гибко.

**Упрощение:** KernelCacheService пока без IStorageBackend — прямая работа с файлами (как FormScriptGenerator). IStorageBackend будет использоваться FilterConfigService. Это уменьшает связность на первом этапе.

**Итог:** KernelCacheService — прямая работа с `base_dir` через std::filesystem. Без внедрения IStorageBackend в эту таску (оставим для FilterConfigService).

---

## 6. ROCm

Заложить вызов: `backend_type_` определяет суффикс `_opencl.bin` или `_rocm.hsaco`. Реализация Load/Save для ROCm binary — заглушка (пока только OpenCL). При `BackendType::ROCm` — писать/читать `*_rocm.hsaco`, но логика идентична (байты в файл).

---

## 7. СТРУКТУРА

```
DrvGPU/services/
├── kernel_cache_service.hpp
└── kernel_cache_service.cpp
```

---

## 8. CMakeLists.txt

Добавить `kernel_cache_service.cpp` в DrvGPU target. Проверить: `#include "common/backend_type.hpp"`.

---

## 9. ТЕСТ

**Файл:** `DrvGPU/tests/test_kernel_cache_service.hpp`

1. Создать `KernelCacheService("Results/TestKernelCache", BackendType::OPENCL)`
2. Подготовить тестовые данные: source = "// test", binary = {0x01, 0x02, 0x03}
3. `Save("test_kernel", source, binary, "meta", "comment")`
4. Проверить: существуют `Results/TestKernelCache/test_kernel.cl`, `Results/TestKernelCache/bin/test_kernel_opencl.bin`, `manifest.json`
5. `Load("test_kernel")` → CacheEntry с тем же source и binary
6. `ListKernels()` → содержит "test_kernel"
7. Повторный `Save("test_kernel", ...)` → VersionOldFiles сработал, старые файлы _00

---

## 10. КРИТЕРИИ ПРИЁМКИ

- [ ] Компиляция успешна
- [ ] KernelCacheService не зависит от OpenCL (только backend_type для суффикса)
- [ ] Save/Load/ListKernels работают
- [ ] VersionOldFiles при перезаписи
- [ ] Тест test_kernel_cache_service проходит
- [ ] ROCm суффикс _rocm.hsaco заложен (при BackendType::ROCm)

---

## 11. ОТЧЁТ

```
✅ TASK-002 выполнено:
- [файлы]
- [сборка]
- [тест]

Проверь (Кодо): компиляция, тест, соответствие PLAN.
```
