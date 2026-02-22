# TASK-001: IStorageBackend + FileStorageBackend

> **План:** [PLAN_KernelCacheService_DrvGPU.md](PLAN_KernelCacheService_DrvGPU.md)  
> **Приоритет:** 1 (фундамент)  
> **Проверка:** Кодо (старшая) — компиляция, тесты, соответствие спецификации

---

## 1. ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ

| № | Файл | Зачем |
|---|------|-------|
| 1 | `CLAUDE.md` | Правила проекта, структура |
| 2 | `PLAN_KernelCacheService_DrvGPU.md` | Разделы 6.3, 8.1, 9 — абстракция, раздельные папки |
| 3 | `DrvGPU/services/` | Текущая структура services |

---

## 2. ЦЕЛЬ

Создать абстрактный механизм read/write для DrvGPU. Разные модули (kernels, filters) будут использовать разные **папки** (base_dir), но один интерфейс.

**Подтверждено:** Раздельные папки — `modules/filters/kernels/`, `modules/signal_generators/kernels/`.

---

## 3. ТРЕБОВАНИЯ

### 3.1. IStorageBackend (интерфейс)

**Файл:** `DrvGPU/services/storage/i_storage_backend.hpp`

```cpp
#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace drv_gpu_lib {

/**
 * @brief Абстрактный backend для хранения данных (файлы, SQLite)
 * Разные экземпляры — разные base_dir (раздельные папки).
 */
struct IStorageBackend {
  virtual ~IStorageBackend() = default;

  virtual void Save(const std::string& key, const std::vector<uint8_t>& data) = 0;
  virtual std::vector<uint8_t> Load(const std::string& key) const = 0;
  virtual std::vector<std::string> List(const std::string& prefix = "") const = 0;
  virtual bool Exists(const std::string& key) const = 0;
};

} // namespace drv_gpu_lib
```

### 3.2. FileStorageBackend (реализация)

**Файл:** `DrvGPU/services/storage/file_storage_backend.hpp` + `.cpp`

- **Конструктор:** `FileStorageBackend(const std::string& base_dir)`
- **Save(key, data):** Записать в `base_dir/key`. Создать поддиректории при необходимости (например, `filters/lp_5000.json` → `base_dir/filters/lp_5000.json`).
- **Load(key):** Прочитать файл. Выбросить `std::runtime_error` если не существует.
- **List(prefix):** Сканировать `base_dir/` (и поддиректории при prefix), вернуть ключи (относительные пути).
- **Exists(key):** Проверить наличие файла.

**Детали:**
- Ключ может содержать `/` — интерпретировать как путь (filters/name.json).
- Использовать `std::filesystem` (C++17).
- Потокобезопасность: не требуется на уровне backend (вызывающий синхронизирует при необходимости).

---

## 4. СТРУКТУРА ФАЙЛОВ

```
DrvGPU/
└── services/
    └── storage/
        ├── i_storage_backend.hpp
        ├── file_storage_backend.hpp
        └── file_storage_backend.cpp
```

---

## 5. CMakeLists.txt

Добавить в `DrvGPU/CMakeLists.txt`:
- `services/storage/file_storage_backend.cpp` в соответствующий target (drvgpu или отдельная lib).

---

## 6. ТЕСТ

**Файл:** `DrvGPU/tests/test_storage_backend.hpp`

Минимальный тест:
1. Создать `FileStorageBackend("Results/TestStorage")`
2. `Save("test/key.bin", {1,2,3})`
3. `Exists("test/key.bin")` → true
4. `Load("test/key.bin")` → {1,2,3}
5. `List("test/")` → содержит "test/key.bin" (или "key.bin" в зависимости от реализации List)
6. Удалить тестовую папку после теста (или использовать временную)

Включить в `DrvGPU/tests/all_test.hpp` (закомментировано или по флагу).

---

## 7. КРИТЕРИИ ПРИЁМКИ (для проверки Кодо)

- [ ] `cmake -B build && cmake --build build` — успешно
- [ ] Файлы созданы: `i_storage_backend.hpp`, `file_storage_backend.hpp`, `file_storage_backend.cpp`
- [ ] Тест `test_storage_backend` проходит
- [ ] Интерфейс IStorageBackend соответствует спецификации
- [ ] FileStorageBackend создаёт поддиректории при Save (key с `/`)
- [ ] Нет зависимостей от OpenCL, signal_generators, filters

---

## 8. ОТЧЁТ ВЫПОЛНИТЕЛЮ

После выполнения:

```
✅ TASK-001 выполнено:
- [список созданных файлов]
- [результат сборки]
- [результат теста]

Проверь, пожалуйста (Кодо):
1. Компиляция
2. Тест test_storage_backend
3. Соответствие PLAN
```
