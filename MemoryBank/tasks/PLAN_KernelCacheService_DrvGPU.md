# План: KernelCacheService в DrvGPU

> **Источник:** TASK_FormSignal_Kernels_OnDiskCache_Refactor.md, шаг 3
> **Статус:** Документирован, ожидает реализации

---

## 1. Текущее состояние

On-disk cache реализован в `FormScriptGenerator` (`form_script_generator.cpp`):

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

### Потребители
- `signal_generators` — FormScriptGenerator (текущий)
- `filters` — при проектировании FIR/IIR (будущее)
- `lch_farrow` — если потребуется кэширование (будущее)

---

## 3. Порядок переноса

1. Создать `DrvGPU/services/kernel_cache_service.hpp` + `.cpp`
2. Перенести логику из FormScriptGenerator (save/load/list/version/manifest)
3. Обновить FormScriptGenerator — использовать KernelCacheService
4. Добавить тесты (C++ unit test)
5. Обновить CMakeLists.txt DrvGPU

---

## 4. Оценка сложности

**Низкая** — перенос чистого кода без изменения логики. ~200 строк.

---

*Создано: 2026-02-18*
