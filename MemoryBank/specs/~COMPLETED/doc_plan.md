---
name: Docs
---

# План: Profiler MD + API Docs

## Контекст

После API Refactoring (api_refactoring.md) нужно:

- Создать документацию по использованию интерфейсов

---



## 6. Документация: полное описание использования API

**Создать**: `MemoryBank/specs/spectrum_maxima_api_usage.md` (или `Doc/`)

**Содержание**:

### 6.1 Интерфейс передачи сигнала

- `InputData<T>`: antenna_count, n_point, data
- Поддерживаемые типы T: `vector`, `cl_mem`, `void*` (SVM)

### 6.2 Интерфейсы получения результата

- 1 пик: `PeakSearchMode::ONE_PEAK` → 4 MaxValue на луч
- 2 пика: `PeakSearchMode::TWO_PEAKS` → 8 MaxValue на луч

### 6.3 Примеры

**Пример 1: CPU вектор + инициализация DrvGPU**

```cpp
DrvGPU gpu(BackendType::OPENCL, 0);
gpu.Initialize();

SpectrumMaximaFinder finder(&gpu.GetBackend());
InputData<std::vector<std::complex<float>>> input{5, 100000, my_data};
ProcessingParams params{.repeat_count = 4, .sample_rate = 1000.0f};
auto results = finder.Process(input, params, PeakSearchMode::ONE_PEAK);
```

**Пример 2: Внешний OpenCL — передача контекста**

- Использование `OpenCLBackendExternal` с `cl_context`, `cl_device_id`, `cl_command_queue`
- `InputData<cl_mem>` с буфером от заказчика

**Пример 3: Выбор OpenCL vs ROCm**

- `DriverType::OPENCL` (по умолчанию) — ROCm если доступен
- `DriverType::OPENCL` — принудительно OpenCL
- `DriverType::ROCM` — принудительно ROCm (когда реализовано)

### 6.4 Куда записываются результаты

- `Process()` возвращает `std::vector<SpectrumResult>`
- Каждый элемент — `SpectrumResult` с полями: antenna_id, center_point, interpolated и т.д.

### 6.5 Логи

- Формат: `Logs/DRVGPU_XX/YYYY-MM-DD/HH-MM-SS.log`
- Настраивается через `configGPU.json` (log_path)

### 6.6 Профилирование

- Формат: `Results/GPU_XX_Profiler/.....'
- Включается в `configGPU.json`: `"is_prof": true`

---

