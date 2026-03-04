# DrvGPU/services — Сервисы инфраструктуры

Фоновые сервисы DrvGPU: профилирование, консоль, кеш ядер, хранилище конфигураций.

---

## Структура

```
services/
├── gpu_profiler.hpp          # Singleton — асинхронный сбор GPU timing
├── profiling_stats.hpp       # Агрегированная статистика (min/max/avg)
├── profiling_types.hpp       # Типы данных профилирования (OpenCL + ROCm)
├── gpu_benchmark_base.hpp    # Базовый класс для GPU бенчмарков (Template Method)
├── console_output.hpp        # Singleton — потокобезопасный вывод (multi-GPU)
├── service_manager.hpp       # Singleton — жизненный цикл сервисов
├── async_service_base.hpp    # Базовый класс асинхронных сервисов
├── batch_manager.hpp/cpp     # Разбиение данных на пакеты (BatchManager)
├── kernel_cache_service.hpp/cpp    # On-disk кеш скомпилированных ядер
├── filter_config_service.hpp/cpp   # Хранение конфигураций фильтров (FIR/IIR)
└── storage/
    ├── i_storage_backend.hpp        # Абстрактный интерфейс хранилища
    └── file_storage_backend.hpp/cpp # Файловая реализация
```

---

## GPUProfiler — профилирование

Singleton. Асинхронно собирает timing из cl_event / ROCm events, агрегирует min/max/avg.

```cpp
auto& prof = GPUProfiler::GetInstance();

// Обязательный порядок: SetGPUInfo ПЕРЕД Start() — иначе в отчёте "Unknown GPU"
prof.SetGPUInfo(gpu_id, gpu_report_info);
prof.Start();

// Запись события
prof.Record(gpu_id, "FFTProcessor", "Execute", opencl_profiling_data);

// Дождаться обработки всей очереди (перед Report)
prof.WaitEmpty();

// Вывод — ТОЛЬКО через эти методы:
prof.PrintReport();
prof.ExportJSON("Results/Profiler/report.json");
prof.ExportMarkdown("Results/Profiler/report.md");

prof.Stop();
prof.Reset();  // очистить данные (например, между warmup и measure)
```

> ⚠️ **ЗАПРЕЩЕНО**: вручную вызывать `GetStats()` + цикл + `ConsoleOutput::Print`.
> Весь вывод профилирования — только через `PrintReport()` / `ExportMarkdown()` / `ExportJSON()`.

---

## GpuBenchmarkBase — базовый класс бенчмарка

Template Method Pattern для стандартного профилирования GPU-модулей.
Production-код модуля остаётся **чистым** (ноль кода профилирования).

**Поток исполнения `Run()`:**
```
InitProfiler (SetGPUInfo + Start)
  → Warmup: n_warmup × ExecuteKernel()     // без timing (прогрев JIT / clock ramp-up)
  → profiler.Reset()                       // очистить данные warmup
  → Measure: n_runs × ExecuteKernelTimed() // с timing → GPUProfiler
```

```cpp
class MyBenchmark : public GpuBenchmarkBase {
protected:
  void ExecuteKernel() override {
    module_.Process(input_, output_);       // warmup — без cl_event
  }
  void ExecuteKernelTimed() override {
    cl_event ev;
    module_.Process(input_, output_, &ev);  // measure — с cl_event
    RecordEvent("Process", ev);             // helper базового класса → GPUProfiler
  }
};

BenchmarkConfig cfg{.n_warmup = 5, .n_runs = 20, .output_dir = "Results/Profiler"};
MyBenchmark bench(&backend, "MyModule", cfg);
bench.Run();     // warmup + measure
bench.Report();  // PrintReport + ExportJSON + ExportMarkdown + Stop
```

`Run()` и `Report()` — **no-op** если `is_prof=false` в `configGPU.json`.
Все бенчмарки модулей наследуют GpuBenchmarkBase и живут в `{module}/tests/`.

---

## ConsoleOutput — потокобезопасный вывод

Singleton. Обязателен при работе с несколькими GPU — иначе строки перемешаются.

```cpp
ConsoleOutput::GetInstance().Print(gpu_id, "Message");
ConsoleOutput::GetInstance().PrintError(gpu_id, "Error message");
```

---

## ServiceManager — жизненный цикл

Singleton. Инициализирует и завершает все сервисы (Logger, Profiler, Console).

```cpp
ServiceManager::GetInstance().Initialize();
// ... работа ...
ServiceManager::GetInstance().Shutdown();
```

---

## BatchManager — разбиение на пакеты

Разбивает большие данные на пакеты с учётом доступной VRAM.

```cpp
BatchManager manager(backend);
auto batches = manager.CreateBatches(total_size, element_size);
for (auto& batch : batches) { /* process */ }
```

---

## KernelCacheService — кеш скомпилированных ядер

On-disk кеш OpenCL/ROCm ядер. Storage-agnostic: не зависит от OpenCL API,
возвращает `{source, binary}` — caller сам создаёт `cl_program`.

```
base_dir/
├── my_kernel.cl               # исходник
├── bin/my_kernel_opencl.bin   # скомпилированный бинарь (или _rocm.hsaco)
└── manifest.json              # индекс + метаданные + timestamps
```

```cpp
KernelCacheService cache("modules/filters/kernels", BackendType::OPENCL);

// Сохранить после компиляции
cache.Save("fir_filter", cl_source, compiled_binary, params_str, "comment");

// Загрузить
auto entry = cache.Load("fir_filter");
if (entry.has_binary()) {
  // быстрый путь: clCreateProgramWithBinary(entry.binary)
} else {
  // fallback: clCreateProgramWithSource(entry.source)
}

auto names = cache.ListKernels();  // из manifest.json
```

Версионирование при перезаписи: `name` → `name_00`, `name_01`, ...

---

## FilterConfigService — конфигурации фильтров

Сохраняет/загружает параметры FIR и IIR фильтров в JSON. Версионирование при перезаписи.

```cpp
FilterConfigService svc("modules/filters/configs");

FilterConfigData cfg;
cfg.type = "fir";
cfg.coefficients = {0.1f, 0.5f, 0.8f, 0.5f, 0.1f};
svc.Save("lp_5000", cfg, "Low-pass 5 kHz");

auto loaded = svc.Load("lp_5000");   // throws если не найден
bool ok     = svc.Exists("lp_5000");
auto names  = svc.ListFilters();
```

Файлы: `base_dir/filters/{name}.json`
IIR секции: `sections[i] = {b0, b1, b2, a1, a2}` (biquad format).

---

## storage/ — абстракция хранилища

| Файл | Класс | Описание |
|------|-------|----------|
| `i_storage_backend.hpp` | `IStorageBackend` | Интерфейс: `Save / Load / List / Exists` |
| `file_storage_backend.hpp/cpp` | `FileStorageBackend` | Файловая реализация (текущая) |

Ключи могут содержать `/` — интерпретируются как подкаталог.
Планируется: `SqliteStorageBackend`.

---

## Тесты

- **GPUProfiler**: `tests/test_gpu_profiler.hpp` — `test_gpu_profiler::run()`
- **ConsoleOutput, ServiceManager**: `tests/test_services.hpp` — `test_services::run()`
