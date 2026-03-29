@page drvgpu_main DrvGPU --- Unified GPU Management

@tableofcontents

@section drvgpu_overview Назначение

**DrvGPU** --- единая точка управления GPU в проекте GPUWorkLib.
Предоставляет абстракцию над тремя бэкендами (OpenCL 3.0, ROCm/HIP 7.2+, Hybrid)
и обеспечивает все модули библиотеки единым интерфейсом `IBackend`.

Основные задачи:

- **Backend Abstraction** --- Bridge Pattern: один интерфейс, несколько реализаций
- **Multi-GPU** --- координация через GPUManager (Round-Robin, Least-Loaded)
- **Memory Management** --- RAII, pooling, статистика через MemoryManager
- **Profiling** --- асинхронный сбор данных через GPUProfiler (OpenCL + ROCm)
- **Console Output** --- потокобезопасный вывод через ConsoleOutput (10 GPU одновременно)
- **Service Lifecycle** --- ServiceManager: Init -> Start -> Work -> Stop

@note Namespace: `drv_gpu_lib`
@note Статус: Active | Backend: OpenCL + ROCm + Hybrid

---

@section drvgpu_classes Ключевые классы

| Класс | Заголовок | Описание |
|-------|-----------|----------|
| @ref drv_gpu_lib::DrvGPU "DrvGPU" | `drv_gpu.hpp` | Facade: init GPU, get backend, создание из внешнего контекста |
| @ref drv_gpu_lib::GPUManager "GPUManager" | `gpu_manager.hpp` | Multi-GPU координатор с load balancing |
| @ref drv_gpu_lib::IBackend "IBackend" | `i_backend.hpp` | Bridge Pattern: универсальный интерфейс бэкенда |
| OpenCLBackend | `opencl_backend.hpp` | Реализация OpenCL 3.0 (все GPU) |
| ROCmBackend | `rocm_backend.hpp` | Реализация ROCm 7.2+ (AMD, hipStream_t) |
| HybridBackend | `hybrid_backend.hpp` | OpenCL + ROCm одновременно + ZeroCopyBridge |
| @ref drv_gpu_lib::MemoryManager "MemoryManager" | `memory_manager.hpp` | GPU memory: alloc/free/statistics/pooling |
| @ref drv_gpu_lib::GPUProfiler "GPUProfiler" | `gpu_profiler.hpp` | Профилирование: Record -> Aggregate -> PrintReport / ExportMarkdown / ExportJSON |
| @ref drv_gpu_lib::ConsoleOutput "ConsoleOutput" | `console_output.hpp` | Singleton, thread-safe вывод для multi-GPU |
| @ref drv_gpu_lib::ServiceManager "ServiceManager" | `service_manager.hpp` | Жизненный цикл фоновых сервисов |
| @ref drv_gpu_lib::BatchManager "BatchManager" | `batch_manager.hpp` | Пакетная обработка с учётом доступной GPU-памяти |

Подробнее об архитектуре: @ref drvgpu_architecture

---

@section drvgpu_arch_overview Архитектура (обзор)

```
Application / Modules (FFT, Filters, Statistics, Heterodyne, ...)
       |  получают IBackend*
       v
   DrvGPU (Facade)
       |
       +-- OpenCLBackend  --> cl_context / cl_command_queue
       +-- ROCmBackend    --> hipStream_t  (ENABLE_ROCM=1)
       +-- HybridBackend  --> OpenCL + ROCm + ZeroCopyBridge
       |
       +-- MemoryManager  --> alloc / free / pool / statistics
       +-- ModuleRegistry --> регистрация compute-модулей
       |
   GPUManager (Multi-GPU)
       +-- DrvGPU[0], DrvGPU[1], ... DrvGPU[N-1]
       +-- Load Balancing: Round-Robin / Least-Loaded / Manual
       |
   ServiceManager (Singleton)
       +-- ConsoleOutput  --> thread-safe stdout
       +-- GPUProfiler    --> async profiling (OpenCL + ROCm)
       +-- Logger (plog)  --> per-GPU файлы в Logs/DRVGPU_XX/
```

@warning Все модули GPUWorkLib используют контекст DrvGPU.
Не создавайте собственные OpenCL/HIP контексты --- получайте `IBackend*` от DrvGPU.

---

@section drvgpu_quickstart Быстрый старт

@subsection drvgpu_qs_single Single GPU (C++)

@code{.cpp}
#include "DrvGPU/include/drv_gpu.hpp"

// Создание и инициализация
drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::ROCM, 0);
gpu.Initialize();

// Получение бэкенда для модулей
auto& backend = gpu.GetBackend();

// Информация об устройстве
gpu.PrintDeviceInfo();

// RAII: gpu.Cleanup() вызывается автоматически в деструкторе
@endcode

@subsection drvgpu_qs_multi Multi-GPU (C++)

@code{.cpp}
#include "DrvGPU/include/gpu_manager.hpp"
#include "DrvGPU/services/service_manager.hpp"

// Инициализация всех GPU в системе
drv_gpu_lib::GPUManager manager;
manager.InitializeAll(drv_gpu_lib::BackendType::OPENCL);

// Запуск сервисов (ConsoleOutput, GPUProfiler, Logger)
auto& sm = drv_gpu_lib::ServiceManager::GetInstance();
sm.InitializeFromConfig("configGPU.json");
sm.StartAll();

// Round-Robin распределение задач
for (int i = 0; i < 100; ++i) {
    auto& gpu = manager.GetNextGPU();
    // ... обработка на gpu ...
}

// Явный выбор GPU
auto& gpu0 = manager.GetGPU(0);
auto& gpu1 = manager.GetGPU(1);

// Балансировка нагрузки
auto& least_loaded = manager.GetLeastLoadedGPU();

// Завершение
sm.StopAll();
@endcode

@subsection drvgpu_qs_external External Context (C++)

DrvGPU поддерживает создание из внешних GPU-контекстов.
Это полезно при интеграции с существующим кодом (hipBLAS, hipFFT, MIOpen и др.).

@code{.cpp}
// Из внешнего OpenCL контекста
cl_context ctx = ...;
cl_device_id dev = ...;
cl_command_queue q = ...;
auto gpu_ocl = drv_gpu_lib::DrvGPU::CreateFromExternalOpenCL(0, ctx, dev, q);
// gpu_ocl.Initialize() вызывать НЕ нужно --- уже инициализирован
// gpu_ocl НЕ освобождает внешние ресурсы при уничтожении

// Из внешнего HIP stream (ROCm)
hipStream_t s;
hipStreamCreate(&s);
auto gpu_rocm = drv_gpu_lib::DrvGPU::CreateFromExternalROCm(0, s);
// gpu_rocm НЕ вызовет hipStreamDestroy --- owns_resources_ = false

// Из обоих контекстов (HybridBackend)
auto gpu_hybrid = drv_gpu_lib::DrvGPU::CreateFromExternalHybrid(0, ctx, dev, q, s);
auto& hybrid = static_cast<drv_gpu_lib::HybridBackend&>(gpu_hybrid.GetBackend());
auto bridge = hybrid.CreateZeroCopyBridge(cl_buf, size);
@endcode

@subsection drvgpu_qs_python Python

@code{.py}
import gpuworklib as gw

# Создание GPU-контекста (ROCm)
ctx = gw.ROCmGPUContext(0)

# ctx передаётся во все модули
fft = gw.FFTProcessor(ctx, n_fft=1024)
gen = gw.SignalGenerator(ctx)
@endcode

---

@section drvgpu_patterns Используемые паттерны

| Паттерн | Где | Назначение |
|---------|-----|-----------|
| **Bridge** | IBackend -> OpenCL/ROCm/Hybrid | Отделение абстракции от реализации |
| **Facade** | DrvGPU, GPUManager | Упрощённый интерфейс для сложной подсистемы |
| **Singleton** | ConsoleOutput, GPUProfiler, ServiceManager | Единая точка доступа к сервисам |
| **Factory Method** | DrvGPU::CreateFromExternal* | Создание объектов с внешними ресурсами |
| **Strategy** | LoadBalancing (Round-Robin, Least-Loaded) | Алгоритмы балансировки нагрузки |
| **RAII** | DrvGPU, MemoryManager | Автоматическое освобождение ресурсов |
| **Observer** | AsyncServiceBase -> ConsoleOutput/GPUProfiler | Асинхронная обработка сообщений |

---

@section drvgpu_tests Тесты

@subsection drvgpu_tests_cpp C++ тесты (DrvGPU/tests/)

| Файл | Статус | Описание |
|------|--------|----------|
| `single_gpu.hpp` | Active | Smoke: init, device info, alloc/write/read, MemoryManager stats |
| `test_services.hpp` | Active | ConsoleOutput (8 threads), AsyncServiceBase stress, ServiceManager |
| `test_gpu_profiler.hpp` | Active | GPUProfiler: Record 100+ events, PrintReport, ExportMarkdown |
| `test_storage_services.hpp` | Active | FileStorageBackend, KernelCacheService, FilterConfigService |
| `example_external_context_usage.hpp` | Off | OpenCL external context (5 примеров) |
| `test_drv_gpu_external.hpp` | Off | DrvGPU::CreateFromExternal (OCL/ROCm/Hybrid) --- 6 тестов |
| `test_rocm_backend.hpp` | ROCm | ROCmBackend: init, alloc, memcpy, sync |
| `test_rocm_external_context.hpp` | Off | ROCmBackend::InitializeFromExternalStream --- 6 тестов |
| `test_hybrid_backend.hpp` | ROCm | HybridBackend: dual sub-backends |
| `test_hybrid_external_context.hpp` | Off | HybridBackend external contexts --- 6 тестов |
| `test_zero_copy.hpp` | ROCm | ZeroCopyBridge: OpenCL <-> HIP zero-copy |

@subsection drvgpu_tests_python Python тесты

DrvGPU --- инфраструктурный слой, тестируется через модули-потребители
(FFT, Filters, Statistics, Heterodyne и др.).

---

@section drvgpu_config Конфигурация

Файл `configGPU.json` управляет поведением сервисов для каждого GPU:

@code{.json}
{
  "gpus": [
    {
      "id": 0,
      "is_console": true,
      "is_prof": true,
      "is_logger": true,
      "log_level": "info"
    },
    {
      "id": 1,
      "is_console": true,
      "is_prof": false,
      "is_logger": true,
      "log_level": "warning"
    }
  ]
}
@endcode

- `is_console` --- включить/отключить ConsoleOutput для данного GPU
- `is_prof` --- включить/отключить GPUProfiler для данного GPU
- `is_logger` --- включить/отключить файловое логирование (plog)
- `log_level` --- уровень: debug, info, warning, error

---

@section drvgpu_results Результаты и логи

| Путь | Содержимое |
|------|-----------|
| `Results/Profiler/` | Markdown + JSON отчёты профилирования |
| `Results/JSON/` | Результаты тестов в JSON |
| `Logs/DRVGPU_XX/` | Per-GPU логи (plog format) |

---

@section drvgpu_related_pages Связанные страницы

- @ref drvgpu_architecture --- Детальная архитектура DrvGPU (бэкенды, IBackend, ServiceManager)
- @ref drvgpu_profiler --- GPUProfiler API: профилирование GPU-операций
- @ref drvgpu_console --- ConsoleOutput: потокобезопасный вывод для multi-GPU
