@page drvgpu_architecture DrvGPU --- Архитектура

@tableofcontents

@section drvgpu_arch_intro Обзор

Архитектура DrvGPU построена на паттерне **Bridge**: абстрактный интерфейс
@ref drv_gpu_lib::IBackend "IBackend" отделён от конкретных реализаций
(OpenCL, ROCm, Hybrid). Класс @ref drv_gpu_lib::DrvGPU "DrvGPU" выступает
фасадом, скрывая детали работы с конкретным бэкендом.

```
                        +-------------------+
                        |    Application    |
                        |   (FFT, Filters,  |
                        |   Statistics...)   |
                        +--------+----------+
                                 |
                                 | получают IBackend*
                                 v
                     +-----------+-----------+
                     |       DrvGPU          |
                     |      (Facade)         |
                     +--+--------+--------+--+
                        |        |        |
              +---------+   +----+----+   +---------+
              |             |         |             |
    +---------v---+  +------v------+  +------v------+
    | OpenCL      |  | ROCm       |  | Hybrid      |
    | Backend     |  | Backend    |  | Backend     |
    +-------------+  +------------+  +------+------+
    | cl_context  |  | hipStream_t|  | OpenCL +    |
    | cl_queue    |  |            |  | ROCm +      |
    |             |  |            |  | ZeroCopy    |
    +-------------+  +------------+  +-------------+
```

---

@section drvgpu_arch_ibackend IBackend --- абстрактный интерфейс

@ref drv_gpu_lib::IBackend "IBackend" определяет контракт для всех GPU-бэкендов.
Файл: `DrvGPU/interface/i_backend.hpp`

@subsection drvgpu_ibackend_lifecycle Жизненный цикл

| Метод | Описание |
|-------|----------|
| `Initialize(device_index)` | Инициализация бэкенда для конкретного GPU |
| `IsInitialized()` | Проверка готовности |
| `Cleanup()` | Освобождение ресурсов (учитывает ownership) |
| `SetOwnsResources(bool)` | Режим владения: true = бэкенд освобождает, false = внешний код |
| `OwnsResources()` | Проверка текущего режима владения |

@subsection drvgpu_ibackend_memory Управление памятью

| Метод | Описание |
|-------|----------|
| `Allocate(size_bytes, flags)` | Выделить память на GPU |
| `AllocateManaged(size_bytes)` | Unified memory (ROCm: hipMallocManaged) |
| `Free(ptr)` | Освободить GPU-память |
| `MemcpyHostToDevice(dst, src, size)` | Копирование Host -> Device |
| `MemcpyDeviceToHost(dst, src, size)` | Копирование Device -> Host |
| `MemcpyDeviceToDevice(dst, src, size)` | Копирование Device -> Device |

@subsection drvgpu_ibackend_native Нативные хэндлы

| Метод | OpenCL | ROCm | Hybrid |
|-------|--------|------|--------|
| `GetNativeContext()` | `cl_context` | `hipCtx_t` | OpenCL sub-backend |
| `GetNativeDevice()` | `cl_device_id` | `hipDevice_t` | OpenCL sub-backend |
| `GetNativeQueue()` | `cl_command_queue` | `hipStream_t` | OpenCL sub-backend |

@subsection drvgpu_ibackend_caps Возможности устройства

| Метод | Описание |
|-------|----------|
| `SupportsSVM()` | Поддержка Shared Virtual Memory |
| `SupportsDoublePrecision()` | Поддержка double precision |
| `GetMaxWorkGroupSize()` | Максимальный размер work group |
| `GetGlobalMemorySize()` | Глобальная память (bytes) |
| `GetFreeMemorySize()` | Свободная память GPU (bytes) |
| `GetLocalMemorySize()` | Локальная память (bytes) |

@subsection drvgpu_ibackend_sync Синхронизация

| Метод | Описание |
|-------|----------|
| `Synchronize()` | Ожидание завершения всех операций |
| `Flush()` | Сброс буфера команд (без ожидания) |

---

@section drvgpu_arch_backends Три бэкенда

@subsection drvgpu_backend_opencl OpenCLBackend

Расположение: `DrvGPU/backends/opencl/`

- Поддержка OpenCL 3.0 (все GPU: AMD, NVIDIA, Intel)
- Управление через `cl_context`, `cl_command_queue`, `cl_device_id`
- Ядро: класс `OpenCLCore` --- низкоуровневое управление OpenCL-ресурсами
- Обнаружение устройств: `OpenCLCore::GetAvailableDeviceCount()`
- SVM (Shared Virtual Memory) при поддержке устройством
- Профилирование через `cl_event` (5 полей: queued, submit, start, end, complete)

@code{.cpp}
// Стандартное создание
drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::OPENCL, 0);
gpu.Initialize();

// Из внешнего контекста
auto gpu_ext = drv_gpu_lib::DrvGPU::CreateFromExternalOpenCL(0, ctx, dev, queue);
// gpu_ext НЕ освобождает ctx/dev/queue при уничтожении
@endcode

@subsection drvgpu_backend_rocm ROCmBackend

Расположение: `DrvGPU/backends/rocm/`

@note Компилируется только при `ENABLE_ROCM=1`

- ROCm 7.2+ для AMD GPU (gfx908, gfx1201 и др.)
- Управление через `hipStream_t`
- `hipMallocManaged` для Unified Memory (отладка без явного D2H)
- `ROCmCore` --- низкоуровневый слой с `owns_stream_` для внешних потоков
- Профилирование через `roctracer` / HIP events (расширенные поля: domain, kind, op, kernel_name)

@code{.cpp}
// Стандартное создание
drv_gpu_lib::DrvGPU gpu(drv_gpu_lib::BackendType::ROCM, 0);
gpu.Initialize();

// Из внешнего HIP stream
hipStream_t s;
hipStreamCreate(&s);
auto gpu_ext = drv_gpu_lib::DrvGPU::CreateFromExternalROCm(0, s);
// gpu_ext НЕ вызовет hipStreamDestroy
@endcode

@subsection drvgpu_backend_hybrid HybridBackend

Расположение: `DrvGPU/backends/hybrid/`

@note Компилируется только при `ENABLE_ROCM=1`

- Одновременная работа OpenCL + ROCm на одном AMD GPU
- Два sub-backend'а: OpenCLBackend + ROCmBackend
- **ZeroCopyBridge** --- нулевое копирование между `cl_mem` и HIP-указателем
  (возможно, т.к. OpenCL и HIP на AMD GPU разделяют одно VRAM)
- Полезно для интеграции: OpenCL kernels + hipBLAS/hipFFT в одном pipeline

@code{.cpp}
// Из внешних контекстов
auto gpu = drv_gpu_lib::DrvGPU::CreateFromExternalHybrid(0, cl_ctx, cl_dev, cl_q, hip_s);

// Доступ к ZeroCopyBridge
auto& hybrid = static_cast<drv_gpu_lib::HybridBackend&>(gpu.GetBackend());
auto bridge = hybrid.CreateZeroCopyBridge(cl_buf, size);
// bridge.GetHIPPointer() --- указатель для hipBLAS/hipFFT
@endcode

---

@section drvgpu_arch_drvgpu Класс DrvGPU (Facade)

Файл: `DrvGPU/include/drv_gpu.hpp`

@ref drv_gpu_lib::DrvGPU "DrvGPU" --- главный класс библиотеки.
**Не является Singleton!** Для каждого GPU создаётся свой экземпляр.

@subsection drvgpu_facade_features Возможности

- Создание из `BackendType` + `device_index`
- Static factory methods для внешних контекстов:
  - `CreateFromExternalOpenCL(device_index, context, device, queue)`
  - `CreateFromExternalROCm(device_index, stream)`
  - `CreateFromExternalHybrid(device_index, context, device, queue, stream)`
- Доступ к подсистемам: `GetBackend()`, `GetMemoryManager()`, `GetModuleRegistry()`
- Синхронизация: `Synchronize()`, `Flush()`
- RAII: автоматическая очистка в деструкторе
- Move-семантика: перемещение без копирования

@subsection drvgpu_facade_ownership Модель владения ресурсами

Критически важно при интеграции с внешними контекстами:

| Способ создания | `owns_resources_` | Cleanup() |
|----------------|-------------------|-----------|
| `DrvGPU(BackendType, index)` + `Initialize()` | `true` | Освобождает все ресурсы |
| `CreateFromExternalOpenCL(...)` | `false` | Обнуляет указатели, НЕ освобождает |
| `CreateFromExternalROCm(...)` | `false` | Обнуляет указатели, НЕ вызывает hipStreamDestroy |
| `CreateFromExternalHybrid(...)` | `false` | Обнуляет указатели обоих sub-backend'ов |

@warning При `owns_resources_ = false` бэкенд **не уничтожает** чужие хэндлы.
Вызывающий код обязан самостоятельно освободить ресурсы.

---

@section drvgpu_arch_gpumanager GPUManager --- Multi-GPU координатор

Файл: `DrvGPU/include/gpu_manager.hpp`

@ref drv_gpu_lib::GPUManager "GPUManager" управляет массивом DrvGPU для multi-GPU сценариев.

@subsection drvgpu_gpumanager_features Возможности

- **Автоматическое обнаружение** всех GPU через `OpenCLCore::GetAvailableDeviceCount()`
- **Инициализация**: `InitializeAll(BackendType)` --- создаёт DrvGPU для каждого устройства
- **Балансировка нагрузки**:
  - `GetNextGPU()` --- Round-Robin
  - `GetLeastLoadedGPU()` --- наименее загруженная GPU
  - `GetGPU(index)` --- явный выбор
- **Auto-fill GPUProfiler** --- автоматическая передача GPUReportInfo при инициализации
- Thread-safe доступ ко всем GPU

@code{.cpp}
drv_gpu_lib::GPUManager manager;
manager.InitializeAll(drv_gpu_lib::BackendType::OPENCL);

std::cout << "Обнаружено GPU: " << manager.GetGPUCount() << std::endl;

// Round-Robin
for (int i = 0; i < 1000; ++i) {
    auto& gpu = manager.GetNextGPU();
    // gpu.GetBackend()...
}
@endcode

---

@section drvgpu_arch_memory MemoryManager

Файл: `DrvGPU/memory/memory_manager.hpp`

@ref drv_gpu_lib::MemoryManager "MemoryManager" управляет GPU-памятью с учётом
статистики и пулинга буферов.

@subsection drvgpu_mem_features Возможности

- Аллокация/деаллокация GPU-памяти
- Статистика: количество аллокаций, текущее потребление, пиковое
- Пулинг буферов для переиспользования
- RAII: автоматическое освобождение при уничтожении DrvGPU
- Работает через `IBackend` (не привязан к конкретному бэкенду)

---

@section drvgpu_arch_batch BatchManager

Файл: `DrvGPU/services/batch_manager.hpp`

@ref drv_gpu_lib::BatchManager "BatchManager" --- универсальный менеджер пакетной обработки.

@subsection drvgpu_batch_features Возможности

- Расчёт оптимального размера пакета на основе свободной GPU-памяти
- Настраиваемый процент доступной памяти (по умолчанию 70%)
- Умное слияние хвоста: если в последнем пакете 1--3 элемента --- объединяется с предыдущим
- Работает с любым `IBackend`

@code{.cpp}
drv_gpu_lib::BatchManager manager;

// Расчёт размера пакета
size_t per_item_memory = nFFT * sizeof(std::complex<float>) * 2;
size_t batch_size = manager.CalculateOptimalBatchSize(
    backend, total_beams, per_item_memory, 0.7);

// Генерация диапазонов
auto batches = manager.CreateBatches(total_beams, batch_size, 3, true);

for (auto& batch : batches) {
    ProcessBatch(input, batch.start, batch.count);
}
@endcode

---

@section drvgpu_arch_services Сервисы

@subsection drvgpu_arch_servicemanager ServiceManager

Файл: `DrvGPU/services/service_manager.hpp`

@ref drv_gpu_lib::ServiceManager "ServiceManager" --- синглтон, управляющий
жизненным циклом всех фоновых сервисов DrvGPU.

**Жизненный цикл:**

1. `GPUManager` создаёт GPU
2. `ServiceManager::InitializeFromConfig("configGPU.json")` --- настройка из JSON
3. `ServiceManager::StartAll()` --- запуск фоновых потоков
4. ... работа GPU, модули вызывают Enqueue() ...
5. `ServiceManager::StopAll()` --- освобождение очередей, join потоков

**Управляемые сервисы:**

| Сервис | Настройка из конфига | Фоновый поток |
|--------|---------------------|---------------|
| ConsoleOutput | `is_console` per GPU | Да |
| GPUProfiler | `is_prof` per GPU | Да |
| Logger (plog) | `is_logger` per GPU | Нет (файловый I/O) |

@code{.cpp}
auto& sm = drv_gpu_lib::ServiceManager::GetInstance();
sm.InitializeFromConfig("configGPU.json");
sm.StartAll();

// ... работа ...

// Статус сервисов
std::cout << sm.GetStatus();

sm.StopAll();
@endcode

@subsection drvgpu_arch_asyncbase AsyncServiceBase

Файл: `DrvGPU/services/async_service_base.hpp`

Шаблонный базовый класс для асинхронных сервисов. Предоставляет:

- Потокобезопасную очередь сообщений
- Фоновый рабочий поток
- Неблокирующий `Enqueue()` для потоков GPU
- `Start()` / `Stop()` для управления жизненным циклом
- Счётчики обработанных сообщений

Наследники: `ConsoleOutput`, `GPUProfiler`

Подробнее о сервисах:
- @ref drvgpu_profiler --- GPUProfiler API
- @ref drvgpu_console --- ConsoleOutput API

---

@section drvgpu_arch_threading Потокобезопасность

| Компонент | Стратегия |
|-----------|-----------|
| DrvGPU | `std::mutex` для внутреннего состояния |
| IBackend | Реализации thread-safe (per-backend mutex) |
| ConsoleOutput | AsyncServiceBase: lock-free enqueue, single consumer |
| GPUProfiler | AsyncServiceBase: lock-free enqueue, single consumer |
| ServiceManager | Single-thread init/start/stop (из main) |
| GPUManager | `std::mutex` для доступа к массиву GPU |

@note Модули, работающие на разных GPU, автоматически thread-safe
(каждый DrvGPU имеет свой бэкенд, свою очередь, свою память).

---

@section drvgpu_arch_modules Взаимодействие с модулями

Все модули GPUWorkLib получают `IBackend*` от DrvGPU и работают через единую архитектуру:

| Слой | Класс | Назначение |
|------|-------|-----------|
| 1 | `GpuContext` | Per-module: backend, stream, compiled module, shared buffers |
| 2 | `IGpuOperation` | Interface: Name, Initialize, IsReady, Release |
| 3 | `GpuKernelOp` | Base: доступ к compiled kernels через GpuContext |
| 4 | `BufferSet<N>` | Compile-time GPU buffer array (zero overhead, trivial move) |
| 5 | Concrete Ops | Маленькие классы: MeanReductionOp, FirFilterOp... |
| 6 | Facade + Strategy | Тонкий фасад (StatisticsProcessor), авто-выбор (MedianStrategy) |

Подробнее: см. `Doc_Addition/PLAN/Ref03_Unified_Architecture.md`
