# DrvGPU/backends — Архитектура бэкендов

Абстракция GPU-вычислений. `IBackend` (`interface/i_backend.hpp`) — единый интерфейс;
три реализации: OpenCL (все платформы), ROCm/HIP (AMD + Linux), Hybrid (оба на одном GPU).

---

## Структура

```
backends/
├── opencl/         # OpenCL backend — основной, все платформы
├── rocm/           # ROCm/HIP backend — AMD GPU, только Linux (ENABLE_ROCM)
├── hybrid/         # HybridBackend — OpenCL + ROCm на одном GPU
└── README.md
```

---

## Бэкенды

### `opencl/` — OpenCLBackend

| Файл | Класс | Роль |
|------|-------|------|
| `opencl_backend.hpp/cpp` | `OpenCLBackend` | `IBackend` реализация. Multi-GPU: каждый экземпляр = один GPU |
| `opencl_core.hpp/cpp` | `OpenCLCore` | Per-device контекст (context, device, platform) |
| `command_queue_pool.hpp/cpp` | `CommandQueuePool` | Пул command queues для параллельного выполнения |
| `opencl_profiling.hpp/cpp` | — | Утилиты профилирования (cl_event → время в нс) |
| `opencl_export.hpp` | — | ZeroCopy утилиты: экспорт `cl_mem` → dma-buf fd / AMD GPU VA (Linux only) |

Платформы: Windows, Linux, AMD, NVIDIA.

Инициализация:
```cpp
OpenCLBackend backend;
backend.Initialize(0);                              // по индексу GPU

// или через внешний контекст (e.g. из Qt / другого SDK):
backend.InitializeFromExternalContext(ctx, device, queue);
```

---

### `rocm/` — ROCmBackend

| Файл | Класс | Роль |
|------|-------|------|
| `rocm_backend.hpp/cpp` | `ROCmBackend` | `IBackend` реализация на базе HIP API |
| `rocm_core.hpp/cpp` | `ROCmCore` | Per-device HIP контекст (stream, device props) |
| `zero_copy_bridge.hpp/cpp` | `ZeroCopyBridge` | Импорт `cl_mem` → HIP address space без копий через CPU |

Платформы: **AMD GPU + Linux + `-DENABLE_ROCM=ON`**. На Windows — stub (бросает исключение).

```cpp
// (Только через HybridBackend в рабочем коде — не создавать напрямую)
ROCmBackend backend;
backend.Initialize(0);
hipStream_t stream = backend.GetCore().GetStream();  // для hipFFT, rocPRIM и т.п.
```

#### ZeroCopy: `cl_mem` ↔ HIP

`ZeroCopyBridge` позволяет использовать один буфер в обоих API без копирования:

| Метод | Расширение | Overhead | Ограничения |
|-------|------------|----------|-------------|
| `AMD_GPU_VA` | `CL_MEM_AMD_GPU_VA` | **нулевой** — тот же VA | AMD GPU + Linux |
| `DMA_BUF` | `cl_khr_external_memory_dma_buf` | ~мкс на импорт | Linux kernel |
| `NONE` | — | — | ZeroCopy недоступен |

> **Важно:** SVM не используется как fallback — `hipSVMAlloc` требует специальной аллокации через `clSVMAlloc`; произвольный `cl_mem` импортировать нельзя.

Приоритет в `ImportFromOpenCl()`: AMD GPU VA → DMA-BUF → `std::runtime_error`.

---

### `hybrid/` — HybridBackend

| Файл | Класс | Роль |
|------|-------|------|
| `hybrid_backend.hpp/cpp` | `HybridBackend` | `IBackend` — OpenCL + ROCm на одном GPU |

Архитектура: хранит `OpenCLBackend` + `ROCmBackend`, делегирует по типу операции:
- Стандартный IBackend (Allocate/Memcpy/Sync) → **OpenCL**
- ROCm-специфика (hipFFT, rocPRIM) → **`GetROCm()`**
- ZeroCopy (данные между API) → **`CreateZeroCopyBridge()`**

```cpp
DrvGPU gpu(BackendType::OPENCLandROCm, 0);
gpu.Initialize();

auto& hybrid = static_cast<HybridBackend&>(gpu.GetBackend());

// OpenCL kernel
auto* cl = hybrid.GetOpenCL();
void* cl_buf = cl->Allocate(size);

// HIP kernel (тот же GPU, без копий)
auto bridge = hybrid.CreateZeroCopyBridge(static_cast<cl_mem>(cl_buf), size);
void* hip_ptr = bridge->GetHipPtr();

// Синхронизация вокруг ZeroCopy
hybrid.SyncBeforeZeroCopy();   // clFinish перед HIP
// ... hipKernel<<<...>>>(hip_ptr, N);
hybrid.SyncAfterZeroCopy();    // hipStreamSync перед OpenCL
```

Платформы: **AMD GPU + Linux + `-DENABLE_ROCM=ON`**. На Windows — stub.

---

## IBackend — интерфейс

```cpp
// Жизненный цикл
void Initialize(int device_index);
void Cleanup();
bool IsInitialized() const;

// Память (OpenCL: cl_mem; ROCm: hipMalloc)
void* Allocate(size_t size_bytes, unsigned int flags = 0);
void  Free(void* ptr);
void  MemcpyHostToDevice(void* dst, const void* src, size_t size);
void  MemcpyDeviceToHost(void* dst, const void* src, size_t size);
void  MemcpyDeviceToDevice(void* dst, const void* src, size_t size);

// Синхронизация
void Synchronize();   // блокирует CPU до завершения GPU
void Flush();         // отправляет команды без ожидания (GPU + CPU параллельно)

// Информация
GPUDeviceInfo  GetDeviceInfo() const;
int            GetDeviceIndex() const;
std::string    GetDeviceName() const;
BackendType    GetType() const;

// Нативные хэндлы (OpenCL: cl_context / cl_device_id / cl_command_queue)
void* GetNativeContext() const;
void* GetNativeDevice() const;
void* GetNativeQueue() const;

// Возможности
bool   SupportsSVM() const;
bool   SupportsDoublePrecision() const;
size_t GetMaxWorkGroupSize() const;
size_t GetGlobalMemorySize() const;
size_t GetFreeMemorySize() const;
size_t GetLocalMemorySize() const;

// MemoryManager
MemoryManager* GetMemoryManager();
```

---

## Выбор бэкенда

| `BackendType` | Класс | Когда использовать |
|---------------|-------|--------------------|
| `OPENCL` | `OpenCLBackend` | Основной. AMD / NVIDIA / Intel, все ОС |
| `ROCm` | `ROCmBackend` | Только HIP kernels (редко напрямую) |
| `OPENCLandROCm` | `HybridBackend` | Модули, использующие hipFFT / rocPRIM совместно с OpenCL |

> На AMD GPU **clFFT не тестируем** — не поддерживает RDNA4+ (gfx1201). Используем hipFFT через HybridBackend.

---

## Тесты

- **OpenCLBackend**: `tests/single_gpu.hpp` — `example_drv_gpu_singl::run()`
- **Внешний контекст**: `tests/example_external_context_usage.hpp` — `external_context_example::run()`
