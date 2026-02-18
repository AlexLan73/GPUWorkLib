# GPUProfiler: передача информации о GPU и вывод данных профилирования

> **⚠️ ВАЖНО для AI-ассистентов (Кодо и др.)**:
> 1. При любом использовании GPUProfiler с `PrintReport()` или `ExportMarkdown()` **ОБЯЗАТЕЛЬНО** вызывать `SetGPUInfo()` **ДО** `profiler.Start()`. Иначе в отчёте будет «GPU -1: Unknown» и «[нет информации о драйверах]».
> 2. **Вывод данных профилирования** — ТОЛЬКО через GPUProfiler: `PrintReport()`, `ExportMarkdown()`, `ExportJSON()`. ЗАПРЕЩЕНО вручную выводить через `GetStats()` + цикл + `con.Print` или `std::cout`.

---

## Зачем это нужно

GPUProfiler выводит отчёт с таблицей событий (Upload, FFT, Download и т.д.). В шапке отчёта отображаются:
- Название GPU (например, «NVIDIA GeForce RTX 3060»)
- Объём памяти (MB)
- Информация о драйверах (OpenCL версия, драйвер, vendor)

**Без вызова `SetGPUInfo()`** отчёт будет содержать:
```
GPU -1: Unknown
Драйверы: [нет информации о драйверах]
```

---

## Как применять

### 1. DrvGPU (стандартная инициализация)

```cpp
#include "drv_gpu.hpp"
#include "DrvGPU/services/gpu_profiler.hpp"

DrvGPU gpu(BackendType::OPENCL, 0);
gpu.Initialize();

auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
auto device_info = gpu.GetDeviceInfo();  // или gpu.GetBackend().GetDeviceInfo()

drv_gpu_lib::GPUReportInfo gpu_info;
gpu_info.gpu_name = device_info.name;
gpu_info.backend_type = BackendType::OPENCL;
gpu_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);

std::map<std::string, std::string> opencl_driver;
opencl_driver["driver_type"] = "OpenCL";
opencl_driver["version"] = device_info.opencl_version;
opencl_driver["driver_version"] = device_info.driver_version;
opencl_driver["vendor"] = device_info.vendor;
gpu_info.drivers.push_back(opencl_driver);

profiler.SetGPUInfo(0, gpu_info);  // gpu_id = 0
profiler.Start();
// ... выполнение GPU операций ...
profiler.PrintReport();
profiler.Stop();
```

### 2. External context (InitializeFromExternalContext)

При использовании внешнего OpenCL контекста `backend->GetDeviceIndex()` может возвращать `-1`. Нужно передать info **и для 0, и для -1**:

```cpp
auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
backend->InitializeFromExternalContext(context, device, queue);

int gpu_id = backend->GetDeviceIndex();
if (gpu_id < 0) gpu_id = 0;

auto device_info = backend->GetDeviceInfo();
drv_gpu_lib::GPUReportInfo gpu_info;
gpu_info.gpu_name = device_info.name.empty() ? "Unknown" : device_info.name;
gpu_info.backend_type = BackendType::OPENCL;
gpu_info.global_mem_mb = device_info.global_memory_size / (1024 * 1024);

std::map<std::string, std::string> opencl_driver;
opencl_driver["driver_type"] = "OpenCL";
opencl_driver["version"] = device_info.opencl_version;
opencl_driver["driver_version"] = device_info.driver_version;
opencl_driver["vendor"] = device_info.vendor;
gpu_info.drivers.push_back(opencl_driver);

profiler.SetGPUInfo(gpu_id, gpu_info);
if (backend->GetDeviceIndex() < 0) profiler.SetGPUInfo(-1, gpu_info);  // ← важно!

profiler.Start();
// ...
```

### 3. Краткий чек-лист

- [ ] Вызвать `SetGPUInfo(gpu_id, gpu_info)` **до** `profiler.Start()`
- [ ] `gpu_id` должен совпадать с тем, что использует `Record()` в коде (часто `backend->GetDeviceIndex()`)
- [ ] Для external context: `GetDeviceIndex()` может быть `-1` → дублировать `SetGPUInfo(-1, gpu_info)`
- [ ] `device_info` брать из `backend->GetDeviceInfo()` или `gpu.GetDeviceInfo()`

---

## Референс

- **Реализация**: `DrvGPU/backends/opencl/opencl_backend.cpp` — `QueryDeviceInfo()` (поддержка external context)
- **Пример в тестах**: `modules/fft_maxima/tests/test_batch_all_maxima.hpp` — `TestBatchWithProfiling()`
