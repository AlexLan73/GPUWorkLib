# DrvGPU/common — Общие типы и утилиты

Общие типы, перечисления и структуры, используемые во всём DrvGPU.

## Классы и типы

| Файл | Описание |
|------|----------|
| `backend_type.hpp` | `BackendType` — enum типов бэкендов (OPENCL, ROCm, AUTO). `BackendTypeToString()` |
| `gpu_device_info.hpp` | `GPUDeviceInfo` — структура с информацией о GPU (name, vendor, memory, compute units, SVM, etc.). `ToString()`, `GetGlobalMemoryGB()` |
| `load_balancing.hpp` | `LoadBalancingStrategy` — enum стратегий балансировки (ROUND_ROBIN, LEAST_LOADED, MANUAL, FASTEST_FIRST). `LoadBalancingStrategyToString()` |

## Использование

```cpp
#include "common/backend_type.hpp"
#include "common/gpu_device_info.hpp"
#include "common/load_balancing.hpp"

DrvGPU gpu(BackendType::OPENCL, 0);
auto info = gpu.GetDeviceInfo();
std::cout << info.name << " " << info.GetGlobalMemoryGB() << " GB\n";
```

## Как тестировать

- **GPUDeviceInfo**: `tests/single_gpu.hpp` — `example_drv_gpu_singl::run()` — вывод `PrintDeviceInfo()`, `GetDeviceInfo()`
- **BackendType**: `tests/single_gpu.hpp` — инициализация `DrvGPU(BackendType::OPENCL, 0)`
- **LoadBalancingStrategy**: `include/gpu_manager.hpp` — `GPUManager` использует при выборе GPU
