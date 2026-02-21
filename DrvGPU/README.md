# DrvGPU — Базовый драйвер GPU

OpenCL backend, Multi-GPU, память, профилирование, логирование.

## Структура и README по папкам

| Папка | README | Описание |
|-------|--------|----------|
| [common/](common/README.md) | ✅ | BackendType, GPUDeviceInfo, LoadBalancingStrategy |
| [config/](config/README.md) | ✅ | GPUConfig, configGPU.json |
| [include/](include/README.md) | ✅ | DrvGPU, GPUManager, ModuleRegistry |
| [interface/](interface/README.md) | ✅ | IBackend, IComputeModule, ILogger, IMemoryBuffer |
| [logger/](logger/README.md) | ✅ | Logger, ConfigLogger, DefaultLogger |
| [services/](services/README.md) | ✅ | GPUProfiler, BatchManager, ConsoleOutput, ServiceManager |
| [memory/](memory/README.md) | ✅ | MemoryManager, GPUBuffer, SVM |
| [backends/](backends/README.md) | ✅ | Архитектура бэкендов |
| [backends/opencl/](backends/opencl/README.md) | ✅ | OpenCLBackend, OpenCLCore |
| [src/](src/README.md) | ✅ | Реализации DrvGPU, ModuleRegistry |
| [tests/](tests/README.md) | ✅ | all_test.hpp, single_gpu, test_services, test_gpu_profiler |

## Запуск тестов

```cpp
// src/main.cpp
#include "DrvGPU/tests/all_test.hpp"
drvgpu_all_test::run();
```

В `tests/all_test.hpp` раскомментировать нужные тесты.
