# Signal Generators Module

> Генерация сигналов на GPU (OpenCL) с поддержкой CPU reference

**Namespace**: `signal_gen`
**Каталог**: `modules/signal_generators/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL

---

## Содержание

| Файл | Описание |
|------|----------|
| [Architecture.md](Architecture.md) | Архитектура модуля, паттерны |
| [API.md](API.md) | Полный API Reference |
| [ScriptGenerator.md](ScriptGenerator.md) | Text DSL -> OpenCL kernel compiler |

---

## Обзор

Модуль предоставляет генерацию сигналов на GPU через единый интерфейс `ISignalGenerator` (Strategy pattern):

| Генератор | Класс | Формула |
|-----------|-------|---------|
| **CW** | `CwGenerator` | `s(t) = A * exp(j*(2*pi*f*t + phase))` |
| **LFM** | `LfmGenerator` | `s(t) = A * exp(j*pi*k*t^2 + j*2*pi*f_start*t)` |
| **Noise** | `NoiseGenerator` | Gaussian / White (Philox-2x32 + Box-Muller) |
| **Script** | `ScriptGenerator` | Text DSL -> OpenCL kernel at runtime |

Дополнительно:
- **SignalGeneratorFactory** — фабрика генераторов (Factory pattern)
- **SignalService** — фасад для генерации CPU/GPU (Facade pattern)

---

## Быстрый старт

### C++

```cpp
#include "signal_service.hpp"

// Создать сервис
signal_gen::SignalService service(backend);

// CW сигнал: 100 Hz, шаг 10 Hz между лучами
signal_gen::CwParams cw{.f0 = 100.0, .freq_step = 10.0};
signal_gen::SystemSampling sys{.fs = 1000.0, .length = 4096};

// CPU (1 луч)
auto cpu_data = service.GenerateCpu(cw, sys);

// GPU (8 лучей) — возвращает cl_mem
cl_mem gpu_data = service.GenerateGpu(cw, sys, 8);
// ... использовать gpu_data ...
clReleaseMemObject(gpu_data);
```

### Python

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.SignalGenerator(ctx)

# CW: 256 лучей, 4096 точек, 100 Hz + шаг 10 Hz
data = gen.generate_cw(256, 4096, 1000.0, f0=100.0, freq_step=10.0)
# data.shape = (256, 4096), dtype = complex64
```

---

## Файлы

```
modules/signal_generators/
├── include/
│   ├── i_signal_generator.hpp          # ISignalGenerator interface
│   ├── signal_generator_factory.hpp    # Factory
│   ├── signal_service.hpp              # Facade
│   ├── params/
│   │   ├── system_sampling.hpp         # SystemSampling {fs, length}
│   │   └── signal_request.hpp          # CwParams, LfmParams, NoiseParams, SignalRequest
│   └── generators/
│       ├── cw_generator.hpp            # CW generator
│       ├── lfm_generator.hpp           # LFM generator
│       ├── noise_generator.hpp         # Noise generator
│       └── script_generator.hpp        # Script DSL generator
├── src/
│   ├── cw_generator.cpp
│   ├── lfm_generator.cpp
│   ├── noise_generator.cpp
│   ├── script_generator.cpp
│   ├── signal_generator_factory.cpp
│   └── signal_service.cpp
└── CMakeLists.txt
```

---

*Обновлено: 2026-02-13*
