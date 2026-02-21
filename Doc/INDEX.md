# Doc — Полный индекс

> Навигация по всей документации GPUWorkLib

---

## DrvGPU (ядро)

| Файл | Описание |
|------|----------|
| [DrvGPU/Quick.md](DrvGPU/Quick.md) | Краткий справочник |
| [DrvGPU/Architecture.md](DrvGPU/Architecture.md) | Архитектура, слои, паттерны |
| [DrvGPU/Memory.md](DrvGPU/Memory.md) | Система памяти (GPUBuffer, SVMBuffer) |
| [DrvGPU/OpenCL.md](DrvGPU/OpenCL.md) | OpenCL бэкенд |
| [DrvGPU/Command.md](DrvGPU/Command.md) | Command Queue |
| [DrvGPU/Classes.md](DrvGPU/Classes.md) | Справочник классов |

---

## Модули

| Модуль | Quick | Full | README |
|--------|-------|------|--------|
| **Signal Generators** | [Quick](Modules/signal_generators/Quick.md) | [Full](Modules/signal_generators/Full.md) | [README](Modules/signal_generators/README.md) |
| **FFT Processor** | [Quick](Modules/fft_processor/Quick.md) | [Full](Modules/fft_processor/Full.md) | [README](Modules/fft_processor/README.md) |
| **FFT Maxima** | — | [Full](Modules/fft_maxima/Full.md) | [README](Modules/fft_maxima/README.md) |
| **Filters** | [Quick](Modules/filters/Quick.md) | [Full](Modules/filters/Full.md) | [README](Modules/filters/README.md) |
| **LchFarrow** | [Quick](Modules/lch_farrow/Quick.md) | [Full](Modules/lch_farrow/Full.md) | [README](Modules/lch_farrow/README.md) |
| **Heterodyne** | — | [Full](Modules/heterodyne/Full.md) | Planned |
| **Python Bindings** | — | — | [README](Modules/python_bindings/README.md) |

---

## Python API

| Модуль | Документ |
|--------|----------|
| Signal Generators | [signal_generators_api.md](Python/signal_generators_api.md) |
| LchFarrow | [lch_farrow_api.md](Python/lch_farrow_api.md) |
| Spectrum Maxima | [spectrum_maxima_api.md](Python/spectrum_maxima_api.md) |

---

## Тесты

- **Python**: `Python_test/test_*.py`
- **C++**: `modules/*/tests/*.hpp` (вызов через `all_test.hpp`)

---

*Обновлено: 2026-02-17*
