# GPUWorkLib Modules

> Документация по модулям библиотеки GPU-вычислений

---

## Модули

| Модуль | Каталог | Статус | Описание |
|--------|---------|--------|----------|
| **Signal Generators** | [signal_generators/](signal_generators/) | Active | CW, LFM, Noise, Script генераторы сигналов на GPU |
| **FFT Processor** | [fft_processor/](fft_processor/) | Active | GPU FFT с режимами Complex / MagPhase / MagPhaseFreq |
| **FFT Maxima** | [fft_maxima/](fft_maxima/) | Active | Поиск максимумов спектра FFT (SpectrumMaximaFinder) |
| **Python Bindings** | [python_bindings/](python_bindings/) | Active | pybind11 модуль `gpuworklib` для Python 3.12 |

---

## Архитектура

```
                    ┌──────────────────────┐
                    │   Python (pybind11)   │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Signal Generators│ │  FFT Processor   │ │   FFT Maxima     │
│ (signal_gen)     │ │ (fft_processor)  │ │ (antenna_fft)    │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │      DrvGPU       │
                    │   (IBackend*)     │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   OpenCL / ROCm   │
                    └───────────────────┘
```

## Зависимости

- Все модули зависят от **DrvGPU** (через `IBackend*`)
- DrvGPU: см. [../DrvGPU/](../DrvGPU/)
- Сборка: CMake 3.20+, C++17, OpenCL 1.2+, clFFT

---

*Обновлено: 2026-02-13*
