# GPUWorkLib Modules

> Документация по модулям библиотеки GPU-вычислений

---

## Модули

| Модуль | Каталог | Статус | Описание |
|--------|---------|--------|----------|
| **Signal Generators** | [signal_generators/](signal_generators/) | Active | CW, LFM, Noise, Script, FormSignal, DelayedFormSignal |
| **FFT Processor** | [fft_processor/](fft_processor/) | Active | GPU FFT с режимами Complex / MagPhase / MagPhaseFreq |
| **FFT Maxima** | [fft_maxima/](fft_maxima/) | Active | Поиск максимумов спектра FFT (SpectrumMaximaFinder) |
| **Filters** | [filters/](filters/) | Active | FIR, IIR, SMA/EMA/DEMA/TEMA, Kalman, KAMA на GPU (OpenCL + ROCm) |
| **LchFarrow** | [lch_farrow/](lch_farrow/) | Active | Дробная задержка Lagrange 48×5 |
| **Heterodyne** | [heterodyne/](heterodyne/) | Active | Дечирп, stretch processing для ЛЧМ-радара (OpenCL + ROCm) |
| **FM Correlator** | [fm_correlator/](fm_correlator/) | Active | ФМ-корреляция с M-последовательностями в частотной области (ROCm) |
| **Python Bindings** | [python_bindings/](python_bindings/) | Active | pybind11 модуль `gpuworklib` для Python 3.12 |

---

## Архитектура

```
                    ┌──────────────────────┐
                    │   Python (pybind11)   │
                    └──────────┬───────────┘
                               │
     ┌─────────────────────────┼─────────────────────────┐
     │                         │                         │
     ▼                         ▼                         ▼
┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐
│   Signal     │  │  FFT Processor   │  │   FFT Maxima      │
│  Generators  │  │ (fft_processor)  │  │  (fft_maxima)     │
└──────┬───────┘  └────────┬─────────┘  └────────┬──────────┘
       │                   │                      │
┌──────┴───────┐  ┌────────┴─────────┐  ┌────────┴──────────┐
│   Filters    │  │   Heterodyne     │  │   LCH Farrow      │
│  (FIR/IIR/  │  │  (LFM Dechirp)   │  │ (Lagrange 5-pt)   │
│  MA/Kalman) │  │                  │  │                   │
└──────┬───────┘  └────────┬─────────┘  └────────┬──────────┘
       │                   │                      │
┌──────┴───────┐           │
│ FM Correlator│           │
│ (ROCm only)  │           │
└──────┬───────┘           │
       └───────────────────┼──────────────────────┘
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
- Сборка: CMake 3.20+, C++17, OpenCL 1.2+, clFFT / hipFFT (ROCm)

---

*Обновлено: 2026-03-04*
