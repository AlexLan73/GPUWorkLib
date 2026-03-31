# GPUWorkLib — Architecture Documentation Index

> **Date**: 2026-03-28
> **Author**: Кодо (AI Assistant)
> **Notation**: C4 Model + DFD + UML Sequence Diagrams

---

## Документы

| # | Документ | Уровень | Описание |
|---|----------|---------|----------|
| 1 | [C1 — System Context](Architecture_C1_SystemContext.md) | Самый высокий | Акторы, внешние системы, границы GPUWorkLib |
| 2 | [C2 — Container Diagram](Architecture_C2_Container.md) | Контейнеры | DrvGPU, модули, биндинги, зависимости |
| 3 | [C3 — Component Diagram](Architecture_C3_Component.md) | Компоненты | Классы внутри каждого контейнера |
| 4 | [C4 — Code Diagram](Architecture_C4_Code.md) | Код | Интерфейсы, сигнатуры, UML |
| 5 | [DFD — Data Flow Diagram](Architecture_DFD.md) | Потоки данных | Level 0/1/2 + pipelines |
| 6 | [Seq — Sequence Diagrams](Architecture_Seq.md) | Сценарии | 6 диаграмм последовательностей |

---

## Quick Reference: Sequence Diagrams

| # | Сценарий | Участники |
|---|----------|-----------|
| Seq-1 | DrvGPU Initialization | DrvGPU → GPUConfig → OpenCLBackend → Logger |
| Seq-2 | Signal → FFT → Peak | Factory → CwGen → FFTProcessor → SpectrumProc |
| Seq-3 | Heterodyne Dechirp | HeterodyneDechirp → LfmConjGen → FFT → Maxima |
| Seq-4 | Python API Usage | GPUContext → PySigGen → PyFFT → PyHeterodyne |
| Seq-5 | Multi-GPU Batch | BatchManager → DrvGPU[0..N] → Merge |
| Seq-6 | Profiling & Export | Module → GPUProfiler → AsyncQueue → FileSystem |

---

## Модули системы (текущее состояние)

| # | Модуль | Backend | Статус | Описание |
|---|--------|---------|--------|----------|
| 0 | **DrvGPU** | OpenCL / ROCm / Hybrid | 🟢 Active | Ядро: backends, память, profiler, services |
| 1 | **fft_func** | ROCm (hipFFT) | 🟢 Active | Пакетный FFT + поиск максимумов спектра |
| 2 | **Statistics** | ROCm only | 🟢 Active | Welford mean/var, медиана, radix sort |
| 3 | **Vector Algebra** | ROCm (rocsolver) | 🟢 Active | Cholesky POTRF/POTRI инверсия |
| 4 | **Filters** | ROCm (HIP) | 🟢 Active | FIR, IIR, SMA/EMA/DEMA/TEMA, Kalman, KAMA |
| 5 | **Signal Generators** | OpenCL / ROCm | 🟢 Active | CW, LFM, Noise, FormSignal, DelayedFormSignal |
| 6 | **LCH Farrow** | OpenCL / ROCm | 🟢 Active | Дробная задержка Lagrange 48×5 |
| 7 | **Heterodyne** | OpenCL / ROCm | 🟢 Active | LFM Dechirp → beat freq → range |
| 8 | **FM Correlator** | ROCm (hipFFT) | 🟢 Active | M-seq LFSR, cyclic shifts, freq-domain correlation |
| 9 | **Strategies** | ROCm (hipBLAS, hipFFT) | 🟢 Active | Цифровое ДН: CGEMM → FFT → post-FFT scenarios |
| 10 | **Capon** | ROCm (rocBLAS, rocsolver) | 🟡 Framework | MVDR: R=YY^H → Cholesky → relief / beamform |
| 11 | **Range Angle** | ROCm | 🟡 Beta | 3D: dechirp → range FFT → 2D beam FFT → peak |

## Предыдущие документы (справочные)

| Документ | Описание |
|----------|----------|
| DrvGPU_Design_C4.md | C4 только для DrvGPU (ранняя версия, удалён) |
| GPUWorkLib_Design_C4_Full.md | Предыдущая полная C4 (до Statistics/VectorAlgebra, удалён) |
| Disane C4.md | Справочный пример C4-модели (удалён) |

---

## Рендеринг PlantUML

Все документы содержат блоки `plantuml` которые можно отрендерить:
- **VS Code**: расширение PlantUML (jebbs.plantuml)
- **Online**: [plantuml.com/plantuml](https://www.plantuml.com/plantuml)
- **CLI**: `java -jar plantuml.jar Architecture_*.md`

---

*Maintained by: Кодо (AI Assistant) | Last updated: 2026-03-28*
