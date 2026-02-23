# GPUWorkLib — Architecture Documentation Index

> **Date**: 2026-02-23
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

## Предыдущие документы (справочные)

| Документ | Описание |
|----------|----------|
| [DrvGPU_Design_C4.md](DrvGPU_Design_C4.md) | C4 только для DrvGPU (ранняя версия) |
| [GPUWorkLib_Design_C4_Full.md](GPUWorkLib_Design_C4_Full.md) | Предыдущая полная C4 (до Heterodyne/Farrow) |
| [Disane C4.md](Disane%20C4.md) | Справочный пример C4-модели |

---

## Рендеринг PlantUML

Все документы содержат блоки `plantuml` которые можно отрендерить:
- **VS Code**: расширение PlantUML (jebbs.plantuml)
- **Online**: [plantuml.com/plantuml](https://www.plantuml.com/plantuml)
- **CLI**: `java -jar plantuml.jar Architecture_*.md`

---

*Maintained by: Кодо (AI Assistant) | Last updated: 2026-02-23*
