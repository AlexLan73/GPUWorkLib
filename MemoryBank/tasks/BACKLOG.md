# BACKLOG — Очередь задач

> **Обновлено**: 2026-02-23
> **Фокус**: ROCm

---

## Перспективные задачи

### Единый механизм профилирования GPU (Средний приоритет) ← НОВЫЙ

**Цель**: Стандартизировать профилирование через `GpuBenchmarkBase` + Template Method Pattern

| Задача | Описание |
|--------|----------|
| **PROF-001** | Реализовать `GpuBenchmarkBase` + `FFTProcessorBenchmark` (OpenCL) — [подробнее](TASK_profiling_fftprocessor.md) |
| PROF-002 | Statistics → `StatisticsBenchmark` (ROCm) |
| PROF-003 | Filters → `FiltersBenchmark` (OpenCL) |
| PROF-004 | Heterodyne → `HeterodyneBenchmark` (OpenCL) |
| PROF-005 | `Doc/Profiling_Migration_Guide.md` — гайд для новых модулей |

**Спека**: `specs/Profil_GPU.md`
**Начать с**: PROF-001 — `modules/fft_processor`

---

### ROCm Backend (Средний приоритет) — **В РАБОТЕ**

**Цель**: Добавить поддержку AMD GPU через ROCm/HIP

| Задача | Описание |
|--------|----------|
| ROCM-001 | Создать `SpectrumProcessorROCm` (hipFFT) |
| ROCM-002 | Интегрировать в SpectrumMaximaFinder через Strategy Pattern |
| ROCM-003 | Тесты на AMD GPU |

**Зависимости**: Нужен доступ к AMD GPU  
**План**: [PLAN_AMD_Radeon_9070_ROCm.md](PLAN_AMD_Radeon_9070_ROCm.md)

---

### Filters Stage 2 (Средний приоритет)

**Цель**: Text→kernel pipeline (FormScriptGenerator-like)

- Кэш скомпилированных kernel на диск
- См. [Doc/Modules/filters/Full.md](../../Doc/Modules/filters/Full.md)

---

### Code Style (Низкий приоритет)

**Цель**: Google C++ Style + 2-пробельная табуляция

**Сфера**: `DrvGPU/`

---

## Отложенные (после основного функционала)

| Задача | Описание |
|--------|----------|
| ~~Statistics модуль~~ | ~~mean, std, variance на GPU~~ **РЕАЛИЗОВАНО** (ROCm) — см. `modules/statistics/` |
| Heterodyne NCO/MixDown | Планируется (LFM Dechirp уже реализован) |
| Overlap-Save/Add | Длинные FIR через FFT |
| AI-003 | Multi-step AI pipeline (LangChain/AutoGen) |

---

*Последнее обновление: 2026-02-23*
