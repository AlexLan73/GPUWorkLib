# COMPLETED — Завершённые задачи

> **Обновлено**: 2026-02-12

---

## 2026-02-12: ТЕМА 1 — API Refactoring

| Задача | Описание |
|--------|----------|
| T-API-REF-001 | InputData<T> объединён с ProcessingParams |
| T-API-REF-002 | Process() с 3 параметрами (input, mode, driver) |
| T-API-REF-003 | Удалён старый Process(vector&) |
| T-API-REF-004 | Обновлены все тесты (test_spectrum_maxima, test_large_batch, test_gpu_generator_integration) |
| T-API-REF-005 | GPUProfiler: экспорт в MD и JSON |
| T-API-REF-006 | Логирование перед/после Process() |
| T-API-REF-007 | Документация: guide + quick reference |

**Результат**: Новый API v2.0 работает!

---

## 2026-02-11: ТЕМА 2 — Batch Processing

| Задача | Описание |
|--------|----------|
| T-BATCH-001..010 | BatchManager интегрирован в SpectrumMaximaFinder |
| T-BATCH-011..016 | test_large_batch.hpp: 256×1.3M точек |
| T-BATCH-014 | GPUProfiler накопление по batch'ам |

**Результат**: Обработка 2.5GB данных за ~900ms (5 batch'ей)

---

## 2026-02-10: ТЕМА 3 — Kernel Refactoring

| Задача | Описание |
|--------|----------|
| KERN-01..06 | Суффикс `_opencl` для всех кернелов |
| KERN-07..08 | OnePeak кернел создан |
| KERN-09 | Исправлена амплитуда TwoPeaks |
| KERN-10..12 | PeakSearchMode интегрирован |

**Результат**: ONE_PEAK (4 MaxValue) и TWO_PEAKS (8 MaxValue)

---

## 2026-02-10: ТЕМА 4 — DrvGPU Optimization

| Задача | Описание |
|--------|----------|
| P1 | OpenCLBackend объединён с OpenCLBackendExternal |

**Результат**: Один backend для внутреннего и внешнего контекста

---

## 2026-02-09: Исправления

| ID | Задача |
|----|--------|
| C-001 | Исправлен Access Violation в clfftEnqueueTransform |
| C-002 | GPUProfiler: дельты вместо абсолютных значений |
| C-003 | Logger path (filesystem::path) |
| C-004 | Убран ThreadID из логов |
| C-005 | Logger связан с configGPU.json |

---

*Последнее обновление: 2026-02-12*
