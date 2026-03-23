# COMPLETED — Завершённые задачи

> **Обновлено**: 2026-03-23

---

## Task_14 — Python Test Bugs + C++ Test Migration ✅ 2026-03-23

### Python: 5 нерешённых багов → ALL FIXED
| # | Баг | Причина | Фикс |
|---|-----|---------|------|
| 5 | Нет биндинга LfmAnalyticalDelayROCm | Отсутствовал файл | `py_lfm_analytical_delay_rocm.hpp` + регистрация |
| 4 | FormSignal window tau_base=-0.1 | 2D shape (1,N) вместо 1D | `if (n_ant<=1) return 1D` |
| 3 | LchFarrow integer delay err=8.0 | float32: `3.0*1e-6*1e6≈3.000001` + hsaco cache | snap delay_samples к nearest int |
| 1 | SpectrumMaxima segfault | GC убивал ctx → dangling ref | `self._ctx` вместо `ctx` |
| 2 | Matrix CSV segfault | То же — GC + ctx | `self._ctx` |

### C++ Test Migration (TASK_CppTest_06) → 4 модуля на test_utils
| Модуль | Файлов | Строк: было→стало | Тесты |
|--------|--------|-------------------|-------|
| signal_generators | 2 | 713→400 (-44%) | 11/11 ✅ |
| fft_func | 4 | 1745→830 (-52%) | 23/23 ✅ |
| heterodyne | 3 | 1096→440 (-60%) | 11/11 ✅ |
| filters | 4 | 1764→860 (-51%) | 22/22 ✅ |
| **Итого** | **13** | **5318→2530 (-52%)** | **67/67 ✅** |

Убрано: 13 дублей helpers, 40+ raw hipMemcpy, 30+ per-test backend creation

---

## Task_13 — Strategies Pipeline ✅ 2026-03-12

- ProcessMagnitudeToBuffer, StrategiesFloatApi, AllocateManaged
- Parallel benchmark, Python 13/13 PASSED
- Связанный backlog-пункт «ProcessMagnitude + Statistics pipeline» (ранее без файла задачи): закрыт тем же контекстом + `test_process_magnitude_rocm`, см. `sessions/2026-03-12.md`

---

## 2026-03-09 — DrvGPU External Context Integration ✅

- ROCmBackend::InitializeFromExternalStream
- HybridBackend::InitializeFromExternalContexts
- DrvGPU Facade Static Factory Methods + 18 тестов

---

## Task_12 — Cholesky Optimize ✅ 2026-02-26

- 341×341 GpuKernel: 1.482→0.941 мс (-36.5%)
- Предаллокация d_info_, отложенная CheckInfo

## Task_11 — VectorAlgebra Cholesky v2 ✅ 2026-02-26

- SymmetrizeMode (Roundtrip/GpuKernel), RAII CholeskyResult, hiprtc
- C++ 23 PASSED, Python 6/6 PASSED
