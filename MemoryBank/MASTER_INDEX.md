# GPUWorkLib — Master Index

> **Проект**: Библиотеки GPU-вычислений (OpenCL, ROCm, HIP)
> **Автор**: Alex
> **AI-ассистент**: Кодо
> **Обновлено**: 2026-03-17

---

## Модули

| Модуль | Статус | Документация |
|--------|--------|-------------|
| **DrvGPU** | 🟢 Active | `Doc/DrvGPU/Full.md` ✅ Quick.md ✅ API.md ✅ |
| signal_generators | 🟢 Active | `Doc/Modules/signal_generators/Full.md` |
| **fft_func** | 🟢 Active | `Doc/Modules/fft_func/Full.md` ✅ Quick.md ✅ API.md ✅ |
| ~~fft_processor~~ | ⚫ Merged → fft_func | `Doc/Modules/~!/fft_processor/Full.md` (архив) |
| ~~fft_maxima~~ | ⚫ Merged → fft_func | `Doc/Modules/~!/fft_maxima/Full.md` (архив) |
| filters | 🟢 Active | `Doc/Modules/filters/Full.md` |
| **lch_farrow** | 🟢 Active | `Doc/Modules/lch_farrow/Full.md` ✅ API.md ✅ |
| heterodyne | 🟢 Active | `Doc/Modules/heterodyne/Full.md` |
| statistics | 🟢 Active | `Doc/Modules/statistics/Full.md` |
| **vector_algebra** | 🟢 Active | `Doc/Modules/vector_algebra/Full.md` ✅ API.md ✅ |
| fm_correlator | 🟢 Active | `Doc/Modules/fm_correlator/Full.md` |
| **strategies** | 🟢 Active | `Doc/Modules/strategies/Full.md` ✅ Quick.md ✅ API.md ✅ |
| **capon** | 🟡 Framework Ready | `Doc/Modules/capon/Full.md` ✅ Quick.md ✅ API.md ✅ |

---

## Текущие задачи (см. MemoryBank/tasks/)

| Задача | Описание | Статус |
|--------|----------|--------|
| **Task_13** | Strategies Pipeline: fft_func + ProcessMagnitudeToBuffer + CPU wrappers + AllocateManaged + benchmark | ✅ COMPLETED 2026-03-12 |

| **capon_rocblas** | rocBLAS CGEMM в CovarianceMatrixOp / CaponReliefOp / AdaptBeamformOp | 🔴 TODO |

*Следующий шаг: реализация rocBLAS CGEMM (нужен rocblas_handle из backend) или Python bindings.*
