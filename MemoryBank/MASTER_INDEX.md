# GPUWorkLib — Master Index

> **Проект**: Библиотеки GPU-вычислений (OpenCL, ROCm, HIP)
> **Автор**: Alex
> **AI-ассистент**: Кодо
> **Обновлено**: 2026-03-23

---

## Модули

| Модуль | Статус | Документация |
|--------|--------|-------------|
| **DrvGPU** | 🟢 Active | `Doc/DrvGPU/Full.md` ✅ Quick.md ✅ API.md ✅ |
| signal_generators | 🟢 Active | `Doc/Modules/signal_generators/Full.md` |
| **fft_func** | 🟢 Active | `Doc/Modules/fft_func/Full.md` ✅ Quick.md ✅ API.md ✅ |
| filters | 🟢 Active | `Doc/Modules/filters/Full.md` |
| **lch_farrow** | 🟢 Active | `Doc/Modules/lch_farrow/Full.md` ✅ API.md ✅ |
| heterodyne | 🟢 Active | `Doc/Modules/heterodyne/Full.md` |
| statistics | 🟢 Active | `Doc/Modules/statistics/Full.md` |
| **vector_algebra** | 🟢 Active | `Doc/Modules/vector_algebra/Full.md` ✅ API.md ✅ |
| fm_correlator | 🟢 Active | `Doc/Modules/fm_correlator/Full.md` |
| **strategies** | 🟢 Active | `Doc/Modules/strategies/Full.md` ✅ Quick.md ✅ API.md ✅ |
| **capon** | 🟡 Framework Ready | `Doc/Modules/capon/Full.md` ✅ Quick.md ✅ API.md ✅ |
| **range_angle** | 🟡 Beta | `Doc/Modules/range_angle/Full.md` ✅ Quick.md ✅ API.md ✅ |

---

## Текущие задачи

| Задача | Описание | Статус |
|--------|----------|--------|
| **Task_14** | Python 5 bugs fixed + C++ test migration (4 модуля, -52% LOC) | ✅ COMPLETED 2026-03-23 |
| **CppTest migration** | Оставшиеся модули: lch_farrow, vector_algebra, fm_correlator, strategies | 🟢 Low priority |
| **capon_rocblas** | rocBLAS CGEMM в CovarianceMatrixOp | 🔴 TODO |
| **GPU+Mellanox Topology** | Авто-детектор пар GPU+NIC по PCIe слотам (сервер 6049GP-TRT) | 📋 PLANNED |

---

## C++ Test Infrastructure

| Компонент | Файл | Статус |
|-----------|------|--------|
| test_result.hpp | `modules/test_utils/` | ✅ Ready |
| test_configs.hpp | `modules/test_utils/` | ✅ Ready |
| validators/ | `modules/test_utils/validators/` | ✅ Ready |
| references/ | `modules/test_utils/references/` | ✅ Ready |
| gpu_transfer.hpp | `modules/test_utils/` | ✅ Ready |
| test_runner.hpp | `modules/test_utils/` | ✅ Ready |
| gpu_test_base.hpp | `modules/test_utils/` | ✅ Ready |

### Миграция модулей

| Модуль | Статус | Тесты |
|--------|--------|-------|
| statistics | ✅ Эталон | 21/21 |
| signal_generators | ✅ Мигрирован | 11/11 |
| fft_func | ✅ Мигрирован | 23/23 |
| heterodyne | ✅ Мигрирован | 11/11 |
| filters | ✅ Мигрирован | 22/22 |
| lch_farrow | ⬜ Pending | — |
| vector_algebra | ⬜ Pending | — |
| fm_correlator | ⬜ Pending | — |
| strategies | ⬜ Pending | — |
