# 🔍 Code Review: modules/filters

> **Дата**: 2026-03-22
> **Ревьюер**: Кодо (AI Assistant)
> **Объём**: ~29 файлов
> **Ветка**: main (ROCm 7.2+)

## ✅ ИСПРАВЛЕНО В ЭТОЙ СЕССИИ

| # | Тип | Описание | Файлы |
|---|-----|----------|-------|
| 1 | 🟡→✅ | Orphan OpenCL headers удалены (fir_filter.hpp, iir_filter.hpp, fir_kernels.hpp, iir_kernels.hpp) | 4 файла |
| 2 | 🟡→✅ | MakeROCmDataFromEvents → shared utility (исправлено ранее в сессии) | fir/iir_filter_rocm.cpp |
| 3 | 🟡→✅ | warp_size → ROCmCore::GetWarpSize() (исправлено ранее) | 3 filter .cpp |

---

## 📊 Общая оценка

| Аспект | Оценка |
|--------|--------|
| Архитектура | ⭐⭐⭐ Legacy (manual hiprtc, raw void*) |
| Фильтры | ⭐⭐⭐⭐ FIR, IIR, SMA/EMA/MMA/DEMA/TEMA, Kalman, Kaufman |
| Kernels | ⭐⭐⭐⭐ Корректные, __launch_bounds__ |
| Windows stubs | ✅ Все классы имеют stubs |
| Тесты | ✅ test_filters_rocm + 3 отдельных + benchmark |

## 🟡 Оставшиеся задачи

| # | Описание | Сложность |
|---|----------|-----------|
| 4 | FirFilterROCm → GpuContext (legacy hiprtc) | Средняя |
| 5 | IirFilterROCm → GpuContext | Средняя |
| 6 | MovingAverageFilterROCm → GpuContext | Средняя |
| 7 | KalmanFilterROCm → GpuContext | Средняя |
| 8 | KaufmanFilterROCm → GpuContext | Средняя |

Все 5 классов используют одинаковый паттерн manual hiprtc — миграция по аналогии с другими модулями.
