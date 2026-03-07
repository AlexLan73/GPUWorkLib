# BACKLOG — Очередь задач

> **Обновлено**: 2026-03-07
> **Фокус**: ROCm

---

## Перспективные задачи

- `strategies`: подробный task-пакет на реализацию ROCm архитектуры
  - См. `MemoryBank/tasks/STRATEGIES_ROCM_EXECUTION.md`
  - Базовая схема: `d_S on GPU -> GEMM -> Window+FFT -> Step2.1/2.2/2.3`
  - Debug точки: `2.1`, `2.2`, `2.3`
  - Weight matrix: Delay-and-sum, auto + external, C++ + Python
  - Post-FFT поиск: в `modules/fft_maxima/`
  - Post-FFT statistics: в `modules/statistics/`

*Последнее обновление: 2026-03-07*
