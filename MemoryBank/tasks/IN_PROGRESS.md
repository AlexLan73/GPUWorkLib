# IN PROGRESS — Текущие задачи

> **Обновлено**: 2026-02-26
> **Фокус**: ROCm Backend — Оптимизация

---

## В работе

- **ROCm Backend** — см. [PLAN_AMD_Radeon_9070_ROCm.md](PLAN_AMD_Radeon_9070_ROCm.md)
- **Аудит модулей** — Python тесты FFTProcessor, документация Statistics/FFTProcessor — см. [PLAN_modules_audit_2026-02-28.md](PLAN_modules_audit_2026-02-28.md)

## Завершено (2026-02-26)

- ✅ **Task_12: Cholesky <1.0ms** — оптимизация CholeskyInverterROCm 341×341
  - Результат: **0.941 мс** (было 1.482 мс, цель <1.0 мс) — **-36.5%**
  - Оптимизации: предаллокация dev_info, отложенная CheckInfo, убран лишний Sync
  - Сессия: `sessions/2026-02-26_Task12_Cholesky_Optimize.md`

- ✅ **Task_11 v2: VectorAlgebra Cholesky** — полностью завершён
  - C++ 23 PASSED, Python 6/6 PASSED
  - Benchmark: hipEvent GPU timing, MD отчёт
  - Отчёт: `Results/Profiler/cholesky_benchmark_2026-02-26.md`

---

## Следующие кандидаты (BACKLOG)

См. [BACKLOG.md](BACKLOG.md)

---

*Последнее обновление: 2026-02-28*
