# 📋 DISCREPANCIES — Heterodyne vs PLAN

> **Дата**: 2026-02-21
> **Файл**: разногласия между ПЛАНОМ и реализацией
> **Статус**: ✅ ВСЕ ИСПРАВЛЕНЫ (2026-02-21)
>
> **Намеренные изменения (НЕ разногласия)**:
> - N=8000, B=2MHz → физически корректнее (delay 500μs << T=667μs, μ=3e9 сохранён)
> - delays=[100,200,300,400,500]μs → соответствует плану
> - 7 тестов → план выполнен
> - OPT-1..OPT-6 → применены

---

## ✅ БАГ-1: OPT-3 в `Process()` — ИСПРАВЛЕН

Убран лишний `GenerateToGpu()` + `clReleaseMemObject(ref_gpu)` из `Process()`.
Теперь `Process()` использует только CPU ref path, `ProcessExternal()` — GPU ref path.

---

## ✅ РАЗ-1: `ProcessExternal()` не кеширует `conj_gen_` — ИСПРАВЛЕН

`ProcessExternal()` теперь использует `EnsureConjugateGenerator()` с проверкой изменения params.

---

## ✅ РАЗ-2: Python тесты — ВСЕ СОЗДАНЫ

| Файл | Статус |
|------|--------|
| `test_heterodyne.py` (4 базовых pytest) | ✅ |
| `test_heterodyne_step_by_step.py` (8 шагов + графики) | ✅ |
| `test_heterodyne_comparison.py` (GPU vs CPU отчёт) | ✅ |

---

## ✅ РАЗ-3: `Results/Plots/heterodyne/` — директория создана

Графики появятся после запуска Python тестов.

---

## ✅ РАЗ-4: `ALGORITHM_Heterodyne_LFM_Dechirp.md` — ОБНОВЛЁН

Параметры синхронизированы: B=2MHz, N=8000, 7 тестов, OPT-1..6, Python биндинги.

---

## ✅ РАЗ-5: `IN_PROGRESS.md` + `CLAUDE.md` — ОБНОВЛЕНЫ

- `IN_PROGRESS.md`: TASK-009 перенесён в COMPLETED
- `CLAUDE.md`: Heterodyne 🟡 → 🟢 Active
- `COMPLETED.md`: добавлена запись TASK-009

---

## ✅ РАЗ-6: Имя файла ядра — синхронизировано в ALGORITHM.md

ALGORITHM.md обновлён, использует фактическое имя `lfm_conjugate.cl`.

---

## 📋 Итог

| # | Разногласие | Статус |
|---|-------------|--------|
| БАГ-1 | OPT-3 в Process() — лишний GenerateToGpu | ✅ Исправлен |
| РАЗ-1 | ProcessExternal не кеширует conj_gen_ | ✅ Исправлен |
| РАЗ-2 | Создать step_by_step и comparison тесты | ✅ Создано |
| РАЗ-3 | Results/Plots/heterodyne/ | ✅ Директория есть |
| РАЗ-4 | Обновить ALGORITHM.md | ✅ Обновлён |
| РАЗ-5 | Обновить IN_PROGRESS.md + CLAUDE.md | ✅ Обновлено |
| РАЗ-6 | Синхронизировать имя cl файла в документации | ✅ Синхронизировано |

**Все разногласия устранены.** ✅

---

*Обновлено: 2026-02-21 | Кодо (AI Assistant)*
