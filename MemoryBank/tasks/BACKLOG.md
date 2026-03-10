# BACKLOG — Очередь задач

> **Обновлено**: 2026-03-10
> **Главный план**: `MemoryBank/tasks/MODULES_WORK_PLAN.md`

---

## Приоритет 🔴 Высокий

### TASK-02 — fft_processor: Python Binding
- Нет `py_fft_processor.hpp` и `py_fft_processor_rocm.hpp`
- Нет Python тестов
- Детали: `MODULES_WORK_PLAN.md#TASK-02`

### TASK-03 — fft_maxima: Python Binding
- Есть `Doc/Python/spectrum_maxima_api.md` (план), но нет реального binding
- Нет `py_spectrum_maxima_finder.hpp` / `_rocm.hpp`
- Зависит от TASK-02
- Детали: `MODULES_WORK_PLAN.md#TASK-03`

---

## Приоритет 🟠 Выше среднего

### TASK-05 — fm_correlator: Тесты + API.md
- Все тесты в `all_test.hpp` закомментированы
- Нет `Doc/Modules/fm_correlator/API.md`
- Нет Python тестов
- Детали: `MODULES_WORK_PLAN.md#TASK-05`

---

## Приоритет 🟡 Средний

### TASK-01 — drvgpu: Раскомментировать External Context тесты
- 18 тестов написаны, закомментированы в `all_test.hpp`
- Быстро (30 мин)
- Детали: `MODULES_WORK_PLAN.md#TASK-01`

### TASK-04 — signal_generators: FormSignalROCm Python Binding
- Нет `py_form_signal_rocm.hpp`
- FormSignalROCm используется в strategies — нужен Python доступ для тестов
- Детали: `MODULES_WORK_PLAN.md#TASK-04`

---

## Перспективные задачи

- `strategies`: подробный task-пакет на реализацию ROCm архитектуры
  - Главный файл: `MemoryBank/tasks/STRATEGIES_ROCM_EXECUTION.md`
  - Статус: 85% готово, осталось SetExternalWeights + GPU запуск + графики

---

## Готовые модули (не требуют работы)

| Модуль | C++ | Python | Docs |
|--------|-----|--------|------|
| statistics | ✅ | ✅ | ✅ |
| lch_farrow | ✅ | ✅ | ✅ |
| vector_algebra | ✅ | ✅ | ✅ |
| filters | ✅ | ✅ | ✅ |
| heterodyne | ✅ | ✅ | ✅ |

*Последнее обновление: 2026-03-10*
