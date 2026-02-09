# 🗂️ GPUWorkLib — Master Index

> **Проект**: Библиотеки GPU-вычислений (OpenCL, ROCm, HIP)
> **Автор**: Alex
> **AI-ассистент**: Кодо
> **Создано**: 2026-02-09

---

## 📦 Модули проекта

| Модуль | Статус | Спецификация | Описание |
|--------|--------|--------------|----------|
| **DrvGPU** | 🟢 Active | [specs/drvgpu.md](specs/drvgpu.md) | Базовый драйвер GPU |
| **FFT/IFFT** | 🟡 WIP | [specs/fft_module.md](specs/fft_module.md) | Быстрое преобразование Фурье |
| **Filters** | ⚪ Planned | [specs/filters_module.md](specs/filters_module.md) | ЦОС фильтры (FIR, IIR, адаптивные) |
| **Statistics** | ⚪ Planned | [specs/statistics_module.md](specs/statistics_module.md) | Статистическая обработка |
| **Heterodyne** | ⚪ Planned | [specs/heterodyne_module.md](specs/heterodyne_module.md) | Гетеродин (перенос частоты) |
| **SignalSynth** | ⚪ Planned | [specs/signal_synth_module.md](specs/signal_synth_module.md) | Синтезатор сигналов |

**Легенда**: 🟢 Active | 🟡 WIP | 🔴 Blocked | ⚪ Planned

---

## 📋 Задачи

| Файл | Описание |
|------|----------|
| [tasks/BACKLOG.md](tasks/BACKLOG.md) | Очередь задач (TODO) |
| [tasks/IN_PROGRESS.md](tasks/IN_PROGRESS.md) | Текущие задачи |
| [tasks/COMPLETED.md](tasks/COMPLETED.md) | Завершённые задачи |

---

## 📊 Changelog & Releases

| Файл | Описание |
|------|----------|
| [changelog/RELEASES.md](changelog/RELEASES.md) | Список релизов |
| [changelog/2026-02.md](changelog/2026-02.md) | Изменения за февраль 2026 |

---

## 🔬 Тесты & Бенчмарки

| Папка | Описание |
|-------|----------|
| [tests/benchmarks/](tests/benchmarks/) | Результаты производительности |
| [tests/validation/](tests/validation/) | Результаты валидации |

---

## 📚 Исследования & Документация

| Файл | Тема | Дата |
|------|------|------|
| [research/AMD_GPU_OpenCL_ROCm_ZeroCopy.md](research/AMD_GPU_OpenCL_ROCm_ZeroCopy.md) | ZeroCopy память AMD | 2026-02-06 |

> 💡 Сюда добавляются найденные материалы, исследования API, примеры кода

---

## 💬 Сессии с AI

| Файл | Описание |
|------|----------|
| [sessions/](sessions/) | История сессий с Кодо |

---

## 🔧 Быстрые команды для Кодо

```
"Покажи статус проекта"     → Этот файл + tasks/IN_PROGRESS.md
"Добавь задачу: ..."        → tasks/BACKLOG.md
"Запиши в спеку FFT: ..."   → specs/fft_module.md
"Сохрани исследование: ..." → research/
"Что мы сделали сегодня?"   → Создать sessions/YYYY-MM-DD.md
```

---

*Последнее обновление: 2026-02-09*
