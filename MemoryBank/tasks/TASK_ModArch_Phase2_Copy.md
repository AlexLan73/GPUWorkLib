# 📦 Фаза 2: Копирование реального кода

> **Индекс**: [`TASK_Modular_Architecture_INDEX.md`](TASK_Modular_Architecture_INDEX.md)
> **Спецификация**: [`specs/modular_architecture_plan.md`](../specs/modular_architecture_plan.md) — раздел 7, Фаза 2
> **Статус**: ⬜ BACKLOG
> **Зависимость**: Фаза 1 должна быть ✅ DONE
> **Платформа**: Windows (копирование файлов, GPU не нужен)
> **Результат**: реальный код в 9 репо, без GPU тестирования

---

## Ключевые правила копирования

> - **Без git history** — простое копирование файлов, `git add -A`, `git commit`
> - **OpenCL `.cl` kernels НЕ копируем** — мы не считаем на OpenCL (только стыковка данных в GPU)
> - **Ветка `nvidia`** / OpenCL-специфичный код — не переносим
> - **`.claude/`** — только локально, не копируем
> - Все файлы из ветки `main` GPUWorkLib
> - **MemoryBank** — остаётся в GPUWorkLib, в DSP не переносится

---

## Чеклист

### 📂 core ← DrvGPU/

- [ ] **C1.1** Скопировать содержимое `GPUWorkLib/DrvGPU/`:
  ```
  DrvGPU/include/  → core/include/dsp/
  DrvGPU/src/      → core/src/
  DrvGPU/config/   → core/src/config/
  ```
- [ ] **C1.2** Скопировать `configGPU.json` → `core/configGPU.json`
- [ ] **C1.3** Скопировать `modules/test_utils/` → `core/test_utils/`
  > Это отдельная target `DspCore::TestUtils`, не публичный API!
- [ ] **C1.4** Скопировать **только DrvGPU-часть** биндингов:
  ```
  # Из gpu_worklib_bindings.cpp берём ТОЛЬКО блок DrvGPU:
  # class GPUContext, DrvGPU init/cleanup/buffer methods
  # → core/python/dsp_core_module.cpp
  ```
  > ⚠️ НЕ копировать весь файл целиком! Он содержит FFTProcessor, StatisticsProcessor и т.д. —
  > они не скомпилируются в core/ (нет заголовков spectrum/stats).
  > Полный разбив биндингов по репо — в Фазе 3b.
- [ ] **C1.5** Скопировать `DrvGPU/tests/` → `core/tests/`
- [ ] **C1.6** git add + commit: `"feat: initial DrvGPU source from GPUWorkLib"`
- [ ] **C1.7** Push

---

### 📂 spectrum ← fft_func + filters + lch_farrow

> ⚠️ statistics НЕ сюда — оно идёт в репо `stats`!

- [ ] **C2.1** `modules/fft_func/` → `spectrum/src/fft_func/`
- [ ] **C2.2** `modules/fft_func/include/` → `spectrum/include/dsp/spectrum/`
- [ ] **C2.3** `modules/filters/` → `spectrum/src/filters/`
- [ ] **C2.4** `modules/filters/include/` → `spectrum/include/dsp/spectrum/`
- [ ] **C2.5** `modules/lch_farrow/` → `spectrum/src/lch_farrow/`
- [ ] **C2.6** `modules/lch_farrow/include/` → `spectrum/include/dsp/spectrum/`
- [ ] **C2.7** Скопировать `.hip` kernel файлы → `spectrum/kernels/rocm/`
  > `.cl` файлы НЕ копируем!
- [ ] **C2.8** Скопировать C++ тесты fft_func / filters / lch_farrow → `spectrum/tests/`
- [ ] **C2.9** git commit + push

---

### 📂 stats ← statistics

- [ ] **C3.1** `modules/statistics/` → `stats/src/statistics/`
- [ ] **C3.2** `modules/statistics/include/` → `stats/include/dsp/stats/`
- [ ] **C3.3** `.hip` kernel файлы statistics → `stats/kernels/rocm/`
- [ ] **C3.4** Скопировать тесты → `stats/tests/`
- [ ] **C3.5** git commit + push

---

### 📂 signal_generators ← signal_generators

- [ ] **C4.1** `modules/signal_generators/` → `signal_generators/src/`
- [ ] **C4.2** `modules/signal_generators/include/` → `signal_generators/include/dsp/signal_generators/`
- [ ] **C4.3** `.hip` kernel файлы → `signal_generators/kernels/rocm/`
- [ ] **C4.4** Скопировать тесты → `signal_generators/tests/`
- [ ] **C4.5** Проверить: нет ли в include прямых путей к `lch_farrow/` (должны через API spectrum)
- [ ] **C4.6** git commit + push

---

### 📂 heterodyne ← heterodyne

- [ ] **C5.1** `modules/heterodyne/` → `heterodyne/src/`
- [ ] **C5.2** `modules/heterodyne/include/` → `heterodyne/include/dsp/heterodyne/`
- [ ] **C5.3** `.hip` kernel файлы → `heterodyne/kernels/rocm/`
- [ ] **C5.4** Скопировать тесты → `heterodyne/tests/`
- [ ] **C5.5** Проверить: нет ли прямых include из signal_generators без cmake-зависимости
- [ ] **C5.6** git commit + push

---

### 📂 linalg ← vector_algebra + capon

- [ ] **C6.1** `modules/vector_algebra/` → `linalg/src/vector_algebra/`
- [ ] **C6.2** `modules/vector_algebra/include/` → `linalg/include/dsp/linalg/`
- [ ] **C6.3** `modules/capon/` → `linalg/src/capon/`
- [ ] **C6.4** `modules/capon/include/` → `linalg/include/dsp/linalg/`
- [ ] **C6.5** `.hip` kernel файлы (vector_algebra + capon) → `linalg/kernels/rocm/`
- [ ] **C6.6** Скопировать тесты обоих модулей → `linalg/tests/`
- [ ] **C6.7** git commit + push

---

### 📂 radar ← range_angle + fm_correlator

- [ ] **C7.1** `modules/range_angle/` → `radar/src/range_angle/`
- [ ] **C7.2** `modules/range_angle/include/` → `radar/include/dsp/radar/`
- [ ] **C7.3** `modules/fm_correlator/` → `radar/src/fm_correlator/`
- [ ] **C7.4** `modules/fm_correlator/include/` → `radar/include/dsp/radar/`
- [ ] **C7.5** `.hip` kernel файлы → `radar/kernels/rocm/`
- [ ] **C7.6** Скопировать тесты → `radar/tests/`
- [ ] **C7.7** git commit + push

---

### 📂 strategies ← strategies

- [ ] **C8.1** `modules/strategies/` → `strategies/src/`
- [ ] **C8.2** `modules/strategies/include/` → `strategies/include/dsp/strategies/`
- [ ] **C8.3** `.hip` kernel файлы → `strategies/kernels/rocm/`
- [ ] **C8.4** Скопировать тесты → `strategies/tests/`
- [ ] **C8.5** git commit + push

---

### 📂 DSP/Python ← Python_test/

> ⚠️ `Python_test/common/gpu_loader.py` импортирует `gpuworklib` (один модуль).
> После Фазы 3b это изменится на 8 отдельных `.pyd`. Пока копируем как есть с пометкой.

- [ ] **C9.1** Переименовать папки при копировании:
  ```
  Python_test/common/              → DSP/Python/common/
  Python_test/fft_maxima/          → DSP/Python/spectrum/  (частично)
  Python_test/filters/             → DSP/Python/spectrum/  (частично)
  Python_test/signal_generators/   → DSP/Python/signal_generators/
  Python_test/heterodyne/          → DSP/Python/heterodyne/
  Python_test/statistics/          → DSP/Python/stats/
  Python_test/lch_farrow/          → DSP/Python/spectrum/  (частично)
  Python_test/vector_algebra/      → DSP/Python/linalg/    (частично)
  Python_test/integration/         → DSP/Python/integration/
  Python_test/strategies/          → DSP/Python/strategies/
  ```
- [ ] **C9.2** Создать пустую `DSP/Python/lib/` + добавить в `.gitignore`
- [ ] **C9.3** Проверить что в тестах нет абсолютных путей к GPUWorkLib:
  ```bash
  grep -r "E:/C++" DSP/Python/
  grep -r "C:\\\\C++" DSP/Python/
  ```
- [ ] **C9.4** Добавить в `DSP/Python/common/gpu_loader.py` комментарий:
  ```python
  # TODO (Фаза 3b): переписать под 8 отдельных .pyd модулей
  # import dsp_core, dsp_spectrum, dsp_stats, ...
  ```
- [ ] **C9.5** git commit + push

---

### 📂 DSP/Doc ← Doc/ + Doc_Addition/

- [ ] **C10.1** `GPUWorkLib/Doc/` → `DSP/Doc/` (архитектурная документация)
- [ ] **C10.2** `GPUWorkLib/Doc_Addition/` → `DSP/Doc/addition/` (потом почистим)
- [ ] **C10.3** **MemoryBank остаётся в GPUWorkLib** — в DSP не переносить
- [ ] **C10.4** git commit + push

---

## Что НЕ копируем (финальный список)

| Что | Почему |
|-----|--------|
| `DrvGPU/backends/opencl/kernels/*.cl` | Только стыковка данных, не вычисления |
| `modules/*/kernels/*.cl` | ROCm kernels в `.hip` / inline — `.cl` не нужны |
| `.claude/` | Локально, не в GitHub |
| `.claude/worktrees/` | Временные |
| `build/` | Артефакты сборки |
| `nvidia` ветка | Только ROCm main |
| `src/main.cpp` | Заменяется examples/ |
| `Logs/`, `Results/` | gitignore |
| `MemoryBank/` | Остаётся в GPUWorkLib |
| `gpu_worklib_bindings.cpp` (весь файл) | Разбиваем в Фазе 3b — брать только DrvGPU-часть |

---

## Definition of Done

- [ ] Все 8 репо (core, spectrum, stats, linalg, signal_gen, heterodyne, radar, strategies) содержат реальный код
- [ ] Нет OpenCL `.cl` kernel файлов: `find . -name "*.cl"` → пусто
- [ ] Нет абсолютных путей в Python-тестах: `grep -r "E:/C++" DSP/Python/` → пусто
- [ ] `.hip` kernel файлы скопированы в `kernels/rocm/` каждого репо
- [ ] `gpu_loader.py` содержит TODO-пометку для Фазы 3b
- [ ] Можно переходить к Фазе 3

---

*Создан: 2026-04-12 | Обновлён: 2026-04-12 (исправлен C1.4, добавлены kernels/rocm/, закрыт вопрос MemoryBank, добавлен TODO gpu_loader) | Автор: Кодо*
