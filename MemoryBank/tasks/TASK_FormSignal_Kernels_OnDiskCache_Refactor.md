# Задача: Вынос kernels в .cl + On-disk cache в DrvGPU

> **Источник:** [FormSignalGenerator_Разногласия.md](FormSignalGenerator_Разногласия.md) — дополнения Alex (131–147)  
> **Проверка:** Главный AI (Кодо) проверяет результат — компиляцию, тесты, соответствие спецификации.

---

## 1. ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ

| № | Файл | Зачем |
|---|------|-------|
| 1 | `CLAUDE.md` | Правила проекта, структура |
| 2 | `MemoryBank/tasks/FormSignalGenerator_Разногласия.md` | Исходные дополнения (разделы 1, 6, 7 в конце) |
| 3 | `MemoryBank/specs/Form_signals.md` | Раздел 5 — расположение kernels, раздел 12 — on-disk кэш |
| 4 | `modules/signal_generators/kernels/README.md` | Текущая структура kernels |
| 5 | `modules/signal_generators/src/form_signal_generator.cpp` | Inline kernel — нужно вынести |
| 6 | `modules/signal_generators/src/form_script_generator.cpp` | On-disk cache — логика |
| 7 | `modules/signal_generators/src/delayed_form_signal_generator.cpp` | Inline kernel — вынести |
| 8 | `modules/example/kernels/vector_ops.cl` | Образец .cl файла |

---

## 2. ПОРЯДОК ВЫПОЛНЕНИЯ

### Шаг 1: Правило в CLAUDE.md

Добавить в `CLAUDE.md` (раздел Architecture & Code Organization или структура):

```
### Kernels — единый стиль
- Все OpenCL kernels — в отдельные `.cl` файлы в `modules/[module]/kernels/`
- Не inline в .cpp — только загрузка из файла или #include через CMake
- Общий include (например prng.cl) — в `modules/[module]/kernels/` или общая папка
- Референс: MemoryBank/specs/Form_signals.md раздел 5
```

Если уже есть — дополнить. Ссылаться в следующих задачах.

### Шаг 2: Вынос kernels в .cl

1. **FormSignalGenerator**  
   - Вынести kernel из `form_signal_generator.cpp` (KERNEL_SOURCE) в `modules/signal_generators/kernels/form_signal.cl`  
   - Philox+Box-Muller — в `kernels/prng.cl` (общий include) или оставить в form_signal.cl по Form_signals.md  
   - Загрузка: чтение файла при инициализации или CMake

2. **FormScriptGenerator**  
   - Генерируем kernel — писать в `modules/signal_generators/kernels/` (уже так?)  
   - Проверить: все .cl в одной папке kernels, единый стандарт

3. **DelayedFormSignalGenerator**  
   - Вынести FRACTIONAL_DELAY_KERNEL_SOURCE в `modules/signal_generators/kernels/delayed_form_signal.cl`  
   - Загрузка из файла

4. **Результат:** `cmake -B build && cmake --build build` — сборка OK

### Шаг 3: On-disk cache в DrvGPU

> **Условие:** Механизм на тестах работает нейтрально (без привязки к signal_generators).

1. **Анализ:** Проверить, что save/load kernel в FormScriptGenerator не зависит от FormParams и специфики signal_generators.  
2. **Перенос:** Вынести логику on-disk cache (save_kernel, load_kernel, manifest, bin/) в `DrvGPU/services/` — общий сервис для модулей.  
3. **Потребители:** signal_generators, filters (при проектировании) и др.  
4. **API:** `KernelCacheService` или аналог — save(name, binary, metadata), load(name) → binary, list_kernels()

**Если перенос сложен:** оставить в signal_generators, но документировать интерфейс для копирования в DrvGPU позже.

### Шаг 4: Тесты и картинки

1. Запустить `python -m pytest Python_test/test_form_signal.py -v`  
2. Запустить `python -m pytest Python_test/test_delayed_form_signal.py -v`  
3. Запустить `python Python_test/example_form_signal.py`  
4. **Директория с картинками после тестов:**

| Тест / скрипт | Директория с PNG |
|---------------|------------------|
| `test_form_signal.py` | `Results/Plots/FormSignal/` |
| `test_delayed_form_signal.py` | `Results/Plots/DelayedFormSignal/` |
| `example_form_signal.py` | `Results/Plots/FormSignal/` |
| `test_gpuworklib.py` и др. | `Results/Plots/` (корень) |

**Полный путь:** `GPUWorkLib/Results/Plots/...` (от корня репозитория)

---

## 3. ЧЕКЛИСТ ПЕРЕД «ГОТОВО»

- [ ] `cmake -B build && cmake --build build` — успешно
- [ ] `python -m pytest Python_test/test_form_signal.py -v` — проходят
- [ ] `python -m pytest Python_test/test_delayed_form_signal.py -v` — проходят
- [ ] `python Python_test/example_form_signal.py` — выполняется
- [ ] CLAUDE.md: правило про kernels добавлено/обновлено
- [ ] Kernels: form_signal.cl, delayed_form_signal.cl (и prng.cl при необходимости) в `modules/signal_generators/kernels/`
- [ ] On-disk cache: перенесён в DrvGPU или документирован план переноса

---

## 4. ПЕРЕДАТЬ ГЛАВНОМУ AI (Кодо)

После выполнения:

```
✅ Выполнено:
- [список сделанного]
- [пути к созданным/изменённым файлам]
- [результат сборки]

Картинки после тестов:
- Results/Plots/FormSignal/
- Results/Plots/DelayedFormSignal/

Проверь, пожалуйста:
1. Компиляция
2. Тесты
3. Соответствие плану
```

---

## 5. ДИРЕКТОРИЯ С КАРТИНКАМИ (для Alex)

```
GPUWorkLib/Results/Plots/FormSignal/       # ФорSignal, example_form_signal
GPUWorkLib/Results/Plots/DelayedFormSignal/  # DelayedFormSignal (Farrow)
GPUWorkLib/Results/Plots/                  # Общие тесты (test_gpuworklib и др.)
```

Флаг `--no-plot` отключает генерацию. По умолчанию картинки создаются.
