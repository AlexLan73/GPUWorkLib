# Задача: Вынос kernels в .cl + On-disk cache в DrvGPU

> **Источник:** Doc/Modules/signal_generators/Full.md — FormScriptGenerator, on-disk кэш  
> **План:** [PLAN_KernelCacheService_DrvGPU.md](PLAN_KernelCacheService_DrvGPU.md)  
> **Проверка:** Кодо (старшая) — компиляция, тесты, соответствие спецификации

---

## 1. СВЯЗЬ С ДЕТАЛЬНЫМИ ТАСКАМИ

On-disk cache и рефакторинг разбиты на **подробные таски** (выполняет другая AI, проверяет Кодо):

| Таска | Описание | Зависимость |
|-------|----------|-------------|
| [TASK_KernelCacheService_001_StorageBackend.md](TASK_KernelCacheService_001_StorageBackend.md) | IStorageBackend + FileStorageBackend | — |
| [TASK_KernelCacheService_002_KernelCacheService.md](TASK_KernelCacheService_002_KernelCacheService.md) | KernelCacheService в DrvGPU | 001 |
| [TASK_KernelCacheService_003_FormScriptRefactor.md](TASK_KernelCacheService_003_FormScriptRefactor.md) | FormScriptGenerator → KernelCacheService | 002 |
| [TASK_KernelCacheService_004_Filters.md](TASK_KernelCacheService_004_Filters.md) | FirFilter, IirFilter → KernelCacheService | 002 |
| [TASK_KernelCacheService_005_FilterConfigService.md](TASK_KernelCacheService_005_FilterConfigService.md) | FilterConfigService (SaveFilter/LoadFilter) | 001 |

**Порядок выполнения:** 001 → 002 → 003, 004 (параллельно), 005.

---

## 2. ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ

| № | Файл | Зачем |
|---|------|-------|
| 1 | `CLAUDE.md` | Правила проекта, структура |
| 2 | `Doc/Modules/signal_generators/Full.md` | FormScriptGenerator, kernels, on-disk кэш |
| 3 | `modules/signal_generators/kernels/README.md` | Текущая структура kernels |
| 4 | `modules/signal_generators/src/form_signal_generator.cpp` | Inline kernel — вынести |
| 5 | `modules/signal_generators/src/form_script_generator.cpp` | On-disk cache — логика |
| 6 | `modules/signal_generators/src/delayed_form_signal_generator.cpp` | Inline kernel — вынести |
| 7 | `PLAN_KernelCacheService_DrvGPU.md` | Полный план, архитектура |

---

## 3. ПОРЯДОК ВЫПОЛНЕНИЯ

### Шаг 1: Правило в CLAUDE.md

Добавить в `CLAUDE.md` (раздел Architecture & Code Organization):

```
### Kernels — единый стиль
- Все OpenCL kernels — в отдельные `.cl` файлы в `modules/[module]/kernels/`
- Не inline в .cpp — только загрузка из файла или #include через CMake
- Общий include (например prng.cl) — в `modules/[module]/kernels/` или общая папка
- Референс: Doc/Modules/signal_generators/Full.md раздел 5
```

Если уже есть — дополнить.

### Шаг 2: Вынос kernels в .cl

1. **FormSignalGenerator**  
   - Вынести kernel из `form_signal_generator.cpp` (KERNEL_SOURCE) в `modules/signal_generators/kernels/form_signal.cl`  
   - Philox+Box-Muller — в `kernels/prng.cl` или оставить в form_signal.cl  
   - Загрузка: чтение файла при инициализации

2. **FormScriptGenerator**  
   - Генерируемый kernel — писать в `modules/signal_generators/kernels/`  
   - Проверить: все .cl в одной папке kernels

3. **DelayedFormSignalGenerator**  
   - Вынести FRACTIONAL_DELAY_KERNEL_SOURCE в `modules/signal_generators/kernels/delayed_form_signal.cl`  
   - Загрузка из файла

4. **Результат:** `cmake -B build && cmake --build build` — OK

### Шаг 3: On-disk cache в DrvGPU

**Выполнять по детальным таскам:**

- **TASK-001** — IStorageBackend, FileStorageBackend
- **TASK-002** — KernelCacheService
- **TASK-003** — Рефакторинг FormScriptGenerator (замена SaveKernel/LoadKernel на KernelCacheService)

Шаг 3 = TASK-001 + TASK-002 + TASK-003. См. файлы тасок для подробностей.

### Шаг 4: Тесты и картинки

1. `python -m pytest Python_test/test_form_signal.py -v`  
2. `python -m pytest Python_test/test_delayed_form_signal.py -v`  
3. `python Python_test/example_form_signal.py`  
4. **Директория с картинками:**

| Тест / скрипт | Директория с PNG |
|---------------|------------------|
| `test_form_signal.py` | `Results/Plots/FormSignal/` |
| `test_delayed_form_signal.py` | `Results/Plots/DelayedFormSignal/` |
| `example_form_signal.py` | `Results/Plots/FormSignal/` |
| `test_gpuworklib.py` и др. | `Results/Plots/` |

**Полный путь:** `GPUWorkLib/Results/Plots/...`

---

## 4. ЧЕКЛИСТ ПЕРЕД «ГОТОВО»

- [ ] `cmake -B build && cmake --build build` — успешно
- [ ] `python -m pytest Python_test/test_form_signal.py -v` — проходят
- [ ] `python -m pytest Python_test/test_delayed_form_signal.py -v` — проходят
- [ ] `python Python_test/example_form_signal.py` — выполняется
- [ ] CLAUDE.md: правило про kernels добавлено/обновлено
- [ ] Kernels: form_signal.cl, delayed_form_signal.cl (и prng.cl при необходимости)
- [ ] TASK-001, 002, 003 выполнены и приняты Кодо

---

## 5. ОТЧЁТ ВЫПОЛНИТЕЛЮ

После выполнения:

```
✅ Выполнено:
- [список сделанного]
- [пути к созданным/изменённым файлам]
- [результат сборки]

Картинки: Results/Plots/FormSignal/, Results/Plots/DelayedFormSignal/

Проверь (Кодо): компиляция, тесты, соответствие плану.
```

---

## 6. ДИРЕКТОРИЯ С КАРТИНКАМИ (для Alex)

```
GPUWorkLib/Results/Plots/FormSignal/
GPUWorkLib/Results/Plots/DelayedFormSignal/
GPUWorkLib/Results/Plots/
```

Флаг `--no-plot` отключает генерацию.
