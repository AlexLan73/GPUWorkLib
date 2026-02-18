# FormSignalGenerator — Файл разногласий (план vs реализация)

> **Дата**: 2026-02-11
> **Цель**: Отражение всех отличий между планом/спецификацией и фактической реализацией

---

## 1. Расположение кернелов

| План / Спецификация | Реализация | Статус |
|---------------------|------------|--------|
| Kernel в `modules/signal_generators/kernels/form_signal.cl` (отдельный файл) | Kernel **inline** в `form_signal_generator.cpp` (строка KERNEL_SOURCE) | ⚠️ Отличие |
| `kernels/prng.cl` — общий include для Philox+Box-Muller | Philox+Box-Muller **встроен** в тот же kernel в form_signal_generator.cpp | ⚠️ Отличие |

**Причина**: Упрощение сборки — один .cpp, не нужен runtime path к .cl. FormScriptGenerator генерирует kernel из DSL, сохраняет в .cl при save_kernel().

**Рекомендация**: Оставить как есть (inline). При необходимости вынести в form_signal.cl позже.

---

## 2. Python тесты — создание картинок

| План | Реализация | Статус |
|------|------------|--------|
| Графики в тестах Python | `make_plots()` создаёт 6 PNG, но **только при флаге `--plot`** | ⚠️ Исправлено |

**Исправление**: Картинки теперь создаются **по умолчанию** при запуске `test_form_signal.py`. Флаг `--no-plot` отключает генерацию.

---

## 3. Output: GPU + metadata

| План / Спецификация | Реализация | Статус |
|---------------------|------------|--------|
| GPU: cl_mem + metadata (адрес, size, num_antennas, points) | `InputData<cl_mem>` — data, antenna_count, n_point, gpu_memory_bytes | ✅ Реализовано |
| Не добавлять kernel_name, backend_type | — | ✅ Соответствует |

**Реализация**: Используется существующая структура `InputData<cl_mem>` из DrvGPU (как в fft_maxima). Метод `GenerateInputData()` возвращает полную метаданную. `Generate()` удалён — везде `GenerateInputData()`.

---

## 4. Output: "cpu" | "gpu"

| План | Реализация | Статус |
|------|------------|--------|
| Python: output="cpu" \| "gpu" | `generate(output="cpu"|"gpu")` — CPU: numpy, GPU: GPUBuffer с .read() | ✅ Реализовано |

**Реализация**: `output="cpu"` (по умолчанию) — readback → numpy. `output="gpu"` — возвращает `GPUBuffer` с методами `read()`, `shape`, `release()`, `antenna_count`, `n_point`.

---

## 5. FormSignalGenerator vs ISignalGenerator

| План | Реализация | Статус |
|------|------------|--------|
| Реализует ISignalGenerator или IFormSignalGenerator | **Standalone** — не наследует ISignalGenerator | ✅ Осознанное решение |

**Причина**: API отличается (antennas, points в FormParams; vector<vector<>> на CPU). SignalService::GenerateFormGpu/Cpu — отдельные методы.

---

## 6. FormScriptGenerator — отдельный класс

| План | Реализация | Статус |
|------|------------|--------|
| Обёртка над ScriptGenerator | **Самостоятельный** генератор: FormParams → DSL → OpenCL kernel | ✅ Реализовано |

FormScriptGenerator не использует ScriptGenerator внутри — генерирует kernel из FormParams сам. Ближе к плану по функциональности.

---

## 7. On-disk кэш — путь

| План / Спецификация | Реализация | Статус |
|---------------------|------------|--------|
| `modules/[module]/kernels/bin/` | `modules/signal_generators/kernels/bin/` | ✅ Соответствует |
| manifest.json локально | manifest.json в kernels/ | ✅ Соответствует |
| Версионирование _00 при коллизии | Реализовано | ✅ |
| Префикс _opencl | Реализовано | ✅ |

---

## 8. Параметры FormParams

| План | Реализация | Статус |
|------|------------|--------|
| delay (tau) | tau_base, tau_step, tau_min, tau_max, tau_seed | ✅ Расширено (TauMode: FIXED/LINEAR/RANDOM) |
| freq_min, freq_max | Реализованы в FormParams | ✅ |

---

## 9. Тесты C++

| План | Реализация | Статус |
|------|------------|--------|
| 6 тестов | 6 тестов: NoNoise, Window, MultiChannel, Noise, Parser, Chirp | ✅ |
| Сравнение с CPU reference | getX_numpy в Python; в C++ — косвенное через GPU vs ожидаемое | ✅ |

**Примечание**: C++ и Python тесты оба сравнивают GPU с CPU-реализацией формулы getX. В C++ используется `GetXReference()` (C++), в Python — `getX_numpy()` (NumPy). Логика проверки одинаковая.

---

## 10. Пример и документация

| План | Реализация | Статус |
|------|------------|--------|
| `Python_test/example_form_signal.py` или `examples/form_signal_demo.py` | `Python_test/example_form_signal.py` | ✅ |
| Графики как на презентацию | 5 демо + 5 графиков в example, 6 графиков в test_form_signal | ✅ |
| Сохранение в Results/Plots/FormSignal/ | Реализовано | ✅ |
| Doc/Python/signal_generators_api.md | Создан | ✅ |

---

## 11. ROCm

| План | Реализация | Статус |
|------|------------|--------|
| FormSignalGeneratorROCm — stub | Реализован (throw) | ✅ |
| form_signal.hip | Заглушка в kernels/rocm/ | ✅ |
| CreateFormROCm | Отдельный метод в Factory | ✅ |

---

## 12. Итоговая сводка

| Категория | Соответствует | Отличие | Отложено |
|-----------|----------------|---------|----------|
| Кернелы | — | Inline vs .cl файл | — |
| Python картинки | — | Было --plot, исправлено | — |
| Output metadata | InputData | — | — |
| Output "gpu" | GPUBuffer | — | — |
| Остальное | Большинство | 1–2 пункта | 2 |

---

*Обновлять при выявлении новых расхождений.*
## 1. Расположение кернелов
 У нас должно быть все в одном стиле ныжно вынести в папку с кернелам она есть в каждом модуле и проще править - это вроде прописано в @CLAUDE.md -если нет проиши и ссылайся в следующий раз

 ## 6. FormScriptGenerator — отдельный класс
 опять про kernel  ScriptGenerator внутри — генерирует kernel ( - 
 пишим в одном стандарте все кернелы в общей папке для них

 ## 7. On-disk кэш — путь
если этот механизм на тестах работает новтально его нужно перенести в DryGPU servis gjnjve что мы будем им пользоваться и в других модулях к примеру при проектировании фильтров

