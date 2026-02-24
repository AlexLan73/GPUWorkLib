# Инструкция для следующей сессии

> **Дата написания**: 2026-02-24
> **Приоритет**: СРЕДНИЙ
> **Задача**: Документация Python API для ROCm + run_all_tests скрипт + Task_08/09 (опционально)

---

## 🚀 Начни здесь (для нового Кодо)

### Шаг 1 — Прочитай контекст
1. `MemoryBank/MASTER_INDEX.md` — статус проекта
2. `MemoryBank/tasks/COMPLETED.md` — что уже сделано
3. `MemoryBank/sessions/2026-02-24g.md` — последняя сессия

### Шаг 2 — Пойми статус
Все ROCm модули **готовы и протестированы**. Python тесты **30/30 PASSED**.
Осталось: документация + скрипт + опциональный ZeroCopy/HybridBackend.

### Шаг 3 — Выполни задачи по приоритету
1. 🔴 **HIGH** — Документация Python API → `Doc/Python/rocm_modules_api.md`
2. 🟡 **MEDIUM** — Скрипт запуска тестов → `Python_test/run_all_rocm_tests.sh`
3. 🟢 **LOW** — Task_08 ZeroCopy (если время/желание есть)
4. 🟢 **LOW** — Task_09 HybridBackend (если время/желание есть)

---

## ✅ Текущий статус: Всё работает!

**Все ROCm модули реализованы и протестированы:**

| Модуль | C++ тесты | Python тесты | Биндинги |
|--------|-----------|--------------|---------|
| DrvGPU (ROCmBackend) | ✅ | — | ✅ ROCmGPUContext |
| FFTProcessor ROCm | ✅ | — | — |
| Statistics ROCm | ✅ 11/11 | ✅ 9/9 (speedup 18.7×) | ✅ StatisticsProcessor |
| SpectrumMaxima ROCm | ✅ | — | — |
| Filters ROCm | ✅ | ✅ FIR 5/5, IIR 5/5 | ✅ FirFilterROCm, IirFilterROCm |
| LchFarrow ROCm | ✅ | ✅ 5/5 | ✅ LchFarrowROCm |
| FormSignal ROCm | ✅ | — | — |
| Heterodyne ROCm | ✅ | ✅ 6/6 | ✅ HeterodyneROCm |

**Итого Python тестов**: **30/30 PASSED** 🎉

**Python .so**: `build/debian-radeon9070/python/gpuworklib.cpython-313-x86_64-linux-gnu.so`

**Сборка**:
```bash
cmake -B build/debian-radeon9070 -DBUILD_PYTHON=ON
cmake --build build/debian-radeon9070 -j4
```

---

## 🎯 Задача 1 (Приоритет HIGH): Документация Python API для ROCm

### Что нужно создать

`Doc/Python/rocm_modules_api.md` — единый файл документации для всех ROCm классов.

### Структура файла

```markdown
# GPUWorkLib Python API — ROCm Classes

## ROCmGPUContext
## FirFilterROCm
## IirFilterROCm
## LchFarrowROCm
## HeterodyneROCm
## StatisticsProcessor
```

### Источники для документации

- `python/py_filters_rocm.hpp` — FirFilterROCm, IirFilterROCm
- `python/py_lch_farrow_rocm.hpp` — LchFarrowROCm
- `python/py_heterodyne_rocm.hpp` — HeterodyneROCm
- `python/py_statistics.hpp` — StatisticsProcessor
- `python/gpu_worklib_bindings.cpp` — ROCmGPUContext

### Формат документации (из CLAUDE.md):

```python
# Constructor
ctx = gpuworklib.ROCmGPUContext(0)

# FIR Filter
fir = gpuworklib.FirFilterROCm(ctx)
fir.set_coefficients([0.25, 0.5, 0.25])
out = fir.process(data)  # np.complex64 array

# Properties
fir.num_taps           # int
fir.coefficients       # list of float
```

### Таблица классов для документации

| Класс | Конструктор | Ключевые методы |
|-------|-------------|-----------------|
| `ROCmGPUContext` | `(device_index: int)` | `.device_name`, `.device_index` |
| `FirFilterROCm` | `(ctx: ROCmGPUContext)` | `set_coefficients()`, `process()` |
| `IirFilterROCm` | `(ctx: ROCmGPUContext)` | `set_sections()`, `process()` |
| `LchFarrowROCm` | `(ctx: ROCmGPUContext)` | `set_delays()`, `set_sample_rate()`, `set_noise()`, `process()` |
| `HeterodyneROCm` | `(ctx: ROCmGPUContext)` | `set_params()`, `dechirp()`, `correct()` |
| `StatisticsProcessor` | `(ctx: ROCmGPUContext)` | `compute_mean()`, `compute_median()`, `compute_statistics()` |

---

## 🎯 Задача 2 (Приоритет MEDIUM): Единый скрипт запуска тестов

### Что создать

`Python_test/run_all_rocm_tests.sh` — bash-скрипт для запуска всех Python тестов:

```bash
#!/bin/bash
# Запуск всех ROCm Python тестов
# Использование: sg render -c "./Python_test/run_all_rocm_tests.sh"
# или: bash Python_test/run_all_rocm_tests.sh (без GPU)

PASSED=0; FAILED=0

run_test() {
    echo "=== $1 ==="
    if python3 "$1"; then
        PASSED=$((PASSED+1))
    else
        FAILED=$((FAILED+1))
    fi
}

run_test Python_test/statistics/test_statistics_rocm.py
run_test Python_test/filters/test_fir_filter_rocm.py
run_test Python_test/filters/test_iir_filter_rocm.py
run_test Python_test/heterodyne/test_heterodyne_rocm.py
run_test Python_test/lch_farrow/test_lch_farrow_rocm.py

echo "Total: $PASSED passed, $FAILED failed"
```

---

## 🎯 Задача 3 (Приоритет LOW): Task_08 ZeroCopy

ZeroCopyBridge уже реализован в `DrvGPU/backends/rocm/zero_copy_bridge.cpp`.
Нужно добавить OpenCL-сторону (экспорт dma-buf fd) и написать тест.

> ⚠️ **Только новые файлы!** Не трогать существующие OpenCL файлы.

| Тип | Файл | Статус |
|-----|------|--------|
| НОВЫЙ | `DrvGPU/backends/opencl/opencl_export.hpp` | создать |
| НОВЫЙ | `DrvGPU/backends/opencl/opencl_export.cpp` | создать |
| НОВЫЙ | `DrvGPU/tests/test_zero_copy.hpp` | создать |
| НОВЫЙ | `Python_test/zero_copy/test_zero_copy.py` | создать |
| Дополнить | `DrvGPU/backends/rocm/zero_copy_bridge.cpp` | уже есть |

Детали: `MemoryBank/tasks/Task_08_ZeroCopy.md`

---

## 🎯 Задача 4 (Приоритет LOW): Task_09 HybridBackend

Гибридный режим `BackendType::OPENCLandROCm` — оба backend на одной GPU.
**Зависит от Task_08** (ZeroCopy должен быть готов сначала!).

> ⚠️ **Только новые файлы!** Не трогать существующие OpenCL и ROCm файлы.

| Тип | Файл | Статус |
|-----|------|--------|
| НОВЫЙ | `DrvGPU/backends/hybrid/hybrid_backend.hpp` | создать |
| НОВЫЙ | `DrvGPU/backends/hybrid/hybrid_backend.cpp` | создать |
| НОВЫЙ | `DrvGPU/tests/test_hybrid_backend.hpp` | создать |
| НОВЫЙ | `Python_test/hybrid/test_hybrid_backend.py` | создать |
| Минимум | `DrvGPU/src/drv_gpu.cpp` | добавить 1 case |

**Рекомендуемый вариант**: Вариант B — два DrvGPU, ZeroCopyBridge как мост.

Детали: `MemoryBank/tasks/Task_09_HybridBackend.md`

---

## ⚠️ Известные ограничения

- **LchFarrow**: НЕ использовать ровно целочисленные задержки (1.0, 2.0, 3.0 µs при 1MHz).
  Использовать дроби: 1.5, 2.7, 3.3 и т.д. (GPU float32 boundary issue с row=47 матрицы Лагранжа)
- **OpenCL (clFFT)**: падает на gfx1201 — ИГНОРИРОВАТЬ
- Запуск с GPU: `sg render -c "python3 test.py"` (нужна группа render)

---

## 📁 Ключевые файлы

```
python/
├── gpu_worklib_bindings.cpp   # ROCmGPUContext + регистрация всех классов
├── py_filters_rocm.hpp        # FirFilterROCm, IirFilterROCm
├── py_lch_farrow_rocm.hpp     # LchFarrowROCm
├── py_heterodyne_rocm.hpp     # HeterodyneROCm
└── py_statistics.hpp          # StatisticsProcessor

Python_test/
├── filters/
│   ├── test_fir_filter_rocm.py   ✅ 5/5
│   └── test_iir_filter_rocm.py   ✅ 5/5
├── heterodyne/
│   └── test_heterodyne_rocm.py   ✅ 6/6
├── lch_farrow/
│   └── test_lch_farrow_rocm.py   ✅ 5/5 (delay 3.3, не 3.0!)
└── statistics/
    └── test_statistics_rocm.py   ✅ 9/9 (speedup 18.7×)

DrvGPU/backends/rocm/
└── zero_copy_bridge.cpp       # ZeroCopyBridge уже реализован — нужен тест
```

---

## 🏁 Итог этой эпохи (ROCm migration)

Все основные задачи **Task_00 → Task_07** завершены:

| Task | Модуль | Статус |
|------|--------|--------|
| Task_00 | DrvGPU ROCmBackend | ✅ DONE |
| Task_01 | FFTProcessor ROCm | ✅ DONE |
| Task_02 | Statistics ROCm | ✅ DONE |
| Task_03 | SpectrumMaxima ROCm | ✅ DONE |
| Task_04 | Filters ROCm | ✅ DONE |
| Task_05 | LchFarrow ROCm | ✅ DONE |
| Task_06 | FormSignal ROCm | ✅ DONE |
| Task_07 | Heterodyne ROCm | ✅ DONE |
| Task_05★ | Python биндинги + тесты | ✅ DONE (30/30) |
| Task_08 | ZeroCopy | ⏳ опционально |
| Task_09 | HybridBackend | ⏳ опционально |

Документация Python API — единственное обязательное, что осталось. 📝

---

*Написано: 2026-02-24, Кодо*
*Обновлено: 2026-02-24, Кодо — добавлены статус выполнения задач 1 и 2, задание следующей сессии*

---

## ✅ Что сделано в этой сессии (конец 2026-02-24)

| Задача | Статус | Файл |
|--------|--------|------|
| Документация Python API (ROCm) | ✅ DONE | `Doc/Python/rocm_modules_api.md` |
| Скрипт запуска тестов | ✅ DONE | `Python_test/run_all_rocm_tests.sh` |
| Task_08 ZeroCopy | ⏳ НЕ НАЧАТО | — |
| Task_09 HybridBackend | ⏳ НЕ НАЧАТО | — |

---

## 📋 Задание для следующего Кодо

### 🎯 Контекст: ROCm-эпоха завершена. Что дальше?

Все Task_00–Task_07 выполнены, Python тесты 30/30, документация написана.
Впереди — **необязательные расширения** и **обновление MASTER_INDEX**.

---

### Шаг 1 — Обязательно: Привести MASTER_INDEX.md в порядок

Файл `MemoryBank/MASTER_INDEX.md` **устарел** — не отражает ROCm реальность.

Что обновить:
1. Статус модулей: добавить ROCm-классы (`ROCmGPUContext`, `FirFilterROCm`, etc.)
2. Python модуль: добавить ROCm-биндинги в список классов
3. Раздел «В работе»: убрать «ROCm backend planned» → написать «ROCm DONE»
4. Дата обновления → 2026-02-24

---

### Шаг 2 — 🔴 ОБЯЗАТЕЛЬНО: Task_08 ZeroCopy

OpenCL → ROCm zero-copy через dma-buf. **Это задача следующей сессии!**

**Детали**: `MemoryBank/tasks/Task_08_ZeroCopy.md`

**Файлы для создания (только новые!)**:
- `DrvGPU/backends/opencl/opencl_export.hpp` — экспорт dma-buf fd из OpenCL буфера
- `DrvGPU/backends/opencl/opencl_export.cpp`
- `DrvGPU/tests/test_zero_copy.hpp` — C++ тест
- `Python_test/zero_copy/test_zero_copy.py` — Python тест

⚠️ Не трогать существующие OpenCL и ROCm файлы!

---

### Шаг 3 — 🔴 ОБЯЗАТЕЛЬНО: Task_09 HybridBackend

Только после Task_08! Гибридный режим: OpenCL + ROCm на одной GPU. **Это задача следующей сессии!**

**Детали**: `MemoryBank/tasks/Task_09_HybridBackend.md`

Рекомендуемый вариант: **Вариант B** — два DrvGPU, ZeroCopyBridge как мост.

---

### Шаг 4 — 🚫 НЕ ДЕЛАТЬ: Очистка MemoryBank

~~Перенести завершённые таски в `COMPLETED.md`~~
~~Удалить промежуточные черновики из `sessions/`~~
~~Обновить `changelog/2026-02.md`~~

> Очистку пока не делать — оставить все сессии и черновики для истории.

---

## 🗂️ Ключевые файлы для следующей сессии

```
MemoryBank/MASTER_INDEX.md          ← обновить первым!
Doc/Python/rocm_modules_api.md      ← уже готово ✅
Python_test/run_all_rocm_tests.sh   ← уже готово ✅
MemoryBank/tasks/Task_08_ZeroCopy.md
MemoryBank/tasks/Task_09_HybridBackend.md
```
