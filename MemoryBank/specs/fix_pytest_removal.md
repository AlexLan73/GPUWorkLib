# План: Удаление всех упоминаний pytest из Python_test/
**Статус:** TODO
**Дата создания:** 2026-03-19
**Приоритет:** Высокий — pytest не используется в проекте!

---

## Диагноз

Проект использует собственную инфраструктуру тестирования:
- `common/runner.py` → `TestRunner` + `SkipTest`
- `common/test_base.py` → `TestBase` (Template Method)
- Запуск: `python Python_test/module/test_xxx.py` напрямую

**`import pytest` НЕ используется нигде** — это хорошо!

Проблема в другом:
1. В комментариях/docstring осталось `pytest Python_test/...` в примерах запуска
2. Несколько файлов описывают себя как "pytest configuration"
3. `test_heterodyne_rocm.py` — архитектурный баг (хардкод пути, не использует GPULoader)

---

## Категория А — Только правка комментариев/docstring

Убрать `pytest` из Usage-секций, заменить на `python file.py`.

### A1. `common/gpu_loader.py` — строки 22
```
# было:  pytest.skip("gpuworklib not found")
# стало: raise SkipTest("gpuworklib not found")
```
*(в примере в docstring)*

### A2. `common/gpu_context.py` — строки 17, 21
```
# было:  @pytest.fixture(scope="session")
#         def gpu_ctx(): ...
# стало: убрать целый пример с pytest.fixture,
#         оставить только: ctx = GPUContextManager.get()
```

### A3. `common/runner.py` — строки 8, 23
```
# было:  SkipTest — исключение для пропуска теста (заменяет pytest.skip)
#         Заменяет pytest.skip(). Бросается внутри setUp() или test_*().
# стало: SkipTest — исключение для пропуска теста
#         Бросается внутри setUp() или test_*().
```

### A4. `common/plotting/__init__.py` — строка 7
```
# было:  Графики никогда не вызываются внутри pytest-тестов напрямую.
# стало: Графики вызываются только из __main__ и plot_*() методов.
```

### A5. `capon/conftest.py` — строка 2
```
# было:  conftest.py — pytest configuration for capon module tests.
# стало: conftest.py — фабричные функции для Python_test/capon/
```

### A6. `strategies/conftest.py` — строка 5
```
# было:  Без pytest. Предоставляет plain factory functions вместо @pytest.fixture.
# стало: Предоставляет фабричные функции. Каждый вызов make_*() создаёт новый объект.
```

### A7. `strategies/test_strategies_step_by_step.py` — строка 32, 35
```
# было:  Если GPU недоступен — Часть 2 пропускается автоматически (pytest.skip).
#         pytest Python_test/strategies/test_strategies_step_by_step.py -v
# стало: Если GPU недоступен — Часть 2 пропускается (SkipTest).
#         python Python_test/strategies/test_strategies_step_by_step.py
```

### A8. `strategies/test_timing_analysis.py` — строки 23, 26, 123
```
# было:  Без этих файлов все тесты будут pytest.skip.
#         pytest Python_test/strategies/test_timing_analysis.py -v
#         # pytest tests
# стало: Без этих файлов все тесты будут SkipTest.
#         python Python_test/strategies/test_timing_analysis.py
#         # Tests
```

### A9. `statistics/test_statistics_rocm.py` — строки 27
```
# было:  pytest Python_test/statistics/test_statistics_rocm.py -v
# стало: (убрать строку — достаточно python ...)
```

### A10. Все файлы с "ЗАПУСК: pytest ..." в docstring

Файлы у которых в заголовке/Usage написано `pytest ...`:
- `fft_func/test_spectrum_maxima_finder_rocm.py` — строка 17
- `capon/test_capon.py` — строка 25
- `filters/test_fir_filter_rocm.py` — Usage секция
- `filters/test_iir_filter_rocm.py` — Usage секция
- `filters/test_moving_average_rocm.py` — Usage секция
- `filters/test_kaufman_rocm.py` — Usage секция
- `filters/test_kalman_rocm.py` — Usage секция
- `lch_farrow/test_lch_farrow_rocm.py` — Usage секция
- `hybrid/test_hybrid_backend.py` — Usage секция
- `zero_copy/test_zero_copy.py` — Usage секция

**Правило**: во всех Usage/Запуск секциях заменить `pytest file.py -v` на `python file.py`.

---

## Категория Б — Архитектурные правки

### Б1. `heterodyne/test_heterodyne_rocm.py` — КРИТИЧНО 🔴

**Проблема 1**: Хардкод пути к `.so` (строка 31):
```python
# УДАЛИТЬ эту строку:
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'debian-radeon9070', 'python'))
```

**Проблема 2**: `HAS_GPU` + `return` вместо `SkipTest`:
```python
# было:
try:
    import gpuworklib
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

def test_dechirp_vs_numpy():
    if not HAS_GPU:
        print("  SKIP: no GPU")
        return

# стало: убрать HAS_GPU, добавить в каждый тест:
def test_dechirp_vs_numpy():
    gw = GPULoader.get()
    if gw is None or not hasattr(gw, 'HeterodyneROCm'):
        raise SkipTest("gpuworklib/HeterodyneROCm недоступен")
    ...
```

**Проблема 3**: `make_ctx_het()` создаёт новый `ROCmGPUContext(0)` в каждом тесте:
```python
# было (создаётся 6 раз за сессию):
def make_ctx_het(...):
    ctx = gpuworklib.ROCmGPUContext(0)

# стало (синглтон, создаётся один раз):
from common.gpu_context import GPUContextManager
ctx = GPUContextManager.get_rocm()
if ctx is None:
    raise SkipTest("ROCm недоступен")
```

**Алгоритм исправления**:
1. Добавить стандартные импорты (GPULoader, GPUContextManager, SkipTest)
2. Удалить `sys.path.insert(...)` на строке 31
3. Удалить `HAS_GPU = True/False` и `try: import gpuworklib`
4. Переписать `make_ctx_het()` через GPUContextManager.get_rocm()
5. В каждой test-функции заменить `if not HAS_GPU: return` → `gw = GPULoader.get(); if not: raise SkipTest`
6. Обновить docstring: убрать `pytest ...`, добавить `python ...`

---

## Категория В — Conftest-файлы

Все `conftest.py` в подпапках — их роль: "фабричные функции для создания тестовых объектов".
Название файла `conftest.py` — это convention из pytest. Рассмотреть переименование.

**Варианты:**
- **Переименовать** в `test_helpers.py` или `fixtures.py` (более нейтрально)
- **Оставить** имя `conftest.py` — оно уже устоялось, не несёт вреда без pytest

**Решение: оставить имя `conftest.py`**, т.к.:
- Уже импортируется везде по этому имени: `from conftest import make_farrow`
- Переименование = менять импорты во всех тестах
- Имя само по себе не добавляет pytest-зависимость

**НО**: Обновить docstring первой строки везде где написано "pytest configuration":

| Файл | Было | Стало |
|------|------|-------|
| `capon/conftest.py` | `pytest configuration for capon` | `фабричные функции для Python_test/capon/` |

Остальные `conftest.py` уже содержат "Без pytest." — оставить как есть.

---

## Категория Г — Дополнительные баги из ревью (не связаны с pytest)

Эти баги обнаружены в ходе ревью. Зафиксировать для отдельного исправления.

### Г1. `gpu_loader.py:131` — логическая ошибка фильтра файлов
```python
# было (некорректная скобочность):
if fp.suffix in (".pyd",) or (fp.suffix == ".so" or ".cpython" in fp.name):

# стало:
if fp.suffix in (".pyd", ".so"):
```

### Г2. `reporters.py:123` — crash при `output_path` без директории
```python
# было:
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# стало:
parent = os.path.dirname(output_path)
if parent:
    os.makedirs(parent, exist_ok=True)
```

### Г3. `result.py:60` — пустой TestResult всегда PASS
```python
# было:
return all(v.passed for v in self.validations)   # all([]) == True

# стало (опционально — зависит от политики):
if not self.validations:
    return False   # нет проверок = не прошёл
return all(v.passed for v in self.validations)
```

### Г4. `strategy_test_base.py:146` — `import math` внутри функции
```python
# стало: перенести на верх файла
import math
```

### Г5. `pipeline_step_validator.py:37-38` — дублирование `import os`
```python
# удалить:
import os as _os
_sys_path_root = _os.path.join(...)
# заменить _os на os (уже импортирован)
```

### Г6. `run_all_rocm_tests.sh` — неполный список тестов
Добавить:
```bash
run_test "Python_test/vector_algebra/test_cholesky_inverter_rocm.py" \
         "CholeskyInverterROCm (POTRF+POTRI, Roundtrip/GpuKernel)"

run_test "Python_test/statistics/test_statistics_float_rocm.py" \
         "StatisticsProcessor float (mean/median float32)"
```

---

## Порядок выполнения

| # | Категория | Файлов | Сложность | Приоритет |
|---|-----------|--------|-----------|-----------|
| 1 | Б1 — `test_heterodyne_rocm.py` полный рефактор | 1 | Средняя | 🔴 Высокий |
| 2 | А — Правка docstring `common/` | 4 | Низкая | 🟡 Средний |
| 3 | А — Правка Usage во всех тестах | ~15 | Низкая | 🟡 Средний |
| 4 | В — `capon/conftest.py` описание | 1 | Низкая | 🟢 Низкий |
| 5 | Г1 — gpu_loader логика | 1 | Низкая | 🔴 Высокий |
| 6 | Г2 — reporters makedirs | 1 | Низкая | 🟡 Средний |
| 7 | Г3-Г5 — мелкие баги | 3 | Низкая | 🟢 Низкий |
| 8 | Г6 — run_all_rocm_tests.sh | 1 | Низкая | 🟡 Средний |

---

## Что НЕ нужно менять

- Имена файлов `conftest.py` — оставить (уже импортируются)
- Структуру standalone `def test_*()` функций — это **намеренный паттерн** прямого запуска
- TestRunner — работает корректно
- GPUContextManager, GPULoader, DataValidator — OK

---

## Проверка после исправления

```bash
# Запустить все тесты без GPU (NumPy-only части):
python Python_test/statistics/test_statistics_rocm.py
python Python_test/heterodyne/test_heterodyne_rocm.py

# С GPU:
sg render -c "bash Python_test/run_all_rocm_tests.sh"

# Убедиться что pytest НЕ упоминается:
grep -r "pytest" Python_test/ --include="*.py" | grep -v ".pyc"
```
