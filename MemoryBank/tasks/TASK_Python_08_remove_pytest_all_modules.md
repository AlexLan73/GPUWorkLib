# TASK Python-08: Удалить pytest из всех остальных модулей

**Статус**: 🔲 TODO
**Приоритет**: 🟠 СРЕДНИЙ — после TASK_Python_07
**Файлы**: все `test_*.py` и `conftest.py` вне `strategies/`
**Стиль**: ООП, SOLID, GRASP, GoF — обязательно!
**Порядок обработки**: fft_func → statistics → signal_generators → heterodyne → filters → vector_algebra → capon → range_angle

---

## 🎯 Цель

Убрать pytest из оставшихся 27 файлов (не strategies/, не common/).
К этому моменту уже есть: `TestRunner`, `SkipTest`, `DataValidator` из TASK 01–02.

---

## 📖 Что прочитать перед реализацией

- `TASK_Python_01` — `TestRunner`, `SkipTest`
- `TASK_Python_02` — `DataValidator`
- `TASK_Python_07` — примеры замен (там расписано подробно)
- Каждый файл перед изменением — прочитать!

---

## 📁 Полный список файлов

### Группа 1: fft_func

| Файл | Содержимое |
|------|-----------|
| `Python_test/fft_func/conftest.py` | pytest.fixture для fft контекста |
| `Python_test/fft_func/test_process_magnitude_rocm.py` | FFT magnitude тесты |
| `Python_test/fft_func/test_spectrum_find_all_maxima_rocm.py` | Поиск максимумов |
| `Python_test/fft_func/test_spectrum_maxima_finder_rocm.py` | SpectrumMaximaFinder |

### Группа 2: statistics

| Файл | Содержимое |
|------|-----------|
| `Python_test/statistics/conftest.py` | pytest.fixture |
| `Python_test/statistics/test_statistics_float_rocm.py` | Статистика GPU |

### Группа 3: signal_generators

| Файл | Содержимое |
|------|-----------|
| `Python_test/signal_generators/conftest.py` | pytest.fixture |
| `Python_test/signal_generators/test_form_signal_rocm.py` | Генераторы сигналов |

### Группа 4: heterodyne

| Файл | Содержимое |
|------|-----------|
| `Python_test/heterodyne/conftest.py` | pytest.fixture |
| `Python_test/heterodyne/test_heterodyne.py` | Гетеродин тесты |

### Группа 5: filters

| Файл | Содержимое |
|------|-----------|
| `Python_test/filters/conftest.py` | pytest.fixture |
| `Python_test/filters/test_ai_filter_pipeline.py` | AI filter |
| `Python_test/filters/ai_pipeline/test_ai_pipeline.py` | AI pipeline |

### Группа 6: vector_algebra

| Файл | Содержимое |
|------|-----------|
| `Python_test/vector_algebra/conftest.py` | pytest.fixture |
| `Python_test/vector_algebra/test_cholesky_inverter_rocm.py` | Холецкий |
| `Python_test/vector_algebra/test_matrix_csv_comparison.py` | CSV сравнение |

### Группа 7: capon

| Файл | Содержимое |
|------|-----------|
| `Python_test/capon/test_capon.py` | Capon beamforming |

### Группа 8: range_angle

| Файл | Содержимое |
|------|-----------|
| `Python_test/range_angle/test_range_angle.py` | Range-angle тесты |

### Группа 9: fm_correlator

| Файл | Содержимое |
|------|-----------|
| `Python_test/fm_correlator/test_fm_correlator.py` | FM correlator |
| `Python_test/fm_correlator/test_fm_correlator_rocm.py` | FM correlator ROCm |

### Группа 10: integration

| Файл | Содержимое |
|------|-----------|
| `Python_test/integration/conftest.py` | pytest.fixture |
| `Python_test/integration/test_fft_integration.py` | FFT интеграционный |
| `Python_test/integration/test_signal_gen_integration.py` | Generator интеграционный |

### Группа 11: lch_farrow

| Файл | Содержимое |
|------|-----------|
| `Python_test/lch_farrow/conftest.py` | pytest.fixture |

---

## 🔧 Алгоритм обработки каждого модуля

### Для каждой группы:

**Шаг 1**: Прочитать `conftest.py` модуля — понять что за fixtures

**Шаг 2**: Прочитать `test_*.py` — понять что тестируется

**Шаг 3**: Создать класс `TestModuleName`:
```python
class TestFftMagnitude:
    def setUp(self):
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("Нет ROCm GPU")
        gw = GPULoader.get()
        if gw is None:
            raise SkipTest("gpuworklib не найден")
        self._ctx = ctx
        self._gw  = gw
        # ... инициализация специфичного для модуля

    def test_something(self) -> TestResult:
        tr = TestResult(test_name="test_something")
        # ... логика теста
        v = DataValidator(tolerance=0.01, metric="max_rel")
        tr.add(v.validate(gpu_result, numpy_ref, name="check_name"))
        return tr
```

**Шаг 4**: Убрать `import pytest`

**Шаг 5**: Добавить точку входа:
```python
if __name__ == "__main__":
    from common.runner import TestRunner
    runner = TestRunner()
    results = runner.run(TestModuleName())
    runner.print_summary(results)
```

---

## ⚠️ Особые случаи

### `conftest.py` в каждом модуле

Если в `conftest.py` только инициализация контекста — **удалить файл полностью**.
Инициализация уже есть в `GPUContextManager.get()` / `get_rocm()`.

Если в `conftest.py` есть специфичная логика (например загрузка CSV) — перенести в `setUp()` тест-класса.

### `test_matrix_csv_comparison.py`

Этот тест загружает CSV файлы. Путь к файлам — проверить, не хардкодить.
В `setUp()`:
```python
self._csv_dir = os.path.join(PROJECT_ROOT, "Results", "CSV")
if not os.path.isdir(self._csv_dir):
    raise SkipTest(f"CSV директория не найдена: {self._csv_dir}")
```

### `test_cholesky_inverter_rocm.py`

Использует `vector_algebra.CholeskyInverterROCm` — сложный модуль.
Проверить что все зависимости (rocSOLVER) доступны через `SkipTest`.

### Тесты без GPU (NumPy only)

Некоторые тесты (`test_scenario_builder.py`) не используют GPU совсем.
Для них не нужен `SkipTest` — просто убрать pytest.

---

## ✅ Критерии готовности

1. Команда ниже возвращает пустой результат:
   ```bash
   grep -r "import pytest\|pytest.skip\|pytest.fixture\|pytest.mark" Python_test/ \
     --include="*.py" | grep -v "conftest.py.old\|backup"
   ```
2. Каждый `test_*.py` запускается напрямую без ошибок импорта
3. `TestRunner.run(TestModuleName())` работает для каждого модуля
4. SKIP (не FAIL) при отсутствии GPU/ROCm/файлов
5. `assert` заменены на `DataValidator.validate()` + `TestResult.add()`

---

## 🔍 Проверка после выполнения

```bash
# Проверить что pytest не осталось:
grep -r "import pytest" Python_test/ --include="*.py"

# Проверить что все тесты запускаются:
python Python_test/fft_func/test_process_magnitude_rocm.py
python Python_test/statistics/test_statistics_float_rocm.py
python Python_test/signal_generators/test_form_signal_rocm.py
# ... и т.д.
```

---

## ❌ Что НЕ делать

- НЕ менять алгоритмическую логику тестов — только обёртку
- НЕ удалять `conftest.py` если там есть полезная логика инициализации
- НЕ делать пустые тест-классы — если тест не нужен, удалить полностью
- НЕ использовать `unittest.SkipTest` — только наш `from common.runner import SkipTest`
