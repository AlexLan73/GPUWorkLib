# TASK Python-01: common/runner.py — TestRunner + SkipTest

**Статус**: 🔲 TODO
**Приоритет**: 🔴 КРИТИЧЕСКИЙ — первый, все остальные зависят от него
**Файл**: `Python_test/common/runner.py` (создать новый)
**Стиль**: ООП, SOLID, GRASP, GoF — обязательно!

---

## 🎯 Цель

Создать минимальную инфраструктуру для запуска тестов **без pytest**.

Два класса:
1. `SkipTest` — исключение для пропуска теста (вместо `pytest.skip()`)
2. `TestRunner` — запускает методы `test_*` в классе, собирает результаты

---

## 📖 Что прочитать перед реализацией

- `Python_test/common/result.py` — классы `TestResult`, `ValidationResult`
- `Python_test/common/test_base.py` — класс `TestBase` (Template Method)
- `Python_test/common/gpu_context.py` — `GPUContextManager` (Singleton)

---

## 📁 Место

```
Python_test/common/runner.py   ← создать
Python_test/common/__init__.py ← добавить экспорт TestRunner и SkipTest
```

---

## 🏗️ Реализация

### Класс `SkipTest`

```python
class SkipTest(Exception):
    """Пропуск теста — GPU недоступен или тест не применим.

    Заменяет pytest.skip(). Бросается внутри setUp() или test_*().
    TestRunner перехватывает и помечает тест как SKIP (не FAIL).

    Usage:
        def setUp(self):
            if not GPUContextManager.is_rocm_available():
                raise SkipTest("ROCm GPU не доступен")
    """
    pass
```

### Класс `TestRunner`

```python
class TestRunner:
    """Запускает методы test_* в объекте, собирает TestResult.

    Coordinator (GRASP): координирует обнаружение и запуск тестов.
    Заменяет pytest discovery — explicit лучше implicit.

    Обнаружение тестов:
        Все методы объекта чьё имя начинается с "test_"
        запускаются в алфавитном порядке.

    Обработка исключений:
        SkipTest  → тест помечается как SKIP в summary
        Exception → тест помечается как FAIL, ошибка сохраняется в TestResult

    Usage:
        runner = TestRunner()
        results = runner.run(MyTestClass())
        runner.print_summary(results)
    """

    def run(self, obj) -> list:
        """Найти и запустить все методы test_* в объекте.

        Args:
            obj: экземпляр класса с методами test_*

        Returns:
            List[TestResult] — результаты всех тестов
        """
        ...

    def run_all(self, objects: list) -> list:
        """Запустить тесты в нескольких объектах.

        Args:
            objects: список экземпляров тест-классов

        Returns:
            List[TestResult] — объединённый список результатов
        """
        ...

    def print_summary(self, results: list) -> None:
        """Вывести сводку: сколько PASS / FAIL / SKIP.

        Формат вывода:
            ========================================
            TEST SUMMARY
            ========================================
            [PASS] TestGemm.test_v1_clean    (3/3 checks)
            [FAIL] TestGemm.test_v3_phase    (1/3 checks) ERROR: ...
            [SKIP] TestGemm.test_v5_file     (ROCm GPU не доступен)
            ----------------------------------------
            Total: 2 passed, 1 failed, 1 skipped
            ========================================
        """
        ...
```

---

## 📋 Детали реализации

### `TestRunner.run(obj)`

1. Получить список методов через `dir(obj)`
2. Отфильтровать: только `callable` + имя начинается с `"test_"`
3. Отсортировать по имени (`sorted(...)`)
4. Для каждого метода:
   ```python
   result = TestResult(test_name=f"{obj.__class__.__name__}.{method_name}")
   try:
       method_result = method()          # вызываем test_*()
       if isinstance(method_result, TestResult):
           result = method_result        # если вернул TestResult — используем его
           result.test_name = ...        # гарантируем имя
   except SkipTest as e:
       result.metadata["skipped"] = True
       result.metadata["skip_reason"] = str(e)
   except Exception as e:
       result.error = e
   results.append(result)
   ```
5. Вернуть `list[TestResult]`

### `TestRunner.print_summary(results)`

Формат каждой строки:
```
[PASS] ClassName.test_name   (N/N checks)
[FAIL] ClassName.test_name   (N/M checks)  ← если есть failed checks
[SKIP] ClassName.test_name   (skip reason)
[ERROR] ClassName.test_name  Exception: ...
```

Итоговая строка:
```
Total: N passed, M failed, K skipped
```

Выводить через обычный `print()`.

---

## ✅ Критерии готовности

1. `SkipTest` — наследник `Exception`, импортируется из `common`
2. `TestRunner.run(obj)` — находит все `test_*` методы, запускает, возвращает `List[TestResult]`
3. `TestRunner.run_all(objects)` — работает с несколькими классами
4. `TestRunner.print_summary(results)` — красиво выводит PASS/FAIL/SKIP/ERROR
5. `SkipTest` перехватывается отдельно (не как FAIL!)
6. Добавить в `Python_test/common/__init__.py`:
   ```python
   from .runner import TestRunner, SkipTest
   ```

---

## ❌ Что НЕ делать

- НЕ использовать `unittest.TestCase` или любой `unittest`
- НЕ использовать `pytest` нигде
- НЕ создавать сложную иерархию — два простых класса
- НЕ добавлять `setUp`/`tearDown` в `TestRunner` — они уже есть в `TestBase`
