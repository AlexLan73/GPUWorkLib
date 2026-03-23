"""
runner.py — TestRunner + SkipTest
==================================

Инфраструктура запуска тестов.

Classes:
    SkipTest   — исключение для пропуска теста
    TestRunner — Coordinator (GRASP): запускает методы test_* в объекте,
                 собирает TestResult, выводит сводку PASS/FAIL/SKIP.

Дизайн:
    Explicit better than implicit — TestRunner не использует магические
    декораторы или метаклассы. Достаточно начать имя метода с "test_".
"""

from .result import TestResult, ValidationResult


class SkipTest(Exception):
    """Пропуск теста — GPU недоступен или тест не применим.

    Бросается внутри setUp() или test_*().
    TestRunner перехватывает и помечает тест как SKIP (не FAIL).

    Usage:
        def setUp(self):
            if not GPUContextManager.is_rocm_available():
                raise SkipTest("ROCm GPU не доступен")
    """
    pass


class TestRunner:
    """Запускает методы test_* в объекте, собирает TestResult.

    Coordinator (GRASP): координирует обнаружение и запуск тестов.
    Explicit лучше implicit.

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
        results = []
        test_methods = sorted([
            name for name in dir(obj)
            if name.startswith("test_") and callable(getattr(obj, name))
        ])

        setup = getattr(obj, "setUp", None)
        teardown = getattr(obj, "tearDown", None)

        for method_name in test_methods:
            method = getattr(obj, method_name)
            full_name = f"{obj.__class__.__name__}.{method_name}"
            result = TestResult(test_name=full_name)

            try:
                if setup:
                    setup()
                method_result = method()
                if isinstance(method_result, TestResult):
                    method_result.test_name = full_name
                    result = method_result
                else:
                    # test used assert-style (returned None) — mark as passed
                    result.metadata["assert_passed"] = True
            except SkipTest as e:
                result.metadata["skipped"] = True
                result.metadata["skip_reason"] = str(e)
            except Exception as e:
                result.error = e
            finally:
                if teardown:
                    try:
                        teardown()
                    except Exception:
                        pass

            results.append(result)

        return results

    def run_all(self, objects: list) -> list:
        """Запустить тесты в нескольких объектах.

        Args:
            objects: список экземпляров тест-классов

        Returns:
            List[TestResult] — объединённый список результатов
        """
        results = []
        for obj in objects:
            results.extend(self.run(obj))
        return results

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
        print("=" * 40)
        print("TEST SUMMARY")
        print("=" * 40)

        n_pass = 0
        n_fail = 0
        n_skip = 0

        for r in results:
            if r.metadata.get("skipped"):
                n_skip += 1
                reason = r.metadata.get("skip_reason", "")
                print(f"[SKIP]  {r.test_name}   ({reason})")

            elif r.error is not None:
                n_fail += 1
                print(f"[ERROR] {r.test_name}  Exception: {r.error}")

            elif r.passed:
                n_pass += 1
                n_total = len(r.validations)
                print(f"[PASS]  {r.test_name}   ({n_total}/{n_total} checks)")

            else:
                n_fail += 1
                n_total = len(r.validations)
                n_ok = sum(1 for v in r.validations if v.passed)
                fail_msg = ""
                for v in r.validations:
                    if not v.passed:
                        fail_msg = (f"  ERROR: {v.metric_name}="
                                    f"{v.actual_value:.4g} > tol={v.threshold:.4g}")
                        break
                print(f"[FAIL]  {r.test_name}   ({n_ok}/{n_total} checks){fail_msg}")

        print("-" * 40)
        print(f"Total: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
        print("=" * 40)
