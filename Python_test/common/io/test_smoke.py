"""
test_smoke.py — smoke-тесты common.io
========================================

Запуск:
    "F:/Program Files (x86)/Python314/python.exe" Python_test/common/io/test_smoke.py

НЕ pytest! Используется TestRunner из common.runner.
Все файлы пишутся в tempfile.TemporaryDirectory — без следов на диске.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Bootstrap: Python_test/ в sys.path (файл живёт внутри common/io/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.io import ResultStore
from common.result import TestResult, ValidationResult
from common.runner import TestRunner


def _assert(tr: TestResult, passed: bool, name: str,
            actual: float = 0.0, threshold: float = 1.0,
            message: str = "") -> None:
    """Хелпер: добавить boolean-check в TestResult."""
    tr.add(ValidationResult(
        passed=passed,
        metric_name=name,
        actual_value=actual if actual else (1.0 if passed else 0.0),
        threshold=threshold,
        message=message,
    ))


class TestIOSmoke:
    """Smoke-тесты I/O — работают без GPU."""

    def test_find_repo_root_sees_git(self) -> TestResult:
        """ResultStore._PROJECT_ROOT должен указывать на корень репо."""
        tr = TestResult(test_name="find_repo_root_sees_git")
        root = ResultStore._PROJECT_ROOT
        has_git = (root / ".git").exists()
        has_python_test = (root / "Python_test").exists()
        _assert(
            tr,
            passed=has_git and has_python_test,
            name="repo_root_valid",
            message=f"root={root}, .git={has_git}, Python_test={has_python_test}",
        )
        return tr

    def test_save_load_array_roundtrip(self) -> TestResult:
        tr = TestResult(test_name="save_load_array_roundtrip")
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            data = np.random.randn(256).astype(np.float32)
            path = store.save_array(data, "arr", "mymod")
            loaded = store.load_array("arr", "mymod")
            ok = path.exists() and np.allclose(data, loaded)
            _assert(tr, ok, "roundtrip", message=f"path={path}")
        return tr

    def test_save_comparison_keys_and_data(self) -> TestResult:
        """save_comparison → .npz с ключами gpu/ref И одинаковыми данными."""
        tr = TestResult(test_name="save_comparison_keys_and_data")
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            gpu = np.random.randn(128).astype(np.complex64)
            ref = np.ones(128, dtype=np.complex64)
            store.save_comparison(gpu, ref, "cmp", "mymod")
            loaded = store.load_comparison("cmp", "mymod")
            ok = (
                "gpu" in loaded and "ref" in loaded
                and np.allclose(loaded["gpu"], gpu)
                and np.allclose(loaded["ref"], ref)
            )
            _assert(tr, ok, "keys_and_data")
        return tr

    def test_array_exists_sees_npz(self) -> TestResult:
        """CRITICAL: array_exists должен возвращать True и для .npz, не только .npy."""
        tr = TestResult(test_name="array_exists_sees_npz")
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            store.save_comparison(
                np.zeros(8, dtype=np.float32),
                np.ones(8, dtype=np.float32),
                "only_npz", "mymod",
            )
            exists = store.array_exists("only_npz", "mymod")
            _assert(tr, exists, "exists_npz")
        return tr

    def test_save_test_result_uses_test_name(self) -> TestResult:
        """save_test_result сохраняет под правильным именем (не 'unknown')."""
        tr = TestResult(test_name="save_test_result_uses_test_name")
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            inner = TestResult(test_name="example_unit_test")
            inner.add(ValidationResult(
                passed=True, metric_name="m",
                actual_value=0.001, threshold=0.01,
            ))
            path = store.save_test_result(inner, "mymod")

            loaded = store.load_json("example_unit_test", "mymod")
            ok = (
                path.exists()
                and loaded["test_name"] == "example_unit_test"
                and loaded["passed"] is True
            )
            _assert(tr, ok, "testresult_roundtrip", message=f"path={path.name}")
        return tr

    def test_save_test_result_rejects_dict_without_name(self) -> TestResult:
        """save_test_result({dict_without_test_name}) → ValueError."""
        tr = TestResult(test_name="save_test_result_rejects_dict_without_name")
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            raised = False
            try:
                store.save_test_result({"foo": "bar"}, "mymod")
            except ValueError:
                raised = True
            _assert(tr, raised, "raises_valueerror")
        return tr

    def test_save_benchmark_roundtrip(self) -> TestResult:
        tr = TestResult(test_name="save_benchmark_roundtrip")
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            store.save_benchmark(
                {"ms_per_call": 1.23, "ops": 4096}, "fft", "mymod",
            )
            loaded = store.load_json("bench_fft", "mymod")
            ok = loaded["ms_per_call"] == 1.23 and loaded["ops"] == 4096
            _assert(tr, ok, "bench_roundtrip")
        return tr

    def test_save_config_dataclass(self) -> TestResult:
        """save_config принимает dataclass через dataclasses.asdict()."""
        from dataclasses import dataclass

        tr = TestResult(test_name="save_config_dataclass")

        @dataclass
        class _Cfg:
            n: int = 4096
            fs: float = 12e6

        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            store.save_config(_Cfg(), "test_cfg", "mymod")
            loaded = store.load_json("test_cfg_config", "mymod")
            ok = loaded["n"] == 4096 and loaded["fs"] == 12e6
            _assert(tr, ok, "config_roundtrip")
        return tr

    def test_list_arrays_and_results(self) -> TestResult:
        tr = TestResult(test_name="list_arrays_and_results")
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            store.save_array(np.zeros(4), "a", "mymod")
            store.save_array(np.ones(4),  "b", "mymod")
            store.save_comparison(np.zeros(4), np.ones(4), "c", "mymod")
            store.save_benchmark({"k": 1}, "x", "mymod")

            arrs = store.list_arrays("mymod")
            ress = store.list_results("mymod")

            ok = (
                sorted(arrs) == ["a", "b", "c"]
                and sorted(ress) == ["bench_x"]
            )
            _assert(
                tr, ok, "list_consistency",
                message=f"arrays={arrs}, results={ress}",
            )
        return tr


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestIOSmoke())
    runner.print_summary(results)
    sys.exit(0 if all(r.passed for r in results) else 1)
