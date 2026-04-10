"""
test_smoke.py — smoke-тесты common.validators
================================================

Запуск:
    "F:/Program Files (x86)/Python314/python.exe" Python_test/common/validators/test_smoke.py

НЕ pytest! Используется собственный TestRunner из common.runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap: Python_test/ в sys.path (файл живёт внутри common/validators/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.result import TestResult, ValidationResult
from common.runner import TestRunner


class TestValidatorsSmoke:
    """Smoke-тесты валидаторов — работают без GPU."""

    # ── Backward compatibility ──────────────────────────────────────────────

    def test_backward_compat_public_attrs(self) -> TestResult:
        """DataValidator должен иметь публичные .tolerance/.metric/.METRICS."""
        from common import DataValidator

        tr = TestResult(test_name="backward_compat_public_attrs")
        v = DataValidator(tolerance=0.01, metric="max_rel")

        # Атрибуты, которые могут использовать внешние тесты
        ok = (
            v.tolerance == 0.01
            and v.metric == "max_rel"
            and DataValidator.METRICS == ("max_rel", "abs", "rmse")
        )
        tr.add(ValidationResult(
            passed=ok,
            metric_name="public_attrs",
            actual_value=1.0 if ok else 0.0,
            threshold=1.0,
            message=f"tolerance={v.tolerance}, metric={v.metric}",
        ))
        return tr

    def test_backward_compat_real_max_rel(self) -> TestResult:
        """DataValidator(max_rel) на real-массивах — как было."""
        from common import DataValidator

        tr = TestResult(test_name="backward_compat_real_max_rel")
        v = DataValidator(tolerance=0.01, metric="max_rel")
        a = np.ones(100, dtype=np.float32)
        r = np.ones(100, dtype=np.float32) * 1.005  # 0.5% отклонение
        tr.add(v.validate(a, r, "real_0.5%"))
        return tr

    def test_backward_compat_complex_detects_im(self) -> TestResult:
        """CRITICAL: DataValidator не должен терять мнимую часть complex64.

        Старый баг (исправленный): .astype(float64) отбрасывал Im → разница 0
        → PASS (ложно). Проверяем, что теперь разница Im-части ловится.
        """
        from common import DataValidator

        tr = TestResult(test_name="backward_compat_complex_detects_im")
        v = DataValidator(tolerance=1e-6, metric="max_rel")
        a = np.ones(100, dtype=np.complex64) * (1 + 0j)
        r = np.ones(100, dtype=np.complex64) * (1 + 0.5j)
        inner = v.validate(a, r, "complex_im_diff")

        # Ожидаем, что inner.passed = False (разница есть), actual ≈ 0.447
        caught = (not inner.passed) and (inner.actual_value > 0.4)
        tr.add(ValidationResult(
            passed=caught,
            metric_name="im_part_detected",
            actual_value=inner.actual_value,
            threshold=1.0,
            message=(
                f"inner.passed={inner.passed}, actual={inner.actual_value:.4f} "
                f"(ожидаем ≈0.447)"
            ),
        ))
        return tr

    # ── Новый API ───────────────────────────────────────────────────────────

    def test_relative_validator_complex_self(self) -> TestResult:
        """RelativeValidator(complex_a, complex_a) → PASS."""
        from common.validators import RelativeValidator

        tr = TestResult(test_name="relative_validator_complex_self")
        v = RelativeValidator(1e-6)
        a = np.array([1 + 0j, 2 + 1j, -1 - 0.5j], dtype=np.complex64)
        tr.add(v.validate(a, a, "self_complex"))
        return tr

    def test_absolute_validator_frequency_hz(self) -> TestResult:
        """AbsoluteValidator для частот в Гц."""
        from common.validators import AbsoluteValidator

        tr = TestResult(test_name="absolute_validator_frequency_hz")
        v = AbsoluteValidator(tolerance=50e3)
        freq_gpu = np.array([2.01e6], dtype=np.float32)
        freq_ref = np.array([2.00e6], dtype=np.float32)
        tr.add(v.validate(freq_gpu, freq_ref, "peak_freq"))
        return tr

    def test_rmse_validator(self) -> TestResult:
        """RmseValidator на шумных данных."""
        from common.validators import RmseValidator

        tr = TestResult(test_name="rmse_validator")
        rng = np.random.default_rng(42)
        ref = np.sin(np.linspace(0, 10, 512)).astype(np.float32)
        gpu = ref + rng.normal(0, 0.002, size=512).astype(np.float32)
        v = RmseValidator(tolerance=0.05)
        tr.add(v.validate(gpu, ref, "sine_plus_noise"))
        return tr

    def test_frequency_validator_standalone(self) -> TestResult:
        """FrequencyValidator — standalone (reference игнорируется)."""
        from common.validators import FrequencyValidator

        tr = TestResult(test_name="frequency_validator_standalone")
        fs, n, f0 = 12e6, 4096, 2e6
        t = np.arange(n) / fs
        cw = np.exp(2j * np.pi * f0 * t).astype(np.complex64)

        v = FrequencyValidator(expected_hz=f0, tolerance_hz=2e3, fs=fs)
        tr.add(v.validate(cw, reference=None, name="cw_peak"))
        return tr

    def test_power_validator_standalone(self) -> TestResult:
        """PowerValidator — амплитуда CW-сигнала."""
        from common.validators import PowerValidator

        tr = TestResult(test_name="power_validator_standalone")
        cw = np.ones(1024, dtype=np.complex64)  # |A| = 1 → power = 1
        v = PowerValidator(expected_power=1.0, tolerance=0.01)
        tr.add(v.validate(cw))
        return tr

    # ── Composite ───────────────────────────────────────────────────────────

    def test_composite_all_pass(self) -> TestResult:
        """CompositeValidator: 2 валидатора, оба PASS."""
        from common.validators import (
            CompositeValidator, RelativeValidator, FrequencyValidator,
        )

        tr = TestResult(test_name="composite_all_pass")
        fs, n, f0 = 12e6, 4096, 2e6
        t = np.arange(n) / fs
        cw = np.exp(2j * np.pi * f0 * t).astype(np.complex64)

        composite = CompositeValidator(
            RelativeValidator(1e-3, "amplitude"),
            FrequencyValidator(expected_hz=f0, tolerance_hz=2e3, fs=fs),
        )
        vr = composite.validate(cw, cw)
        # Дополнительно проверяем что actual_value = n_pass, threshold = n_total
        ok = vr.passed and vr.actual_value == 2.0 and vr.threshold == 2.0
        tr.add(ValidationResult(
            passed=ok,
            metric_name="composite_agg",
            actual_value=vr.actual_value,
            threshold=vr.threshold,
            message=vr.message,
        ))
        return tr

    def test_composite_fluent_add(self) -> TestResult:
        """CompositeValidator.add() fluent API."""
        from common.validators import CompositeValidator, RelativeValidator

        tr = TestResult(test_name="composite_fluent_add")
        a = np.ones(100, dtype=np.float32)
        c = CompositeValidator()
        c.add(RelativeValidator(0.01)).add(RelativeValidator(0.1))
        tr.add(c.validate(a, a, "dual"))
        return tr

    # ── Factory ─────────────────────────────────────────────────────────────

    def test_factory_create(self) -> TestResult:
        """ValidatorFactory.create() возвращает правильный тип."""
        from common.validators import (
            ValidatorFactory, RelativeValidator, AbsoluteValidator, RmseValidator,
        )

        tr = TestResult(test_name="factory_create")

        mapping = [
            ("max_rel", RelativeValidator),
            ("abs",     AbsoluteValidator),
            ("rmse",    RmseValidator),
        ]
        for metric, cls in mapping:
            v = ValidatorFactory.create(metric, 0.01)
            ok = isinstance(v, cls)
            tr.add(ValidationResult(
                passed=ok,
                metric_name=f"factory_{metric}",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
            ))
        return tr

    def test_factory_unknown_metric(self) -> TestResult:
        """ValidatorFactory.create("unknown") → ValueError."""
        from common.validators import ValidatorFactory

        tr = TestResult(test_name="factory_unknown_metric")
        try:
            ValidatorFactory.create("not_a_metric", 0.01)
            passed = False
            msg = "ValueError НЕ был выброшен"
        except ValueError:
            passed = True
            msg = "ValueError OK"

        tr.add(ValidationResult(
            passed=passed,
            metric_name="raises_valueerror",
            actual_value=1.0 if passed else 0.0,
            threshold=1.0,
            message=msg,
        ))
        return tr

    def test_factory_create_for_signal(self) -> TestResult:
        """ValidatorFactory.create_for_signal(...) → Composite с 2 валидаторами."""
        from common.validators import ValidatorFactory, CompositeValidator

        tr = TestResult(test_name="factory_create_for_signal")
        composite = ValidatorFactory.create_for_signal(
            expected_hz=2e6, fs=12e6, tolerance_hz=1e3,
        )
        ok = isinstance(composite, CompositeValidator) and len(composite) == 2
        tr.add(ValidationResult(
            passed=ok,
            metric_name="composite_2",
            actual_value=float(len(composite)),
            threshold=2.0,
        ))
        return tr

    # ── Fail-fast: None reference ────────────────────────────────────────────

    def test_comparative_raises_on_none_ref(self) -> TestResult:
        """RelativeValidator.validate(x, None) → ValueError (fail-fast)."""
        from common.validators import RelativeValidator

        tr = TestResult(test_name="comparative_raises_on_none_ref")
        a = np.ones(10, dtype=np.float32)
        v = RelativeValidator(0.01)
        try:
            v.validate(a, None)
            passed = False
            msg = "ValueError НЕ был выброшен"
        except ValueError as exc:
            passed = True
            msg = str(exc)

        tr.add(ValidationResult(
            passed=passed,
            metric_name="raises_valueerror",
            actual_value=1.0 if passed else 0.0,
            threshold=1.0,
            message=msg,
        ))
        return tr


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestValidatorsSmoke())
    runner.print_summary(results)
    # Exit code для CI
    sys.exit(0 if all(r.passed for r in results) else 1)
