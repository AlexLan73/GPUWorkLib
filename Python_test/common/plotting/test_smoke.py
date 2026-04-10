"""
test_smoke.py — smoke-тесты common.plotting
==============================================

Запуск:
    "F:/Program Files (x86)/Python314/python.exe" Python_test/common/plotting/test_smoke.py

Работает БЕЗ GPU. Все PNG пишутся в tempfile.TemporaryDirectory.
Если matplotlib не установлен — SkipTest.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Bootstrap: Python_test/ в sys.path (файл живёт внутри common/plotting/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.result import TestResult, ValidationResult
from common.runner import SkipTest, TestRunner


def _check_matplotlib() -> None:
    """Пропускает smoke-тесты если matplotlib недоступен."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SkipTest(f"matplotlib не установлен: {exc}")


def _cw(fs: float, n: int, f0: float) -> np.ndarray:
    t = np.arange(n) / fs
    return np.exp(2j * np.pi * f0 * t).astype(np.complex64)


def _lfm(fs: float, n: int, f_start: float, f_end: float) -> np.ndarray:
    T = n / fs
    t = np.arange(n) / fs
    k = (f_end - f_start) / T
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t * t)
    return np.exp(1j * phase).astype(np.complex64)


class TestPlottingSmoke:
    """Smoke-тесты плоттеров — без GPU, во временной директории."""

    def setUp(self) -> None:
        _check_matplotlib()

    def test_spectrum_plotter_saves_png(self) -> TestResult:
        """SpectrumPlotter сохраняет PNG + slugify работает со спецсимволами."""
        from common.plotting import PlotterFactory

        tr = TestResult(test_name="spectrum_plotter_saves_png")
        cw = _cw(12e6, 4096, 2e6)
        with tempfile.TemporaryDirectory() as tmp:
            factory = PlotterFactory(
                "test_module", out_dir=tmp, save=True, show=False,
            )
            # Заголовок со спецсимволами — проверяем slugify
            path = factory.spectrum().plot(
                cw, 12e6, title="CW 2MHz: f0=2e6/test",
            )
            p = Path(path)
            exists = p.exists()
            no_colon = ":" not in p.name
            no_slash = "/" not in p.name
            ok = exists and no_colon and no_slash

            tr.add(ValidationResult(
                passed=ok,
                metric_name="spectrum_saved_slugified",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
                message=f"name={p.name}",
            ))
        return tr

    def test_time_plotter_iq(self) -> TestResult:
        """TimePlotter.plot() строит I/Q субплоты."""
        from common.plotting import PlotterFactory

        tr = TestResult(test_name="time_plotter_iq")
        lfm = _lfm(12e6, 4096, 0.0, 2e6)
        with tempfile.TemporaryDirectory() as tmp:
            factory = PlotterFactory(
                "test_module", out_dir=tmp, save=True, show=False,
            )
            path = factory.timeseries().plot(lfm, 12e6, title="LFM")
            ok = Path(path).exists()
            tr.add(ValidationResult(
                passed=ok,
                metric_name="time_saved",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
                message=f"path={path}",
            ))
        return tr

    def test_time_plotter_magnitude(self) -> TestResult:
        """TimePlotter.plot_magnitude() строит |A|(t)."""
        from common.plotting import PlotterFactory

        tr = TestResult(test_name="time_plotter_magnitude")
        cw = _cw(12e6, 2048, 1e6)
        with tempfile.TemporaryDirectory() as tmp:
            factory = PlotterFactory(
                "test_module", out_dir=tmp, save=True, show=False,
            )
            path = factory.timeseries().plot_magnitude(cw, 12e6, title="CW mag")
            ok = Path(path).exists()
            tr.add(ValidationResult(
                passed=ok,
                metric_name="mag_saved",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
            ))
        return tr

    def test_subdir_creates_new_config(self) -> TestResult:
        """factory.spectrum(subdir='FormSignal') → подпапка."""
        from common.plotting import PlotterFactory

        tr = TestResult(test_name="subdir_creates_new_config")
        with tempfile.TemporaryDirectory() as tmp:
            factory = PlotterFactory(
                "test_module", out_dir=tmp, save=True, show=False,
            )
            cw = _cw(12e6, 1024, 1e6)
            path = factory.spectrum(subdir="FormSignal").plot(cw, 12e6, title="sub")

            ok = "FormSignal" in str(path) and Path(path).exists()
            tr.add(ValidationResult(
                passed=ok,
                metric_name="subdir_ok",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
                message=f"path={path}",
            ))
        return tr

    def test_plot_compare_two_spectra(self) -> TestResult:
        """plot_compare рисует два спектра на одной figure."""
        from common.plotting import PlotterFactory

        tr = TestResult(test_name="plot_compare_two_spectra")
        gpu = _cw(12e6, 2048, 1e6)
        ref = gpu * 1.001
        with tempfile.TemporaryDirectory() as tmp:
            factory = PlotterFactory(
                "test_module", out_dir=tmp, save=True, show=False,
            )
            path = factory.spectrum().plot_compare(
                gpu, ref, 12e6, title="GPU vs Ref",
            )
            ok = Path(path).exists()
            tr.add(ValidationResult(
                passed=ok,
                metric_name="compare_saved",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
            ))
        return tr

    def test_factory_config_property(self) -> TestResult:
        """PlotterFactory.config — property, возвращает PlotConfig."""
        from common.plotting import PlotterFactory, PlotConfig

        tr = TestResult(test_name="factory_config_property")
        factory = PlotterFactory("test_module")
        cfg = factory.config
        ok = isinstance(cfg, PlotConfig) and "test_module" in cfg.out_dir
        tr.add(ValidationResult(
            passed=ok,
            metric_name="config_property",
            actual_value=1.0 if ok else 0.0,
            threshold=1.0,
            message=f"out_dir={cfg.out_dir}",
        ))
        return tr


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestPlottingSmoke())
    runner.print_summary(results)
    sys.exit(0 if all(r.passed for r in results) else 1)
