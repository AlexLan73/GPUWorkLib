"""
test_timing_analysis.py — анализ JSON из C++ TimingPerStepTest (T4)
====================================================================

ЗАЧЕМ:
    После того как C++ тесты прогнали TimingPerStepTest — этот Python файл
    берёт сохранённые JSON файлы из Results/strategies/ и делает из них:
    - Таблицу с временем каждого шага GPU (gpu_ms) и wall (wall_ms)
    - Bar chart → Results/Plots/strategies/timing_{signal}.png

    Это позволяет увидеть какой шаг pipeline самый долгий и где узкое место.

ЧТО ПРОВЕРЯЕТ:
    Что timing_*.json файлы существуют (иначе skip — нужно сначала запустить C++).
    Структуру JSON: поля signal, n_ant, n_samples, steps с gpu_ms/wall_ms.
    Sanity: FullProcess < 1000 мс, все gpu_ms >= 0.
    Строит и сохраняет bar chart если matplotlib доступен.

GPU: НЕ НУЖЕН — читает уже готовые JSON, не запускает GPU.

ЗАВИСИМОСТЬ: нужно сначала запустить C++ тест TimingPerStepTest,
    который создаёт Results/strategies/timing_*.json.
    Без этих файлов все тесты будут SkipTest.

ЗАПУСК (из корня проекта):
    python Python_test/strategies/test_timing_analysis.py
"""

import sys
import os
import json
import glob
import numpy as np
from pathlib import Path

_PYTHON_TEST_DIR_EARLY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PYTHON_TEST_DIR_EARLY not in sys.path:
    sys.path.insert(0, _PYTHON_TEST_DIR_EARLY)
from common.runner import SkipTest

_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(os.path.dirname(_DIR))
_RESULTS_DIR = os.path.join(_ROOT, "Results", "strategies")
_PLOTS_DIR   = os.path.join(_ROOT, "Results", "Plots", "strategies")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_timing_json(path: str) -> dict | None:
    """Загрузить JSON файл timing из C++ TimingPerStepTest."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _find_timing_files() -> list[str]:
    """Найти все timing_*.json в Results/strategies/."""
    pattern = os.path.join(_RESULTS_DIR, "timing_*.json")
    return sorted(glob.glob(pattern))


def _print_timing_table(data: dict) -> None:
    """Вывести таблицу timing в консоль."""
    print(f"\n{'─'*52}")
    print(f"  Timing: signal={data.get('signal','?')}  "
          f"n_ant={data.get('n_ant','?')}  n_samples={data.get('n_samples','?')}")
    print(f"{'─'*52}")
    print(f"  {'Step':<20} {'GPU ms':>8}  {'Wall ms':>8}")
    print(f"{'─'*52}")
    steps = data.get("steps", [])
    for s in steps:
        print(f"  {s['name']:<20} {s['gpu_ms']:>8.3f}  {s['wall_ms']:>8.3f}")
    print(f"{'─'*52}")


def _plot_timing_bars(data: dict, out_dir: str) -> str | None:
    """Построить bar chart GPU timing и сохранить в out_dir.

    Returns:
        Путь к сохранённому файлу или None если matplotlib недоступен.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    steps  = data.get("steps", [])
    names  = [s["name"] for s in steps]
    gpu_ms = [s["gpu_ms"] for s in steps]
    signal = data.get("signal", "unknown")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#4C72B0" if n != "FullProcess" else "#DD8452" for n in names]
    bars    = ax.bar(names, gpu_ms, color=colors, edgecolor="black", linewidth=0.5)

    ax.bar_label(bars, labels=[f"{v:.2f}" for v in gpu_ms],
                 padding=3, fontsize=9)
    ax.set_xlabel("Pipeline Step")
    ax.set_ylabel("GPU time (ms)")
    ax.set_title(
        f"AntennaProcessor Step Timing — {signal}\n"
        f"n_ant={data.get('n_ant')}  n_samples={data.get('n_samples')}  "
        f"fs={data.get('fs', 0)/1e6:.2f} MHz"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"timing_{signal}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_timing_files_exist():
    """Проверить что JSON файлы timing созданы C++ TimingPerStepTest."""
    files = _find_timing_files()
    if not files:
        raise SkipTest(
            f"No timing_*.json found in {_RESULTS_DIR}. "
            "Run C++ TimingPerStepTest first."
        )
    print(f"\nFound {len(files)} timing file(s):")
    for f in files:
        print(f"  {f}")
    assert len(files) > 0


def test_timing_json_valid():
    """Проверить структуру JSON файлов."""
    files = _find_timing_files()
    if not files:
        raise SkipTest("No timing files found.")

    for path in files:
        data = _load_timing_json(path)
        assert data is not None, f"Failed to load {path}"
        assert "signal"   in data, f"Missing 'signal' in {path}"
        assert "steps"    in data, f"Missing 'steps' in {path}"
        assert "n_ant"    in data, f"Missing 'n_ant' in {path}"
        assert "n_samples" in data, f"Missing 'n_samples' in {path}"
        assert len(data["steps"]) > 0, f"Empty steps in {path}"

        for step in data["steps"]:
            assert "name"    in step, f"Missing name in step: {step}"
            assert "gpu_ms"  in step, f"Missing gpu_ms in step: {step}"
            assert "wall_ms" in step, f"Missing wall_ms in step: {step}"
            assert step["gpu_ms"] >= 0, f"Negative gpu_ms: {step}"


def test_timing_sanity():
    """Sanity: FullProcess < 1000 ms, все шаги > 0 ms."""
    files = _find_timing_files()
    if not files:
        raise SkipTest("No timing files found.")

    for path in files:
        data = _load_timing_json(path)
        _print_timing_table(data)

        for step in data["steps"]:
            assert step["gpu_ms"] >= 0, f"Negative gpu_ms: {step['name']}"
            if step["name"] == "FullProcess":
                assert step["gpu_ms"] < 1000, \
                    f"FullProcess {step['gpu_ms']:.1f} ms > 1000 ms"


def test_plot_timing_bars():
    """Построить bar chart для всех найденных JSON файлов."""
    files = _find_timing_files()
    if not files:
        raise SkipTest("No timing files found.")

    for path in files:
        data = _load_timing_json(path)
        out  = _plot_timing_bars(data, _PLOTS_DIR)
        if out:
            print(f"  Plot saved: {out}")
        else:
            print("  matplotlib not available — plot skipped")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone script
# ─────────────────────────────────────────────────────────────────────────────

def parse_and_report(results_dir: str = _RESULTS_DIR,
                     plots_dir: str   = _PLOTS_DIR) -> None:
    """Разобрать все timing JSON и вывести таблицу + графики.

    Можно вызвать напрямую:
        python test_timing_analysis.py
    """
    files = sorted(glob.glob(os.path.join(results_dir, "timing_*.json")))
    if not files:
        print(f"No timing files found in {results_dir}")
        return

    for path in files:
        data = _load_timing_json(path)
        if data is None:
            continue
        _print_timing_table(data)
        out = _plot_timing_bars(data, plots_dir)
        if out:
            print(f"  Plot saved: {out}")


if __name__ == "__main__":
    parse_and_report()
