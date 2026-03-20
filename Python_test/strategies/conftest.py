"""
conftest.py — фабричные функции для Python_test/strategies/
============================================================

Предоставляет фабричные функции. Каждый вызов make_*() создаёт новый объект.

Использование:
    from conftest import make_farrow, make_scenario_8ant, strategy_plot_dir

    def test_something():
        farrow = make_farrow()
        scenario = make_scenario_8ant()
        ...
"""

import os
import sys

# Добавить strategies/ в sys.path
_STRATEGIES_DIR = os.path.dirname(os.path.abspath(__file__))
if _STRATEGIES_DIR not in sys.path:
    sys.path.insert(0, _STRATEGIES_DIR)

# Добавить Python_test/ в sys.path
_PT_DIR = os.path.dirname(_STRATEGIES_DIR)
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Пути
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(_PT_DIR)
strategy_plot_dir: str = os.path.join(_PROJECT_ROOT, "Results", "Plots", "strategies")
os.makedirs(strategy_plot_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Фабричные функции
# ─────────────────────────────────────────────────────────────────────────────

def make_farrow():
    """FarrowDelay — numpy реализация (не требует GPU)."""
    from farrow_delay import FarrowDelay
    return FarrowDelay()


def make_scenario_8ant():
    """Сценарий: 8 антенн, 1 цель @ theta=30°, fs=12МГц."""
    from scenario_builder import make_single_target
    return make_single_target(
        n_ant=8,
        theta_deg=30.0,
        fs=12e6,
        n_samples=4096,
        fdev_hz=1e6,
    )


def make_scenario_multi():
    """Сценарий: 8 антенн, 3 цели @ 15°/30°/45°."""
    from scenario_builder import make_multi_target
    return make_multi_target(
        n_ant=8,
        thetas=[15.0, 30.0, 45.0],
        f0s=[2e6, 3e6, 4e6],
        fdevs=[1e6, 1e6, 1e6],
        fs=12e6,
        n_samples=4096,
    )


def make_pipeline_runner():
    """PipelineRunner без checkpoint'ов (нет вывода на диск)."""
    from pipeline_runner import PipelineRunner
    return PipelineRunner(output_dir=None)
