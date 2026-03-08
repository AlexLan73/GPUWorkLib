"""
conftest.py — pytest fixtures для Python_test/strategies/
==========================================================

Fixtures для тестов pipeline (beamforming, Farrow, scenario).
Наследует gpu_ctx, gw из корневого conftest.py.

Fixtures:
    strategy_plot_dir — Results/Plots/strategies/
    scenario_8ant     — стандартный сценарий 8 антенн, 1 цель, theta=30°
    scenario_multi    — сценарий с 3 целями
    pipeline_runner   — PipelineRunner без checkpoint'ов
    farrow            — FarrowDelay (numpy, не GPU)
"""

import os
import sys
import pytest
import numpy as np

# Добавить strategies/ в sys.path для импорта scenario_builder, farrow_delay
_STRATEGIES_DIR = os.path.dirname(os.path.abspath(__file__))
if _STRATEGIES_DIR not in sys.path:
    sys.path.insert(0, _STRATEGIES_DIR)


@pytest.fixture(scope="session")
def strategy_plot_dir(plot_dir) -> str:
    """Директория Results/Plots/strategies/."""
    path = os.path.join(plot_dir, "strategies")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def farrow():
    """FarrowDelay — numpy реализация (не требует GPU)."""
    from farrow_delay import FarrowDelay
    return FarrowDelay()


@pytest.fixture(scope="module")
def scenario_8ant():
    """Сценарий: 8 антенн, 1 цель @ theta=30°, fs=12МГц."""
    from scenario_builder import make_single_target
    return make_single_target(
        n_ant=8,
        theta_deg=30.0,
        fs=12e6,
        n_samples=4096,
        fdev_hz=1e6,
    )


@pytest.fixture(scope="module")
def scenario_multi():
    """Сценарий: 8 антенн, 3 цели @ 15°/30°/45°."""
    from scenario_builder import make_multi_target
    return make_multi_target(
        n_ant=8,
        thetas_deg=[15.0, 30.0, 45.0],
        fs=12e6,
        n_samples=4096,
        fdev_hz=1e6,
    )


@pytest.fixture(scope="session")
def pipeline_runner():
    """PipelineRunner без checkpoint'ов (нет вывода на диск)."""
    from pipeline_runner import PipelineRunner
    return PipelineRunner(output_dir=None)
