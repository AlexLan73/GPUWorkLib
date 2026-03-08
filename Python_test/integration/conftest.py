"""
conftest.py — pytest fixtures для Python_test/integration/
===========================================================

Интеграционные тесты проверяют совместную работу нескольких модулей:
  FFTProcessor + SignalGenerator, ScriptGenerator, ...

Fixtures:
    integration_plot_dir — Results/Plots/integration/
    sig_gen              — SignalGenerator (scope=session)
    fft_proc             — FFTProcessor (scope=session)
    script_gen           — ScriptGenerator (scope=session, если доступен)
"""

import os
import pytest


@pytest.fixture(scope="session")
def integration_plot_dir(plot_dir) -> str:
    """Директория Results/Plots/integration/."""
    path = os.path.join(plot_dir, "integration")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def sig_gen(gw, gpu_ctx):
    """SignalGenerator (scope=session)."""
    return gw.SignalGenerator(gpu_ctx)


@pytest.fixture(scope="session")
def fft_proc(gw, gpu_ctx):
    """FFTProcessor (scope=session)."""
    return gw.FFTProcessor(gpu_ctx)


@pytest.fixture(scope="session")
def script_gen(gw, gpu_ctx):
    """ScriptGenerator (scope=session, skip если недоступен)."""
    if not hasattr(gw, "ScriptGenerator"):
        pytest.skip("ScriptGenerator не доступен в этой сборке")
    return gw.ScriptGenerator(gpu_ctx)
