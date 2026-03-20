"""
conftest.py — фабричные функции для Python_test/integration/
=============================================================

Предоставляет factory functions для интеграционных тестов.
"""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
integration_plot_dir: str = os.path.join(_PROJECT_ROOT, "Results", "Plots", "integration")
os.makedirs(integration_plot_dir, exist_ok=True)


def make_sig_gen(gw, gpu_ctx):
    """SignalGenerator."""
    return gw.SignalGenerator(gpu_ctx)


def make_fft_proc(gw, gpu_ctx):
    """FFTProcessor."""
    return gw.FFTProcessor(gpu_ctx)


def make_script_gen(gw, gpu_ctx):
    """ScriptGenerator. SkipTest если недоступен."""
    from common.runner import SkipTest
    if not hasattr(gw, "ScriptGenerator"):
        raise SkipTest("ScriptGenerator не доступен в этой сборке")
    return gw.ScriptGenerator(gpu_ctx)
