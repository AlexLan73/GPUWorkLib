"""
conftest.py — корневые pytest fixtures для Python_test/
========================================================

Fixtures доступны во всех тестах Python_test/ без явного импорта.

Fixtures:
    gw          — модуль gpuworklib (scope=session)
    gpu_ctx     — GPU-контекст GPUContext (scope=session)
    project_root — путь к корню проекта
    plot_dir    — путь к директории Results/Plots/

Использование в тестах:
    def test_something(gpu_ctx):
        filter = gpuworklib.FirFilterROCm(gpu_ctx, ...)
        ...

    def test_without_gpu():
        # не принимает gpu_ctx → работает без GPU
        ...
"""

import os
import sys
import pytest

# Добавить Python_test/ в sys.path для импорта common.*
_PT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

from common.gpu_loader import GPULoader
from common.gpu_context import GPUContextManager


# ─────────────────────────────────────────────────────────────────────────────
# Session-scope fixtures (создаются один раз для всей pytest-сессии)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def gw():
    """Модуль gpuworklib. Skip если не найден."""
    module = GPULoader.get()
    if module is None:
        pytest.skip(
            "gpuworklib не найден. "
            f"Проверьте пути: {GPULoader._SEARCH_PATHS if hasattr(GPULoader, '_SEARCH_PATHS') else 'build/'}"
        )
    return module


@pytest.fixture(scope="session")
def gpu_ctx(gw):
    """GPU-контекст. Skip если GPU недоступен."""
    ctx = GPUContextManager.get()
    if ctx is None:
        pytest.skip("GPU-контекст не создан (нет GPU или ошибка драйвера)")
    return ctx


@pytest.fixture(scope="session")
def project_root() -> str:
    """Абсолютный путь к корню проекта GPUWorkLib."""
    return os.path.dirname(_PT_DIR)


@pytest.fixture(scope="session")
def plot_dir(project_root) -> str:
    """Путь к директории Results/Plots/."""
    path = os.path.join(project_root, "Results", "Plots")
    os.makedirs(path, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Function-scope fixtures (создаются для каждого теста)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    """NumPy RNG с фиксированным seed (воспроизводимость)."""
    import numpy as np
    return np.random.default_rng(seed=42)
