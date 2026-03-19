"""
conftest.py — общие helpers и синглтоны для Python_test/
=========================================================

Без pytest. Предоставляет ленивую инициализацию GPU-контекста
через GPULoader / GPUContextManager.

Использование в тестах:
    from conftest import get_gw, get_gpu_ctx, get_rocm_ctx, PROJECT_ROOT, PLOT_DIR

    def test_something():
        ctx = get_rocm_ctx()   # SkipTest если ROCm недоступен
        ...

    def test_without_gpu():
        ...  # не вызывает GPU-функции → работает без GPU
"""

import os
import sys
import numpy as np

# Добавить Python_test/ в sys.path для импорта common.*
_PT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

from common.runner import SkipTest
from common.gpu_loader import GPULoader
from common.gpu_context import GPUContextManager

# ─────────────────────────────────────────────────────────────────────────────
# Константы (модульный уровень)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT: str = os.path.dirname(_PT_DIR)
PLOT_DIR: str = os.path.join(PROJECT_ROOT, "Results", "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy GPU helpers (синглтоны, создаются при первом вызове)
# ─────────────────────────────────────────────────────────────────────────────

def get_gw():
    """Модуль gpuworklib. SkipTest если не найден."""
    module = GPULoader.get()
    if module is None:
        raise SkipTest(
            "gpuworklib не найден. "
            f"Проверьте пути: {getattr(GPULoader, '_SEARCH_PATHS', 'build/')}"
        )
    return module


def get_gpu_ctx():
    """GPU-контекст (OpenCL). SkipTest если GPU недоступен."""
    get_gw()  # убедиться что gpuworklib загружен
    ctx = GPUContextManager.get()
    if ctx is None:
        raise SkipTest("GPU-контекст не создан (нет GPU или ошибка драйвера)")
    return ctx


def get_rocm_ctx():
    """ROCmGPUContext. SkipTest если ROCm недоступен."""
    get_gw()  # убедиться что gpuworklib загружен
    ctx = GPUContextManager.get_rocm()
    if ctx is None:
        raise SkipTest("ROCmGPUContext не создан (нет ROCm GPU или ошибка драйвера)")
    return ctx


def make_rng(seed: int = 42) -> np.random.Generator:
    """NumPy RNG с фиксированным seed (воспроизводимость)."""
    return np.random.default_rng(seed=seed)
