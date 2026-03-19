"""
GPUWorkLib: SpectrumMaximaFinder ROCm Tests (AMD, hipFFT)
==========================================================

ROCm-специфичные тесты для поиска максимумов спектра.
Используют ROCmGPUContext + hipFFT (не clFFT).

Тесты проверяют pipeline: FFT (hipFFT) → SpectrumProcessorROCm → peaks.

Примечание: Прямого Python API для SpectrumProcessorROCm нет.
Проверка выполняется через HeterodyneDechirp (который использует spectrum pipeline на ROCm)
или через запуск C++ тестов.

Tests:
  1. rocm_context_available — ROCmGPUContext создаётся
  2. spectrum_via_heterodyne — косвенная проверка: dechirp находит f_beat

Author: Kodo (AI Assistant)
Date: 2026-03-02
"""

import sys
import os
import subprocess

# Path to gpuworklib
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PT_DIR = os.path.join(PROJECT_ROOT, "Python_test")
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.runner import SkipTest
for subdir in ["build/python", "build/debian-radeon9070/python", "build/python/Release", "build/python/Debug"]:
    p = os.path.join(PROJECT_ROOT, subdir.replace("/", os.sep))
    if os.path.isdir(p):
        sys.path.insert(0, p)
        break

try:
    import gpuworklib
    import numpy as np
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def _is_amd_rocm():
    """Проверить, что ROCm доступен (AMD GPU)."""
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=3)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ============================================================================
# Test 1: ROCm context available
# ============================================================================
def test_rocm_context_available():
    """ROCmGPUContext создаётся на AMD."""
    if not HAS_GPU:
        raise SkipTest("gpuworklib not found")
    if not _is_amd_rocm():
        raise SkipTest("ROCm not available (need AMD GPU)")
    if not hasattr(gpuworklib, "ROCmGPUContext"):
        raise SkipTest("gpuworklib built without ROCm")
    ctx = gpuworklib.ROCmGPUContext(0)
    assert ctx is not None
    assert "AMD" in ctx.device_name or "Radeon" in ctx.device_name
    print(f"  ROCm device: {ctx.device_name}")


# ============================================================================
# Test 2: Spectrum pipeline via HeterodyneDechirp (ROCm)
# ============================================================================
def test_spectrum_via_heterodyne_rocm():
    """
    Косвенная проверка spectrum pipeline: HeterodyneDechirp использует
    FFT + SpectrumProcessorROCm внутри. Генерируем LFM с задержкой,
    dechirp должен найти f_beat.
    """
    if not HAS_GPU:
        raise SkipTest("gpuworklib not found")
    if not _is_amd_rocm():
        raise SkipTest("ROCm not available (need AMD GPU)")
    if not hasattr(gpuworklib, "HeterodyneDechirp") or not hasattr(gpuworklib, "ROCmGPUContext"):
        raise SkipTest("HeterodyneDechirp or ROCmGPUContext not available")

    ctx = gpuworklib.ROCmGPUContext(0)
    het = gpuworklib.HeterodyneDechirp(ctx)
    fs = 12e6
    f_start, f_end = 0.0, 2e6
    n = 8000
    het.set_params(float(f_start), float(f_end), float(fs), int(n), 1)

    # Генерируем LFM с задержкой 100 us -> f_beat = 300 kHz
    t = np.arange(n, dtype=np.float32) / fs
    mu = (f_end - f_start) / (n / fs)
    delay_us = 100.0
    tau = delay_us * 1e-6
    t_delayed = t - tau
    phase = 2 * np.pi * (0.5 * mu * t_delayed**2 + f_start * t_delayed)
    rx = np.exp(1j * phase).astype(np.complex64)

    result = het.process(rx)
    assert result["success"], f"Process failed: {result.get('error_message', '')}"
    assert len(result["antennas"]) == 1
    f_beat = result["antennas"][0]["f_beat_hz"]
    expected = 3e9 * 100e-6  # 300 kHz
    err = abs(f_beat - expected)
    assert err < 5000, f"f_beat error {err:.0f} Hz (expected ~{expected:.0f})"
    print(f"  f_beat={f_beat/1e3:.1f} kHz, expected={expected/1e3:.0f} kHz, err={err:.0f} Hz")
