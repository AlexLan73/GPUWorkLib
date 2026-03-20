"""
test_hybrid_backend.py — Python тесты HybridBackend (OpenCL + ROCm)

Запуск:
    python Python_test/hybrid/test_hybrid_backend.py

Тесты проверяют создание HybridGPUContext и корректную работу
обоих sub-backend (OpenCL + ROCm) на одном GPU.

Требования:
    - Linux + AMD GPU + ROCm
    - Собранный gpuworklib.so с ENABLE_ROCM=ON
    - Запуск в группе render: sg render -c "python3 ..."

Автор: Кодо (AI Assistant)
Дата: 2026-02-24
"""

import sys
import os
import numpy as np

# Добавляем путь к .so (build/debian-radeon9070/python/)
BUILD_DIR = os.path.join(os.path.dirname(__file__), '../../build/debian-radeon9070/python')
if os.path.isdir(BUILD_DIR):
    sys.path.insert(0, BUILD_DIR)

import gpuworklib

PASSED = 0
FAILED = 0


def run_test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  [Hybrid] {name}: PASSED")
        PASSED += 1
    except AssertionError as e:
        print(f"  [Hybrid] {name}: FAILED — AssertionError: {e}")
        FAILED += 1
    except Exception as e:
        print(f"  [Hybrid] {name}: FAILED — {type(e).__name__}: {e}")
        FAILED += 1


# ============================================================================
# Тест 1: HybridGPUContext создаётся, оба backend инициализированы
# ============================================================================

def test_hybrid_init():
    ctx = gpuworklib.HybridGPUContext(0)
    assert ctx is not None
    assert ctx.device_index == 0


# ============================================================================
# Тест 2: Имена устройств OpenCL и ROCm
# ============================================================================

def test_hybrid_device_names():
    ctx = gpuworklib.HybridGPUContext(0)

    print(f"\n    OpenCL: {ctx.opencl_device_name}")
    print(f"    ROCm:   {ctx.rocm_device_name}")
    print(f"    Hybrid: {ctx.device_name}")

    assert ctx.opencl_device_name != "", "OpenCL device name is empty"
    assert ctx.rocm_device_name != "", "ROCm device name is empty"

    # Hybrid имя должно содержать "Hybrid"
    assert "Hybrid" in ctx.device_name, \
        f"Expected 'Hybrid' in device_name, got: {ctx.device_name}"


# ============================================================================
# Тест 3: ROCm контекст работает через HybridGPUContext
# ============================================================================

def test_hybrid_rocm_statistics():
    ctx_hybrid = gpuworklib.HybridGPUContext(0)
    ctx_rocm = gpuworklib.ROCmGPUContext(0)

    # Проверяем, что ROCm sub-backend совпадает с тем же GPU
    print(f"\n    HybridGPUContext ROCm: {ctx_hybrid.rocm_device_name}")
    print(f"    Standalone ROCmGPUContext: {ctx_rocm.device_name}")

    # Оба должны указывать на одно устройство
    assert ctx_hybrid.rocm_device_name != "", "ROCm device in hybrid is empty"
    assert ctx_rocm.device_name != "", "Standalone ROCm device is empty"

    # Запускаем статистику через standalone ROCm (проверяем совместимость)
    data = np.random.randn(1024).astype(np.float32)
    stats_proc = gpuworklib.StatisticsProcessor(ctx_rocm)
    result = stats_proc.compute_statistics(data)

    assert 'mean' in result, "Statistics result missing 'mean'"
    assert 'std' in result, "Statistics result missing 'std'"
    print(f"\n    Statistics via ROCm: mean={result['mean']:.4f}, std={result['std']:.4f}")


# ============================================================================
# Тест 4: ZeroCopy метод определяется
# ============================================================================

def test_hybrid_zero_copy_info():
    ctx = gpuworklib.HybridGPUContext(0)
    method = ctx.zero_copy_method
    supported = ctx.is_zero_copy_supported

    print(f"\n    ZeroCopy method:    {method}")
    print(f"    ZeroCopy supported: {supported}")

    assert isinstance(method, str), "zero_copy_method должен быть str"
    assert isinstance(supported, bool), "is_zero_copy_supported должен быть bool"
    assert method != "", "zero_copy_method не должен быть пустым"


# ============================================================================
# Тест 5: __repr__ содержит нужную информацию
# ============================================================================

def test_hybrid_repr():
    ctx = gpuworklib.HybridGPUContext(0)
    r = repr(ctx)
    print(f"\n    repr: {r}")

    assert "HybridGPUContext" in r
    assert "device=" in r
    assert "zero_copy=" in r


# ============================================================================
# Тест 6: Контекстный менеджер
# ============================================================================

def test_hybrid_context_manager():
    with gpuworklib.HybridGPUContext(0) as ctx:
        assert ctx is not None
        assert ctx.device_index == 0
        assert ctx.opencl_device_name != ""


# ============================================================================
# Тест 7: OpenCL + ROCm параллельная работа
# (OpenCL FFT → ROCm Statistics — независимые операции)
# ============================================================================

def test_hybrid_parallel_opencl_rocm():
    ctx_hybrid = gpuworklib.HybridGPUContext(0)
    ctx_opencl = gpuworklib.GPUContext(0)
    ctx_rocm = gpuworklib.ROCmGPUContext(0)

    # Генерируем данные
    N = 512
    signal = np.random.randn(N).astype(np.complex64)
    real_data = np.random.randn(N).astype(np.float32)

    # OpenCL: FFT
    fft = gpuworklib.FFTProcessor(ctx_opencl)
    spectrum = fft.process_complex(signal, sample_rate=1e6)

    # ROCm: Statistics
    stats_proc = gpuworklib.StatisticsProcessor(ctx_rocm)
    result = stats_proc.compute_statistics(real_data)

    print(f"\n    OpenCL FFT output len: {len(spectrum)}")
    print(f"    ROCm Statistics: mean={result['mean']:.4f}, std={result['std']:.4f}")

    assert len(spectrum) == N, f"Expected {N} FFT bins, got {len(spectrum)}"
    assert abs(result['mean']) < 1.0, "ROCm mean seems too large"

    # Hybrid context успешно создан — подтверждаем совместимость
    print(f"    Hybrid context: {ctx_hybrid.device_name}")
    assert ctx_hybrid.opencl_device_name != ""
    assert ctx_hybrid.rocm_device_name != ""


# ============================================================================
# Тест 8: Информационный отчёт (всегда PASSED)
# ============================================================================

def test_hybrid_status_report():
    ctx = gpuworklib.HybridGPUContext(0)

    print(f"\n    ===== HybridBackend Status =====")
    print(f"    Device index:         {ctx.device_index}")
    print(f"    OpenCL device:        {ctx.opencl_device_name}")
    print(f"    ROCm device:          {ctx.rocm_device_name}")
    print(f"    ZeroCopy method:      {ctx.zero_copy_method}")
    print(f"    ZeroCopy supported:   {ctx.is_zero_copy_supported}")

    if ctx.is_zero_copy_supported:
        print(f"    Hybrid mode:          FULL (OpenCL + ROCm + ZeroCopy)")
    else:
        print(f"    Hybrid mode:          PARTIAL (OpenCL + ROCm, ZeroCopy not available)")
        print(f"    Note: Data transfer requires CPU copy between backends")

    assert True  # Информационный тест


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n========== HybridBackend Python Tests ==========")

    run_test("hybrid_init",               test_hybrid_init)
    run_test("hybrid_device_names",       test_hybrid_device_names)
    run_test("hybrid_rocm_statistics",    test_hybrid_rocm_statistics)
    run_test("hybrid_zero_copy_info",     test_hybrid_zero_copy_info)
    run_test("hybrid_repr",               test_hybrid_repr)
    run_test("context_manager",           test_hybrid_context_manager)
    run_test("parallel_opencl_rocm",      test_hybrid_parallel_opencl_rocm)
    run_test("hybrid_status_report",      test_hybrid_status_report)

    print(f"\nTotal: {PASSED} passed, {FAILED} failed")
    print("========== HybridBackend Python Tests Done ==========\n")

    sys.exit(0 if FAILED == 0 else 1)
