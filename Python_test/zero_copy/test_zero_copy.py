"""
test_zero_copy.py — Python тесты ZeroCopy Bridge (OpenCL → ROCm)

Запуск:
    python Python_test/zero_copy/test_zero_copy.py

Тесты проверяют доступность ZeroCopy и корректность определения метода.
Если устройство не поддерживает ZeroCopy — тесты gracefully пропускаются.

Требования:
    - Linux + AMD GPU + ROCm
    - Собранный gpuworklib.so с ENABLE_ROCM=ON
    - Запуск в группе render: sg render -c "python3 ..."

Автор: Кодо (AI Assistant)
Дата: 2026-02-24
"""

import sys
import os

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
        print(f"  [ZeroCopy] {name}: PASSED")
        PASSED += 1
    except AssertionError as e:
        print(f"  [ZeroCopy] {name}: FAILED — AssertionError: {e}")
        FAILED += 1
    except Exception as e:
        print(f"  [ZeroCopy] {name}: FAILED — {type(e).__name__}: {e}")
        FAILED += 1


# ============================================================================
# Тест 1: HybridGPUContext создаётся успешно
# ============================================================================

def test_hybrid_context_creates():
    ctx = gpuworklib.HybridGPUContext(0)
    assert ctx is not None
    assert ctx.device_index == 0


# ============================================================================
# Тест 2: Имена устройств не пустые
# ============================================================================

def test_device_names_not_empty():
    ctx = gpuworklib.HybridGPUContext(0)
    print(f"\n    OpenCL device: {ctx.opencl_device_name}")
    print(f"    ROCm device:   {ctx.rocm_device_name}")
    print(f"    Combined:      {ctx.device_name}")
    assert ctx.opencl_device_name != ""
    assert ctx.rocm_device_name != ""
    assert ctx.device_name != ""


# ============================================================================
# Тест 3: ZeroCopy метод определяется (не крэшится)
# ============================================================================

def test_zero_copy_method_detection():
    ctx = gpuworklib.HybridGPUContext(0)
    method = ctx.zero_copy_method
    print(f"\n    ZeroCopy method: {method}")

    # Метод должен быть одним из известных значений
    known_methods = [
        "AMD GPU VA (CL_MEM_AMD_GPU_VA)",
        "DMA-BUF (cl_khr_external_memory_dma_buf)",
        "SVM (Shared Virtual Memory)",
        "None (ZeroCopy not supported)",
    ]
    assert method in known_methods, f"Unknown ZeroCopy method: '{method}'"


# ============================================================================
# Тест 4: is_zero_copy_supported имеет тип bool
# ============================================================================

def test_zero_copy_supported_is_bool():
    ctx = gpuworklib.HybridGPUContext(0)
    supported = ctx.is_zero_copy_supported
    print(f"\n    is_zero_copy_supported: {supported}")
    assert isinstance(supported, bool)


# ============================================================================
# Тест 5: __repr__ корректно формируется
# ============================================================================

def test_hybrid_repr():
    ctx = gpuworklib.HybridGPUContext(0)
    r = repr(ctx)
    print(f"\n    repr: {r}")
    assert "HybridGPUContext" in r
    assert "device=" in r
    assert "zero_copy=" in r


# ============================================================================
# Тест 6: Контекстный менеджер (with statement)
# ============================================================================

def test_context_manager():
    with gpuworklib.HybridGPUContext(0) as ctx:
        assert ctx is not None
        assert ctx.device_index == 0


# ============================================================================
# Тест 7: ZeroCopy report (информационный тест — всегда PASSED)
# ============================================================================

def test_zero_copy_report():
    ctx = gpuworklib.HybridGPUContext(0)

    print(f"\n    === ZeroCopy Report ===")
    print(f"    Device:               {ctx.opencl_device_name}")
    print(f"    ZeroCopy method:      {ctx.zero_copy_method}")
    print(f"    ZeroCopy supported:   {ctx.is_zero_copy_supported}")

    if ctx.is_zero_copy_supported:
        print(f"    Status: ZeroCopy AVAILABLE ✓")
    else:
        print(f"    Status: ZeroCopy NOT available (dma-buf / AMD GPU VA not supported)")
        print(f"    Hint: Requires AMD GPU with cl_khr_external_memory_dma_buf or cl_amd_svm")

    # Тест всегда проходит — это информационный отчёт
    assert True


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n========== ZeroCopy Python Tests ==========")

    run_test("hybrid_context_creates",       test_hybrid_context_creates)
    run_test("device_names_not_empty",        test_device_names_not_empty)
    run_test("zero_copy_method_detection",    test_zero_copy_method_detection)
    run_test("zero_copy_supported_is_bool",   test_zero_copy_supported_is_bool)
    run_test("hybrid_repr",                   test_hybrid_repr)
    run_test("context_manager",               test_context_manager)
    run_test("zero_copy_report",              test_zero_copy_report)

    print(f"\nTotal: {PASSED} passed, {FAILED} failed")
    print("========== ZeroCopy Python Tests Done ==========\n")

    sys.exit(0 if FAILED == 0 else 1)
