"""
Smoke-тест GeneratorFactory -- все зарегистрированные генераторы создаются.
Требует GPU (gpuworklib).

Запуск: python Python_test/Core/generators/test_generators_smoke.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from common import GPUContextManager, SkipTest
from common.configs import SignalConfig
from common.runner import TestRunner
from common.result import TestResult, ValidationResult
from Core.generators import GeneratorFactory


class TestGeneratorsSmoke:
    """Smoke-тест: все зарегистрированные генераторы создаются и генерируют."""

    def test_factory_available(self):
        result = TestResult(test_name="factory_available")
        avail = GeneratorFactory.available()
        expected = ["cw", "lfm", "noise"]
        ok = avail == expected
        result.add(ValidationResult(
            passed=ok,
            metric_name="available_types",
            actual_value=1.0 if ok else 0.0,
            threshold=1.0,
            message=f"available={avail}, expected={expected}"
        ))
        return result

    def test_factory_unknown_type(self):
        result = TestResult(test_name="factory_unknown_type")
        try:
            GeneratorFactory.create("unknown_type_xyz", None, SignalConfig())
            ok = False
            msg = "ValueError not raised"
        except ValueError as e:
            ok = True
            msg = str(e)
        except Exception as e:
            ok = False
            msg = f"wrong exception: {type(e).__name__}: {e}"
        result.add(ValidationResult(
            passed=ok,
            metric_name="unknown_type_error",
            actual_value=1.0 if ok else 0.0,
            threshold=1.0,
            message=msg
        ))
        return result

    def test_all_generators_gpu(self):
        try:
            ctx = GPUContextManager.get_rocm()
        except Exception as e:
            raise SkipTest(f"GPU недоступен: {e}")
        if ctx is None:
            raise SkipTest("ROCm контекст не создан")

        result = TestResult(test_name="all_generators_gpu")
        params = SignalConfig(fs=12e6, f0_hz=2e6, fdev_hz=1e6, n_samples=4096)

        for gen_type in GeneratorFactory.available():
            gen = GeneratorFactory.create(gen_type, ctx, params)
            sig = gen.generate(params.n_samples)
            ok_dtype = sig.dtype == np.complex64
            ok_shape = sig.shape == (params.n_samples,)
            ok_type = gen.generator_type == gen_type
            ok_fs = gen.sample_rate == params.fs
            all_ok = ok_dtype and ok_shape and ok_type and ok_fs
            result.add(ValidationResult(
                passed=all_ok,
                metric_name=f"{gen_type}_generate",
                actual_value=1.0 if all_ok else 0.0,
                threshold=1.0,
                message=(f"dtype={sig.dtype}, shape={sig.shape}, "
                         f"type={gen.generator_type}, fs={gen.sample_rate}, "
                         f"max_abs={np.abs(sig).max():.4f}")
            ))

        return result

    def test_set_params(self):
        try:
            ctx = GPUContextManager.get_rocm()
        except Exception as e:
            raise SkipTest(f"GPU недоступен: {e}")
        if ctx is None:
            raise SkipTest("ROCm контекст не создан")

        result = TestResult(test_name="set_params")
        params = SignalConfig(fs=12e6, f0_hz=2e6, n_samples=4096)

        gen = GeneratorFactory.create("cw", ctx, params)
        gen.set_params(f0=3e6, amplitude=0.5)
        sig = gen.generate(4096)

        ok = sig.dtype == np.complex64 and sig.shape == (4096,)
        result.add(ValidationResult(
            passed=ok,
            metric_name="set_params_cw",
            actual_value=1.0 if ok else 0.0,
            threshold=1.0,
            message=f"after set_params: max_abs={np.abs(sig).max():.4f}"
        ))
        return result


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestGeneratorsSmoke())
    runner.print_summary(results)
