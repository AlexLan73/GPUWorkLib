"""
test_strategies_pipeline.py — Тест pipeline strategies (5 вариантов сигнала)
=============================================================================

Запуск:
    python Python_test/strategies/test_strategies_pipeline.py

Или через TestRunner:
    from common.runner import TestRunner
    runner = TestRunner()
    test = TestStrategiesPipeline()
    test.setUp()
    results = runner.run(test)
    runner.print_summary(results)

Ожидаемый результат:
    [PASS] V1_clean, [PASS] V2_noise, [PASS] V3_phase, [PASS] V4_phase_noise,
    [SKIP] V5_file

GPU API (из py_strategies_rocm.hpp):
    proc = gpuworklib.AntennaProcessorTest(ctx, n_ant=N, n_samples=M,
               sample_rate=fs, signal_frequency_hz=f0, debug_mode=True)
"""

import sys
import os

# Добавить Python_test/ в path
_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

from common.runner import TestRunner, SkipTest
from common.result import TestResult
from common.gpu_context import GPUContextManager
from common.gpu_loader import GPULoader

from .numpy_reference import NumpyReference
from .signal_factory import SignalSourceFactory, SignalVariant, SignalConfig
from .pipeline_step_validator import PipelineStepValidator


# ── Дефолтные параметры сценария ─────────────────────────────────────────────

_DEFAULT_CFG = SignalConfig(
    n_ant     = 5,
    n_samples = 8000,
    fs        = 12e6,
    f0        = 2e6,
    tau_step  = 100e-6,
    snr_db    = 20.0,
    n_fft     = 8192,
)


class TestStrategiesPipeline:
    """Тест полного pipeline strategies — 5 вариантов сигнала.

    Каждый метод test_v* = один вариант сигнала.
    Запускается через TestRunner.

    Template Method (GoF): _run_pipeline() — общий скелет для всех тестов.
    Strategy (GoF): SignalSourceFactory выбирает источник сигнала.

    Объект proc создаётся ОДИН РАЗ в setUp() и переиспользуется во всех test_v*.
    """

    def __init__(self, cfg: SignalConfig = _DEFAULT_CFG):
        """
        Args:
            cfg: параметры сигнала (n_ant, n_samples, fs, f0, ...)
        """
        self._cfg  = cfg
        self._proc = None   # AntennaProcessorTest — создаётся один раз в setUp()

    def setUp(self) -> None:
        """Создать ROCm-контекст и AntennaProcessorTest.

        Вызывается ВРУЧНУЮ перед запуском тестов (не через TestRunner).
        Если ROCm недоступен → SkipTest (весь набор тестов пропускается).

        GPU API (py_strategies_rocm.hpp):
            gpuworklib.AntennaProcessorTest(ctx, n_ant, n_samples,
                sample_rate, signal_frequency_hz, debug_mode=True)
        """
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("ROCmGPUContext недоступен — ROCm не установлен или нет GPU")

        gw = GPULoader.get()
        if gw is None or not hasattr(gw, "AntennaProcessorTest"):
            raise SkipTest("gpuworklib.AntennaProcessorTest не найден")

        # AntennaProcessorTest создаётся ОДИН РАЗ — живёт весь набор тестов
        self._proc = gw.AntennaProcessorTest(
            ctx,
            n_ant               = self._cfg.n_ant,
            n_samples           = self._cfg.n_samples,
            sample_rate         = float(self._cfg.fs),
            signal_frequency_hz = float(self._cfg.f0),
            debug_mode          = True,
        )

    # ── Тесты ─────────────────────────────────────────────────────────────────

    def test_v1_cw_clean(self) -> TestResult:
        """V1: CW без шума, W = Identity.

        Самый простой вариант: GEMM тривиален (X = I @ S = S).
        Включает перекрёстную проверку CHECK-3b (stats после GEMM = stats входа).
        """
        return self._run_pipeline(
            variant       = SignalVariant.V1_CW_CLEAN,
            is_identity_w = True,
            variant_name  = "V1_clean",
        )

    def test_v2_cw_noise(self) -> TestResult:
        """V2: CW + AWGN (SNR=20 дБ), W = Identity.

        Проверяем что шум не разрушает pipeline.
        Пик должен быть виден несмотря на шум.
        """
        return self._run_pipeline(
            variant       = SignalVariant.V2_CW_NOISE,
            is_identity_w = True,
            variant_name  = "V2_noise",
        )

    def test_v3_cw_phase_delay(self) -> TestResult:
        """V3: CW с фазовой задержкой, W = delay_and_sum, без шума.

        Нетривиальный GEMM: X ≠ S. Формирование луча (beamforming).
        """
        return self._run_pipeline(
            variant       = SignalVariant.V3_CW_PHASE_DELAY,
            is_identity_w = False,
            variant_name  = "V3_phase",
        )

    def test_v4_cw_phase_noise(self) -> TestResult:
        """V4: CW с задержкой + AWGN, W = delay_and_sum.

        Полный реальный сценарий: задержки + шум.
        """
        return self._run_pipeline(
            variant       = SignalVariant.V4_CW_PHASE_NOISE,
            is_identity_w = False,
            variant_name  = "V4_phase_noise",
        )

    def test_v5_from_file(self) -> TestResult:
        """V5: Загрузка из файла (заглушка → SKIP).

        Когда появятся реальные данные — задать SignalConfig.file_path.
        Сейчас FileSignalSource бросает SkipTest.
        """
        return self._run_pipeline(
            variant       = SignalVariant.V5_FROM_FILE,
            is_identity_w = True,
            variant_name  = "V5_file",
        )

    # ── Шаблонный метод ───────────────────────────────────────────────────────

    def _run_pipeline(self, variant: SignalVariant,
                      is_identity_w: bool,
                      variant_name: str) -> TestResult:
        """Template Method: общий скелет для всех вариантов теста.

        Шаги:
            1. Проверить GPU-контекст (SkipTest если недоступен)
            2. Ленивая инициализация proc (если setUp не вызван)
            3. Сгенерировать сигнал (SignalSourceFactory)
            4. Создать NumpyReference (CPU-эталон)
            5. PipelineStepValidator.run_all() → TestResult
        """
        # 1. Проверить GPU
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("ROCm контекст недоступен")

        # 2. Ленивая инициализация proc (если setUp() не был вызван)
        if self._proc is None:
            self.setUp()  # может бросить SkipTest

        # 3. Сгенерировать сигнал (V5 бросит SkipTest если нет файла)
        source = SignalSourceFactory.create(variant)
        data = source.generate(self._cfg)  # SignalData (d_S, d_W — CPU numpy arrays)

        # 4. Создать CPU-эталон
        ref = NumpyReference(
            S     = data.S_ref,
            W     = data.W_ref,
            fs    = self._cfg.fs,
            f0    = self._cfg.f0,
            n_fft = self._cfg.n_fft,
        )

        # 5. Валидация всех шагов pipeline
        psv = PipelineStepValidator(
            proc         = self._proc,
            ref          = ref,
            save_to_disk = False,
        )

        result = psv.run_all(
            d_S           = data.d_S,
            d_W           = data.d_W,
            is_identity_w = is_identity_w,
            variant_name  = variant_name,
        )
        result.test_name = f"TestStrategiesPipeline.{variant_name}"
        return result


# ── Точка входа (прямой запуск) ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("  strategies pipeline — 5 вариантов сигнала")
    print("=" * 60)
    print(f"  N_ANT={_DEFAULT_CFG.n_ant}  N_SAMPLES={_DEFAULT_CFG.n_samples}")
    print(f"  fs={_DEFAULT_CFG.fs/1e6:.0f} МГц  f0={_DEFAULT_CFG.f0/1e6:.0f} МГц")
    print(f"  n_fft={_DEFAULT_CFG.n_fft}")
    print()

    runner = TestRunner()
    test   = TestStrategiesPipeline()

    try:
        test.setUp()
        print(f"  GPU: AntennaProcessorTest создан ✓")
    except SkipTest as e:
        print(f"  GPU: НЕДОСТУПЕН — {e}")
        print("  Все тесты будут SKIP\n")

    results = runner.run(test)
    runner.print_summary(results)
