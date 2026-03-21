# TASK Python-06: strategies/test_strategies_pipeline.py — 5 тестов

**Статус**: ✅ DONE (2026-03-20)
**Приоритет**: 🟡 ВЫСОКИЙ
**Файл**: `Python_test/strategies/test_strategies_pipeline.py` (создать новый)
**Зависимости**: TASK_Python_01..05 (все предыдущие)
**Стиль**: ООП, SOLID, GRASP, GoF — обязательно!

---

## 🎯 Цель

Создать главный тест-класс `TestStrategiesPipeline` с 5 методами (V1–V5).
Каждый тест — один вариант сигнала, полный прогон pipeline через `PipelineStepValidator`.

Запускается через `TestRunner`, НЕ через pytest.

---

## 📖 Что прочитать перед реализацией

- `TASK_Python_01` — `TestRunner`, `SkipTest`
- `TASK_Python_02` — `DataValidator`
- `TASK_Python_03` — `NumpyReference`
- `TASK_Python_04` — `SignalSourceFactory`, `SignalVariant`, `SignalConfig`
- `TASK_Python_05` — `PipelineStepValidator`
- `Python_test/common/gpu_context.py` — `GPUContextManager`
- `modules/strategies/include/antenna_processor_test.hpp` — API

---

## ⚠️ Как создать AntennaProcessorTest в Python

Нужно проверить реальный API pybind11. Ищи в Python_test/strategies/:
```bash
grep -r "AntennaProcessorTest\|create_processor\|AntennaProcessor" Python_test/ --include="*.py"
```

Возможный вариант:
```python
import gpuworklib as gw
cfg = gw.AntennaProcessorConfig()
cfg.n_ant = 5
cfg.n_samples = 8000
cfg.sample_rate = 12e6
proc = gw.AntennaProcessorTest(rocm_ctx, cfg)
```

**Уточни реальное имя класса и его параметры прежде чем писать тест!**

---

## 🏗️ Реализация

```python
"""
test_strategies_pipeline.py — Тест pipeline strategies (5 вариантов сигнала)
=============================================================================

Запуск (без pytest!):
    from common.runner import TestRunner
    runner = TestRunner()
    results = runner.run(TestStrategiesPipeline())
    runner.print_summary(results)

Или из main:
    python -m Python_test.strategies.test_strategies_pipeline
"""

import sys
import os
import numpy as np

# Добавить корень Python_test в path
_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

from common.runner import TestRunner, SkipTest
from common.result import TestResult
from common.gpu_context import GPUContextManager

from .numpy_reference import NumpyReference
from .signal_factory import SignalSourceFactory, SignalVariant, SignalConfig
from .pipeline_step_validator import PipelineStepValidator


# ── Параметры сценария (дефолты) ─────────────────────────────────────────────
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

    Каждый метод test_vN_* = один вариант сигнала.
    Запускается через TestRunner (без pytest!).

    Template Method (GoF): _run_pipeline() — общий скелет для всех тестов.
    Strategy (GoF): SignalSourceFactory выбирает источник сигнала.
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

        Вызывается TestRunner перед test_* методами.
        Если ROCm недоступен → SkipTest.
        """
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("ROCmGPUContext недоступен — ROCm не установлен или нет GPU")

        gw = _get_gpuworklib()

        # Создать конфигурацию AntennaProcessor
        ap_cfg = gw.AntennaProcessorConfig()   # <-- проверить реальное имя!
        ap_cfg.n_ant        = self._cfg.n_ant
        ap_cfg.n_samples    = self._cfg.n_samples
        ap_cfg.sample_rate  = self._cfg.fs

        # AntennaProcessorTest создаётся ОДИН РАЗ — живёт весь тест
        self._proc = gw.AntennaProcessorTest(ctx, ap_cfg)  # <-- проверить API!

    # ── Тесты ────────────────────────────────────────────────────────────────

    def test_v1_cw_clean(self) -> TestResult:
        """V1: CW без шума, W = Identity.

        Самый простой вариант: GEMM тривиален (X = I @ S = S).
        Используется для базовой проверки pipeline.
        """
        return self._run_pipeline(
            variant=SignalVariant.V1_CW_CLEAN,
            is_identity_w=True,
            variant_name="V1_clean",
        )

    def test_v2_cw_noise(self) -> TestResult:
        """V2: CW + AWGN (SNR=20дБ), W = Identity.

        Проверяем что шум не разрушает pipeline.
        """
        return self._run_pipeline(
            variant=SignalVariant.V2_CW_NOISE,
            is_identity_w=True,
            variant_name="V2_noise",
        )

    def test_v3_cw_phase_delay(self) -> TestResult:
        """V3: CW с фазовой задержкой, W = delay_and_sum, без шума.

        Нетривиальный GEMM: X ≠ S. Формирование луча.
        """
        return self._run_pipeline(
            variant=SignalVariant.V3_CW_PHASE_DELAY,
            is_identity_w=False,
            variant_name="V3_phase",
        )

    def test_v4_cw_phase_noise(self) -> TestResult:
        """V4: CW с задержкой + AWGN, W = delay_and_sum.

        Полный реальный сценарий: задержки + шум.
        """
        return self._run_pipeline(
            variant=SignalVariant.V4_CW_PHASE_NOISE,
            is_identity_w=False,
            variant_name="V4_phase_noise",
        )

    def test_v5_from_file(self) -> TestResult:
        """V5: Загрузка из файла (заглушка).

        Когда появятся реальные тестовые данные — раскомментировать:
        cfg_v5 = SignalConfig(**vars(self._cfg), file_path="/path/to/data.npz")

        Сейчас бросает SkipTest.
        """
        return self._run_pipeline(
            variant=SignalVariant.V5_FROM_FILE,
            is_identity_w=True,
            variant_name="V5_file",
        )

    # ── Шаблонный метод ───────────────────────────────────────────────────────

    def _run_pipeline(self, variant: SignalVariant,
                      is_identity_w: bool,
                      variant_name: str) -> TestResult:
        """Template Method: общий скелет для всех вариантов теста.

        Шаги:
            1. Получить ROCm-контекст
            2. Сгенерировать сигнал (SignalSourceFactory)
            3. Создать NumpyReference
            4. Создать PipelineStepValidator
            5. run_all() → собрать результаты
        """
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("ROCm контекст недоступен")

        # 1. Сгенерировать сигнал (может бросить SkipTest для V5)
        source = SignalSourceFactory.create(variant)
        data = source.generate(ctx, self._cfg)  # SignalData

        # 2. Создать CPU-эталон
        ref = NumpyReference(
            S     = data.S_ref,
            W     = data.W_ref,
            fs    = self._cfg.fs,
            f0    = self._cfg.f0,
            n_fft = self._cfg.n_fft,
        )

        # 3. Создать валидатор (proc создан в setUp)
        psv = PipelineStepValidator(
            proc         = self._proc,
            ref          = ref,
            save_to_disk = False,        # True → для отладки
        )

        # 4. Запустить все шаги
        result = psv.run_all(
            d_S          = data.d_S,
            d_W          = data.d_W,
            is_identity_w= is_identity_w,
            variant_name = variant_name,
        )
        result.test_name = f"TestStrategiesPipeline.{variant_name}"
        return result


# ── Вспомогательная функция ───────────────────────────────────────────────────

def _get_gpuworklib():
    """Получить модуль gpuworklib. SkipTest если недоступен."""
    from common.gpu_loader import GPULoader
    gw = GPULoader.get()
    if gw is None:
        raise SkipTest("gpuworklib.so не найден")
    return gw


# ── Точка входа (прямой запуск) ───────────────────────────────────────────────

if __name__ == "__main__":
    runner = TestRunner()
    test   = TestStrategiesPipeline()
    test.setUp()                          # создать proc один раз
    results = runner.run(test)            # запустить test_v1..test_v5
    runner.print_summary(results)
```

---

## ✅ Критерии готовности

1. Файл `test_strategies_pipeline.py` создан, импорты работают
2. `TestStrategiesPipeline` — **не** наследует `TestBase` (у него другой скелет)
3. `setUp()` — создаёт `AntennaProcessorTest` один раз, бросает `SkipTest` если нет ROCm
4. Методы `test_v1_*` ... `test_v5_*` — каждый вызывает `_run_pipeline()`
5. `test_v5_from_file()` — корректно пробрасывает `SkipTest` из `FileSignalSource`
6. `_run_pipeline()` — создаёт `NumpyReference` и `PipelineStepValidator`
7. После каждого `test_v4` и `test_v3` — в `Results/Plots/strategies/` есть .png
8. Прямой запуск `python test_strategies_pipeline.py` работает
9. `TestRunner.run(test)` возвращает 5 `TestResult` (PASS / FAIL / SKIP)

---

## ❌ Что НЕ делать

- НЕ создавать `proc` в каждом `test_vN_*` — только в `setUp()`!
- НЕ использовать `from common.runner import SkipTest` нигде
- НЕ хардкодить пути к gpuworklib.so
- НЕ делать FAIL если V5 → SkipTest — это ожидаемое поведение
- НЕ добавлять `` / `@pytest.mark`
