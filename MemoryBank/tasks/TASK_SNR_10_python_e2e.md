# TASK SNR_10: Python e2e тест `test_snr_estimator.py`

> **Дата**: 2026-04-09
> **Модуль**: `Python_test/statistics/`
> **Приоритет**: Medium
> **Статус**: BACKLOG
> **Зависимости**: **[SNR_07](TASK_SNR_07_python_bindings.md)** (Python bindings)
> **Ревьюер**: Кодо
>
> 📐 **План**: **Часть 4 → Python e2e тест** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

Написать Python e2e тест — проверяет полный pipeline:
```
signal_generators (GPU) → LFM сигнал с шумом →
heterodyne (GPU, дечирп) →
StatisticsProcessor.compute_snr_db (GPU) → результат

Параллельно (numpy reference):
cfar_estimator.py из PyPanelAntennas/SNR/ → SNR_fft

Сравнение: |GPU result - numpy reference| < 1 dB
```

---

## 📁 Файл (создать)

```
Python_test/statistics/test_snr_estimator.py
```

---

## 📝 Структура теста

**Запрещено:** `pytest`, `@pytest.*`, `pytest.skip`, `import pytest`.
**Обязательно:** `common/runner.py::TestRunner` + `SkipTest`.

```python
"""
e2e тест SNR-estimator: signal_generators → heterodyne → statistics.compute_snr_db
Референс: numpy cfar_estimator из PyPanelAntennas/SNR/
"""
import sys
import os
from pathlib import Path
import numpy as np

# Добавляем корень репо в path для импортов
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "PyPanelAntennas" / "SNR"))

# Локальная инфраструктура тестов (не pytest!)
from Python_test.common.runner import TestRunner, SkipTest

# GPUWorkLib bindings
from gpu_work_lib import (
    StatisticsProcessor, SnrEstimationConfig, BranchSelector, BranchType,
    SignalGenerator, HeterodyneDechirp,
    GPUContext
)

# Numpy reference из PyPanelAntennas/SNR/
from cfar_estimator import CfarEstimator  # type: ignore


class TestSnrEstimator:
    """e2e тесты SNR-estimator"""

    def __init__(self):
        self.ctx = GPUContext()
        self.stat_proc = StatisticsProcessor(self.ctx)

    # ==========================================================
    # test_01: 1 антенна, SNR_in = 20 dB, проверка с numpy
    # ==========================================================
    def test_01_single_antenna_high_snr(self):
        n_samp = 5000
        snr_in_db = 20.0
        amplitude = 10.0 ** (snr_in_db / 20.0)  # A = sqrt(SNR_linear)

        # Generate CW + noise (dechirped = CW)
        np.random.seed(42)
        t = np.arange(n_samp, dtype=np.float32)
        signal = (amplitude * np.exp(1j * 2 * np.pi * 0.15 * t)).astype(np.complex64)
        noise = (np.random.randn(n_samp) + 1j * np.random.randn(n_samp)).astype(np.complex64) / np.sqrt(2)
        data = (signal + noise).reshape(1, n_samp)  # (1, 5000)

        # GPU
        cfg = SnrEstimationConfig()
        cfg.target_n_fft = 0  # auto → 2048
        result = self.stat_proc.compute_snr_db(data, n_antennas=1, n_samples=n_samp, config=cfg)
        gpu_snr_db = result.snr_db_global

        # numpy reference
        ref = CfarEstimator(target_n_fft=2048, guard_bins=3, ref_bins=8)
        ref_snr_db = ref.estimate(data[0])  # numpy path

        diff = abs(gpu_snr_db - ref_snr_db)
        print(f"test_01: GPU={gpu_snr_db:.1f} dB, numpy={ref_snr_db:.1f} dB, diff={diff:.2f} dB")
        assert diff < 1.0, f"GPU vs numpy diff > 1 dB: {diff}"
        assert gpu_snr_db > 40.0, f"snr_db_global too low: {gpu_snr_db}"

    # ==========================================================
    # test_02: 50 антенн, SNR_in = 5 dB, проверка медианы
    # ==========================================================
    def test_02_multi_antenna_medium_snr(self):
        n_ant = 50
        n_samp = 5000
        snr_in_db = 5.0
        amp = 10.0 ** (snr_in_db / 20.0)

        np.random.seed(123)
        data = np.zeros((n_ant, n_samp), dtype=np.complex64)
        t = np.arange(n_samp, dtype=np.float32)
        for a in range(n_ant):
            freq = 0.1 + 0.15 * np.random.rand()
            signal = amp * np.exp(1j * 2 * np.pi * freq * t)
            noise = (np.random.randn(n_samp) + 1j * np.random.randn(n_samp)) / np.sqrt(2)
            data[a] = (signal + noise).astype(np.complex64)

        cfg = SnrEstimationConfig()
        result = self.stat_proc.compute_snr_db(data, n_ant, n_samp, cfg)

        print(f"test_02: GPU snr_db={result.snr_db_global:.1f}, used_ant={result.used_antennas}")
        # SNR_fft = 5 + 10*log10(1666) ≈ 5 + 32 = 37 dB (± CFAR bias)
        assert 30.0 < result.snr_db_global < 45.0
        assert result.used_antennas == n_ant  # step_antennas=0 → auto = 1 (n_ant <= 50)

    # ==========================================================
    # test_03: только шум — артефакт CFAR ≈ 8-10 dB, BranchSelector → Low
    # ==========================================================
    def test_03_noise_only_branch_low(self):
        n_ant = 50
        n_samp = 5000

        np.random.seed(7)
        data = (np.random.randn(n_ant, n_samp) +
                1j * np.random.randn(n_ant, n_samp)) / np.sqrt(2)
        data = data.astype(np.complex64)

        cfg = SnrEstimationConfig()
        cfg.thresholds.low_to_mid_db = 12.0  # выше артефакта
        cfg.thresholds.mid_to_high_db = 18.0

        result = self.stat_proc.compute_snr_db(data, n_ant, n_samp, cfg)
        print(f"test_03: noise-only snr_db={result.snr_db_global:.1f}")
        assert 5.0 < result.snr_db_global < 15.0  # CFAR артефакт

        # BranchSelector
        selector = BranchSelector()
        branch = selector.select(result.snr_db_global, cfg.thresholds)
        assert branch == BranchType.Low, f"Expected Low, got {branch}"


def main():
    runner = TestRunner()
    results = runner.run(TestSnrEstimator())
    runner.print_summary(results)


if __name__ == "__main__":
    main()
```

---

## ✅ Definition of Done

- [ ] Файл `Python_test/statistics/test_snr_estimator.py` создан
- [ ] 3 теста: single antenna, multi antenna, noise only
- [ ] **НЕТ** `pytest` — только `TestRunner` из `common/runner.py`
- [ ] Использует Python API из `gpu_work_lib` биндингов
- [ ] Сравнение с numpy reference (`cfar_estimator.py` из `PyPanelAntennas/SNR/`)
- [ ] `BranchSelector.select()` вызывается в test_03 → ожидается `Low`
- [ ] `if __name__ == "__main__"` — запуск через `python Python_test/statistics/test_snr_estimator.py`
- [ ] Код НЕ запускается сегодня (нужен собранный GPU модуль в понедельник)
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ **Нет `import pytest`**, **нет `@pytest.fixture`**, **нет `pytest.skip`**
- ✅ `TestRunner` + `TestSnrEstimator` class (обычный Python class, не pytest class)
- ✅ `dtype=np.complex64` — НЕ `complex128`
- ✅ `cfg.target_n_fft` — lowercase snake_case
- ✅ `cfg.search_full_spectrum` если используется (не `search_left_right`)
- ✅ `result.n_actual`, `result.used_bins`, `result.snr_db_global` — читаются
- ✅ `result.branch` **НЕ используется** (`branch` — это `BranchSelector`)
- ✅ Assert'ы — **диапазоны** (`5 < x < 15`), не точные равенства
- ✅ Tolerance при сравнении с numpy: `|gpu - numpy| < 1 dB`
- ✅ Путь к `cfar_estimator.py` через `sys.path.insert`

---

## 🚫 Запреты

- ❌ **ЗАПРЕЩЕНО pytest** любого вида (import, декоратор, pytest.skip, pytest.mark, etc.)
- ❌ НЕ использовать `result.branch` (его нет!)
- ❌ НЕ использовать устаревшие имена (`target_N_fft`, `search_left_right`)
- ❌ НЕ запускать сегодня — только написание кода

---

## 🔗 Связанные таски

- **Требует:** [SNR_07](TASK_SNR_07_python_bindings.md) (Python bindings), [SNR_00](TASK_SNR_00_python_model.md) (cfar_estimator.py из Python модели)
- **Идёт параллельно:** SNR_08, SNR_09

---

*Created 2026-04-09 | Кодо*
