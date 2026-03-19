# TASK Python-05: strategies/pipeline_step_validator.py — PipelineStepValidator

**Статус**: 🔲 TODO
**Приоритет**: 🟡 ВЫСОКИЙ
**Файл**: `Python_test/strategies/pipeline_step_validator.py` (создать новый)
**Зависимости**: TASK_Python_02 (DataValidator), TASK_Python_03 (NumpyReference)
**Стиль**: ООП, SOLID, GRASP, GoF — обязательно!

---

## 🎯 Цель

Создать класс `PipelineStepValidator` — координатор валидации pipeline по шагам.

Вызывает `AntennaProcessorTest.step_N()` в нужном порядке, читает результаты,
сравнивает с `NumpyReference` через `DataValidator`, возвращает `TestResult`.

**Паттерны**:
- **Facade (GoF)**: скрывает детали вызова `step_N()` и сравнения
- **Template Method (GoF)**: `run_all()` — фиксированный порядок, шаги переопределяемы
- **Coordinator (GRASP)**: координирует `proc`, `ref`, `DataValidator`

---

## 📖 Что прочитать перед реализацией

- `modules/strategies/include/antenna_processor_test.hpp` — ВСЕ методы `step_N()`
  **ВНИМАТЕЛЬНО!** Каждый метод возвращает `AntennaResult` или `vector<complex<float>>`
- `modules/strategies/include/result_types.hpp` — структуры `AntennaResult`, `StatisticsResult`
- `Python_test/strategies/numpy_reference.py` — класс `NumpyReference` (TASK_Python_03)
- `Python_test/common/validators.py` — `DataValidator` (TASK_Python_02)
- `MemoryBank/specs/python_test_refactoring.md` — таблица CHECK-точек

---

## ⚠️ Ключевое понимание: порядок вызова шагов

```
proc.step_0_prepare_input(d_S, d_W)  → пишет d_S_, d_W_ в state
proc.step_1_debug_input()             → читает d_S_, возвращает AntennaResult
proc.step_2_gemm()                    → читает d_S_/d_W_, пишет d_X_, возвращает vector
proc.step_3_debug_post_gemm()         → читает d_X_, возвращает AntennaResult
proc.step_4_window_fft()              → читает d_X_, пишет d_spectrum_, возвращает vector
proc.step_5_debug_post_fft()          → читает d_magnitudes_, возвращает AntennaResult
proc.step_6_1_one_max_parabola()      → читает d_spectrum_/d_magnitudes_, возвращает AntennaResult
proc.step_6_2_all_maxima()            → аналогично
proc.step_6_3_global_minmax()         → аналогично
```

Объект `proc` создаётся ОДИН раз снаружи и передаётся в конструктор.

---

## ⚠️ Типы возвращаемых данных из pybind11

`AntennaResult` в Python — это объект с полями:
```python
result.pre_input_stats  # List[StatisticsResult]
result.post_gemm_stats  # List[StatisticsResult]
result.post_fft_stats   # List[StatisticsResult]
result.one_max          # List[OneMaxParabolaLite]
result.minmax           # List[MinMaxResult]
```

`StatisticsResult` в Python:
```python
sr.mean            # complex (complex64)
sr.variance        # float
sr.std_dev         # float
sr.mean_magnitude  # float
```

`step_2_gemm()` и `step_4_window_fft()` возвращают `List[complex]` (Python list).
Нужно преобразовать: `np.array(result, dtype=np.complex64).reshape(n_ant, n_samples)`.

---

## 🏗️ Реализация

```python
"""
pipeline_step_validator.py — Валидация pipeline strategies по шагам
====================================================================

Facade (GoF): скрывает детали step_N() вызовов и сравнения
Template Method (GoF): run_all() — фиксированный порядок шагов
Coordinator (GRASP): управляет proc, ref, DataValidator

Объект proc (AntennaProcessorTest) создаётся СНАРУЖИ и передаётся сюда.
PipelineStepValidator его НЕ создаёт и НЕ уничтожает.
"""

import numpy as np
import os
from typing import Optional

from common.result import TestResult, ValidationResult
from common.validators import DataValidator
from .numpy_reference import NumpyReference


class PipelineStepValidator:
    """Валидация pipeline AntennaProcessorTest по шагам.

    Вызывает step_N() в порядке 0→1→2→3→4→5→6.1→6.2→6.3.
    Каждый шаг: вызов GPU → сравнение с NumpyReference → TestResult.

    Usage:
        proc = ctx.create_antenna_processor_test(cfg)
        ref  = NumpyReference(S_ref, W_ref, fs=12e6, f0=2e6, n_fft=8192)
        psv  = PipelineStepValidator(proc, ref)
        psv.run_step_0(data.d_S, data.d_W)   # установить данные
        result = psv.run_all()                 # прогнать все шаги
        print(result.summary())
    """

    def __init__(self,
                 proc,                         # AntennaProcessorTest (pybind11 obj)
                 ref: NumpyReference,
                 save_to_disk: bool = False,
                 output_dir: Optional[str] = None):
        """
        Args:
            proc:        AntennaProcessorTest — создан снаружи, живёт весь тест
            ref:         NumpyReference — CPU-эталон
            save_to_disk: True → сохранять промежуточные данные в файлы
            output_dir:  куда сохранять (если None и save_to_disk=True → "Results/debug/")
        """
        self._proc = proc
        self._ref  = ref
        self._save = save_to_disk
        self._out  = output_dir or "Results/debug/strategies"

    # ── Публичный API ────────────────────────────────────────────────────────

    def run_step_0(self, d_S, d_W) -> TestResult:
        """STEP 0: подготовить входные данные.

        Вызывает proc.step_0_prepare_input(d_S, d_W).
        Проверяет что данные корректны (не None).
        """
        tr = TestResult(test_name="step_0_prepare_input")
        self._proc.step_0_prepare_input(d_S, d_W)
        # Простая проверка: proc не упал
        tr.add(ValidationResult(
            passed=True,
            metric_name="prepare_input_ok",
            actual_value=0.0,
            threshold=0.0,
            message="d_S and d_W registered"
        ))
        return tr

    def run_step_1(self) -> TestResult:
        """STEP 1: статистика входного сигнала d_S.

        CHECK-1a: mean (complex64)      tol=0.01 max_rel
        CHECK-1b: variance (float32)    tol=0.01 max_rel
        CHECK-1c: std_dev (float32)     tol=0.01 max_rel
        CHECK-1d: mean_magnitude(float32) tol=0.01 max_rel
        """
        tr = TestResult(test_name="step_1_debug_input")
        gpu_result = self._proc.step_1_debug_input()
        stats = gpu_result.pre_input_stats  # List[StatisticsResult]

        v = DataValidator(tolerance=0.01, metric="max_rel")
        n = len(stats)

        # Собрать GPU данные в массивы
        gpu_mean  = np.array([s.mean for s in stats], dtype=np.complex64)
        gpu_var   = np.array([s.variance for s in stats], dtype=np.float32)
        gpu_std   = np.array([s.std_dev for s in stats], dtype=np.float32)
        gpu_mmag  = np.array([s.mean_magnitude for s in stats], dtype=np.float32)

        # CPU reference
        ref_mean  = np.array([r.mean for r in self._ref.input_stats], dtype=np.complex64)
        ref_var   = np.array([r.variance for r in self._ref.input_stats], dtype=np.float32)
        ref_std   = np.array([r.std_dev for r in self._ref.input_stats], dtype=np.float32)
        ref_mmag  = np.array([r.mean_magnitude for r in self._ref.input_stats], dtype=np.float32)

        tr.add(v.validate(gpu_mean,  ref_mean,  name="CHECK-1a_mean"))
        tr.add(v.validate(gpu_var,   ref_var,   name="CHECK-1b_variance"))
        tr.add(v.validate(gpu_std,   ref_std,   name="CHECK-1c_std_dev"))
        tr.add(v.validate(gpu_mmag,  ref_mmag,  name="CHECK-1d_mean_magnitude"))

        if self._save:
            self._save_arrays("step1", mean=gpu_mean, var=gpu_var)
        return tr

    def run_step_2(self) -> TestResult:
        """STEP 2: GEMM X = W × S.

        CHECK-2: d_X vs X_ref = W_ref @ S_ref    tol=1e-3 max_rel
        """
        tr = TestResult(test_name="step_2_gemm")
        raw = self._proc.step_2_gemm()  # List[complex] (flat)
        d_X = np.array(raw, dtype=np.complex64).reshape(
            self._ref.n_ant, self._ref.n_samples)

        v = DataValidator(tolerance=1e-3, metric="max_rel")
        tr.add(v.validate(d_X, self._ref.X_ref, name="CHECK-2_gemm_output"))

        if self._save:
            self._save_arrays("step2", d_X=d_X)
        return tr

    def run_step_3(self, is_identity_w: bool = False) -> TestResult:
        """STEP 3: статистика после GEMM d_X.

        CHECK-3a: stats(d_X) vs stats(X_ref)       tol=0.01
        CHECK-3b: если W=I → stats(d_X) ≈ stats(d_S) (перекрёстный)
        """
        tr = TestResult(test_name="step_3_debug_post_gemm")
        gpu_result = self._proc.step_3_debug_post_gemm()
        stats = gpu_result.post_gemm_stats

        v = DataValidator(tolerance=0.01, metric="max_rel")

        gpu_mean = np.array([s.mean for s in stats], dtype=np.complex64)
        gpu_var  = np.array([s.variance for s in stats], dtype=np.float32)
        gpu_std  = np.array([s.std_dev for s in stats], dtype=np.float32)
        gpu_mmag = np.array([s.mean_magnitude for s in stats], dtype=np.float32)

        ref_mean = np.array([r.mean for r in self._ref.gemm_stats], dtype=np.complex64)
        ref_var  = np.array([r.variance for r in self._ref.gemm_stats], dtype=np.float32)
        ref_std  = np.array([r.std_dev for r in self._ref.gemm_stats], dtype=np.float32)
        ref_mmag = np.array([r.mean_magnitude for r in self._ref.gemm_stats], dtype=np.float32)

        tr.add(v.validate(gpu_mean, ref_mean, name="CHECK-3a_mean"))
        tr.add(v.validate(gpu_var,  ref_var,  name="CHECK-3a_variance"))
        tr.add(v.validate(gpu_std,  ref_std,  name="CHECK-3a_std_dev"))
        tr.add(v.validate(gpu_mmag, ref_mmag, name="CHECK-3a_mean_magnitude"))

        if is_identity_w:
            # W=I: stats(d_X) должна совпасть со stats(d_S)
            ref_in_mean = np.array([r.mean for r in self._ref.input_stats], dtype=np.complex64)
            tr.add(v.validate(gpu_mean, ref_in_mean, name="CHECK-3b_cross_mean_W=I"))

        return tr

    def run_step_4(self, variant_name: str = "") -> TestResult:
        """STEP 4: Hamming + FFT → спектр.

        CHECK-4a: |spec_gpu| vs mag_ref    tol=0.01 max_rel
        CHECK-4b: peak_bin vs expected_bin  tol=2 (abs, в бинах)
        PLOT: рисует спектр → Results/Plots/strategies/spectrum_{variant}.png
        """
        tr = TestResult(test_name="step_4_window_fft")
        raw = self._proc.step_4_window_fft()  # List[complex] (flat)
        spec_gpu = np.array(raw, dtype=np.complex64).reshape(
            self._ref.n_ant, self._ref.n_fft)
        mag_gpu = np.abs(spec_gpu).astype(np.float32)

        # CHECK-4a: амплитуды спектра
        v_rel = DataValidator(tolerance=0.01, metric="max_rel")
        tr.add(v_rel.validate(mag_gpu, self._ref.mag_ref, name="CHECK-4a_spectrum_mag"))

        # CHECK-4b: позиция пика (в бинах)
        peak_bin_gpu = int(np.argmax(mag_gpu[0]))  # beam 0
        v_abs = DataValidator(tolerance=2, metric="abs")
        tr.add(v_abs.validate(peak_bin_gpu, self._ref.expected_peak_bin,
                              name="CHECK-4b_peak_bin"))

        # PLOT спектр (beam 0)
        self._plot_spectrum(mag_gpu, variant_name)

        if self._save:
            self._save_arrays("step4", spec_gpu=spec_gpu)
        return tr

    def run_step_5(self) -> TestResult:
        """STEP 5: статистика |spectrum|.

        CHECK-5: stats(|spectrum|)_gpu vs stats(mag_ref)   tol=0.01
        """
        tr = TestResult(test_name="step_5_debug_post_fft")
        gpu_result = self._proc.step_5_debug_post_fft()
        stats = gpu_result.post_fft_stats

        v = DataValidator(tolerance=0.01, metric="max_rel")

        gpu_mmag = np.array([s.mean_magnitude for s in stats], dtype=np.float32)
        gpu_var  = np.array([s.variance for s in stats], dtype=np.float32)

        ref_mmag = np.array([r.mean_magnitude for r in self._ref.fft_stats], dtype=np.float32)
        ref_var  = np.array([r.variance for r in self._ref.fft_stats], dtype=np.float32)

        tr.add(v.validate(gpu_mmag, ref_mmag, name="CHECK-5_mean_magnitude"))
        tr.add(v.validate(gpu_var,  ref_var,  name="CHECK-5_variance"))
        return tr

    def run_step_6_1(self) -> TestResult:
        """STEP 6.1: один максимум + параболическая интерполяция.

        CHECK-6.1: |refined_freq_hz - f0| < 50 кГц per beam
        """
        tr = TestResult(test_name="step_6_1_one_max_parabola")
        gpu_result = self._proc.step_6_1_one_max_parabola()
        one_max = gpu_result.one_max  # List[OneMaxParabolaLite]

        freq_gpu = np.array([m.refined_freq_hz for m in one_max], dtype=np.float32)
        freq_ref = np.full(len(one_max), self._ref.f0, dtype=np.float32)

        v = DataValidator(tolerance=50e3, metric="abs")
        tr.add(v.validate(freq_gpu, freq_ref, name="CHECK-6.1_refined_freq_hz"))
        return tr

    def run_step_6_2(self) -> TestResult:
        """STEP 6.2: все локальные максимумы.

        CHECK-6.2: количество пиков >= 1
        """
        tr = TestResult(test_name="step_6_2_all_maxima")
        gpu_result = self._proc.step_6_2_all_maxima()
        all_maxima = gpu_result.all_maxima  # List[AllMaximaBeamResult]

        # Проверить что хотя бы в beam 0 есть пик
        count = len(all_maxima[0].peaks) if all_maxima else 0
        tr.add(ValidationResult(
            passed=count >= 1,
            metric_name="CHECK-6.2_peak_count",
            actual_value=float(count),
            threshold=1.0,
            message=f"beam_0 has {count} peaks"
        ))
        return tr

    def run_step_6_3(self) -> TestResult:
        """STEP 6.3: глобальный MIN/MAX + dynamic_range_dB.

        CHECK-6.3a: min_magnitude < max_magnitude per beam
        CHECK-6.3b: dynamic_range_dB > 0 per beam
        """
        tr = TestResult(test_name="step_6_3_global_minmax")
        gpu_result = self._proc.step_6_3_global_minmax()
        minmax = gpu_result.minmax  # List[MinMaxResult]

        for mm in minmax:
            tr.add(ValidationResult(
                passed=mm.min_magnitude < mm.max_magnitude,
                metric_name=f"CHECK-6.3a_min<max_beam{mm.beam_id}",
                actual_value=float(mm.min_magnitude),
                threshold=float(mm.max_magnitude),
                message=f"min={mm.min_magnitude:.4f} max={mm.max_magnitude:.4f}"
            ))
            tr.add(ValidationResult(
                passed=mm.dynamic_range_dB > 0,
                metric_name=f"CHECK-6.3b_dyn_range_beam{mm.beam_id}",
                actual_value=float(mm.dynamic_range_dB),
                threshold=0.0,
                message=f"dynamic_range={mm.dynamic_range_dB:.1f} dB"
            ))
        return tr

    def run_all(self, d_S=None, d_W=None,
                is_identity_w: bool = False,
                variant_name: str = "") -> TestResult:
        """Прогнать все шаги последовательно.

        Template Method (GoF): фиксированный порядок шагов.

        Args:
            d_S:          GPU-указатель на сигнал (если None — step_0 пропускается)
            d_W:          GPU-указатель на весовую матрицу
            is_identity_w: True если W = Identity (включает CHECK-3b)
            variant_name:  имя варианта для графика (например "V1_clean")

        Returns:
            TestResult с объединёнными результатами всех шагов
        """
        combined = TestResult(test_name=f"pipeline_all_steps_{variant_name}")
        steps = []

        if d_S is not None and d_W is not None:
            steps.append(self.run_step_0(d_S, d_W))

        steps.extend([
            self.run_step_1(),
            self.run_step_2(),
            self.run_step_3(is_identity_w=is_identity_w),
            self.run_step_4(variant_name=variant_name),
            self.run_step_5(),
            self.run_step_6_1(),
            self.run_step_6_2(),
            self.run_step_6_3(),
        ])

        # Объединить все ValidationResult в один TestResult
        for step_result in steps:
            for vr in step_result.validations:
                combined.add(vr)
            if step_result.error:
                combined.error = step_result.error
                break  # Остановиться при первой ошибке

        return combined

    # ── Приватные методы ─────────────────────────────────────────────────────

    def _plot_spectrum(self, mag_gpu: np.ndarray, variant_name: str) -> None:
        """Нарисовать спектр GPU vs NumPy Reference для beam 0.

        Сохраняет в Results/Plots/strategies/spectrum_{variant_name}.png
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")  # без GUI

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            n_fft = self._ref.n_fft
            fs = self._ref.fs
            freqs = np.fft.fftfreq(n_fft, d=1.0 / fs) / 1e6  # в МГц

            # Beam 0
            ax_ref = axes[0]
            ax_ref.plot(freqs[:n_fft//2], self._ref.mag_ref[0, :n_fft//2])
            ax_ref.set_title(f"NumPy Reference  {variant_name}")
            ax_ref.set_xlabel("Frequency (MHz)")
            ax_ref.set_ylabel("|FFT|")
            ax_ref.axvline(self._ref.f0 / 1e6, color="red",
                           linestyle="--", label=f"f0={self._ref.f0/1e6:.1f} MHz")
            ax_ref.legend()

            ax_gpu = axes[1]
            ax_gpu.plot(freqs[:n_fft//2], mag_gpu[0, :n_fft//2])
            ax_gpu.set_title(f"GPU Result  {variant_name}")
            ax_gpu.set_xlabel("Frequency (MHz)")
            ax_gpu.axvline(self._ref.f0 / 1e6, color="red",
                           linestyle="--", label=f"f0={self._ref.f0/1e6:.1f} MHz")
            ax_gpu.legend()

            plt.tight_layout()

            # Сохранить
            out_dir = os.path.join("Results", "Plots", "strategies")
            os.makedirs(out_dir, exist_ok=True)
            fname = f"spectrum_{variant_name}.png" if variant_name else "spectrum.png"
            out_path = os.path.join(out_dir, fname)
            plt.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"  [PLOT] {out_path}")

        except Exception as e:
            print(f"  [PLOT] предупреждение: не удалось нарисовать спектр: {e}")

    def _save_arrays(self, step_name: str, **arrays) -> None:
        """Сохранить массивы на диск (для отладки).

        Сохраняет .npz файл в self._out/
        """
        if not self._save:
            return
        os.makedirs(self._out, exist_ok=True)
        path = os.path.join(self._out, f"{step_name}.npz")
        np.savez(path, **arrays)
        print(f"  [SAVE] {path}")
```

---

## ✅ Критерии готовности

1. `run_step_0` — вызывает `proc.step_0_prepare_input`, возвращает `TestResult`
2. `run_step_1` — 4 CHECK (mean, variance, std_dev, mean_magnitude)
3. `run_step_2` — 1 CHECK (GEMM output vs X_ref), reshape `[n_ant, n_samples]`
4. `run_step_3` — 4 CHECK + перекрёстный CHECK-3b при `is_identity_w=True`
5. `run_step_4` — 2 CHECK + PLOT спектра в `Results/Plots/strategies/`
6. `run_step_5` — 2 CHECK (mean_magnitude, variance)
7. `run_step_6_1` — 1 CHECK (refined_freq_hz < 50кГц)
8. `run_step_6_2` — 1 CHECK (count >= 1)
9. `run_step_6_3` — 2 CHECK per beam (min<max, dyn_range>0)
10. `run_all()` — объединяет все TestResult в один
11. `save_to_disk=True` → сохраняет .npz файлы
12. PLOT: `matplotlib.use("Agg")` — без GUI, сохранить в .png

---

## ❌ Что НЕ делать

- НЕ создавать `proc` внутри `PipelineStepValidator` — он приходит снаружи
- НЕ вызывать `run_all()` без предшествующего `run_step_0()` — или передать d_S, d_W в run_all
- НЕ останавливаться при FAIL отдельного CHECK — продолжать, собирать все результаты
- НЕ падать если matplotlib не установлен — `try/except` вокруг plot
