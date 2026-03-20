"""
pipeline_step_validator.py — Валидация pipeline strategies по шагам
====================================================================

Facade (GoF): скрывает детали step_N() вызовов и сравнения
Template Method (GoF): run_all() — фиксированный порядок шагов
Coordinator (GRASP): управляет proc, ref, DataValidator

Объект proc (AntennaProcessorTest) создаётся СНАРУЖИ и передаётся сюда.
PipelineStepValidator его НЕ создаёт и НЕ уничтожает.

Реальный pybind11 API (py_strategies_rocm.hpp):
    step_1_debug_input()    → dict {"stats": [{"beam_id":, "mean_real":,
                                "mean_imag":, "mean_magnitude":, "variance":,
                                "std_dev":},...], "beam_count": N}
    step_2_gemm()           → np.ndarray [n_ant, n_samples] complex64
    step_3_debug_post_gemm()→ dict (same as step_1)
    step_4_window_fft()     → np.ndarray [n_ant, nFFT] complex64
    step_5_debug_post_fft() → dict (same as step_1)
    step_6_1_one_max_parabola() → list of dicts
                              [{"beam_id":, "bin_index":, "magnitude":,
                                "freq_offset":, "refined_freq_hz":}]
    step_6_2_all_maxima()   → list of dicts
                              [{"antenna_id":, "num_maxima":, "maxima": [...]}]
    step_6_3_global_minmax()→ list of dicts
                              [{"beam_id":, "min_magnitude":, "min_bin":,
                                "min_frequency_hz":, "max_magnitude":,
                                "max_bin":, "max_frequency_hz":,
                                "dynamic_range_dB":}]
"""

import os
import sys
import numpy as np
from typing import Optional

_sys_path_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _sys_path_root not in sys.path:
    sys.path.insert(0, _sys_path_root)

from common.result import TestResult, ValidationResult
from common.validators import DataValidator
from .numpy_reference import NumpyReference


class PipelineStepValidator:
    """Валидация pipeline AntennaProcessorTest по шагам.

    Вызывает step_N() в порядке 0→1→2→3→4→5→6.1→6.2→6.3.
    Каждый шаг: вызов GPU → сравнение с NumpyReference → TestResult.

    Usage:
        proc = gpuworklib.AntennaProcessorTest(ctx, n_ant=5, n_samples=8000,
                   sample_rate=12e6, signal_frequency_hz=2e6)
        ref  = NumpyReference(S_ref, W_ref, fs=12e6, f0=2e6, n_fft=8192)
        psv  = PipelineStepValidator(proc, ref)
        result = psv.run_all(d_S=S_flat, d_W=W_flat,
                             is_identity_w=True, variant_name="V1_clean")
        print(result.summary())
    """

    def __init__(self,
                 proc,
                 ref: NumpyReference,
                 save_to_disk: bool = False,
                 output_dir: Optional[str] = None):
        """
        Args:
            proc:        AntennaProcessorTest pybind11-объект (создан снаружи)
            ref:         NumpyReference — CPU-эталон
            save_to_disk: True → сохранять промежуточные данные в .npz
            output_dir:  куда сохранять (если None → "Results/debug/strategies")
        """
        self._proc = proc
        self._ref  = ref
        self._save = save_to_disk
        self._out  = output_dir or os.path.join("Results", "debug", "strategies")

    # ── Публичный API ────────────────────────────────────────────────────────

    def run_step_0(self, d_S: np.ndarray, d_W: np.ndarray) -> TestResult:
        """STEP 0: подготовить входные данные.

        Вызывает proc.step_0_prepare_input(d_S, d_W).
        pybind11 делает hipMemcpy внутри — d_S и d_W это CPU-массивы.

        CHECK-0: proc не упал (passed = True)
        """
        tr = TestResult(test_name="step_0_prepare_input")
        self._proc.step_0_prepare_input(d_S, d_W)
        tr.add(ValidationResult(
            passed=True,
            metric_name="prepare_input_ok",
            actual_value=0.0,
            threshold=0.0,
            message="d_S and d_W uploaded to GPU"
        ))
        return tr

    def run_step_1(self) -> TestResult:
        """STEP 1: статистика входного сигнала d_S.

        CHECK-1a: mean (complex64)          tol=0.01 max_rel
        CHECK-1b: variance (float32)        tol=0.01 max_rel
        CHECK-1c: std_dev (float32)         tol=0.01 max_rel
        CHECK-1d: mean_magnitude (float32)  tol=0.01 max_rel
        """
        tr = TestResult(test_name="step_1_debug_input")
        gpu_dict = self._proc.step_1_debug_input()
        stats = gpu_dict["stats"]  # list of dicts

        v = DataValidator(tolerance=0.01, metric="max_rel")

        gpu_mean  = np.array([s["mean_real"] + 1j * s["mean_imag"]
                               for s in stats], dtype=np.complex64)
        gpu_var   = np.array([s["variance"]       for s in stats], dtype=np.float32)
        gpu_std   = np.array([s["std_dev"]         for s in stats], dtype=np.float32)
        gpu_mmag  = np.array([s["mean_magnitude"]  for s in stats], dtype=np.float32)

        ref_mean  = np.array([r.mean           for r in self._ref.input_stats],
                              dtype=np.complex64)
        ref_var   = np.array([r.variance       for r in self._ref.input_stats],
                              dtype=np.float32)
        ref_std   = np.array([r.std_dev        for r in self._ref.input_stats],
                              dtype=np.float32)
        ref_mmag  = np.array([r.mean_magnitude for r in self._ref.input_stats],
                              dtype=np.float32)

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
        d_X = self._proc.step_2_gemm()  # np.ndarray [n_ant, n_samples] complex64

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
        gpu_dict = self._proc.step_3_debug_post_gemm()
        stats = gpu_dict["stats"]

        v = DataValidator(tolerance=0.01, metric="max_rel")

        gpu_mean = np.array([s["mean_real"] + 1j * s["mean_imag"]
                              for s in stats], dtype=np.complex64)
        gpu_var  = np.array([s["variance"]      for s in stats], dtype=np.float32)
        gpu_std  = np.array([s["std_dev"]        for s in stats], dtype=np.float32)
        gpu_mmag = np.array([s["mean_magnitude"] for s in stats], dtype=np.float32)

        ref_mean = np.array([r.mean           for r in self._ref.gemm_stats],
                             dtype=np.complex64)
        ref_var  = np.array([r.variance       for r in self._ref.gemm_stats],
                             dtype=np.float32)
        ref_std  = np.array([r.std_dev        for r in self._ref.gemm_stats],
                             dtype=np.float32)
        ref_mmag = np.array([r.mean_magnitude for r in self._ref.gemm_stats],
                             dtype=np.float32)

        tr.add(v.validate(gpu_mean, ref_mean, name="CHECK-3a_mean"))
        tr.add(v.validate(gpu_var,  ref_var,  name="CHECK-3a_variance"))
        tr.add(v.validate(gpu_std,  ref_std,  name="CHECK-3a_std_dev"))
        tr.add(v.validate(gpu_mmag, ref_mmag, name="CHECK-3a_mean_magnitude"))

        if is_identity_w:
            # W=I: stats(d_X) должна совпасть со stats(d_S)
            ref_in_mean = np.array([r.mean for r in self._ref.input_stats],
                                    dtype=np.complex64)
            tr.add(v.validate(gpu_mean, ref_in_mean, name="CHECK-3b_cross_mean_W=I"))

        return tr

    def run_step_4(self, variant_name: str = "") -> TestResult:
        """STEP 4: Hamming + FFT → спектр.

        CHECK-4a: |spec_gpu| vs mag_ref    tol=0.01 max_rel
        CHECK-4b: peak_bin vs expected_bin  tol=2 (abs, в бинах)
        PLOT: рисует спектр → Results/Plots/strategies/spectrum_{variant}.png
        """
        tr = TestResult(test_name="step_4_window_fft")
        spec_gpu = self._proc.step_4_window_fft()  # [n_ant, nFFT] complex64
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
        gpu_dict = self._proc.step_5_debug_post_fft()
        stats = gpu_dict["stats"]

        v = DataValidator(tolerance=0.01, metric="max_rel")

        gpu_mmag = np.array([s["mean_magnitude"] for s in stats], dtype=np.float32)
        gpu_var  = np.array([s["variance"]       for s in stats], dtype=np.float32)

        ref_mmag = np.array([r.mean_magnitude for r in self._ref.fft_stats],
                             dtype=np.float32)
        ref_var  = np.array([r.variance       for r in self._ref.fft_stats],
                             dtype=np.float32)

        tr.add(v.validate(gpu_mmag, ref_mmag, name="CHECK-5_mean_magnitude"))
        tr.add(v.validate(gpu_var,  ref_var,  name="CHECK-5_variance"))
        return tr

    def run_step_6_1(self) -> TestResult:
        """STEP 6.1: один максимум + параболическая интерполяция.

        CHECK-6.1: |refined_freq_hz - f0| < 50 кГц per beam
        """
        tr = TestResult(test_name="step_6_1_one_max_parabola")
        one_max = self._proc.step_6_1_one_max_parabola()  # list of dicts

        freq_gpu = np.array([m["refined_freq_hz"] for m in one_max], dtype=np.float32)
        freq_ref = np.full(len(one_max), self._ref.f0, dtype=np.float32)

        v = DataValidator(tolerance=50e3, metric="abs")
        tr.add(v.validate(freq_gpu, freq_ref, name="CHECK-6.1_refined_freq_hz"))
        return tr

    def run_step_6_2(self) -> TestResult:
        """STEP 6.2: все локальные максимумы.

        CHECK-6.2: количество пиков >= 1 в beam 0
        """
        tr = TestResult(test_name="step_6_2_all_maxima")
        all_maxima = self._proc.step_6_2_all_maxima()  # list of dicts

        # Проверить что хотя бы в beam 0 есть пик
        count = all_maxima[0]["num_maxima"] if all_maxima else 0
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
        CHECK-6.3c: |GPU dyn_range - NumPy dyn_range| < 1.0 dB per beam
        """
        tr = TestResult(test_name="step_6_3_global_minmax")
        minmax = self._proc.step_6_3_global_minmax()  # list of dicts

        # CHECK-6.3c: эталон dynamic_range per beam
        ref_dyn_range = self._ref.compute_dynamic_range_db()  # [n_ant] float32

        for mm in minmax:
            beam_id = mm["beam_id"]
            tr.add(ValidationResult(
                passed=mm["min_magnitude"] < mm["max_magnitude"],
                metric_name=f"CHECK-6.3a_min<max_beam{beam_id}",
                actual_value=float(mm["min_magnitude"]),
                threshold=float(mm["max_magnitude"]),
                message=(f"min={mm['min_magnitude']:.4f} "
                         f"max={mm['max_magnitude']:.4f}")
            ))
            tr.add(ValidationResult(
                passed=mm["dynamic_range_dB"] > 0,
                metric_name=f"CHECK-6.3b_dyn_range_beam{beam_id}",
                actual_value=float(mm["dynamic_range_dB"]),
                threshold=0.0,
                message=f"dynamic_range={mm['dynamic_range_dB']:.1f} dB"
            ))
            # CHECK-6.3c: GPU dynamic_range ≈ NumPy эталон (допуск 1.0 dB)
            diff_db = abs(float(mm["dynamic_range_dB"]) - float(ref_dyn_range[beam_id]))
            tr.add(ValidationResult(
                passed=diff_db < 1.0,
                metric_name=f"CHECK-6.3c_dyn_range_vs_ref_beam{beam_id}",
                actual_value=diff_db,
                threshold=1.0,
                message=(f"GPU={mm['dynamic_range_dB']:.2f} dB "
                         f"ref={ref_dyn_range[beam_id]:.2f} dB "
                         f"diff={diff_db:.3f} dB")
            ))
        return tr

    def run_all(self,
                d_S: Optional[np.ndarray] = None,
                d_W: Optional[np.ndarray] = None,
                is_identity_w: bool = False,
                variant_name: str = "") -> TestResult:
        """Прогнать все шаги последовательно.

        Template Method (GoF): фиксированный порядок шагов.

        Args:
            d_S:          CPU-массив complex64 flat [n_ant*n_samples]
                          (если None — step_0 пропускается, данные уже загружены)
            d_W:          CPU-массив complex64 flat [n_ant*n_ant]
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
            import matplotlib
            matplotlib.use("Agg")  # без GUI
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            n_fft = self._ref.n_fft
            fs = self._ref.fs
            freqs = np.fft.fftfreq(n_fft, d=1.0 / fs) / 1e6  # в МГц

            half = n_fft // 2

            ax_ref = axes[0]
            ax_ref.plot(freqs[:half], self._ref.mag_ref[0, :half])
            ax_ref.set_title(f"NumPy Reference  {variant_name}")
            ax_ref.set_xlabel("Frequency (MHz)")
            ax_ref.set_ylabel("|FFT|")
            ax_ref.axvline(self._ref.f0 / 1e6, color="red",
                           linestyle="--", label=f"f0={self._ref.f0/1e6:.1f} MHz")
            ax_ref.legend()

            ax_gpu = axes[1]
            ax_gpu.plot(freqs[:half], mag_gpu[0, :half])
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
