# Обзор модулей GPUWorkLib {#modules_overview_page}

## DrvGPU — ядро библиотеки

Единственная точка управления GPU. Все модули принимают `IBackend*`.

**Три backend'а**:

| Backend | Условие | Платформа |
|---------|---------|-----------|
| `OpenCLBackend` | всегда | AMD / NVIDIA / Intel |
| `ROCmBackend` | `ENABLE_ROCM=1` | AMD (Linux) |
| `HybridBackend` | `ENABLE_ROCM=1` | OpenCL + ROCm одновременно |

**External Context API** — встраивание в чужой контекст:
```cpp
// Чужой hipBLAS/hipFFT stream
hipStream_t s;  hipStreamCreate(&s);
auto gpu = DrvGPU::CreateFromExternalROCm(0, s);
// DrvGPU НЕ вызовет hipStreamDestroy — owns_resources_=false
```

*Подробно*: @ref drvgpu_main, @ref drvgpu_architecture

---

## fft_func — FFT и поиск максимумов спектра

Объединяет бывшие модули `fft_processor` и `fft_maxima`.
Пакетный FFT через hipFFT + поиск пиков (ONE_PEAK, ALL_MAXIMA).

*Подробно*: @ref fft_func_overview | *Формулы*: @ref fft_func_formulas | *Тесты*: @ref fft_func_tests

---

## statistics — GPU статистика

Per-beam Mean, Std, Variance (Welford one-pass) + Median (Radix Sort / Histogram).

*Подробно*: @ref statistics_overview | *Формулы*: @ref statistics_formulas | *Тесты*: @ref statistics_tests

---

## vector_algebra — Cholesky инверсия

Cholesky decomposition + inversion (rocsolver POTRF/POTRI) для HPD матриц.
Два режима symmetrize: Roundtrip, GpuKernel.

*Подробно*: @ref vector_algebra_overview | *Формулы*: @ref vector_algebra_formulas | *Тесты*: @ref vector_algebra_tests

---

## filters — GPU фильтры

5 типов: FIR, IIR (Biquad DFII-T), MovingAverage (SMA/EMA/DEMA/TEMA), Kalman, KAMA.

*Подробно*: @ref filters_overview | *Формулы*: @ref filters_formulas | *Тесты*: @ref filters_tests

---

## signal_generators — Генераторы сигналов

8 типов: CW, LFM, Noise, FormSignal, DelayedFormSignal, LfmAnalyticalDelay, FormScript.

*Подробно*: @ref signal_generators_overview | *Формулы*: @ref signal_generators_formulas | *Тесты*: @ref signal_generators_tests

---

## lch_farrow — Дробная задержка

Lagrange 48×5 интерполяция для sub-sample fractional delay.

*Подробно*: @ref lch_farrow_overview | *Формулы*: @ref lch_farrow_formulas | *Тесты*: @ref lch_farrow_tests

---

## heterodyne — LFM Dechirp

Stretch-processing ЛЧМ → beat frequency → range (м), SNR (дБ).

*Подробно*: @ref heterodyne_overview | *Формулы*: @ref heterodyne_formulas | *Тесты*: @ref heterodyne_tests

---

## fm_correlator — FM-корреляция

M-sequence LFSR + cyclic shifts + freq-domain correlation (hipFFT R2C/C2R).

*Подробно*: @ref fm_correlator_overview | *Формулы*: @ref fm_correlator_formulas | *Тесты*: @ref fm_correlator_tests

---

## strategies — Цифровое ДН

7-step pipeline: CGEMM beamforming → Hamming+FFT → post-FFT scenarios.

*Подробно*: @ref strategies_overview | *Формулы*: @ref strategies_formulas | *Тесты*: @ref strategies_tests

---

## capon — MVDR Beamformer

Адаптивное подавление помех: R=YY^H/N+μI → Cholesky → R^{-1} → relief / beamform.

*Подробно*: @ref capon_overview | *Формулы*: @ref capon_formulas | *Тесты*: @ref capon_tests

---

## range_angle — 3D Range-Angle

Dechirp → Range FFT → 2D Beam FFT → Peak Search. Status: Beta (stubs).

*Подробно*: @ref range_angle_overview | *Формулы*: @ref range_angle_formulas | *Тесты*: @ref range_angle_tests

---

## GpuBenchmarkBase — шаблон бенчмарка

Все бенчмарки наследуют `GpuBenchmarkBase`:
```cpp
class MyBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
protected:
    void RunIteration() override { /* один прогон */ }
};
```

Запуск: `Run(iterations=100)` → GPUProfiler → `ExportMarkdown()`.
