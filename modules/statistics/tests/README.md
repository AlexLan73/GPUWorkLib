# Statistics Module — C++ Tests

## 📋 Обзор тестов

### Основные (существующие)
- `test_statistics_rocm.hpp` — 15 тестов mean/median/variance/std (ROCm)
- `test_statistics_float_rocm.hpp` — float API + ProcessMagnitude→Statistics
- `statistics_compute_all_benchmark.hpp` + runner — ComputeAll benchmark

### SNR-estimator (SNR_08, SNR_09 — добавлено 2026-04-09)

#### `snr_test_helpers.hpp`
Утилиты для SNR тестов (namespace `snr_test_helpers`):
- `MakeDechirpedCW(n, freq_norm, A)` — CW тон (после дечирпа LFM)
- `MakeNoise(n, noise_power, seed)` — AWGN через LCG + Box-Muller
- `AddNoise(signal, noise_power, seed)` — in-place добавление шума
- `CopyToGpu(data)` / `FreeGpu(ptr)` — hipMalloc буфер helpers
- `GetTestBackend()` — shared ROCm backend (singleton)

#### `test_snr_estimator_rocm.hpp`
7 тестов CA-CFAR SNR-estimator'а (namespace `test_snr_estimator_rocm`):

| # | Test | Что проверяет |
|---|------|---------------|
| 01 | `noise_only_artifact` | Артефакт CFAR на чистом шуме ≈ 8-15 dB, BranchSelector→Low |
| 02 | `basic_signal` | CW SNR_in=20 dB → snr_db_global > 38 dB |
| 03 | `negative_freq` | search_full_spectrum: full vs half — отрицательная частота |
| 04 | `scenario_a` | 2500 × 5000, auto params, used_antennas = 50 |
| 05 | `scenario_b` | 256 × 1.3M (2.66 GB GPU!), used_antennas = 43 |
| 06 | `scenario_b_noise` | 256 × 1.3M только шум — CFAR artifact стабилен |
| 06b | `scenario_c` | 9000 × 10000, used_antennas = 50 |

Все assert'ы используют **диапазоны**, а не точные значения — физика имеет разброс.

#### `snr_estimator_benchmark.hpp` + `test_snr_estimator_benchmark.hpp`
Benchmark (namespace `test_snr_estimator`, runner `test_snr_estimator_benchmark`):
- Наследник `GpuBenchmarkBase` (3 warmup + 10 measurements)
- Измеряет end-to-end `ComputeSnrDb` через hipEvent пары → `RecordROCmEvent` → GPUProfiler
- 6 сценариев: Py-Small, A, B (главный), C, B с nfft=1024/4096
- Export: `Results/Profiler/GPU_00_SNR_Estimator_ROCm/SNR_*.md/.json`

## 🚀 Запуск (в понедельник на Debian/AMD)

Вызовы закомментированы в `all_test.hpp` до проверки на реальном GPU:

```cpp
// test_snr_estimator_rocm::run_all();
// test_snr_estimator_benchmark::run_benchmark();
```

После запуска раскомментировать если всё ок.

## 📐 Зависимости (цепочка SNR_01..SNR_06)

```
statistics_types.hpp  (SNR_01: types)
   ↓
gather_decimated_kernel.hpp  (SNR_03)
peak_cfar_kernel.hpp         (SNR_05)
snr_estimator_op.hpp         (SNR_05: Layer 5 Op)
branch_selector.hpp          (SNR_05: hysteresis)
   ↓
statistics_processor.{hpp,cpp}  (SNR_06: ComputeSnrDb facade)
   ↓
py_statistics.hpp  (SNR_07: Python bindings)
```

FFT часть (SNR_02, SNR_02b, SNR_04):
- `modules/fft_func/include/types/window_type.hpp`
- `modules/fft_func/include/kernels/fft_processor_kernels_rocm.hpp` — `pad_data_windowed` + magnitude
- `modules/fft_func/include/operations/pad_data_op.hpp` — `window` param
- `modules/fft_func/include/operations/magnitude_op.hpp` — `squared` param
- `modules/fft_func/include/fft_processor_rocm.hpp` — `ProcessMagnitudesToGPU`

## 📊 Калибровка (Python Эксп.5, 2026-04-09)

- Window:            **Hann** (решает −27 dB bias от sinc sidelobes)
- CFAR estimator:    CA-CFAR (mean)
- guard_bins:        **5** (было 3 до калибровки)
- ref_bins:          **16** (было 8 до калибровки)
- low_to_mid_db:     **15.0**
- mid_to_high_db:    **30.0**
- P_correct:         **97.9%**

Источник: `PyPanelAntennas/SNR/results/exp5_thresholds.json`
