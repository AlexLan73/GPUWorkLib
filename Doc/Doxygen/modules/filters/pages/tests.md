@page filters_tests Filters — Тесты и бенчмарки

@tableofcontents

@section filters_tests_cpp C++ тесты

Расположение: `modules/filters/tests/`

| Файл | Описание |
|------|----------|
| `test_filters_rocm.hpp` | FIR/IIR: 8ch × 4096pts, CW 100Hz+5kHz, GPU vs CPU |
| `test_moving_average_rocm.hpp` | SMA/EMA/MMA/DEMA/TEMA: ring buffer, window |
| `test_kalman_rocm.hpp` | 256 channels, convergence, Re/Im independence |
| `test_kaufman_rocm.hpp` | KAMA: ER bounds (0–1), smoothing constants |

@section filters_tests_python Python тесты

Расположение: `Python_test/filters/`

| Файл | Описание |
|------|----------|
| `test_fir_filter_rocm.py` | ROCm FIR, direct convolution |
| `test_iir_filter_rocm.py` | ROCm IIR biquad cascade |
| `test_iir_plot.py` | IIR frequency response plots |
| `test_moving_average_rocm.py` | SMA/EMA/MMA/DEMA/TEMA |
| `test_kalman_rocm.py` | Kalman filter validation |
| `test_kaufman_rocm.py` | KAMA with efficiency ratio |
| `test_ai_filter_pipeline.py` | LLM-based filter design (scipy → GPU) |
| `test_ai_fir_demo.py` | Natural language → FIR coefficients |

@section filters_tests_benchmarks Бенчмарки

| Класс | Метрики |
|-------|---------|
| `FirFilterROCmBenchmark` | Upload (H2D), Kernel |
| `IirFilterROCmBenchmark` | Upload (H2D), Kernel |

@section filters_tests_plots Графики

@subsection filters_plots_ma Moving Average — SMA / EMA / MMA / DEMA / TEMA
@image html filters/report_task20_moving_average.png "Сравнение 5 типов скользящих средних" width=800px

@subsection filters_plots_kalman Kalman Filter — convergence
@image html filters/report_task21_kalman_filter.png "Kalman: сходимость на зашумлённых данных" width=800px

@subsection filters_plots_kama KAMA — Kaufman Adaptive Moving Average
@image html filters/report_task22_kaufman_kama.png "KAMA: Efficiency Ratio и адаптивное сглаживание" width=800px

@subsection filters_plots_ai AI-designed filters
@image html filters/ai_fir_lowpass.png "AI FIR Lowpass (natural language → coefficients)" width=600px
@image html filters/ai_iir_highpass.png "AI IIR Highpass" width=600px

@section filters_tests_seealso См. также

- @ref filters_overview — Обзор модуля
- @ref filters_formulas — Математика фильтров
