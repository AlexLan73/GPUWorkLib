# Python_test — тесты и примеры GPUWorkLib

---

## Как запустить тест в PyCharm — пошагово

### Шаг 1: Открыть проект

Открой корневую папку `GPUWorkLib/` как проект PyCharm.
Не открывай `Python_test/` отдельно — нужен именно корень.

---

### Шаг 2: Настроить Python interpreter

`File → Settings → Project → Python Interpreter`

Выбери тот же Python, которым собирался C++ модуль.
Если не знаешь — запусти в терминале:
```bash
cmake -B build -DBUILD_PYTHON=ON && cmake --build build
```
и посмотри какой `python3` был найден (`Python3_EXECUTABLE`).

---

### Шаг 3: Настроить Run Configuration (один раз)

`Run → Edit Configurations → + → Python`

| Поле | Значение |
|------|----------|
| **Script path** | `Python_test/strategies/test_base_pipeline.py` |
| **Working directory** | `/home/alex/C++/GPUWorkLib` ← **ОБЯЗАТЕЛЬНО корень!** |
| **Environment variables** | *(пусто — для тестов без GPU)* |

> ⚠️ **Working directory = корень проекта** — это критично.
> Все тесты рассчитывают найти `build/`, `Results/`, `Python_test/` именно отсюда.

---

### Шаг 4: Запустить

Нажми **Run** (▶) или **Debug** (🐛).

PyCharm покажет вывод в консоль.

Альтернатива через терминал из корня проекта:
```bash
python Python_test/strategies/test_base_pipeline.py
```

---

## Какие тесты требуют GPU, а какие нет

| Тип | Что нужно | Примеры |
|-----|-----------|---------|
| **Чистый NumPy** | Только Python, GPU не нужен | `strategies/test_base_pipeline.py` |
| **OpenCL GPU** | `gpuworklib.so` + OpenCL-драйвер | `filters/test_fir_filter_rocm.py` |
| **ROCm GPU** | `gpuworklib.so` + AMD GPU + ROCm | `statistics/test_statistics_rocm.py`, `heterodyne/test_heterodyne_rocm.py` |

`gpuworklib.so` ищется автоматически через `GPULoader` — он сам перебирает пути:
`build/python/Release` → `build/python/Debug` → `build/debian-radeon9070/python` → `build/python` → авто-поиск.

Если модуль не найден → тест пропускается (`SkipTest`), не падает.

Можно явно указать путь через переменную окружения:
```
GPUWORKLIB_BUILD_DIR=build/debian-radeon9070/python
```

---

## Настройка для GPU-тестов в PyCharm

Если тест требует GPU, добавь в Run Configuration:

`Environment variables`:
```
GPUWORKLIB_BUILD_DIR=build/debian-radeon9070/python
```

Или не добавляй ничего — `GPULoader` найдёт `.so` сам, если сборка лежит в стандартном месте.

---

## Структура по модулям

| Модуль | Директория | Нужен GPU? | Тесты |
|--------|------------|------------|-------|
| **strategies** | `Python_test/strategies/` | NumPy тесты — без GPU; остальные — GPU опционален | `test_params.py` (конфиг), `test_base_pipeline.py`, `test_debug_steps.py`, `test_scenario_builder.py`, `test_farrow_pipeline.py`, `test_timing_analysis.py`, `test_strategies_step_by_step.py`, `test_strategies_pipeline.py` |
| **signal_generators** | `Python_test/signal_generators/` | Частично | `test_form_signal.py`, `test_form_signal_rocm.py`, `test_delayed_form_signal.py`, `test_lfm_analytical_delay.py`, `example_form_signal.py` |
| **filters** | `Python_test/filters/` | Частично | `test_filters_stage1.py`, `test_ai_fir_demo.py`, `test_fir_filter_rocm.py`, `test_iir_filter_rocm.py`, `test_iir_plot.py`, `test_kalman_rocm.py`, `test_kaufman_rocm.py`, `test_moving_average_rocm.py`, `test_ai_filter_pipeline.py`, `plot_report_filters.py` |
| **heterodyne** | `Python_test/heterodyne/` | Да (ROCm) | `test_heterodyne.py`, `test_heterodyne_rocm.py`, `test_heterodyne_step_by_step.py`, `test_heterodyne_comparison.py` |
| **statistics** | `Python_test/statistics/` | Да (ROCm) | `test_statistics_rocm.py`, `test_statistics_float_rocm.py` |
| **fft_func** | `Python_test/fft_func/` | Да (ROCm) | `test_spectrum_find_all_maxima_rocm.py`, `test_spectrum_maxima_finder_rocm.py`, `test_process_magnitude_rocm.py` |
| **vector_algebra** | `Python_test/vector_algebra/` | Да (ROCm) | `test_cholesky_inverter_rocm.py`, `test_matrix_csv_comparison.py` |
| **lch_farrow** | `Python_test/lch_farrow/` | Частично | `test_lch_farrow.py`, `test_lch_farrow_rocm.py` |
| **fm_correlator** | `Python_test/fm_correlator/` | Да (ROCm) | `test_fm_correlator.py`, `test_fm_correlator_rocm.py` |
| **capon** | `Python_test/capon/` | Да | `test_capon.py` |
| **range_angle** | `Python_test/range_angle/` | Да (ROCm) | `test_range_angle.py` |
| **integration** | `Python_test/integration/` | Частично | `test_gpuworklib.py`, `test_fft_integration.py`, `test_signal_gen_integration.py` |
| **zero_copy** | `Python_test/zero_copy/` | Да | `test_zero_copy.py` |
| **hybrid** | `Python_test/hybrid/` | Да | `test_hybrid_backend.py` |

---

## Описание тестов

### strategies/ — Pipeline Strategies

`test_base_pipeline.py` — **GPU не нужен**, чистый NumPy.
Pipeline: S → GEMM → Hamming + FFT → peak finding. 4 варианта сигнала: SIN, LFM_NO_DELAY, LFM_WITH_DELAY, LFM_FARROW.
Проверяет: peak_freq ≈ f0 (±2 бина), dynamic_range > 20 дБ.

| Файл | GPU? | Что тестирует |
|------|------|----------------|
| `test_params.py` | ❌ нет | ⚙️ Не тест! Dataclass AntennaTestParams + SignalVariant — конфигурация для всех тестов |
| `test_base_pipeline.py` | ❌ нет | Полный NumPy pipeline: GEMM + FFT + peak, 4 варианта сигнала. **Старт отсюда!** |
| `test_debug_steps.py` | ❌ нет | Каждый шаг по отдельности с числовыми критериями (GEMM gain, FFT bin, DR) |
| `test_scenario_builder.py` | ❌ нет | Физическая модель ULA: задержки, CW/LFM, шум, матрица W |
| `test_farrow_pipeline.py` | ❌ нет | Pipeline A (фаза) vs Pipeline B (Farrow задержка) — когда нужен Farrow? |
| `test_timing_analysis.py` | ❌ нет | Анализ timing JSON от C++ TimingPerStepTest — нужны файлы из `Results/strategies/` |
| `test_strategies_step_by_step.py` | ⚡ GPU опц. | Детальный GPU vs NumPy по каждому шагу pipeline |

Графики → `Results/Plots/strategies/`

---

### signal_generators/test_delayed_form_signal.py — DelayedFormSignalGenerator (Farrow 48×5)

| № | Функция | Что тестирует |
|---|---------|----------------|
| 1 | `test_integer_delay()` | Целая задержка (5 сэмплов): GPU vs NumPy; max_error < 1e-2 |
| 2 | `test_fractional_delay()` | Дробная задержка (2.7 сэмпла): GPU vs Lagrange; max_error < 1e-2 |
| 3 | `test_multichannel_delay()` | 8 антенн, задержки 0…10.5 мкс; допуск max_err < 1.0 |
| 4 | `test_zero_delay()` | Задержка 0 = FormSignalGenerator без шума; max_error < 1e-4 |
| 5 | `test_delay_with_noise()` | Задержка + шум: мощность шума, ratio 0.5–2.0 |

Графики → `Results/Plots/signal_generators/DelayedFormSignal/`:
`plot1_integer_delay.png`, `plot2_fractional_delay.png`, `plot3_multichannel_waterfall.png`, `plot4_delay_sweep.png`

---

### signal_generators/test_lfm_analytical_delay.py

Аналитическая задержка ЛЧМ по формуле S(t−τ) без Farrow/Lagrange.
Графики → `Results/Plots/signal_generators/LfmAnalyticalDelay/`:
`plot1_real_delay_overlay.png`, `plot2_fractional_delay_boundary.png`, `plot3_multiantenna_delays.png`

---

### heterodyne/ — LFM Dechirp (ROCm)

| Файл | Что тестирует |
|------|----------------|
| `test_heterodyne.py` | Базовые тесты HeterodyneDechirp |
| `test_heterodyne_rocm.py` | ROCm: дечирп, NCO, MixDown/MixUp на AMD GPU |
| `test_heterodyne_step_by_step.py` | Пошаговая проверка пайплайна |
| `test_heterodyne_comparison.py` | GPU vs NumPy-эталон (точность, корреляция) |

---

### statistics/ — StatisticsProcessor (ROCm)

| Файл | Что тестирует |
|------|----------------|
| `test_statistics_rocm.py` | Welford fused (mean/std/var), Radix sort, Extract medians — vs NumPy |
| `test_statistics_float_rocm.py` | Float крайние случаи (NaN, Inf, минимальные размеры) |

---

### fft_func/ — FFT функции (ROCm)

| Файл | Что тестирует |
|------|----------------|
| `test_spectrum_find_all_maxima_rocm.py` | FindAllMaxima в спектре на GPU |
| `test_spectrum_maxima_finder_rocm.py` | SpectrumMaximaFinder — batch поиск пиков |
| `test_process_magnitude_rocm.py` | ProcessMagnitude — |FFT| + нормировка |

---

### filters/

| Файл | Что тестирует |
|------|----------------|
| `test_filters_stage1.py` | FIR/IIR базовые тесты (scipy → GPU) |
| `test_fir_filter_rocm.py` | FIR фильтр на ROCm GPU |
| `test_iir_filter_rocm.py` | IIR фильтр на ROCm GPU |
| `test_kalman_rocm.py` | Калман-фильтр на GPU |
| `test_kaufman_rocm.py` | Адаптивный фильтр Кауфмана на GPU |
| `test_moving_average_rocm.py` | Скользящее среднее на GPU |
| `test_ai_filter_pipeline.py` | AI-pipeline: LLM parser → filter designer |
| `plot_report_filters.py` | Сводный отчёт-графики по всем фильтрам |

Графики → `Results/Plots/filters/`

---

### lch_farrow/

| Файл | Что тестирует |
|------|----------------|
| `test_lch_farrow.py` | Standalone Lagrange 48×5 fractional delay (CPU) |
| `test_lch_farrow_rocm.py` | LchFarrow на ROCm GPU |

---

### vector_algebra/test_matrix_csv_comparison.py

Инверсия матриц: загрузка R_inv_85.csv / R_inv_341.csv → CholeskyInverterROCm → сравнение с эталоном.
Отчёт → `Results/Reports/vector_algebra/matrix_csv_comparison_report.md`

---

### range_angle/test_range_angle.py — 3D FFT радарный процессор (ROCm)

| № | Функция | GPU? | Что тестирует |
|---|---------|------|----------------|
| 1 | `test_default_params` | ❌ | Дефолты RangeAngleParams: n_ant_az=16, n_ant_el=16, n_samples=1.3M |
| 2 | `test_params_helpers` | ❌ | `get_bandwidth/duration/chirp_rate/n_antennas` vs NumPy |
| 3 | `test_peak_mode_enum` | ❌ | Enum TOP_1 / TOP_N, поле n_peaks |
| 4 | `test_repr_objects` | ❌ | repr() RangeAngleParams и TargetInfo |
| 5 | `test_processor_set_get_params` | ✅ | Roundtrip set_params → get_params |
| 6 | `test_range_basic` *(xfail)* | ✅ | LFM τ=0.5 мс → R=75 000 м ± 1 000 м |
| 7 | `test_power_cube_shape` | ✅ | power_cube_numpy() shape (n_rbins, N_AZ, N_EL), float32 |

Тест 6 — `xfail` до полной реализации GPU kernels.

---

### fm_correlator/, capon/, zero_copy/, hybrid/

| Файл | Что тестирует |
|------|----------------|
| `fm_correlator/test_fm_correlator.py` | FM корреляторный детектор (CPU) |
| `fm_correlator/test_fm_correlator_rocm.py` | FM коррелятор на ROCm GPU |
| `capon/test_capon.py` | Адаптивный метод Capon (MVDR) |
| `zero_copy/test_zero_copy.py` | Zero-copy: SVM/pinned memory CPU↔GPU |
| `hybrid/test_hybrid_backend.py` | Hybrid backend: OpenCL + ROCm одновременно |

---

### integration/

| Файл | Что тестирует |
|------|----------------|
| `test_gpuworklib.py` | Сводный: FormSignal, Script, FFT, FindAllMaxima. Графики → `Results/Plots/integration/` |
| `test_fft_integration.py` | Интеграционный тест FFT pipeline |
| `test_signal_gen_integration.py` | Интеграционный тест генераторов сигналов |

---

## Где сохраняются графики

`Results/Plots/[модуль]/` относительно корня проекта:

| Модуль | Папка |
|--------|-------|
| strategies | `Results/Plots/strategies/` |
| filters | `Results/Plots/filters/` |
| heterodyne | `Results/Plots/heterodyne/` |
| statistics | `Results/Plots/statistics/` |
| fft_func | `Results/Plots/fft_func/` |
| fm_correlator | `Results/Plots/fm_correlator/` |
| capon | `Results/Plots/capon/` |
| range_angle | `Results/Plots/range_angle/` |
| lch_farrow | `Results/Plots/lch_farrow/` |
| integration | `Results/Plots/integration/` |
| signal_generators/FormSignal | `Results/Plots/signal_generators/FormSignal/` |
| signal_generators/DelayedFormSignal | `Results/Plots/signal_generators/DelayedFormSignal/` |
| signal_generators/LfmAnalyticalDelay | `Results/Plots/signal_generators/LfmAnalyticalDelay/` |

---

*Обновлено: 2026-03-18*
