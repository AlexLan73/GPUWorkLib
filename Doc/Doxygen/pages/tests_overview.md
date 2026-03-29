# Тесты GPUWorkLib {#tests_overview_page}

> Сводка C++ и Python тестов всех 12 модулей.
> Подробности — на странице каждого модуля (ссылки в колонке Docs).

---

## Сводная таблица

| # | Модуль | C++ тестов | Python тестов | Benchmark | Plots | Docs |
|---|--------|-----------|---------------|-----------|-------|------|
| 0 | DrvGPU | 12 файлов | - | GPUProfiler | Profiler MD/JSON | @ref drvgpu_main |
| 1 | fft_func | 6 + 4 bench | 3 | FFT, Maxima | fft_maxima/ | @ref fft_func_tests |
| 2 | statistics | 2 + 2 bench | 3 | ComputeAll | statistics/ | @ref statistics_tests |
| 3 | vector_algebra | 4 + stage prof | 2 | Symmetrize | - | @ref vector_algebra_tests |
| 4 | filters | 4 + 2 bench | 10 | FIR, IIR | filters/ (10 PNG) | @ref filters_tests |
| 5 | signal_generators | 2 + 4 bench | 4+ | FormSignal | signal_generators/ (28 PNG) | @ref signal_generators_tests |
| 6 | lch_farrow | 1 + 2 bench | 2 | OCL + ROCm | - | @ref lch_farrow_tests |
| 7 | heterodyne | 2 + 4 bench | 4 | Dechirp, Correct | heterodyne/ | @ref heterodyne_tests |
| 8 | fm_correlator | 2 + 4 bench | 2 | Parametric sweep | - | @ref fm_correlator_tests |
| 9 | strategies | 6+ (GoF) | 7+ | Profiling, Streams | strategies/ (4 PNG) | @ref strategies_tests |
| 10 | capon | 4 + 2 bench | 1 | Relief, Beamform | - | @ref capon_tests |
| 11 | range_angle | 2 (stubs) | 1 (stub) | - | - | @ref range_angle_tests |

---

## Как запускать

### C++ тесты

```bash
# Порядок из config/tests_order.txt (комментировать # для пропуска)
./GPUWorkLib

# Все модули
./GPUWorkLib all

# Один модуль
./GPUWorkLib drvgpu
./GPUWorkLib fft_func
./GPUWorkLib statistics
./GPUWorkLib capon
# ... и т.д.

# Из произвольного файла
./GPUWorkLib --file my_order.txt
```

### Python тесты

```bash
# Каждый файл — независимый (НЕ pytest!)
python Python_test/fft_func/test_process_magnitude_rocm.py
python Python_test/statistics/test_compute_all.py
python Python_test/filters/test_fir_filter_rocm.py
python Python_test/heterodyne/test_heterodyne_comparison.py
```

---

## Результаты

| Тип | Путь | Формат |
|-----|------|--------|
| Profiler reports | `Results/Profiler/GPU_00_*/` | Markdown + JSON |
| Test JSON | `Results/JSON/` | JSON |
| Plots | `Results/Plots/{module}/` | PNG |
| Logs | `Logs/DRVGPU_XX/` | plog text |

---

## Порядок запуска (config/tests_order.txt)

```
drvgpu              #  0. Драйвер GPU
fft_func            #  1. FFT + спектр
statistics          #  2. Mean, Median, Variance
vector_algebra      #  3. Cholesky inversion
filters             #  4. FIR, IIR, MA, Kalman, KAMA
signal_generators   #  5. CW, LFM, Noise, FormSignal
lch_farrow          #  6. Lagrange 48x5
heterodyne          #  7. LFM Dechirp
fm_correlator       #  8. FM M-sequence
strategies          #  9. Digital beamforming
capon               # 10. MVDR Capon
range_angle         # 11. 3D range-angle
```
