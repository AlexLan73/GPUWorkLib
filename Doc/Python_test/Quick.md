# Python_test — Краткий справочник

> Тестовая инфраструктура Python для всех GPU-модулей GPUWorkLib

---

## Структура

```
Python_test/
├── signal_generators/   # 4 файла — CW, LFM, Delayed, Analytical
├── filters/             # 5 файлов — FIR, IIR, AI pipeline
├── heterodyne/          # 4 файла — LFM dechirp
├── fft_maxima/          # 3 файла — поиск максимумов спектра
├── lch_farrow/          # 2 файла — Lagrange 48×5 задержка
├── statistics/          # 1 файл  — mean/median/Welford (ROCm)
├── vector_algebra/      # 2 файла — Cholesky inversion (ROCm)
├── integration/         # 1 файл  — full pipeline
├── hybrid/              # 1 файл  — OpenCL + ROCm context
└── zero_copy/           # 1 файл  — GPU memory bridge
```

**Итого: 10 модулей, 24 тестовых файла + 1 example**

---

## Быстрый старт

### Сборка

```bash
cmake -B build -DBUILD_PYTHON=ON && cmake --build build --config Release
```

### Запуск

```bash
# Один тест
python Python_test/signal_generators/test_form_signal.py

# Без графиков
python Python_test/signal_generators/test_form_signal.py --no-plot

# Через pytest
pytest Python_test/ -v

# Только ROCm тесты (Linux)
bash Python_test/run_all_rocm_tests.sh
```

### PyCharm

- Working dir: корень `GPUWorkLib/`
- PYTHONPATH: `$ProjectFileDir$/build/python`
- Environment: `GPUWORKLIB_PLOT=1`

---

## Покрытие модулей

| Модуль | Файлов | OpenCL | ROCm | Эталон |
|--------|--------|--------|------|--------|
| signal_generators | 4 | ✅ | ✅ | NumPy |
| filters | 5 | ✅ | ✅ | SciPy |
| heterodyne | 4 | ✅ | ✅ | NumPy FFT |
| fft_maxima | 3 | ✅ | ✅* | SciPy find_peaks |
| lch_farrow | 2 | ✅ | ✅ | CPU Lagrange |
| statistics | 1 | — | ✅ | NumPy |
| vector_algebra | 2 | — | ✅ | NumPy + CSV |
| integration | 1 | ✅ | — | Combined |
| hybrid | 1 | ✅ | ✅ | — |
| zero_copy | 1 | ✅ | ✅ | — |

---

## Типичные пороги

| Операция | Порог | Причина |
|----------|-------|---------|
| Identity (delay=0) | 1e-4 — 1e-6 | Только copy |
| CW/LFM генерация | 1e-3 | float32 vs float64 |
| FIR фильтрация | 1e-3 | Линейная свёртка |
| IIR фильтрация | 1e-2 | Рекурсивная ошибка |
| Farrow задержка | 1e-2 — 1e-3 | Lagrange approx |

---

## Графики

Все графики → `Results/Plots/{module}/`:
- `signal_generators/FormSignal/`, `DelayedFormSignal/`, `LfmAnalyticalDelay/`
- `fft_maxima/`, `filters/`, `heterodyne/`, `lch_farrow/`, `integration/`

Отчёты → `Results/Reports/vector_algebra/`

---

## Зависимости

| Пакет | Обязательный | Назначение |
|-------|-------------|------------|
| numpy | ✅ | CPU reference |
| scipy | ⚠️ рекомендуется | Фильтры, find_peaks |
| matplotlib | ⚠️ рекомендуется | Графики |
| pytest | ⚠️ рекомендуется | Test runner |

---

## Python API (pybind11)

```python
# Контексты
ctx = gpuworklib.GPUContext(0)          # OpenCL
ctx = gpuworklib.ROCmGPUContext(0)      # ROCm

# Генераторы
gen = gpuworklib.FormSignalGenerator(ctx)
gen = gpuworklib.DelayedFormSignalGenerator(ctx)

# FFT & Spectrum
fft = gpuworklib.FFTProcessor(ctx)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

# Фильтры
fir = gpuworklib.FirFilter(ctx)         # OpenCL
iir = gpuworklib.IirFilterROCm(ctx)     # ROCm

# Signal Processing
farrow = gpuworklib.LchFarrow(ctx)
het = gpuworklib.HeterodyneDechirp(ctx)
stats = gpuworklib.StatisticsProcessor(ctx)  # ROCm only
```

Подробнее: `Doc/Python/*.md`

---

## Ссылки

- [Full](Full.md) — полное описание, все тесты, архитектура, binding'ы
- [Doc/Python/](../Python/) — Python API документация по модулям

---

*Обновлено: 2026-03-02*
