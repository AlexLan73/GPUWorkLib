# SpectrumMaximaFinder — Python API

> **Модуль**: `gpuworklib.SpectrumMaximaFinder`
> **C++ класс**: `antenna_fft::SpectrumMaximaFinder`
> **Обновлено**: 2026-02-14

## Обзор

GPU-ускоренный поиск ВСЕХ локальных максимумов в FFT-спектре.

**Алгоритм**: Обнаружение -> Префиксная сумма (Blelloch Scan) -> Stream Compaction

Локальный максимум в позиции `i`: `magnitude[i] > magnitude[i-1]` И `magnitude[i] > magnitude[i+1]`

## Быстрый старт

```python
import numpy as np
import gpuworklib

# Настройка
ctx = gpuworklib.GPUContext(0)
fft = gpuworklib.FFTProcessor(ctx)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

# Генерация сигнала (3 частоты)
fs = 1000.0
t = np.arange(1024, dtype=np.float32)
signal = (np.sin(2*np.pi*50*t/fs) +
          np.sin(2*np.pi*120*t/fs) +
          np.sin(2*np.pi*200*t/fs)).astype(np.complex64)

# FFT на GPU
spectrum = fft.process_complex(signal, sample_rate=fs)

# Поиск ВСЕХ локальных максимумов
result = finder.find_all_maxima(spectrum, sample_rate=fs)

print(f"Найдено {result['num_maxima']} пиков")
print(f"Частоты: {result['frequencies']} Гц")
print(f"Амплитуды:  {result['magnitudes']}")
print(f"Позиции бинов: {result['positions']}")
```

## Конструктор

```python
finder = gpuworklib.SpectrumMaximaFinder(ctx)
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `ctx` | `GPUContext` | GPU-контекст (обязательный) |

## Методы

### find_all_maxima()

```python
result = finder.find_all_maxima(
    fft_data,                # numpy complex64 массив
    sample_rate,             # float, Гц
    beam_count=0,            # автоопределение из shape
    nFFT=0,                  # автоопределение из shape
    search_start=0,          # 0 = по умолчанию (1, пропуск DC)
    search_end=0             # 0 = по умолчанию (nFFT/2)
)
```

| Параметр | Тип | По умолч. | Описание |
|----------|-----|-----------|----------|
| `fft_data` | `np.ndarray` complex64 | обязат. | Результат FFT. 1D `(nFFT,)` или 2D `(лучи, nFFT)` |
| `sample_rate` | `float` | обязат. | Частота дискретизации, Гц |
| `beam_count` | `int` | 0 | Количество лучей (0 = авто из shape) |
| `nFFT` | `int` | 0 | Размер FFT (0 = авто из shape) |
| `search_start` | `int` | 0 | Начальный бин поиска (0 = 1, пропуск DC) |
| `search_end` | `int` | 0 | Конечный бин поиска (0 = nFFT/2) |

### Возвращаемое значение

**Один луч** (1D вход или beam_count=1): `dict`

```python
{
    "positions":   np.array([51, 123, 205, ...], dtype=np.uint32),
    "magnitudes":  np.array([478.8, 497.7, 479.0, ...], dtype=np.float32),
    "frequencies": np.array([49.8, 120.1, 200.2, ...], dtype=np.float32),
    "num_maxima":  12
}
```

**Много лучей** (2D вход): `list[dict]`

```python
[
    {
        "antenna_id":  0,
        "positions":   np.array([...], dtype=np.uint32),
        "magnitudes":  np.array([...], dtype=np.float32),
        "frequencies": np.array([...], dtype=np.float32),
        "num_maxima":  7
    },
    {
        "antenna_id":  1,
        ...
    },
    ...
]
```

## Примеры

### Многолучевая обработка

```python
ctx = gpuworklib.GPUContext(0)
sig = gpuworklib.SignalGenerator(ctx)
fft = gpuworklib.FFTProcessor(ctx)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

# Генерация 8 лучей: 100, 200, 300, ..., 800 Гц
signals = sig.generate_cw(freq=100, fs=4000, length=4096,
                           beam_count=8, freq_step=100)
spectra = fft.process_complex(signals, sample_rate=4000)

# Поиск всех пиков
results = finder.find_all_maxima(spectra, sample_rate=4000)

for i, beam in enumerate(results):
    main_peak_idx = np.argmax(beam["magnitudes"])
    main_freq = beam["frequencies"][main_peak_idx]
    print(f"Луч {beam['antenna_id']}: главный пик на {main_freq:.1f} Гц "
          f"({beam['num_maxima']} пиков всего)")
```

### Сравнение с SciPy

```python
from scipy.signal import find_peaks

# Пики на GPU
gpu_result = finder.find_all_maxima(spectrum, sample_rate=fs)

# Пики через SciPy
magnitude = np.abs(spectrum[:nFFT//2])
scipy_peaks, _ = find_peaks(magnitude)

# Сравнение
gpu_set = set(gpu_result["positions"].tolist())
scipy_set = set(scipy_peaks.tolist())
print(f"Совпадение: {len(gpu_set & scipy_set)}/{len(scipy_set)}")
```

### Визуализация (Matplotlib)

```python
import matplotlib.pyplot as plt

result = finder.find_all_maxima(spectrum, sample_rate=fs)
nFFT = len(spectrum)
freq_axis = np.arange(nFFT // 2) * fs / nFFT
magnitude = np.abs(spectrum[:nFFT // 2])

plt.figure(figsize=(12, 5))
plt.plot(freq_axis, magnitude, 'b-', label='Спектр')
mask = result["positions"] < nFFT // 2
plt.plot(result["frequencies"][mask], result["magnitudes"][mask],
         'rv', markersize=10, label=f'Пики ({result["num_maxima"]})')
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Производительность

Бенчмарк на NVIDIA GeForce RTX 2080 Ti:

| Конфигурация | Общее время | На луч |
|-------------|------------|--------|
| 1 луч x 1024 FFT | 0.03 мс | 0.03 мс |
| 5 лучей x 512 FFT | 0.09 мс | 0.02 мс |
| 256 лучей x 4096 FFT | ~22 мс | 0.088 мс |

Разбивка по GPU-ядрам (256 x 4096):
- Обнаружение (Detection): ~0.02 мс
- Префиксная сумма (Scan): ~0.004 мс
- Уплотнение (Compaction): ~0.035 мс
- Накладные расходы хоста (выделение памяти, чтение): ~22 мс

## Замечания

- На вход подаётся **результат FFT** (комплексный спектр), а не сырой сигнал
- Диапазон поиска по умолчанию: `[1, nFFT/2)` (пропуск DC и отрицательных частот)
- Для конвейера «сырой сигнал -> пики» сначала используйте FFTProcessor:
  `сигнал -> fft.process_complex() -> finder.find_all_maxima()`
- Обнаружение использует модуль: `sqrt(re^2 + im^2)`
- Результаты всегда возвращаются на CPU (OutputDestination::CPU в Python API)

## Тесты

Файл Python-тестов: `Python_test/test_spectrum_find_all_maxima.py`

```bash
python Python_test/test_spectrum_find_all_maxima.py
```

Графики сохраняются в: `Results/Plots/`
- `test1_single_tone.png` — Спектр одной частоты + пики
- `test2_three_tones.png` — Три частоты + пики
- `test3_multi_beam.png` — 5 лучей с разными частотами
- `test4_gpu_vs_scipy.png` — Сравнение GPU vs SciPy find_peaks
- `test5_performance.png` — Гистограмма времени бенчмарка
