# FFT Maxima (SpectrumMaximaFinder) — Краткий справочник

> GPU-поиск пиков FFT-спектра: один пик с параболической интерполяцией или все локальные максимумы

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
После FFT у нас есть спектр — массив амплитуд по частотам. Но нас обычно интересует не весь спектр, а конкретные пики — частоты, на которых есть сигналы. Этот модуль находит эти пики на GPU прямо в спектре.

---

### Два режима поиска

**Один пик (ONE_PEAK)** — находит самый большой пик во всём спектре и уточняет его частоту с субчастотной точностью через параболическую интерполяцию. Результат: одна частота + амплитуда на луч. Используй для простых задач: определить частоту тона, измерить beat-частоту в радаре.

**Все максимумы (AllMaxima / FindAllMaxima)** — находит все локальные максимумы: где амплитуда больше соседей. Возвращает список пар (частота, амплитуда) на каждый луч. Используй для сложных сценариев: несколько сигналов одновременно, поиск всех целей в радаре.

---

### Три метода — чем отличаются

**`Process()`** — принимает сырой сигнал, внутри делает FFT, потом ищет 1 или 2 пика. Всё-в-одном.

**`FindAllMaxima()`** — принимает сырой сигнал, внутри делает FFT, потом ищет все локальные максимумы. Тоже всё-в-одном.

**`AllMaxima()`** — принимает уже готовый FFT-спектр. Только поиск, без FFT. Используй когда FFT уже был сделан (через FFTProcessor) и не хочешь делать его второй раз. В Python — всегда именно этот путь.

---

### Важная деталь про Python

В Python-биндинге нет метода с встроенным FFT. Нужно сначала сделать `fft.process_complex()` через FFTProcessor, и только потом передать результат в `finder.find_all_maxima()`. Это позволяет переиспользовать спектр для разных целей.

---

## Алгоритм

```
signal  → Zero-Pad → clFFT → peak reduction → δ = 0.5*(yL-yR)/(yL-2yC+yR) → f_refined
signal  → Zero-Pad → clFFT → detect → prefix_sum → compact → AllMaximaResult
spectrum → compute_magnitudes → detect → prefix_sum → compact → AllMaximaResult
```

---

## Три метода

| Метод | Вход | Назначение |
|-------|------|------------|
| `Process()` | Сырой сигнал | 1 или 2 пика с параболической интерполяцией |
| `FindAllMaxima()` | Сырой сигнал | FFT внутри + все локальные максимумы |
| `AllMaxima()` | FFT-спектр | Только поиск пиков, без FFT |

---

## Быстрый старт

### C++ — один пик

```cpp
#include "spectrum_maxima_finder.h"
#include "interface/spectrum_input_data.hpp"

antenna_fft::SpectrumMaximaFinder finder(backend);
// Initialize() вызывается автоматически при первом Process

antenna_fft::InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 5, .n_point = 100000, .data = my_signal,
    .repeat_count = 4, .sample_rate = 1000.0f
};

auto results = finder.Process(input, antenna_fft::PeakSearchMode::ONE_PEAK,
                              antenna_fft::DriverType::OPENCL);
// results[i].interpolated.refined_frequency, .magnitude
```

### C++ — все пики

```cpp
antenna_fft::InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 64, .n_point = 1024, .data = raw_signal, .sample_rate = 1000.0f
};
auto result = finder.FindAllMaxima(input, antenna_fft::OutputDestination::CPU);
// result.beams[i].maxima[j].index, .refined_frequency, .magnitude
```

### Python

```python
import gpuworklib, numpy as np

ctx = gpuworklib.GPUContext(0)
fft = gpuworklib.FFTProcessor(ctx)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

fs, nFFT = 1000.0, 1024
t = np.arange(nFFT, dtype=np.float32)
signal = np.sin(2 * np.pi * 100 * t / fs).astype(np.complex64)

# Python принимает FFT, не сырой сигнал!
spectrum = fft.process_complex(signal, sample_rate=fs)
result = finder.find_all_maxima(spectrum, sample_rate=fs)

print(result['frequencies'])    # np.array float32
print(result['magnitudes'])     # np.array float32
print(result['num_maxima'])     # int
```

### Python — несколько лучей

```python
# signals shape: (beam_count, nFFT)
spectra = fft.process_complex(signals, sample_rate=fs)
result = finder.find_all_maxima(spectra, sample_rate=fs)
# result → list[dict]: 'frequencies', 'magnitudes', 'positions', 'antenna_id'
```

---

## Ключевые параметры (InputData\<T\>)

| Поле | Тип | Описание | Умолчание |
|------|-----|----------|-----------|
| antenna_count | uint32 | Количество лучей/антенн | — |
| n_point | uint32 | Точек на луч (для AllMaxima: n_point=nFFT!) | — |
| repeat_count | uint32 | nFFT = nextPow2(n_point) × r | 2 |
| sample_rate | float | Гц | 1000.0 |
| memory_limit | float | Доля GPU памяти для batch | 0.80 |
| max_maxima_per_beam | size_t | Лимит максимумов на луч | 1000 |
| search_range | uint32 | 0 = auto = nFFT/4 | 0 |

---

## Типы T для InputData

| T | Когда |
|---|-------|
| `std::vector<std::complex<float>>` | CPU данные (upload на GPU) |
| `cl_mem` | GPU данные (zero-copy) |

---

## ⚠️ Частые ошибки

| Ошибка | Правильно |
|--------|-----------|
| AllMaxima с сырым сигналом | AllMaxima только с FFT! `n_point = nFFT` |
| Python: передать сырой сигнал | Сначала `fft.process_complex()`, потом `find_all_maxima()` |
| Dest=GPU: не освободить буферы | `clReleaseMemObject(result.gpu_maxima)` обязателен |
| clFFT на AMD RDNA4+ (gfx1201) | Использовать только ROCm backend |

---

## Ссылки

- [Full.md](Full.md) — полная документация, математика, pipeline, C4, все тесты
- [API.md](API.md) — детальный API reference (FFTPlanCache, FFTBatchAdapter)
- [FindAllMaxima_MaxValue_Guide.md](FindAllMaxima_MaxValue_Guide.md) — формат MaxValue, CPU/GPU пути
- [Doc/Python/spectrum_maxima_api.md](../../Python/spectrum_maxima_api.md) — Python API

---

*Обновлено: 2026-03-02*
