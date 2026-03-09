# Heterodyne — Краткий справочник

> Дечирп (stretch processing) ЛЧМ-радара на GPU

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
Это ключевая обработка в FMCW (частотно-модулированном непрерывном) радаре. Радар посылает ЛЧМ-сигнал (chirp) — сигнал, чья частота линейно растёт. Отражённый сигнал приходит с задержкой, пропорциональной расстоянию до цели.

---

### Как это работает (без формул)

Принятый сигнал перемножается с сопряжённой копией опорного (переданного) chirp-сигнала. После умножения мгновенная разность фаз становится постоянной — получается тон (beat-сигнал), чья частота пропорциональна расстоянию. Затем FFT этого beat-сигнала показывает пик на beat-частоте. По этой частоте вычисляется дальность до цели.

**Аналогия**: как в музыке — два похожих тона при сложении дают биение с разностной частотой. Здесь то же самое, только chirp против своей задержанной копии.

---

### Что конкретно делает модуль

**Dechirp** — основная операция: `dc = conj(rx × ref)`. Входные данные — массив принятых сигналов по всем антеннам. Выход — beat-сигнал для каждой антенны + его FFT + пик + дальность.

**Correct** — коррекция фазы beat-сигнала: умножение на `exp(−j·2π·f_beat/fs·n)`. Сдвигает пик к DC. Используется для верификации и в `HeterodyneROCm`.

**DechirpFromGPU / DechirpWithGPURef** — варианты для GPU-пайплайна: входные данные и/или опорный сигнал уже лежат в GPU-памяти — не нужна лишняя перекачка по PCIe (OPT-3).

---

### Откуда берётся опорный сигнал (ref)?

**OpenCL**: из модуля SignalGenerators — класс `LfmConjugateGenerator`, генерирует conj(s_tx).
**ROCm**: CPU-функция `GenerateConjugateLfmCpu()` — fallback, т.к. LfmConjugateGenerator использует OpenCL.

---

### Какой класс брать

| Задача | Класс |
|--------|-------|
| Получить f_beat + дальность + SNR | `HeterodyneDechirp` (C++) или `gpuworklib.HeterodyneDechirp` (Python) |
| Встроить дечирп в свой пайплайн (без FFT) | `gpuworklib.HeterodyneROCm` (Python, ROCm-only) |
| Замерить производительность kernel | `HeterodyneDechirpBenchmark` / `HeterodyneCorrectBenchmark` в `tests/` |

---

### Связи с другими модулями

- **signal_generators** → `LfmConjugateGenerator` — генерирует ref
- **fft_maxima** → `SpectrumMaximaFinder` — FFT + поиск пика beat-частоты (pad N=8000 → 8192)
- **DrvGPU** → `IBackend*`, `GPUProfiler`, `ConsoleOutput`, `KernelCacheService`

---

## Алгоритм

```
dc = conj(rx × ref)  →  FFT (pad N→2^k)  →  f_beat  →  R = c·T·f_beat / (2·B)
ref = conj(s_tx),  f_beat = mu·tau = (B/T)·(2R/c)
```

---

## Быстрый старт

### C++ — OpenCL

```cpp
#include "heterodyne_dechirp.hpp"

drv_gpu_lib::HeterodyneDechirp het(backend);
het.SetParams({.f_start=0, .f_end=2e6f, .sample_rate=12e6f,
               .num_samples=8000, .num_antennas=5});

// rx_data: flat complex<float>[antennas × N]
auto result = het.Process(rx_data);

if (result.success) {
    for (auto& a : result.antennas)
        // a.f_beat_hz, a.range_m, a.peak_snr_db
}
```

### C++ — ROCm (ENABLE_ROCM=1, Linux + AMD GPU)

```cpp
// Внимание: BackendType::ROCm — строчная m!
drv_gpu_lib::HeterodyneDechirp het(backend, BackendType::ROCm);
het.SetParams({.f_start=0, .f_end=2e6f, .sample_rate=12e6f,
               .num_samples=8000, .num_antennas=5});
auto result = het.Process(rx_data);
```

### C++ — rx уже на GPU (без PCIe round-trip, OPT-3)

```cpp
// OpenCL: gpu_ptr = &cl_mem_handle
// ROCm:   gpu_ptr = &hip_device_ptr
auto result = het.ProcessExternal(gpu_ptr, params);
// ВАЖНО: буфер НЕ освобождается фасадом
```

### Python — HeterodyneDechirp (полный пайплайн)

```python
import gpuworklib, numpy as np

ctx = gpuworklib.GPUContext(0)
het = gpuworklib.HeterodyneDechirp(ctx)
het.set_params(0.0, 2e6, 12e6, 8000, 5)   # f_start, f_end, fs, N, antennas

rx = np.zeros(5 * 8000, dtype=np.complex64)  # flat [antennas × N]
result = het.process(rx)
# result['success']
# result['antennas'][i] → {'f_beat_hz', 'f_beat_bin', 'range_m', 'peak_amplitude', 'peak_snr_db'}
```

### Python — HeterodyneROCm (ROCm-only, сырой дечирп без FFT)

```python
ctx = gpuworklib.ROCmGPUContext(0)
het = gpuworklib.HeterodyneROCm(ctx)
het.set_params(f_start=0, f_end=2e6, sample_rate=12e6, num_samples=8000, num_antennas=5)

dc = het.dechirp(rx_flat, ref)          # → ndarray[complex64]
corrected = het.correct(dc, f_beats)    # → ndarray[complex64]
p = het.params                          # dict с параметрами
```

---

## Ключевые параметры

| Параметр | Тип | Пример | Описание |
|----------|-----|--------|----------|
| f_start, f_end | float | 0, 2e6 | ЛЧМ полоса B [Гц] |
| sample_rate | float | 12e6 | fs [Гц] |
| num_samples | int | 8000 | N точек на антенну |
| num_antennas | int | 5 | Количество каналов |

**Производные** (методы HeterodyneParams):
- `GetBandwidth()` = f_end − f_start = 2 МГц
- `GetDuration()` = N/fs = 666.67 мкс
- `GetChirpRate()` = B/T = 3·10⁹ Гц/с
- `GetBinWidth()` = fs/N = 1500 Гц/бин

---

## Стейджи профилирования

| Backend | Benchmark | Стейджи |
|---------|-----------|---------|
| OpenCL | Dechirp | `Upload_Rx`, `Upload_Ref`, `Kernel_Multiply`, `Download` |
| OpenCL | Correct | `Upload_DC`, `Upload_PhaseStep`, `Kernel_Correct`, `Download` |
| ROCm | Dechirp | `Upload_Rx`, `Upload_Ref`, `Kernel_Multiply`, `Download` |
| ROCm | Correct | `Upload_DC`, `Upload_PhaseStep`, `Kernel_Correct`, `Download` |

Активация: `"is_prof": true` в `configGPU.json`, раскомментировать `test_heterodyne_benchmark::run()` в `tests/all_test.hpp`.

---

## Тесты

| Файл | Тесты |
|------|-------|
| `tests/test_heterodyne_basic.hpp` | OpenCL: 1-single ant, 2-5 ant linear, 3-correction, 6-random delays |
| `tests/test_heterodyne_pipeline.hpp` | OpenCL: 4-full pipeline, 5-process_external, 7-AllMaxima |
| `tests/test_heterodyne_rocm.hpp` | ROCm: 6 тестов (Linux + AMD GPU) |
| `Python_test/heterodyne/test_heterodyne.py` | Python: 4 теста (HeterodyneDechirp) |
| `Python_test/heterodyne/test_heterodyne_rocm.py` | Python: 6 тестов (HeterodyneROCm vs NumPy) |

**Типовые параметры тестов**: fs=12 МГц, B=2 МГц, N=8000, delays=[100..500] мкс, tolerance=±5 кГц.

---

## Ограничения

- delay > T (666.67 мкс) → сигнал пустой, f_beat = 0
- `BackendType::ROCM` не компилируется — нужна строчная `m`: `BackendType::ROCm`
- `HeterodyneROCm` (Python) — только ROCm; нет FFT; нет дальности; для встраивания в свой пайплайн
- ProcessExternal НЕ освобождает переданный GPU буфер

---

## Ссылки

- [Full.md](Full.md) — полное описание, математика, pipeline, C4, все тесты
- [API.md](API.md) — API-справочник по всем классам и методам

---

*Обновлено: 2026-03-09*
