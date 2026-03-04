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

**Dechirp** — основная операция: принятый сигнал × conj(ref). Входные данные — массив принятых сигналов по всем антеннам. Выход — beat-сигнал для каждой антенны + его FFT + пик + дальность.

**Correct** — коррекция фазы beat-сигнала. Применяется после Dechirp, если нужна доп. фазовая компенсация (например, при многоканальной обработке).

**DechirpFromGPU / DechirpWithGPURef** — варианты для GPU-пайплайна: входные данные и/или опорный сигнал уже лежат в GPU-памяти — не нужна лишняя перекачка по PCIe.

---

### Откуда берётся опорный сигнал (ref)?

Из модуля SignalGenerators — класс `LfmConjugateGenerator`. Он генерирует conj(s_tx) — сопряжённую копию переданного chirp-сигнала. Именно поэтому `LfmConjugateGenerator` самостоятельно не используется — только в связке с Heterodyne.

---

## Алгоритм

```
dc = conj(rx × ref)  →  FFT  →  f_beat  →  R = c·T·f_beat / (2·B)
ref = conj(s_tx),  f_beat = mu·tau = (B/T)·(2R/c)
```

---

## Быстрый старт

### C++ — OpenCL

```cpp
#include "heterodyne_dechirp.hpp"

drv_gpu_lib::HeterodyneDechirp het(backend);
het.SetParams({.f_start=0, .f_end=1e6f, .sample_rate=12e6f,
               .num_samples=4000, .num_antennas=5});

auto result = het.Process(rx_data);  // rx_data: flat [antennas × N]
// result.antennas[i].f_beat_hz, .range_m, .peak_snr_db
```

### C++ — ROCm (ENABLE_ROCM=1, Linux)

```cpp
drv_gpu_lib::HeterodyneDechirp het(backend, BackendType::ROCM);
het.SetParams({...});
auto result = het.Process(rx_data);
```

### Python

```python
het = gpuworklib.HeterodyneDechirp(ctx)
het.set_params(f_start=0, f_end=1e6, sample_rate=12e6, num_samples=4000, num_antennas=5)
result = het.process(rx_signal)
```

---

## Параметры

| Параметр | Описание | Пример |
|----------|----------|--------|
| f_start, f_end | ЛЧМ полоса B [Hz] | 0, 1e6 |
| sample_rate | fs [Hz] | 12e6 |
| num_samples | N точек на антенну | 4000 |
| num_antennas | Каналов | 5 |

---

## Стейджи профилирования

| Backend | Benchmark | Стейджи |
|---------|-----------|---------|
| OpenCL | Dechirp | `Upload_Rx`, `Upload_Ref`, `Kernel_Multiply`, `Download` |
| OpenCL | Correct | `Upload_DC`, `Upload_PhaseStep`, `Kernel_Correct`, `Download` |
| ROCm | Dechirp | `Upload_Rx`, `Upload_Ref`, `Kernel_Multiply`, `Download` |
| ROCm | Correct | `Upload_DC`, `Upload_PhaseStep`, `Kernel_Correct`, `Download` |

---

## Тесты

| Файл | Тесты |
|------|-------|
| `tests/test_heterodyne_basic.hpp` | OpenCL: 3 теста (single, 5-ant, correction) |
| `tests/test_heterodyne_pipeline.hpp` | OpenCL: 2 теста (full pipeline, external) |
| `tests/test_heterodyne_rocm.hpp` | ROCm: 6 тестов (Linux + AMD GPU) |

---

## Ссылки

- [Full](Full.md) — полное описание, математика, pipeline, тесты

---

*Обновлено: 2026-03-02*
