# Heterodyne — Краткий справочник

> Дечирп (stretch processing) ЛЧМ-радара на GPU

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
