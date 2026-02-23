# Heterodyne — Краткий справочник

> Дечирп (stretch processing) ЛЧМ-радара на GPU

---

## Алгоритм

```
s_dc = s_rx × conj(s_tx)  →  FFT  →  f_beat  →  R = c·T·f_beat / (2·B)
```

---

## Быстрый старт

### C++

```cpp
#include "heterodyne_dechirp.hpp"

drv_gpu_lib::HeterodyneDechirp het(backend);
het.SetParams({.f_start=0, .f_end=2e6f, .sample_rate=12e6f,
               .num_samples=8000, .num_antennas=5});

auto result = het.Process(rx_data);  // rx_data: flat [antennas × N]
// result.antennas[i].f_beat_hz, .range_m, .peak_snr_db
```

### Python

```python
het = gpuworklib.HeterodyneDechirp(ctx)
het.set_params(f_start=0, f_end=2e6, sample_rate=12e6, num_samples=8000, num_antennas=5)
result = het.process(rx_signal)
```

---

## Параметры

| Параметр | Описание | Пример |
|----------|----------|--------|
| f_start, f_end | ЛЧМ полоса B [Hz] | 0, 2e6 |
| sample_rate | fs [Hz] | 12e6 |
| num_samples | N точек на антенну | 8000 |
| num_antennas | Каналов | 5 |

---

## Ссылки

- [Full](Full.md) — полное описание, математика, pipeline, тесты

---

*Обновлено: 2026-02-23*
