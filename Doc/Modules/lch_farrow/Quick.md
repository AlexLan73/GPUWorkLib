# LchFarrow — Краткий справочник

> Дробная задержка Lagrange 48×5 на GPU

---

## Алгоритм

```
delay_samples = delay_us * 1e-6 * sample_rate
D = floor(delay_samples), mu = delay_samples - D
row = int(mu * 48) % 48
output[n] = sum(L[row][k] * input[n-D-1+k], k=0..4)
```

---

## Быстрый старт

### C++

```cpp
#include "lch_farrow.hpp"

lch_farrow::LchFarrow proc(backend);
proc.SetSampleRate(1e6f);
proc.SetDelays({0.0f, 2.7f, 5.0f});

auto result = proc.Process(input_buf, antennas, points);
```

### Python

```python
proc = gpuworklib.LchFarrow(ctx)
proc.set_sample_rate(1e6)
proc.set_delays([0.0, 2.7, 5.0])
delayed = proc.process(signal)
```

---

## Ссылки

- [Full](Full.md) | [lch_farrow_api.md](../../Python/lch_farrow_api.md)

---

*Обновлено: 2026-02-17*
