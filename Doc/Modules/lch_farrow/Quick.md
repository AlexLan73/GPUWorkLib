# LchFarrow — Краткий справочник

> Дробная задержка Lagrange 48×5 на GPU

---

## Алгоритм

```
delay_samples = delay_us * 1e-6 * sample_rate
read_pos = n - delay_samples
if read_pos < 0 → output[n] = 0
center = floor(read_pos), frac = read_pos - center
row = int(frac * 48) % 48
output[n] = sum(L[row][k] * input[center - 1 + k], k=0..4)
```

> ⚠️ `row` — по `frac` позиции чтения, **не** по μ задержки!

---

## Быстрый старт

### C++ — OpenCL

```cpp
#include "lch_farrow.hpp"

lch_farrow::LchFarrow proc(backend);
proc.SetSampleRate(1e6f);
proc.SetDelays({0.0f, 2.7f, 5.0f});

auto result = proc.Process(input_buf, antennas, points);
// caller: clReleaseMemObject(result.data)
```

### C++ — ROCm (ENABLE_ROCM=1, Linux)

```cpp
#include "lch_farrow_rocm.hpp"

lch_farrow::LchFarrowROCm proc(rocm_backend);
proc.SetSampleRate(1e6f);
proc.SetDelays({0.0f, 2.7f, 5.0f});

// CPU→GPU (ProcessFromCPU — плоский вектор)
auto result = proc.ProcessFromCPU(flat_signal, antennas, points);
// caller: hipFree(result.data)
```

### Python

```python
proc = gpuworklib.LchFarrow(ctx)
proc.set_sample_rate(1e6)
proc.set_delays([0.0, 2.7, 5.0])
delayed = proc.process(signal)
```

---

## Стейджи профилирования

| Backend | Стейджи | output_dir |
|---------|---------|-----------|
| OpenCL | `Upload_delay`, `Kernel` | `Results/Profiler/GPU_00_LchFarrow/` |
| ROCm | `Upload_input`, `Upload_delay`, `Kernel` | `Results/Profiler/GPU_00_LchFarrow_ROCm/` |

---

## Тесты

| Файл | Тесты |
|------|-------|
| `tests/test_lch_farrow.hpp` | OpenCL: 3 теста (zero, int5, frac2.7) |
| `tests/test_lch_farrow_rocm.hpp` | ROCm: 4 теста (+ multi-antenna) |
| `Python_test/lch_farrow/test_lch_farrow.py` | Python: 5 тестов |

---

## Ссылки

- [Full](Full.md) | [Python API](../../Python/lch_farrow_api.md) | [tests/README.md](../../../modules/lch_farrow/tests/README.md)

---

*Обновлено: 2026-03-02*
