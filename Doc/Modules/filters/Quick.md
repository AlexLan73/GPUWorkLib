# Filters — Краткий справочник

> FIR и IIR фильтры на GPU

---

## Классы

| Класс | Назначение |
|-------|------------|
| `FirFilter` | FIR convolution (direct-form) |
| `IirFilter` | IIR biquad cascade (DFII-T) |

---

## Быстрый старт

### C++

```cpp
#include "filters/fir_filter.hpp"

filters::FirFilter fir(backend);
fir.SetCoefficients({0.1f, 0.2f, 0.4f, 0.2f, 0.1f});

auto result = fir.Process(input_buf, channels, points);
// result.data — cl_mem
```

### Python

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
fir = gpuworklib.FirFilter(ctx)
fir.set_coefficients([0.1, 0.2, 0.4, 0.2, 0.1])
out = fir.process(signal)
```

---

## Тесты

- C++: `modules/filters/tests/test_fir_basic.hpp`, `test_iir_basic.hpp`
- Python: `Python_test/test_filters*.py`

---

*Обновлено: 2026-02-17*
