# Filters — Краткий справочник

> FIR и IIR фильтры на GPU

---

## Классы

| Класс | Назначение |
|-------|------------|
| `FirFilter` | FIR convolution (direct-form) |
| `IirFilter` | IIR biquad cascade (DFII-T) |

---

## On-disk kernel cache

FirFilter и IirFilter используют DrvGPU [KernelCacheService](../../DrvGPU/Services/Quick.md):
- **Первый запуск:** компиляция → Save в `modules/filters/kernels/bin/`
- **Повторный:** Load binary (~1 мс вместо ~50 мс компиляции)
- **Fallback:** при отсутствии cache — компиляция из source

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

## FilterConfigService (сохранение конфигов)

DrvGPU [FilterConfigService](../../DrvGPU/Services/Quick.md) — сохранение/загрузка коэффициентов FIR/IIR в JSON:
- `filters/{name}.json` — тип, коэффициенты, comment
- Версионирование при перезаписи: `name_00.json`, `name_01.json`
- Интеграция SaveFilterConfig/LoadFilterConfig в FirFilter/IirFilter — планируется (TASK-006)

---

## Тесты

- C++: `modules/filters/tests/test_fir_basic.hpp`, `test_iir_basic.hpp`
- Python: `Python_test/filters/test_filters_stage1.py`

---

*Обновлено: 2026-02-23*
