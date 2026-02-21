# Filters — Полная документация

> FIR и IIR фильтры на GPU (OpenCL)

**Namespace**: `filters`
**Каталог**: `modules/filters/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL

---

## Обзор

| Класс | Алгоритм | Коэффициенты |
|-------|----------|--------------|
| **FirFilter** | Direct-form convolution | scipy.signal.firwin, JSON |
| **IirFilter** | Biquad cascade DFII-T | scipy.signal.butter (SOS) |

### FIR

```
y[ch][n] = sum_{k=0}^{N-1} h[k] * x[ch][n-k]
```

- Коэффициенты: `__constant` (≤16K taps) или `__global`
- 2D NDRange: каналы × точки

### IIR

- Biquad cascade (order 2–10+)
- Все секции в одном kernel
- SOS формат (b0, b1, b2, a1, a2)

---

## API

### FirFilter

```cpp
FirFilter fir(backend);
fir.SetCoefficients(std::vector<float> coeffs);
fir.LoadConfig("lowpass.json");  // JSON: { "type":"fir", "coefficients":[...] }
auto result = fir.Process(cl_mem input, uint32_t channels, uint32_t points);
auto cpu_result = fir.ProcessCpu(vector<complex<float>>, channels, points);
```

### IirFilter

```cpp
IirFilter iir(backend);
iir.SetSos(sos_matrix);  // [[b0,b1,b2,a1,a2], ...]
iir.LoadConfig("butterworth.json");
auto result = iir.Process(input, channels, points);
```

---

## Файлы

```
modules/filters/
├── include/
│   ├── filters/fir_filter.hpp
│   ├── filters/iir_filter.hpp
│   ├── types/filter_params.hpp
│   └── kernels/fir_kernels.hpp
├── src/
│   ├── fir_filter.cpp
│   └── iir_filter.cpp
└── tests/
    ├── test_fir_basic.hpp
    └── test_iir_basic.hpp
```

---

## Ссылки

- [Quick](Quick.md)
- [README](README.md)
- Python: `Doc/Python/` (если есть)

---

*Обновлено: 2026-02-17*
