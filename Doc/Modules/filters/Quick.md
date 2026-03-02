# Filters — Краткий справочник

> FIR и IIR фильтры на GPU: OpenCL (все платформы) + ROCm (AMD, ENABLE_ROCM=1)

**Namespace**: `filters` | **Каталог**: `modules/filters/`

---

## Алгоритмы

```
FIR:  y[ch][n] = Σ h[k] · x[ch][n-k],  k=0..N-1   (direct-form)
IIR:  y[n] = b0·x[n] + w1              (DFII-T, biquad cascade)
      w1   = b1·x[n] - a1·y[n] + w2
      w2   = b2·x[n] - a2·y[n]
```

---

## Классы

| Класс | Backend | Назначение |
|-------|---------|------------|
| `FirFilter` | OpenCL | FIR direct-form, cl_mem |
| `IirFilter` | OpenCL | IIR biquad cascade DFII-T, cl_mem |
| `FirFilterROCm` | ROCm/HIP | FIR, hiprtc, void* ptr |
| `IirFilterROCm` | ROCm/HIP | IIR biquad, hiprtc, void* ptr |

---

## Быстрый старт

### C++ — FirFilter (OpenCL)

```cpp
#include "filters/fir_filter.hpp"

filters::FirFilter fir(backend);
fir.SetCoefficients({0.1f, 0.2f, 0.4f, 0.2f, 0.1f});  // или LoadConfig(json)

auto result = fir.Process(input_buf, channels, points);
clReleaseMemObject(result.data);  // caller owns!
```

### C++ — IirFilter (OpenCL)

```cpp
#include "filters/iir_filter.hpp"

filters::BiquadSection sec;
sec.b0 = 0.02008337f;  sec.b1 = 0.04016673f;  sec.b2 = 0.02008337f;
sec.a1 = -1.56101808f; sec.a2 = 0.64135154f;  // Butterworth 2nd order, fc=0.1

filters::IirFilter iir(backend);
iir.SetBiquadSections({sec});

auto result = iir.Process(input_buf, channels, points);
clReleaseMemObject(result.data);
```

### C++ — FirFilterROCm (Linux + AMD GPU)

```cpp
#include "filters/fir_filter_rocm.hpp"

filters::FirFilterROCm fir(rocm_backend);
fir.SetCoefficients(coeffs);

// Из CPU (upload + process)
auto res = fir.ProcessFromCPU(cpu_data, channels, points);
hipFree(res.data);  // caller owns!

// Из GPU ptr
auto res2 = fir.Process(gpu_ptr, channels, points);
hipFree(res2.data);
```

### Python — FirFilter (OpenCL)

```python
import gpuworklib as gw
import scipy.signal as sig

ctx = gw.GPUContext(0)
fir = gw.FirFilter(ctx)

taps = sig.firwin(64, 0.1)
fir.set_coefficients(taps.tolist())

result = fir.process(signal)  # (channels, points) или 1D complex64
```

### Python — IirFilter (OpenCL)

```python
iir = gw.IirFilter(ctx)
sos = sig.butter(2, 0.1, output='sos')
sections = [
    {'b0': float(r[0]), 'b1': float(r[1]), 'b2': float(r[2]),
     'a1': float(r[4]), 'a2': float(r[5])}
    for r in sos
]
iir.set_sections(sections)
result = iir.process(signal)  # (channels, points) complex64
```

### Python — FirFilterROCm (Linux + AMD GPU)

```python
ctx = gw.ROCmGPUContext(0)
fir = gw.FirFilterROCm(ctx)
fir.set_coefficients(sig.firwin(64, 0.1).tolist())
result = fir.process(data)  # np.ndarray complex64
```

---

## Ключевые нюансы

| Параметр | FIR (OpenCL) | IIR (OpenCL) |
|----------|-------------|-------------|
| NDRange | 2D `(ch, ⌈pts/256⌉×256)` | 1D `(ch,)` |
| Коэффициенты | `__constant` ≤ 16 000 тапов, иначе `__global` | `__constant` SOS-матрица |
| Параллелизм | По каналам И семплам | Только по каналам |
| Рекомендовано | ≥ 8 каналов | ≥ 8 каналов |

---

## On-disk kernel cache

FirFilter и IirFilter используют DrvGPU KernelCacheService:
- **Первый запуск:** компиляция → Save в `modules/filters/kernels/bin/`
- **Повторный:** Load binary (~1 мс вместо ~50 мс компиляции)
- **Fallback:** при отсутствии cache — компиляция из source

---

## Важные ловушки

| # | Ловушка |
|---|---------|
| ⚠️ | `clReleaseMemObject(result.data)` — caller owns (OpenCL) |
| ⚠️ | `hipFree(result.data)` — caller owns (ROCm) |
| ⚠️ | IIR single-channel = медленнее CPU! GPU выгоден только при ≥ 8 каналах |
| ⚠️ | SOS scipy: `a1=row[4], a2=row[5]` (пропускаем `a0=row[3]`, он всегда 1) |
| ⚠️ | `kMaxConstantTaps = 16000`: при больше → автоматически `__global` (медленнее) |
| ⚠️ | Бенчмарк: OpenCL queue **обязательно** с `CL_QUEUE_PROFILING_ENABLE` |
| ⚠️ | ROCm FirFilter на Windows — compile-only stub (throws runtime_error) |

---

## Тесты

| Файл | Что тестирует |
|------|---------------|
| `tests/test_fir_basic.hpp` | OpenCL FIR 64-tap, GPU vs CPU, 8ch × 4096pts |
| `tests/test_iir_basic.hpp` | OpenCL IIR Butterworth 2nd, GPU vs CPU, 8ch × 4096pts |
| `tests/test_filters_rocm.hpp` | ROCm: 6 тестов FIR/IIR (Linux + AMD GPU) |
| `tests/filters_benchmark.hpp` | OpenCL: FirFilterBenchmark, IirFilterBenchmark |
| `Python_test/filters/test_filters_stage1.py` | Python: 5 тестов FIR+IIR vs scipy |
| `Python_test/filters/test_fir_filter_rocm.py` | Python ROCm FIR: 5 тестов |
| `Python_test/filters/test_iir_filter_rocm.py` | Python ROCm IIR: multi-section, GPU ptr |

---

## Ссылки

- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, все тесты с rationale
- [gpu_filters_research.md](gpu_filters_research.md) — Overlap-Save/Add, tiled FIR — будущие алгоритмы

---

*Обновлено: 2026-03-02*
