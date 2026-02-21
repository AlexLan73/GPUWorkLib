# FFT Maxima — Полная документация

> Поиск максимумов спектра FFT (SpectrumMaximaFinder)

**Namespace**: `antenna_fft`
**Каталог**: `modules/fft_maxima/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL, clFFT

---

## Обзор

Pipeline: Input → Zero-Pad → clFFT → Post-Processing → Peak Search → Parabolic Interpolation → Results

### Режимы поиска

- **ONE_PEAK**: лучший из левого/правого
- **TWO_PEAKS**: оба пика

---

## API

```cpp
antenna_fft::InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 256,
    .n_point = 1300000,
    .data = my_vector,
    .repeat_count = 2,
    .sample_rate = 1000.0f
};

auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

---

## Python

```python
finder = gpuworklib.SpectrumMaximaFinder(ctx)
result = finder.find_all_maxima(fft_data, sample_rate)
```

См. [Doc/Python/spectrum_maxima_api.md](../../Python/spectrum_maxima_api.md)

---

## Файлы

- [README](README.md) | [API](API.md)

---

*Обновлено: 2026-02-17*
