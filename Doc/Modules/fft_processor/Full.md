# FFT Processor — Полная документация

> GPU FFT с вариантами вывода (Complex / MagPhase / MagPhaseFreq)

**Namespace**: `fft_processor`
**Каталог**: `modules/fft_processor/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL, clFFT

---

## Обзор

`FFTProcessor` возвращает **полный спектр** в нужном формате. Отдельно: `SpectrumMaximaFinder` ищет максимумы.

### Pipeline

```
Input → Zero-Pad (pre-callback) → clFFT → [COMPLEX | MagPhase Kernel] → Output
```

---

## API

```cpp
FFTProcessor fft(backend);
FFTProcessorParams params{beam_count, n_point, sample_rate, output_mode};
auto results = fft.ProcessComplex(vector<complex<float>>, params);
// или ProcessGpu(cl_mem, params)
```

---

## Файлы

- [README](README.md) | [API](API.md)

---

*Обновлено: 2026-02-17*
