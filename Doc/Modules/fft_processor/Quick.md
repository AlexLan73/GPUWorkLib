# FFT Processor — Краткий справочник

> GPU FFT (clFFT) с режимами вывода

---

## Режимы (FFTOutputMode)

| Режим | Выход |
|-------|-------|
| `COMPLEX` | complex<float>[nFFT] |
| `MAGNITUDE_PHASE` | mag[nFFT] + phase[nFFT] |
| `MAGNITUDE_PHASE_FREQ` | + freq[nFFT] (Hz) |

---

## Быстрый старт

### C++

```cpp
fft_processor::FFTProcessor fft(backend);
fft_processor::FFTProcessorParams params;
params.beam_count = 16;
params.n_point = 4096;
params.sample_rate = 1000.0f;
params.output_mode = FFTOutputMode::COMPLEX;

auto results = fft.ProcessComplex(data, params);
```

### Python

```python
fft = gpuworklib.FFTProcessor(ctx)
spectrum = fft.process(signal, sample_rate=1000.0, output_mode='complex')
```

---

*Обновлено: 2026-02-17*
