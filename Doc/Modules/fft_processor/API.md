# FFT Processor — API Reference

**Namespace**: `fft_processor`

---

## Типы данных

### FFTOutputMode

```cpp
enum class FFTOutputMode {
    COMPLEX,              // Сырой комплексный спектр (float2)
    MAGNITUDE_PHASE,      // |FFT|, phase(FFT)
    MAGNITUDE_PHASE_FREQ  // |FFT|, phase, freq_hz = bin * fs / nFFT
};
```

### FFTProcessorParams

```cpp
struct FFTProcessorParams {
    uint32_t beam_count = 1;          // Количество лучей
    uint32_t n_point = 0;             // Входных точек на луч
    float sample_rate = 1000.0f;      // Частота дискретизации (Hz)
    FFTOutputMode output_mode = FFTOutputMode::COMPLEX;
    uint32_t repeat_count = 1;        // Множитель nFFT (1 = nextPow2(n_point))
    float memory_limit = 0.80f;       // Лимит GPU памяти для batch (0.0-1.0)
};
```

### FFTBeamResult (base)

```cpp
struct FFTBeamResult {
    uint32_t beam_id = 0;
    uint32_t nFFT = 0;
    float sample_rate = 0.0f;
};
```

### FFTComplexResult

```cpp
struct FFTComplexResult : FFTBeamResult {
    std::vector<std::complex<float>> spectrum;  // nFFT комплексных значений
};
```

### FFTMagPhaseResult

```cpp
struct FFTMagPhaseResult : FFTBeamResult {
    std::vector<float> magnitude;    // nFFT магнитуд
    std::vector<float> phase;        // nFFT фаз (радианы)
    std::vector<float> frequency;    // nFFT частот (Hz), только MAGNITUDE_PHASE_FREQ
};
```

### FFTProfilingData

```cpp
struct FFTProfilingData {
    double upload_time_ms = 0.0;
    double fft_time_ms = 0.0;
    double post_processing_time_ms = 0.0;
    double download_time_ms = 0.0;
    double total_time_ms = 0.0;
};
```

---

## FFTProcessor

**Файл**: `include/fft_processor.hpp`

### Конструктор

```cpp
explicit FFTProcessor(IBackend* backend);
```

### Complex output

```cpp
// CPU данные
std::vector<FFTComplexResult> ProcessComplex(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params);

// GPU данные (cl_mem)
std::vector<FFTComplexResult> ProcessComplex(
    cl_mem gpu_data,
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes = 0);
```

### Magnitude + Phase output

```cpp
// CPU данные
std::vector<FFTMagPhaseResult> ProcessMagPhase(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params);

// GPU данные (cl_mem)
std::vector<FFTMagPhaseResult> ProcessMagPhase(
    cl_mem gpu_data,
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes = 0);
```

### Информация

```cpp
const FFTProfilingData& GetProfilingData() const;
uint32_t GetNFFT() const;
```

---

## Важные замечания

### FFT Plan Caching

FFTProcessor кеширует clFFT plan для текущего размера данных. **При изменении `n_point` необходимо создать новый экземпляр FFTProcessor**.

```cpp
// ПРАВИЛЬНО: разные экземпляры для разных размеров
FFTProcessor fft_4096(backend);  // для n_point = 4096
FFTProcessor fft_8192(backend);  // для n_point = 8192

// НЕПРАВИЛЬНО: один экземпляр для разных размеров
FFTProcessor fft(backend);
fft.ProcessComplex(data_4096, params_4096);  // OK
fft.ProcessComplex(data_8192, params_8192);  // ОШИБКА! CL_INVALID_VALUE
```

### GPU Memory Ownership

- CPU данные (`vector<complex<float>>`) копируются на GPU внутри ProcessXxx
- GPU данные (`cl_mem`) используются напрямую, ownership не передаётся
- Результаты всегда возвращаются на CPU

### nFFT вычисление

```
nFFT = nextPowerOf2(n_point) * repeat_count
```

Пример: n_point=1000, repeat_count=2 → nFFT = 1024 * 2 = 2048

---

## Примеры

### Complex FFT

```cpp
FFTProcessor fft(backend);

FFTProcessorParams params;
params.beam_count = 4;
params.n_point = 1024;
params.sample_rate = 44100.0f;
params.output_mode = FFTOutputMode::COMPLEX;

auto results = fft.ProcessComplex(data, params);
for (auto& r : results) {
    // r.spectrum[i] — complex<float>, i = 0..nFFT-1
    auto peak = std::max_element(r.spectrum.begin(), r.spectrum.end(),
        [](auto& a, auto& b) { return std::abs(a) < std::abs(b); });
}
```

### Magnitude + Phase + Frequency

```cpp
params.output_mode = FFTOutputMode::MAGNITUDE_PHASE_FREQ;

auto results = fft.ProcessMagPhase(data, params);
for (auto& r : results) {
    // r.magnitude[i], r.phase[i], r.frequency[i]
    auto peak_idx = std::distance(r.magnitude.begin(),
        std::max_element(r.magnitude.begin(), r.magnitude.end()));
    printf("Peak: %.1f Hz, mag=%.2f\n", r.frequency[peak_idx], r.magnitude[peak_idx]);
}
```

### Pipeline: SignalGenerator → FFTProcessor

```cpp
// Генерация сигнала
signal_gen::SignalService service(backend);
cl_mem signal = service.GenerateGpu(cw_params, {1000.0, 4096}, 16);

// FFT
FFTProcessor fft(backend);
FFTProcessorParams params{.beam_count = 16, .n_point = 4096, .sample_rate = 1000.0f};
auto results = fft.ProcessComplex(signal, params, 16 * 4096 * sizeof(complex<float>));

clReleaseMemObject(signal);
```

---

*Обновлено: 2026-02-13*
