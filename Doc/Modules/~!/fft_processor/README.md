# FFT Processor Module

> GPU FFT с вариантами вывода (Complex / MagPhase / MagPhaseFreq)

**Namespace**: `fft_processor`
**Каталог**: `modules/fft_processor/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL, clFFT

---

## Содержание

| Файл | Описание |
|------|----------|
| [API.md](API.md) | Полный API Reference |

---

## Обзор

`FFTProcessor` — модуль GPU FFT обработки, отдельный от `SpectrumMaximaFinder`:

| Модуль | Назначение |
|--------|------------|
| **FFTProcessor** | Возвращает **полный спектр** в нужном формате |
| **SpectrumMaximaFinder** | Ищет **максимумы** спектра (пики) |

### Режимы вывода (FFTOutputMode)

| Режим | Описание | Данные |
|-------|----------|--------|
| `COMPLEX` | Сырой комплексный спектр | `complex<float>[nFFT]` |
| `MAGNITUDE_PHASE` | Магнитуда + фаза | `float[nFFT]` + `float[nFFT]` |
| `MAGNITUDE_PHASE_FREQ` | Магнитуда + фаза + частота | + `float[nFFT]` (Hz) |

---

## Быстрый старт

### C++

```cpp
#include "fft_processor.hpp"

fft_processor::FFTProcessor fft(backend);

fft_processor::FFTProcessorParams params;
params.beam_count = 16;
params.n_point = 4096;
params.sample_rate = 1000.0f;
params.output_mode = fft_processor::FFTOutputMode::COMPLEX;

// CPU данные
std::vector<std::complex<float>> data = /* ... */;
auto results = fft.ProcessComplex(data, params);

for (auto& r : results) {
    // r.beam_id, r.nFFT, r.spectrum[nFFT]
}
```

### Python

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
fft = gpuworklib.FFTProcessor(ctx)

# signal: numpy array (beam_count, n_point), dtype=complex64
spectrum = fft.process(signal, sample_rate=1000.0, output_mode='complex')
# spectrum.shape == signal.shape (complex64)
```

---

## Архитектура

```
FFTProcessor
├── clFFT (clfftPlanHandle)    — FFT plan с pre-callback padding
├── Pre-callback userdata      — [32B header][input data] → zero-pad до nFFT
├── Post-processing kernel     — complex → magnitude + phase (OpenCL kernel)
├── Batch processing           — автоматическое разбиение по памяти GPU
└── Profiling                  — upload/fft/post/download timing (cl_event)
```

### Pipeline данных

```
CPU/GPU Input
    │
    ▼
Upload → [pre_callback_userdata] → zero-pad → [fft_input]
    │
    ▼
clFFT Execute → [fft_output]
    │
    ├── COMPLEX → ReadBack → FFTComplexResult
    │
    └── MAGNITUDE_PHASE → MagPhase Kernel → [mag_output, phase_output]
                              │
                              └── ReadBack → FFTMagPhaseResult
```

---

## Ключевые особенности

- **nFFT**: автоматически вычисляется как `nextPow2(n_point) * repeat_count`
- **Pre-callback**: zero-padding входных данных до nFFT в kernel-е
- **Batch processing**: при нехватке GPU памяти автоматически разбивает на batch'и
- **Dual input**: принимает CPU `vector<complex<float>>` или GPU `cl_mem`
- **Profiling**: детальное время каждого этапа через OpenCL events

---

## Файлы

```
modules/fft_processor/
├── include/
│   ├── fft_processor.hpp              # FFTProcessor class
│   ├── fft_processor_types.hpp        # FFTOutputMode, params, results
│   ├── kernels/
│   │   └── fft_processor_kernels.hpp  # OpenCL kernel sources
│   └── services/
│       └── batch_manager.hpp          # Batch processing
├── src/
│   └── fft_processor.cpp
└── CMakeLists.txt
```

---

*Обновлено: 2026-02-13*
