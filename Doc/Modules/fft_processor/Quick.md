# FFT Processor — Краткий справочник

> GPU FFT с вариантами вывода (Complex / MagPhase / MagPhaseFreq)
> OpenCL backend: `FFTProcessor` (clFFT) | ROCm backend: `FFTProcessorROCm` (hipFFT)

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
FFT (быстрое преобразование Фурье) переводит сигнал из временной области в частотную. На выходе видно — на каких частотах есть энергия, а на каких нет. Это фундаментальная операция в ЦОС: детектирование сигналов, измерение частоты, дечирп в радаре, спектральный анализ.

Модуль делает FFT сразу для N лучей (batch) на GPU — всё параллельно за один вызов.

---

### Три варианта что получить на выходе

**COMPLEX** — сырой спектр, комплексные числа. Используй когда дальше нужна обработка спектра (перемножение, IFFT, корреляция).

**MAGNITUDE_PHASE** — амплитуда и фаза каждой гармоники. Используй когда нужно видеть «силу» каждой частоты и её фазу.

**MAGNITUDE_PHASE_FREQ** — то же плюс массив частот в Hz. Удобно для графиков и вывода — не надо считать частоту руками.

---

### FFTProcessor vs FFTProcessorROCm

**FFTProcessor (OpenCL / clFFT)** — работает на любом GPU через OpenCL. Но на AMD RDNA4+ (gfx1201 и новее) clFFT не поддерживается — используй ROCm вариант.

**FFTProcessorROCm (ROCm / hipFFT)** — только AMD GPU на Linux. Быстрее на современных AMD картах. Умеет хранить два плана в кеше — при переключении между двумя разными размерами FFT не пересоздаёт план.

**ComplexToMagPhaseROCm** — это не FFT! Просто переводит массив комплексных чисел в (|z|, arg(z)). Нужно когда FFT уже сделан, и надо только извлечь амплитуду/фазу.

---

### Важная деталь: размер FFT

Внутри FFT размер всегда округляется вверх до ближайшей степени двойки (`nextPow2`). Параметр `repeat_count` умножает этот размер — например, если хочешь zero-padding для лучшего частотного разрешения.

---

## Классы

| Класс | Backend | Назначение |
|-------|---------|------------|
| `FFTProcessor` | OpenCL/clFFT | FFT → полный спектр |
| `FFTProcessorROCm` | ROCm/hipFFT | То же (AMD gfx1201+) |
| `ComplexToMagPhaseROCm` | ROCm | `|z|` + `arg(z)` **без FFT** |

> ⚠️ На AMD (RDNA4+) используй `FFTProcessorROCm` — clFFT не работает на gfx1201+

---

## Режимы вывода (FFTOutputMode)

| Режим | Поля результата |
|-------|-----------------|
| `COMPLEX` | `spectrum[]` (complex\<float\>[nFFT]) |
| `MAGNITUDE_PHASE` | `magnitude[]` + `phase[]` |
| `MAGNITUDE_PHASE_FREQ` | + `frequency[]` (Hz) |

---

## Быстрый старт

### C++ — OpenCL

```cpp
#include "fft_processor.hpp"

fft_processor::FFTProcessor fft(backend);
fft_processor::FFTProcessorParams params;
params.beam_count  = 16;
params.n_point     = 4096;
params.sample_rate = 1000.0f;
params.output_mode = fft_processor::FFTOutputMode::COMPLEX;

auto results = fft.ProcessComplex(data, params);
// results[i].spectrum[], .nFFT, .beam_id
```

### C++ — ROCm/hipFFT

```cpp
#include "fft_processor_rocm.hpp"

fft_processor::FFTProcessorROCm fft(backend);
fft_processor::FFTProcessorParams params;
params.beam_count  = 64;
params.n_point     = 1024;
params.sample_rate = 1e6f;

auto results = fft.ProcessComplex(data, params);

// С детальным профилированием
fft_processor::ROCmProfEvents events;
fft.ProcessComplex(data, params, &events);
// events["Upload"].end_ns - events["Upload"].start_ns
```

### C++ — ComplexToMagPhaseROCm (без FFT)

```cpp
#include "complex_to_mag_phase_rocm.hpp"

fft_processor::ComplexToMagPhaseROCm converter(backend);
fft_processor::MagPhaseParams params;
params.beam_count = 4;
params.n_point    = 2048;

// CPU → CPU
auto results = converter.Process(data, params);

// CPU → GPU (для GPU pipeline, без download)
void* gpu_out = converter.ProcessToGPU(data, params);
// Обязательно: backend.Free(gpu_out)
```

### Python

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
fft = gpuworklib.FFTProcessor(ctx)

results = fft.process_complex(signal, beam_count=8, n_point=1024, sample_rate=50000.0)
results = fft.process_mag_phase(signal, 8, 1024, 50000.0, include_freq=True)
# results[i].magnitude, .phase, .frequency
```

---

## Параметры

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `beam_count` | uint32_t | 1 | Количество лучей |
| `n_point` | uint32_t | — | Входных точек на луч |
| `sample_rate` | float | 1000.0f | Hz |
| `output_mode` | FFTOutputMode | COMPLEX | Формат вывода |
| `repeat_count` | uint32_t | 1 | `nFFT = nextPow2(n_point) × repeat_count` |

---

## Важные нюансы

1. **Plan cache**: OpenCL — 1 план, при смене `n_point` создавай новый экземпляр. ROCm — two-plan cache, автоматически переключается между 2 размерами.
2. **HSACO disk cache**: первый запуск — JIT (~100-500ms), далее — загрузка из `kernels/bin/` (~1ms).
3. **ProcessToGPU** возвращает `void*` — **caller owner**, нужно `backend.Free(ptr)`.
4. **GPU-input overload**: `Process(void* gpu_ptr, params, byte_size)` — пропускает upload.

---

*Обновлено: 2026-03-02*
