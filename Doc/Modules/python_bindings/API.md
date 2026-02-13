# Python Bindings — API Reference

**Module**: `gpuworklib`

---

## gpuworklib.GPUContext

OpenCL GPU контекст.

### Constructor

```python
ctx = gpuworklib.GPUContext(device_index: int = 0)
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| `device_index` | `int` | Индекс GPU устройства (0-based) |

### Properties

| Свойство | Тип | Описание |
|----------|-----|----------|
| `device_name` | `str` | Название GPU |
| `device_index` | `int` | Индекс устройства |
| `global_memory_mb` | `float` | Глобальная память (MB) |

---

## gpuworklib.SignalGenerator

Генерация сигналов на GPU.

### Constructor

```python
gen = gpuworklib.SignalGenerator(ctx: GPUContext)
```

### Methods

#### generate_cw

CW сигнал: `s(t) = A * exp(j*(2*pi*f*t + phase))`

```python
data = gen.generate_cw(
    beam_count: int,       # Количество лучей
    n_point: int,          # Точек на луч
    sample_rate: float,    # Частота дискретизации (Hz)
    f0: float = 100.0,     # Частота (Hz)
    phase: float = 0.0,    # Начальная фаза (rad)
    amplitude: float = 1.0,# Амплитуда
    freq_step: float = 0.0 # Шаг частоты между лучами (Hz)
) -> numpy.ndarray  # shape=(beam_count, n_point), dtype=complex64
```

Для `beam_count=1` возвращает 1D массив `(n_point,)`.

Multi-beam: `freq_i = f0 + i * freq_step`

#### generate_lfm

LFM chirp: `s(t) = A * exp(j*pi*k*t^2 + j*2*pi*f_start*t)`

```python
data = gen.generate_lfm(
    beam_count: int,
    n_point: int,
    sample_rate: float,
    f_start: float = 100.0,  # Начальная частота (Hz)
    f_end: float = 500.0,    # Конечная частота (Hz)
    amplitude: float = 1.0
) -> numpy.ndarray  # shape=(beam_count, n_point), dtype=complex64
```

#### generate_noise

Шум: Gaussian / White (GPU: Philox-2x32 + Box-Muller).

```python
data = gen.generate_noise(
    beam_count: int,
    n_point: int,
    noise_type: str = 'gaussian',  # 'gaussian' или 'white'
    power: float = 1.0,            # Мощность (дисперсия)
    seed: int = 0                  # 0 = random
) -> numpy.ndarray  # shape=(beam_count, n_point), dtype=complex64
```

---

## gpuworklib.FFTProcessor

GPU FFT обработка (clFFT wrapper).

### Constructor

```python
fft = gpuworklib.FFTProcessor(ctx: GPUContext)
```

### Methods

#### process

```python
result = fft.process(
    data: numpy.ndarray,       # Input: (beam_count, n_point) complex64
    sample_rate: float = 1000.0,
    output_mode: str = 'complex'  # 'complex' | 'mag_phase' | 'mag_phase_freq'
)
```

**Возвращает** (зависит от `output_mode`):

| output_mode | Возвращает | Типы |
|-------------|-----------|------|
| `'complex'` | `spectrum` | `ndarray complex64` |
| `'mag_phase'` | `(magnitude, phase)` | `tuple(ndarray float32, ndarray float32)` |
| `'mag_phase_freq'` | `(magnitude, phase, frequency)` | `tuple(ndarray, ndarray, ndarray)` |

**Важно**: FFTProcessor кеширует план для текущего размера данных. Для данных другого размера создайте новый экземпляр.

---

## gpuworklib.ScriptGenerator

Text DSL → OpenCL kernel compiler.

### Constructor

```python
sg = gpuworklib.ScriptGenerator(ctx: GPUContext)
```

### Methods

#### load

Загрузить и скомпилировать скрипт из строки.

```python
sg.load(script_text: str)
```

Формат скрипта:
```
[Params]
ANTENNAS = N
POINTS = M

[Defs]
var = expression  (OpenCL C syntax)

[Signal]
res = expression       # real output
res_re = ...; res_im = ...  # complex IQ output
```

#### load_file

Загрузить из файла.

```python
sg.load_file(file_path: str)
```

#### generate

Генерация сигнала на GPU.

```python
data = sg.generate() -> numpy.ndarray
# shape=(antennas, points) для antennas > 1
# shape=(points,) для antennas == 1
# dtype=complex64
```

### Properties

| Свойство | Тип | Описание |
|----------|-----|----------|
| `antennas` | `int` | Количество антенн |
| `points` | `int` | Количество точек |
| `kernel_source` | `str` | Сгенерированный OpenCL kernel (для отладки) |
| `is_ready` | `bool` | Скрипт загружен и скомпилирован |

---

## Примеры

### Matplotlib визуализация

```python
import gpuworklib
import numpy as np
import matplotlib.pyplot as plt

ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.SignalGenerator(ctx)
fft = gpuworklib.FFTProcessor(ctx)

# Сигнал: 4 луча с разными частотами
signal = gen.generate_cw(4, 4096, 1000.0, f0=50.0, freq_step=50.0)

# FFT
mag, phase, freq = fft.process(signal, sample_rate=1000.0,
                                output_mode='mag_phase_freq')

# График
fig, axes = plt.subplots(4, 1, figsize=(10, 8))
for i in range(4):
    axes[i].plot(freq[i], mag[i])
    axes[i].set_title(f'Beam {i}: f = {50 + i*50} Hz')
    axes[i].set_xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()
```

### Benchmark

```python
import time
ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.SignalGenerator(ctx)

# Warm up
_ = gen.generate_cw(1, 1024, 1000.0)

# Benchmark
start = time.perf_counter()
data = gen.generate_cw(512, 16384, 1000.0, f0=100.0, freq_step=1.0)
elapsed = time.perf_counter() - start

samples = 512 * 16384
print(f"{samples:,} samples in {elapsed*1000:.1f} ms")
print(f"Throughput: {samples/elapsed/1e6:.0f} Msamples/s")
```

---

## Error Handling

Все ошибки OpenCL и FFT выбрасываются как `RuntimeError`:

```python
try:
    sg.load("[Signal]\nres = undefined_var")
except RuntimeError as e:
    print(f"Compilation error: {e}")
    # Содержит: generated kernel source + OpenCL build log
```

---

*Обновлено: 2026-02-13*
