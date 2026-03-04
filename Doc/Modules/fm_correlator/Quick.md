# FM Correlator — Краткий справочник

> Корреляционная обработка ФМ-сигналов с M-последовательностями в частотной области (ROCm-only)

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
Система посылает зондирующий сигнал с псевдослучайной FM-модуляцией (M-последовательностью). Принятый отражённый сигнал задержан — на величину, пропорциональную расстоянию до цели. Модуль находит эту задержку путём корреляции принятого сигнала с циклически сдвинутыми копиями эталона.

---

### Что такое M-последовательность

M-последовательность (максимальная линейная последовательность) — псевдослучайная бинарная последовательность длиной N=2^31-1, генерируемая через сдвиговый регистр с обратной связью (LFSR). Ключевое свойство: её автокорреляция — острый пик при нулевом сдвиге и почти ноль везде. Это делает её идеальным зондирующим сигналом: точно видно на каком сдвиге "попало".

---

### Что делает модуль (без формул)

1. **Генерирует M-последовательность** (один раз через LFSR) — это эталонный сигнал.
2. **Делает K циклических сдвигов** эталона и для каждого сдвига вычисляет FFT.
3. **Принимает S входных сигналов** и делает их FFT (вещественное R2C — экономия памяти).
4. **Перемножает спектры**: для каждой пары (сигнал, сдвиг) — сопряжённый спектр эталона × спектр сигнала.
5. **Обратное FFT (IFFT)** каждого произведения — это и есть корреляция.
6. **Берёт первые n_kg точек** — это временные задержки от 0 до n_kg/fs.

Где стоит пик в выходном массиве — там и задержка (расстояние до цели).

---

### Зачем нужны K сдвигов

Один сдвиг — одна гипотеза о задержке сигнала. K сдвигов — K параллельных гипотез, покрывающих разные дальности. При обнаружении — смотрим какой из K сдвигов дал наибольший пик.

---

### Частотная область vs временная

Корреляция во временной области — это свёртка через цикл. Долго. В частотной области — просто перемножение двух FFT и обратное FFT. GPU делает все S×K пары параллельно. Это в разы быстрее.

---

### ROCm-only, два потока

Модуль использует два HIP-потока: в первом параллельно обрабатывается загрузка и FFT входных сигналов, во втором — FFT эталонных сдвигов. Синхронизируются перед финальным умножением.

---

## Алгоритм

```
ref_shifted[k] = circshift(ref, k)
corr(s,k) = IFFT{ conj(FFT{ref_shifted[k]}) · R2C_FFT{inp[s]} } / N
peaks[s,k,j] = |corr[j]|,  j = 0..n_kg-1
```

> R2C + C2R экономят 50% памяти vs C2C. hipFFT не нормирует IFFT → делим на N.

---

## Быстрый старт

### C++ — ROCm

```cpp
#include "fm_correlator.hpp"

drv_gpu_lib::FMCorrelator corr(backend);
corr.SetParams({.fft_size=32768, .num_shifts=32, .num_signals=5,
                .num_output_points=2000});

corr.PrepareReference();                   // LFSR M-seq + upload (один раз)
auto result = corr.Process(input_signals); // input_signals: flat [S × N] float
// result.at(signal, shift, point) -> float
```

### C++ — тестовый паттерн (данные не покидают GPU)

```cpp
corr.PrepareReference();
auto result = corr.RunTestPattern(/*shift_step=*/2);
// Пик (s, k) в позиции (s*2 - k) mod N
```

### Python

```python
import gpuworklib

ctx  = gpuworklib.ROCmGPUContext(0)
corr = gpuworklib.FMCorrelatorROCm(ctx)
corr.set_params(fft_size=32768, num_shifts=32, num_signals=5)
corr.prepare_reference()

# Вариант 1: тестовый паттерн
peaks = corr.run_test_pattern(shift_step=2)  # numpy [S, K, n_kg]

# Вариант 2: внешние данные
import numpy as np
ref     = corr.generate_msequence(seed=1)
signals = np.stack([np.roll(ref, -s*2) for s in range(5)])
corr.prepare_reference_from_data(ref)
peaks   = corr.process(signals.astype(np.float32))  # [S, K, n_kg]
```

---

## Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `fft_size` | size_t | 32768 (2^15) | Размер FFT, степень 2 |
| `num_shifts` | int | 32 | K — циклических сдвигов ref |
| `num_signals` | int | 5 | S — входных сигналов |
| `num_output_points` | int | 2000 | n_kg — первых точек из IFFT |
| `lfsr_polynomial` | uint32_t | 0xB8000000 | Полином LFSR (степень 31) |
| `lfsr_seed` | uint32_t | 0x1 | Начальное состояние LFSR |

---

## Размерности буферов (N=32768, K=32, S=5)

| Буфер | Размер | Тип |
|-------|--------|-----|
| `d_ref_complex/fft` | K × N | float2 (in-place) |
| `d_inp_float` | S × N | float |
| `d_inp_fft` | S × **(N/2+1)** | float2 (hermitian!) |
| `d_corr_fft` | S × K × **(N/2+1)** | float2 |
| `d_corr_time` | S × K × N | **float** (C2R) |
| `d_peaks` | S × K × n_kg | float |

---

## Профилирование

```cpp
auto& profiler = backend->GetProfiler();
profiler.SetGPUInfo(backend->GetDeviceName(), backend->GetDriverVersion());
profiler.Start("FM_Correlator_Process");
corr.Process(inp);
profiler.Stop("FM_Correlator_Process");
profiler.PrintReport();
profiler.ExportMarkdown("Results/Profiler/fm_correlator_YYYY-MM-DD.md");
```

---

## Тесты

| Файл | Тест | Что проверяет |
|------|------|---------------|
| `tests/test_fm_msequence.hpp` | LFSR генератор | Первые 10 бит известной M-seq |
| `tests/test_fm_basic.hpp` | Автокорреляция | SNR > 10 в j=0 |
| `tests/test_fm_basic.hpp` | Сдвиговый паттерн | Пик (s,k) в ожидаемой позиции, CPU vs GPU atol=1e-4 |
| `tests/test_fm_benchmark_rocm.hpp` | Бенчмарк | warmup 3 + 20 runs + hipEvent |
| `Python_test/fm_correlator/test_fm_correlator_rocm.py` | Python: autocorr | SNR > 10 |
| `Python_test/fm_correlator/test_fm_correlator_rocm.py` | Python: shift_pattern | argmax == expected_pos |
| `Python_test/fm_correlator/test_fm_correlator_rocm.py` | Python: cpu_vs_gpu | atol=1e-4 |

---

## Частые ошибки

| Ошибка | Решение |
|--------|---------|
| R2C даёт N/2+1 точек, код ожидает N | `half_N = N/2+1` везде для спектральных буферов |
| IFFT результат не нормализован | Делить на N в `extract_magnitudes` |
| Plan создаётся в каждом Process() | Создавать при `SetParams()`, хранить постоянно |
| `hipfftSetStream` не вызван | Вызывать после `hipfftPlanMany` для каждого плана |
| Забыт sync перед Step3 | `hipStreamSynchronize` обоих потоков перед multiply |
| S не помещается в GPU | BatchManager разобьёт по лучам автоматически |

---

## Ссылки

- [Full.md](Full.md) — математика, pipeline, kernels, Python bindings, бенчмарк
- [ROCm/HIP Optimization Guide](../../Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md)
- [hipFFT API](https://rocm.docs.amd.com/projects/hipFFT/en/latest/)

---

*Обновлено: 2026-03-04*
