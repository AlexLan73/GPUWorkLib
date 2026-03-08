# 🔍 Алгоритм поиска регрессий в ROCm-модулях GPUWorkLib

> **Контекст**: Создан после регрессии в `fft_maxima` OpenCL (commit `47edd7b`, 2026-03-07).
> **Проверено**: Статический анализ ROCm кода — явных дефектов не найдено.
> **Назначение**: Чеклист при следующей оптимизации kernels. Запускать на Linux + AMD GPU.

---

## 📌 Что сломалось в fft_maxima OpenCL (урок)

Две независимые ошибки после "безопасной" оптимизации:

| # | Ошибка | Симптом | Файл |
|---|--------|---------|------|
| 1 | Kernel переведён на 2D NDRange, хост остался 1D | Multi-beam: только beam 0 работает | `spectrum_maxima_finder_all_maxima.cpp` |
| 2 | `/ %` → bitwise `>> &` в clFFT callback | ONE_PEAK/TWO_PEAKS дают неверные результаты | `fft_kernel_sources.hpp` |

---

## 🚦 Алгоритм проверки ROCm-модуля после оптимизации

### Шаг 1: Статический анализ (можно без GPU)

```bash
# 1a. Найти все 2D HIP запуски (grid_y > 1)
grep -rn "grid_y\|gridDim.y\|blockIdx.y" modules/ --include="*.hpp" --include="*.cpp" --include="*.cl"

# 1b. Найти bitwise-оптимизации в индексации
grep -rn ">> log2\|>> nFFT\|& (n.*- 1)\|& nFFT\|>> beam" modules/ --include="*.hpp" --include="*.cpp"

# 1c. Найти hipModuleLaunchKernel и dim3 вызовы — сравнить с kernel
grep -rn "hipModuleLaunchKernel\|hipLaunchKernel\|dim3" modules/ --include="*.cpp"
```

### Шаг 2: Матрица kernel/host dimensions (заполнить перед тестом)

Для каждого ROCm kernel проверить соответствие:

| Kernel | Kernel dimensions | Host launch dims | Статус |
|--------|------------------|-----------------|--------|
| `detect_all_maxima` (fft_maxima) | 2D: x=nFFT, y=beam | 2D: `{nFFT, beam_count}` | ✅ |
| `prefix_sum_scan` (fft_maxima) | 1D | 1D | ✅ |
| `compact_maxima` (fft_maxima) | 1D | 1D | ✅ |
| `post_kernel` (fft_maxima) | 1D: groups=beam_count | 1D: `beam_count` blocks | ✅ |
| `dechirp_multiply` (heterodyne) | 2D: x=samples, y=antennas | 2D: `(samples, antennas, 1)` | ✅ |
| `fir_filter_cf32` (filters) | 2D: x=points, y=channels | 2D: `(grid_x, channels, 1)` | ✅ |
| `iir_biquad_cascade_cf32` (filters) | 1D: channels | 1D: `grid(channels)` | ✅ |
| `welford_fused` (statistics) | 1D | 1D | ✅ |
| `extract_medians` (statistics) | 1D | 1D | ✅ |

### Шаг 3: Тест на Linux + AMD GPU (обязательно)

```bash
cd build && cmake .. -DENABLE_ROCM=ON
cmake --build . --target GPUWorkLib -j8

# Порядок запуска (от базового к сложному):
./GPUWorkLib drvgpu        # базовый ROCm backend
./GPUWorkLib fft_processor # hipFFT base
./GPUWorkLib fft_maxima    # ROCm spectrum + AllMaxima
./GPUWorkLib filters       # FIR, IIR, Kaufman, MA, Kalman
./GPUWorkLib heterodyne    # dechirp LFM
./GPUWorkLib statistics    # welford, medians
```

### Шаг 4: Проверочные Python тесты на ROCm

```bash
python Python_test/fft_maxima/test_spectrum_find_all_maxima_rocm.py
python Python_test/filters/test_fir_filter_rocm.py
python Python_test/filters/test_iir_filter_rocm.py
python Python_test/heterodyne/test_heterodyne_rocm.py
python Python_test/statistics/test_statistics_rocm.py
```

---

## 📋 Рекомендации по модулям

### fft_maxima

**OpenCL** — исправлено (2026-03-07):
- ✅ `detect_all_maxima`: запуск 2D `{nFFT, beam_count}` — синхронизирован с kernel
- ✅ pre-callback: возвращён `/ %` вместо `>> &`

**ROCm** — статический анализ чистый, но при тестировании проверить:
- `all_maxima_pipeline_rocm.cpp`: `detect_all_maxima_beam` запускается 2D — убедиться что оба dim передаются
- `spectrum_processor_rocm.cpp`: hipFFT план создаётся корректно под `nFFT`
- **Эталонные значения**: сравнить с OpenCL результатами (должны совпадать с точностью float32)

```
Тест-минимум:
  - Single-beam: 1 тон → находится 1 максимум
  - Multi-beam: 5 лучей с разными freq → каждый луч свой максимум
  - CPU vs GPU: результаты AllMaxima совпадают побитово (не float!)
  - FullPipeline CPU data → hipFFT → peaks: частота в допуске ±0.5 бина
```

---

### fft_processor

**OpenCL** — тесты отключены (clFFT не работает на RDNA4+ gfx1201).

**ROCm** — hipFFT активен, проверить после оптимизации:
- `fft_processor_kernels_rocm.hpp`: `pad_data_kernel` использует 2D launch (x=nFFT, y=batch) — ✅ корректно
- `complex_to_mag_phase_rocm.cpp`: 1D launch с batch * nFFT элементов — ✅ корректно
- При оптимизации `fft_processor_kernels.cl`: не менять размерность IFFT batch без обновления хоста

```
Тест-минимум:
  - FFT сигнала с известной частотой → пик в нужном бине (±1)
  - IFFT(FFT(x)) ≈ x (roundtrip с точностью 1e-5)
  - Multi-batch (256 × 4096): все батчи независимы
```

---

### filters

**OpenCL** — рабочий, тесты проходят.

**ROCm** — в commit `47edd7b` значительно изменены:
- `fir_kernels_rocm.hpp`: вероятна оптимизация inner loop — проверить coeffs alignment
- `iir_kernels_rocm.hpp`: biquad cascade — проверить per-channel state корректность
- `kaufman_kernels_rocm.hpp`: 44+ строк изменено — проверить adaptive EMA корректность

**Алгоритм проверки FIR**:
```python
# ETA: GPU FIR должен совпасть с scipy.signal.lfilter
import scipy.signal
b = [0.1, 0.2, 0.4, 0.2, 0.1]  # FIR coeffs
y_scipy = scipy.signal.lfilter(b, [1.0], x)
y_gpu = fir_filter_gpu(x, b)
assert max(abs(y_scipy - y_gpu)) < 1e-4, "FIR regression!"
```

**Алгоритм проверки IIR**:
```python
# IIR biquad: первый и последний отсчёты должны совпасть
y_scipy = scipy.signal.sosfilt(sos, x)
y_gpu   = iir_filter_gpu(x, sos)
assert max(abs(y_scipy - y_gpu)) < 1e-4, "IIR regression!"
```

---

### heterodyne

**OpenCL + ROCm** — оба активны. В `47edd7b` изменено:
- `heterodyne_kernels_rocm.hpp`: 94 строки изменений
- `heterodyne_processor_rocm.hpp`: изменена структура processора

Kernel `dechirp_multiply` использует 2D: x=samples, y=antennas — ✅ хост запускает 2D корректно.

**Проверить после оптимизации**:
- Dechirp для одного луча: аналитический результат `f_beat = f1 - f0`
- Dechirp для N лучей: каждый луч дает свой `f_beat`
- `DechirpFromGPU`: передача cl_mem не теряет данные

```
Эталонный тест:
  LFM от 2.0 до 3.0 Hz за 100000 точек, fs=1000 Hz
  После dechirp: пик на бите ~0 (нулевой биение)
  Допуск: ±1 бин
```

---

### statistics

**ROCm only**. В `47edd7b` изменено 143 строки в `statistics_kernels_rocm.hpp`.

Проверить:
- `welford_fused`: mean/variance для нормального распределения N(0,1) → mean≈0, var≈1
- `extract_medians`: медиана массива [1,2,3,4,5] = 3.0
- Radix sort: массив случайных чисел → отсортирован

```
Тест-минимум:
  N = 10000, беамов = 64
  mean должен быть в [-0.1, 0.1] (для N(0,1))
  std  должен быть в [0.95, 1.05]
```

---

### signal_generators

**OpenCL + ROCm**. В `47edd7b` изменены `lfm_generator.cpp` и `lfm_generator_analytical_delay.cpp`.

**Проверить**:
- CW генератор: частота пика в FFT совпадает с заданной
- LFM генератор: sweep rate правильный (проверить через autocorrelation width)
- LFM analytical delay: задержка в семплах = заданной (проверить cross-correlation)

---

## ⚠️ Общие правила при оптимизации kernels

### Запрещённые оптимизации (без валидации):

```cpp
// ❌ НЕ делать: bitwise вместо div/mod в callback/index
beam_idx   = inoffset >> log2_nFFT;   // ОПАСНО!
pos_in_fft = inoffset & (nFFT - 1);  // ОПАСНО для callbacks!

// ✅ Безопасно:
beam_idx   = inoffset / nFFT;
pos_in_fft = inoffset % nFFT;
```

```cpp
// ❌ НЕ делать: менять NDRange без синхронизации с хостом
// Kernel:
const int beam = blockIdx.y;  // Kernel стал 2D

// Host (СТАРЫЙ!):
hipLaunchKernel(kernel, dim3(N), dim3(256), ...);  // 1D — СЛОМАЕТ multi-beam!

// ✅ Синхронизировать:
hipLaunchKernel(kernel, dim3(grid_x, beam_count), dim3(256, 1), ...);
```

### Чеклист перед коммитом оптимизации:

```
[ ] Каждый изменённый kernel: проверить kernel dims == host launch dims
[ ] Callback функции (clFFT, hipFFT pre/post): НЕ трогать div/mod без теста
[ ] Bitwise >> / & для индексов: только если nFFT гарантированно pow2 И есть проверка
[ ] После оптимизации: single-beam test + multi-beam test
[ ] После оптимизации: CPU reference comparison (не просто "не упало")
[ ] Если изменена структура userdata: обновить sizeof() в хосте
```

---

## 🛠️ Скрипт быстрой проверки (Windows, OpenCL)

```powershell
# Запустить из e:\C++\GPUWorkLib\build\Debug\
.\GPUWorkLib.exe drvgpu
.\GPUWorkLib.exe fft_processor
.\GPUWorkLib.exe fft_maxima

# Python тесты:
& "F:\Program Files (x86)\Python312\python.exe" Python_test\fft_maxima\test_spectrum_find_all_maxima.py
& "F:\Program Files (x86)\Python312\python.exe" Python_test\fft_maxima\test_find_all_maxima_maxvalue.py
```

---

## 🐧 Скрипт быстрой проверки (Linux, ROCm)

```bash
cd /path/to/GPUWorkLib/build

# C++ тесты всех ROCm-зависимых модулей
./GPUWorkLib drvgpu
./GPUWorkLib fft_processor
./GPUWorkLib fft_maxima
./GPUWorkLib filters
./GPUWorkLib heterodyne
./GPUWorkLib statistics

# Python тесты ROCm
python3 Python_test/fft_maxima/test_spectrum_find_all_maxima_rocm.py
python3 Python_test/filters/test_fir_filter_rocm.py
python3 Python_test/filters/test_iir_filter_rocm.py
python3 Python_test/heterodyne/test_heterodyne_rocm.py
python3 Python_test/statistics/test_statistics_rocm.py
```

---

## 📊 Статус проверки после commit 47edd7b

| Платформа | Модуль | C++ | Python | Дата |
|-----------|--------|-----|--------|------|
| Windows/OpenCL | DrvGPU | ✅ 3/3 | — | 2026-03-08 |
| Windows/OpenCL | fft_processor | ✅ (OpenCL tests disabled on gfx1201) | — | 2026-03-08 |
| Windows/OpenCL | fft_maxima | ✅ 7/7 | ✅ 5/5 + 2/2 | 2026-03-08 |
| Linux/ROCm | DrvGPU | ❓ не тестировалось | — | — |
| Linux/ROCm | fft_maxima | ❓ не тестировалось | ❓ skip | — |
| Linux/ROCm | filters | ❓ не тестировалось | ❓ не тестировалось | — |
| Linux/ROCm | heterodyne | ❓ не тестировалось | ❓ не тестировалось | — |
| Linux/ROCm | statistics | ❓ не тестировалось | ❓ не тестировалось | — |

> **Примечание**: Статический анализ ROCm кода чистый — паттерны OpenCL регрессии не найдены.
> Финальная верификация возможна только на Linux + AMD GPU (gfx1201 или совместимый).

---

*Создан: 2026-03-08*
*Автор: Кодо (AI Assistant)*
*Поводок: fft_maxima OpenCL regression fix + commit 47edd7b review*
