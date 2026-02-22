# 📡 Алгоритм LFM Dechirp — полный цикл обработки

> **Дата**: 2026-02-21
> **Автор**: Кодо (AI Assistant)
> **Статус**: ✅ Реализовано, все 7 тестов PASSED
> **Модуль**: `modules/heterodyne/`

---

## 🎯 Задача

Stretch-processing (dechirp) ЛЧМ радарного сигнала:
- Принятый сигнал s_rx(t) = s_tx(t − τ) — задержанная копия передающего ЛЧМ
- Цель: найти частоту биений f_beat = μ·τ → вычислить дальность R
- SNR оценка из спектра: 20·log10(peak / noise_estimate)

---

## 📥 Входные данные

Пользователь задаёт `HeterodyneParams` (`heterodyne_params.hpp`):

| Параметр | Описание | Пример |
|----------|----------|--------|
| `f_start` | Начальная частота ЛЧМ [Hz] | 0 |
| `f_end` | Конечная частота ЛЧМ [Hz] | 2e6 |
| `sample_rate` | Частота дискретизации fs [Hz] | 12e6 |
| `num_samples` | Число отсчётов N на антенну | 8000 |
| `num_antennas` | Число каналов (антенн) | 5 |

Производные:
- B = f_end − f_start = 2 МГц (полоса)
- T = N / fs = 666.67 мкс (длительность чирпа)
- μ = B / T = 3·10⁹ Гц/с (скорость перестройки)

Входной сигнал: `rx_data` — плоский вектор `complex<float>[num_antennas × num_samples]`

---

## 🔧 Этап 1: Генерация сопряжённого опорного ЛЧМ

**Модуль**: `signal_generators` → `LfmConjugateGenerator`
**Ядро GPU**: `lfm_conjugate.cl`

Генерирует `ref[n] = conj(s_tx(t))` — один вектор на N точек, общий для всех антенн.

### Формула:
```
t = n / fs
phase = −(π · μ · t² + 2π · f_start · t)
ref[n] = A · (cos(phase) + j·sin(phase))
```

Это `exp(−j·(π·μ·t² + 2π·f₀·t))` — комплексно-сопряжённый передающий ЛЧМ.

### Код (OPT-4: кешированный генератор):
```cpp
// heterodyne_dechirp.cpp
EnsureConjugateGenerator();  // lazy-init, rebuild only on SetParams()
auto ref_cpu = conj_gen_->GenerateToCpu();  // vector<complex<float>>, размер N
```

---

## 🖥️ Этап 2: Дечирп-перемножение на GPU

**Файл**: `dechirp_multiply.cl`
**Запуск**: 1D kernel (OPT-5), `global = num_antennas × num_samples`

### Входы:
- `rx[antennas × N]` — принятый сигнал (все антенны плоско)
- `ref[N]` — опорный сопряжённый ЛЧМ (broadcast на все антенны)

### Ядро:
```c
int gid = get_global_id(0);
int n   = gid % num_samples;

rx_v = rx[gid];
re_v = ref[n];               // broadcast

// conj(rx · ref)
dc_out.x =  rx_v.x * re_v.x - rx_v.y * re_v.y;
dc_out.y = -(rx_v.x * re_v.y + rx_v.y * re_v.x);
```

### Почему conj от результата?

Прямое `s_rx · conj(s_tx)` даёт тон на **отрицательной** частоте `−μτ`.
Взяв conjugate от результата, получаем `conj(s_rx) · s_tx` — тон на **положительной** `+μτ`.

Это нужно чтобы:
- Пик FFT был в нижней половине спектра [0, N/2)
- Коррекция `exp(−j·2π·f·t)` корректно сдвигала к DC

### Математика:
```
s_rx(t) = exp(j·(π·μ·(t−τ)² + 2π·f₀·(t−τ)))     // задержанный ЛЧМ
s_tx(t) = exp(j·(π·μ·t² + 2π·f₀·t))               // передающий ЛЧМ

conj(s_rx) · s_tx:
  phase = −[π·μ·(t−τ)² + 2π·f₀·(t−τ)] + [π·μ·t² + 2π·f₀·t]
        = 2π·μ·τ·t − π·μ·τ² + 2π·f₀·τ
        = 2π·(μ·τ)·t + const

→ f_beat = μ·τ  (положительная!)
```

### GPU pipeline (OPT-1/2: кешированные ядра и буферы):
```cpp
// heterodyne_processor_opencl.cpp, Dechirp()
EnsureBuffers(params);  // OPT-2: allocate only when size changes
clEnqueueWriteBuffer(buf_rx_, ...);
clEnqueueWriteBuffer(buf_ref_, ...);
clEnqueueNDRangeKernel(kernel_multiply_, 1D, total);  // OPT-1: cached kernel
clEnqueueReadBuffer(buf_dc_, ...);
```

---

## 📊 Этап 3: FFT + поиск максимума (SpectrumMaximaFinder)

**Модуль**: `fft_maxima` → `SpectrumMaximaFinder`
**Файл**: `heterodyne_dechirp.cpp`, BuildResult()

Отдаём дечирпнутые данные в отлаженный модуль `SpectrumMaximaFinder`:

```cpp
SpectrumMaximaFinder finder(backend_);

InputData<vector<complex<float>>> input;
input.antenna_count = num_antennas;
input.n_point = num_samples;       // N=8000
input.data = dc_data;
input.repeat_count = 1;
input.sample_rate = fs;

auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

### Внутри SpectrumMaximaFinder (всё на GPU):
1. **Upload** — данные CPU → GPU
2. **FFT** (clFFT) — N=8000 → pad до nFFT=8192, FFT по каждой антенне
3. **PostKernel** — ищет один максимум в спектре [1, nFFT/2) с **параболической интерполяцией**:
   - Находит bin с max |spectrum[i]|
   - 3 точки: `y0 = |spec[bin−1]|, y1 = |spec[bin]|, y2 = |spec[bin+1]|`
   - `offset = 0.5 · (y0 − y2) / (y0 − 2·y1 + y2)`
   - `refined_frequency = (bin + offset) · fs / nFFT`

### Результат:
- `SpectrumResult.interpolated.refined_frequency` — частота биений [Гц]
- `SpectrumResult.interpolated.index` — номер бина
- `SpectrumResult.interpolated.magnitude` — амплитуда пика

---

## 📏 Этап 4: Расчёт дальности + SNR

**Формула дальности**:
```
R = (c · T · f_beat) / (2 · B)
```

Где:
- c = 3×10⁸ м/с — скорость света
- T = N / fs — длительность чирпа [с]
- B = f_end − f_start — полоса [Гц]
- f_beat — найденная частота биений [Гц]

**SNR** (из соседних точек спектра):
```cpp
float noise_est = (left_mag + right_mag) * 0.5f;
float snr_db = 20.0f * log10(peak_mag / noise_est);
```

**Код**: `HeterodyneResult::CalcRange()` и `BuildResult()` в `heterodyne_dechirp.cpp`

---

## 🔄 Этап 5 (опциональный): Коррекция частоты

**Ядро GPU**: `dechirp_correct.cl`
**Оптимизация OPT-6**: принимает `phase_step[]` (предвычислено на CPU)

Сдвигает пик к DC (для верификации):
```c
int gid = get_global_id(0);
int ant = gid / num_samples;
int n   = gid % num_samples;
float phase = phase_step[ant] * (float)n;
corrected = dc_in * (cos(phase), sin(phase));
```

Где `phase_step[ant] = -2·pi·f_beat[ant] / fs` (предвычислено на CPU, OPT-6).

После FFT corrected → пик должен быть на bin 0 (DC). Используется в Test 3.

---

## ⚡ Оптимизации (OPT-1..OPT-6)

| OPT | Описание | Где |
|-----|----------|-----|
| OPT-1 | Кеширование cl_kernel объектов | `heterodyne_processor_opencl.hpp` |
| OPT-2 | Кеширование GPU буферов (EnsureBuffers) | `heterodyne_processor_opencl.cpp` |
| OPT-3 | GPU ref без PCIe round-trip (ProcessExternal) | `heterodyne_dechirp.cpp` |
| OPT-4 | Кеширование LfmConjugateGenerator | `heterodyne_dechirp.cpp` |
| OPT-5 | 1D kernel вместо 2D | `dechirp_multiply.cl`, `dechirp_correct.cl` |
| OPT-6 | phase_step предвычислен на CPU | `dechirp_correct.cl` |

---

## 📐 Схема pipeline

```
rx_data (CPU, flat complex<float>[antennas × N])
    │
    ▼
┌─────────────────────────────────────┐
│ 1. LfmConjugateGenerator (GPU)     │  → ref = conj(s_tx), размер N
│    lfm_conjugate.cl                 │     signal_generators модуль
│    [OPT-4: кешируется]             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. dechirp_multiply.cl (GPU)       │  dc = conj(rx[gid] × ref[n])
│    1D kernel: global=ant*N (OPT-5) │  → тон на +f_beat = μ·τ
│    [OPT-1/2: cached kernel/buf]    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. SpectrumMaximaFinder (GPU)      │  FFT(clFFT, N→8192) + OnePeak
│    fft_maxima модуль                │  + параболическая интерполяция
│                                     │  → f_beat [Hz], bin, magnitude
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. CalcRange + SNR (CPU)            │  R = c·T·f_beat / (2·B)
│                                     │  SNR = 20·log10(peak/noise)
└─────────────────────────────────────┘
    │
    ▼
HeterodyneResult {f_beat_hz, range_m, peak_amplitude, peak_snr_db, ...}
```

---

## ✅ Результаты тестирования

### Параметры тестов
```
fs       = 12 МГц
B        = 2 МГц (f_start=0, f_end=2e6)
N        = 8000 точек
T        = 666.67 мкс
μ        = 3·10⁹ Гц/с
antennas = 5
delays   = [100, 200, 300, 400, 500] мкс
F_BEAT_TOL = 5000 Гц
```

### Все тесты (7/7 PASSED)

| # | Тест | Файл | Результат |
|---|------|------|-----------|
| 1 | Single antenna dechirp (delay=100мкс) | basic.hpp | ✅ PASSED |
| 2 | 5 antennas, linear delays [100..500]мкс | basic.hpp | ✅ ALL PASSED |
| 3 | Dechirp correction (peak → DC) | basic.hpp | ✅ PASSED |
| 4 | Full pipeline Process() | pipeline.hpp | ✅ PASSED |
| 5 | ProcessExternal (external cl_mem) | pipeline.hpp | ✅ PASSED |
| 6 | Random delays (seed=42) | basic.hpp | ✅ ALL PASSED |
| 7 | AllMaxima control | pipeline.hpp | ✅ PASSED |

### Python тесты

| Файл | Описание | Тестов |
|------|----------|--------|
| `test_heterodyne.py` | Базовые pytest тесты | 4 |
| `test_heterodyne_step_by_step.py` | Пошаговый pipeline с графиками | 8 шагов |
| `test_heterodyne_comparison.py` | GPU vs CPU сравнительный отчёт | 1 report |

### Python биндинги
- `python/py_heterodyne.hpp` — `register_heterodyne()` → `HeterodyneDechirp` class
- Методы: `set_params()`, `process()`, `process_external()`, `get_params()`

---

## 📁 Файлы модуля

```
modules/heterodyne/
├── include/
│   ├── heterodyne_dechirp.hpp          # Facade: HeterodyneDechirp
│   ├── heterodyne_params.hpp           # HeterodyneParams, Result types
│   ├── i_heterodyne_processor.hpp      # Strategy interface
│   └── processors/
│       ├── heterodyne_processor_opencl.hpp  # OPT-1/2: cached kernels/buffers
│       └── heterodyne_processor_rocm.hpp    # ROCm stub
├── src/
│   ├── heterodyne_dechirp.cpp          # Facade (OPT-3/4, SNR)
│   ├── heterodyne_processor_opencl.cpp # GPU kernel launch (~350 строк)
│   └── heterodyne_processor_rocm.cpp   # ROCm stub
├── kernels/opencl/
│   ├── dechirp_multiply.cl             # 1D conj(rx × ref) kernel (OPT-5)
│   └── dechirp_correct.cl             # 1D phase_step correction (OPT-5/6)
├── tests/
│   ├── all_test.hpp                    # Test registry (7 tests)
│   ├── test_heterodyne_basic.hpp       # Tests 1-3, 6
│   ├── test_heterodyne_pipeline.hpp    # Tests 4-5, 7
│   └── README.md
└── CMakeLists.txt
```

### Зависимости
- `drvgpu` — GPU backend (IBackend, OpenCL context/queue)
- `signal_generators` — LfmConjugateGenerator (conj(s_tx))
- `spectrum_maxima` (fft_maxima) — FFT + OnePeak поиск максимума
- `OpenCL::OpenCL` — OpenCL runtime

---

## ⚠️ Важные нюансы

1. **Задержки должны быть < T** (длительности чирпа). При delay > T сигнал пустой.
2. **conj от произведения** — без этого частота биений отрицательная и пик в верхней половине FFT.
3. **`-cl-fast-relaxed-math`** — компиляция ядер с relaxed math, ошибка sin/cos ~1e-4.
4. **SpectrumMaximaFinder** — FFT pad до степени 2 (8000→8192), search_range автоматически.
5. **OPT-3** — `DechirpWithGPURef()` работает только в `ProcessExternal()` (rx уже на GPU).
   Для `Process()` (rx на CPU) используется CPU ref path.

---

*Обновлено: 2026-02-21 | Кодо (AI Assistant)*
