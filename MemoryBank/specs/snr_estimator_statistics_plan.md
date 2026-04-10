# Plan: SNR-estimator в statistics (ROCm 7.2+)

> Обновлено: 2026-04-09 (v5 — после Python моделирования) | Кодо
>
> **v5 изменения (после Python моделирования, [snr_estimator_review_v6_findings_2026-04-09.md](snr_estimator_review_v6_findings_2026-04-09.md)):**
>
> 🔴 **КРИТИЧЕСКОЕ ОТКРЫТИЕ Python модели:** CA-CFAR с `guard=3, ref=8` на rectangular-окне даёт **−27 dB bias** из-за sinc sidelobes (хвосты пика попадают в ref-окно и завышают noise_mean). Для default сценариев rectangular-окно **неприменимо**.
>
> ✅ **Решение — Hann window + увеличенные guard/ref:**
> - **Window function**: Hann (sidelobes −32 dB вместо −13 dB)
> - **guard_bins**: 3 → **5**
> - **ref_bins**: 8 → **16**
> - **CFAR estimator**: CA-CFAR (mean) — подтверждено калибровкой
> - **Default `kDefaultWindow = Hann`** в snr_defaults
>
> ✅ **Новое: расширение `PadDataOp` параметром `WindowType`** (см. новый раздел 2.0.1)
> - enum `WindowType { None, Hann, Hamming, Blackman }`
> - kernel применяет window НА ЛЕТУ до zero-padding
> - Default `WindowType::None` — существующий API `PadDataOp::Execute(...)` работает без изменений
> - `FFTProcessorROCm::ProcessMagnitudesToGPU` принимает window параметр
>
> ✅ **Откалиброванные пороги (Эксп.5):**
> - `low_to_mid_db = 15.0` (было 6.0)
> - `mid_to_high_db = 30.0` (было 12.0)
> - P_correct = **97.9%** для Hann + mean
>
> ✅ **Подтверждено математикой:**
> - Hann gives **coherent gain + Hann loss**: `SNR_fft = SNR_in + 10·log10(N_actual) − 1.76 dB`
> - Bias стабильный (~−2..−4 dB), легко компенсируется калибровкой порогов
> - H0 артефакт (только шум) ≈ 10 dB → `low_to_mid > 12 dB` обязательно
>
> 📊 **Python модель:** [`PyPanelAntennas/SNR/`](../../PyPanelAntennas/SNR/)
> - 5 экспериментов × 8 комбинаций (window × CFAR estimator)
> - 3 финальных графика: `debug_cfar_bias.png`, `debug_window_vs_guard.png`, `debug_peak_grows.png`
> - JSON результаты: `results/exp5_thresholds.json` ← главный выход
>
> **v4 изменения (сохранены):**
> - Добавлен раздел **2.2.7 HIP kernel `peak_cfar_kernel`** с полным псевдокодом (argmax LDS-reduction + CFAR wraparound + защита log10)
> - Семантика `target_n_fft`: любое `N_actual` автоматически выравнивается до `NextPowerOf2(N_actual)` через существующий `CalculateNFFT`
> - Проверка памяти разделена на CPU-overload (учитываем INPUT upload) и GPU-overload (только новые scratch-буферы)
> - Стиль: `target_N_fft → target_n_fft`, `actual_N_actual → n_actual` (snake_case по CLAUDE.md)
> - `search_left_right → search_full_spectrum`
> - `BranchSelector::Select` — NaN/Inf guard (invalid snr_db → keep current branch)
> - `SnrEstimationConfig::Validate()` — проверка `2*(guard+ref)+1 < target_n_fft` (wraparound safety)
>
> **v3 изменения (сохранены):**
> - `target_n_fft` — гибкий параметр (default 2048), не догма 1024
> - Pipeline: `FFTProcessorROCm::ProcessMagnitudesToGPU` (новый метод) вместо `ISpectrumProcessor`
> - Square-law detector через новый kernel `complex_to_magnitude_squared` (БЕЗ sqrt, ~7× быстрее)
> - Median через существующий `MedianRadixSortOp::ExecuteFloat(1, n_ant_used)` (не новый kernel!)
> - Hysteresis через отдельный класс `BranchSelector` (SOLID), facade `StatisticsProcessor` остаётся stateless
> - BatchManager не нужен, данные целиком + проверка памяти с exception
> - Папка `PyPanelAantenns` переименована в `PyPanelAntennas` (опечатка исправлена)

---

## Сценарии использования

| Сценарий | n_antennas | n_samples | Входной размер (complex float) |
|---|---|---|---|
| **Py-Small** (Python model) | 5 лучей | 1 300 000 | 50 MB |
| **A — C++ стандарт** | 2500 (50×50) | 5 000 | 100 MB |
| **B — C++ большой** | 256 | 1 300 000 | 2.66 GB |
| **C — C++ огромный** | 9000 (квадрат 95×95) | 10 000 | 720 MB |

> ⚠️ **complex float = 8 байт** (2×float32).
> RADEON 9070 = 16 GB VRAM → все сценарии помещаются.
> Защита: проверка `required_bytes < free_vram × 0.8` → `throw std::runtime_error(...)` если не влезет.
> BatchManager НЕ используется в первой версии (см. Q-6 в ревью).

---

## Принятые допущения

| Параметр | Решение |
|---|---|
| Тип данных | `std::complex<float>` = 2×float32 = 8 байт/сэмпл |
| Децимация | **Downsampling** (каждый N-й) — быстро, алиасинг допустим |
| Пик-поиск | **Весь спектр** [0..n_fft-1] — `search_full_spectrum=true` |
| FFT-размер | **Гибкий** — `target_n_fft` параметр, default **2048**. Входные данные после децимации могут иметь любой размер `N_actual`; `FFTProcessorROCm::CalculateNFFT` **автоматически** вычисляет `nFFT = NextPowerOf2(N_actual) × repeat_count` (см. [fft_processor_rocm.cpp:565](../../modules/fft_func/src/fft_processor_rocm.cpp#L565)), а `PadDataOp` дополняет нулями до nFFT. Caller указывает `target_n_fft` как желаемый нижний bound, фактический nFFT = ближайшая степень 2 сверху. |
| Авто-выбор step | Если оба `step_samples=0` и `target_n_fft=0` → `target=2048`, `step=ceil(n_samples/target)` |
| **Window function** | **Hann** (default) — решает проблему sinc sidelobes, sidelobes −32 dB вместо −13 dB. Применяется в расширенном `PadDataOp(window=Hann)`. Без window rect-обработка даёт −27 dB bias. |
| **CFAR estimator** | **CA-CFAR (mean)** — подтверждено калибровкой в Python (P_correct = 97.9%). OS-CFAR (median) даёт +1-3 dB точности но сложнее в реализации, отложен как опция. |
| **guard_bins / ref_bins** | **5 / 16** (было 3/8) — калибровано в Python модели. С Hann окна этих значений достаточно, без window нужно guard≥200. |
| Детектор | **Square-law** (`\|X\|²`) — новый kernel `complex_to_magnitude_squared` (БЕЗ sqrt, ~7× быстрее), вызов через `MagnitudeOp::Execute(..., squared=true)` |
| Агрегат | `median(snr_db_per_antenna)` через существующий `MedianRadixSortOp::ExecuteFloat(1, n_ant_used)` |
| Hysteresis | Отдельный класс `BranchSelector` — facade остаётся stateless |
| BatchManager | НЕ используется, данные целиком + проверка памяти с exception |
| **Пороги бренчей** | **low_to_mid = 15 dB, mid_to_high = 30 dB** (калибровано Python, P_correct=97.9%) |
| Цель | Быстрый **грубый** SNR для переключения ветвей |

---

## Математика

### SNR после дечирпа + FFT

complex float → после гетеродина (дечирп) → CW тональный сигнал:

```
s_d(t) = A * exp(j * 2π * f_d * t)
```

FFT длины N_fft (без нормировки), N_actual = реальное число ненулевых сэмплов:

```
P_peak  = |X[k_peak]|²  ≈  A² × N_actual²     (когерентное накопление!)
P_noise = mean(|X[k_ref]|²)  ≈  σ² × N_actual  (σ² = мощность шума на сэмпл)

SNR_fft = P_peak / P_noise  =  SNR_in × N_actual
SNR_fft_dB  =  SNR_in_dB  +  10·log10(N_actual)
```

> ⚠️ **Coherent gain зависит от `N_actual` (число ненулевых сэмплов), НЕ от `n_fft`!**
> Zero-padding не добавляет энергию — просто sinc-интерполяция спектра.
> Пример: при target_n_fft=2048 и n_samples=5000 → step=3, N_actual=1666, gain = 10·log10(1666) ≈ **32.2 dB**.
> При target_n_fft=1024 → gain ≈ 30 dB. При target_n_fft=4096 → gain ≈ 36 dB.

### Авто-выбор step_samples / target_n_fft (гибкий)

Config имеет **два параметра**, любой можно оставить 0 = автовычисление:

```cpp
struct SnrEstimationConfig {
  uint32_t target_n_fft = 0;    // 0 = auto (default 2048 если step тоже 0)
  uint32_t step_samples = 0;    // 0 = auto из target_n_fft
  // ...
};
```

**Логика авто-выбора (вычисляется в `StatisticsProcessor::ComputeSnrDb`):**

```
Case 1: оба = 0 (default)
    target_n_fft = 2048
    step_samples = ceil(n_samples / 2048)
    N_actual     = n_samples / step_samples

Case 2: задан только target_n_fft
    step_samples = ceil(n_samples / target_n_fft)
    N_actual     = n_samples / step_samples

Case 3: задан только step_samples
    N_actual     = n_samples / step_samples
    target_n_fft = (не требуется — nFFT вычислит CalculateNFFT)

Case 4: заданы оба
    caller берёт ответственность, валидация n_samples >= step × N_actual
```

**Далее передаём в `FFTProcessorROCm::ProcessMagnitudesToGPU`:**
```
params.n_point = N_actual;
// CalculateNFFT сам вычислит nFFT_ = NextPowerOf2(N_actual) × repeat_count
// PadDataOp дополнит нулями до nFFT_
```

То есть `target_n_fft` — это желаемый нижний bound, фактический `nFFT` = ближайшая степень 2 сверху. Для всех сценариев A/B/C совпадает с target (см. таблицу ниже).

**Почему default 2048?**
- Компромисс между 1024 (мало разрешения) и 4096 (избыточно для маленьких сценариев)
- 2048 = 2^11, быстрый rocFFT
- Для сценария A (5000 samples): `step=3 → N_actual=1666 → NextPowerOf2(1666)=2048`, gain = 32.2 dB
- Для сценария B (1.3M samples): `step=635 → N_actual=2047 → NextPowerOf2(2047)=2048`, gain = 33.1 dB
- Для сценария C (10K samples): `step=5 → N_actual=2000 → NextPowerOf2(2000)=2048`, gain = 33.0 dB

> ℹ️ **Про non-power-of-2 target_n_fft** (например 3200, 4000): caller может указать любое значение, `NextPowerOf2` округлит вверх. Например `target_n_fft=4000, N_actual=4000 → nFFT=4096` (следующая 2^i). Для SNR-оценки это **не имеет значения**: coherent gain зависит от `N_actual` (ненулевые сэмплы), разница между pad до 4000 и pad до 4096 — ~0.01 dB в sinc-боковых лепестках.

### Таблица масштабирования (пример для target_n_fft = 2048)

| n_samples | step_samples | N_actual | nFFT (NextPowerOf2) | Когерентный gain |
|---|---|---|---|---|
| 2 000 | 1 | 2 000 | 2048 | 33.0 dB |
| 4 000 | 2 | 2 000 | 2048 | 33.0 dB |
| **5 000** | **3** | **1 666** | **2048** | **32.2 dB** |
| 8 000 | 4 | 2 000 | 2048 | 33.0 dB |
| **10 000** (Сцен. C) | 5 | 2 000 | 2048 | 33.0 dB |
| 16 000 | 8 | 2 000 | 2048 | 33.0 dB |
| 64 000 | 32 | 2 000 | 2048 | 33.0 dB |
| 256 000 | 125 | 2 048 | 2048 | 33.1 dB |
| 1 000 000 | 489 | 2 044 | 2048 | 33.1 dB |
| **1 300 000** (Сцен. B) | **635** | **2 047** | **2048** | **33.1 dB** |

> Когерентный gain стабилен **32-33 dB** для всех n_samples при target_n_fft=2048 — ключевое свойство.
> При target=1024 → gain ≈ 30 dB. При target=4096 → gain ≈ 36 dB.
> Рост gain линейный по N_actual: `gain = 10·log10(N_actual)`.

### Память gather_decimated output

```
output_size = n_ant_used × N_actual × 8 байт   (complex float)
Пример при target_n_fft = 2048:

Py-Small: 5 ant × 2048 × 8  =  81 920 байт  ≈ 80 KB
Сцен. A:  50 ant × 2048 × 8  =  819 200 байт  ≈ 800 KB
Сцен. B:  43 ant × 2048 × 8  =  704 512 байт  ≈ 688 KB
Сцен. C:  180 ant × 2048 × 8 = 2 949 120 байт ≈ 2.8 MB
```

> ⚠️ В сценарии B **используется ~43 антенны** после авто-step_antennas=6
> (256 / 6 = 42, ceil → 43), не все 256.
> В сценарии C **~180 антенн** (9000 / 50 = 180), не все 9000.

### CA-CFAR

```
k_peak = argmax(|X[k]|²,  k = 0..nFFT-1)

P_noise_est = mean(|X[k]|²  для  k ∈ ref_window)
ref_window  = [k_peak ± (guard_bins + 1 .. guard_bins + ref_bins)] mod nFFT

SNR_fft_dB = 10 · log10(|X[k_peak]|² / P_noise_est)
```

> ⚠️ **Требование на nFFT:** `2 × (guard_bins + ref_bins) + 1 < nFFT`, иначе
> ref-window перекроется сам с собой через wraparound. Default `guard=3, ref=8 → 23 < 2048` — OK.
> Валидация выполняется в `SnrEstimationConfig::Validate()` (см. раздел 2.1).

---

## Часть 1 — Python-модель (этап 0)

> Цель: найти оптимальные параметры до написания C++.

### Структура файлов

```
PyPanelAntennas/SNR/
├── snr_estimator_model.py    — главный скрипт, запускает все эксперименты
├── lfm_signal_generator.py   — генератор ЛЧМ + AWGN (complex float)
├── dechirp_numpy.py          — numpy гетеродин (дечирп)
├── cfar_estimator.py         — CA-CFAR SNR оценщик
├── plots/                    — PNG графики экспериментов
└── results/                  — JSON с численными результатами
```

### Запуск

```bash
# Windows (дома)
"F:\Program Files (x86)\Python314\python.exe" PyPanelAntennas/SNR/snr_estimator_model.py

# Debian (работа)
python3 PyPanelAntennas/SNR/snr_estimator_model.py
```

### Эксперимент 1 — Базовая кривая SNR

**Цель**: проверить математику, найти offset.

- Сценарий A: 2500 антенн, 5000 сэмплов
- step_samples=1 (без децимации), 1 антенна
- SNR_in = 40, 35, 30, 25, 20, 15, 10, 5 dB
- Измерить SNR_fft, сравнить с теорией SNR_in + 10·log10(5000) = SNR_in + 37 dB

**График**: `SNR_in` vs `SNR_fft_measured` (прямая + теория).

### Эксперимент 2 — Масштабирование n_samples (2K → 1.3M)

**Цель**: исследовать поведение оценщика при росте данных.

**Фиксировано: 5 антенн** (медиана по 5 антеннам) — изолируем влияние n_samples.

**Нелинейный шаг** (логарифмический, 11 точек):

```python
n_antennas   = 5       # фиксировано!
n_samples_list = [2_000, 4_000, 8_000, 16_000, 32_000,
                  64_000, 128_000, 256_000, 512_000,
                  1_000_000, 1_300_000]
```

Для каждого n_samples:
- Авто-step_samples (`target_n_fft=0` → default **2048**)
- SNR_in = 5, 10, 20 dB (три репрезентативных)
- Измерить: SNR_fft_measured, ошибку оценки σ, время (numpy)

**Графики**:
- `n_samples` vs `SNR_fft_measured` (лог. ось X) — 3 кривые (5/10/20 dB)
- `n_samples` vs `σ(SNR_error)` — как падает разброс при росте данных
- `n_samples` vs `N_fft` — автоматически подобранный размер FFT
- `n_samples` vs `step_samples` — автоматически подобранный шаг

### Эксперимент 3 — Влияние step_samples при фиксированном n_samples=5000

**Цель**: найти "колено" точность vs скорость.

- step = 1, 2, 4, 8, 16 → N_actual = 5000, 2500, 1250, 625, 312 → pad → N_fft = **8192, 4096, 2048, 1024, 512** (степени 2!)
- SNR_in = 40..5 dB
- Измерить: ошибку оценки σ(SNR_error) vs step

**График**: `step_samples` vs `σ(SNR_error)` для разных SNR_in.

### Эксперимент 4 — Стабилизация по антеннам

**Цель**: найти минимальное число антенн для медианы.

- Сценарий A: step_antennas = 1, 2, 5, 10, 25, 50 (→ 2500..50 антенн)
- Сценарий B: step_antennas = 1, 2, 4, 8, 16, 32 (→ 256..8 антенн)
- Измерить: дисперсию медианы SNR vs N_ants

**График**: `N_ants_used` vs `Var(median SNR_fft)`.
**Вывод**: минимальное N_ants при Var < 1 dB.

### Эксперимент 5 — Пороги переключения

**Цель**: найти надёжные пороги SNR_fft для переключения ветвей.

- Для каждого SNR_in (шаг 1 dB от -30 до +20 dB)
- 1000 реализаций шума
- Измерить P(правильного переключения) vs порог
- Найти пороги при P_correct > 90%

**Графики**: ROC-кривые, таблица финальных порогов.

---

## Часть 2 — C++ реализация (ПОНЕДЕЛЬНИК, Debian/ROCm)

> ⚠️ **Этот раздел выполняется в понедельник** на Debian с AMD GPU.
> Код пишут другие помощники, Кодо — **только ревьюер**.
> Таски будут созданы в `MemoryBank/tasks/TASK_snr_estimator_*.md` (тоже помощниками).

### 2.0 Расширение fft_func — НОВЫЙ этап (Этап 1.5)

**Цель:** добавить режим `|X|²` (square-law) в существующий `MagnitudeOp` без ломания API.

**Файл 1:** `modules/fft_func/include/kernels/complex_to_mag_phase_kernels_rocm.hpp`
- Добавить новый kernel `complex_to_magnitude_squared` в **обе** функции:
  - `GetComplexToMagnitudeKernelSource()`
  - `GetCombinedC2MPKernelSource()`
- Реализация:
  ```cpp
  __launch_bounds__(BLOCK_SIZE)
  extern "C" __global__ void complex_to_magnitude_squared(
      const float2_t* __restrict__ input,
      float* __restrict__ output,
      float inv_n,
      unsigned int total)
  {
      unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid >= total) return;
      float2_t z = input[gid];
      output[gid] = (z.x * z.x + z.y * z.y) * inv_n;  // NO sqrt — ~7× faster
  }
  ```

**Файл 2:** `modules/fft_func/include/operations/magnitude_op.hpp`
- Добавить параметр `bool squared = false` в `MagnitudeOp::Execute(...)` (БЕЗ ломания API — default сохраняет текущее поведение)
- Внутри: выбор kernel по имени (`squared ? "complex_to_magnitude_squared" : "complex_to_magnitude"`)
- Нормировка `inv_n` одна и та же для обоих kernel'ов
- Scope: **только ROCm**, `.cl` OpenCL версию НЕ трогаем

**Критерии ревью (Кодо проверит):**
- ✅ Default `squared=false` — существующие callers не меняются
- ✅ Kernel добавлен в **обе** функции source (иначе Combined compile не увидит)
- ✅ `__restrict__` и `__launch_bounds__` как у соседних kernel'ов
- ✅ Нет `sqrt`/`sqrtf`/`__fsqrt_rn` в новом kernel
- ✅ Существующие тесты `fft_func` проходят без изменений

---

### 2.0.1 Расширение `PadDataOp` параметром `WindowType` — НОВЫЙ Этап 1.6

**Цель:** добавить window function (Hann/Hamming/Blackman) в существующий `PadDataOp` без ломания API.

**Обоснование:** Python моделирование показало что CA-CFAR с rectangular-окном даёт **−27 dB bias** из-за sinc sidelobes. Hann window решает проблему полностью — sidelobes падают с −13 до −32 dB, CFAR начинает видеть реальный шум. Для грубого SNR-оценщика это критично.

**Файл 1:** новый файл `modules/fft_func/include/types/window_type.hpp`
```cpp
#pragma once

namespace fft_processor {

/// Window function для предварительной обработки signal'а перед FFT.
/// Default = None (rectangular, существующее поведение).
enum class WindowType : uint32_t {
  None    = 0,  // rectangular (default, существующее поведение)
  Hann    = 1,  // w[n] = 0.5*(1 - cos(2πn/(N-1)))
  Hamming = 2,  // w[n] = 0.54 - 0.46*cos(2πn/(N-1))
  Blackman = 3, // w[n] = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))
};

}  // namespace fft_processor
```

**Файл 2:** расширить `modules/fft_func/include/kernels/fft_processor_kernels_rocm.hpp`
- Добавить kernel `pad_data_windowed` (рядом с существующим `pad_data`)
- Kernel принимает параметр `window_type` (int) и коэффициенты окна через precomputed LUT, либо вычисляет формулу inline

```cpp
// NEW kernel — pad_data с window function
__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void pad_data_windowed(
    const float2_t* __restrict__ input,   // [beam × n_point]
    float2_t* __restrict__ fft_input,      // [beam × nFFT]
    unsigned int n_point,
    unsigned int nFFT,
    int window_type)                        // WindowType enum
{
    unsigned int bx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int by = blockIdx.y;
    if (bx >= nFFT) return;

    float2_t* fft_row = fft_input + (size_t)by * nFFT;

    if (bx < n_point) {
        float2_t z = input[(size_t)by * n_point + bx];
        // Применение окна
        float w = 1.0f;  // WindowType::None
        if (window_type == 1) {  // Hann
            float theta = 2.0f * M_PI_F * (float)bx / (float)(n_point - 1);
            w = 0.5f * (1.0f - __cosf(theta));
        } else if (window_type == 2) {  // Hamming
            float theta = 2.0f * M_PI_F * (float)bx / (float)(n_point - 1);
            w = 0.54f - 0.46f * __cosf(theta);
        } else if (window_type == 3) {  // Blackman
            float theta = 2.0f * M_PI_F * (float)bx / (float)(n_point - 1);
            w = 0.42f - 0.5f * __cosf(theta) + 0.08f * __cosf(2.0f * theta);
        }
        fft_row[bx].x = z.x * w;
        fft_row[bx].y = z.y * w;
    } else {
        fft_row[bx].x = 0.0f;
        fft_row[bx].y = 0.0f;
    }
}
```

**Файл 3:** расширить `modules/fft_func/include/operations/pad_data_op.hpp`
```cpp
class PadDataOp : public drv_gpu_lib::GpuKernelOp {
public:
  /**
   * @brief Execute pad kernel с опциональным window function.
   *
   * @param window  WindowType::None (default) — существующее поведение.
   *                Hann/Hamming/Blackman — применить window перед padding.
   */
  void Execute(void* input_buf, void* fft_input_buf,
               size_t beam_count, uint32_t n_point, uint32_t nFFT,
               WindowType window = WindowType::None) {
    // Выбор kernel по имени:
    const char* kernel_name = (window == WindowType::None)
        ? "pad_data"
        : "pad_data_windowed";

    // Остальной код без изменений — просто добавить window_type в args
    int w_type = static_cast<int>(window);
    void* args[] = { &input_buf, &fft_input_buf, &np, &nf, &w_type };
    // ... hipModuleLaunchKernel(kernel(kernel_name), ...)
  }
};
```

**Файл 4:** `FFTProcessorROCm::ProcessMagnitudesToGPU` — принимает параметр `window`
```cpp
void ProcessMagnitudesToGPU(
    void* gpu_data,
    void* gpu_out_magnitudes,
    const FFTProcessorParams& params,
    bool squared = false,
    WindowType window = WindowType::None,   // ← NEW
    ROCmProfEvents* prof_events = nullptr);
```
Внутри: `pad_op_.Execute(..., window)` вместо `pad_op_.Execute(...)`.

**Критерии ревью:**
- ✅ Default `WindowType::None` — существующие callers `PadDataOp::Execute(...)` работают без изменений
- ✅ Новый kernel `pad_data_windowed` компилируется в тот же HSACO module что `pad_data`
- ✅ Использованы fast intrinsics (`__cosf` вместо `cosf`) для скорости
- ✅ `#define M_PI_F 3.14159265358979323846f` в kernel source
- ✅ `FFTProcessorROCm` методы `ProcessComplex`/`ProcessMagPhase` — НЕ ТРОГАЕМ (они используют default `window=None`)
- ✅ Существующие тесты `fft_func` проходят без изменений

**Python reference реализация:** [`PyPanelAntennas/SNR/cfar_estimator.py:make_window()`](../../PyPanelAntennas/SNR/cfar_estimator.py)

---

### 2.1 Дополнить `statistics_types.hpp`

> ⚠️ Файл уже существует. Добавляем новые типы, не переписываем.

```cpp
// Default constants — КАЛИБРОВАНО в Python model (Эксп.5)
namespace snr_defaults {
  static constexpr uint32_t kTargetNFft           = 2048;  // гибкий, не догма 1024
  static constexpr uint32_t kGuardBins            = 5;     // было 3, откалибровано для Hann
  static constexpr uint32_t kRefBins              = 16;    // было 8, откалибровано для Hann
  static constexpr uint32_t kTargetAntennasMedian = 50;
  static constexpr float    kHysteresisDb         = 2.0f;

  // NEW: window function default (из Python Эксп.0)
  // Hann — sidelobes −32 dB, решает проблему sinc для rect (−13 dB)
  static constexpr fft_processor::WindowType kDefaultWindow =
      fft_processor::WindowType::Hann;
}

enum class BranchType { Low, Mid, High };

struct BranchThresholds {
  // Калибровано в Python Эксп.5: Hann + mean, P_correct = 97.9%
  float low_to_mid_db  = 15.0f;  // было 6.0, калибровка 2026-04-09
  float mid_to_high_db = 30.0f;  // было 12.0, калибровка 2026-04-09
  float hysteresis_db  = snr_defaults::kHysteresisDb;
};

struct SnrEstimationConfig {
  // FFT-size management — ГИБКИЙ!
  uint32_t target_n_fft = 0;      // 0 → auto (default 2048 если step_samples=0)
  uint32_t step_samples = 0;      // 0 → auto из target_n_fft

  // Antenna management
  uint32_t step_antennas = 0;     // 0 → ceil(n_antennas / kTargetAntennasMedian)

  // CFAR parameters (калибровано для Hann window)
  uint32_t guard_bins = snr_defaults::kGuardBins;   // default 5
  uint32_t ref_bins   = snr_defaults::kRefBins;     // default 16
  bool     search_full_spectrum = true;  // true = [0..nFFT-1], false = [0..nFFT/2]

  // NEW: Window function — применяется в PadDataOp перед FFT
  // Hann — default (решает проблему sinc sidelobes → −32 dB vs −13 dB rect)
  fft_processor::WindowType window = snr_defaults::kDefaultWindow;

  // Optional dechirp
  bool     with_dechirp = false;

  // Branch thresholds (без hysteresis — см. BranchSelector отдельный класс)
  BranchThresholds thresholds;

  /**
   * @brief Валидация параметров конфига.
   * @throws std::invalid_argument при нарушении инвариантов
   *
   * Проверяет:
   * - 2*(guard_bins + ref_bins) + 1 < target_n_fft (если target_n_fft > 0)
   *   иначе ref-window перекроется сам с собой через wraparound
   */
  void Validate() const {
    if (target_n_fft > 0) {
      uint32_t min_nfft = 2 * (guard_bins + ref_bins) + 1;
      if (min_nfft >= target_n_fft) {
        throw std::invalid_argument(
            "SnrEstimationConfig: 2*(guard+ref)+1=" + std::to_string(min_nfft) +
            " must be < target_n_fft=" + std::to_string(target_n_fft));
      }
    }
    // Аналогичная проверка с фактическим nFFT — в SnrEstimatorOp после CalculateNFFT
  }
};

struct SnrEstimationResult {
  float snr_db_global;                   // медиана по антеннам
  std::vector<float> snr_db_per_antenna; // per-antenna (пусто если не запрошено)
  uint32_t used_antennas;
  uint32_t used_bins;                    // реальный nFFT (после NextPowerOf2)
  uint32_t actual_step_samples;          // реальный step (если был auto)
  uint32_t n_actual;                     // n_samples / step (ненулевые sample'ы, до pad)
  // БЕЗ BranchType — это ответственность BranchSelector (см. раздел 2.5)
};
```

### 2.1.1 Расширить `shared_buf` slots

Модуль statistics уже использует slots 0..3 (`kInput`, `kMagnitudes`, `kResult`, `kMediansCompact`).
Для SNR-estimator добавляем:

```cpp
namespace shared_buf {
  // Existing (не трогаем):
  static constexpr size_t kInput          = 0;
  static constexpr size_t kMagnitudes     = 1;
  static constexpr size_t kResult         = 2;
  static constexpr size_t kMediansCompact = 3;

  // NEW for SNR-estimator:
  static constexpr size_t kGatherOutput   = 4;  // [n_ant_used × N_actual × complex]
  static constexpr size_t kFftMagSquared  = 5;  // [n_ant_used × N_fft × float] (|X|²)
  static constexpr size_t kSnrPerAntenna  = 6;  // [n_ant_used × float] SNR_db для медианы

  static constexpr size_t kCount          = 7;  // было 4, стало 7
}
```

### 2.2 HIP kernel — `gather_decimated`

Файл: `modules/statistics/kernels/gather_decimated_kernel.hpp`

**Thread mapping: «один поток на антенну, sequential loop по samples» (Вариант A из ревью).**

```cpp
// Вырезает из 2D массива [n_antennas][n_samples] (complex float, row-major)
// подвыборку [n_ant_out][n_samp_out] с шагами step_antennas, step_samples.
//
// Thread mapping: один поток = одна АНТЕННА (не один элемент!).
// Внутри потока — sequential loop по samples вдоль строки.
// L2 prefetcher видит линейный паттерн → эффективное чтение
// даже для больших step_samples (1000+).

__launch_bounds__(64)
__global__ void gather_decimated_kernel(
    const hipFloatComplex* __restrict__ src,   // complex float вход
    hipFloatComplex*       __restrict__ dst,   // complex float выход
    uint32_t n_samples,          // ширина исходной матрицы (строка)
    uint32_t n_samp_out,         // ширина выходной матрицы
    uint32_t step_antennas,      // шаг по строкам (gather по антеннам)
    uint32_t step_samples,       // шаг по столбцам (downsampling)
    uint32_t n_ant_out)          // число выходных антенн
{
    uint32_t ant = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant >= n_ant_out) return;

    const hipFloatComplex* src_row = src + ant * step_antennas * n_samples;
    hipFloatComplex*       dst_row = dst + ant * n_samp_out;

    // Sequential loop — L2 prefetcher работает эффективно
    for (uint32_t s = 0; s < n_samp_out; ++s) {
        dst_row[s] = src_row[s * step_samples];
    }
}

// Launch: grid(ceil(n_ant_out/64), 1), block(64, 1)
```

> Kernel читает `src[ant*step_antennas][s*step_samples]` → записывает `dst[ant][s]`.
> Все данные — complex float (hipFloatComplex = float2).

> 📌 **ПАМЯТКА для будущей оптимизации (если gather станет bottleneck'ом):**
>
> Если бенчмарк покажет что **gather > 5-10% общего времени** SNR-estimator'а
> **И** частота вызовов > 100 Hz → рассмотреть **Вариант C из ревью**:
>
> **Расширить `PadDataOp::Execute(..., step_samples=N, start_offset=N)`** в fft_func:
> - Decimation встраивается в pad kernel, убирая отдельный gather
> - Экономим +1 kernel launch (~5-10 µs)
> - Экономим +1 промежуточный буфер (kGatherOutput, ~2.8 MB в сценарии C)
> - `ProcessMagnitudesToGPU` принимает step_samples и передаёт в PadDataOp
> - Убирает `gather_decimated_kernel` полностью
>
> **Альтернатива (ещё более радикально):** Welch's method — batched FFT по фиксированным блокам + усреднение power spectra. Даёт правильнее PSD, убирает decimation вообще.
>
> Детали: см. [`snr_estimator_review_2026-04-09.md`](snr_estimator_review_2026-04-09.md) раздел про Q-2.

### 2.2.5 Расширить `FFTProcessorROCm` — новый метод `ProcessMagnitudesToGPU`

**Файл:** `modules/fft_func/include/fft_processor_rocm.hpp`

**Добавить новый публичный метод** (БЕЗ ломания существующего API):

```cpp
class FFTProcessorROCm {
public:
  // ... существующие методы ProcessComplex, ProcessMagPhase (не трогаем) ...

  /**
   * @brief Process FFT and write magnitudes directly to caller GPU buffer.
   *
   * No D2H copy. Pipeline: PadDataOp → hipfftExecC2C → MagnitudeOp(squared).
   * Reuses internal BufferSet<4> for pad/fft buffers.
   *
   * nFFT вычисляется автоматически через CalculateNFFT:
   *   nFFT_ = NextPowerOf2(params.n_point) × params.repeat_count
   * Caller указывает params.n_point = N_actual (ненулевые сэмплы после децимации),
   * PadDataOp зеро-пэдит остаток до nFFT_.
   *
   * @param gpu_data            Input [beam_count × n_point × complex<float>] on GPU
   * @param gpu_out_magnitudes  Output [beam_count × nFFT × float] — caller owns
   * @param params              FFT params (beam_count, n_point=N_actual, sample_rate, ...)
   * @param squared             false = |X| (default), true = |X|² (square-law, no sqrt)
   * @param window              WindowType::None (default) / Hann / Hamming / Blackman
   *                            Применяется в PadDataOp перед FFT.
   *                            Для SNR-estimator использовать Hann (см. 2.0.1).
   * @param prof_events         Optional profiling events collector
   */
  void ProcessMagnitudesToGPU(
      void* gpu_data,
      void* gpu_out_magnitudes,
      const FFTProcessorParams& params,
      bool squared = false,
      WindowType window = WindowType::None,
      ROCmProfEvents* prof_events = nullptr);
};
```

**Реализация:** использует существующие `PadDataOp + hipfftExecC2C + MagnitudeOp::Execute(..., squared)`. БЕЗ `ReadMagPhaseResults()` (нет D2H). BatchManager **не используется** — caller гарантирует что данные помещаются в VRAM (проверка на уровне `StatisticsProcessor::ComputeSnrDb`).

**Критерии ревью (Кодо проверит):**
- ✅ Существующие `ProcessComplex`/`ProcessMagPhase` не изменены
- ✅ Метод переиспользует `PadDataOp`, `MagnitudeOp` через `bufs_` (BufferSet)
- ✅ Caller владеет `gpu_out_magnitudes` — метод только пишет, не аллоцирует
- ✅ `hipfftExecC2C` вызывается через тот же plan management (LRU-2 cache)
- ✅ Никакого D2H в методе
- ✅ nFFT вычислен через существующий `CalculateNFFT` (NextPowerOf2 × repeat_count)

---

### 2.2.7 HIP kernel — `peak_cfar_kernel`

**Файл:** `modules/statistics/kernels/peak_cfar_kernel.hpp`

**Thread mapping:** один блок на антенну, BLOCK_SIZE=256 потоков на блок.

**Обоснование выбора:**
- `n_ant_used ~50` (после децимации антенн), `nFFT ~2048`
- 50 блоков × 256 threads = 12 800 threads — достаточная occupancy для AMD GPU
- Альтернатива «один поток на антенну» не работает: внутри антенны нужна parallel reduction для argmax над 2048 bin'ами
- Одна антенна = один CU (compute unit), без inter-block sync

**Алгоритм (two-pass внутри одного блока):**
1. Pass 1 — parallel argmax по всему nFFT через LDS-reduction
2. Pass 2 — sum по ref_window с wraparound `(k + nFFT) % nFFT`
3. Поток 0 считает `SNR_db = 10·log10(peak² / noise_mean)` и пишет результат

```cpp
// Файл: modules/statistics/kernels/peak_cfar_kernel.hpp
#pragma once

#if ENABLE_ROCM

namespace statistics {
namespace kernels {

inline const char* GetPeakCfarKernelSource() {
    return R"HIP(

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// ═══════════════════════════════════════════════════════════════
// Kernel: peak_cfar
// Argmax + CA-CFAR per antenna на уже вычисленных |X|²
//
// Thread mapping: один БЛОК на антенну (blockIdx.x = ant_id).
// Результат: snr_db_out[ant] = 10·log10(peak² / noise_mean)
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void peak_cfar(
    const float* __restrict__ mag_sq,     // [n_ant × nFFT] |X|² из MagnitudeOp(squared=true)
    float*       __restrict__ snr_db_out, // [n_ant]
    unsigned int nFFT,
    unsigned int guard_bins,
    unsigned int ref_bins)
{
    __shared__ float        s_max_val[BLOCK_SIZE];
    __shared__ unsigned int s_max_idx[BLOCK_SIZE];
    __shared__ float        s_ref_sum;
    __shared__ unsigned int s_ref_count;

    unsigned int ant = blockIdx.x;
    unsigned int tid = threadIdx.x;
    const float* row = mag_sq + (size_t)ant * nFFT;

    // ─── Pass 1: parallel argmax ────────────────────────────────
    float        my_max = -1.0f;
    unsigned int my_idx = 0;
    for (unsigned int k = tid; k < nFFT; k += BLOCK_SIZE) {
        float v = row[k];
        if (v > my_max) { my_max = v; my_idx = k; }
    }
    s_max_val[tid] = my_max;
    s_max_idx[tid] = my_idx;
    __syncthreads();

    // Reduction: argmax в s_max_val[0] / s_max_idx[0]
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_val[tid + s] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + s];
                s_max_idx[tid] = s_max_idx[tid + s];
            }
        }
        __syncthreads();
    }

    unsigned int k_peak = s_max_idx[0];
    float        peak   = s_max_val[0];

    // ─── Pass 2: ref-window sum с wraparound ─────────────────────
    if (tid == 0) { s_ref_sum = 0.0f; s_ref_count = 0; }
    __syncthreads();

    // Ref индексы: k_peak ± (guard+1 .. guard+ref) mod nFFT
    // Всего 2*ref_bins точек (ref с каждой стороны)
    unsigned int total_ref = 2u * ref_bins;
    for (unsigned int i = tid; i < total_ref; i += BLOCK_SIZE) {
        int offset;
        if (i < ref_bins) {
            // Левая сторона: k_peak - (guard+1+i)
            offset = -(int)(guard_bins + 1u + i);
        } else {
            // Правая сторона: k_peak + (guard+1+(i-ref_bins))
            offset = (int)(guard_bins + 1u + (i - ref_bins));
        }
        // Wraparound: +nFFT для защиты от отрицательных при малых k_peak
        int k_ref = ((int)k_peak + offset + (int)nFFT) % (int)nFFT;
        atomicAdd(&s_ref_sum, row[k_ref]);
        atomicAdd(&s_ref_count, 1u);
    }
    __syncthreads();

    // ─── Результат (только поток 0) ─────────────────────────────
    if (tid == 0) {
        float noise_mean = (s_ref_count > 0) ? (s_ref_sum / (float)s_ref_count) : 1.0f;
        // Защита от log10(0) и log10(inf)
        float ratio = (noise_mean > 1e-30f) ? (peak / noise_mean) : 1.0f;
        ratio = fmaxf(ratio, 1e-30f);
        snr_db_out[ant] = 10.0f * __log10f(ratio);
    }
}

)HIP";
}

}  // namespace kernels
}  // namespace statistics

#endif  // ENABLE_ROCM
```

**Launch config:**
```cpp
// В SnrEstimatorOp::ExecutePeakCfar:
dim3 grid(n_ant_used, 1, 1);
dim3 block(256, 1, 1);
void* args[] = { &mag_sq_ptr, &snr_out_ptr, &nFFT, &guard, &ref };
hipModuleLaunchKernel(kernel("peak_cfar"), grid.x, 1, 1, block.x, 1, 1, 0, stream(), args, nullptr);
```

**Shared memory per block:** `BLOCK_SIZE × (4 + 4) + 8 = ~2 KB` (s_max_val + s_max_idx + s_ref_sum + s_ref_count) — с огромным запасом до 64 KB лимита.

**Критерии ревью (Кодо проверит):**
- ✅ Argmax через **reduction в LDS**, не atomic (atomic порядок не детерминирован при равных max)
- ✅ Wraparound: `((int)k + offset + (int)nFFT) % (int)nFFT` — `+nFFT` защищает от отрицательных при малых `k_peak`
- ✅ Guard zone исключена — смещения начинаются с `guard_bins + 1`
- ✅ Защита `noise_mean > 1e-30f` перед делением, `ratio > 1e-30f` перед `log10`
- ✅ `__log10f` intrinsic (быстрый, single precision достаточно для dB-оценки)
- ✅ `__launch_bounds__(BLOCK_SIZE)` для register allocation
- ✅ `__restrict__` на указателях для auto-vectorization
- ✅ `size_t ant * nFFT` (не `unsigned int`) — защита от overflow при больших nFFT

**Edge cases (покрывает test_01 и test_03):**
- Пик на `k=0`: левая ref-сторона попадёт в `nFFT - guard - ref .. nFFT - guard - 1` через wraparound
- Пик на `k=nFFT-1`: правая сторона попадёт в `0 .. ref_bins-1` через wraparound
- `noise_mean == 0`: snr_db_out = `10·log10(1) = 0` (безопасный fallback)
- Только шум: формула даёт `~10·log10(H_nFFT) ≈ 9.1 dB` для nFFT=2048 (см. test_01)

---

### 2.3 Op-класс — `SnrEstimatorOp` (Layer 5, Ref03)

Файл: `modules/statistics/include/operations/snr_estimator_op.hpp`

```
Слои Ref03:
  Layer 5: SnrEstimatorOp
    ↳ gather_decimated_kernel   — 2D downsampling → kGatherOutput[n_ant_used × N_actual × cx]
                                  Thread mapping: один поток на антенну (Вариант A из ревью)

    ↳ FFTProcessorROCm::ProcessMagnitudesToGPU(
            gpu_in  = kGatherOutput,
            gpu_out = kFftMagSquared,
            params  = {beam_count=n_ant_used, n_point=N_actual, ...},
            squared = true,
            window  = WindowType::Hann)        ← НОВЫЙ метод (см. 2.2.5 + 2.0.1)
                                — FFT параллельно по ВСЕМ антеннам
                                — внутри: PadDataOp(window=Hann) → hipfftExecC2C → MagnitudeOp(squared=true)
                                — Hann применяется НА ЛЕТУ в pad_data_windowed kernel (2.0.1)
                                — pad до nFFT = next rocfft size (гибко)
                                — результат: |X[k]|² в kFftMagSquared (без sqrt, без D2H!)

    ↳ peak_cfar_kernel          — argmax + CA-CFAR per antenna
                                — читает kFftMagSquared (уже |X|²)
                                — пишет в kSnrPerAntenna[n_ant_used]
                                — square-law detector, правильная статистика Exponential

    ↳ MedianRadixSortOp::ExecuteFloat(beam_count=1, n_point=n_ant_used)
                                — ПЕРЕИСПОЛЬЗУЕМ существующий Op из statistics!
                                — читает kSnrPerAntenna (через kMagnitudes slot)
                                — пишет медиану в kMediansCompact[0]
                                — работает для любого n_ant_used < kHistogramThreshold (100K)
```

> ⚠️ **Используется `FFTProcessorROCm`**, НЕ `ISpectrumProcessor` (тот только для peak data).
> ⚠️ **`hipfftExecC2C` напрямую НЕ вызываем** — только через `FFTProcessorROCm::ProcessMagnitudesToGPU`.
> ⚠️ **НЕ пишем новый median kernel** — переиспользуем `MedianRadixSortOp::ExecuteFloat(1, n_ant_used)`.

**Важно про буферы:**
`MedianRadixSortOp::ExecuteFloat` читает из `shared_buf::kMagnitudes`. Нам нужно либо:
1. Скопировать `kSnrPerAntenna → kMagnitudes` через `hipMemcpyAsync` (D2D, ~1 µs)
2. **Или** записать SNR_db сразу в `kMagnitudes` (переиспользовать slot) — эффективнее
3. **Или** добавить параметр `slot_in` в `MedianRadixSortOp::ExecuteFloat` (расширить API)

**Рекомендую вариант 2** — в `peak_cfar_kernel` писать результат прямо в `kMagnitudes` (slot переиспользуется).

### 2.4 Фасад — `StatisticsProcessor` (stateless!)

```cpp
// Добавить в statistics_processor.hpp:

// Из CPU-данных
SnrEstimationResult ComputeSnrDb(
    const std::vector<std::complex<float>>& data,
    uint32_t n_antennas,
    uint32_t n_samples,
    const SnrEstimationConfig& config);

// Из GPU-буфера (complex float, row-major)
SnrEstimationResult ComputeSnrDb(
    void*    gpu_data,
    uint32_t n_antennas,
    uint32_t n_samples,
    const SnrEstimationConfig& config);
```

**Проверка памяти перед allocation** (см. Q-6 в ревью):

Разная логика для CPU-overload (надо загрузить INPUT на GPU) и GPU-overload (INPUT уже на GPU, считаем только **новые** аллокации):

```cpp
// Helper: размер новых GPU аллокаций SnrEstimator
size_t SnrEstimatorNewAllocBytes(uint32_t n_ant_used, uint32_t nFFT) {
  return n_ant_used * nFFT * sizeof(std::complex<float>)   // kGatherOutput
       + n_ant_used * nFFT * sizeof(float)                  // kFftMagSquared
       + n_ant_used * sizeof(float);                        // kSnrPerAntenna
}

// Универсальный чекер
void CheckVramAvailable(size_t required, const char* ctx) {
  size_t free_vram = 0, total_vram = 0;
  hipMemGetInfo(&free_vram, &total_vram);
  if (required > free_vram * 0.8) {
    throw std::runtime_error(
        std::string("SnrEstimator[") + ctx + "]: need " +
        std::to_string(required / (1024*1024)) + " MB, only " +
        std::to_string(free_vram / (1024*1024)) + " MB free (× 0.8 limit)");
  }
}

// === CPU overload ===
// INPUT надо загрузить на GPU → учитываем его в required
size_t required_cpu = n_antennas * n_samples * sizeof(std::complex<float>)  // INPUT upload
                    + SnrEstimatorNewAllocBytes(n_ant_used, nFFT);           // scratch
CheckVramAvailable(required_cpu, "CPU overload");

// === GPU overload ===
// INPUT уже аллоцирован caller'ом, hipMemGetInfo вернёт остаток после него.
// НЕ учитываем n_antennas * n_samples — это был бы двойной учёт.
size_t required_gpu = SnrEstimatorNewAllocBytes(n_ant_used, nFFT);
CheckVramAvailable(required_gpu, "GPU overload");
```

### 2.5 `BranchSelector` — отдельный класс для hysteresis (SOLID)

**Файл:** `modules/statistics/include/branch_selector.hpp`

> ⚠️ **Facade `StatisticsProcessor` остаётся stateless!** Hysteresis живёт здесь.

```cpp
namespace statistics {

/**
 * @brief Stateful branch selector with hysteresis.
 *
 * Caller creates one instance, keeps it alive across multiple SNR measurements.
 * Each call to Select() updates internal state (prev_branch) and applies
 * hysteresis to prevent thrashing on threshold boundaries.
 *
 * Example:
 *   BranchSelector selector;
 *   for (;;) {
 *     auto result = proc.ComputeSnrDb(data, ant, samp, config);
 *     BranchType b = selector.Select(result.snr_db_global, config.thresholds);
 *     // use b...
 *   }
 */
class BranchSelector {
public:
  BranchSelector() = default;

  /**
   * @brief Select branch for given SNR with hysteresis.
   * @param snr_db          Current SNR in dB
   * @param thresholds      Low/Mid/High thresholds + hysteresis_db
   * @return Current branch (Low/Mid/High)
   */
  BranchType Select(float snr_db, const BranchThresholds& thresholds);

  /// Current branch without updating state (for monitoring)
  BranchType Current() const { return current_; }

  /// Force reset to Low (useful for tests and reinitialization)
  void Reset(BranchType to = BranchType::Low) { current_ = to; }

private:
  BranchType current_ = BranchType::Low;
};

}  // namespace statistics
```

**Реализация Select():**
```cpp
BranchType BranchSelector::Select(float snr_db, const BranchThresholds& thr) {
  // Защита от NaN/Inf: невалидное измерение → оставляем текущую ветку
  // (ломать переключение одним плохим фреймом нельзя)
  if (!std::isfinite(snr_db)) {
    return current_;
  }

  const float h = thr.hysteresis_db;
  switch (current_) {
    case BranchType::Low:
      if (snr_db > thr.low_to_mid_db + h) current_ = BranchType::Mid;
      break;
    case BranchType::Mid:
      if (snr_db > thr.mid_to_high_db + h) current_ = BranchType::High;
      else if (snr_db < thr.low_to_mid_db - h) current_ = BranchType::Low;
      break;
    case BranchType::High:
      if (snr_db < thr.mid_to_high_db - h) current_ = BranchType::Mid;
      break;
  }
  return current_;
}
```

- ✅ Facade `StatisticsProcessor` stateless (SOLID: Single Responsibility)
- ✅ `BranchSelector` легко тестировать изолированно (без GPU)
- ✅ Thread-safety простая: один caller — один selector (**NOT thread-safe** между инстансами)
- ✅ NaN/Inf защита: `!isfinite(snr_db)` → возвращаем `current_` без обновления состояния
- ✅ `SnrEstimationResult` не содержит `BranchType` — чистое число SNR

### 2.6 Памятка — потенциальная оптимизация

> 📌 **Когда вернуться и ускорить**:
> Если бенчмарк покажет что `gather_decimated` занимает **> 10% общего времени**
> **И** `SnrEstimator` вызывается чаще **100 Hz** → рассмотреть:
>
> 1. **Расширение `PadDataOp::Execute(..., step_samples, start_offset)`** в fft_func
>    — decimation встраивается в pad, убирая gather kernel полностью
>    — `-1 kernel launch, -1 промежуточный буфер`
>    — переиспользуется существующая архитектура fft_func
>
> 2. **Welch's method**: batched FFT по фиксированным блокам + усреднение PSD
>    — математически правильнее decimation
>    — требует переписывания части 2 плана
>
> Подробности: [`snr_estimator_review_2026-04-09.md`](snr_estimator_review_2026-04-09.md) раздел «Памятка C-3».

---

## Часть 3 — Python-обёртки (pybind11)

Файл: `modules/statistics/python/statistics_bindings.cpp` (добавить к существующим)

Экспортируются:
- `SnrEstimationConfig` (struct)
- `SnrEstimationResult` (struct, **БЕЗ** BranchType!)
- `BranchThresholds` (struct)
- `BranchType` (enum: Low, Mid, High)
- `BranchSelector` (class с методами Select, Current, Reset)
- `StatisticsProcessor.compute_snr_db()` (метод)

```python
# Python API — пример использования
from gpu_work_lib import (
    StatisticsProcessor, SnrEstimationConfig,
    BranchSelector, BranchType
)

cfg = SnrEstimationConfig()
cfg.target_n_fft    = 0      # 0 = auto (default 2048)
cfg.step_samples    = 0      # 0 = auto из target_n_fft
cfg.step_antennas   = 0      # 0 = auto (ceil(n_antennas / 50))
cfg.guard_bins      = 5      # default — калибровано для Hann window
cfg.ref_bins        = 16     # default — калибровано для Hann window
cfg.search_full_spectrum = True
cfg.window          = WindowType.Hann  # default — решает sinc sidelobes

# Пороги (калиброваны в Python Эксп.5, P_correct = 97.9%)
cfg.thresholds.low_to_mid_db  = 15.0   # артефакт CFAR на шуме ≈ 10 dB → 15 с запасом
cfg.thresholds.mid_to_high_db = 30.0
cfg.thresholds.hysteresis_db  = 2.0

proc = StatisticsProcessor(context)
selector = BranchSelector()  # stateful, hysteresis

# Цикл измерений
for frame in data_stream:
    result = proc.compute_snr_db(frame, n_antennas, n_samples, cfg)
    branch = selector.Select(result.snr_db_global, cfg.thresholds)

    print(f"SNR: {result.snr_db_global:.1f} dB → branch: {branch}")
    print(f"  N_fft: {result.used_bins}, "
          f"antennas: {result.used_antennas}, "
          f"step: {result.actual_step_samples}")
```

`data_np` — `numpy.ndarray` dtype=`complex64`, shape=(n_antennas, n_samples).

---

## Часть 4 — Тесты

### Архитектура тестов (по паттерну capon)

Не плодим сущности! Один файл утилит + тесты по образцу `test_capon_rocm.hpp`.

```
modules/statistics/tests/
├── snr_test_helpers.hpp          — утилиты: MakeDechirpedCW, MakeNoise, AddAwgn
├── test_snr_estimator_rocm.hpp   — C++ тесты (7 тестов)
├── all_test.hpp                  — добавить include
└── README.md                     — добавить описание тестов SNR
```

### snr_test_helpers.hpp — утилиты (namespace `snr_test_helpers`)

```cpp
namespace snr_test_helpers {

// Генерация комплексного CW (тональный сигнал после дечирпа LFM)
// freq_norm = f_d / f_s ∈ (-0.5, 0.5), amplitude = A
std::vector<std::complex<float>> MakeDechirpedCW(
    uint32_t n_samples, float freq_norm, float amplitude);

// Генерация комплексного AWGN (σ² = noise_power)
std::vector<std::complex<float>> MakeNoise(
    uint32_t n_samples, float noise_power, uint32_t seed = 42u);

// Добавить шум к сигналу → SNR_in = 10*log10(amplitude²/noise_power)
void AddNoise(std::vector<std::complex<float>>& signal,
              float noise_power, uint32_t seed = 0u);

// Скопировать CPU данные в hipMalloc буфер
// (эмуляция: "данные пришли с сетевой карты через OpenCL, лежат на GPU")
void* CopyToGpu(const std::vector<std::complex<float>>& data);
void  FreeGpu(void* ptr);  // hipFree

} // namespace snr_test_helpers
```

> `MakeDechirpedCW` + `AddNoise` = синтетические данные **после гетеродина**.
> Это то что реально придёт в `ComputeSnrDb` от гетеродина в боевом коде.

### Эмуляция OpenCL данных → HIP (Паттерн B из capon)

```cpp
// В тестах симулируем: "данные от сетевой карты записаны OpenCL в GPU-память"
// Реальный путь: NIC DMA → OpenCL cl_mem → hipMalloc через SVM

// Шаг 1: CPU-данные (синтетика, после дечирпа)
auto signal_cpu = snr_test_helpers::MakeDechirpedCW(n_samples, freq=0.1f, A);
snr_test_helpers::AddNoise(signal_cpu, noise_power, seed=42);

// Шаг 2: hipMalloc + hipMemcpy HostToDevice — данные готовы на GPU
void* gpu_data = snr_test_helpers::CopyToGpu(signal_cpu);
//
// ⚠️ ВАЖНО: В продакшне источник данных на GPU — ОТДЕЛЬНЫЙ архитектурный вопрос:
//  - OpenCL SVM и ROCm hipMalloc это РАЗНЫЕ runtime'ы и разные heap'ы
//  - Прямая передача указателя OpenCL→HIP не работает — нужен DMABuf/HSA interop
//  - Тесты эмулируют "данные уже на ROCm GPU" через hipMalloc (как если бы data path
//    изначально был ROCm)
//  - Вопрос реального path (NIC → ROCm) выходит за рамки этого плана

// Шаг 3: передаём в тестируемую функцию
auto result = proc.ComputeSnrDb(gpu_data, n_antennas, n_samples, cfg);

// Шаг 4: освобождение
snr_test_helpers::FreeGpu(gpu_data);
```

### Тесты — test_snr_estimator_rocm.hpp

> Все тесты используют `cfg.target_n_fft = 0` (auto → 2048) если не указано иное.

#### test_01 — Только шум (нет сигнала!)

```
Цель: понять как ведёт себя алгоритм при чистом AWGN.

Данные: только MakeNoise(), без сигнала.

Математика (что ожидать):
  CA-CFAR находит случайный пик — максимум из N бинов.
  |X[k]|² ~ Exp(N·σ²)  (точное распределение square-law)
  E[max_k(|X|²)] ≈ σ² × N × H_N, где H_N = ln(N) + γ ≈ ln(N) + 0.577
  E[P_noise_ref] ≈ σ² × N
  SNR_fft_noise ≈ H_N = ln(2048) + 0.577 ≈ 8.2 → ≈ 9.1 dB

Ожидаемое поведение:
  - result.snr_db_global ≈ 8-10 dB  (артефакт CA-CFAR на чистом шуме)
  - После selector.Select(...) → BranchType::Low
  - Значение стабильно при разных seed (не зависит от амплитуды шума)

⚠️ Вывод для порогов: low_to_mid_db ДОЛЖЕН быть > 10 dB,
   иначе чистый шум попадёт в Mid-branch!
```

#### test_02 — Сигнал + шум (базовый)

```
Данные:
  n_antennas=1, n_samples=5000, target_n_fft=0 (auto → 2048)
  CW freq=0.15 (умеренный, не на краю), SNR_in = 20 dB
  1 антенна → проверяем без медианы

Ожидаемое:
  - step_samples auto = 3 → N_actual=1666 → pad до 2048
  - coherent gain = 10·log10(1666) ≈ 32.2 dB
  - snr_db_global ≈ 20 + 32.2 = 52.2 dB (теория без биаса CFAR)
  - С учётом биаса CFAR от sinc-боковых лепестков: 48-52 dB

Проверки:
  - snr_db_global > 40 dB (гарантированно High)
  - selector.Select(...) → BranchType::High
  - used_bins == 2048
  - n_actual == 1666
```

#### test_03 — Пик в отрицательной части (search_full_spectrum)

```
Данные: CW freq=-0.2 (отрицательная частота = сигнал "приближается")
        search_full_spectrum=true / false — два прогона

Проверки:
  - С true: пик найден, snr_db_global > 30 dB
  - С false (только [0..N/2]): пик пропущен, snr_db_global < 15 dB
```

#### test_04 — Сценарий A (2500 × 5000, авто)

```
Данные: 2500 антенн × 5000 сэмплов, SNR_in=15 dB
        Разные freq_norm для каждой антенны (±0.05..0.3)
        target_n_fft=0 (auto), step_antennas=auto

Ожидаемое:
  - step_antennas auto = 50 → used_antennas = 50
  - step_samples auto = 3 → N_actual=1666 → pad 2048
  - coherent gain ≈ 32.2 dB
  - snr_db_global ≈ 15 + 32.2 = 47.2 dB (медиана по 50 антеннам)

Проверки:
  - used_antennas == 50
  - used_bins == 2048
  - n_actual == 1666
  - snr_db_global в диапазоне [42, 50] dB
  - selector.Select(...) → BranchType::High
```

#### test_05 — Сценарий B (256 × 1 300 000, авто)

```
Данные: 256 антенн × 1,300,000 сэмплов, complex float (2.66 GB)
        SNR_in=10 dB
        target_n_fft=0 (auto → 2048)
        step_samples=авто → step=635 → N_actual=2047 → pad 2048
        step_antennas=auto → step=6 → used_antennas=43

Проверки:
  - actual_step_samples == 635
  - used_antennas == 43
  - used_bins == 2048
  - n_actual in [2040, 2050]
  - snr_db_global стабилен (медиана по 43 антеннам)
  - Время < 50 мс (замеряется в бенчмарке Этап 5.5)
```

#### test_06 — Только шум в Сценарии B (256 × 1 300 000)

```
Данные: 256 × 1,300,000, ТОЛЬКО MakeNoise() — нет сигнала!

Проверки:
  - snr_db_global ≈ 8-10 dB (артефакт CA-CFAR, см. test_01)
  - selector.Select(...) → BranchType::Low
  - Стабильность: повторный запуск → отклонение < 1 dB
```

#### test_06b — Сценарий C огромный (9000 × 10 000)

```
Данные: 9000 антенн × 10 000 сэмплов (720 MB), SNR_in=10 dB
        target_n_fft=0 (auto → 2048), step_antennas=auto

Ожидаемое:
  - step_antennas = ceil(9000/50) = 180 → used_antennas = 50
  - step_samples = ceil(10000/2048) = 5 → N_actual=2000 → pad 2048
  - coherent gain = 10·log10(2000) ≈ 33.0 dB
  - snr_db_global ≈ 10 + 33.0 = 43 dB

Проверки:
  - used_antennas == 50
  - used_bins == 2048
  - n_actual == 2000
  - snr_db_global > 38 dB
  - Время < 100 мс
```

#### test_07 — JSON export

```
Запустить test_04, сохранить результат:
  Results/JSON/snr_estimator_scenarioA.json
Проверить что файл создан и валиден.
```

### Python e2e тест — сигнал из генератора

Файл: `Python_test/statistics/test_snr_estimator.py`

```
Схема:
  signal_generators (GPU) → LFM сигнал с шумом →
  heterodyne (GPU, дечирп) →
  ComputeSnrDb (GPU) → результат

  Параллельно (numpy референс):
  то же самое через cfar_estimator.py (PyPanelAntennas/SNR/)

Сравнение:
  GPU result.snr_db_global vs numpy SNR_fft
  Допустимое отклонение: < 1 dB

Тест-кейсы:
  1. SNR_in = 20 dB, 1 антенна — базовая проверка
  2. SNR_in = 5 dB, 50 антенн  — слабый сигнал, медиана
  3. Только шум, 50 антенн     — проверка артефакта 8-10 dB
```

> Не создаём новых helper-классов! Используем существующие:
> `signal_generators` модуль + `heterodyne` + `cfar_estimator.py` из PyPanelAntennas/SNR/

---

## Часть 5 — API-документация (дописать/исправить)

Файлы для обновления после реализации:

| Файл | Что добавить |
|---|---|
| `Doc/Modules/statistics/Full.md` | Раздел SNR Estimator: архитектура, формулы, примеры |
| `Doc/Modules/statistics/API.md` | `ComputeSnrDb` сигнатуры, типы, пороги |
| `Doc/Modules/statistics/Quick.md` | Пример 5 строк Python |
| `Doc/Python/statistics_api.md` | Python API + numpy dtype=complex64 |
| `modules/statistics/tests/README.md` | Описание тест-кейсов SNR |

---

## Пороги переключения (откалиброваны в Python Эксп.5, 2026-04-09)

> ✅ **Финальные значения после калибровки** (Hann + CA-CFAR mean, P_correct = 97.9%)
> Обработка: `window=Hann, cfar=mean, guard=5, ref=16, target_n_fft=2048`

| SNR_fft | Branch | ≈ SNR_in |
|---|---|---|
| < 15 dB | **Low** (накопление, слабый сигнал) | < −18 dB |
| 15–30 dB | **Mid** (стандартная обработка) | −18 .. 0 dB |
| > 30 dB | **High** (точная обработка, сильный сигнал) | > 0 dB |

> ⚠️ Под H0 (только AWGN) CFAR даёт артефакт ≈ **10 dB** (см. test_01),
> поэтому `low_to_mid_db = 15 dB` — с запасом от артефакта.

**Истинные границы классов (для калибровки):**
- Low:  `SNR_in < -15 dB` (слабый сигнал, нужно накопление)
- Mid:  `-15 ≤ SNR_in < 0 dB` (стандартная обработка)
- High: `SNR_in ≥ 0 dB` (сильный сигнал, можно точную обработку)

**Результаты калибровки по всем комбинациям (из `PyPanelAntennas/SNR/results/exp5_thresholds.json`):**

| Combo | P_correct | low_to_mid | mid_to_high |
|---|---|---|---|
| **Hann + mean** (DEFAULT) | **97.9%** | **15.0** | **30.0** |
| Hamming + mean | 97.9% | 15.0 | 30.0 |
| rect + mean | 97.7% | 16.0 | 28.0 |
| Blackman + mean | 97.4% | 14.0 | 29.0 |
| Hann + median | 97.2% | 16.0 | 31.0 |

**Гистерезис:** ±2 dB, применяется через отдельный класс `BranchSelector` (см. раздел 2.5).

**Если используется другой `target_n_fft` (грубое приближение):**
- `target_n_fft = 1024` → gain ≈ 30 dB → сдвинуть пороги Low/Mid/High на ≈ −3 dB
- `target_n_fft = 4096` → gain ≈ 36 dB → сдвинуть на ≈ +3 dB
- Формула для сигнала: `shift_signal_db = 10·log10(target_n_fft / 2048)`

> ⚠️ **Формула — только стартовое приближение для сигнала!**
>
> Пороги находятся **между** CFAR-артефактом на H0 (~9 dB) и сигналом на H1 (>30 dB),
> а эти два уровня сдвигаются **по-разному** при изменении nFFT:
>
> | Переход 2048 → 4096 | Сдвиг |
> |---|---|
> | Coherent gain (сигнал) | `+3.01 dB` |
> | CFAR-артефакт (`10·log10(H_nFFT)`, гармоническое число) | `+0.35 dB` |
>
> То есть при удвоении nFFT сигнал отодвигается от шума на ~2.66 dB, и порог
> `low_to_mid_db` можно сдвинуть **меньше чем на +3 dB** — иначе потеряем чувствительность.
>
> **Правильный путь:** заново прогнать Эксп.5 (Python) для нового `target_n_fft` и
> взять откалиброванные значения. Формула выше — только быстрый старт.

---

## 🚦 Workflow и распределение ролей

### Порядок работ (зафиксировано Alex, 2026-04-09)

```
   СЕГОДНЯ (Windows, до понедельника)                ПОНЕДЕЛЬНИК (Debian, AMD GPU)
  ┌───────────────────────────────────────┐         ┌──────────────────────────┐
  │  1. Python анализ (5 экспериментов)   │         │   Тестирование           │
  │     PyPanelAntennas/SNR/              │         │   ─ C++ тесты            │
  │     параметры, пороги, калибровка     │         │   ─ Python e2e           │
  │                                       │────────▶│   ─ Профилирование       │
  │  2. Написание ВСЕГО кода (БЕЗ тестов) │         │   ─ Отладка              │
  │     ─ C++ расширение fft_func         │         │   ─ Бенчмарки            │
  │     ─ HIP kernel gather_decimated     │         │                          │
  │     ─ SnrEstimatorOp                  │         │                          │
  │     ─ ComputeSnrDb в StatisticsProc   │         │                          │
  │     ─ pybind11 биндинги               │         │                          │
  │     ─ Python обёртки и примеры        │         │                          │
  └───────────────────────────────────────┘         └──────────────────────────┘
```

1. **СЕГОДНЯ (до понедельника, Windows)** — Python анализ + написание **всего** C++ и Python кода. **БЕЗ тестов**, не запускаем (нет AMD GPU под Windows на ветке `main`).
2. **ПОНЕДЕЛЬНИК (Debian, AMD GPU)** — тестирование C++ и Python e2e, профилирование, отладка.

### Распределение ролей

| Роль | Кто | Ответственность |
|---|---|---|
| **Автор плана** | Кодо | `MemoryBank/specs/snr_estimator_statistics_plan.md` |
| **Автор тасков** | Другие агенты/помощники | `MemoryBank/tasks/TASK_snr_*.md` |
| **Автор кода** | Другие агенты/помощники | C++ и Python код |
| **Старший ревьюер** ⭐ | **Кодо** | Проверка тасков, проверка кода, согласованность с планом и стандартами GPUWorkLib |

> ⚠️ **Кодо НЕ пишет код самостоятельно** — только проверяет.
> **Кодо НЕ пишет таски** — только проверяет.
> Исключение: Python моделирование (Этап 0) — исследовательская часть, может делать Кодо.

---

## Этапы

| # | Этап | Когда | Исполнитель | Результат |
|---|---|---|---|---|
| **0** | Python анализ (5 экспериментов) | 🟢 Сегодня | Кодо / Alex | Параметры + пороги + графики (`PyPanelAntennas/SNR/`) |
| **1** | Типы: Config, Result, BranchType | 🟢 Сегодня | помощник → Кодо ревью | `statistics_types.hpp` (дополнить) |
| **1.5** | fft_func: `complex_to_magnitude_squared` + параметр `squared` в `MagnitudeOp` | 🟢 Сегодня | помощник → Кодо ревью | `kernels/complex_to_mag_phase_kernels_rocm.hpp` (+1 kernel), `operations/magnitude_op.hpp` (+1 параметр) |
| **2** | HIP kernel: `gather_decimated` | 🟢 Сегодня | помощник → Кодо ревью | `kernels/gather_decimated_kernel.hpp` |
| **3** | `SnrEstimatorOp` (Layer 5) | 🟢 Сегодня | помощник → Кодо ревью | `operations/snr_estimator_op.hpp` |
| **4** | `ComputeSnrDb` в фасаде | 🟢 Сегодня | помощник → Кодо ревью | `statistics_processor.hpp/.cpp` |
| **4.5** | Python bindings (pybind11) | 🟢 Сегодня | помощник → Кодо ревью | `modules/statistics/python/statistics_bindings.cpp` (дополнить) |
| **5** | C++ тесты (7 тестов) + helpers (КОД, не запуск!) | 🟢 Сегодня | помощник → Кодо ревью | `tests/snr_test_helpers.hpp` + `tests/test_snr_estimator_rocm.hpp` |
| **5.5** | Бенчмарк `GpuBenchmarkBase` (КОД) | 🟢 Сегодня | помощник → Кодо ревью | `tests/snr_estimator_benchmark.hpp` + runner |
| **6** | Python e2e тест (КОД, не запуск!) | 🟢 Сегодня | помощник → Кодо ревью | `Python_test/statistics/test_snr_estimator.py` |
| **Т1** | 🔵 **Тестирование C++** | 🟡 Понедельник | Alex + Кодо ревью логов | Запуск на Debian/ROCm, отладка, профилирование |
| **Т2** | 🔵 **Тестирование Python e2e** | 🟡 Понедельник | Alex + Кодо ревью логов | Запуск pipeline signal_generators → heterodyne → SNR |
| **7** | API-документация | 🔵 После тестов | помощник → Кодо ревью | `Doc/Modules/statistics/` |

Легенда: 🟢 сегодня (код без запуска), 🟡 понедельник (тесты на Debian), 🔵 после тестов (документация).

> ⚠️ **Вся разработка кода — СЕГОДНЯ.** В понедельник только запускаем на AMD GPU и дебажим. Документация — после успешных тестов.

---

## Параметры для тестирования (default target_n_fft=2048)

| | Py-Small (Python model) | Сцен. A | Сцен. B | Сцен. C |
|---|---|---|---|---|
| n_antennas | 5 лучей | 2500 (50×50) | 256 | 9000 (95×95) |
| n_samples | 1 300 000 | 5 000 | 1 300 000 | 10 000 |
| Входной размер | 50 MB | 100 MB | 2.66 GB | 720 MB |
| target_n_fft | 2048 (auto) | 2048 (auto) | 2048 (auto) | 2048 (auto) |
| step_samples (auto, `ceil(N/2048)`) | 635 | 3 | 635 | 5 |
| N_actual | 2 047 | 1 666 | 2 047 | 2 000 |
| N_fft после pad | 2048 | 2048 | 2048 | 2048 |
| step_antennas (auto, `ceil(N/50)`) | 1 (все 5) | 50 (→ 50 ant) | 6 (→ 43 ant) | 180 (→ 50 ant) |
| used_antennas | 5 | 50 | 43 | 50 |
| guard_bins | 3 | 3 | 3 | 3 |
| ref_bins | 8 | 8 | 8 | 8 |
| search_full_spectrum | true | true | true | true |
| coherent gain (dB) | 33.1 | 32.2 | 33.1 | 33.0 |
| gather output size | 160 KB | 800 KB | 688 KB | 800 KB |
| Целевое время | < 10 ms (numpy) | < 5 ms (GPU) | < 50 ms (GPU) | < 100 ms (GPU) |

> 📌 Таблица показывает **default auto** поведение. Пользователь может задать `target_n_fft=4096` (другие пороги, лучше разрешение) или любое другое значение — всё работает.

---

*Обновлено: 2026-04-09 (v4 — блокеры ревью v6 закрыты, добавлен peak_cfar_kernel) | Кодо*