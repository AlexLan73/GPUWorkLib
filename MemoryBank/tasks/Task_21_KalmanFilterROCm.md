# Task_21: KalmanFilterROCm
## 1D Скалярный фильтр Калмана — ROCm GPU реализация

**Статус**: ✅ TESTED (5/5 PASSED, 2026-03-05, Radeon 9070 gfx1201)
**Приоритет**: High
**Модуль**: `modules/filters`
**Документация**: `Doc_Addition/Filters/2_Kalman_Filter.md`
**Ветка**: worktree `claude/sleepy-williamson`

---

## Цель

Реализовать класс `KalmanFilterROCm` — GPU-фильтр Калмана (1D скалярный) для сглаживания зашумлённых радарных сигналов.

**Применение в радаре**:
- Сглаживание `f_beat[t]` (частота биений → дальность цели) по временной оси
- Фильтрация амплитуды пика A[t] от зондирования к зондированию
- Подавление шума в оценках угла прихода θ[t]

**Принцип**: Фильтр применяется к каждому лучу (channel) **независимо**.
Внутри луча — последовательный цикл `predict → update` по всем `points`.
Re и Im части обрабатываются **независимо** (два скалярных фильтра Калмана).

---

## Создаваемые файлы

```
modules/filters/
├── include/
│   ├── filters/
│   │   └── kalman_filter_rocm.hpp          ← NEW
│   ├── kernels/
│   │   └── kalman_kernels_rocm.hpp         ← NEW (R"HIP(...)HIP")
│   └── types/
│       └── filter_params.hpp               ← UPDATE (KalmanParams)
├── src/
│   └── kalman_filter_rocm.cpp              ← NEW
├── tests/
│   ├── test_kalman_rocm.hpp                ← NEW
│   └── all_test.hpp                        ← UPDATE
└── CMakeLists.txt                          ← UPDATE
```

---

## TASK 21.1 — filter_params.hpp: добавить KalmanParams

**Файл**: `modules/filters/include/types/filter_params.hpp`

Добавить в namespace `filters`:

```cpp
// ════════════════════════════════════════════════════════════════════════════
// Kalman Filter (ROCm)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct KalmanParams
 * @brief Параметры 1D скалярного фильтра Калмана
 *
 * Применяется к Re и Im частям независимо.
 *
 * Выбор параметров:
 *   R — дисперсия шума измерения. Начать с: R = (FFT_bin_size)^2 / 12
 *       Чем больше R, тем больше сглаживание (меньше доверие измерению).
 *   Q — дисперсия шума процесса. Начать с: Q = R / 100
 *       Чем больше Q, тем быстрее фильтр реагирует на изменения.
 *   Q/R << 1: сильное сглаживание, медленная реакция на изменения
 *   Q/R >> 1: слабое сглаживание, быстрая реакция
 */
struct KalmanParams {
  float Q  = 0.1f;   ///< Дисперсия шума процесса (process noise variance)
  float R  = 25.0f;  ///< Дисперсия шума измерения (measurement noise variance)
  float x0 = 0.0f;   ///< Начальная оценка состояния
  float P0 = 25.0f;  ///< Начальная дисперсия ошибки (обычно = R)
};
```

---

## TASK 21.2 — kalman_kernels_rocm.hpp

**Файл**: `modules/filters/include/kernels/kalman_kernels_rocm.hpp`

### Алгоритм (напоминание)

```
Состояние: x̂ (оценка), P (дисперсия ошибки)
Инициализация: x̂ = x0, P = P0

На каждый отсчёт z[n]:
  // 1. Predict (предсказание)
  x_pred = x̂               // модель: следующее состояние = текущее
  P_pred = P + Q            // дисперсия растёт (неопределённость растёт)

  // 2. Update (обновление по измерению)
  K = P_pred / (P_pred + R) // коэффициент Калмана: [0..1]
  x̂  = x_pred + K * (z[n] - x_pred)  // скорректировать оценку
  P  = (1 - K) * P_pred     // дисперсия уменьшается

  output[n] = x̂
```

### Упрощение для constant-state модели (x_pred = x̂)

```
  K = P_pred / (P_pred + R)
  x̂ += K * (z[n] - x̂)     // инновация * коэффициент
  P  = (1 - K) * P_pred
```

### Kernel source

```cpp
inline const char* kKalmanKernelSrc = R"HIP(

struct float2_t { float x; float y; };

// Один kernel — обрабатывает Re и Im независимо
extern "C" __global__ __launch_bounds__(256)
void kalman_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    float Q,    // process noise variance
    float R,    // measurement noise variance
    float x0,   // initial state
    float P0)   // initial error covariance
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;

    // Состояния для Re и Im — независимые скалярные фильтры
    float x_re = x0, x_im = x0;
    float P_re = P0, P_im = P0;

    for (unsigned int n = 0; n < points; n++) {
        float2_t z = in[base + n];

        // ─ Re ─────────────────────────────────────────
        // Predict
        float P_pred_re = P_re + Q;
        // Update
        float K_re = P_pred_re / (P_pred_re + R);
        x_re = x_re + K_re * (z.x - x_re);
        P_re = (1.0f - K_re) * P_pred_re;

        // ─ Im ─────────────────────────────────────────
        float P_pred_im = P_im + Q;
        float K_im = P_pred_im / (P_pred_im + R);
        x_im = x_im + K_im * (z.y - x_im);
        P_im = (1.0f - K_im) * P_pred_im;

        out[base + n].x = x_re;
        out[base + n].y = x_im;
    }
}

)HIP";
```

### Замечание об оптимизации

В steady-state (после ~20-50 отсчётов) коэффициент `K` сходится к постоянному значению:
```
K_ss = (-R + sqrt(R^2 + 4*Q*R)) / (2*Q)
```
Для очень длинных сигналов можно предварительно вычислить K_ss и использовать его — убирает деление в цикле. **Реализовать как опцию** через параметр `use_steady_state = false` (по умолчанию — полный алгоритм).

---

## TASK 21.3 — kalman_filter_rocm.hpp

**Файл**: `modules/filters/include/filters/kalman_filter_rocm.hpp`

```cpp
#pragma once

#if ENABLE_ROCM

#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "types/filter_params.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cstdint>

namespace filters {

/**
 * @class KalmanFilterROCm
 * @brief 1D скалярный фильтр Калмана на ROCm GPU
 *
 * Обрабатывает multi-channel IQ сигнал.
 * Re и Im части фильтруются независимыми скалярными фильтрами Калмана.
 * Каждый channel имеет независимое состояние (x_hat, P).
 *
 * Применение: сглаживание f_beat[t] (дальность) по временной оси,
 * фильтрация амплитуды пика, угловое сопровождение.
 *
 * Grid: 1D — (channels + block_size - 1) / block_size
 * 1 thread = 1 channel, последовательный цикл по points
 */
class KalmanFilterROCm {
public:
  explicit KalmanFilterROCm(drv_gpu_lib::IBackend* backend,
                             unsigned int block_size = 256);
  ~KalmanFilterROCm();

  KalmanFilterROCm(const KalmanFilterROCm&) = delete;
  KalmanFilterROCm& operator=(const KalmanFilterROCm&) = delete;
  KalmanFilterROCm(KalmanFilterROCm&&) noexcept;
  KalmanFilterROCm& operator=(KalmanFilterROCm&&) noexcept;

  void SetParams(const KalmanParams& params);
  void SetParams(float Q, float R, float x0 = 0.0f, float P0 = 25.0f);

  // GPU ptr -> GPU output (caller делает hipFree на result.data)
  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr, uint32_t channels, uint32_t points);

  // CPU input -> GPU output
  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t channels, uint32_t points);

  // CPU reference (для тестирования)
  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels, uint32_t points) const;

  const KalmanParams& GetParams() const { return params_; }
  bool  IsReady() const { return kernel_compiled_; }

private:
  void CompileKernel();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;
  hipStream_t   stream_  = nullptr;
  hipModule_t   module_  = nullptr;
  hipFunction_t kernel_  = nullptr;
  bool          kernel_compiled_ = false;

  KalmanParams params_;  // Q, R, x0, P0

  void*  cached_input_buf_  = nullptr;
  size_t cached_input_size_ = 0;

  unsigned int block_size_ = 256;
};

} // namespace filters

#else  // stub для non-ROCm сборки

namespace filters {

class KalmanFilterROCm {
public:
  explicit KalmanFilterROCm(void*, unsigned int = 256) {}
  void SetParams(const KalmanParams&) {}
  void SetParams(float, float, float = 0.0f, float = 25.0f) {}
  bool IsReady() const { return false; }
};

} // namespace filters

#endif // ENABLE_ROCM
```

---

## TASK 21.4 — kalman_filter_rocm.cpp

**Файл**: `modules/filters/src/kalman_filter_rocm.cpp`

### Структура реализации

```cpp
#if ENABLE_ROCM

// ─── CompileKernel ────────────────────────────────────────────────────────────
// hiprtc стандартный паттерн:
// 1. hiprtcCreateProgram(kKalmanKernelSrc, "kalman", 0, nullptr, nullptr)
// 2. hiprtcCompileProgram с opts {"-O3"}
// 3. При ошибке: hiprtcGetProgramLog → throw std::runtime_error
// 4. hiprtcGetCodeSize → hiprtcGetCode → vector<char>
// 5. hipModuleLoadData → kernel_
// 6. hipModuleGetFunction(module_, "kalman_kernel") → kernel_

// ─── SetParams ────────────────────────────────────────────────────────────────
// Валидация: Q > 0, R > 0, P0 > 0
// Сохранить в params_

// ─── Process ──────────────────────────────────────────────────────────────────
// 1. Выделить out_buf: hipMalloc(channels * points * sizeof(float2))
// 2. Собрать args:
//    void* args[] = {&input_ptr, &out_buf, &channels, &points,
//                    &params_.Q, &params_.R, &params_.x0, &params_.P0}
// 3. Grid: (channels + block_size_ - 1) / block_size_ × 1 × 1
// 4. Block: block_size_ × 1 × 1
// 5. hipModuleLaunchKernel(kernel_, gx, 1, 1, block_size_, 1, 1, ...)
// 6. hipStreamSynchronize(stream_)
// 7. return InputData<void*>{out_buf, channels, points}

// ─── ProcessFromCPU ───────────────────────────────────────────────────────────
// Паттерн cached_input_buf_ (переиспользование GPU буфера)

// ─── ProcessCpu ───────────────────────────────────────────────────────────────
// CPU эталон — точная копия алгоритма из kernel:
// for each channel:
//   x_re = params_.x0, x_im = params_.x0
//   P_re = params_.P0, P_im = params_.P0
//   for each point n:
//     P_pred_re = P_re + Q
//     K_re = P_pred_re / (P_pred_re + R)
//     x_re += K_re * (in[ch*pts+n].real() - x_re)
//     P_re = (1 - K_re) * P_pred_re
//     // same for Im
//     out[ch*pts+n] = complex<float>(x_re, x_im)

#endif
```

### Замечание по x0 (начальное состояние)

Два подхода:
1. **x0 = 0** (умолчание) — подходит для нулевого среднего
2. **x0 = z[0]** (первое измерение) — быстрее выходит на режим

Реализовать оба варианта через параметр `init_from_first = false` в `SetParams()` или просто использовать `x0 = params_.x0` и позволить пользователю передать `x0 = 0` или конкретное значение.

---

## TASK 21.5 — test_kalman_rocm.hpp

**Файл**: `modules/filters/tests/test_kalman_rocm.hpp`

### Тест 1: Константный сигнал + белый шум

```
Физический смысл: постоянный сигнал (например, f_beat от неподвижной цели),
зашумлённый белым шумом с дисперсией sigma^2.

Параметры: Q=0.01, R=25, x0=0, P0=25
Сигнал: x[n] = 100 + noise, noise ~ N(0, 5)
Каналы: 8, Points: 1024

Ожидание:
  - Первые 50 отсчётов: фильтр "разогревается"
  - После 100 отсчётов: filtered_x близко к 100 (±шум)
  - RMS(filtered - 100) << RMS(noisy - 100) — фильтр улучшает SNR
  - Ожидаемое снижение шума: ~sqrt(R/Q) ≈ 50x при Q=0.01, R=25
```

### Тест 2: GPU vs CPU (основной тест точности)

```
8 каналов, 4096 отсчётов, случайный complex<float> сигнал
Запуск ProcessCpu → reference
Запуск Process    → GPU result
Сравнение:
  max |GPU[i] - CPU[i]| < 1e-4f
```

### Тест 3: Независимость каналов

```
256 каналов, каждый с уникальным const-сигналом
const_val[ch] = (float)ch * 10.0f
x[ch * points + n] = const_val[ch] + small_noise

После 500 отсчётов: |output[ch * points + 499] - const_val[ch]| < 1.0f
Каналы не смешиваются — состояния независимы
```

### Тест 4: Шаговый отклик (step response)

```
x[n] = 0  при n < 512
x[n] = 100 при n >= 512
Параметры: Q=1.0, R=25

Ожидание: фильтр постепенно поднимается к 100 после скачка
Скорость реакции определяется K_ss
```

### Тест 5: Выбор параметров Q/R (практический тест)

```
Для радара: сигнал = f_beat от цели на дальности 100 (бины FFT)
Шум: sigma = 5 бин (что даёт R = 25)
Цель медленно движется: delta_f_beat ~ 0.1 бин/мс → Q << R

Оптимальные параметры: Q = 0.01, R = 25
Проверить: RMS после 200 отсчётов < 1 бин
```

### Структура кода теста

```cpp
namespace test_kalman_rocm {

void run(drv_gpu_lib::IBackend* backend) {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print("[KalmanROCm] Running tests...\n");

    filters::KalmanFilterROCm kalman(backend);
    kalman.SetParams(0.1f, 25.0f, 0.0f, 25.0f);

    // test_const_signal
    // test_gpu_vs_cpu
    // test_channel_independence
    // test_step_response

    con.Print("[KalmanROCm] All tests passed!\n");
}

} // namespace test_kalman_rocm
```

---

## TASK 21.6 — CMakeLists.txt и all_test.hpp

### CMakeLists.txt

```cmake
if(ROCM_ENABLED)
    list(APPEND MODULE_SOURCES
        src/kalman_filter_rocm.cpp     # NEW
    )
endif()
```

### all_test.hpp

```cpp
// Kalman ROCm
// #include "tests/test_kalman_rocm.hpp"
// test_kalman_rocm::run(backend);
```

---

## GPU dispatch параметры

```
Grid:  (channels + block_size - 1) / block_size × 1 × 1
Block: block_size × 1 × 1

Thread model: 1 thread = 1 channel
Loop: sequential predict-update для всех points

При channels=256, block_size=256: 1 блок × 256 потоков
При channels=4096, block_size=256: 16 блоков × 256 потоков
```

---

## Математика (для проверки реализации)

### Скалярный Калман step-by-step

```
Инициализация: x̂ = x0, P = P0

Шаг n (измерение z = in[n]):
  P_pred = P + Q                     // predict covariance
  K      = P_pred / (P_pred + R)     // Kalman gain [0..1]
  x̂      = x̂ + K * (z - x̂)          // update state
  P      = (1 - K) * P_pred          // update covariance

output[n] = x̂
```

### Steady-state gain

После многих итераций P и K сходятся к:
```
P_ss = Q/2 + sqrt((Q/2)^2 + Q*R)
K_ss = P_ss / (P_ss + R)

При Q=0.1, R=25:
  P_ss ≈ 1.56
  K_ss ≈ 0.059
```

---

## Валидация против Python

```python
import numpy as np

def kalman_filter_cpu(signal, Q=0.1, R=25.0, x0=0.0, P0=25.0):
    """
    1D скалярный фильтр Калмана — эталон для Python-тестов
    signal: 1D array, float
    """
    n = len(signal)
    x = np.zeros(n)
    x_hat = x0
    P = P0
    for i, z in enumerate(signal):
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x_hat = x_hat + K * (z - x_hat)
        P = (1 - K) * P_pred
        x[i] = x_hat
    return x

# Для complex<float>: применить к real и imag отдельно
def kalman_cf32(signal_cf32, **kwargs):
    re = kalman_filter_cpu(signal_cf32.real, **kwargs)
    im = kalman_filter_cpu(signal_cf32.imag, **kwargs)
    return re + 1j * im
```

---

## TASK 21.7 — 🎯 Красивый тест: LFM-радар с 5 целями и Калманом

> **Это главный демонстрационный тест для отчёта**
> Физически корректный радарный сценарий: 5 целей на разных дальностях.
> Показывает улучшение качества измерения дальности после фильтра Калмана.

**Файл**: добавить в `modules/filters/tests/test_kalman_rocm.hpp`

---

### Физический сценарий

**ЛЧМ-радар** с 5 антеннами (лучами), каждый направлен на цель на своей дальности.

После смешения принятого сигнала с опорным (Dechirp) — на выходе **тон биений** (beat signal):
```
x[n] = A · exp(j·2π·f_beat·n/fs) + шум_AWGN
```

Это идеальный сигнал для Калмана с моделью `x_pred = x_hat` (постоянное состояние).

**Параметры ЛЧМ:**

```
fs     = 10 МГц          Частота дискретизации
fdev   = 2 МГц           Девиация частоты
N      = 16384            Точек на импульс
Ti     = N / fs = 1.638 мс   Длительность
mu     = fdev / Ti = 1220.7 МГц/с  Скорость чирп (chirp rate)
Δbin   = fs / N = 610 Гц     Разрешение FFT
```

**Цели (линейные задержки):**

```
Ant | tau     | f_beat        | bin FFT | Range   | Физика
----+---------+---------------+---------+---------+------------------
  0 |  50 мкс |  61.03 кГц   |   100   |  7.5 км | Ближняя цель
  1 | 100 мкс | 122.07 кГц   |   200   | 15.0 км | Умеренная дальность
  2 | 150 мкс | 183.10 кГц   |   300   | 22.5 км | Средняя дальность
  3 | 200 мкс | 244.14 кГц   |   400   | 30.0 км | Дальняя цель
  4 | 250 мкс | 305.18 кГц   |   500   | 37.5 км | Дальняя цель
```

Бины FFT выбраны кратными 100 — легко верифицируются визуально!

**Шум:**
```
noise_sigma = 0.30     → SNR_raw ≈ 10 дБ (сложные условия)
```

**Параметры Калмана:**
```
Q   = 0.001   процессный шум малый (тон почти не меняется)
R   = 0.09    = sigma^2 (измерительный шум = дисперсия AWGN)
x0  = 0.0     начальное состояние
P0  = 0.09    начальная неопределённость = R
```

---

### Генерация тестового сигнала в C++

```cpp
// ── Параметры ─────────────────────────────────────────────────────────────────
const float    fs        = 10e6f;
const float    fdev      = 2e6f;
const uint32_t N         = 16384;
const float    Ti        = (float)N / fs;
const float    mu        = fdev / Ti;           // chirp rate [Hz/s]
const float    bin_hz    = fs / (float)N;       // FFT bin size [Hz]
const float    c         = 3e8f;                // speed of light

const uint32_t n_ant     = 5;
const float    noise_sigma = 0.30f;

// ── Задержки и f_beat ────────────────────────────────────────────────────────
const float tau_us[5]  = {50.0f, 100.0f, 150.0f, 200.0f, 250.0f};   // [мкс]
float tau[5], f_beat[5], range_km[5];
for (uint32_t a = 0; a < n_ant; a++) {
    tau[a]      = tau_us[a] * 1e-6f;
    f_beat[a]   = mu * tau[a];
    range_km[a] = c * tau[a] / 2.0f / 1000.0f;
}

// ── Генерация дечирпованного сигнала (beat tone + AWGN) ──────────────────────
std::vector<std::complex<float>> signal(n_ant * N);
std::mt19937 rng(42);
std::normal_distribution<float> noise(0.0f, noise_sigma);

for (uint32_t ch = 0; ch < n_ant; ch++) {
    const float omega = 2.0f * M_PI * f_beat[ch] / fs;
    for (uint32_t n = 0; n < N; n++) {
        float re = cosf(omega * n) + noise(rng);
        float im = sinf(omega * n) + noise(rng);
        signal[ch * N + n] = {re, im};
    }
}
```

---

### Применение KalmanFilterROCm и оценка SNR

```cpp
// ── Фильтр Калмана ───────────────────────────────────────────────────────────
KalmanFilterROCm kalman(backend);
kalman.SetParams(
    0.001f,   // Q — процессный шум
    0.09f,    // R = sigma^2
    0.0f,     // x0
    0.09f     // P0
);

// GPU processing
auto filtered_result = kalman.ProcessFromCPU(signal, n_ant, N);
// Скопировать результат обратно в CPU: filtered[n_ant * N]

// ── Для оценки f_beat и SNR использовать FFTProcessor ──────────────────────
// FFTProcessor fft_proc(backend);
// fft_proc.Process(InputData{signal_ptr, n_ant, N}, FFTMode::MAGNITUDE)
// → получить |FFT[ch][k]| для каждого канала
// Для raw:      fft_raw[ch][k]
// Для filtered: fft_flt[ch][k]

// ── Вычисление f_beat из FFT (максимум) ─────────────────────────────────────
// f_est_raw[ch] = argmax(fft_raw[ch]) * bin_hz
// f_est_flt[ch] = argmax(fft_flt[ch]) * bin_hz

// ── Вычисление SNR (Signal-to-Noise Ratio) ──────────────────────────────────
// Для каждого канала:
//   peak_mag   = |FFT[peak_bin]|
//   noise_bins = все бины кроме [peak_bin-5 .. peak_bin+5]
//   noise_rms  = sqrt(mean(noise_bins^2))
//   SNR_dB[ch] = 20 * log10(peak_mag / noise_rms)
```

---

### Красивый вывод для отчёта

```cpp
// ── Заголовок ─────────────────────────────────────────────────────────────────
con.Print("\n");
con.Print("╔══════════════════════════════════════════════════════════════╗\n");
con.Print("║       LFM RADAR — KALMAN FILTER DEMONSTRATION               ║\n");
con.Print("╚══════════════════════════════════════════════════════════════╝\n");
con.Print("\n");

snprintf(buf, sizeof(buf),
    "  Параметры ЛЧМ:\n"
    "    fs    = %.1f МГц  |  fdev  = %.1f МГц  |  N     = %u точек\n"
    "    Ti    = %.3f мс   |  mu    = %.1f МГц/с |  Δbin  = %.1f Гц\n"
    "    Шум:  σ = %.2f    |  SNR_raw ≈ %.1f дБ\n\n",
    fs/1e6f, fdev/1e6f, N,
    Ti*1e3f, mu/1e6f, bin_hz,
    noise_sigma, 20.0f*log10f(1.0f/noise_sigma)
);
con.Print(buf);

// ── Таблица целей ─────────────────────────────────────────────────────────────
con.Print("  Цели (линейные задержки):\n");
con.Print("  ─────┬──────────┬──────────┬────────────┬──────────\n");
con.Print("   Ант │ Дальн.   │ tau      │ f_beat     │  бин FFT\n");
con.Print("  ─────┼──────────┼──────────┼────────────┼──────────\n");
for (uint32_t a = 0; a < n_ant; a++) {
    snprintf(buf, sizeof(buf),
        "    %u  │ %5.1f км │ %3.0f мкс │ %6.2f кГц │  #%3.0f\n",
        a, range_km[a], tau_us[a], f_beat[a]/1e3f, f_beat[a]/bin_hz
    );
    con.Print(buf);
}

// ── Таблица результатов Калмана ───────────────────────────────────────────────
con.Print("\n  Параметры Калмана: Q=0.001  R=0.09  x0=0  P0=0.09\n\n");
con.Print("  Ант │ f_beat    │ без Калмана │ с Калманом  │ SNR raw │ SNR flt │  ΔdB\n");
con.Print("  ────┼───────────┼─────────────┼─────────────┼─────────┼─────────┼──────\n");

float total_snr_improvement = 0.0f;
bool all_passed = true;

for (uint32_t a = 0; a < n_ant; a++) {
    // ... вычислить f_est_raw[a], f_est_flt[a], snr_raw[a], snr_flt[a]
    float snr_delta = snr_flt[a] - snr_raw[a];
    total_snr_improvement += snr_delta;

    snprintf(buf, sizeof(buf),
        "   %u  │ %6.2f кГц │  %6.2f кГц │  %6.2f кГц │  %5.1f  │  %5.1f  │ +%.1f\n",
        a,
        f_beat[a]/1e3f,
        f_est_raw[a]/1e3f,
        f_est_flt[a]/1e3f,
        snr_raw[a], snr_flt[a], snr_delta
    );
    con.Print(buf);

    // Проверка: ошибка Калмана < 2 бина
    float err_flt = fabsf(f_est_flt[a] - f_beat[a]);
    if (err_flt > 2.0f * bin_hz) all_passed = false;
}

snprintf(buf, sizeof(buf),
    "\n  Среднее улучшение SNR:  +%.1f дБ\n",
    total_snr_improvement / n_ant
);
con.Print(buf);

// ── Итог ─────────────────────────────────────────────────────────────────────
con.Print("\n");
if (all_passed)
    con.Print("  ✓ Все 5 антенн: ошибка f_beat < 2 бина после Калмана\n");
else
    con.Print("  ✗ FAIL: некоторые антенны превышают допуск\n");
con.Print("╚══════════════════════════════════════════════════════════════╝\n");
```

---

### Ожидаемый вывод (пример)

```
╔══════════════════════════════════════════════════════════════╗
║       LFM RADAR — KALMAN FILTER DEMONSTRATION               ║
╚══════════════════════════════════════════════════════════════╝

  Параметры ЛЧМ:
    fs    = 10.0 МГц  |  fdev  = 2.0 МГц  |  N     = 16384 точек
    Ti    = 1.638 мс  |  mu    = 1220.7 МГц/с  |  Δbin  = 610.4 Гц
    Шум:  σ = 0.30    |  SNR_raw ≈ 10.5 дБ

  Цели (линейные задержки):
  ─────┬──────────┬──────────┬────────────┬──────────
   Ант │ Дальн.   │ tau      │ f_beat     │  бин FFT
  ─────┼──────────┼──────────┼────────────┼──────────
    0  │   7.5 км │  50 мкс  │  61.03 кГц │  # 100
    1  │  15.0 км │ 100 мкс  │ 122.07 кГц │  # 200
    2  │  22.5 км │ 150 мкс  │ 183.10 кГц │  # 300
    3  │  30.0 км │ 200 мкс  │ 244.14 кГц │  # 400
    4  │  37.5 км │ 250 мкс  │ 305.18 кГц │  # 500

  Параметры Калмана: Q=0.001  R=0.09  x0=0  P0=0.09

  Ант │ f_beat    │ без Калмана │ с Калманом  │ SNR raw │ SNR flt │  ΔdB
  ────┼───────────┼─────────────┼─────────────┼─────────┼─────────┼──────
   0  │  61.03 кГц│  60.20 кГц  │  61.03 кГц  │  10.3   │  18.1   │ +7.8
   1  │ 122.07 кГц│ 121.47 кГц  │ 122.07 кГц  │   9.8   │  18.0   │ +8.2
   2  │ 183.10 кГц│ 184.32 кГц  │ 183.10 кГц  │  10.1   │  18.2   │ +8.1
   3  │ 244.14 кГц│ 243.53 кГц  │ 244.14 кГц  │   9.9   │  18.1   │ +8.2
   4  │ 305.18 кГц│ 306.41 кГц  │ 305.18 кГц  │  10.3   │  17.9   │ +7.6

  Среднее улучшение SNR:  +7.98 дБ

  ✓ Все 5 антенн: ошибка f_beat < 2 бина после Калмана
╚══════════════════════════════════════════════════════════════╝
```

### Проверки (ASSERT)

```cpp
// Для каждой антенны:
// 1. Ошибка Калмана < 2 бина = 2 * 610 = 1220 Гц
ASSERT_LT(fabsf(f_est_flt[a] - f_beat[a]), 2.0f * bin_hz, "f_beat error");

// 2. Калман лучше raw
ASSERT_GT(snr_flt[a], snr_raw[a] + 5.0f, "SNR improvement > 5 dB");

// 3. Дальность (косвенная проверка через f_beat):
// R = c * f_beat_flt / (2 * mu), tolerance < 500 м
float R_est = c * f_est_flt[a] / (2.0f * mu);
ASSERT_NEAR(R_est, range_km[a] * 1000.0f, 500.0f, "Range estimation");
```

---

## TASK 21.8 — Оптимизация (по чеклисту `Doc_Addition/Roc hip kernel оптимизация.md`)

**Статус**: ✅ DONE (2026-03-01)

### Kernel-level оптимизации (kalman_kernels_rocm.hpp)

| # | Оптимизация | Было | Стало | Эффект |
|---|------------|------|-------|--------|
| 1 | Калман gain Re | `K_re = P_pred_re / (P_pred_re + R)` (full division ~20 cycles) | `K_re = P_pred_re * __frcp_rn(P_pred_re + R)` (~4 cycles) | **~16 cycles/iter saved** |
| 2 | Калман gain Im | `K_im = P_pred_im / (P_pred_im + R)` | `K_im = P_pred_im * __frcp_rn(P_pred_im + R)` | **~16 cycles/iter saved** |

**Итого**: ~32 cycles saved per point per channel. При 8 каналов × 4096 точек = ~1M cycles.

> **Заметка по точности**: `__frcp_rn()` даёт ~1 ULP ошибку для float32. Для Калмана с типичными значениями `P_pred ∈ [0.1, 25]`, `R ∈ [0.01, 100]` — ошибка пренебрежимо мала. Tolerance GPU vs CPU остаётся < 1e-4f.

### Host-level оптимизации (kalman_filter_rocm.cpp)

| # | Оптимизация | Было | Стало | Эффект |
|---|------------|------|-------|--------|
| 1 | Compile flags | `{"-O3"}` | `{"-O3", "-DWARP_SIZE=32", "--offload-arch=gfxXXXX"}` | **Корректный ISA** |
| 2 | KernelCacheService | Нет кеша | HSACO кешируется через `KernelCacheService(ROCm)` | **~100-200ms saved** |

### Не оптимизировано (осознанно)

| Элемент | Причина |
|---------|---------|
| Steady-state K_ss precompute | Уменьшает точность при коротких сигналах; добавить как опцию при необходимости |
| Нет ring buffer | Калман — чисто рекуррентный (state = 4 float), occupancy максимальная |

---

## Ссылки

- Документация: `Doc_Addition/Filters/2_Kalman_Filter.md`
- Паттерн ROCm: аналогично `modules/filters/src/iir_filter_rocm.cpp`
- LFM параметры: `Doc/Modules/signal_generators/Full.md` (FormParams)
- Дечирп: `modules/heterodyne/` (HeterodyneDechirp)
- Смежный класс: `MovingAverageFilterROCm` (Task_20)
- Связанный класс: `KaufmanFilterROCm` (Task_22) — адаптивный вариант
- **Оптимизация**: `Doc_Addition/Roc hip kernel оптимизация.md`, `Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`
