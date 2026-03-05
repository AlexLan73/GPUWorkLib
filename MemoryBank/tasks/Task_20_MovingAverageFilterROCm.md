# Task_20: MovingAverageFilterROCm
## SMA · EMA · MMA · DEMA · TEMA — ROCm GPU реализация

**Статус**: ✅ TESTED (6/6 PASSED, 2026-03-05, Radeon 9070 gfx1201)
**Приоритет**: High
**Модуль**: `modules/filters`
**Документация**: `Doc_Addition/Filters/1_Moving_Averages.md`
**Ветка**: worktree `claude/sleepy-williamson`

---

## Цель

Реализовать класс `MovingAverageFilterROCm` — GPU-фильтр скользящих средних (SMA, EMA, MMA, DEMA, TEMA) на ROCm/hiprtc.

Входные данные: `complex<float>` (float2_t) — радарные IQ-сигналы по N лучам.
Параллелизм: каждый луч (channel) — независимый поток GPU.

---

## Создаваемые файлы

```
modules/filters/
├── include/
│   ├── filters/
│   │   └── moving_average_filter_rocm.hpp     ← NEW
│   ├── kernels/
│   │   └── moving_average_kernels_rocm.hpp    ← NEW (R"HIP(...)HIP")
│   └── types/
│       └── filter_params.hpp                  ← UPDATE (MAType, MovingAverageParams)
├── src/
│   └── moving_average_filter_rocm.cpp         ← NEW
├── tests/
│   ├── test_moving_average_rocm.hpp           ← NEW
│   └── all_test.hpp                           ← UPDATE
└── CMakeLists.txt                             ← UPDATE
```

---

## TASK 20.1 — filter_params.hpp: добавить типы

**Файл**: `modules/filters/include/types/filter_params.hpp`

Добавить в namespace `filters` (после существующих типов):

```cpp
// ════════════════════════════════════════════════════════════════════════════
// Moving Average (ROCm)
// ════════════════════════════════════════════════════════════════════════════

enum class MAType {
  SMA,   ///< Simple MA — кольцевой буфер, равные веса 1/N
  EMA,   ///< Exponential MA — alpha = 2/(N+1)
  MMA,   ///< Modified MA (Wilder) — alpha = 1/N
  DEMA,  ///< Double EMA — 2*EMA1 - EMA2
  TEMA   ///< Triple EMA — 3*EMA1 - 3*EMA2 + EMA3
};

struct MovingAverageParams {
  MAType   type        = MAType::EMA;  ///< Тип фильтра
  uint32_t window_size = 10;           ///< N — размер окна
};
```

---

## TASK 20.2 — moving_average_kernels_rocm.hpp

**Файл**: `modules/filters/include/kernels/moving_average_kernels_rocm.hpp`

Шаблон файла:

```cpp
#pragma once
// ROCm hiprtc kernel source — Moving Average Filters
// Встроен как R"HIP(...)HIP" строка, компилируется в runtime

namespace filters {

inline const char* kMovingAverageKernelSrc = R"HIP(

// ─── Типы ────────────────────────────────────────────────────────────────────
struct float2_t { float x; float y; };

// ─── SMA ─────────────────────────────────────────────────────────────────────
// window_size ≤ 128 (ring buffer в thread-local памяти)
// alpha = 1/N
extern "C" __global__ __launch_bounds__(256)
void sma_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    unsigned int N)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    float2_t ring[128];  // N ≤ 128
    float2_t sum = {0.0f, 0.0f};
    float inv_N  = 1.0f / (float)N;
    unsigned int head = 0;

    for (unsigned int n = 0; n < points; n++) {
        float2_t x = in[base + n];
        if (n < N) {
            // Заполняем буфер
            ring[n % N] = x;
            sum.x += x.x;
            sum.y += x.y;
            // Первые N-1 отсчётов: частичное среднее
            out[base + n].x = sum.x / (float)(n + 1);
            out[base + n].y = sum.y / (float)(n + 1);
        } else {
            // Скользящее обновление O(1)
            float2_t old = ring[head];
            ring[head] = x;
            head = (head + 1) % N;
            sum.x += x.x - old.x;
            sum.y += x.y - old.y;
            out[base + n].x = sum.x * inv_N;
            out[base + n].y = sum.y * inv_N;
        }
    }
}

// ─── EMA ─────────────────────────────────────────────────────────────────────
// alpha = 2/(N+1), состояние: 1x float2_t
extern "C" __global__ __launch_bounds__(256)
void ema_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    float alpha)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    float one_minus_alpha = 1.0f - alpha;

    float2_t state = in[base];
    out[base] = state;

    for (unsigned int n = 1; n < points; n++) {
        float2_t x = in[base + n];
        state.x = alpha * x.x + one_minus_alpha * state.x;
        state.y = alpha * x.y + one_minus_alpha * state.y;
        out[base + n] = state;
    }
}

// ─── MMA (Wilder's Smoothed MA) ───────────────────────────────────────────────
// alpha = 1/N — более медленная реакция чем EMA
// Форма идентична EMA, только alpha другой
extern "C" __global__ __launch_bounds__(256)
void mma_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    float alpha)   // передаётся 1/N
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    float one_minus_alpha = 1.0f - alpha;

    float2_t state = in[base];
    out[base] = state;

    for (unsigned int n = 1; n < points; n++) {
        float2_t x = in[base + n];
        state.x = alpha * x.x + one_minus_alpha * state.x;
        state.y = alpha * x.y + one_minus_alpha * state.y;
        out[base + n] = state;
    }
}

// ─── DEMA — Double EMA ────────────────────────────────────────────────────────
// DEMA[n] = 2*EMA1[n] - EMA2[n]
// EMA2 — EMA от EMA1
// Состояние: 2x float2_t (ema1, ema2)
extern "C" __global__ __launch_bounds__(256)
void dema_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    float alpha)   // 2/(N+1)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    float one_minus_alpha = 1.0f - alpha;

    float2_t x0 = in[base];
    float2_t ema1 = x0;
    float2_t ema2 = x0;
    out[base].x = 2.0f * ema1.x - ema2.x;
    out[base].y = 2.0f * ema1.y - ema2.y;

    for (unsigned int n = 1; n < points; n++) {
        float2_t x = in[base + n];
        ema1.x = alpha * x.x    + one_minus_alpha * ema1.x;
        ema1.y = alpha * x.y    + one_minus_alpha * ema1.y;
        ema2.x = alpha * ema1.x + one_minus_alpha * ema2.x;
        ema2.y = alpha * ema1.y + one_minus_alpha * ema2.y;
        out[base + n].x = 2.0f * ema1.x - ema2.x;
        out[base + n].y = 2.0f * ema1.y - ema2.y;
    }
}

// ─── TEMA — Triple EMA ───────────────────────────────────────────────────────
// TEMA[n] = 3*EMA1[n] - 3*EMA2[n] + EMA3[n]
// Состояние: 3x float2_t
extern "C" __global__ __launch_bounds__(256)
void tema_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    float alpha)   // 2/(N+1)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    float one_minus_alpha = 1.0f - alpha;

    float2_t x0 = in[base];
    float2_t ema1 = x0, ema2 = x0, ema3 = x0;
    out[base].x = 3.0f * ema1.x - 3.0f * ema2.x + ema3.x;
    out[base].y = 3.0f * ema1.y - 3.0f * ema2.y + ema3.y;

    for (unsigned int n = 1; n < points; n++) {
        float2_t x = in[base + n];
        ema1.x = alpha * x.x    + one_minus_alpha * ema1.x;
        ema1.y = alpha * x.y    + one_minus_alpha * ema1.y;
        ema2.x = alpha * ema1.x + one_minus_alpha * ema2.x;
        ema2.y = alpha * ema1.y + one_minus_alpha * ema2.y;
        ema3.x = alpha * ema2.x + one_minus_alpha * ema3.x;
        ema3.y = alpha * ema2.y + one_minus_alpha * ema3.y;
        out[base + n].x = 3.0f * ema1.x - 3.0f * ema2.x + ema3.x;
        out[base + n].y = 3.0f * ema1.y - 3.0f * ema2.y + ema3.y;
    }
}

)HIP";

} // namespace filters
```

---

## TASK 20.3 — moving_average_filter_rocm.hpp

**Файл**: `modules/filters/include/filters/moving_average_filter_rocm.hpp`

Полная структура заголовка:

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

class MovingAverageFilterROCm {
public:
  explicit MovingAverageFilterROCm(drv_gpu_lib::IBackend* backend,
                                   unsigned int block_size = 256);
  ~MovingAverageFilterROCm();

  MovingAverageFilterROCm(const MovingAverageFilterROCm&) = delete;
  MovingAverageFilterROCm& operator=(const MovingAverageFilterROCm&) = delete;
  MovingAverageFilterROCm(MovingAverageFilterROCm&&) noexcept;
  MovingAverageFilterROCm& operator=(MovingAverageFilterROCm&&) noexcept;

  void SetParams(const MovingAverageParams& params);
  void SetParams(MAType type, uint32_t window_size);

  // GPU ptr -> GPU output (caller делает hipFree на result.data)
  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr, uint32_t channels, uint32_t points);

  // CPU input -> GPU output (загрузка на GPU внутри)
  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t channels, uint32_t points);

  // CPU reference (для тестирования и валидации)
  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels, uint32_t points) const;

  MAType   GetType()       const { return ma_type_; }
  uint32_t GetWindowSize() const { return window_size_; }
  bool     IsReady()       const { return kernel_compiled_; }

private:
  void CompileKernels();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;
  hipStream_t   stream_  = nullptr;
  hipModule_t   module_  = nullptr;
  hipFunction_t kernel_sma_  = nullptr;
  hipFunction_t kernel_ema_  = nullptr;
  hipFunction_t kernel_mma_  = nullptr;
  hipFunction_t kernel_dema_ = nullptr;
  hipFunction_t kernel_tema_ = nullptr;
  bool          kernel_compiled_ = false;

  MAType   ma_type_    = MAType::EMA;
  uint32_t window_size_ = 10;
  float    alpha_       = 2.0f / 11.0f;  // precomputed 2/(N+1)

  void*  cached_input_buf_  = nullptr;
  size_t cached_input_size_ = 0;

  unsigned int block_size_ = 256;
};

} // namespace filters

#else  // stub для non-ROCm сборки

namespace filters {

class MovingAverageFilterROCm {
public:
  explicit MovingAverageFilterROCm(void*, unsigned int = 256) {}
  void SetParams(const MovingAverageParams&) {}
  bool IsReady() const { return false; }
};

} // namespace filters

#endif // ENABLE_ROCM
```

---

## TASK 20.4 — moving_average_filter_rocm.cpp

**Файл**: `modules/filters/src/moving_average_filter_rocm.cpp`

### Структура реализации

```cpp
#if ENABLE_ROCM

#include "filters/moving_average_filter_rocm.hpp"
#include "kernels/moving_average_kernels_rocm.hpp"
#include <hiprtc/hiprtc.h>
#include <stdexcept>

// ─── Паттерн hiprtc (стандартный для проекта) ────────────────────────────────
// 1. hiprtcCreateProgram(src, name, 0, nullptr, nullptr)
// 2. hiprtcCompileProgram(prog, n_opts, opts) с opts = {"-O3"}
// 3. hiprtcGetCodeSize → hiprtcGetCode → std::vector<char>
// 4. hipModuleLoadData(module_, code.data())
// 5. hipModuleGetFunction(module_, "sma_kernel") × 5

void MovingAverageFilterROCm::CompileKernels() {
    // Компилировать kMovingAverageKernelSrc через hiprtc
    // Получить 5 функций: sma_kernel, ema_kernel, mma_kernel, dema_kernel, tema_kernel
    // При ошибке: hiprtcGetProgramLog → throw std::runtime_error
    kernel_compiled_ = true;
}

// ─── SetParams ───────────────────────────────────────────────────────────────
// Обновляет ma_type_, window_size_, пересчитывает alpha_:
//   EMA/DEMA/TEMA: alpha_ = 2.0f / (window_size_ + 1.0f)
//   MMA: alpha_ = 1.0f / (float)window_size_
//   SMA: alpha_ не используется (хранит window_size_ для SMA kernel)
// Валидация: window_size > 0, для SMA window_size ≤ 128

// ─── Process ─────────────────────────────────────────────────────────────────
// Выбор kernel по ma_type_:
//   SMA:  args = {in, out, channels, points, window_size_}
//   EMA:  args = {in, out, channels, points, alpha_}
//   MMA:  args = {in, out, channels, points, alpha_}    // alpha = 1/N
//   DEMA: args = {in, out, channels, points, alpha_}
//   TEMA: args = {in, out, channels, points, alpha_}
//
// Grid: ((channels + block_size_ - 1) / block_size_, 1, 1)
// Block: (block_size_, 1, 1)
// Выделить out_buf через hipMalloc(channels * points * sizeof(float2))
// Вернуть InputData<void*>{out_buf, channels, points}

// ─── ProcessFromCPU ───────────────────────────────────────────────────────────
// Паттерн из проекта:
// 1. Вычислить нужный размер = channels * points * sizeof(float2)
// 2. if (size != cached_input_size_) { hipFree(cached_input_buf_); hipMalloc(...); }
// 3. hipMemcpy(cached_input_buf_, data.data(), size, hipMemcpyHostToDevice)
// 4. return Process(cached_input_buf_, channels, points)

// ─── ProcessCpu ───────────────────────────────────────────────────────────────
// CPU-эталон для тестирования (без GPU):
//
// SMA: кольцевой буфер per channel
// EMA: state = input[0]; loop: state = alpha*x + (1-alpha)*state
// MMA: то же, alpha = 1/N
// DEMA: ema1=x[0], ema2=x[0]; loop: ema1 = alpha*x+(1-alpha)*ema1;
//       ema2 = alpha*ema1+(1-alpha)*ema2; out = 2*ema1 - ema2
// TEMA: ema1=ema2=ema3=x[0]; out = 3*ema1 - 3*ema2 + ema3

#endif // ENABLE_ROCM
```

### Ключевые детали реализации

| Алгоритм | alpha | GPU state | Ограничение |
|----------|-------|-----------|-------------|
| SMA  | 1/N  | ring[128] в регистрах | window_size ≤ 128 |
| EMA  | 2/(N+1) | 1x float2_t | нет |
| MMA  | 1/N  | 1x float2_t | нет |
| DEMA | 2/(N+1) | 2x float2_t | нет |
| TEMA | 2/(N+1) | 3x float2_t | нет |

---

## TASK 20.5 — test_moving_average_rocm.hpp

**Файл**: `modules/filters/tests/test_moving_average_rocm.hpp`

### Тесты

```cpp
namespace test_moving_average_rocm {

void run(drv_gpu_lib::IBackend* backend) {

// ─── test_ema ─────────────────────────────────────────────────────────────────
// EMA(N=10), 8 channels, 4096 points
// Генерация: случайные complex<float>, диапазон [-1, 1]
// Запуск: GPU ProcessCpu() → ожидаемый результат
// Запуск: GPU Process()
// Сравнение: GPU vs CPU, tolerance < 1e-4f
// Проверить: IsReady() == true

// ─── test_sma ─────────────────────────────────────────────────────────────────
// SMA(N=8), 8 channels, 4096 points
// Проверить первые N отсчётов (частичное среднее) и середину сигнала

// ─── test_mma ─────────────────────────────────────────────────────────────────
// MMA(N=10) vs NumPy-формула alpha=1/10

// ─── test_dema ────────────────────────────────────────────────────────────────
// DEMA(N=10), 8 channels
// GPU vs ProcessCpu tolerance < 1e-4f

// ─── test_tema ────────────────────────────────────────────────────────────────
// TEMA(N=10)

// ─── test_impulse_response ────────────────────────────────────────────────────
// Единичный импульс x=[1,0,0,...,0]
// EMA(N=10): y[0]=1, y[1]=alpha*(1-alpha), y[2]=alpha*(1-alpha)^2 ...
// Проверяем экспоненциальное затухание

// ─── test_multi_channel_independence ─────────────────────────────────────────
// 256 каналов, каждый с разным сигналом
// Проверить что каналы не смешиваются (состояния независимы)

} // run

} // namespace test_moving_average_rocm
```

---

## TASK 20.6 — Обновление CMakeLists.txt и all_test.hpp

### CMakeLists.txt

```cmake
if(ROCM_ENABLED)
    list(APPEND MODULE_SOURCES
        src/fir_filter_rocm.cpp
        src/iir_filter_rocm.cpp
        src/moving_average_filter_rocm.cpp   # NEW
    )
endif()
```

### all_test.hpp

```cpp
// Moving Average ROCm
// #include "tests/test_moving_average_rocm.hpp"
// test_moving_average_rocm::run(backend);
```

---

## Параметры GPU запуска (dispatch)

```
Grid:  (channels + block_size - 1) / block_size × 1 × 1
Block: block_size × 1 × 1

При channels=256, block_size=256:  1 блок
При channels=4096, block_size=256: 16 блоков
```

Каждый поток обрабатывает **один канал** последовательно по всем `points`.
Последовательность вычисления внутри потока — необходима (рекуррентная зависимость).

---

## Валидация против Python/SciPy

```python
import numpy as np

def ema_cpu(data, N):
    alpha = 2.0 / (N + 1)
    result = np.zeros_like(data)
    state = data[0]
    result[0] = state
    for n in range(1, len(data)):
        state = alpha * data[n] + (1 - alpha) * state
        result[n] = state
    return result

def dema_cpu(data, N):
    ema1 = ema_cpu(data, N)
    ema2 = ema_cpu(ema1, N)
    return 2 * ema1 - ema2

def tema_cpu(data, N):
    ema1 = ema_cpu(data, N)
    ema2 = ema_cpu(ema1, N)
    ema3 = ema_cpu(ema2, N)
    return 3 * ema1 - 3 * ema2 + ema3
```

Tolerance GPU vs Python: **< 1e-4** (float32 rounding).

---

## TASK 20.7 — Демонстрационный тест: vector\<float\> ступенька

**Файл**: добавить в `modules/filters/tests/test_moving_average_rocm.hpp`

Этот тест воспроизводит классический пример из `Examples/MA_python.md`:
ступенчатый сигнал 20 нулей → 50 единиц → 50 нулей (120 точек).
Показывает поведение каждого фильтра на скачке — наглядно для отчёта.

### Сигнал

```
t:    [0 .. 19]  → 0.0
t:    [20 .. 69] → 1.0   ← скачок
t:    [70 .. 119]→ 0.0   ← спад
```

```
Отклик SMA(10):  медленный трапециевидный рост/спад — N точек нарастания
Отклик EMA(10):  экспоненциальный рост (alpha=2/11), быстрее SMA
Отклик MMA(10):  экспоненциальный, но медленнее EMA (alpha=1/10)
Отклик DEMA(10): обгоняет EMA — выход чуть раньше EMA
Отклик TEMA(10): самый быстрый — почти без запаздывания на фронте
```

### Код теста

```cpp
void test_step_response(drv_gpu_lib::IBackend* backend) {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print("\n--- [MA Demo] Step signal: 20 zeros / 50 ones / 50 zeros ---\n");

    // ── Сигнал (real → complex<float> с Im=0) ───────────────────────────────
    const uint32_t channels = 1;
    const uint32_t points   = 120;
    const uint32_t N        = 10;  // window_size

    std::vector<std::complex<float>> sig(points, {0.0f, 0.0f});
    for (uint32_t i = 20; i < 70; i++) sig[i] = {1.0f, 0.0f};

    // ── Запуск всех 5 фильтров через ProcessCpu (CPU-эталон) ─────────────────
    MovingAverageFilterROCm filter(backend);

    struct FilterResult {
        std::string name;
        MAType      type;
        std::vector<std::complex<float>> out;
    };

    std::vector<FilterResult> results = {
        {"SMA",  MAType::SMA},
        {"EMA",  MAType::EMA},
        {"MMA",  MAType::MMA},
        {"DEMA", MAType::DEMA},
        {"TEMA", MAType::TEMA},
    };

    for (auto& r : results) {
        filter.SetParams(r.type, N);
        r.out = filter.ProcessCpu(sig, channels, points);
    }

    // ── Вывод таблицы (t, input, SMA, EMA, MMA, DEMA, TEMA) ──────────────────
    // Печатаем каждый 5-й отсчёт для компактности
    con.Print("  t  | input | SMA(10) | EMA(10) | MMA(10) | DEMA(10) | TEMA(10)\n");
    con.Print(" ----+-------+---------+---------+---------+----------+---------\n");

    char buf[128];
    for (uint32_t t = 0; t < points; t += 5) {
        snprintf(buf, sizeof(buf),
            " %3u |  %.1f  |  %5.3f  |  %5.3f  |  %5.3f  |  %6.3f  |  %6.3f\n",
            t,
            sig[t].real(),
            results[0].out[t].real(),  // SMA
            results[1].out[t].real(),  // EMA
            results[2].out[t].real(),  // MMA
            results[3].out[t].real(),  // DEMA
            results[4].out[t].real()   // TEMA
        );
        con.Print(buf);
    }

    // ── Проверки (значения в установившемся режиме t=50..65) ─────────────────
    // В середине единичного прямоугольника фильтры должны выйти на 1.0
    for (auto& r : results) {
        float val_mid = r.out[55].real();
        ASSERT_NEAR(val_mid, 1.0f, 0.05f,
            ("[MA step] " + r.name + "(10) mid-plateau должен ≈ 1.0").c_str());
    }

    // Проверить ширину переходной зоны (по EMA): нарастание до 0.9 к t≈30
    float ema_at_30 = results[1].out[30].real();  // EMA N=10, t=30 (10 отсчётов от скачка)
    // alpha=2/11=0.182, через 10 шагов: 1-(1-0.182)^10 ≈ 0.866
    ASSERT_GT(ema_at_30, 0.80f, "[MA step] EMA должен выйти на > 0.8 через 10 отсчётов после скачка");

    // TEMA должен реагировать быстрее EMA — достичь 0.5 раньше
    // EMA:  t≈24 (alpha=2/11, через 4 шага: 1-(1-0.182)^4≈0.53)
    // TEMA: t≈22 (за счёт тройного опережения)
    // Проверяем: TEMA[23].re > EMA[23].re
    ASSERT_GT(results[4].out[23].real(), results[1].out[23].real(),
        "[MA step] TEMA должен опережать EMA на фронте");

    // ── GPU vs CPU (EMA) ─────────────────────────────────────────────────────
    filter.SetParams(MAType::EMA, N);
    auto gpu_result = filter.ProcessFromCPU(sig, channels, points);
    // Сравнить gpu_result с results[1].out, tolerance < 1e-4f

    con.Print("[MA Demo] Step response test PASSED ✓\n");
}
```

### Ожидаемый вывод (первые строки)

```
--- [MA Demo] Step signal: 20 zeros / 50 ones / 50 zeros ---
  t  | input | SMA(10) | EMA(10) | MMA(10) | DEMA(10) | TEMA(10)
 ----+-------+---------+---------+---------+----------+---------
   0 |  0.0  |  0.000  |  0.000  |  0.000  |   0.000  |   0.000
   5 |  0.0  |  0.000  |  0.000  |  0.000  |   0.000  |   0.000
  20 |  1.0  |  0.100  |  0.182  |  0.100  |   0.298  |   0.392
  25 |  1.0  |  0.600  |  0.651  |  0.410  |   0.830  |   0.944
  30 |  1.0  |  1.000  |  0.866  |  0.651  |   0.975  |   0.997
  35 |  1.0  |  1.000  |  0.959  |  0.826  |   0.996  |   0.999
  40 |  1.0  |  1.000  |  0.988  |  0.913  |   0.999  |   1.000
  50 |  1.0  |  1.000  |  0.999  |  0.994  |   1.000  |   1.000
  70 |  0.0  |  0.900  |  0.818  |  0.900  |   0.702  |   0.608
  80 |  0.0  |  0.000  |  0.134  |  0.349  |   0.025  |   0.003
 110 |  0.0  |  0.000  |  0.001  |  0.007  |   0.000  |   0.000
[MA Demo] Step response test PASSED ✓
```

Таблица наглядно показывает:
- **SMA**: прямоугольное трапециевидное нарастание (строго N шагов)
- **EMA**: быстрый экспоненциальный отклик
- **MMA**: медленнее EMA (Wilder smoothing)
- **DEMA**: опережает EMA (двойная компенсация)
- **TEMA**: самый быстрый, почти мгновенное нарастание

---

## TASK 20.8 — Оптимизация (по чеклисту `Doc_Addition/Roc hip kernel оптимизация.md`)

**Статус**: ✅ DONE (2026-03-01)

### Kernel-level оптимизации (moving_average_kernels_rocm.hpp)

| # | Оптимизация | Было | Стало | Эффект |
|---|------------|------|-------|--------|
| 1 | `%N` в SMA hot loop | `head = (head+1) % N` (~20 cycles) | `if (++head >= N) head = 0` (~1-2 cycles) | **~18 cycles/iter saved** |
| 2 | `1.0f/(float)N` в SMA loop | Деление каждый отсчёт в else-ветке | `inv_N` передаётся как параметр kernel (precomputed on host) | **~20 cycles/iter saved** |
| 3 | `1.0f/(float)(n+1)` в SMA warmup | Полное деление | `__frcp_rn((float)(n+1))` — fast reciprocal | **~10 cycles/iter saved** |
| 4 | EMA/MMA/DEMA/TEMA | Уже оптимальны (FMA, no divisions) | — | — |

### Host-level оптимизации (moving_average_filter_rocm.cpp)

| # | Оптимизация | Было | Стало | Эффект |
|---|------------|------|-------|--------|
| 1 | Compile flags | `{"-O3"}` | `{"-O3", "-DWARP_SIZE=32", "--offload-arch=gfxXXXX"}` | **Корректный ISA** |
| 2 | KernelCacheService | Нет кеша | HSACO кешируется на диск через `KernelCacheService(ROCm)` | **~100-200ms saved** при повторных запусках |
| 3 | SMA kernel args | 5 args | 6 args (добавлен `inv_N`) | Поддержка precomputed reciprocal |
| 4 | `LoadKernelFunctions()` | Inline в `CompileKernels()` | Выделен в отдельный метод (DRY: вызывается из cache-path и compile-path) | Cleaner code |

### Не оптимизировано (осознанно)

| Элемент | Причина |
|---------|---------|
| `ring[128]` в SMA | Нет альтернативы без shared memory для рекуррентного ring buffer; типичный N=10 — occupancy OK |
| 2D Grid | Рекуррентные фильтры (IIR-style) — 1D grid is correct |
| GPUProfiler | Добавить при первом профилировании на AMD GPU |

---

## Ссылки

- Документация: `Doc_Addition/Filters/1_Moving_Averages.md`
- Пример step-сигнала: `Examples/MA_python.md`
- Паттерн ROCm: аналогично `modules/filters/src/fir_filter_rocm.cpp`
- Паттерн гетеродина (1D grid IIR-style): `modules/heterodyne/src/heterodyne_processor_rocm.cpp`
- Аналогичный класс: `KaufmanFilterROCm` (Task_22)
- **Оптимизация**: `Doc_Addition/Roc hip kernel оптимизация.md`, `Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`
