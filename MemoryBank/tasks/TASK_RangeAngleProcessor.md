# 📋 TASK: RangeAngleProcessor — 3D FFT модуль для антенной решётки

> **Статус**: BACKLOG → IN_PROGRESS → COMPLETED
> **Дата создания**: 2026-03-17  |  **Автор**: Кодо
> **Исполнитель**: *(назначить)*
> **Ревьюер**: Кодо
>
> 📐 **Спецификация**: `MemoryBank/specs/PLAN_range_angle_processor.md`
> 📖 **Теория**: `MemoryBank/specs/2fft_lfm_processing_simple.md`
> 🏛️ **Архитектура**: `Doc_Addition/PLAN/Ref03_Unified_Architecture.md`

---

## 🎯 Кратко: что нужно сделать

Создать модуль `modules/range_angle/` по шаблону `modules/capon/` и `modules/statistics/`.

**Входные данные**: `[256 × 1 300 000]` комплексных IQ-отсчётов (16×16 антенн × 1.3М точек)
**Выход**: 3D куб мощности `[650K × 16 × 16]` (дальность × азимут × элевация) + список целей

**Пайплайн**:
```
[256 × 1.3M] → Dechirp+Hamming → Range FFT (batch=256) → Transpose
             → 2D Beam FFT (hipfftPlanMany [16×16], batch=650K) → |·|² → 3D куб
             → argmax3D → паrabola → (R, θ_az, θ_el)
```

---

## 🔑 Ключевые параметры (вшить в дефолты)

| Параметр | Значение |
|---------|---------|
| n_ant_az = n_ant_el | 16 (итого 256 антенн) |
| n_samples | 1 300 000 |
| f_start / f_end | -5e6 / +5e6 (B=10 МГц, baseband) |
| sample_rate | 12e6 Гц |
| nfft_range | auto → 2^21 = 2 097 152 |
| carrier_freq | 435e6 Гц (LFM 430→440 МГц) |
| antenna_spacing | 0.345 м (λ/2 при 435 МГц) |
| n_range_bins | ~650 000 = B/Δf = 10M / (12M/1.3M) |
| Δθ (угловое разрешение) | 7.2° по каждой оси |
| ΔR (дальностное разрешение) | 15 м |

---

## ✅ Критерии приёмки (DoD — Definition of Done)

- [ ] Проект собирается без ошибок: `cmake .. -DENABLE_ROCM=ON && make -j8`
- [ ] Тест малый PASSED: `./gpu_work_lib range_angle` (128 антенн × 50K точек, 8×8 решётка)
- [ ] Тест большой PASSED: `./gpu_work_lib range_angle bench` (256 антенн × 1.3M точек, 16×16)
- [ ] Дальность совпадает с эталоном: погрешность < 1 бин (15 м)
- [ ] Углы совпадают с эталоном: погрешность < 1 бин (7.2°) или < 0.5° с параболой
- [ ] GPUProfiler отчёт сохранён в `Results/Profiler/range_angle_benchmark.json`
- [ ] Python биндинги работают: `pytest Python_test/range_angle/test_range_angle.py`
- [ ] Графики сохранены в `Results/Plots/range_angle/`

---

## 📁 Файловая структура (создать)

```
modules/range_angle/
├── include/
│   ├── range_angle_processor.hpp     ← Фасад (Layer 6)
│   ├── range_angle_params.hpp        ← Структуры параметров
│   ├── range_angle_types.hpp         ← PeakSearchMode, TargetInfo, Result, shared_buf
│   └── operations/
│       ├── dechirp_window_op.hpp     ← Op: dechirp × conj(ref) + Hamming + zero-pad
│       ├── range_fft_op.hpp          ← Op: hipFFT batch по строкам
│       ├── transpose_op.hpp          ← Op: [n_ant × N_r] → [N_r × n_ant]
│       ├── beam_fft_op.hpp           ← Op: 2D hipFFT + 2D fftshift
│       └── peak_search_op.hpp        ← Op: |·|² + argmax3D + парабола
├── src/
│   ├── range_angle_processor.cpp     ← Реализация фасада
│   ├── dechirp_window_kernel.hip     ← HIP kernel: dechirp+window+zeropad
│   ├── transpose_kernel.hip          ← HIP kernel: tiled transpose 32×32
│   └── fftshift2d_kernel.hip         ← HIP kernel: своп 4 квадрантов [N_az×N_el]
├── tests/
│   ├── all_test.hpp                  ← Точка входа всех тестов модуля
│   ├── test_range_angle_basic.hpp    ← Тест 1: дальность + угол синтетика
│   ├── test_range_angle_benchmark.hpp ← Тест 2: большой benchmarkS
│   └── README.md                     ← Описание тестов
└── CMakeLists.txt
```

---

## 🏗️ ФАЗА 1 — Скелет и параметры

**Файлы**: `range_angle_params.hpp`, `range_angle_types.hpp`, `CMakeLists.txt`, пустой фасад

### 1.1 `include/range_angle_types.hpp`

```cpp
#pragma once
#if ENABLE_ROCM

namespace range_angle {

enum class PeakSearchMode {
  TOP_1,  // один argmax
  TOP_N,  // до n_peaks пиков по порогу
};

struct TargetInfo {
  float range_m;       // дальность, метры
  float angle_az_deg;  // азимут, градусы (±53.1° при d=λ/2, N=16)
  float angle_el_deg;  // элевация, градусы
  float range_bin;     // дробный бин (после параболы)
  float az_bin;
  float el_bin;
  float power_db;
  float snr_db;
};

struct RangeAngleResult {
  bool success = false;
  uint32_t n_range_bins = 0;
  uint32_t n_ant_az     = 0;
  uint32_t n_ant_el     = 0;
  std::vector<float>      power_cube;   // [n_range_bins × n_ant_az × n_ant_el] float32
  void*                   gpu_power_cube = nullptr;
  std::vector<TargetInfo> targets;
  std::string             error_message;
};

namespace shared_buf {
  static constexpr size_t kInput      = 0;
  static constexpr size_t kRef        = 1;
  static constexpr size_t kDechirped  = 2;
  static constexpr size_t kRangeFFT   = 3;
  static constexpr size_t kTransposed = 4;
  static constexpr size_t kBeamFFT    = 5;
  static constexpr size_t kPowerCube  = 6;
  static constexpr size_t kCount      = 7;
}

}  // namespace range_angle
#endif
```

### 1.2 `include/range_angle_params.hpp`

```cpp
#pragma once
#if ENABLE_ROCM
#include "range_angle_types.hpp"
#include <cstdint>

namespace range_angle {

struct RangeAngleParams {
  // 2D URA решётка
  uint32_t n_ant_az  = 16;
  uint32_t n_ant_el  = 16;
  uint32_t n_samples = 1'300'000;
  uint32_t GetNAntennas() const { return n_ant_az * n_ant_el; }

  // ЛЧМ (baseband)
  float f_start      = -5e6f;
  float f_end        = +5e6f;
  float sample_rate  = 12e6f;

  // FFT
  uint32_t nfft_range = 0;  // 0 = auto → следующая 2^n ≥ n_samples

  // Физика антенн
  float antenna_spacing = 0.345f;  // λ/2 при 435 МГц
  float carrier_freq    = 435e6f;  // f_c = (430+440)/2

  // Вычисляемые (заполняет SetParams)
  uint32_t n_range_bins = 0;
  float    range_res_m  = 0.0f;

  // Поиск пиков
  PeakSearchMode peak_mode = PeakSearchMode::TOP_1;
  uint32_t       n_peaks   = 1;

  // Helpers
  float GetBandwidth() const { return f_end - f_start; }
  float GetDuration()  const { return float(n_samples) / sample_rate; }
  float GetChirpRate() const { return GetBandwidth() / GetDuration(); }
};

}  // namespace range_angle
#endif
```

### 1.3 `CMakeLists.txt`

Взять за образец `modules/statistics/CMakeLists.txt`. Ключевые отличия:
- Добавить `hipfft` в `target_link_libraries`
- Добавить `.hip` файлы в `set_source_files_properties(... PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)`

```cmake
if(NOT ROCM_ENABLED)
  return()
endif()

cmake_minimum_required(VERSION 3.21)

set(MODULE_NAME range_angle)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HIP_SOURCES "src/*.hip")

set_source_files_properties(${HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

add_library(${MODULE_NAME} STATIC ${SOURCES} ${HIP_SOURCES})

target_include_directories(${MODULE_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ROCM_PATH}/include
)

target_link_libraries(${MODULE_NAME} PUBLIC
  drv_gpu
  signal_generators
  hipfft
  hip::host
)

target_compile_definitions(${MODULE_NAME} PUBLIC ENABLE_ROCM=1)
```

### 1.4 Добавить в корневой CMakeLists.txt

```cmake
add_subdirectory(modules/range_angle)
```

### 1.5 Пустой фасад (чтобы проект собирался)

`include/range_angle_processor.hpp` — объявление класса без реализации.
`src/range_angle_processor.cpp` — заглушки всех методов.

**Проверка Фазы 1**: `cmake && make` — OK, нет ошибок компиляции.

---

## 🏗️ ФАЗА 2 — Генерация опорного ЛЧМ

**Файлы**: реализация `BuildRefSignal()` в `range_angle_processor.cpp`

### 2.1 Использовать `LfmConjugateGenerator`

📄 **API**: `modules/signal_generators/include/generators/lfm_conjugate_generator.hpp`

```cpp
void RangeAngleProcessor::BuildRefSignal() {
  if (ref_built_) return;  // пересоздавать только при смене params

  // Настраиваем LFM параметры (baseband)
  LfmParams lfm;
  lfm.f_start    = params_.f_start;     // -5e6
  lfm.f_end      = params_.f_end;       // +5e6
  lfm.amplitude  = 1.0f;
  lfm.complex_iq = true;

  SystemSampling samp;
  samp.fs     = params_.sample_rate;    // 12e6
  samp.length = params_.n_samples;      // 1 300 000

  // Генерируем conj(ref_lfm) на GPU
  LfmConjugateGenerator gen(backend_, lfm);
  gen.SetSampling(samp);
  void* d_ref = gen.GenerateToGpu();    // complex<float>[n_samples] на GPU

  // Сохраняем в shared buffer kRef
  size_t ref_bytes = params_.n_samples * sizeof(hipFloatComplex);
  void* d_ref_shared = ctx_.RequireShared(shared_buf::kRef, ref_bytes);
  hipMemcpyDtoDAsync(d_ref_shared, d_ref, ref_bytes, ctx_.stream());

  ref_built_ = true;
}
```

**⚠️ Важно**: `BuildRefSignal()` вызывается один раз при `SetParams()`, не в каждом `Process()`.

---

## 🏗️ ФАЗА 3 — DechirpWindowOp

**Файлы**: `include/operations/dechirp_window_op.hpp`, `src/dechirp_window_kernel.hip`

### 3.1 HIP kernel `dechirp_window_kernel.hip`

```cpp
// dechirp_window_kernel.hip
// Функция: rx[ant, i] * ref_conj[i] * hamming[i], запись в out[ant, i] или 0 если i>=n_samples
// Grid: dim3(ceil(nfft_r/256), n_ant, 1)   Block: dim3(256, 1, 1)

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

__global__ void dechirp_window_kernel(
    const hipFloatComplex* __restrict__ rx,       // [n_ant × n_samples]
    const hipFloatComplex* __restrict__ ref_conj, // [n_samples]
    const float*           __restrict__ window,   // [n_samples] Hamming
    hipFloatComplex*       __restrict__ out,       // [n_ant × nfft_r] (инициализирован нулями!)
    uint32_t n_ant, uint32_t n_samples, uint32_t nfft_r)
{
    uint32_t ant = blockIdx.y;
    uint32_t i   = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant >= n_ant || i >= nfft_r) return;

    hipFloatComplex val = make_hipFloatComplex(0.f, 0.f);
    if (i < n_samples) {
        hipFloatComplex r = rx[ant * n_samples + i];
        hipFloatComplex c = ref_conj[i];
        float w = window[i];
        // dechirp: r * conj(c)  → c уже conj от генератора!
        val = make_hipFloatComplex(
            (r.x * c.x - r.y * c.y) * w,
            (r.x * c.y + r.y * c.x) * w);
    }
    out[ant * nfft_r + i] = val;  // i >= n_samples → 0 (zero-pad)
}
```

### 3.2 `DechirpWindowOp` (Layer 5)

```cpp
// include/operations/dechirp_window_op.hpp
class DechirpWindowOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "DechirpWindow"; }

  // Предвычислить окно Хэмминга при инициализации
  void Initialize(uint32_t n_samples) {
    // Выделить буфер d_window_[n_samples] и заполнить Hamming
    // hamming[i] = 0.54 - 0.46 * cos(2π*i / (n_samples-1))
    // Загрузить через hipMemcpy H2D
  }

  void Execute(uint32_t n_ant, uint32_t n_samples, uint32_t nfft_r) {
    // Запустить dechirp_window_kernel
    // Grid: dim3(ceil(nfft_r/256), n_ant), Block: dim3(256)
    // ctx_->GetShared(kInput), ctx_->GetShared(kRef), d_window_
    // ctx_->RequireShared(kDechirped, n_ant * nfft_r * sizeof(hipFloatComplex))
  }

protected:
  void OnRelease() override { hipFree(d_window_); d_window_ = nullptr; }

private:
  float* d_window_ = nullptr;  // Hamming window [n_samples] на GPU
};
```

---

## 🏗️ ФАЗА 4 — Range FFT Op

**Файлы**: `include/operations/range_fft_op.hpp`

### 4.1 `RangeFftOp` (hipFFT plan 1D, batch)

```cpp
// include/operations/range_fft_op.hpp
#include <hipfft/hipfft.h>

class RangeFftOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "RangeFFT"; }

  // Создать план при инициализации
  void Initialize(uint32_t nfft_r, uint32_t batch /*n_ant*/) {
    hipfftPlan1d(&plan_, (int)nfft_r, HIPFFT_C2C, (int)batch);
    // ⚠️ Привязать plan к stream: hipfftSetStream(plan_, ctx_->stream())
    nfft_r_ = nfft_r;
  }

  // Выполнить FFT + crop до n_range_bins
  void Execute(uint32_t n_ant, uint32_t n_range_bins) {
    void* d_in  = ctx_->GetShared(shared_buf::kDechirped);   // [n_ant × nfft_r]
    void* d_out = ctx_->RequireShared(shared_buf::kRangeFFT,
                  (size_t)n_ant * nfft_r_ * sizeof(hipFloatComplex));

    hipfftExecC2C(plan_,
        (hipfftComplex*)d_in, (hipfftComplex*)d_out, HIPFFT_FORWARD);

    // Crop: kRangeFFT теперь содержит nfft_r столбцов, нам нужны только n_range_bins.
    // Размер буфера уменьшаем при следующем RequireShared (lazy resize down не делаем).
    // Достаточно запомнить что используем [n_ant × n_range_bins] из первых столбцов.
  }

  ~RangeFftOp() { if (plan_) hipfftDestroy(plan_); }

protected:
  void OnRelease() override { if (plan_) { hipfftDestroy(plan_); plan_ = 0; } }

private:
  hipfftHandle plan_ = 0;
  uint32_t nfft_r_   = 0;
};
```

**⚠️ Важно**: `hipfftSetStream(plan_, stream)` — обязательно, иначе FFT будет в default stream!

---

## 🏗️ ФАЗА 5 — Transpose Op

**Файлы**: `include/operations/transpose_op.hpp`, `src/transpose_kernel.hip`

### 5.1 Tiled HIP transpose kernel `transpose_kernel.hip`

```cpp
// Tile 32×32, избегаем bank conflicts через +1 padding
// Вход:  in[n_rows × n_cols]   (n_rows = n_ant, n_cols = n_range_bins)
// Выход: out[n_cols × n_rows]  (= [n_range_bins × n_ant])
// Grid: dim3(ceil(n_cols/32), ceil(n_rows/32))  Block: dim3(32, 32)

__global__ void transpose_kernel(
    const hipFloatComplex* __restrict__ in,
    hipFloatComplex*       __restrict__ out,
    uint32_t n_rows, uint32_t n_cols)
{
    __shared__ hipFloatComplex tile[32][33];  // +1 против bank conflict

    uint32_t x = blockIdx.x * 32 + threadIdx.x;  // col исходной
    uint32_t y = blockIdx.y * 32 + threadIdx.y;  // row исходной

    if (x < n_cols && y < n_rows)
        tile[threadIdx.y][threadIdx.x] = in[y * n_cols + x];
    __syncthreads();

    // Записываем транспонированно
    uint32_t out_x = blockIdx.y * 32 + threadIdx.x;  // col = row исходной
    uint32_t out_y = blockIdx.x * 32 + threadIdx.y;  // row = col исходной
    if (out_x < n_rows && out_y < n_cols)
        out[out_y * n_rows + out_x] = tile[threadIdx.x][threadIdx.y];
}
```

### 5.2 `TransposeOp`

```cpp
// include/operations/transpose_op.hpp
class TransposeOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "Transpose"; }

  void Execute(uint32_t n_rows /*n_ant*/, uint32_t n_cols /*n_range_bins*/) {
    void* d_in = ctx_->GetShared(shared_buf::kRangeFFT);
    void* d_out = ctx_->RequireShared(shared_buf::kTransposed,
                  (size_t)n_cols * n_rows * sizeof(hipFloatComplex));

    dim3 block(32, 32);
    dim3 grid((n_cols + 31) / 32, (n_rows + 31) / 32);
    hipLaunchKernelGGL(transpose_kernel, grid, block, 0, ctx_->stream(),
        (const hipFloatComplex*)d_in, (hipFloatComplex*)d_out, n_rows, n_cols);
  }
};
```

---

## 🏗️ ФАЗА 6 — 2D Beam FFT + fftshift Op

**Файлы**: `include/operations/beam_fft_op.hpp`, `src/fftshift2d_kernel.hip`

### 6.1 `fftshift_2d_kernel.hip`

```cpp
// Свапает 4 квадранта в матрице [n_az × n_el] для каждого range-bin
// Данные: [n_range_bins × n_az × n_el], row-major
// Grid: dim3(n_range_bins, n_az/2, n_el/2)  Block: dim3(1)
// (для n_az=16, n_el=16: grid.y=8, grid.z=8)

__global__ void fftshift2d_kernel(
    hipFloatComplex* data,
    uint32_t n_range_bins, uint32_t n_az, uint32_t n_el)
{
    uint32_t r  = blockIdx.x;   // range bin
    uint32_t az = blockIdx.y;   // 0..n_az/2-1
    uint32_t el = blockIdx.z;   // 0..n_el/2-1

    uint32_t half_az = n_az / 2;
    uint32_t half_el = n_el / 2;

    // 4 пары квадрантов: (az, el) ↔ (az+half_az, el+half_el)
    // и (az, el+half_el) ↔ (az+half_az, el)

    auto idx = [&](uint32_t a, uint32_t e) {
        return r * n_az * n_el + a * n_el + e;
    };

    // Квадрант Q1 ↔ Q4
    hipFloatComplex tmp = data[idx(az, el)];
    data[idx(az, el)] = data[idx(az + half_az, el + half_el)];
    data[idx(az + half_az, el + half_el)] = tmp;

    // Квадрант Q2 ↔ Q3
    tmp = data[idx(az, el + half_el)];
    data[idx(az, el + half_el)] = data[idx(az + half_az, el)];
    data[idx(az + half_az, el)] = tmp;
}
```

### 6.2 `BeamFftOp` с hipfftPlanMany

```cpp
// include/operations/beam_fft_op.hpp
class BeamFftOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "BeamFFT2D"; }

  // Создать 2D FFT план
  void Initialize(uint32_t n_az, uint32_t n_el, uint32_t n_range_bins) {
    int n[2] = { (int)n_az, (int)n_el };
    int total_2d = (int)(n_az * n_el);

    hipfftPlanMany(&plan_, 2, n,
        nullptr, 1, total_2d,   // inembed, istride, idist
        nullptr, 1, total_2d,   // onembed, ostride, odist
        HIPFFT_C2C, (int)n_range_bins);

    hipfftSetStream(plan_, ctx_->stream());
    n_az_ = n_az; n_el_ = n_el; n_range_bins_ = n_range_bins;
  }

  void Execute() {
    void* d_in  = ctx_->GetShared(shared_buf::kTransposed);
    size_t out_bytes = (size_t)n_range_bins_ * n_az_ * n_el_ * sizeof(hipFloatComplex);
    void* d_out = ctx_->RequireShared(shared_buf::kBeamFFT, out_bytes);

    hipfftExecC2C(plan_,
        (hipfftComplex*)d_in, (hipfftComplex*)d_out, HIPFFT_FORWARD);

    // 2D fftshift
    dim3 grid(n_range_bins_, n_az_ / 2, n_el_ / 2);
    hipLaunchKernelGGL(fftshift2d_kernel, grid, dim3(1), 0, ctx_->stream(),
        (hipFloatComplex*)d_out, n_range_bins_, n_az_, n_el_);
  }

  ~BeamFftOp() { if (plan_) hipfftDestroy(plan_); }

protected:
  void OnRelease() override { if (plan_) { hipfftDestroy(plan_); plan_ = 0; } }

private:
  hipfftHandle plan_ = 0;
  uint32_t n_az_ = 0, n_el_ = 0, n_range_bins_ = 0;
};
```

---

## 🏗️ ФАЗА 7 — PeakSearchOp

**Файлы**: `include/operations/peak_search_op.hpp`

### 7.1 |·|² HIP kernel (inplace)

```cpp
// magnitude_sq_kernel: заменяет complex → float (re²+im²)
// Запускать с n_range_bins × n_az × n_el потоками
__global__ void magnitude_sq_kernel(
    const hipFloatComplex* __restrict__ in,
    float* __restrict__ out,
    uint32_t total)  // n_range_bins * n_az * n_el
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = in[i].x * in[i].x + in[i].y * in[i].y;
}
```

### 7.2 CPU: argmax3D + парабола + бин→координаты

После `hipMemcpy` куба `float[n_range_bins × n_az × n_el]` на CPU:

```cpp
// Найти 3D максимум
size_t peak_idx = std::max_element(power_cube, power_cube + total) - power_cube;
uint32_t r_bin  = peak_idx / (n_az * n_el);
uint32_t az_bin = (peak_idx % (n_az * n_el)) / n_el;
uint32_t el_bin =  peak_idx % n_el;

// Параболическая интерполяция по каждой оси
auto parabola = [&](const float* arr, uint32_t k, uint32_t N) -> float {
    if (k == 0 || k == N-1) return float(k);
    float a = arr[k-1], b = arr[k], g = arr[k+1];
    float denom = a - 2*b + g;
    if (fabsf(denom) < 1e-10f) return float(k);
    return k + 0.5f * (a - g) / denom;
};

// Срезы для параболы
std::vector<float> r_slice(n_range_bins), az_slice(n_az), el_slice(n_el);
for (uint32_t i = 0; i < n_range_bins; i++) r_slice[i]  = power_cube[i * n_az * n_el + az_bin * n_el + el_bin];
for (uint32_t i = 0; i < n_az;         i++) az_slice[i] = power_cube[r_bin * n_az * n_el + i * n_el + el_bin];
for (uint32_t i = 0; i < n_el;         i++) el_slice[i] = power_cube[r_bin * n_az * n_el + az_bin * n_el + i];

float r_fine  = parabola(r_slice.data(),  r_bin,  n_range_bins);
float az_fine = parabola(az_slice.data(), az_bin, n_az);
float el_fine = parabola(el_slice.data(), el_bin, n_el);

// Бин → физические координаты
float f_beat     = r_fine * params.sample_rate / params.nfft_range;
float R          = f_beat * 3e8f / (2.f * params.GetChirpRate());

float k_az       = az_fine - params.n_ant_az / 2.f;
float k_el       = el_fine - params.n_ant_el / 2.f;
float sin_az     = k_az * 2.f / float(params.n_ant_az);
float sin_el     = k_el * 2.f / float(params.n_ant_el);
float theta_az   = degrees(asinf(std::clamp(sin_az, -1.f, 1.f)));
float theta_el   = degrees(asinf(std::clamp(sin_el, -1.f, 1.f)));
```

---

## 🏗️ ФАЗА 8 — Сборка фасада

**Файл**: `src/range_angle_processor.cpp`

### 8.1 `SetParams()` — пересчитать вычисляемые поля

```cpp
void RangeAngleProcessor::SetParams(const RangeAngleParams& p) {
  params_ = p;

  // Авто-nfft_range: следующая 2^n ≥ n_samples
  if (params_.nfft_range == 0) {
    uint32_t nfft = 1;
    while (nfft < params_.n_samples) nfft <<= 1;
    params_.nfft_range = nfft;  // 2^21 для N=1.3M
  }

  // Полезных бинов = B / Δf = (f_end-f_start) / (sample_rate/n_samples)
  float df = params_.sample_rate / float(params_.n_samples);
  params_.n_range_bins = uint32_t(params_.GetBandwidth() / df);  // ~650 000

  // Разрешение по дальности
  params_.range_res_m = 3e8f / (2.f * params_.GetBandwidth());  // 15 м

  ref_built_  = false;  // пересоздать опорный сигнал
  compiled_   = false;  // пересоздать FFT планы
}
```

### 8.2 `Process()` — полный пайплайн

```cpp
RangeAngleResult RangeAngleProcessor::Process(
    const std::vector<std::complex<float>>& data, bool download)
{
  EnsureCompiled();  // инициализировать Op-ы и FFT планы
  BuildRefSignal();  // генерировать conj(ref) если ещё нет
  UploadData(data.data(), data.size());  // H2D: data → kInput

  // Pipeline
  dechirp_op_.Execute(params_.GetNAntennas(), params_.n_samples, params_.nfft_range);
  range_fft_op_.Execute(params_.GetNAntennas(), params_.n_range_bins);
  transpose_op_.Execute(params_.GetNAntennas(), params_.n_range_bins);
  beam_fft_op_.Execute();
  peak_op_.Execute(params_, download);

  backend_->Synchronize();

  return peak_op_.GetResult();
}
```

### 8.3 `EnsureCompiled()` — ленивая инициализация

```cpp
void RangeAngleProcessor::EnsureCompiled() {
  if (compiled_) return;

  // Инициализировать dechirp (окно Хэмминга)
  dechirp_op_.Initialize(params_.n_samples);
  dechirp_op_.AttachContext(&ctx_);

  // FFT планы
  range_fft_op_.Initialize(params_.nfft_range, params_.GetNAntennas());
  range_fft_op_.AttachContext(&ctx_);

  transpose_op_.AttachContext(&ctx_);

  beam_fft_op_.Initialize(params_.n_ant_az, params_.n_ant_el, params_.n_range_bins);
  beam_fft_op_.AttachContext(&ctx_);

  peak_op_.AttachContext(&ctx_);

  compiled_ = true;
}
```

---

## 🏗️ ФАЗА 9 — Тесты

### 9.1 `tests/test_range_angle_basic.hpp` — три теста

**Тест 1: дальность одной цели**

```
Синтез данных (CPU):
  - Одна антенна, задержка τ = 1.0 мс
  - ref_lfm[i] = exp(j × π × μ × (i/fs)²)
  - rx[i]      = ref_lfm[i - τ×fs]  (задержанный ЛЧМ)
  - Остальные антенны = rx (угол 0°, все одинаково)

Ожидание:
  - R = c × τ / 2 = 3e8 × 1e-3 / 2 = 150 000 м = 150 км
  - Погрешность < 1 range_bin = 15 м
```

**Тест 2: два ЛЧМ на разных дальностях**

```
rx = signal1(τ=0.5мс) + signal2(τ=1.5мс)
Ожидание: два пика → TOP_N режим находит оба
```

**Тест 3: сигнал под известным углом**

```
Синтез с пространственным сдвигом фазы:
  rx[ant_az, ant_el, i] = rx_base[i] × exp(j × 2π × d/λ × (sin_az × ant_az + sin_el × ant_el))

При θ_az = 14.5° (грубо один бин):
  k_az = round(sin(14.5°) × n_ant_az / 2) ≈ 2
  → пик на az_bin = n_az/2 + k_az = 10

Ожидание:
  |az_peak - 10| < 1
```

### 9.2 `tests/test_range_angle_benchmark.hpp`

```cpp
// Большой тест: 16×16 × 1.3M
// 1. Генерировать синтетику на GPU (LfmConjugateGenerator)
// 2. Process() с download_result=false
// 3. GPUProfiler: SetGPUInfo() → Start() → Process() × 5 → PrintReport() + ExportJSON()
// 4. Сохранить в Results/Profiler/range_angle_benchmark.json

// Малый тест: 8×8 × 50K
// 1. Загрузить с CPU
// 2. Process() с download_result=true
// 3. Сохранить power_cube в Results/JSON/range_angle_small_cube.json
//    Формат: {n_range_bins, n_ant_az, n_ant_el, targets:[{R,az,el,power_db}], cube:[...]}
```

### 9.3 `tests/all_test.hpp`

```cpp
#pragma once
#include "test_range_angle_basic.hpp"
#include "test_range_angle_benchmark.hpp"

inline void RunAllRangeAngleTests(drv_gpu_lib::IBackend* backend) {
  TestRangeAngleBasic(backend);      // T1–T3: basic correctness
  TestRangeAngleBenchmark(backend);  // T4: benchmark 16×16×1.3M
}
```

### 9.4 Подключить в `src/main.cpp`

```cpp
#include "../modules/range_angle/tests/all_test.hpp"
// ...
RunAllRangeAngleTests(backend);
```

---

## 🏗️ ФАЗА 10 — Python биндинги

**Файл**: `python/py_range_angle.hpp`

### 10.1 Структура биндинга

```cpp
#pragma once
#if ENABLE_ROCM
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../modules/range_angle/include/range_angle_processor.hpp"

namespace py = pybind11;
using namespace range_angle;

inline void register_range_angle(py::module& m) {
  py::enum_<PeakSearchMode>(m, "RangeAnglePeakMode")
    .value("TOP_1", PeakSearchMode::TOP_1)
    .value("TOP_N", PeakSearchMode::TOP_N);

  py::class_<RangeAngleParams>(m, "RangeAngleParams")
    .def(py::init<>())
    .def_readwrite("n_ant_az",        &RangeAngleParams::n_ant_az)
    .def_readwrite("n_ant_el",        &RangeAngleParams::n_ant_el)
    .def_readwrite("n_samples",       &RangeAngleParams::n_samples)
    .def_readwrite("f_start",         &RangeAngleParams::f_start)
    .def_readwrite("f_end",           &RangeAngleParams::f_end)
    .def_readwrite("sample_rate",     &RangeAngleParams::sample_rate)
    .def_readwrite("carrier_freq",    &RangeAngleParams::carrier_freq)
    .def_readwrite("antenna_spacing", &RangeAngleParams::antenna_spacing)
    .def_readwrite("peak_mode",       &RangeAngleParams::peak_mode)
    .def_readwrite("n_peaks",         &RangeAngleParams::n_peaks);

  py::class_<TargetInfo>(m, "TargetInfo")
    .def_readonly("range_m",      &TargetInfo::range_m)
    .def_readonly("angle_az_deg", &TargetInfo::angle_az_deg)
    .def_readonly("angle_el_deg", &TargetInfo::angle_el_deg)
    .def_readonly("power_db",     &TargetInfo::power_db)
    .def_readonly("snr_db",       &TargetInfo::snr_db);

  py::class_<RangeAngleResult>(m, "RangeAngleResult")
    .def_readonly("success",      &RangeAngleResult::success)
    .def_readonly("n_range_bins", &RangeAngleResult::n_range_bins)
    .def_readonly("n_ant_az",     &RangeAngleResult::n_ant_az)
    .def_readonly("n_ant_el",     &RangeAngleResult::n_ant_el)
    .def_readonly("targets",      &RangeAngleResult::targets)
    .def_readonly("error_message",&RangeAngleResult::error_message)
    .def("power_cube_numpy", [](const RangeAngleResult& r) {
      // Вернуть numpy array [n_range_bins, n_ant_az, n_ant_el] float32
      std::vector<ssize_t> shape = {
          (ssize_t)r.n_range_bins, (ssize_t)r.n_ant_az, (ssize_t)r.n_ant_el};
      return py::array_t<float>(shape, r.power_cube.data());
    });

  py::class_<RangeAngleProcessor>(m, "RangeAngleProcessor")
    .def(py::init<drv_gpu_lib::IBackend*>())
    .def("set_params", &RangeAngleProcessor::SetParams)
    .def("get_params", &RangeAngleProcessor::GetParams,
         py::return_value_policy::reference_internal)
    .def("process", [](RangeAngleProcessor& p,
                       py::array_t<std::complex<float>> data,
                       bool download) {
      auto buf = data.request();
      std::vector<std::complex<float>> vec(
          static_cast<std::complex<float>*>(buf.ptr),
          static_cast<std::complex<float>*>(buf.ptr) + buf.size);
      return p.Process(vec, download);
    }, py::arg("data"), py::arg("download_result") = true);
}
#endif
```

### 10.2 Подключить в `python/gpu_worklib_bindings.cpp`

```cpp
#if ENABLE_ROCM
#include "py_range_angle.hpp"
// ...
register_range_angle(m);
#endif
```

### 10.3 Python тест `Python_test/range_angle/test_range_angle.py`

```python
import pytest, numpy as np, json
from pathlib import Path
import gpu_worklib as gw

# Тест 1: дальность совпадает с эталоном (малый: 8×8 × 50K)
def test_range_basic():
    ctx = gw.ROCmGPUContext(0)
    proc = gw.RangeAngleProcessor(ctx)

    p = gw.RangeAngleParams()
    p.n_ant_az = 8; p.n_ant_el = 8; p.n_samples = 50_000
    p.f_start = -5e6; p.f_end = 5e6; p.sample_rate = 12e6
    p.carrier_freq = 435e6; p.antenna_spacing = 0.345
    proc.set_params(p)

    # Синтез: задержка τ = 0.5 мс → R = 75 км
    N, fs = 50_000, 12e6
    tau = 0.5e-3; B = 10e6; T = N/fs; mu = B/T
    t = np.arange(N) / fs
    ref_lfm = np.exp(1j * np.pi * mu * t**2)
    delay_samples = int(tau * fs)
    rx = np.zeros(N, dtype=complex)
    rx[delay_samples:] = ref_lfm[:N-delay_samples]

    # Все 64 антенны одинаковый сигнал (угол 0°)
    data = np.tile(rx, (64, 1)).astype(np.complex64)
    result = proc.process(data.flatten(), download_result=True)

    assert result.success
    assert len(result.targets) >= 1
    R_expected = 3e8 * tau / 2  # 75 000 м
    assert abs(result.targets[0].range_m - R_expected) < 500  # < 500 м

# Тест 2: визуализация куба (только если GPU доступен)
def test_cube_visualization():
    # ... аналогично, сохранить PNG срезы
    pass
```

---

## ⚠️ Важные нюансы для исполнителя

### Требования к оформлению кода
- **Google C++ Style Guide** + 2-пробельные отступы (как везде в проекте)
- **Один класс — один файл** (Op в `operations/`, kernel в `src/*.hip`)
- **Логирование**: только `ConsoleOutput::Print(gpu_id, "RangeAngle", msg)` — не `std::cout`
- **Профилирование**: только `GPUProfiler` → `PrintReport()` / `ExportJSON()`
  - ⚠️ `SetGPUInfo()` ПЕРЕД `Start()` — иначе «Unknown» в отчёте!

### Синхронизация stream
- Все операции в одном `ctx_.stream()` → `Synchronize()` только в конце `Process()`
- FFT планы: `hipfftSetStream(plan_, ctx_.stream())` — обязательно!

### Память
- `RequireShared` — аллоцирует (или переиспользует) буфер в `GpuContext`
- Буфер `kDechirped` должен быть инициализирован нулями перед DechirpWindowKernel!
  (или kernel сам пишет 0 для i ≥ n_samples — это уже есть в kernel выше)

### Справочные файлы в проекте
| Что изучить | Файл |
|---|---|
| Как устроен Layer 5 Op | [mean_reduction_op.hpp](modules/statistics/include/operations/mean_reduction_op.hpp) |
| Как работает GpuContext | [gpu_context.hpp](DrvGPU/interface/gpu_context.hpp) |
| Образец фасада | [capon_processor.cpp](modules/capon/src/capon_processor.cpp) |
| Образец CMakeLists | [modules/statistics/CMakeLists.txt](modules/statistics/CMakeLists.txt) |
| HIP оптимизация | [Info_ROCm_HIP_Optimization_Guide.md](Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) |
| Профайлер | [GPUProfiler_SetGPUInfo.md](Examples/GPUProfiler_SetGPUInfo.md) |

---

## 📦 Чеклист сдачи

```
Фаза 1 — Скелет:
  [ ] modules/range_angle/ создан, cmake собирается

Фаза 2 — Ref signal:
  [ ] BuildRefSignal() использует LfmConjugateGenerator
  [ ] conj(ref) сохраняется в kRef

Фаза 3 — DechirpWindow:
  [ ] HIP kernel: dechirp + Hamming + zero-pad
  [ ] Окно Хэмминга предвычислено на GPU

Фаза 4 — RangeFFT:
  [ ] hipfftPlan1d batch=n_ant
  [ ] hipfftSetStream

Фаза 5 — Transpose:
  [ ] Tiled 32×32 transpose, тест на некратных 32

Фаза 6 — 2D BeamFFT:
  [ ] hipfftPlanMany rank=2 [16×16] batch=n_range_bins
  [ ] 2D fftshift (4 квадранта)

Фаза 7 — PeakSearch:
  [ ] |·|² → float куб
  [ ] argmax3D → CPU парабола → бин в координаты
  [ ] TOP_N режим

Фаза 8 — Фасад:
  [ ] SetParams() заполняет вычисляемые поля
  [ ] Process() и ProcessFromGPU()
  [ ] EnsureCompiled() ленивая инициализация
  [ ] GPUProfiler в benchmark

Фаза 9 — Тесты:
  [ ] Тест 1: дальность < 15 м погрешность
  [ ] Тест 2: два пика — найдены оба
  [ ] Тест 3: угловой бин ±1
  [ ] Benchmark 16×16×1.3M + JSON отчёт

Фаза 10 — Python:
  [ ] py_range_angle.hpp зарегистрирован
  [ ] pytest test_range_angle.py PASSED
  [ ] Графики срезов куба сохранены
```

---

*Создано: 2026-03-17 | Кодо | Ревью: Кодо после реализации*
