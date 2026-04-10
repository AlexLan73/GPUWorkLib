# TASK SNR_05: `SnrEstimatorOp` (Layer 5) + `peak_cfar_kernel` + `BranchSelector`

> **Дата**: 2026-04-09
> **Модуль**: `modules/statistics/`
> **Приоритет**: High
> **Статус**: BACKLOG
> **Зависимости**: **[SNR_01](TASK_SNR_01_types.md)** + **[SNR_03](TASK_SNR_03_gather_kernel.md)** + **[SNR_04](TASK_SNR_04_fft_process_to_gpu.md)**
> **Ревьюер**: Кодо
>
> 📐 **План**: разделы **2.2.7**, **2.3**, **2.5** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> ⚠️ **Ядро `peak_cfar` — читать раздел 2.2.7 плана** (добавлен в v4 после ревью)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

1. Написать `SnrEstimatorOp` (Layer 5 Ref03) — pipeline: gather → FFT |X|² → CFAR → median
2. Написать `peak_cfar_kernel` (CA-CFAR + argmax per antenna)
3. Написать отдельный класс `BranchSelector` (stateful, hysteresis, SOLID)

---

## 📁 Файлы (создать)

```
modules/statistics/
├── include/
│   ├── operations/
│   │   └── snr_estimator_op.hpp      ← Layer 5 Op class
│   ├── kernels/
│   │   └── peak_cfar_kernel.hpp       ← CA-CFAR HIP kernel source
│   └── branch_selector.hpp            ← Stateful hysteresis selector
```

---

## 📝 Часть 1 — `peak_cfar_kernel.hpp` (HIP kernel source)

Файл: `modules/statistics/include/kernels/peak_cfar_kernel.hpp`

> ⚠️ **Точный псевдокод — раздел 2.2.7 плана v4.** Здесь приведена эталонная версия.

**Kernel работает с `|X|²`** (input от `MagnitudeOp(squared=true)`).
**Thread mapping:** один блок = одна антенна, BLOCK_SIZE=256.
**search_full_spectrum НЕ передаётся в ядро** — caller передаёт `nFFT` или `nFFT/2` через параметр.

```cpp
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
//
// search_full_spectrum НЕ параметр ядра — caller передаёт
// либо nFFT (поиск по всему спектру) либо nFFT/2 (только [0..N/2]).
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

    // ─── Pass 2: ref-window sum с wraparound (параллельный) ─────
    if (tid == 0) { s_ref_sum = 0.0f; s_ref_count = 0; }
    __syncthreads();

    // Ref индексы: k_peak ± (guard+1 .. guard+ref) mod nFFT
    // Всего 2*ref_bins точек
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
        // Wraparound: +nFFT защищает от отрицательных при малых k_peak
        int k_ref = ((int)k_peak + offset + (int)nFFT) % (int)nFFT;
        atomicAdd(&s_ref_sum, row[k_ref]);
        atomicAdd(&s_ref_count, 1u);
    }
    __syncthreads();

    // ─── Результат (только поток 0) ─────────────────────────────
    if (tid == 0) {
        float noise_mean = (s_ref_count > 0) ? (s_ref_sum / (float)s_ref_count) : 1.0f;
        // Защита от log10(0) и log10(отрицательного)
        float ratio = (noise_mean > 1e-30f) ? (peak / noise_mean) : 1.0f;
        ratio = fmaxf(ratio, 1e-30f);  // вторая защита
        snr_db_out[ant] = 10.0f * __log10f(ratio);
    }
}

)HIP";
}

}  // namespace kernels
}  // namespace statistics
```

**Launch config (в `SnrEstimatorOp::ExecutePeakCfar`):**
```cpp
// search_full_spectrum управляется через параметр nFFT:
unsigned int search_nfft = config.search_full_spectrum ? n_fft : n_fft / 2;

dim3 grid(n_ant_used, 1, 1);
dim3 block(256, 1, 1);
void* args[] = { &mag_sq_ptr, &snr_out_ptr, &search_nfft, &guard, &ref };
hipModuleLaunchKernel(kernel("peak_cfar"), grid.x, 1, 1, block.x, 1, 1, 0, stream(), args, nullptr);
```

---

## 📝 Часть 2 — `snr_estimator_op.hpp` (Layer 5 Op)

Файл: `modules/statistics/include/operations/snr_estimator_op.hpp`

```cpp
#pragma once

/**
 * @file snr_estimator_op.hpp
 * @brief SnrEstimatorOp — Layer 5 Ref03 Op for SNR estimation via CFAR.
 *
 * Pipeline:
 *   gather_decimated → ProcessMagnitudesToGPU(squared=true) → peak_cfar → median
 *
 * @author ...
 * @date 2026-04-09
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"
#include "statistics_types.hpp"
#include "operations/median_radix_sort_op.hpp"
#include "fft_processor_rocm.hpp"

#include <hip/hip_runtime.h>
#include <memory>
#include <stdexcept>

namespace statistics {

class SnrEstimatorOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "SnrEstimator"; }

  /**
   * @brief Setup — создаёт FFT processor и median op
   * @param fft_backend — backend для FFTProcessorROCm (IBackend*)
   */
  void SetupFft(drv_gpu_lib::IBackend* fft_backend) {
    fft_processor_ = std::make_unique<fft_processor::FFTProcessorROCm>(fft_backend);
  }

  /**
   * @brief Execute full SNR pipeline
   * @param gpu_input       Input complex<float> [n_antennas × n_samples] on GPU
   * @param n_antennas      Number of input antennas
   * @param n_samples       Samples per antenna
   * @param config          SnrEstimationConfig (validated!)
   * @param out_result      Result struct (populated by this method)
   */
  void Execute(void* gpu_input,
               uint32_t n_antennas, uint32_t n_samples,
               const SnrEstimationConfig& config,
               SnrEstimationResult& out_result) {
    config.Validate();

    // 1. Compute auto parameters
    uint32_t target_n_fft = config.target_n_fft > 0
        ? config.target_n_fft
        : snr_defaults::kTargetNFft;  // 2048
    uint32_t step_samples = config.step_samples > 0
        ? config.step_samples
        : (n_samples + target_n_fft - 1) / target_n_fft;  // ceil
    uint32_t step_antennas = config.step_antennas > 0
        ? config.step_antennas
        : (n_antennas + snr_defaults::kTargetAntennasMedian - 1)
              / snr_defaults::kTargetAntennasMedian;

    uint32_t n_actual  = n_samples / step_samples;
    uint32_t n_ant_out = (n_antennas + step_antennas - 1) / step_antennas;

    // 2. Validate nFFT vs (guard + ref) — см. SnrEstimationConfig::Validate()
    //    (Дополнительная проверка с фактическим n_actual)
    if (2 * (config.guard_bins + config.ref_bins) + 1 >= n_actual) {
      throw std::invalid_argument("SnrEstimator: ref window >= n_actual");
    }

    // 3. Allocate shared buffers
    size_t gather_bytes = (size_t)n_ant_out * n_actual * sizeof(float) * 2;
    ctx_->RequireShared(shared_buf::kGatherOutput, gather_bytes);

    // nFFT = NextPowerOf2(n_actual) — совпадает с FFTProcessorROCm::CalculateNFFT
    // (repeat_count=1, NextPowerOf2 из fft_processor_rocm.cpp:559)
    // После ProcessMagnitudesToGPU уточняем через fft_processor_->GetNFFT()
    uint32_t n_fft_est = NextPowerOf2(n_actual);
    size_t mag_bytes = (size_t)n_ant_out * n_fft_est * sizeof(float);
    ctx_->RequireShared(shared_buf::kFftMagSquared, mag_bytes);

    ctx_->RequireShared(shared_buf::kSnrPerAntenna,
                        (size_t)n_ant_out * sizeof(float));

    // 4. Stage 1: gather_decimated_kernel
    ExecuteGather(gpu_input, n_antennas, n_samples,
                  step_antennas, step_samples, n_ant_out, n_actual);

    // 5. Stage 2: FFT → |X|² (через FFTProcessorROCm::ProcessMagnitudesToGPU)
    fft_processor::FFTProcessorParams fft_params;
    fft_params.beam_count = n_ant_out;
    fft_params.n_point = n_actual;

    // КРИТИЧНО: window=Hann! решает sinc sidelobes (без окна −27 dB bias).
    // Калибровано в Python Эксп.5 (2026-04-09).
    fft_processor_->ProcessMagnitudesToGPU(
        ctx_->GetShared(shared_buf::kGatherOutput),
        ctx_->GetShared(shared_buf::kFftMagSquared),
        fft_params,
        /*squared=*/true,       // ← square-law (|X|²)
        config.window);         // ← Hann (из SnrEstimationConfig, default)

    // Уточняем фактический nFFT после CalculateNFFT внутри ProcessMagnitudesToGPU
    uint32_t n_fft = fft_processor_->GetNFFT();

    // 6. Stage 3: peak_cfar_kernel → kSnrPerAntenna
    // search_full_spectrum передаётся через параметр nFFT (не в kernel!)
    ExecutePeakCfar(n_ant_out, n_fft,
                    config.guard_bins, config.ref_bins,
                    config.search_full_spectrum);

    // 7. Stage 4: median → kMediansCompact[0]
    //    Переиспользуем MedianRadixSortOp! (он для малых массивов)
    //    ВАЖНО: kSnrPerAntenna → kMagnitudes (D2D копия, или прямо писать в kMagnitudes)
    hipMemcpyAsync(ctx_->GetShared(shared_buf::kMagnitudes),
                   ctx_->GetShared(shared_buf::kSnrPerAntenna),
                   n_ant_out * sizeof(float),
                   hipMemcpyDeviceToDevice, stream());

    median_op_.ExecuteFloat(/*beam_count=*/1, /*n_point=*/n_ant_out);

    // 8. D2H — читаем один float (медиана)
    float median_snr_db = 0.0f;
    hipMemcpyAsync(&median_snr_db,
                   ctx_->GetShared(shared_buf::kMediansCompact),
                   sizeof(float),
                   hipMemcpyDeviceToHost, stream());
    hipStreamSynchronize(stream());

    // 9. Populate result
    out_result.snr_db_global = median_snr_db;
    out_result.used_antennas = n_ant_out;
    out_result.used_bins = n_fft;
    out_result.actual_step_samples = step_samples;
    out_result.n_actual = n_actual;
    // snr_db_per_antenna заполняется опционально (если config требует)
  }

protected:
  /// Вызывается после Attach(&ctx_) — инициализируем дочерние Op'ы
  void OnInitialize() override {
    median_op_.Attach(ctx_);  // разделяем тот же GpuContext (stream, buffers)
  }

  void OnRelease() override {
    fft_processor_.reset();
    median_op_.Release();
  }

private:
  std::unique_ptr<fft_processor::FFTProcessorROCm> fft_processor_;
  MedianRadixSortOp median_op_;

  void ExecuteGather(void* gpu_input,
                     uint32_t n_antennas, uint32_t n_samples,
                     uint32_t step_ant, uint32_t step_samp,
                     uint32_t n_ant_out, uint32_t n_samp_out);

  // search_full_spectrum: если false — передаёт n_fft/2 в kernel как search_nfft
  void ExecutePeakCfar(uint32_t n_ant_out, uint32_t n_fft,
                       uint32_t guard_bins, uint32_t ref_bins,
                       bool search_full_spectrum);

  // Совпадает с FFTProcessorROCm::NextPowerOf2 (fft_processor_rocm.cpp:559)
  // Используется только для pre-allocation kFftMagSquared.
  // После ProcessMagnitudesToGPU фактический nFFT берётся из GetNFFT().
  static uint32_t NextPowerOf2(uint32_t n) {
    if (n == 0) return 1;
    --n;
    n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    return n + 1;
  }
};

}  // namespace statistics

#endif  // ENABLE_ROCM
```

---

## 📝 Часть 3 — `branch_selector.hpp` (stateful)

Файл: `modules/statistics/include/branch_selector.hpp`

```cpp
#pragma once

#include "statistics_types.hpp"

namespace statistics {

/**
 * @brief Stateful branch selector with hysteresis.
 *
 * Caller creates one instance, keeps it alive across multiple SNR measurements.
 * Hysteresis prevents thrashing on threshold boundaries.
 *
 * NOT thread-safe — one selector per thread/pipeline.
 */
class BranchSelector {
public:
  BranchSelector() = default;

  /// Select branch for given SNR with hysteresis. Updates internal state.
  BranchType Select(float snr_db, const BranchThresholds& thr) {
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
        if (snr_db > thr.mid_to_high_db + h) {
          current_ = BranchType::High;
        } else if (snr_db < thr.low_to_mid_db - h) {
          current_ = BranchType::Low;
        }
        break;
      case BranchType::High:
        if (snr_db < thr.mid_to_high_db - h) current_ = BranchType::Mid;
        break;
    }
    return current_;
  }

  BranchType Current() const { return current_; }
  void Reset(BranchType to = BranchType::Low) { current_ = to; }

private:
  BranchType current_ = BranchType::Low;
};

}  // namespace statistics
```

---

## ✅ Definition of Done

- [ ] `peak_cfar_kernel.hpp` создан (HIP kernel source inline)
- [ ] `snr_estimator_op.hpp` создан (Layer 5 Op)
- [ ] `branch_selector.hpp` создан (stateful class)
- [ ] SnrEstimatorOp вызывает gather → `FFTProcessorROCm::ProcessMagnitudesToGPU(squared=true)` → peak_cfar → `MedianRadixSortOp::ExecuteFloat(1, n_ant_out)`
- [ ] Нет нового median kernel — переиспользуется существующий
- [ ] `hipfftExecC2C` НЕ вызывается напрямую — только через FFT processor
- [ ] `BranchSelector` совершенно отдельный от `StatisticsProcessor` (facade остаётся stateless)
- [ ] Код компилируется на Debian (понедельник)
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ `peak_cfar_kernel` работает с `|X|²` (НЕ `|X|`) — `snr_db = 10*log10(peak²/noise²)` где peak² и noise² уже квадраты
- ✅ `peak_cfar_kernel` — ring buffer `((int)k_peak + offset + (int)nFFT) % (int)nFFT` для wraparound
- ✅ `peak_cfar_kernel` НЕ принимает `search_full_spectrum` — caller передаёт `nFFT` или `nFFT/2`
- ✅ `peak_cfar_kernel` — `ratio = fmaxf(ratio, 1e-30f)` перед `__log10f` (двойная защита)
- ✅ `SnrEstimatorOp::Execute` использует `MagnitudeOp(squared=true)` через `FFTProcessorROCm::ProcessMagnitudesToGPU(..., squared=true)`
- ✅ Median через `MedianRadixSortOp::ExecuteFloat(1, n_ant_out)` — НЕ новый kernel
- ✅ `BranchSelector` не зависит от `StatisticsProcessor`, только от `BranchThresholds` (типа из statistics_types.hpp)
- ✅ `SnrEstimationResult` **НЕ содержит** `BranchType` (остаётся в BranchSelector)
- ✅ Auto-вычисления `target_n_fft=0 → 2048`, `step_samples=0 → ceil(n/target)`, `step_antennas=0 → ceil(n/50)`
- ✅ Нет `hipfftExecC2C` напрямую в `SnrEstimatorOp`
- ✅ `BranchSelector::Select` — первая строка `if (!std::isfinite(snr_db)) return current_;`
- ✅ `median_op_.Attach(ctx_)` вызывается в `OnInitialize()` (не в конструкторе!)
- ✅ `NextPowerOf2` реализован inline — совпадает с `FFTProcessorROCm::NextPowerOf2`
- ✅ Фактический `n_fft = fft_processor_->GetNFFT()` после `ProcessMagnitudesToGPU`

---

## 🚫 Запреты

- ❌ НЕ писать новый median kernel — переиспользовать `MedianRadixSortOp::ExecuteFloat`
- ❌ НЕ добавлять `BranchType branch` в `SnrEstimationResult` — это ответственность `BranchSelector`
- ❌ НЕ делать `StatisticsProcessor` stateful — hysteresis только в `BranchSelector`
- ❌ НЕ вызывать `hipfftExecC2C` напрямую — только через `FFTProcessorROCm`

---

## 🔗 Связанные таски

- **Требует:** SNR_01 (типы + WindowType поле), SNR_02 (squared kernel), SNR_02b (WindowType + pad_data_windowed), SNR_03 (gather kernel), SNR_04 (ProcessMagnitudesToGPU с window param)
- **Блокирует:** [SNR_06](TASK_SNR_06_facade.md) (фасад ComputeSnrDb использует этот Op)

---

*Created 2026-04-09 | Кодо*
