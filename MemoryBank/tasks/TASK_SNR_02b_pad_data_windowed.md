# TASK SNR_02b: `WindowType` enum + расширение `PadDataOp` (Hann/Hamming/Blackman)

> **Дата**: 2026-04-09 (добавлено после Python калибровки)
> **Модуль**: `modules/fft_func` (расширение!)
> **Приоритет**: High (блокирует SNR-estimator)
> **Статус**: BACKLOG
> **Зависимости**: — (параллельно с SNR_02, SNR_03)
> **Ревьюер**: Кодо
>
> 📐 **План**: раздел **2.0.1** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)
>
> 🔴 **КРИТИЧЕСКОЕ ОБОСНОВАНИЕ:** Python моделирование (2026-04-09) показало что CA-CFAR с rectangular-окном даёт **−27 dB bias** из-за sinc sidelobes. Hann window решает полностью — sidelobes −32 dB вместо −13 dB.

---

## 🎯 Цель

Добавить в `fft_func`:
1. Новый enum `WindowType { None, Hann, Hamming, Blackman }`
2. Новый kernel `pad_data_windowed` рядом с существующим `pad_data`
3. Параметр `WindowType window = WindowType::None` в `PadDataOp::Execute()`
4. Параметр `window` в `FFTProcessorROCm::ProcessMagnitudesToGPU()`

**API не ломать** — default `WindowType::None` = rectangular = существующее поведение.

---

## 📝 Изменения (4 файла)

### Файл 1: новый `modules/fft_func/include/types/window_type.hpp`

```cpp
#pragma once

/**
 * @file window_type.hpp
 * @brief Window function types для FFT pre-processing (Hann, Hamming, Blackman).
 *
 * Применяется в PadDataOp перед zero-padding для подавления sinc sidelobes
 * спектра конечного сигнала. Default WindowType::None = rectangular
 * (существующее поведение).
 *
 * @author ...
 * @date 2026-04-09
 */

#include <cstdint>

namespace fft_processor {

/// Window function type
enum class WindowType : uint32_t {
  None     = 0,  ///< rectangular (default, без обработки)
  Hann     = 1,  ///< Hann: w[n] = 0.5*(1 - cos(2π·n/(N-1))), sidelobe -32 dB
  Hamming  = 2,  ///< Hamming: w[n] = 0.54 - 0.46*cos(...), sidelobe -43 dB
  Blackman = 3,  ///< Blackman: three-cos, sidelobe -58 dB
};

}  // namespace fft_processor
```

### Файл 2: расширить `modules/fft_func/include/kernels/fft_processor_kernels_rocm.hpp`

Добавить рядом с существующим `pad_data` новый kernel `pad_data_windowed`:

```cpp
// ═══════════════════════════════════════════════════════════════
// Kernel: pad_data_windowed
// pad_data + window function (Hann/Hamming/Blackman) inline.
// One thread per output element. 2D grid: X=nFFT, Y=beam.
// ═══════════════════════════════════════════════════════════════

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void pad_data_windowed(
    const float2_t* __restrict__ input,   // [beam × n_point]
    float2_t* __restrict__ fft_input,      // [beam × nFFT]
    unsigned int n_point,
    unsigned int nFFT,
    int window_type)                        // WindowType enum (as int)
{
    unsigned int bx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int by = blockIdx.y;
    if (bx >= nFFT) return;

    size_t dst_idx = (size_t)by * nFFT + bx;

    if (bx < n_point) {
        size_t src_idx = (size_t)by * n_point + bx;
        float2_t z = input[src_idx];

        // Применение window (inline, без LUT)
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

        fft_input[dst_idx].x = z.x * w;
        fft_input[dst_idx].y = z.y * w;
    } else {
        // Zero-padding хвоста
        fft_input[dst_idx].x = 0.0f;
        fft_input[dst_idx].y = 0.0f;
    }
}
```

**Важно:**
- Добавить в **обе** функции source: `GetFftProcessorKernelsSource()` и в Combined source (если есть).
- **Не удалять** существующий `pad_data` — он продолжит работать для `window=None`.
- Используется `__cosf` (fast intrinsic), не `cosf` — быстрее.

### Файл 3: расширить `modules/fft_func/include/operations/pad_data_op.hpp`

Добавить параметр `WindowType window` в `Execute()`:

```cpp
#include "types/window_type.hpp"

namespace fft_processor {

class PadDataOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "PadData"; }

  /**
   * @brief Execute zero-padding + optional window function.
   *
   * @param input_buf     Input [beam_count × n_point × complex<float>]
   * @param fft_input_buf Output [beam_count × nFFT × complex<float>]
   * @param beam_count    Number of beams
   * @param n_point       Real samples per beam (before padding)
   * @param nFFT          FFT size (after padding)
   * @param window        Window function (default None = rectangular, legacy)
   */
  void Execute(void* input_buf, void* fft_input_buf,
               size_t beam_count, uint32_t n_point, uint32_t nFFT,
               WindowType window = WindowType::None) {
    // 1. memset output (как в существующей версии)
    hipMemsetAsync(fft_input_buf, 0,
                   beam_count * nFFT * sizeof(float) * 2,
                   stream());

    // 2. Выбор kernel по параметру window
    const char* kernel_name = (window == WindowType::None)
        ? "pad_data"
        : "pad_data_windowed";

    // 3. Launch config — тот же что у pad_data
    unsigned int block_x = 256;
    unsigned int grid_x = (nFFT + block_x - 1) / block_x;
    unsigned int grid_y = static_cast<unsigned int>(beam_count);

    unsigned int np = n_point;
    unsigned int nf = nFFT;
    int w_type = static_cast<int>(window);

    // Args: существующий pad_data принимает (input, output, n_point, nFFT),
    // pad_data_windowed принимает ещё и window_type
    // Можно унифицировать args через always-present window_type
    // (для pad_data игнорируется — но для pad_data_windowed нужен).
    // Альтернатива: две разных ветки void* args[] + два hipModuleLaunchKernel.
    //
    // РЕКОМЕНДАЦИЯ: две ветки чтобы не ломать существующий pad_data:

    if (window == WindowType::None) {
      void* args[] = { &input_buf, &fft_input_buf, &np, &nf };
      hipModuleLaunchKernel(
          kernel("pad_data"),
          grid_x, grid_y, 1, block_x, 1, 1,
          0, stream(), args, nullptr);
    } else {
      void* args[] = { &input_buf, &fft_input_buf, &np, &nf, &w_type };
      hipModuleLaunchKernel(
          kernel("pad_data_windowed"),
          grid_x, grid_y, 1, block_x, 1, 1,
          0, stream(), args, nullptr);
    }
  }
};

}  // namespace fft_processor
```

### Файл 4: `FFTProcessorROCm::ProcessMagnitudesToGPU` — параметр `window`

(Это также часть [TASK_SNR_04](TASK_SNR_04_fft_process_to_gpu.md), но упоминаем здесь для полноты цепочки.)

```cpp
void ProcessMagnitudesToGPU(
    void* gpu_data,
    void* gpu_out_magnitudes,
    const FFTProcessorParams& params,
    bool squared = false,
    WindowType window = WindowType::None,  // ← NEW
    ROCmProfEvents* prof_events = nullptr);
```

Внутри: `pad_op_.Execute(..., window)` — передаём параметр в PadDataOp.

---

## ✅ Definition of Done

- [ ] Файл `modules/fft_func/include/types/window_type.hpp` создан
- [ ] `WindowType` enum определён (None/Hann/Hamming/Blackman)
- [ ] Kernel `pad_data_windowed` добавлен в source (fft_processor_kernels_rocm.hpp)
- [ ] `PadDataOp::Execute(..., WindowType window = WindowType::None)` — параметр добавлен в КОНЕЦ, default None
- [ ] `FFTProcessorROCm::ProcessMagnitudesToGPU` принимает параметр `window`
- [ ] Существующие callers `PadDataOp::Execute(...)` компилируются без изменений (проверить grep)
- [ ] Код компилируется на Debian (понедельник)
- [ ] Существующие тесты `fft_func` проходят без изменений (понедельник)
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ Новый файл `window_type.hpp` в `types/` каталоге
- ✅ `WindowType` — `enum class` (не plain enum)
- ✅ `WindowType::None = 0` (чтобы `0` default работал)
- ✅ Kernel `pad_data_windowed` использует `__cosf` (fast intrinsic), не `cosf`
- ✅ `#define M_PI_F` в начале kernel source (hiprtc не знает M_PI)
- ✅ Zero-padding для `bx >= n_point` идентичен существующему `pad_data`
- ✅ `PadDataOp::Execute` — default `WindowType::None`, existing callers не меняются
- ✅ Две ветки в `PadDataOp::Execute` (одна для `pad_data`, вторая для `pad_data_windowed`) — аргументы разные
- ✅ Существующий kernel `pad_data` НЕ ИЗМЕНЁН
- ✅ `PadDataOp::Name()` возвращает `"PadData"` (общее имя, не `PadDataWindowed`)
- ✅ GetFftProcessorKernelsSource() содержит **оба** kernel'а в одном source

---

## 🚫 Запреты

- ❌ НЕ менять существующий kernel `pad_data` — оставляем как есть
- ❌ НЕ удалять `PadDataOp::Execute(...)` без параметра — добавляем параметр в конец с default
- ❌ НЕ использовать медленный `cosf` — только `__cosf`
- ❌ НЕ делать `if (window_type)` внутри общего pad kernel — это два РАЗНЫХ kernel'а

---

## 📝 Заметки

**Почему не LUT (look-up table)?**
- LUT требует precompute на CPU и upload в GPU memory
- Для наших размеров (n_point ~ 2000..10000) вычисление `__cosf` inline быстрее
- Регистры дешевле memory loads на GPU

**Почему отдельный kernel `pad_data_windowed`, а не `if` внутри `pad_data`?**
- Runtime `if` добавляет branch divergence даже для default случая
- Два kernel'а → компилятор оптимизирует каждый отдельно
- Выбор на host-side (по имени через `kernel()`) — zero overhead

**Processing loss** (чисто информационно):
- Hann:     −1.76 dB
- Hamming:  −1.34 dB
- Blackman: −2.37 dB

Это компенсируется калиброванными порогами (см. `BranchThresholds` в SNR_01).

---

## 🔗 Связанные таски

- **Блокирует:** [TASK_SNR_04](TASK_SNR_04_fft_process_to_gpu.md) (ProcessMagnitudesToGPU передаёт window)
- **Блокирует:** [TASK_SNR_05](TASK_SNR_05_snr_estimator_op.md) (SnrEstimatorOp использует Hann)
- **Параллельно:** [TASK_SNR_02](TASK_SNR_02_fft_func_squared.md) (тот же модуль fft_func, но разные файлы)
- **Python reference:** [`PyPanelAntennas/SNR/cfar_estimator.py:make_window()`](../../PyPanelAntennas/SNR/cfar_estimator.py)

---

*Created 2026-04-09 | Кодо | После Python калибровки*
