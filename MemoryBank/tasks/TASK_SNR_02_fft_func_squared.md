# TASK SNR_02: `complex_to_magnitude_squared` + параметр `squared` в MagnitudeOp

> **Дата**: 2026-04-09
> **Модуль**: `modules/fft_func` (расширение!)
> **Приоритет**: High
> **Статус**: BACKLOG
> **Зависимости**: — (параллельно с SNR_00, SNR_01, SNR_03)
> **Ревьюер**: Кодо
>
> 📐 **План**: раздел **2.0** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

Добавить в `fft_func` новый kernel `complex_to_magnitude_squared` (БЕЗ `sqrt`, ~7× быстрее) и параметр `bool squared = false` в существующий `MagnitudeOp::Execute()`. **API не ломать** — default сохраняет текущее поведение.

---

## 📝 Изменения (2 файла)

### Файл 1: `modules/fft_func/include/kernels/complex_to_mag_phase_kernels_rocm.hpp`

Добавить новый kernel в **обе** inline функции — `GetComplexToMagnitudeKernelSource()` и `GetCombinedC2MPKernelSource()`:

```cpp
// ═══════════════════════════════════════════════════════════════
// Kernel: complex_to_magnitude_squared
// Converts complex data to |X|² * inv_n (power spectrum, no sqrt).
// ~7× faster than complex_to_magnitude — no transcendental.
// One thread per element. 1D grid.
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void complex_to_magnitude_squared(
    const float2_t* __restrict__ input,   // Complex input: {re, im}
    float* __restrict__ output,           // Float output: (re² + im²) * inv_n
    float inv_n,                          // Normalization factor (same as non-squared)
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    output[gid] = (z.x * z.x + z.y * z.y) * inv_n;  // NO sqrt!
}
```

**Важно:** добавить в **обе** функции (`GetComplexToMagnitudeKernelSource` + `GetCombinedC2MPKernelSource`), иначе Combined compile не увидит новый kernel.

### Файл 2: `modules/fft_func/include/operations/magnitude_op.hpp`

Модифицировать `MagnitudeOp::Execute()`:

```cpp
class MagnitudeOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "Magnitude"; }

  /**
   * @brief Execute complex → magnitude conversion
   * @param input          Device pointer complex<float>
   * @param output         Device pointer float
   * @param total_elements beam_count × n_point
   * @param inv_n          Normalization factor
   * @param squared        false = |X| (default, legacy), true = |X|² (no sqrt, ~7× faster)
   */
  void Execute(void* input, void* output,
               size_t total_elements, float inv_n,
               bool squared = false) {                       // ← NEW parameter
    unsigned int total = static_cast<unsigned int>(total_elements);
    unsigned int block_size = 256;
    unsigned int grid_size = (total + block_size - 1) / block_size;

    void* args[] = { &input, &output, &inv_n, &total };

    const char* kernel_name = squared
        ? "complex_to_magnitude_squared"
        : "complex_to_magnitude";

    hipError_t err = hipModuleLaunchKernel(
        kernel(kernel_name),                                  // ← dynamic name
        grid_size, 1, 1,
        block_size, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MagnitudeOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }
};
```

---

## ✅ Definition of Done

- [ ] Kernel `complex_to_magnitude_squared` добавлен в **обе** inline функции в `complex_to_mag_phase_kernels_rocm.hpp`
- [ ] `MagnitudeOp::Execute(..., bool squared = false)` — параметр добавлен в конец, default `false`
- [ ] Существующие вызовы `MagnitudeOp::Execute(...)` компилируются без изменений (проверить через grep)
- [ ] `.cl` OpenCL версия (`C2MP_kernels.cl`) **НЕ трогается**
- [ ] Код компилируется на Debian (в понедельник)
- [ ] Существующие тесты `fft_func` проходят без изменений (понедельник)
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ Новый kernel **БЕЗ** `sqrt`/`sqrtf`/`__fsqrt_rn` — только `z.x*z.x + z.y*z.y`
- ✅ `__restrict__` и `__launch_bounds__(BLOCK_SIZE)` как у соседних kernel'ов
- ✅ `struct float2_t` не дублируется (используется существующий)
- ✅ `inv_n` параметр — тот же самый для обоих kernel'ов
- ✅ Kernel добавлен в **обе** функции source (`GetComplexToMagnitudeKernelSource` + `GetCombinedC2MPKernelSource`)
- ✅ Default `squared=false` — existing callers не меняются
- ✅ Выбор kernel по имени на host-side (zero runtime overhead внутри kernel)
- ✅ OpenCL `.cl` файл НЕ изменён

---

## 🚫 Запреты

- ❌ НЕ менять сигнатуру существующих вызовов `MagnitudeOp::Execute(...)` — только добавить параметр в конец
- ❌ НЕ трогать OpenCL `.cl` файлы (ветка `main` ROCm only)
- ❌ НЕ использовать `if (squared)` **внутри** kernel — два отдельных kernel'а!

---

## 🔗 Связанные таски

- **Блокирует:** [TASK_SNR_04](TASK_SNR_04_fft_process_to_gpu.md) — `ProcessMagnitudesToGPU` использует `squared=true`
- **Параллельно:** SNR_00, SNR_01, SNR_03

---

*Created 2026-04-09 | Кодо*
