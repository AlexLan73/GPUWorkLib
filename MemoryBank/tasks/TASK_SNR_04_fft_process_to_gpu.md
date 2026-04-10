# TASK SNR_04: `FFTProcessorROCm::ProcessMagnitudesToGPU` — новый метод

> **Дата**: 2026-04-09
> **Модуль**: `modules/fft_func/` (расширение)
> **Приоритет**: High
> **Статус**: BACKLOG
> **Зависимости**:
> - **[TASK_SNR_02](TASK_SNR_02_fft_func_squared.md)** (нужен `MagnitudeOp::Execute(..., squared)`)
> - **[TASK_SNR_02b](TASK_SNR_02b_pad_data_windowed.md)** (нужен `WindowType` + `PadDataOp(..., window)`)
> **Ревьюер**: Кодо
>
> 📐 **План**: раздел **2.2.5** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

Добавить в `FFTProcessorROCm` новый публичный метод `ProcessMagnitudesToGPU(...)` который выполняет полный FFT pipeline (`pad → fft → magnitude`) **без D2H копии**, записывая результат `|X|` или `|X|²` в caller-provided GPU буфер.

**Существующий API НЕ ломать** — `ProcessComplex` и `ProcessMagPhase` остаются без изменений.

---

## 📝 Изменения (2 файла)

### Файл 1: `modules/fft_func/include/fft_processor_rocm.hpp`

Добавить в public секцию класса `FFTProcessorROCm`:

```cpp
class FFTProcessorROCm {
public:
  // ... существующие методы ProcessComplex, ProcessMagPhase (НЕ ТРОГАТЬ!) ...

  // =========================================================================
  // Public API -- Magnitudes directly to GPU buffer (no D2H)
  // =========================================================================

  /**
   * @brief Process FFT and write magnitudes directly to caller GPU buffer.
   *
   * Pipeline: PadDataOp(window) → hipfftExecC2C → MagnitudeOp(squared).
   * No D2H copy. Reuses internal BufferSet<4> for pad/fft buffers.
   *
   * @param gpu_data            GPU input [beam_count × n_point × complex<float>]
   * @param gpu_out_magnitudes  GPU output [beam_count × nFFT × float] (caller owns)
   * @param params              FFT params (beam_count, n_point, sample_rate, ...)
   * @param squared             false = |X| (default), true = |X|² (square-law, no sqrt)
   * @param window              WindowType::None (default) / Hann / Hamming / Blackman
   *                            Для SNR-estimator использовать Hann (решает sinc sidelobes).
   * @param prof_events         Optional profiling events collector
   */
  void ProcessMagnitudesToGPU(
      void* gpu_data,
      void* gpu_out_magnitudes,
      const FFTProcessorParams& params,
      bool squared = false,
      WindowType window = WindowType::None,  // ← NEW
      ROCmProfEvents* prof_events = nullptr);

  // ... остальное без изменений ...
};
```

### Файл 2: `modules/fft_func/src/fft_processor_rocm.cpp`

Реализация метода:

```cpp
void FFTProcessorROCm::ProcessMagnitudesToGPU(
    void* gpu_data,
    void* gpu_out_magnitudes,
    const FFTProcessorParams& params,
    bool squared,
    WindowType window,                     // ← NEW
    ROCmProfEvents* prof_events)
{
  // 1. Validate inputs
  if (!gpu_data || !gpu_out_magnitudes) {
    throw std::invalid_argument(
        "ProcessMagnitudesToGPU: null GPU pointer");
  }

  // 2. Ensure kernels compiled (lazy)
  EnsureCompiled();

  // 3. Calculate nFFT from params
  CalculateNFFT(params);  // устанавливает nFFT_ и n_point_
  n_point_ = params.n_point;

  // 4. Allocate internal buffers (kInputBuf, kFftBuf — для pad/fft)
  //    ВАЖНО: kMagPhaseInterleaved НЕ нужен — выходной буфер от caller'а!
  AllocateBuffers(params.beam_count, FFTOutputMode::COMPLEX);

  // 5. Create/reuse hipFFT plan (LRU-2 cache)
  CreateFFTPlan(params.beam_count);

  // 6. Copy input gpu_data → kInputBuf (D2D)
  CopyGpuData(gpu_data, 0,
              (size_t)params.beam_count * params.n_point);

  // 7. Pad: kInputBuf → kFftBuf (zero-padding + optional window)
  //    Для SNR-estimator передаём window=Hann — решает sinc sidelobes
  pad_op_.Execute(bufs_.Get(kInputBuf), bufs_.Get(kFftBuf),
                  params.beam_count, n_point_, nFFT_,
                  window);                    // ← NEW

  // 8. FFT in-place in kFftBuf
  hipfftResult fft_err = hipfftExecC2C(
      plan_,
      (hipfftComplex*)bufs_.Get(kFftBuf),
      (hipfftComplex*)bufs_.Get(kFftBuf),
      HIPFFT_FORWARD);
  if (fft_err != HIPFFT_SUCCESS) {
    throw std::runtime_error("ProcessMagnitudesToGPU: hipfftExecC2C failed");
  }

  // 9. Magnitude: kFftBuf → gpu_out_magnitudes (caller-provided!)
  //    NB: inv_n = 1.0f для raw mag/mag². Caller сам нормирует если нужно.
  size_t total = (size_t)params.beam_count * nFFT_;
  mag_phase_op_.Execute(bufs_.Get(kFftBuf), gpu_out_magnitudes,
                        total, 1.0f, squared);
  // ⚠️ mag_phase_op_ — это существующий MagnitudeOp (см. SNR_02)

  // 10. Optional profiling
  if (prof_events) {
    // Добавить события pad, fft, magnitude через ROCmProfilingData
    // (аналогично ProcessComplex)
  }
}
```

---

## ✅ Definition of Done

- [ ] Метод объявлен в `fft_processor_rocm.hpp` (public секция, после `ProcessMagPhase`)
- [ ] Реализация в `fft_processor_rocm.cpp`
- [ ] Default `squared = false` — семантика совместима с `|X|`
- [ ] Caller владеет `gpu_out_magnitudes` (метод только пишет туда)
- [ ] Внутри используется existing `PadDataOp`, `hipfftExecC2C`, `MagnitudeOp` через `ctx_` и `bufs_`
- [ ] LRU-2 plan cache переиспользуется (как в `ProcessComplex`)
- [ ] Существующие `ProcessComplex` и `ProcessMagPhase` **не изменены**
- [ ] Код компилируется на Debian (понедельник)
- [ ] Существующие тесты `fft_func` проходят (понедельник)
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ НЕТ чтения результата в CPU vector — никакого `ReadMagPhaseResults()`
- ✅ НЕТ `hipMemcpy(..., hipMemcpyDeviceToHost)` в методе
- ✅ Caller-provided `gpu_out_magnitudes` проверен на `nullptr`
- ✅ Переиспользуется existing `PadDataOp`, `MagnitudeOp` (не новые Op'ы!)
- ✅ Нет параметра `gpu_memory_bytes` — рудимент убран (v4 changelog)
- ✅ `MagnitudeOp::Execute(..., squared)` вызывается с параметром
- ✅ `hipfftExecC2C` вызывается через тот же plan management что в `ProcessComplex`
- ✅ При ошибке `hipfft` — понятное исключение с именем метода
- ✅ `params.beam_count` и `params.n_point` используются правильно

---

## 🚫 Запреты

- ❌ НЕ создавать отдельный hipFFT plan — используется `plan_` из класса
- ❌ НЕ менять `BufferSet<kBufCount>` — используется существующий
- ❌ НЕ аллоцировать `gpu_out_magnitudes` внутри метода — caller владеет
- ❌ НЕ делать D2H копию результата

---

## 📝 Заметки для исполнителя

**Про `inv_n = 1.0f`:**
В этом методе мы возвращаем **raw** `|X|` или `|X|²` — caller сам решит нормировать или нет. В SnrEstimatorOp (SNR_05) нормализация не нужна, т.к. CFAR ratio `|X_peak|² / mean(|X_ref|²)` — нормализация сокращается.

**Про `AllocateBuffers`:**
Существующий метод `AllocateBuffers(batch, FFTOutputMode)` выделяет `kInputBuf` + `kFftBuf` + `kMagPhaseInterleaved`. Для `ProcessMagnitudesToGPU` последний буфер **не нужен** — рассмотреть вариант пропустить его через новый параметр `mode` или новый `AllocateBuffersForMagnitudesToGPU(batch)`. Это оптимизация — **можно оставить на потом**, пока пусть аллоцируется, не ломает работу.

---

## 🔗 Связанные таски

- **Требует:** [TASK_SNR_02](TASK_SNR_02_fft_func_squared.md) (параметр `squared` в `MagnitudeOp`)
- **Требует:** [TASK_SNR_02b](TASK_SNR_02b_pad_data_windowed.md) (параметр `window` в `PadDataOp`)
- **Блокирует:** [TASK_SNR_05](TASK_SNR_05_snr_estimator_op.md) (`SnrEstimatorOp` вызывает этот метод с `window=Hann`)

---

*Created 2026-04-09 | Кодо*
