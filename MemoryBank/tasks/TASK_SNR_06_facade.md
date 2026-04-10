# TASK SNR_06: `ComputeSnrDb` в `StatisticsProcessor` (Facade, stateless)

> **Дата**: 2026-04-09
> **Модуль**: `modules/statistics/`
> **Приоритет**: High
> **Статус**: BACKLOG
> **Зависимости**: **[SNR_05](TASK_SNR_05_snr_estimator_op.md)** (SnrEstimatorOp нужен)
> **Ревьюер**: Кодо
>
> 📐 **План**: раздел **2.4** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

Добавить в `StatisticsProcessor` (Layer 6 Facade) два метода `ComputeSnrDb` (CPU-данные и GPU-данные). Facade **остаётся stateless** — hysteresis живёт отдельно в `BranchSelector`.

---

## 📝 Изменения (2 файла)

### Файл 1: `modules/statistics/include/statistics_processor.hpp`

Добавить в public секцию после существующих методов:

```cpp
class StatisticsProcessor {
public:
  // ... существующие методы ...

  // =========================================================================
  // Public API -- SNR estimation (NEW)
  // =========================================================================

  /**
   * @brief Compute SNR (dB) from CPU data via CA-CFAR
   *
   * Pipeline: upload → gather → FFT |X|² → CFAR → median
   *
   * @param data        CPU complex<float> [n_antennas × n_samples] (row-major)
   * @param n_antennas  Number of antennas
   * @param n_samples   Samples per antenna
   * @param config      SNR estimation config (see snr_defaults)
   * @return Result with snr_db_global, used_antennas, used_bins, actual_step_samples
   *
   * NB: Result does NOT contain BranchType — use BranchSelector for branching.
   */
  SnrEstimationResult ComputeSnrDb(
      const std::vector<std::complex<float>>& data,
      uint32_t n_antennas,
      uint32_t n_samples,
      const SnrEstimationConfig& config);

  /**
   * @brief Compute SNR (dB) from GPU data (production path)
   *
   * Pipeline: gather → FFT |X|² → CFAR → median (данные уже на GPU!)
   *
   * @param gpu_data    GPU complex<float> [n_antennas × n_samples] (row-major)
   * @param n_antennas  Number of antennas
   * @param n_samples   Samples per antenna
   * @param config      SNR estimation config
   * @return Result (see above)
   */
  SnrEstimationResult ComputeSnrDb(
      void* gpu_data,
      uint32_t n_antennas,
      uint32_t n_samples,
      const SnrEstimationConfig& config);

private:
  // ... существующие приватные поля ...

  // NEW для SNR-estimator:
  drv_gpu_lib::IBackend* backend_ = nullptr;  // ← сохраняем из конструктора
  SnrEstimatorOp snr_estimator_op_;            // Layer 5 Op
  bool snr_op_initialized_ = false;            // ← вместо HasContext() (которого нет!)
};
```

**Добавить include в верху файла:**
```cpp
#include "operations/snr_estimator_op.hpp"
```

**В конструкторе `StatisticsProcessor` сохранить `backend_`:**
```cpp
StatisticsProcessor::StatisticsProcessor(drv_gpu_lib::IBackend* backend)
    : ctx_(backend), backend_(backend), ...
```

### Файл 2: `modules/statistics/src/statistics_processor.cpp`

Реализации методов + memory check:

```cpp
// =========================================================================
// SNR estimation
// =========================================================================

namespace {
  /// Проверка что данные поместятся в VRAM с запасом (см. Q-6 в ревью)
  void CheckVramAvailable(size_t required_bytes, const std::string& context) {
    size_t free_vram = 0, total_vram = 0;
    hipError_t err = hipMemGetInfo(&free_vram, &total_vram);
    if (err != hipSuccess) {
      // Если хип не отвечает — пропускаем проверку
      return;
    }
    if (required_bytes > free_vram * 8 / 10) {
      throw std::runtime_error(
          context + ": need " +
          std::to_string(required_bytes / (1024 * 1024)) + " MB, only " +
          std::to_string(free_vram / (1024 * 1024)) + " MB free VRAM");
    }
  }
}

SnrEstimationResult StatisticsProcessor::ComputeSnrDb(
    const std::vector<std::complex<float>>& data,
    uint32_t n_antennas, uint32_t n_samples,
    const SnrEstimationConfig& config)
{
  if (data.size() != (size_t)n_antennas * n_samples) {
    throw std::invalid_argument(
        "ComputeSnrDb (CPU): data.size() != n_antennas * n_samples");
  }

  // Validate config early
  config.Validate();

  // Memory check
  size_t input_bytes = data.size() * sizeof(std::complex<float>);
  // +gather + mag² buffers (grubo ~2x input для запаса)
  CheckVramAvailable(input_bytes * 2, "ComputeSnrDb (CPU)");

  EnsureCompiled();

  // Upload to kInput
  UploadComplexData(data.data(), data.size());

  // Delegate to GPU path
  return ComputeSnrDb(
      ctx_.GetShared(shared_buf::kInput),
      n_antennas, n_samples, config);
}

SnrEstimationResult StatisticsProcessor::ComputeSnrDb(
    void* gpu_data,
    uint32_t n_antennas, uint32_t n_samples,
    const SnrEstimationConfig& config)
{
  if (!gpu_data) {
    throw std::invalid_argument("ComputeSnrDb (GPU): null gpu_data");
  }

  config.Validate();

  // Memory check — только промежуточные scratch буферы.
  // НЕ учитываем n_antennas * n_samples — gpu_data уже на GPU у caller'а,
  // двойной учёт даст ложную ошибку!
  //
  // Оценка: n_ant_used * n_fft_est * 3 (gather complex + mag_sq float + snr float)
  // Грубо: n_antennas/50 антенн, max 4096 bins
  uint32_t n_ant_used_est = (n_antennas + 49u) / 50u;
  size_t est_scratch =
      (size_t)n_ant_used_est * n_samples / 50u * sizeof(float) * 2  // kGatherOutput (est)
    + (size_t)n_ant_used_est * 4096u * sizeof(float)                 // kFftMagSquared
    + (size_t)n_ant_used_est * sizeof(float);                         // kSnrPerAntenna
  CheckVramAvailable(est_scratch, "ComputeSnrDb (GPU)");

  EnsureCompiled();

  // Ленивая инициализация snr_estimator_op (один раз)
  if (!snr_op_initialized_) {
    snr_estimator_op_.Attach(&ctx_);
    snr_estimator_op_.SetupFft(backend_);
    snr_op_initialized_ = true;
  }

  SnrEstimationResult result;
  snr_estimator_op_.Execute(gpu_data, n_antennas, n_samples, config, result);
  return result;
}
```

---

## ✅ Definition of Done

- [ ] 2 новых метода `ComputeSnrDb` в `StatisticsProcessor` (CPU + GPU overload)
- [ ] CPU overload делегирует в GPU overload после upload
- [ ] Проверка памяти через `hipMemGetInfo` с `throw std::runtime_error` при нехватке
- [ ] `config.Validate()` вызывается в начале обоих методов
- [ ] `SnrEstimatorOp` добавлен как private поле `snr_estimator_op_`
- [ ] `StatisticsProcessor` остаётся **stateless** (нет поля `prev_branch`)
- [ ] Существующие методы `ComputeMean`, `ComputeMedian`, `ComputeStatistics`, `ComputeAll` не изменены
- [ ] Код компилируется на Debian (понедельник)
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ Нет поля `prev_branch` или аналогичного state в `StatisticsProcessor`
- ✅ Нет вызова `BranchSelector` внутри facade — caller применяет отдельно
- ✅ `config.Validate()` вызван в обоих overload
- ✅ Memory check бросает `std::runtime_error` (не silent-failure)
- ✅ CPU overload делает `UploadComplexData` + делегирует в GPU overload
- ✅ `snr_op_initialized_` флаг вместо `HasContext()` (метода не существует!)
- ✅ `backend_` сохранён в конструкторе (нужен для `SetupFft`)
- ✅ GPU overload memory check НЕ учитывает `n_antennas * n_samples` (double-count!)
- ✅ Нет утечек: если throw — ранее выделенные буферы освобождаются через RAII (`ctx_`)

---

## 🚫 Запреты

- ❌ НЕ добавлять `BranchType` в `SnrEstimationResult`
- ❌ НЕ хранить `BranchSelector` внутри `StatisticsProcessor` (facade stateless!)
- ❌ НЕ дублировать SNR-логику — вся в `SnrEstimatorOp`
- ❌ НЕ пропускать проверку памяти ради простоты

---

## 🔗 Связанные таски

- **Требует:** [SNR_05](TASK_SNR_05_snr_estimator_op.md)
- **Блокирует:** [SNR_07](TASK_SNR_07_python_bindings.md), [SNR_08](TASK_SNR_08_cpp_tests.md), [SNR_09](TASK_SNR_09_benchmark.md)

---

*Created 2026-04-09 | Кодо*
