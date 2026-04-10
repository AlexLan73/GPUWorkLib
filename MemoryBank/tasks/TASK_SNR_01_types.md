# TASK SNR_01: Дополнить `statistics_types.hpp` — типы SNR-estimator

> **Дата**: 2026-04-09
> **Модуль**: `modules/statistics/include/statistics_types.hpp`
> **Приоритет**: High
> **Статус**: BACKLOG
> **Зависимости**: — (можно параллельно с SNR_00)
> **Ревьюер**: Кодо
>
> 📐 **План**: раздел **2.1** и **2.1.1** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

**Дополнить** существующий файл `modules/statistics/include/statistics_types.hpp` новыми типами для SNR-estimator'а. Файл **не переписывать**, только добавить.

---

## 📝 Что добавить (точный код — в плане, раздел 2.1)

### 1. Namespace `snr_defaults` — константы (ОБНОВЛЕНО после Python калибровки)

```cpp
#include "types/window_type.hpp"  // из SNR_02 (расширение PadDataOp)

namespace snr_defaults {
  static constexpr uint32_t kTargetNFft           = 2048;  // гибкий, не догма 1024
  static constexpr uint32_t kGuardBins            = 5;     // было 3, калибровано для Hann
  static constexpr uint32_t kRefBins              = 16;    // было 8, калибровано для Hann
  static constexpr uint32_t kTargetAntennasMedian = 50;
  static constexpr float    kHysteresisDb         = 2.0f;

  // NEW: window function default (из Python Эксп.0)
  // Hann — sidelobes −32 dB, решает проблему sinc (rect даёт −27 dB bias!)
  static constexpr fft_processor::WindowType kDefaultWindow =
      fft_processor::WindowType::Hann;
}
```

### 2. `BranchType` enum + `BranchThresholds` struct (КАЛИБРОВАННЫЕ значения)

```cpp
enum class BranchType { Low, Mid, High };

struct BranchThresholds {
  // Калибровано в Python Эксп.5: Hann + mean, P_correct = 97.9%
  // Источник: PyPanelAntennas/SNR/results/exp5_thresholds.json
  float low_to_mid_db  = 15.0f;  // было 6.0, обновлено 2026-04-09
  float mid_to_high_db = 30.0f;  // было 12.0, обновлено 2026-04-09
  float hysteresis_db  = snr_defaults::kHysteresisDb;
};
```

### 3. `SnrEstimationConfig` struct (NEW: поле `window`)

```cpp
struct SnrEstimationConfig {
  uint32_t target_n_fft  = 0;   // 0 → auto (default 2048)
  uint32_t step_samples  = 0;   // 0 → auto из target_n_fft
  uint32_t step_antennas = 0;   // 0 → ceil(n_antennas / kTargetAntennasMedian)
  uint32_t guard_bins = snr_defaults::kGuardBins;  // default 5 (было 3)
  uint32_t ref_bins   = snr_defaults::kRefBins;    // default 16 (было 8)
  bool     search_full_spectrum = true;

  // NEW: Window function — Hann по умолчанию (решает sinc sidelobes)
  fft_processor::WindowType window = snr_defaults::kDefaultWindow;

  bool     with_dechirp = false;
  BranchThresholds thresholds;

  void Validate() const;  // см. реализацию в плане
};
```

### 4. `SnrEstimationResult` struct — БЕЗ BranchType!

```cpp
struct SnrEstimationResult {
  float snr_db_global;                    // медиана по антеннам
  std::vector<float> snr_db_per_antenna;  // per-antenna (может быть пустой)
  uint32_t used_antennas;
  uint32_t used_bins;                     // реальный nFFT (после padding)
  uint32_t actual_step_samples;
  uint32_t n_actual;                      // n_samples / step (до padding)
  // НЕТ BranchType — см. BranchSelector класс (SNR_05)
};
```

### 5. Расширить `shared_buf` slots

```cpp
namespace shared_buf {
  // СУЩЕСТВУЮЩИЕ (не трогать):
  static constexpr size_t kInput          = 0;
  static constexpr size_t kMagnitudes     = 1;
  static constexpr size_t kResult         = 2;
  static constexpr size_t kMediansCompact = 3;

  // НОВЫЕ для SNR-estimator:
  static constexpr size_t kGatherOutput   = 4;  // complex gather
  static constexpr size_t kFftMagSquared  = 5;  // float |X|²
  static constexpr size_t kSnrPerAntenna  = 6;  // float SNR_db

  static constexpr size_t kCount          = 7;  // было 4, стало 7
}
```

---

## ✅ Definition of Done

- [ ] Все 5 новых типов добавлены в `statistics_types.hpp`
- [ ] `shared_buf::kCount` обновлён с `4` на `7`
- [ ] Добавить `#include <stdexcept>` если нужно для `Validate()`
- [ ] `SnrEstimationConfig::Validate()` реализован (код в плане L378-388)
- [ ] Существующие типы (`StatisticsParams`, `MeanResult`, `MedianResult`, `StatisticsResult`, `FullStatisticsResult`) **не изменены**
- [ ] Код компилируется (на Debian в понедельник)
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ `target_n_fft` — lowercase, НЕ `target_N_fft`
- ✅ `search_full_spectrum` — НЕ `search_left_right`
- ✅ `n_actual` — НЕ `actual_N_actual`
- ✅ `SnrEstimationResult` **НЕ содержит** поле `BranchType branch` (оно в `BranchSelector`)
- ✅ Все константы через `snr_defaults::` namespace
- ✅ **`kGuardBins = 5`** (НЕ 3 — калибровано для Hann)
- ✅ **`kRefBins = 16`** (НЕ 8 — калибровано для Hann)
- ✅ **`kDefaultWindow = WindowType::Hann`** — обязательно
- ✅ **`low_to_mid_db = 15.0f`** (НЕ 6.0 — калибровано Python)
- ✅ **`mid_to_high_db = 30.0f`** (НЕ 12.0 — калибровано Python)
- ✅ `SnrEstimationConfig` содержит поле `window` типа `WindowType`
- ✅ `#include "types/window_type.hpp"` (из SNR_02) добавлен
- ✅ `Validate()` бросает `std::invalid_argument` при нарушении `2*(guard+ref)+1 < target_n_fft`
- ✅ `shared_buf::kCount = 7`

---

## 🚫 Запреты

- ❌ НЕ переписывать существующий файл — только добавлять
- ❌ НЕ менять индексы существующих `shared_buf::kInput..kMediansCompact`
- ❌ НЕ добавлять `BranchType` в `SnrEstimationResult`

---

## 🔗 Связанные таски

- **Блокирует:** [TASK_SNR_05](TASK_SNR_05_snr_estimator_op.md) — SnrEstimatorOp использует эти типы
- **Параллельно:** [TASK_SNR_02](TASK_SNR_02_fft_func_squared.md), [TASK_SNR_03](TASK_SNR_03_gather_kernel.md)

---

*Created 2026-04-09 | Кодо*
