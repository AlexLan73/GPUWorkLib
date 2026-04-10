# TASK SNR_09: Benchmark (`GpuBenchmarkBase`) для SNR-estimator

> **Дата**: 2026-04-09
> **Модуль**: `modules/statistics/tests/`
> **Приоритет**: Medium
> **Статус**: BACKLOG
> **Зависимости**: **[SNR_06](TASK_SNR_06_facade.md)**
> **Ревьюер**: Кодо
>
> 📐 **План**: Этап 5.5 в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

Написать benchmark-класс для SNR-estimator по паттерну других модулей (heterodyne, fft_maxima, filters, lch_farrow). Наследник `GpuBenchmarkBase`, namespace `test_snr_estimator`. Профилирование **только через GPUProfiler**.

---

## 📁 Файлы (создать)

```
modules/statistics/tests/
├── snr_estimator_benchmark.hpp         — класс наследник GpuBenchmarkBase
└── test_snr_estimator_benchmark.hpp    — runner (namespace test_snr_estimator_benchmark)
```

---

## 📝 Reference-пример

Посмотреть существующие benchmark классы как образец:
- `modules/heterodyne/tests/heterodyne_benchmark_rocm.hpp`
- `modules/fft_maxima/tests/fft_maxima_benchmark_rocm.hpp`
- `modules/filters/tests/filters_benchmark_rocm.hpp`
- `modules/lch_farrow/tests/lch_farrow_benchmark_rocm.hpp`

**Следовать тому же паттерну!**

---

## 📝 Что должен измерять benchmark

### Конфигурации (минимум 4)
1. **Py-Small**: 5 ant × 1.3M samp
2. **Scenario A**: 2500 × 5000
3. **Scenario B**: 256 × 1.3M (2.66 GB) — **главный замер**
4. **Scenario C**: 9000 × 10000

### Метрики (через GPUProfiler events)
- `gather_decimated` time
- `pad + fft + magnitude_squared` time (FFT pipeline)
- `peak_cfar` time
- `median` (radix sort) time
- `total` time (end-to-end)

### Варьируемые параметры
- `target_n_fft`: {1024, 2048, 4096} — сравнение
- `squared`: всегда `true` (мы только для SNR-estimator)

---

## 📝 Шаблон `snr_estimator_benchmark.hpp`

```cpp
#pragma once

#if ENABLE_ROCM

#include "tests/gpu_benchmark_base.hpp"  // Базовый класс GpuBenchmarkBase
#include "statistics_processor.hpp"
#include "snr_test_helpers.hpp"
#include "services/gpu_profiler.hpp"
#include "services/console_output.hpp"

#include <vector>
#include <string>

namespace test_snr_estimator {

struct SnrBenchConfig {
  std::string name;
  uint32_t n_antennas;
  uint32_t n_samples;
  uint32_t target_n_fft;
};

class SnrEstimatorBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  const char* Name() const override { return "SnrEstimator"; }

  void Setup() override {
    backend_ = snr_test_helpers::GetTestBackend();
    proc_ = std::make_unique<statistics::StatisticsProcessor>(backend_);

    // SetGPUInfo перед Start() — обязательно (CLAUDE.md правило!)
    SetGPUInfoFromBackend(backend_);
  }

  void Run() override {
    std::vector<SnrBenchConfig> configs = {
      {"Py-Small",   5,    1'300'000, 2048},
      {"Scenario A", 2500, 5000,       2048},
      {"Scenario B", 256,  1'300'000, 2048},
      {"Scenario C", 9000, 10000,      2048},
      // Опционально — сравнение target_n_fft
      {"B (nfft=1024)", 256, 1'300'000, 1024},
      {"B (nfft=4096)", 256, 1'300'000, 4096},
    };

    for (const auto& cfg : configs) {
      RunSingle(cfg);
    }
  }

  void Teardown() override {
    proc_.reset();
  }

private:
  drv_gpu_lib::IBackend* backend_ = nullptr;
  std::unique_ptr<statistics::StatisticsProcessor> proc_;

  void RunSingle(const SnrBenchConfig& cfg) {
    using namespace drv_gpu_lib;
    auto& con = ConsoleOutput::GetInstance();
    con.Print(0, "SnrBench",
              "Running " + cfg.name + " (" +
              std::to_string(cfg.n_antennas) + " x " +
              std::to_string(cfg.n_samples) + ")");

    // Generate test data on GPU через hipMalloc
    // (чтобы не держать сценарий B = 2.66 GB в CPU vector)
    void* gpu_data = nullptr;
    size_t bytes = (size_t)cfg.n_antennas * cfg.n_samples * sizeof(std::complex<float>);
    hipError_t err = hipMalloc(&gpu_data, bytes);
    if (err != hipSuccess) {
      con.Print(0, "SnrBench",
                "SKIP " + cfg.name + ": hipMalloc failed (" +
                std::to_string(bytes / (1024 * 1024)) + " MB)");
      return;
    }

    // Fill с синтетическими данными (noise + one CW tone)
    FillGpuBufferWithNoisePlusCW(gpu_data, cfg.n_antennas, cfg.n_samples);

    statistics::SnrEstimationConfig scfg;
    scfg.target_n_fft = cfg.target_n_fft;

    // Measure via GpuBenchmarkBase (uses GPUProfiler internally)
    StartTimer(cfg.name);
    auto result = proc_->ComputeSnrDb(gpu_data, cfg.n_antennas, cfg.n_samples, scfg);
    StopTimer(cfg.name);

    con.Print(0, "SnrBench",
              cfg.name + ": snr_db=" + std::to_string(result.snr_db_global) +
              " used_ant=" + std::to_string(result.used_antennas) +
              " used_bins=" + std::to_string(result.used_bins));

    hipFree(gpu_data);
  }

  void FillGpuBufferWithNoisePlusCW(void* gpu_data,
                                    uint32_t n_ant, uint32_t n_samp);
  // ... helper — генерирует noise + CW tone через hipMemcpy блочно ...
};

}  // namespace test_snr_estimator

#endif  // ENABLE_ROCM
```

---

## 📝 Шаблон `test_snr_estimator_benchmark.hpp` (runner)

```cpp
#pragma once

#if ENABLE_ROCM

#include "snr_estimator_benchmark.hpp"
#include "services/gpu_profiler.hpp"

namespace test_snr_estimator_benchmark {

inline void run_benchmark() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "SnrBench", "=== SNR Estimator Benchmark ===");

  test_snr_estimator::SnrEstimatorBenchmark bench;
  bench.Setup();
  bench.Run();
  bench.Teardown();

  // Export через GPUProfiler (обязательно!)
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  profiler.PrintReport();
  profiler.ExportMarkdown("Results/Profiler/snr_estimator_benchmark.md");
  profiler.ExportJSON("Results/Profiler/snr_estimator_benchmark.json");
}

}  // namespace test_snr_estimator_benchmark

#endif  // ENABLE_ROCM
```

---

## ✅ Definition of Done

- [ ] `snr_estimator_benchmark.hpp` — класс наследник `GpuBenchmarkBase`
- [ ] `test_snr_estimator_benchmark.hpp` — runner
- [ ] Бенчмарк покрывает 4 сценария: Py-Small, A, B, C
- [ ] Опционально — сравнение `target_n_fft` ∈ {1024, 2048, 4096}
- [ ] Профилирование **только через** `GPUProfiler::PrintReport()` + `ExportMarkdown()` + `ExportJSON()`
- [ ] Вывод экспорта: `Results/Profiler/snr_estimator_benchmark.{md,json}`
- [ ] `SetGPUInfo` вызывается до `Start()` (см. `Examples/GPUProfiler_SetGPUInfo.md`)
- [ ] Обновить `all_test.hpp` — добавить include + закомментированный вызов
- [ ] Код НЕ запускается сегодня (запуск в понедельник)
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ Наследник `GpuBenchmarkBase` — не свой класс
- ✅ namespace `test_snr_estimator` для класса, `test_snr_estimator_benchmark` для runner
- ✅ Профилирование только через `GPUProfiler` — нет ручного `GetStats() + cout`
- ✅ Вывод на консоль только через `ConsoleOutput::GetInstance()`
- ✅ `SetGPUInfo` перед `Start()` (без этого в отчёте будет "Unknown")
- ✅ Для больших сценариев (B, C) — данные генерируются на GPU через `hipMalloc`, НЕ через CPU vector (2.66 GB → OOM)
- ✅ Пути: `Results/Profiler/snr_estimator_benchmark.{md,json}` (relative)
- ✅ `all_test.hpp` вызов **закомментирован** (откроется в понедельник)

---

## 🚫 Запреты

- ❌ НЕ использовать свой Timer вместо `GpuBenchmarkBase`
- ❌ НЕ писать ручной `GetStats() + std::cout` — только `PrintReport`/`ExportMarkdown`/`ExportJSON`
- ❌ НЕ создавать CPU vector на 2.66 GB (сценарий B)
- ❌ НЕ запускать сегодня — только компиляция в понедельник

---

## 📝 Заметки

**Memory management для сценария B:**
```cpp
// НЕ ТАК (CPU OOM):
std::vector<std::complex<float>> data(256 * 1'300'000);  // 2.66 GB!

// ТАК (генерируем на GPU через hipMalloc + hipMemcpy блоками):
void* gpu_data = nullptr;
hipMalloc(&gpu_data, 256 * 1'300'000 * 8);
// Fill блоками по одной антенне
```

---

## 🔗 Связанные таски

- **Требует:** [SNR_06](TASK_SNR_06_facade.md), [SNR_08](TASK_SNR_08_cpp_tests.md) (helpers)
- **Параллельно:** SNR_07, SNR_10

---

*Created 2026-04-09 | Кодо*
