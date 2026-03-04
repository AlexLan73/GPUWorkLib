# FM Correlator: Профилирование по шагам

**Дата**: 2026-03-04
**Статус**: В работе
**Платформа**: ROCm/HIP (AMD). Пишем на Windows, тестируем на AMD.

---

## Концепция

Pipeline = **2 шага**:
- **step1** = `PrepareReference()` — H2D ref → apply_cyclic_shifts → C2C FFT(ref)
- **step2** = `Process(inp)` — H2D inp → R2C FFT(inp) → multiply_conj_fused → C2R IFFT → extract_magnitudes → D2H

> Steps 2 и 3 из исходного ТЗ уже объединены внутри `Process()` — это то что нам нужно.
> **Production-класс НЕ изменяем.** Таймируем снаружи через `hipEvent`.

---

## Порядок разработки

- **2.1, 2.2, 2.3** — пишем все **одновременно**
- Тестируем **2.1 и 2.2** — получаем положительный результат
- После принятия 2.1+2.2 — тестируем **2.3**

---

## Файлы (5 задач)

```
modules/fm_correlator/tests/
├── test_fm_benchmark_rocm_all_time.hpp   ← TASK-1: переименовать (код не менять)
├── test_fm_step_profiling.hpp            ← TASK-2: новый тест 2.1
├── test_fm_avg_summary.hpp               ← TASK-3: новый тест 2.2
├── test_fm_combined.hpp                  ← TASK-4: новый тест 2.3
└── all_test.hpp                          ← TASK-5: обновить includes + вызовы
```

---

## TASK-1: Переименовать test_fm_benchmark_rocm.hpp → test_fm_benchmark_rocm_all_time.hpp

### Действия

1. Переименовать файл:
   - `tests/test_fm_benchmark_rocm.hpp` → `tests/test_fm_benchmark_rocm_all_time.hpp`
   - Содержимое файла **не менять вообще**

2. В `tests/all_test.hpp` строку:
   ```cpp
   #include "test_fm_benchmark_rocm.hpp"
   ```
   заменить на:
   ```cpp
   #include "test_fm_benchmark_rocm_all_time.hpp"
   ```

### Проверка
- Компилируется без ошибок (проверить что include разрешается)

---

## TASK-2: Создать test_fm_step_profiling.hpp (Тест 2.1 — детальное профилирование)

### Назначение
20 замеров step1 + 20 замеров step2. **Каждый замер = отдельный `Record` в GPUProfiler**.
Итого: 20 записей `"step1"` + 20 записей `"step2"` → `PrintReport` показывает min/max/avg по каждому шагу.

### Файл
`modules/fm_correlator/tests/test_fm_step_profiling.hpp`

### Точная структура

```cpp
#pragma once
#if ENABLE_ROCM

/**
 * @file test_fm_step_profiling.hpp
 * @brief FM Correlator — детальное профилирование по шагам (тест 2.1)
 *
 * step1 = PrepareReference(): 20 прогревов + 20 замеров → 20 Record("step1")
 * step2 = Process(inp):       20 прогревов + 20 замеров → 20 Record("step2")
 * Вывод: profiler.PrintReport() + ExportJSON + ExportMarkdown
 */

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "fm_correlator.hpp"
#include "test_fm_benchmark_rocm_all_time.hpp"   // ← FmGetDateForFilename(), GetTestBackend()
#include "services/console_output.hpp"
#include "services/gpu_profiler.hpp"

namespace fm_correlator::tests {

// ── Параметры теста 2.1 — изменить для другой конфигурации ─────────────────
constexpr size_t kSpFftSize    = 32768;
constexpr int    kSpNumShifts  = 32;
constexpr int    kSpNumSignals = 5;
constexpr int    kSpNumOutPts  = 2000;
constexpr int    kSpWarmup     = 20;
constexpr int    kSpRuns       = 20;
// ────────────────────────────────────────────────────────────────────────────

inline void run_step_profiling() {
  auto& con     = drv_gpu_lib::ConsoleOutput::GetInstance();
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  const int gpu_id = 0;

  con.Print(gpu_id, "FM_Step", "════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Step", "  FM Correlator Step Profiling (2.1)");
  con.Print(gpu_id, "FM_Step", "  step1=PrepareReference  step2=Process");
  con.Print(gpu_id, "FM_Step", "  warmup=" + std::to_string(kSpWarmup) +
            "  runs=" + std::to_string(kSpRuns));
  con.Print(gpu_id, "FM_Step", "════════════════════════════════════════════════");

  // ── Backend + Correlator ──────────────────────────────────────────────────
  auto* backend = GetTestBackend();
  drv_gpu_lib::FMCorrelator corr(backend);

  drv_gpu_lib::FMCorrelatorParams params;
  params.fft_size          = kSpFftSize;
  params.num_shifts        = kSpNumShifts;
  params.num_signals       = kSpNumSignals;
  params.num_output_points = kSpNumOutPts;
  corr.SetParams(params);

  // Плоский вектор входного сигнала: num_signals × fft_size
  std::vector<float> inp(kSpFftSize * static_cast<size_t>(kSpNumSignals), 1.0f);

  // Начальный вызов PrepareReference для корректного состояния
  corr.PrepareReference();

  // ── Warmup: kSpWarmup итераций (прогрев GPU + hiprtc compile) ────────────
  con.Print(gpu_id, "FM_Step", "  Warmup...");
  for (int w = 0; w < kSpWarmup; ++w) {
    corr.PrepareReference();
    (void)corr.Process(inp);
    (void)hipDeviceSynchronize();
  }

  // ── GPUProfiler setup (SetGPUInfo перед Start — обязательно!) ────────────
  auto dev = backend->GetDeviceInfo();
  drv_gpu_lib::GPUReportInfo report_info;
  report_info.gpu_name      = dev.name;
  report_info.backend_type  = drv_gpu_lib::BackendType::ROCm;
  report_info.global_mem_mb = dev.global_memory_size / (1024 * 1024);
  std::map<std::string, std::string> drv_map;
  drv_map["driver_type"]    = "ROCm";
  drv_map["driver_version"] = dev.driver_version;
  report_info.drivers.push_back(drv_map);
  profiler.SetGPUInfo(gpu_id, report_info);
  profiler.Start();

  // ── hipEvent (создаём один раз, переиспользуем) ───────────────────────────
  hipEvent_t ev_start, ev_stop;
  (void)hipEventCreate(&ev_start);
  (void)hipEventCreate(&ev_stop);

  // ── Замер step1: PrepareReference — kSpRuns итераций ─────────────────────
  con.Print(gpu_id, "FM_Step", "  Measuring step1 (PrepareReference)...");
  for (int r = 0; r < kSpRuns; ++r) {
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(ev_start, nullptr);

    corr.PrepareReference();

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);

    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);

    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns      = static_cast<uint64_t>(ms * 1.0e6f);
    pd.kernel_name = "PrepareReference";
    profiler.Record(gpu_id, "FM_Step", "step1", pd);
  }

  // ── Замер step2: Process — kSpRuns итераций ───────────────────────────────
  // Ссылка валидна после последнего PrepareReference в цикле step1.
  con.Print(gpu_id, "FM_Step", "  Measuring step2 (Process)...");
  for (int r = 0; r < kSpRuns; ++r) {
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(ev_start, nullptr);

    (void)corr.Process(inp);

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);

    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);

    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns      = static_cast<uint64_t>(ms * 1.0e6f);
    pd.kernel_name = "Process";
    profiler.Record(gpu_id, "FM_Step", "step2", pd);
  }

  // ── Cleanup hipEvents ─────────────────────────────────────────────────────
  (void)hipEventDestroy(ev_start);
  (void)hipEventDestroy(ev_stop);

  // ── Отчёт (ТОЛЬКО через GPUProfiler) ─────────────────────────────────────
  profiler.Stop();
  profiler.PrintReport();

  std::string base = "../Results/Profiler/fm_correlator/fm_step_profiling_"
                   + FmGetDateForFilename();
  profiler.ExportJSON(base + ".json");
  profiler.ExportMarkdown(base + ".md");

  con.Print(gpu_id, "FM_Step", "  Report: " + base + ".md");
  con.Print(gpu_id, "FM_Step", "  Step profiling complete ✅");
}

}  // namespace fm_correlator::tests

#endif  // ENABLE_ROCM
```

### Ожидаемый вывод PrintReport
GPUProfiler покажет 2 группы событий:
- `step1` (N=20): min/avg/max время PrepareReference
- `step2` (N=20): min/avg/max время Process

---

## TASK-3: Создать test_fm_avg_summary.hpp (Тест 2.2 — среднее для планирования)

### Назначение
20 замеров step1 + 20 замеров step2. **Вычислить avg, записать 1 синтетическое событие на шаг**.
Дополнительно: compact summary через `con.Print()` — для задач планирования временных окон.

### Отличие от 2.1
| | Тест 2.1 | Тест 2.2 |
|--|----------|----------|
| Замеры | 20 Record("step1") + 20 Record("step2") | 1 Record("step1") + 1 Record("step2") |
| Данные | Полная статистика (min/max/avg/N=20) | Только avg |
| Доп. вывод | нет | compact summary через con.Print() |
| Назначение | Детальный анализ | Планирование временных окон |

### Файл
`modules/fm_correlator/tests/test_fm_avg_summary.hpp`

### Точная структура

```cpp
#pragma once
#if ENABLE_ROCM

/**
 * @file test_fm_avg_summary.hpp
 * @brief FM Correlator — среднее время шагов для планирования (тест 2.2)
 *
 * step1 = PrepareReference(): 20 замеров → avg → 1 Record("step1")
 * step2 = Process(inp):       20 замеров → avg → 1 Record("step2")
 * Compact summary: con.Print() с двумя строками avg
 * Назначение: знать на какое время закладываться при планировании.
 */

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "fm_correlator.hpp"
#include "test_fm_benchmark_rocm_all_time.hpp"   // ← FmGetDateForFilename(), GetTestBackend()
#include "services/console_output.hpp"
#include "services/gpu_profiler.hpp"

namespace fm_correlator::tests {

// ── Параметры теста 2.2 — изменить для другой конфигурации ─────────────────
constexpr size_t kAvgFftSize    = 32768;
constexpr int    kAvgNumShifts  = 32;
constexpr int    kAvgNumSignals = 5;
constexpr int    kAvgNumOutPts  = 2000;
constexpr int    kAvgWarmup     = 20;
constexpr int    kAvgRuns       = 20;
// ────────────────────────────────────────────────────────────────────────────

inline void run_avg_summary() {
  auto& con     = drv_gpu_lib::ConsoleOutput::GetInstance();
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  const int gpu_id = 0;

  con.Print(gpu_id, "FM_Avg", "════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Avg", "  FM Correlator Avg Summary (2.2)");
  con.Print(gpu_id, "FM_Avg", "  Цель: среднее время step1 и step2 для планирования");
  con.Print(gpu_id, "FM_Avg", "  warmup=" + std::to_string(kAvgWarmup) +
            "  runs=" + std::to_string(kAvgRuns));
  con.Print(gpu_id, "FM_Avg", "════════════════════════════════════════════════");

  // ── Backend + Correlator ──────────────────────────────────────────────────
  auto* backend = GetTestBackend();
  drv_gpu_lib::FMCorrelator corr(backend);

  drv_gpu_lib::FMCorrelatorParams params;
  params.fft_size          = kAvgFftSize;
  params.num_shifts        = kAvgNumShifts;
  params.num_signals       = kAvgNumSignals;
  params.num_output_points = kAvgNumOutPts;
  corr.SetParams(params);

  std::vector<float> inp(kAvgFftSize * static_cast<size_t>(kAvgNumSignals), 1.0f);
  corr.PrepareReference();

  // ── Warmup ────────────────────────────────────────────────────────────────
  con.Print(gpu_id, "FM_Avg", "  Warmup...");
  for (int w = 0; w < kAvgWarmup; ++w) {
    corr.PrepareReference();
    (void)corr.Process(inp);
    (void)hipDeviceSynchronize();
  }

  // ── GPUProfiler setup ─────────────────────────────────────────────────────
  auto dev = backend->GetDeviceInfo();
  drv_gpu_lib::GPUReportInfo report_info;
  report_info.gpu_name      = dev.name;
  report_info.backend_type  = drv_gpu_lib::BackendType::ROCm;
  report_info.global_mem_mb = dev.global_memory_size / (1024 * 1024);
  std::map<std::string, std::string> drv_map;
  drv_map["driver_type"]    = "ROCm";
  drv_map["driver_version"] = dev.driver_version;
  report_info.drivers.push_back(drv_map);
  profiler.SetGPUInfo(gpu_id, report_info);
  profiler.Start();

  // ── hipEvent ──────────────────────────────────────────────────────────────
  hipEvent_t ev_start, ev_stop;
  (void)hipEventCreate(&ev_start);
  (void)hipEventCreate(&ev_stop);

  // ── Замер step1: собираем в вектор → вычисляем avg ────────────────────────
  std::vector<double> step1_times(kAvgRuns);
  con.Print(gpu_id, "FM_Avg", "  Measuring step1 (PrepareReference)...");
  for (int r = 0; r < kAvgRuns; ++r) {
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(ev_start, nullptr);

    corr.PrepareReference();

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    step1_times[r] = static_cast<double>(ms);
  }
  const double avg_step1_ms =
      std::accumulate(step1_times.begin(), step1_times.end(), 0.0) / kAvgRuns;

  // ── Замер step2: собираем в вектор → вычисляем avg ────────────────────────
  std::vector<double> step2_times(kAvgRuns);
  con.Print(gpu_id, "FM_Avg", "  Measuring step2 (Process)...");
  for (int r = 0; r < kAvgRuns; ++r) {
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(ev_start, nullptr);

    (void)corr.Process(inp);

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    step2_times[r] = static_cast<double>(ms);
  }
  const double avg_step2_ms =
      std::accumulate(step2_times.begin(), step2_times.end(), 0.0) / kAvgRuns;

  // ── Cleanup hipEvents ─────────────────────────────────────────────────────
  (void)hipEventDestroy(ev_start);
  (void)hipEventDestroy(ev_stop);

  // ── Записываем 1 синтетическое событие на шаг в GPUProfiler ──────────────
  {
    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns      = static_cast<uint64_t>(avg_step1_ms * 1.0e6);
    pd.kernel_name = "PrepareReference_avg";
    profiler.Record(gpu_id, "FM_Avg", "step1", pd);
  }
  {
    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns      = static_cast<uint64_t>(avg_step2_ms * 1.0e6);
    pd.kernel_name = "Process_avg";
    profiler.Record(gpu_id, "FM_Avg", "step2", pd);
  }

  // ── Отчёт через GPUProfiler ───────────────────────────────────────────────
  profiler.Stop();
  profiler.PrintReport();

  std::string base = "../Results/Profiler/fm_correlator/fm_avg_summary_"
                   + FmGetDateForFilename();
  profiler.ExportJSON(base + ".json");
  profiler.ExportMarkdown(base + ".md");

  // ── Compact summary для планирования (con.Print — НЕ профилирующие данные) ─
  char buf[128];
  con.Print(gpu_id, "FM_Avg", "  ┌─────────────────────────────────────────┐");
  std::snprintf(buf, sizeof(buf), "  │ step1 (PrepareRef):  avg = %8.3f ms  │", avg_step1_ms);
  con.Print(gpu_id, "FM_Avg", buf);
  std::snprintf(buf, sizeof(buf), "  │ step2 (Process):     avg = %8.3f ms  │", avg_step2_ms);
  con.Print(gpu_id, "FM_Avg", buf);
  con.Print(gpu_id, "FM_Avg", "  └─────────────────────────────────────────┘");
  con.Print(gpu_id, "FM_Avg", "  Report: " + base + ".md");
  con.Print(gpu_id, "FM_Avg", "  Avg summary complete ✅");
}

}  // namespace fm_correlator::tests

#endif  // ENABLE_ROCM
```

---

## TASK-4: Создать test_fm_combined.hpp (Тест 2.3 — объединённый)

### Назначение
Один прогон GPUProfiler с 3 синтетическими событиями: `"all_time"` + `"step1"` + `"step2"`.
Объединяет подход из `_all_time` (полное время) и из 2.2 (средние по шагам).

### Файл
`modules/fm_correlator/tests/test_fm_combined.hpp`

### Точная структура

```cpp
#pragma once
#if ENABLE_ROCM

/**
 * @file test_fm_combined.hpp
 * @brief FM Correlator — комбинированный тест (2.3)
 *
 * Один GPUProfiler прогон, 3 синтетических события:
 *   "all_time" = avg(PrepareReference + Process) за 20 замеров
 *   "step1"    = avg(PrepareReference) за 20 замеров
 *   "step2"    = avg(Process) за 20 замеров
 *
 * Внимание: тест 2.3 отлаживать только после принятия теста 2.2.
 */

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "fm_correlator.hpp"
#include "test_fm_benchmark_rocm_all_time.hpp"   // ← FmGetDateForFilename(), GetTestBackend()
#include "services/console_output.hpp"
#include "services/gpu_profiler.hpp"

namespace fm_correlator::tests {

// ── Параметры теста 2.3 — изменить для другой конфигурации ─────────────────
constexpr size_t kCmbFftSize    = 32768;
constexpr int    kCmbNumShifts  = 32;
constexpr int    kCmbNumSignals = 5;
constexpr int    kCmbNumOutPts  = 2000;
constexpr int    kCmbWarmup     = 20;
constexpr int    kCmbRuns       = 20;
// ────────────────────────────────────────────────────────────────────────────

inline void run_combined() {
  auto& con     = drv_gpu_lib::ConsoleOutput::GetInstance();
  auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();
  const int gpu_id = 0;

  con.Print(gpu_id, "FM_Cmb", "════════════════════════════════════════════════");
  con.Print(gpu_id, "FM_Cmb", "  FM Correlator Combined Test (2.3)");
  con.Print(gpu_id, "FM_Cmb", "  all_time + step1 + step2 в одном прогоне");
  con.Print(gpu_id, "FM_Cmb", "  warmup=" + std::to_string(kCmbWarmup) +
            "  runs=" + std::to_string(kCmbRuns));
  con.Print(gpu_id, "FM_Cmb", "════════════════════════════════════════════════");

  // ── Backend + Correlator ──────────────────────────────────────────────────
  auto* backend = GetTestBackend();
  drv_gpu_lib::FMCorrelator corr(backend);

  drv_gpu_lib::FMCorrelatorParams params;
  params.fft_size          = kCmbFftSize;
  params.num_shifts        = kCmbNumShifts;
  params.num_signals       = kCmbNumSignals;
  params.num_output_points = kCmbNumOutPts;
  corr.SetParams(params);

  std::vector<float> inp(kCmbFftSize * static_cast<size_t>(kCmbNumSignals), 1.0f);
  corr.PrepareReference();

  // ── Warmup ────────────────────────────────────────────────────────────────
  con.Print(gpu_id, "FM_Cmb", "  Warmup...");
  for (int w = 0; w < kCmbWarmup; ++w) {
    corr.PrepareReference();
    (void)corr.Process(inp);
    (void)hipDeviceSynchronize();
  }

  // ── GPUProfiler setup ─────────────────────────────────────────────────────
  auto dev = backend->GetDeviceInfo();
  drv_gpu_lib::GPUReportInfo report_info;
  report_info.gpu_name      = dev.name;
  report_info.backend_type  = drv_gpu_lib::BackendType::ROCm;
  report_info.global_mem_mb = dev.global_memory_size / (1024 * 1024);
  std::map<std::string, std::string> drv_map;
  drv_map["driver_type"]    = "ROCm";
  drv_map["driver_version"] = dev.driver_version;
  report_info.drivers.push_back(drv_map);
  profiler.SetGPUInfo(gpu_id, report_info);
  profiler.Start();

  // ── hipEvent ──────────────────────────────────────────────────────────────
  hipEvent_t ev_start, ev_stop;
  (void)hipEventCreate(&ev_start);
  (void)hipEventCreate(&ev_stop);

  // ── (A) Замер all_time: { PrepareReference + Process } ───────────────────
  std::vector<double> all_times(kCmbRuns);
  con.Print(gpu_id, "FM_Cmb", "  Measuring all_time...");
  for (int r = 0; r < kCmbRuns; ++r) {
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(ev_start, nullptr);

    corr.PrepareReference();
    (void)corr.Process(inp);

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    all_times[r] = static_cast<double>(ms);
  }
  const double avg_all_ms =
      std::accumulate(all_times.begin(), all_times.end(), 0.0) / kCmbRuns;

  // ── (B) Замер step1: PrepareReference ─────────────────────────────────────
  std::vector<double> step1_times(kCmbRuns);
  con.Print(gpu_id, "FM_Cmb", "  Measuring step1 (PrepareReference)...");
  for (int r = 0; r < kCmbRuns; ++r) {
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(ev_start, nullptr);

    corr.PrepareReference();

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    step1_times[r] = static_cast<double>(ms);
  }
  const double avg_step1_ms =
      std::accumulate(step1_times.begin(), step1_times.end(), 0.0) / kCmbRuns;

  // ── (C) Замер step2: Process ───────────────────────────────────────────────
  std::vector<double> step2_times(kCmbRuns);
  con.Print(gpu_id, "FM_Cmb", "  Measuring step2 (Process)...");
  for (int r = 0; r < kCmbRuns; ++r) {
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(ev_start, nullptr);

    (void)corr.Process(inp);

    (void)hipEventRecord(ev_stop, nullptr);
    (void)hipEventSynchronize(ev_stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
    step2_times[r] = static_cast<double>(ms);
  }
  const double avg_step2_ms =
      std::accumulate(step2_times.begin(), step2_times.end(), 0.0) / kCmbRuns;

  // ── Cleanup hipEvents ─────────────────────────────────────────────────────
  (void)hipEventDestroy(ev_start);
  (void)hipEventDestroy(ev_stop);

  // ── Записать 3 синтетических события в GPUProfiler ────────────────────────
  {
    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns = static_cast<uint64_t>(avg_all_ms * 1.0e6);
    pd.kernel_name = "PrepareReference+Process";
    profiler.Record(gpu_id, "FM_Combined", "all_time", pd);
  }
  {
    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns = static_cast<uint64_t>(avg_step1_ms * 1.0e6);
    pd.kernel_name = "PrepareReference_avg";
    profiler.Record(gpu_id, "FM_Combined", "step1", pd);
  }
  {
    drv_gpu_lib::ROCmProfilingData pd{};
    pd.end_ns = static_cast<uint64_t>(avg_step2_ms * 1.0e6);
    pd.kernel_name = "Process_avg";
    profiler.Record(gpu_id, "FM_Combined", "step2", pd);
  }

  // ── Отчёт через GPUProfiler ───────────────────────────────────────────────
  profiler.Stop();
  profiler.PrintReport();

  std::string base = "../Results/Profiler/fm_correlator/fm_combined_"
                   + FmGetDateForFilename();
  profiler.ExportJSON(base + ".json");
  profiler.ExportMarkdown(base + ".md");

  // ── Compact summary (con.Print — НЕ профилирующие данные) ─────────────────
  char buf[128];
  con.Print(gpu_id, "FM_Cmb", "  ┌─────────────────────────────────────────┐");
  std::snprintf(buf, sizeof(buf), "  │ all_time:            avg = %8.3f ms  │", avg_all_ms);
  con.Print(gpu_id, "FM_Cmb", buf);
  std::snprintf(buf, sizeof(buf), "  │ step1 (PrepareRef):  avg = %8.3f ms  │", avg_step1_ms);
  con.Print(gpu_id, "FM_Cmb", buf);
  std::snprintf(buf, sizeof(buf), "  │ step2 (Process):     avg = %8.3f ms  │", avg_step2_ms);
  con.Print(gpu_id, "FM_Cmb", buf);
  con.Print(gpu_id, "FM_Cmb", "  └─────────────────────────────────────────┘");
  con.Print(gpu_id, "FM_Cmb", "  Report: " + base + ".md");
  con.Print(gpu_id, "FM_Cmb", "  Combined test complete ✅");
}

}  // namespace fm_correlator::tests

#endif  // ENABLE_ROCM
```

---

## TASK-5: Обновить all_test.hpp

### Файл
`modules/fm_correlator/tests/all_test.hpp`

### Изменения

1. Строку `#include "test_fm_benchmark_rocm.hpp"` заменить на:
   ```cpp
   #include "test_fm_benchmark_rocm_all_time.hpp"
   ```

2. После этой строки добавить три новых include:
   ```cpp
   #include "test_fm_step_profiling.hpp"
   #include "test_fm_avg_summary.hpp"
   #include "test_fm_combined.hpp"
   ```

3. В функции `run()` добавить закомментированные вызовы после существующего `// fm_correlator::tests::run_benchmark();`:
   ```cpp
   // Профилирование по шагам:
   // fm_correlator::tests::run_step_profiling();   // 2.1 — детальное (20 Record на шаг)
   // fm_correlator::tests::run_avg_summary();      // 2.2 — среднее для планирования
   // fm_correlator::tests::run_combined();         // 2.3 — после принятия 2.2
   ```

4. `run_sweep_correlations()` **оставить активным** (как сейчас).

---

## Правила (напоминание)

- `#if ENABLE_ROCM` обязателен во всех новых файлах
- `profiler.SetGPUInfo(gpu_id, report_info)` — **перед** `profiler.Start()`
- Вывод данных профилирования — ТОЛЬКО через `profiler.PrintReport()` / `ExportJSON` / `ExportMarkdown`
- `con.Print()` — только для вспомогательного вывода (заголовки, compact summary)
- Google C++ Style, отступ 2 пробела
