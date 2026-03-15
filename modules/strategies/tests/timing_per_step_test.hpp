#pragma once

/**
 * @file timing_per_step_test.hpp
 * @brief TimingPerStepTest — измерение времени каждого Step (T4)
 *
 * Цель: получить таблицу GPU-времени по каждому шагу для дальнейшей
 * оптимизации кода (TestStrategia.md §Тест измерения времени).
 *
 * Отличие от StrategiesProfilingBenchmark:
 *   - Один прогон (не n_runs итераций) — быстрый срез
 *   - Результат — таблица в консоли (ms/step)
 *   - Данные экспортируются в JSON для анализа в Python (test_timing_analysis.py)
 *
 * Вывод:
 *   ConsoleOutput → таблица шагов
 *   JSON → Results/strategies/timing_{signal}.json
 *
 * @date 2026-03-15
 */

#if ENABLE_ROCM

#include "strategy_test_base.hpp"
#include "antenna_processor_test.hpp"

#include "services/console_output.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <chrono>

namespace test_strategies {

// ─────────────────────────────────────────────────────────────────────────────
// StepTiming — результат замера одного шага
// ─────────────────────────────────────────────────────────────────────────────

struct StepTiming {
  std::string name;
  float       gpu_ms  = 0.0f;  ///< hipEvent elapsed time
  float       wall_ms = 0.0f;  ///< wall-clock (chrono)
};

// ─────────────────────────────────────────────────────────────────────────────
// TimingPerStepTest
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Быстрый замер времени каждого шага AntennaProcessor
 *
 * Один тёплый прогон (после одного warmup) для каждого шага.
 * Результаты — таблица в ConsoleOutput + JSON файл для Python анализа.
 */
class TimingPerStepTest : public StrategyTestBase {
public:
  using StrategyTestBase::StrategyTestBase;

  std::string GetName() const override { return "TimingPerStepTest"; }

protected:
  void Execute() override {
    cfg_.debug_mode = false;
    strategies::AntennaProcessorTest proc(backend_, cfg_);
    proc.step_0_prepare_input(d_S_, d_W_);

    // Warmup: 1 полный прогон
    proc.process_full();
    hipDeviceSynchronize();

    // Подготовка: d_X нужен для FFT/MaxSteps
    proc.step_2_gemm();
    hipDeviceSynchronize();
    proc.step_4_window_fft();
    hipDeviceSynchronize();

    // ── Замеряем каждый шаг ────────────────────────────────────────────────
    timings_.clear();

    // GEMM
    timings_.push_back(MeasureOne("GEMM", [&]() {
      proc.step_2_gemm();
    }));

    // WindowFFT (нужен свежий d_X)
    proc.step_2_gemm(); hipDeviceSynchronize();
    timings_.push_back(MeasureOne("WindowFFT", [&]() {
      proc.step_4_window_fft();
    }));

    // Три post-FFT шага (нужен свежий спектр)
    proc.step_2_gemm(); hipDeviceSynchronize();
    proc.step_4_window_fft(); hipDeviceSynchronize();

    timings_.push_back(MeasureOne("OneMax", [&]() {
      proc.step_6_1_one_max_parabola();
    }));
    timings_.push_back(MeasureOne("AllMaxima", [&]() {
      proc.step_6_2_all_maxima();
    }));
    timings_.push_back(MeasureOne("MinMax", [&]() {
      proc.step_6_3_global_minmax();
    }));

    // Full pipeline (все шаги последовательно)
    timings_.push_back(MeasureOne("FullProcess", [&]() {
      proc.process_full();
    }));
  }

  void Validate() override {
    auto& c = drv_gpu_lib::ConsoleOutput::GetInstance();

    c.Print(0, "TimingTest", "");
    c.Print(0, "TimingTest", "  ┌──────────────────────┬──────────┬──────────┐");
    c.Print(0, "TimingTest", "  │ Step                 │ GPU ms   │ Wall ms  │");
    c.Print(0, "TimingTest", "  ├──────────────────────┼──────────┼──────────┤");

    float total_gpu = 0.0f;
    for (const auto& t : timings_) {
      char row[128];
      std::snprintf(row, sizeof(row),
          "  │ %-20s │ %8.3f │ %8.3f │",
          t.name.c_str(), t.gpu_ms, t.wall_ms);
      c.Print(0, "TimingTest", row);
      if (t.name != "FullProcess") total_gpu += t.gpu_ms;
    }

    c.Print(0, "TimingTest", "  └──────────────────────┴──────────┴──────────┘");

    char sumline[128];
    std::snprintf(sumline, sizeof(sumline),
        "  Sum(steps) = %.3f ms   FullProcess GPU = %.3f ms",
        total_gpu,
        timings_.empty() ? 0.0f : timings_.back().gpu_ms);
    c.Print(0, "TimingTest", sumline);

    // Sanity: полный pipeline < 500 мс (при n_ant=100, n_samples=5000)
    if (!timings_.empty()) {
      const float full_ms = timings_.back().gpu_ms;
      if (full_ms > 500.0f) {
        c.Print(0, "TimingTest", "  [WARN] FullProcess > 500ms — check n_ant/n_samples");
      }
    }
  }

  void SaveResults() override {
    if (timings_.empty()) return;

    // Экспорт в JSON для анализа в Python (test_timing_analysis.py)
    const std::string path = params_.output_dir + "timing_" +
                             GetSignalName() + ".json";

    // TODO: создать директорию (std::filesystem C++17)
    std::ofstream f(path);
    if (!f.is_open()) {
      drv_gpu_lib::ConsoleOutput::GetInstance().Print(
          0, "TimingTest", ("  [WARN] Cannot write " + path).c_str());
      return;
    }

    f << "{\n";
    f << "  \"signal\": \"" << GetSignalName() << "\",\n";
    f << "  \"n_ant\": "    << params_.n_ant    << ",\n";
    f << "  \"n_samples\": " << params_.n_samples << ",\n";
    f << "  \"fs\": "        << params_.fs        << ",\n";
    f << "  \"steps\": [\n";
    for (size_t i = 0; i < timings_.size(); ++i) {
      char row[128];
      std::snprintf(row, sizeof(row),
          "    {\"name\": \"%s\", \"gpu_ms\": %.4f, \"wall_ms\": %.4f}%s",
          timings_[i].name.c_str(), timings_[i].gpu_ms, timings_[i].wall_ms,
          (i + 1 < timings_.size()) ? "," : "");
      f << row << "\n";
    }
    f << "  ]\n}\n";

    drv_gpu_lib::ConsoleOutput::GetInstance().Print(
        0, "TimingTest", ("  Timing saved: " + path).c_str());
  }

private:
  std::vector<StepTiming> timings_;

  // Имя варианта сигнала (для имени файла)
  std::string GetSignalName() const {
    // Заполняется из params_.signal_variant
    return SignalVariantName(params_.signal_variant);
  }

  // Одиночный замер через hipEvent
  StepTiming MeasureOne(const std::string& name,
                        std::function<void()> fn) {
    using SClock = std::chrono::steady_clock;
    using MS     = std::chrono::duration<float, std::milli>;

    hipEvent_t ev_start, ev_stop;
    hipEventCreate(&ev_start);
    hipEventCreate(&ev_stop);

    hipDeviceSynchronize();
    auto wall_start = SClock::now();
    hipEventRecord(ev_start, nullptr);

    fn();

    hipEventRecord(ev_stop, nullptr);
    hipEventSynchronize(ev_stop);
    auto wall_end = SClock::now();

    float gpu_ms = 0.0f;
    hipEventElapsedTime(&gpu_ms, ev_start, ev_stop);

    hipEventDestroy(ev_start);
    hipEventDestroy(ev_stop);

    StepTiming t;
    t.name    = name;
    t.gpu_ms  = gpu_ms;
    t.wall_ms = std::chrono::duration_cast<MS>(wall_end - wall_start).count();
    return t;
  }
};

}  // namespace test_strategies

#endif  // ENABLE_ROCM
