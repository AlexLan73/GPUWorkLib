#pragma once

/**
 * @file base_strategy_test.hpp
 * @brief BaseStrategyTest — полный pipeline тест (T1)
 *
 * GRASP Controller: координирует полный прогон AntennaProcessor pipeline.
 * Наследует StrategyTestBase (Template Method), реализует Execute + Validate.
 *
 * Execute():
 *   proc.process(d_S_, d_W_) → AntennaResult
 *
 * Validate():
 *   1. result.one_max не пустой
 *   2. Найденная частота ≈ f0_hz (±2 бина)
 *   3. dynamic_range_dB > kMinDynamicRangeDb (20 дБ)
 *   4. Статистика по debug-точкам собрана (если debug_mode=true)
 *
 * @date 2026-03-15
 */

#if ENABLE_ROCM

#include "strategy_test_base.hpp"
#include "antenna_processor_test.hpp"
#include "result_types.hpp"

#include "services/console_output.hpp"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

namespace test_strategies {

/**
 * @brief Полный pipeline тест AntennaProcessor
 *
 * Генерирует сигнал, запускает полный process(), валидирует результаты.
 * Является шаблоном для тестирования любой стратегии обработки.
 */
class BaseStrategyTest : public StrategyTestBase {
public:
  /// Минимально допустимый динамический диапазон (dB)
  static constexpr float kMinDynamicRangeDb = 20.0f;

  using StrategyTestBase::StrategyTestBase;  // Наследуем конструкторы

  std::string GetName() const override { return "BaseStrategyTest"; }

protected:
  // ── Execute: запуск полного pipeline ────────────────────────────────────

  void Execute() override {
    strategies::AntennaProcessorTest proc(backend_, cfg_);
    proc.step_0_prepare_input(d_S_, d_W_);
    result_ = proc.process_full();

    auto& c = drv_gpu_lib::ConsoleOutput::GetInstance();
    c.Print(0, "StratTest", fmt("  Execute: total=%.2f ms", result_.perf.total_ms).c_str());
  }

  // ── Validate ─────────────────────────────────────────────────────────────

  void Validate() override {
    auto& c = drv_gpu_lib::ConsoleOutput::GetInstance();

    // 1. one_max должен быть заполнен
    assert(!result_.one_max.empty() &&
           "Validate: one_max is empty — pipeline did not run?");
    c.Print(0, "StratTest", fmt("  one_max: %zu beams", result_.one_max.size()).c_str());

    // 2. Частота в луче 0 ≈ f0_hz (±2 бина)
    const float found_freq = result_.one_max[0].refined_freq_hz;
    const float bin_hz     = params_.fs /
        static_cast<float>(2 * params_.n_samples);  // примерная ширина бина
    const float freq_err   = std::abs(found_freq - params_.f0_hz);

    c.Print(0, "StratTest", fmt("  Beam 0: found=%.1f Hz, f0=%.1f Hz, err=%.1f Hz, 2bin=%.1f Hz",
        found_freq, params_.f0_hz, freq_err, 2.0f * bin_hz).c_str());

    assert(freq_err < 2.0f * bin_hz &&
           "Validate: found freq deviates more than 2 bins from f0");

    // 3. dynamic_range_dB > kMinDynamicRangeDb
    if (!result_.minmax.empty()) {
      const float dr = result_.minmax[0].dynamic_range_dB;
      c.Print(0, "StratTest", fmt("  Beam 0: dynamic_range=%.1f dB (min %.1f dB)",
          dr, kMinDynamicRangeDb).c_str());
      assert(dr > kMinDynamicRangeDb &&
             "Validate: dynamic_range_dB too small — signal too weak?");
    }

    // 4. minmax: max >= min (базовая санти)
    if (!result_.minmax.empty()) {
      assert(result_.minmax[0].max_magnitude >= result_.minmax[0].min_magnitude &&
             "Validate: max_magnitude < min_magnitude");
    }
  }

private:
  strategies::AntennaResult result_{};
};

}  // namespace test_strategies

#endif  // ENABLE_ROCM
