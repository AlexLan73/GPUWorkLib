#pragma once

/**
 * @file test_debug_steps.hpp
 * @brief Runner для DebugStepTest × 4 сигнала
 *
 * @date 2026-03-15
 */

#if ENABLE_ROCM

#include "debug_step_test.hpp"
#include "signal_strategy_factory.hpp"
#include "antenna_test_params.hpp"

#include "interface/i_backend.hpp"
#include "services/console_output.hpp"

namespace test_debug_steps {

/// Запустить DebugStepTest для одного варианта сигнала
inline void run_single(drv_gpu_lib::IBackend* backend,
                       test_strategies::SignalVariant variant,
                       const test_strategies::AntennaTestParams& params) {
  auto sig = test_strategies::SignalStrategyFactory::Create(variant);
  test_strategies::DebugStepTest test(backend, std::move(sig), params);
  test.Run();
}

/**
 * @brief Запустить DebugStepTest для всех 4 вариантов сигнала
 *
 * Логирует каждый шаг и проверяет корректность выходных данных.
 * Для записи промежуточных данных — передать params с save_to_files=true.
 */
inline void run_all(
    drv_gpu_lib::IBackend* backend,
    const test_strategies::AntennaTestParams& params
        = test_strategies::AntennaTestParams::Small())
{
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  const int g = 0;

  con.Print(g, "DebugStep", "");
  con.Print(g, "DebugStep", "════════════════════════════════════════════════════");
  con.Print(g, "DebugStep", " DebugStepTest × 4 signal variants");
  con.Print(g, "DebugStep", "════════════════════════════════════════════════════");

  using V = test_strategies::SignalVariant;
  for (auto v : {V::SIN, V::LFM_NO_DELAY, V::LFM_WITH_DELAY, V::LFM_FARROW}) {
    try {
      run_single(backend, v, params);
    } catch (const std::exception& e) {
      con.Print(g, "DebugStep",
          (std::string("[FAIL] ") + test_strategies::SignalVariantName(v) +
           ": " + e.what()).c_str());
    }
  }

  con.Print(g, "DebugStep", " DebugStepTest DONE");
  con.Print(g, "DebugStep", "════════════════════════════════════════════════════");
}

/// Отладка с записью данных в файлы (output_dir = Results/strategies/debug/)
inline void run_with_save(drv_gpu_lib::IBackend* backend,
                          test_strategies::SignalVariant variant
                              = test_strategies::SignalVariant::SIN) {
  auto params      = test_strategies::AntennaTestParams::Debug(variant);
  run_single(backend, variant, params);
}

}  // namespace test_debug_steps

#endif  // ENABLE_ROCM
