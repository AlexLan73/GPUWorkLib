#pragma once

/**
 * @file test_base_strategy.hpp
 * @brief Runner: BaseStrategyTest × 4 варианта сигнала (Composite GoF)
 *
 * Запускает BaseStrategyTest для всех вариантов сигнала:
 *   SIN / LFM_NO_DELAY / LFM_WITH_DELAY / LFM_FARROW
 *
 * Composite: StrategyTestSuite — список тестов, RunAll() проходит все.
 *
 * @date 2026-03-15
 */

#if ENABLE_ROCM

#include "base_strategy_test.hpp"
#include "signal_strategy_factory.hpp"
#include "antenna_test_params.hpp"

#include "interface/i_backend.hpp"
#include "services/console_output.hpp"

#include <vector>
#include <memory>
#include <string>

namespace test_base_strategy {

// ─────────────────────────────────────────────────────────────────────────────
// run_single — запуск одного варианта
// ─────────────────────────────────────────────────────────────────────────────

inline void run_single(drv_gpu_lib::IBackend* backend,
                       test_strategies::SignalVariant variant,
                       const test_strategies::AntennaTestParams& params) {
  auto sig  = test_strategies::SignalStrategyFactory::Create(variant);
  test_strategies::BaseStrategyTest test(backend, std::move(sig), params);
  test.Run();
}

// ─────────────────────────────────────────────────────────────────────────────
// run_all_variants — все 4 сигнала, параметры по умолчанию
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Запустить BaseStrategyTest для всех 4 вариантов сигнала
 *
 * По умолчанию используются AntennaTestParams::Small() (n_ant=100,
 * квадратная матрица, быстрая компиляция и прогон).
 *
 * Для полного теста 2500×5000 — передать AntennaTestParams::FullSpec():
 * @code
 *   run_all_variants(backend, AntennaTestParams::FullSpec());
 * @endcode
 *
 * @note FullSpec() требует n_beams в AntennaProcessorConfig (TODO).
 */
inline void run_all_variants(
    drv_gpu_lib::IBackend* backend,
    const test_strategies::AntennaTestParams& params
        = test_strategies::AntennaTestParams::Small())
{
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  const int g = 0;

  con.Print(g, "BaseTest", "");
  con.Print(g, "BaseTest", "════════════════════════════════════════════════════");
  con.Print(g, "BaseTest", " BaseStrategyTest × 4 signal variants");
  con.Print(g, "BaseTest", "════════════════════════════════════════════════════");

  using V = test_strategies::SignalVariant;
  const V variants[] = {
      V::SIN, V::LFM_NO_DELAY, V::LFM_WITH_DELAY, V::LFM_FARROW
  };

  int passed = 0;
  for (auto v : variants) {
    try {
      run_single(backend, v, params);
      ++passed;
    } catch (const std::exception& e) {
      con.Print(g, "BaseTest",
          (std::string("[FAIL] ") + test_strategies::SignalVariantName(v) +
           ": " + e.what()).c_str());
    }
  }

  char buf[128];
  std::snprintf(buf, sizeof(buf),
      " %d/4 variants PASSED", passed);
  con.Print(g, "BaseTest", buf);
  con.Print(g, "BaseTest", "════════════════════════════════════════════════════");
}

// ─────────────────────────────────────────────────────────────────────────────
// run_sin_only — быстрый smoke-тест с SIN (для первого запуска)
// ─────────────────────────────────────────────────────────────────────────────

inline void run_sin_only(drv_gpu_lib::IBackend* backend) {
  run_single(backend,
      test_strategies::SignalVariant::SIN,
      test_strategies::AntennaTestParams::Small());
}

}  // namespace test_base_strategy

#endif  // ENABLE_ROCM
