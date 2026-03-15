#pragma once

/**
 * @file strategy_test_base.hpp
 * @brief StrategyTestBase — Template Method (GoF) для тестов стратегии
 *
 * Определяет инвариантный скелет теста:
 *   Run() → Setup() → GenerateSignals() → PrepareMatrix() → Upload()
 *         → Execute() → Validate() → SaveResults() → Teardown()
 *
 * Подклассы переопределяют Execute() и Validate() (pure virtual).
 * Хуки Setup/SaveResults/Teardown — опциональны (default: пустые).
 *
 * GoF Template Method:
 *   - Run()    — invariant (НЕ переопределять!)
 *   - Execute  — abstract (ОБЯЗАТЕЛЬНО переопределить)
 *   - Validate — abstract (ОБЯЗАТЕЛЬНО переопределить)
 *   - Setup/SaveResults/Teardown — hooks (опционально)
 *
 * GRASP Controller: StrategyTestBase координирует весь поток теста.
 * DIP: зависит от ISignalStrategy, не от конкретных классов.
 *
 * @date 2026-03-15
 */

#if ENABLE_ROCM

#include "antenna_test_params.hpp"
#include "i_signal_strategy.hpp"
#include "signal_strategy_factory.hpp"

#include "config/antenna_processor_config.hpp"
#include "weight_generator.hpp"

#include "interface/i_backend.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <cstdio>
#include <stdexcept>

namespace test_strategies {

/**
 * @brief Базовый класс всех тестов антенной стратегии
 *
 * Инициализация:
 * @code
 *   class MyTest : public StrategyTestBase {
 *     void Execute() override { ... }
 *     void Validate() override { ... }
 *     std::string GetName() const override { return "MyTest"; }
 *   };
 *
 *   auto sig = SignalStrategyFactory::Create(SignalVariant::LFM_NO_DELAY);
 *   MyTest test(backend, std::move(sig), AntennaTestParams::Small());
 *   test.Run();
 * @endcode
 */
class StrategyTestBase {
public:
  // ── Constructor ────────────────────────────────────────────────────────────
  StrategyTestBase(drv_gpu_lib::IBackend* backend,
                   std::unique_ptr<ISignalStrategy> signal_strategy,
                   const AntennaTestParams& params)
    : backend_(backend),
      signal_strategy_(std::move(signal_strategy)),
      params_(params) {}

  virtual ~StrategyTestBase() { CleanupGpu(); }

  // ── Template Method — НЕ переопределять ───────────────────────────────────

  /**
   * @brief Запустить тест (Template Method)
   *
   * Порядок вызовов:
   *   Setup() → GenerateSignals() → PrepareMatrix() → Execute()
   *           → Validate() → SaveResults() → Teardown()
   *
   * При любом исключении вызывается Teardown() + CleanupGpu().
   */
  void Run() {
    auto& con   = drv_gpu_lib::ConsoleOutput::GetInstance();
    const int g = 0;
    const std::string mod = "StratTest";

    con.Print(g, mod, "");
    con.Print(g, mod, ("─── " + GetName() +
                       " [" + signal_strategy_->Name() + "] ───").c_str());

    try {
      Setup();
      BuildConfig();
      GenerateSignals();
      PrepareMatrix();
      Execute();
      Validate();
      SaveResults();
    } catch (const std::exception& ex) {
      auto& c = drv_gpu_lib::ConsoleOutput::GetInstance();
      c.Print(0, "StratTest", (std::string("EXCEPTION: ") + ex.what()).c_str());
      CleanupGpu();
      Teardown();
      throw;
    }

    CleanupGpu();
    Teardown();
    auto& c = drv_gpu_lib::ConsoleOutput::GetInstance();
    c.Print(0, "StratTest", ("[+] PASS: " + GetName()).c_str());
  }

  // ── Abstract hooks (SRP: каждый подкласс реализует свою логику) ──────────

  virtual void Execute()  = 0;   ///< Логика теста (GEMM / step-by-step / profiling)
  virtual void Validate() = 0;   ///< Проверка результатов
  virtual std::string GetName() const = 0;

  // ── Optional hooks ────────────────────────────────────────────────────────

  virtual void Setup()       {}  ///< Инициализация доп. ресурсов
  virtual void SaveResults() {}  ///< Запись промежуточных данных в файлы
  virtual void Teardown()    {}  ///< Освобождение доп. ресурсов

protected:
  // ── Доступно подклассам ───────────────────────────────────────────────────

  drv_gpu_lib::IBackend*      backend_;
  AntennaTestParams           params_;
  strategies::AntennaProcessorConfig cfg_;

  void*  d_S_ = nullptr;  ///< GPU сигнал [n_ant × n_samples], owned by test
  void*  d_W_ = nullptr;  ///< GPU матрица весов [n_ant × n_ant], owned by test

  // ── Helper: форматирование строки ────────────────────────────────────────

  template<typename... Args>
  static std::string fmt(const char* format, Args... args) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), format, args...);
    return buf;
  }

private:
  std::unique_ptr<ISignalStrategy> signal_strategy_;

  // ── Шаги Run() (private — Template Method не переопределяется) ───────────

  /// Строим AntennaProcessorConfig из AntennaTestParams
  void BuildConfig() {
    cfg_.n_ant               = params_.n_ant;
    cfg_.n_samples           = params_.n_samples;
    cfg_.sample_rate         = params_.fs;
    cfg_.signal_frequency_hz = params_.f0_hz;
    cfg_.scenario_mode       = strategies::PostFftScenarioMode::ALL_REQUIRED;
    cfg_.debug_mode          = false;
    // NOTE: n_beams не поддержан в AntennaProcessorConfig.
    // При добавлении n_beams → cfg_.n_beams = params_.n_beams;
  }

  /// Генерируем сигнал через стратегию → d_S_
  void GenerateSignals() {
    auto sd = signal_strategy_->Generate(backend_, params_);
    d_S_     = sd.d_S;
    auto& c = drv_gpu_lib::ConsoleOutput::GetInstance();
    c.Print(0, "StratTest", fmt("  Signal '%s': %u ant x %u pts",
        signal_strategy_->Name().c_str(), params_.n_ant, params_.n_samples).c_str());
  }

  /**
   * @brief Подготовить матрицу весов W на GPU → d_W_
   *
   * Использует WeightGenerator::generate_delay_and_sum (квадратная n_ant×n_ant).
   *
   * @note Для не-квадратной матрицы (n_beams=100, n_ant=2500):
   *       нужно добавить n_beams в AntennaProcessorConfig + GemmStep.
   *       TODO: реализовать generate_identity_rectangular(n_ant, n_beams).
   */
  void PrepareMatrix() {
    strategies::WeightParams wp;
    wp.n_ant    = params_.n_ant;
    wp.f0       = static_cast<double>(params_.f0_hz);
    wp.tau_base = 0.0;
    wp.tau_step = 0.0;  // identity-like: нет задержки → W = I / sqrt(n_ant)
    auto W_cpu  = strategies::WeightGenerator::generate_delay_and_sum(wp);
    d_W_        = strategies::WeightGenerator::upload_to_gpu(backend_, W_cpu);

    auto& c = drv_gpu_lib::ConsoleOutput::GetInstance();
    c.Print(0, "StratTest", fmt("  W matrix: %u×%u (identity-like)",
        params_.n_ant, params_.n_ant).c_str());
  }

  /// Освобождение GPU ресурсов (идемпотентно)
  void CleanupGpu() {
    if (d_S_) { hipFree(d_S_);           d_S_ = nullptr; }
    if (d_W_) { backend_->Free(d_W_);    d_W_ = nullptr; }
  }
};

}  // namespace test_strategies

#endif  // ENABLE_ROCM
