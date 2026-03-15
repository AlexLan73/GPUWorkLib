#pragma once

/**
 * @file signal_strategy_factory.hpp
 * @brief SignalStrategyFactory — Factory Method (GoF)
 *
 * Централизованное создание ISignalStrategy по SignalVariant.
 * OCP: новый вариант = добавить кейс в switch + новый класс в signal_strategies.hpp.
 *
 * @date 2026-03-15
 */

#if ENABLE_ROCM

#include "i_signal_strategy.hpp"
#include "signal_strategies.hpp"
#include "antenna_test_params.hpp"

#include <memory>
#include <stdexcept>

namespace test_strategies {

/**
 * @brief Фабрика сигнальных стратегий (Factory Method GoF)
 *
 * Использование:
 * @code
 *   auto strategy = SignalStrategyFactory::Create(SignalVariant::LFM_FARROW);
 *   SignalData data = strategy->Generate(backend, params);
 * @endcode
 */
class SignalStrategyFactory {
public:
  /**
   * @brief Создать стратегию по варианту сигнала
   * @param variant  Один из SignalVariant
   * @return unique_ptr<ISignalStrategy>
   * @throws std::invalid_argument при неизвестном варианте
   */
  static std::unique_ptr<ISignalStrategy> Create(SignalVariant variant) {
    switch (variant) {
      case SignalVariant::SIN:
        return std::make_unique<SinSignalStrategy>();
      case SignalVariant::LFM_NO_DELAY:
        return std::make_unique<LfmNoDelayStrategy>();
      case SignalVariant::LFM_WITH_DELAY:
        return std::make_unique<LfmWithDelayStrategy>();
      case SignalVariant::LFM_FARROW:
        return std::make_unique<LfmFarrowStrategy>();
      default:
        throw std::invalid_argument(
            "SignalStrategyFactory::Create: unknown SignalVariant");
    }
  }
};

}  // namespace test_strategies

#endif  // ENABLE_ROCM
