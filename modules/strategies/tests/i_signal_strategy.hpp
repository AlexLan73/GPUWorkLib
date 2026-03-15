#pragma once

/**
 * @file i_signal_strategy.hpp
 * @brief ISignalStrategy — интерфейс стратегии генерации сигнала (GoF Strategy)
 *
 * Паттерн Strategy (GoF): различные алгоритмы генерации сигнала
 * (SIN / LFM / LFM+задержки / LFM+Farrow) взаимозаменяемы без изменения
 * клиентского кода (StrategyTestBase).
 *
 * OCP (SOLID): добавить новый тип сигнала = добавить новый класс,
 *              НЕ изменяя ISignalStrategy или StrategyTestBase.
 *
 * DIP (SOLID): StrategyTestBase зависит от абстракции ISignalStrategy,
 *              не от конкретных SignalStrategies.
 *
 * @date 2026-03-15
 */

#if ENABLE_ROCM

#include "antenna_test_params.hpp"

#include <string>
#include <cstdint>

namespace drv_gpu_lib { class IBackend; }

namespace test_strategies {

// ─────────────────────────────────────────────────────────────────────────────
// SignalData — результат генерации сигнала
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief GPU-данные сигнала, возвращаемые ISignalStrategy::Generate()
 *
 * d_S — указатель на GPU-буфер complex<float> [n_ant × n_samples].
 *       Выделен через hipMalloc внутри стратегии.
 *       ВЛАДЕЛЕЦ — стратегия (не тест!): вызывается Free() в деструкторе.
 *
 * @note Стратегия ОБЯЗАНА освободить d_S при своём уничтожении или явном
 *       вызове Free(d_S, backend).
 */
struct SignalData {
  void*    d_S      = nullptr;  ///< GPU указатель (hipMalloc)
  uint32_t n_ant    = 0;
  uint32_t n_samples = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// ISignalStrategy — Strategy GoF interface
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Интерфейс стратегии генерации входного сигнала
 *
 * Реализации:
 *   SinSignalStrategy      — простой синус (fdev=0)
 *   LfmNoDelayStrategy     — ЛЧМ без задержек
 *   LfmWithDelayStrategy   — ЛЧМ с линейными задержками
 *   LfmFarrowStrategy      — ЛЧМ + дробные задержки через LchFarrowROCm
 */
class ISignalStrategy {
public:
  virtual ~ISignalStrategy() = default;

  /**
   * @brief Сгенерировать сигнал на GPU
   *
   * @param backend  Инициализированный ROCm backend
   * @param params   Параметры теста (n_ant, n_samples, fs, f0_hz, ...)
   * @return SignalData с указателем d_S на GPU-память
   *
   * @note Возвращённый d_S является СОБСТВЕННОСТЬЮ вызывающего кода.
   *       Освободить через hipFree(result.d_S) или backend->Free(result.d_S).
   */
  virtual SignalData Generate(drv_gpu_lib::IBackend* backend,
                              const AntennaTestParams& params) = 0;

  /// Читаемое имя стратегии для логов
  virtual std::string Name() const = 0;
};

}  // namespace test_strategies

#endif  // ENABLE_ROCM
