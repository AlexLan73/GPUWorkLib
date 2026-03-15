#pragma once

/**
 * @file antenna_test_params.hpp
 * @brief AntennaTestParams — единый носитель всех тестовых параметров стратегии
 *
 * GRASP Information Expert: содержит ВСЁ что нужно для создания теста,
 * включая вариант сигнала, размеры матриц и параметры ЛЧМ.
 *
 * Базовые параметры (TestStrategia.md):
 *   n_ant=2500, n_samples=5000, fs=0.5 МГц, n_beams=100 (не квадратная!)
 *   fdev=90 кГц, матрица — единичная 2500×100
 *
 * @date 2026-03-15
 */

#include <cstdint>
#include <string>

namespace test_strategies {

// ─────────────────────────────────────────────────────────────────────────────
// SignalVariant — 4 варианта сигнала (Strategy GoF)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Варианты входного сигнала для тестов стратегии
 *
 * SIN           — простой синусоидальный сигнал (fdev=0)
 * LFM_NO_DELAY  — ЛЧМ без задержек (2.1 по TestStrategia.md)
 * LFM_WITH_DELAY — ЛЧМ с задержками, без lch_farrow (2.2.1)
 * LFM_FARROW    — ЛЧМ с дробными задержками через lch_farrow (2.2.2)
 */
enum class SignalVariant {
  SIN,
  LFM_NO_DELAY,
  LFM_WITH_DELAY,
  LFM_FARROW
};

/// Читаемое имя варианта для логов
inline const char* SignalVariantName(SignalVariant v) {
  switch (v) {
    case SignalVariant::SIN:           return "SIN";
    case SignalVariant::LFM_NO_DELAY:  return "LFM_NO_DELAY";
    case SignalVariant::LFM_WITH_DELAY: return "LFM_WITH_DELAY";
    case SignalVariant::LFM_FARROW:    return "LFM_FARROW";
    default:                           return "UNKNOWN";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// AntennaTestParams — Information Expert (GRASP)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Параметры теста антенной стратегии
 *
 * Используется как единый носитель параметров для StrategyTestBase
 * и всех наследников. Не зависит от GPU.
 *
 * Дефолтные значения — "малый" тест для быстрой компиляции и проверки.
 * Для полного теста (2500 антенн × 5000 отсчётов) — AntennaTestParams::FullSpec().
 *
 * @note n_beams != n_ant → НЕ квадратная матрица W (2500×100)
 *       Поддержка non-square требует n_beams в AntennaProcessorConfig:
 *       TODO: добавить n_beams в AntennaProcessorConfig + GemmStep
 */
struct AntennaTestParams {
  // ── Размеры (TestStrategia.md: n_ant=2500, n_samples=5000) ──────────────
  uint32_t n_ant     = 100;      ///< Число антенн (малый тест по умолчанию)
  uint32_t n_samples = 5000;     ///< Отсчётов на антенну
  uint32_t n_beams   = 100;      ///< Столбцы матрицы W (=n_ant → квадратная)

  // ── Частоты ─────────────────────────────────────────────────────────────
  float fs      = 0.5e6f;        ///< Частота дискретизации, Гц
  float fdev_hz = 90e3f;         ///< Девиация ЛЧМ, Гц (0 = CW/SIN)
  float f0_hz   = 100e3f;        ///< Целевая частота для валидации (≈ fs/5)

  // ── Задержки (для LFM_WITH_DELAY и LFM_FARROW) ──────────────────────────
  float tau_step_us = 2.0f;      ///< Шаг задержки на антенну, микросекунды

  // ── Вариант сигнала ──────────────────────────────────────────────────────
  SignalVariant signal_variant = SignalVariant::SIN;

  // ── Опции вывода ─────────────────────────────────────────────────────────
  bool        save_to_files = false;
  std::string output_dir    = "Results/strategies/";

  // ── Фабричные методы ─────────────────────────────────────────────────────

  /**
   * @brief Полный тест по TestStrategia.md: 2500 антенн, 5000 отсчётов
   *
   * @note Non-square W (2500×100): требует n_beams в AntennaProcessorConfig
   *       До добавления — использовать с n_beams=n_ant (квадратная)
   */
  static AntennaTestParams FullSpec(SignalVariant variant = SignalVariant::SIN) {
    AntennaTestParams p;
    p.n_ant           = 2500;
    p.n_beams         = 100;    // non-square!
    p.n_samples       = 5000;
    p.fs              = 0.5e6f;
    p.fdev_hz         = 90e3f;
    p.f0_hz           = 100e3f;
    p.tau_step_us     = 2.0f;
    p.signal_variant  = variant;
    return p;
  }

  /**
   * @brief Быстрый тест: 100 антенн, квадратная матрица (компилируется без n_beams)
   */
  static AntennaTestParams Small(SignalVariant variant = SignalVariant::SIN) {
    AntennaTestParams p;
    p.n_ant           = 100;
    p.n_beams         = 100;    // square: current AntennaProcessorConfig compatible
    p.n_samples       = 5000;
    p.signal_variant  = variant;
    return p;
  }

  /**
   * @brief Debug тест с записью в файлы
   */
  static AntennaTestParams Debug(SignalVariant variant = SignalVariant::SIN) {
    auto p          = Small(variant);
    p.save_to_files = true;
    p.output_dir    = "Results/strategies/debug/";
    return p;
  }
};

}  // namespace test_strategies
