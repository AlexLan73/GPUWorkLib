#pragma once

/**
 * @file signal_request.hpp
 * @brief Типы запросов на генерацию сигналов
 *
 * SignalKind — тип сигнала (CW, LFM, NOISE)
 * CwParams, LfmParams, NoiseParams — параметры конкретных генераторов
 * SignalRequest — единый запрос с variant
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "system_sampling.hpp"
#include <cstdint>
#include <variant>

namespace signal_gen {

// ════════════════════════════════════════════════════════════════════════════
// Типы сигналов
// ════════════════════════════════════════════════════════════════════════════

enum class SignalKind { CW, LFM, NOISE };

// ════════════════════════════════════════════════════════════════════════════
// Параметры CW (continuous wave — синусоида)
// ════════════════════════════════════════════════════════════════════════════

struct CwParams {
    double f0 = 100.0;           ///< Частота (Hz)
    double phase = 0.0;          ///< Начальная фаза (rad)
    double amplitude = 1.0;      ///< Амплитуда
    bool complex_iq = true;      ///< true = exp(j*phase), false = real only (imag=0)

    // Для multi-beam: freq_i = f0 + i * freq_step
    double freq_step = 0.0;      ///< Шаг частоты между лучами (Hz), 0 = все одинаковые
};

// ════════════════════════════════════════════════════════════════════════════
// Параметры LFM (linear frequency modulation — chirp)
// ════════════════════════════════════════════════════════════════════════════

struct LfmParams {
    double f_start = 100.0;      ///< Начальная частота (Hz)
    double f_end = 500.0;        ///< Конечная частота (Hz)
    double amplitude = 1.0;      ///< Амплитуда
    bool complex_iq = true;      ///< Комплексный выход

    /// Скорость изменения частоты: k = (f_end - f_start) / duration
    double GetChirpRate(double duration) const {
        return (f_end - f_start) / duration;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Параметры шума
// ════════════════════════════════════════════════════════════════════════════

enum class NoiseType { WHITE, GAUSSIAN };

struct NoiseParams {
    NoiseType type = NoiseType::GAUSSIAN;
    double power = 1.0;          ///< Мощность шума (дисперсия для Gaussian)
    uint64_t seed = 0;           ///< 0 = random seed
};

// ════════════════════════════════════════════════════════════════════════════
// Единый запрос
// ════════════════════════════════════════════════════════════════════════════

struct SignalRequest {
    SignalKind kind;
    SystemSampling system;
    std::variant<CwParams, LfmParams, NoiseParams> params;
};

} // namespace signal_gen
