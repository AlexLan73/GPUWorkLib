#pragma once

/**
 * @file signal_strategies.hpp
 * @brief Конкретные стратегии генерации сигналов (GoF Strategy pattern)
 *
 * Реализации ISignalStrategy для 4 вариантов сигнала:
 *
 *   SinSignalStrategy      — синус (fdev=0, tau_step=0)
 *   LfmNoDelayStrategy     — ЛЧМ без задержек (fdev=90кГц, tau_step=0)
 *   LfmWithDelayStrategy   — ЛЧМ с линейными задержками (tau_step>0)
 *   LfmFarrowStrategy      — ЛЧМ + дробные задержки через LchFarrowROCm
 *
 * OCP: добавление нового типа сигнала = новый класс здесь + строка в Factory.
 *
 * @date 2026-03-15
 */

#if ENABLE_ROCM

#include "i_signal_strategy.hpp"
#include "antenna_test_params.hpp"

#include "generators/form_signal_generator_rocm.hpp"
#include "params/form_params.hpp"
#include "lch_farrow_rocm.hpp"

#include "interface/i_backend.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <stdexcept>

namespace test_strategies {

// ─────────────────────────────────────────────────────────────────────────────
// helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/// Строит FormParams на основе AntennaTestParams и fdev/tau_step
inline signal_gen::FormParams MakeFormParams(const AntennaTestParams& p,
                                             float fdev, float tau_step_us) {
  signal_gen::FormParams fp;
  fp.antennas        = p.n_ant;
  fp.points          = p.n_samples;
  fp.fs              = static_cast<double>(p.fs);
  fp.f0              = static_cast<double>(p.f0_hz);
  fp.fdev            = static_cast<double>(fdev);
  fp.amplitude       = 1.0;
  fp.noise_amplitude = 0.0;
  fp.tau_base        = 0.0;
  fp.tau_step        = static_cast<double>(tau_step_us) * 1e-6;  // мкс → с
  return fp;
}

/// Генерирует через FormSignalGeneratorROCm → возвращает SignalData
inline SignalData GenerateViaFormGen(drv_gpu_lib::IBackend* backend,
                                     const AntennaTestParams& p,
                                     float fdev, float tau_step_us) {
  signal_gen::FormParams fp = MakeFormParams(p, fdev, tau_step_us);
  signal_gen::FormSignalGeneratorROCm gen(backend);
  gen.SetParams(fp);
  auto input = gen.GenerateInputData();  // returns InputData<void*>

  SignalData sd;
  sd.d_S      = input.data;  // hipMalloc ptr — caller must hipFree
  sd.n_ant    = p.n_ant;
  sd.n_samples = p.n_samples;
  return sd;
}

}  // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// 1. SinSignalStrategy — синус (fdev=0)
// ─────────────────────────────────────────────────────────────────────────────

class SinSignalStrategy : public ISignalStrategy {
public:
  SignalData Generate(drv_gpu_lib::IBackend* backend,
                      const AntennaTestParams& params) override {
    return detail::GenerateViaFormGen(backend, params,
        /*fdev=*/0.0f, /*tau_step_us=*/0.0f);
  }

  std::string Name() const override { return "SIN"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// 2. LfmNoDelayStrategy — ЛЧМ без задержек (2.1)
// ─────────────────────────────────────────────────────────────────────────────

class LfmNoDelayStrategy : public ISignalStrategy {
public:
  SignalData Generate(drv_gpu_lib::IBackend* backend,
                      const AntennaTestParams& params) override {
    return detail::GenerateViaFormGen(backend, params,
        /*fdev=*/params.fdev_hz, /*tau_step_us=*/0.0f);
  }

  std::string Name() const override { return "LFM_NO_DELAY"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// 3. LfmWithDelayStrategy — ЛЧМ + линейные задержки (2.2.1, без lch_farrow)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Задержки реализованы через tau_step в FormSignalGeneratorROCm.
 * Целочисленные задержки (сдвиг отсчётов), без субсэмпловой интерполяции.
 */
class LfmWithDelayStrategy : public ISignalStrategy {
public:
  SignalData Generate(drv_gpu_lib::IBackend* backend,
                      const AntennaTestParams& params) override {
    return detail::GenerateViaFormGen(backend, params,
        /*fdev=*/params.fdev_hz,
        /*tau_step_us=*/params.tau_step_us);
  }

  std::string Name() const override { return "LFM_WITH_DELAY"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// 4. LfmFarrowStrategy — ЛЧМ + дробные задержки LchFarrowROCm (2.2.2)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Алгоритм:
 *   1. Генерируем ЛЧМ без задержек (tau_step=0) через FormSignalGeneratorROCm
 *   2. Строим вектор дробных задержек: delays[i] = i * tau_step_us (мкс)
 *   3. Применяем LchFarrowROCm::Process() → новый GPU-буфер
 *   4. Освобождаем исходный буфер, возвращаем буфер после Farrow
 */
class LfmFarrowStrategy : public ISignalStrategy {
public:
  SignalData Generate(drv_gpu_lib::IBackend* backend,
                      const AntennaTestParams& params) override {
    // 1. Генерируем ЛЧМ без задержек
    SignalData base = detail::GenerateViaFormGen(backend, params,
        /*fdev=*/params.fdev_hz, /*tau_step_us=*/0.0f);

    // 2. Вектор дробных задержек в мкс
    std::vector<float> delay_us(params.n_ant);
    for (uint32_t i = 0; i < params.n_ant; ++i) {
      delay_us[i] = static_cast<float>(i) * params.tau_step_us;
    }

    // 3. Применяем LchFarrow
    lch_farrow::LchFarrowROCm farrow(backend);
    farrow.SetDelays(delay_us);
    farrow.SetSampleRate(params.fs);

    auto result = farrow.Process(base.d_S, params.n_ant, params.n_samples);

    // 4. Освобождаем исходный буфер
    hipFree(base.d_S);

    SignalData sd;
    sd.d_S      = result.data;  // новый буфер (caller must hipFree)
    sd.n_ant    = params.n_ant;
    sd.n_samples = params.n_samples;
    return sd;
  }

  std::string Name() const override { return "LFM_FARROW"; }
};

}  // namespace test_strategies

#endif  // ENABLE_ROCM
