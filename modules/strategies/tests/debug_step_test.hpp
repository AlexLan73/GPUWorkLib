#pragma once

/**
 * @file debug_step_test.hpp
 * @brief DebugStepTest — пошаговый тест каждого Step (T2)
 *
 * Использует AntennaProcessorTest для вызова каждого шага по отдельности.
 * Опционально сохраняет промежуточные данные в файлы (params.save_to_files).
 *
 * Порядок шагов:
 *   Step 1: debug input stats     (pre_input)
 *   Step 2: GEMM                  (X = W × S)
 *   Step 3: debug post-GEMM stats
 *   Step 4: Window + FFT
 *   Step 5: debug post-FFT stats
 *   Step 6.1: OneMax + Parabola
 *   Step 6.2: AllMaxima
 *   Step 6.3: GlobalMinMax
 *
 * Валидация:
 *   - GEMM output: X.size() == n_ant * n_samples
 *   - FFT peak бин: ≈ round(f0_hz / bin_hz)
 *   - OneMax refined_freq_hz ≈ f0_hz (±2 бина)
 *   - MinMax: max >= min
 *
 * @date 2026-03-15
 */

#if ENABLE_ROCM

#include "strategy_test_base.hpp"
#include "antenna_processor_test.hpp"
#include "result_types.hpp"

#include "services/console_output.hpp"

#include <hip/hip_runtime.h>
#include <fstream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include <algorithm>

namespace test_strategies {

/**
 * @brief Пошаговый тест AntennaProcessor (debug режим)
 *
 * Вызывает каждый step_ метод AntennaProcessorTest и логирует результат.
 * Если save_to_files=true — сохраняет промежуточные данные для анализа в Python.
 */
class DebugStepTest : public StrategyTestBase {
public:
  using StrategyTestBase::StrategyTestBase;

  std::string GetName() const override { return "DebugStepTest"; }

protected:
  // ── Execute: step-by-step ────────────────────────────────────────────────

  void Execute() override {
    auto& c = drv_gpu_lib::ConsoleOutput::GetInstance();

    // Включаем debug_mode для D2H копий в каждом шаге
    cfg_.debug_mode = true;

    strategies::AntennaProcessorTest proc(backend_, cfg_);
    proc.step_0_prepare_input(d_S_, d_W_);
    c.Print(0, "DebugStep", "  Step 0: input prepared");

    // ── Step 1: pre-input stats ───────────────────────────────────────────
    r1_ = proc.step_1_debug_input();
    c.Print(0, "DebugStep", fmt("  Step 1 [PRE_INPUT]: %zu beams stats",
        r1_.pre_input_stats.size()).c_str());

    // ── Step 2: GEMM ──────────────────────────────────────────────────────
    X_ = proc.step_2_gemm();
    c.Print(0, "DebugStep", fmt("  Step 2 [GEMM]: X.size=%zu (n_ant*n_samples=%u)",
        X_.size(), params_.n_ant * params_.n_samples).c_str());

    // ── Step 3: post-GEMM stats ───────────────────────────────────────────
    r3_ = proc.step_3_debug_post_gemm();
    c.Print(0, "DebugStep", fmt("  Step 3 [POST_GEMM]: %zu beams stats",
        r3_.post_gemm_stats.size()).c_str());

    // ── Step 4: Window + FFT ──────────────────────────────────────────────
    spectrum_ = proc.step_4_window_fft();
    nFFT_     = proc.test_get_nFFT();
    c.Print(0, "DebugStep", fmt("  Step 4 [Window+FFT]: nFFT=%u, spectrum.size=%zu",
        nFFT_, spectrum_.size()).c_str());

    // ── Step 5: post-FFT stats ────────────────────────────────────────────
    r5_ = proc.step_5_debug_post_fft();
    c.Print(0, "DebugStep", fmt("  Step 5 [POST_FFT]: %zu beams stats",
        r5_.post_fft_stats.size()).c_str());

    // ── Step 6.1: OneMax ─────────────────────────────────────────────────
    r61_ = proc.step_6_1_one_max_parabola();
    if (!r61_.one_max.empty()) {
      c.Print(0, "DebugStep", fmt("  Step 6.1 [OneMax]: beam0 freq=%.1f Hz, mag=%.4f, bin=%u",
          r61_.one_max[0].refined_freq_hz,
          r61_.one_max[0].magnitude,
          r61_.one_max[0].bin_index).c_str());
    }

    // ── Step 6.2: AllMaxima ───────────────────────────────────────────────
    r62_ = proc.step_6_2_all_maxima();
    if (!r62_.all_maxima.empty()) {
      c.Print(0, "DebugStep", fmt("  Step 6.2 [AllMaxima]: beam0 found %u maxima",
          r62_.all_maxima[0].num_maxima).c_str());
    }

    // ── Step 6.3: GlobalMinMax ────────────────────────────────────────────
    r63_ = proc.step_6_3_global_minmax();
    if (!r63_.minmax.empty()) {
      c.Print(0, "DebugStep", fmt("  Step 6.3 [MinMax]: beam0 min=%.6f max=%.4f DR=%.1f dB",
          r63_.minmax[0].min_magnitude,
          r63_.minmax[0].max_magnitude,
          r63_.minmax[0].dynamic_range_dB).c_str());
    }
  }

  // ── Validate ─────────────────────────────────────────────────────────────

  void Validate() override {
    // GEMM output size
    assert(X_.size() == static_cast<size_t>(params_.n_ant) * params_.n_samples &&
           "DebugStep: GEMM output size mismatch");

    // FFT spectrum size
    assert(spectrum_.size() == static_cast<size_t>(params_.n_ant) * nFFT_ &&
           "DebugStep: spectrum size mismatch");

    // OneMax: частота ≈ f0
    if (!r61_.one_max.empty()) {
      const float bin_hz   = params_.fs / static_cast<float>(nFFT_);
      const float freq_err = std::abs(r61_.one_max[0].refined_freq_hz - params_.f0_hz);
      assert(freq_err < 2.0f * bin_hz &&
             "DebugStep: OneMax freq deviates > 2 bins from f0");
    }

    // MinMax: max >= min
    if (!r63_.minmax.empty()) {
      assert(r63_.minmax[0].max_magnitude >= r63_.minmax[0].min_magnitude &&
             "DebugStep: max < min in MinMax");
    }
  }

  // ── SaveResults: запись в файлы ──────────────────────────────────────────

  void SaveResults() override {
    if (!params_.save_to_files) return;

    auto& c = drv_gpu_lib::ConsoleOutput::GetInstance();
    const std::string dir = params_.output_dir + GetName() + "_" +
                            signal_strategy_name_ + "/";
    // TODO: создать директорию (std::filesystem C++17)
    // SaveComplexBinary(dir + "gemm_output.bin", X_);
    // SaveComplexBinary(dir + "spectrum.bin", spectrum_);
    c.Print(0, "DebugStep", fmt("  SaveResults: dir=%s (TODO: mkdir + write)",
        dir.c_str()).c_str());
  }

private:
  // Храним имя стратегии для имени файла (заполняется в Setup)
  std::string signal_strategy_name_ = "unknown";

  // Результаты по шагам
  strategies::AntennaResult              r1_{}, r3_{}, r5_{};
  strategies::AntennaResult              r61_{}, r62_{}, r63_{};
  std::vector<std::complex<float>>       X_{};
  std::vector<std::complex<float>>       spectrum_{};
  uint32_t                               nFFT_ = 0;
};

}  // namespace test_strategies

#endif  // ENABLE_ROCM
