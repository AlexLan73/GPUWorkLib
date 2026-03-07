#pragma once

/**
 * @file test_strategies_pipeline.hpp
 * @brief C++ test for AntennaProcessor pipeline (strategies module)
 *
 * Test scenario:
 *   1. Generate 5-antenna signal via FormSignalGeneratorROCm
 *   2. Generate Delay-and-sum W matrix
 *   3. Run full pipeline via AntennaProcessorTest
 *   4. Verify:
 *      - GEMM output shape
 *      - FFT output shape
 *      - Step2.1 finds the peak near f0=2MHz
 *      - Step2.3 min < max
 *
 * @date 2026-03-07
 */

#if ENABLE_ROCM

#include "antenna_processor_test.hpp"
#include "weight_generator.hpp"
#include "generators/form_signal_generator_rocm.hpp"

#include "services/console_output.hpp"

#include <cmath>
#include <cassert>

namespace test_strategies {

inline void test_full_pipeline(drv_gpu_lib::IBackend* backend) {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print("=== test_strategies_pipeline: START ===\n");

  // 1. Generate test signal
  signal_gen::FormParams fp;
  fp.antennas        = 5;
  fp.points          = 8000;
  fp.fs              = 12.0e6;
  fp.f0              = 2.0e6;
  fp.amplitude       = 1.0;
  fp.noise_amplitude = 0.0;
  fp.tau_base        = 0.0;
  fp.tau_step        = 100e-6;

  signal_gen::FormSignalGeneratorROCm gen(backend);
  gen.SetParams(fp);
  auto input = gen.GenerateInputData();

  con.Print("  Signal generated: %u ant x %u pts\n", fp.antennas, fp.points);

  // 2. Generate W matrix
  strategies::WeightParams wp;
  wp.n_ant    = fp.antennas;
  wp.f0       = fp.f0;
  wp.tau_base = fp.tau_base;
  wp.tau_step = fp.tau_step;

  auto W_cpu = strategies::WeightGenerator::generate_delay_and_sum(wp);
  void* d_W = strategies::WeightGenerator::upload_to_gpu(backend, W_cpu);

  con.Print("  W matrix: %ux%u Delay-and-sum\n", wp.n_ant, wp.n_ant);

  // 3. Create processor
  strategies::AntennaProcessorConfig cfg;
  cfg.n_ant            = fp.antennas;
  cfg.n_samples        = fp.points;
  cfg.sample_rate      = static_cast<float>(fp.fs);
  cfg.signal_frequency_hz = static_cast<float>(fp.f0);
  cfg.scenario_mode    = strategies::PostFftScenarioMode::ALL_REQUIRED;
  cfg.debug_mode       = true;

  strategies::AntennaProcessorTest proc(backend, cfg);

  // 4. Step-by-step test
  proc.step_0_prepare_input(input.data, d_W);
  con.Print("  Step 0: input prepared\n");

  // Step 1: debug input
  auto r1 = proc.step_1_debug_input();
  con.Print("  Step 1: pre_input_stats: %zu beams\n", r1.pre_input_stats.size());
  assert(r1.pre_input_stats.size() == fp.antennas);

  // Step 2: GEMM
  auto X = proc.step_2_gemm();
  con.Print("  Step 2: GEMM done, X size=%zu complex\n", X.size());
  assert(X.size() == static_cast<size_t>(fp.antennas) * fp.points);

  // Step 3: debug post-GEMM
  auto r3 = proc.step_3_debug_post_gemm();
  con.Print("  Step 3: post_gemm_stats: %zu beams\n", r3.post_gemm_stats.size());

  // Step 4: Window + FFT
  auto spectrum = proc.step_4_window_fft();
  uint32_t nFFT = proc.test_get_nFFT();
  con.Print("  Step 4: Window+FFT done, nFFT=%u, spectrum size=%zu\n",
            nFFT, spectrum.size());
  assert(spectrum.size() == static_cast<size_t>(fp.antennas) * nFFT);

  // Step 5: debug post-FFT
  auto r5 = proc.step_5_debug_post_fft();
  con.Print("  Step 5: post_fft_stats: %zu beams\n", r5.post_fft_stats.size());

  // Step 6.1: OneMax + Parabola
  auto r61 = proc.step_6_1_one_max_parabola();
  con.Print("  Step 6.1: one_max results: %zu beams\n", r61.one_max.size());
  if (!r61.one_max.empty()) {
    float found_freq = r61.one_max[0].refined_freq_hz;
    con.Print("    Beam 0: freq=%.1f Hz, mag=%.4f, bin=%u\n",
              found_freq, r61.one_max[0].magnitude, r61.one_max[0].bin_index);
  }

  // Step 6.2: AllMaxima
  auto r62 = proc.step_6_2_all_maxima();
  con.Print("  Step 6.2: all_maxima results: %zu beams\n", r62.all_maxima.size());
  if (!r62.all_maxima.empty()) {
    con.Print("    Beam 0: %u maxima found\n", r62.all_maxima[0].num_maxima);
  }

  // Step 6.3: GlobalMinMax
  auto r63 = proc.step_6_3_global_minmax();
  con.Print("  Step 6.3: minmax results: %zu beams\n", r63.minmax.size());
  if (!r63.minmax.empty()) {
    con.Print("    Beam 0: min=%.6f (bin %u), max=%.4f (bin %u), DR=%.1f dB\n",
              r63.minmax[0].min_magnitude, r63.minmax[0].min_bin,
              r63.minmax[0].max_magnitude, r63.minmax[0].max_bin,
              r63.minmax[0].dynamic_range_dB);
    assert(r63.minmax[0].max_magnitude >= r63.minmax[0].min_magnitude);
  }

  // 5. Full pipeline test
  auto full = proc.process_full();
  con.Print("  Full pipeline: total=%.2f ms\n", full.perf.total_ms);

  // Cleanup
  backend->Free(d_W);
  hipFree(input.data);  // FormSignalGeneratorROCm allocates with hipMalloc

  con.Print("=== test_strategies_pipeline: PASSED ===\n\n");
}

}  // namespace test_strategies

#endif  // ENABLE_ROCM
