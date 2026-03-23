#pragma once

/**
 * @file test_signal_generators_rocm_basic.hpp
 * @brief ROCm тесты: CW, LFM, Noise, LfmConjugate генераторы (GPU vs CPU)
 *
 * ✅ MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 *   1. cw_gpu_vs_cpu       — CW single beam, GPU vs CPU reference
 *   2. cw_multi_beam       — CW 8 beams with freq_step
 *   3. lfm_gpu_vs_cpu      — LFM chirp, GPU vs CPU
 *   4. noise_statistics    — mean ~0, power ~1
 *   5. lfm_conjugate       — conj(LFM), GPU vs CPU + negative phase check
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22 (migrated 2026-03-23)
 */

#if ENABLE_ROCM

#include "generators/cw_generator_rocm.hpp"
#include "generators/lfm_generator_rocm.hpp"
#include "generators/noise_generator_rocm.hpp"
#include "generators/lfm_conjugate_generator_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"

// test_utils — единая тестовая инфраструктура
#include "modules/test_utils/test_utils.hpp"

#include <vector>
#include <complex>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_signal_generators_rocm_basic {

using namespace signal_gen;
using namespace drv_gpu_lib;
using namespace gpu_test_utils;

// =========================================================================
// run() — TestRunner (функциональный стиль)
// =========================================================================

inline void run() {
  int gpu_id = 0;

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  TestRunner runner(&backend, "SigGen ROCm", gpu_id);

  // ── Test 1: CW GPU vs CPU ─────────────────────────────────────

  runner.test("cw_gpu_vs_cpu", [&]() {
    CwParams cw;
    cw.f0 = 250.0;
    cw.amplitude = 1.5;
    cw.phase = 0.3;
    SystemSampling sys{4000.0, 4096};

    CwGeneratorROCm gen(&backend);
    auto cpu_data = gen.GenerateToCpu(sys, cw, 1);

    auto gpu_result = gen.GenerateToGpu(sys, cw, 1);
    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, sys.length);

    return AbsError(gpu_data.data(), cpu_data.data(), sys.length,
                    tolerance::kComplex32, "cw_max_err");
  });

  // ── Test 2: CW multi-beam (8 beams, freq_step) ────────────────

  runner.test("cw_multi_beam", [&]() -> TestResult {
    CwParams cw;
    cw.f0 = 100.0;
    cw.freq_step = 50.0;
    cw.amplitude = 1.0;
    SystemSampling sys{2000.0, 2048};
    uint32_t beam_count = 8;

    CwGeneratorROCm gen(&backend);
    auto gpu_result = gen.GenerateToGpu(sys, cw, beam_count);

    size_t total = static_cast<size_t>(beam_count) * sys.length;
    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, total);

    TestResult tr{"cw_multi_beam"};
    for (uint32_t b = 0; b < beam_count; ++b) {
      float freq = static_cast<float>(cw.f0 + b * cw.freq_step);
      float fs = static_cast<float>(sys.fs);

      std::vector<std::complex<float>> cpu_beam(sys.length);
      for (size_t i = 0; i < sys.length; ++i) {
        float t = static_cast<float>(i) / fs;
        float ph = 2.0f * static_cast<float>(M_PI) * freq * t
                  + static_cast<float>(cw.phase);
        cpu_beam[i] = {static_cast<float>(cw.amplitude) * std::cos(ph),
                       static_cast<float>(cw.amplitude) * std::sin(ph)};
      }

      tr.add(AbsError(gpu_data.data() + b * sys.length,
                       cpu_beam.data(), sys.length,
                       tolerance::kComplex32, "beam" + std::to_string(b)));
    }
    return tr;
  });

  // ── Test 3: LFM GPU vs CPU ────────────────────────────────────

  runner.test("lfm_gpu_vs_cpu", [&]() {
    LfmParams lfm;
    lfm.f_start = 100.0;
    lfm.f_end = 500.0;
    SystemSampling sys{4000.0, 4096};

    LfmGeneratorROCm gen(&backend);
    auto cpu_data = gen.GenerateToCpu(sys, lfm, 1);

    auto gpu_result = gen.GenerateToGpu(sys, lfm, 1);
    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, sys.length);

    return AbsError(gpu_data.data(), cpu_data.data(), sys.length,
                    tolerance::kComplex32, "lfm_max_err");
  });

  // ── Test 4: Noise statistics ──────────────────────────────────

  runner.test("noise_statistics", [&]() -> TestResult {
    NoiseParams np;
    np.power = 1.0;
    np.seed = 42;
    SystemSampling sys{4000.0, 8192};

    NoiseGeneratorROCm gen(&backend);
    auto gpu_result = gen.GenerateToGpu(sys, np, 1);
    auto data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, sys.length);

    double sum_re = 0, sum_im = 0, sum_sq = 0;
    for (size_t i = 0; i < sys.length; ++i) {
      sum_re += data[i].real();
      sum_im += data[i].imag();
      sum_sq += std::norm(data[i]);
    }
    double mean_re = sum_re / sys.length;
    double mean_im = sum_im / sys.length;
    double mean_power = sum_sq / sys.length;

    TestResult tr{"noise_statistics"};
    tr.add(ScalarAbsError(mean_re, 0.0, 0.1, "mean_re"));
    tr.add(ScalarAbsError(mean_im, 0.0, 0.1, "mean_im"));
    // Complex Gaussian: total power = var_re + var_im = 2 * np.power
    tr.add(ScalarAbsError(mean_power, 2.0 * np.power, 0.3, "power"));
    return tr;
  });

  // ── Test 5: LfmConjugate GPU vs CPU ───────────────────────────

  runner.test("lfm_conjugate", [&]() -> TestResult {
    LfmParams lfm;
    lfm.f_start = 100.0;
    lfm.f_end = 500.0;
    SystemSampling sys{4000.0, 4096};

    LfmConjugateGeneratorROCm gen(&backend, lfm);
    gen.SetSampling(sys);

    auto cpu_data = gen.GenerateToCpu();
    void* gpu_ptr = gen.GenerateToGpu();
    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_ptr, sys.length);

    TestResult tr{"lfm_conjugate"};
    tr.add(AbsError(gpu_data.data(), cpu_data.data(), sys.length,
                     tolerance::kComplex32, "conj_max_err"));

    // Verify conjugate property: phase should be negative
    float phase1 = std::arg(gpu_data[1]);
    tr.add(ValidationResult{
        phase1 < 0.0f, "neg_phase",
        static_cast<double>(phase1), 0.0,
        phase1 < 0.0f ? "phase[1] < 0 OK" : "phase[1] >= 0 FAIL"});
    return tr;
  });

  runner.print_summary();
}

}  // namespace test_signal_generators_rocm_basic

#endif  // ENABLE_ROCM
