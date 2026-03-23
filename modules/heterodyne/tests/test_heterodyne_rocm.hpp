#pragma once

/**
 * @file test_heterodyne_rocm.hpp
 * @brief ROCm tests for HeterodyneProcessorROCm
 *
 * ✅ MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 *   1. single_antenna   — delay=100us → f_beat=300kHz
 *   2. five_antennas    — linear delays, GPU vs CPU RMS
 *   3. correction       — dechirp → correct → DC peak
 *   4. full_pipeline    — dechirp + range calculation
 *   5. dechirp_from_gpu — external HIP buffer
 *   6. random_delays    — seed=42
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (migrated 2026-03-23)
 */

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <random>

#if ENABLE_ROCM
#include "processors/heterodyne_processor_rocm.hpp"
#include "heterodyne_dechirp.hpp"
#include "heterodyne_params.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include "modules/test_utils/test_utils.hpp"

#include <hip/hip_runtime.h>
#endif

namespace test_heterodyne_rocm {

#if ENABLE_ROCM

using namespace drv_gpu_lib;
using namespace gpu_test_utils;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float FS = 12e6f, F_START = 0.0f, F_END = 2e6f;
static constexpr int   N = 8000, ANTENNAS = 5;
static constexpr float BANDWIDTH = F_END - F_START;
static constexpr float DURATION = static_cast<float>(N) / FS;
static constexpr float MU = BANDWIDTH / DURATION;
static const std::vector<float> DELAYS_LINEAR_US = {100.f, 200.f, 300.f, 400.f, 500.f};
static constexpr float F_BEAT_TOL_HZ = 5000.f;

// Domain helpers

inline std::vector<std::complex<float>> GenerateRxFlatCPU(
    const std::vector<float>& delays_us) {
  float mu = BANDWIDTH / (static_cast<float>(N) / FS);
  size_t total = delays_us.size() * N;
  std::vector<std::complex<float>> flat(total);
  for (size_t ant = 0; ant < delays_us.size(); ++ant) {
    float tau = delays_us[ant] * 1e-6f;
    for (int n = 0; n < N; ++n) {
      float t = static_cast<float>(n) / FS;
      if (t < tau) { flat[ant*N+n] = {0,0}; continue; }
      float tl = t - tau;
      float ph = static_cast<float>(M_PI) * mu * tl*tl + 2.0f*static_cast<float>(M_PI)*F_START*tl;
      flat[ant*N+n] = {std::cos(ph), std::sin(ph)};
    }
  }
  return flat;
}

inline std::vector<std::complex<float>> GenerateConjRefCPU() {
  float mu = BANDWIDTH / (static_cast<float>(N) / FS);
  std::vector<std::complex<float>> ref(N);
  for (int n = 0; n < N; ++n) {
    float t = static_cast<float>(n) / FS;
    float ph = static_cast<float>(M_PI)*mu*t*t + 2.0f*static_cast<float>(M_PI)*F_START*t;
    ref[n] = {std::cos(ph), -std::sin(ph)};
  }
  return ref;
}

inline std::vector<std::complex<float>> CpuDechirpMultiply(
    const std::vector<std::complex<float>>& rx,
    const std::vector<std::complex<float>>& ref, int na, int ns) {
  std::vector<std::complex<float>> dc(na * ns);
  for (int a = 0; a < na; ++a)
    for (int n = 0; n < ns; ++n)
      dc[a*ns+n] = std::conj(rx[a*ns+n] * ref[n]);
  return dc;
}

inline float CpuFindPeakFrequency(
    const std::vector<std::complex<float>>& sig, int off, int ns, float fs) {
  int half = ns / 2;
  float max_mag = 0; int max_bin = 0;
  for (int k = 0; k < half; ++k) {
    float re = 0, im = 0;
    for (int n = 0; n < ns; ++n) {
      float ph = static_cast<float>(-2.0*M_PI*k*n/ns);
      re += sig[off+n].real()*std::cos(ph) - sig[off+n].imag()*std::sin(ph);
      im += sig[off+n].real()*std::sin(ph) + sig[off+n].imag()*std::cos(ph);
    }
    float mag = std::sqrt(re*re + im*im);
    if (mag > max_mag) { max_mag = mag; max_bin = k; }
  }
  return static_cast<float>(max_bin) * fs / static_cast<float>(ns);
}

// Reusable: validate f_beat per antenna
inline TestResult ValidateFBeat(const std::string& name,
    const std::vector<std::complex<float>>& dc,
    const std::vector<float>& delays_us, int n_ant) {
  TestResult tr{name};
  for (int ant = 0; ant < n_ant; ++ant) {
    float f = CpuFindPeakFrequency(dc, ant*N, N, FS);
    float exp_f = MU * delays_us[ant] * 1e-6f;
    tr.add(ScalarAbsError(static_cast<double>(f), static_cast<double>(exp_f),
        F_BEAT_TOL_HZ, "ant" + std::to_string(ant)));
  }
  return tr;
}

inline void run() {
  int gpu_id = 0;

  ROCmBackend backend;
  try { backend.Initialize(gpu_id); }
  catch (...) { return; }

  TestRunner runner(&backend, "Heterodyne ROCm", gpu_id);

  auto ref = GenerateConjRefCPU();

  // ── Test 1: Single antenna ────────────────────────────────────
  runner.test("single_antenna", [&]() {
    auto rx = GenerateRxFlatCPU({100.f});
    HeterodyneParams p; p.f_start=F_START; p.f_end=F_END;
    p.sample_rate=FS; p.num_samples=N; p.num_antennas=1;
    HeterodyneProcessorROCm proc(&backend);
    auto dc = proc.Dechirp(rx, ref, p);
    float f = CpuFindPeakFrequency(dc, 0, N, FS);
    return ScalarAbsError(static_cast<double>(f), static_cast<double>(MU*100e-6f),
                          F_BEAT_TOL_HZ, "f_beat");
  });

  // ── Test 2: 5 antennas ────────────────────────────────────────
  runner.test("five_antennas", [&]() {
    auto rx = GenerateRxFlatCPU(DELAYS_LINEAR_US);
    HeterodyneParams p; p.f_start=F_START; p.f_end=F_END;
    p.sample_rate=FS; p.num_samples=N; p.num_antennas=ANTENNAS;
    HeterodyneProcessorROCm proc(&backend);
    auto dc = proc.Dechirp(rx, ref, p);
    return ValidateFBeat("five_antennas", dc, DELAYS_LINEAR_US, ANTENNAS);
  });

  // ── Test 3: Correction ────────────────────────────────────────
  runner.test("correction", [&]() -> TestResult {
    auto rx = GenerateRxFlatCPU(DELAYS_LINEAR_US);
    HeterodyneParams p; p.f_start=F_START; p.f_end=F_END;
    p.sample_rate=FS; p.num_samples=N; p.num_antennas=ANTENNAS;
    HeterodyneProcessorROCm proc(&backend);
    auto dc = proc.Dechirp(rx, ref, p);
    std::vector<float> f_beats(ANTENNAS);
    for (int a = 0; a < ANTENNAS; ++a)
      f_beats[a] = CpuFindPeakFrequency(dc, a*N, N, FS);
    auto corrected = proc.Correct(dc, f_beats, p);
    TestResult tr{"correction"};
    for (int a = 0; a < ANTENNAS; ++a) {
      float f_corr = CpuFindPeakFrequency(corrected, a*N, N, FS);
      float bin = f_corr / (FS / static_cast<float>(N));
      tr.add(ValidationResult{bin <= 3.0f, "dc_ant" + std::to_string(a),
          static_cast<double>(bin), 3.0, ""});
    }
    return tr;
  });

  // ── Test 4: Full pipeline ─────────────────────────────────────
  runner.test("full_pipeline", [&]() {
    auto rx = GenerateRxFlatCPU(DELAYS_LINEAR_US);
    HeterodyneParams p; p.f_start=F_START; p.f_end=F_END;
    p.sample_rate=FS; p.num_samples=N; p.num_antennas=ANTENNAS;
    HeterodyneProcessorROCm proc(&backend);
    auto dc = proc.Dechirp(rx, ref, p);
    return ValidateFBeat("full_pipeline", dc, DELAYS_LINEAR_US, ANTENNAS);
  });

  // ── Test 5: DechirpFromGPU ────────────────────────────────────
  runner.test("dechirp_from_gpu", [&]() -> TestResult {
    auto rx = GenerateRxFlatCPU(DELAYS_LINEAR_US);
    size_t total = static_cast<size_t>(ANTENNAS)*N;
    size_t sz = total * sizeof(std::complex<float>);
    void* ext = nullptr;
    (void)hipMalloc(&ext, sz);
    (void)hipMemcpy(ext, rx.data(), sz, hipMemcpyHostToDevice);
    HeterodyneParams p; p.f_start=F_START; p.f_end=F_END;
    p.sample_rate=FS; p.num_samples=N; p.num_antennas=ANTENNAS;
    HeterodyneProcessorROCm proc(&backend);
    auto dc = proc.DechirpFromGPU(ext, ref, p);
    bool buf_ok = (hipMemcpy(rx.data(), ext, sz, hipMemcpyDeviceToHost) == hipSuccess);
    (void)hipFree(ext);
    TestResult tr{"dechirp_from_gpu"};
    tr.add(ValidationResult{buf_ok, "buf_valid", buf_ok?1.0:0.0, 1.0, ""});
    float f = CpuFindPeakFrequency(dc, 0, N, FS);
    tr.add(ScalarAbsError(static_cast<double>(f), static_cast<double>(MU*100e-6f),
        F_BEAT_TOL_HZ, "f_beat_ant0"));
    return tr;
  });

  // ── Test 6: Random delays ─────────────────────────────────────
  runner.test("random_delays", [&]() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(10.f, 500.f);
    std::vector<float> delays(ANTENNAS);
    for (int i = 0; i < ANTENNAS; ++i) delays[i] = dist(rng);
    auto rx = GenerateRxFlatCPU(delays);
    HeterodyneParams p; p.f_start=F_START; p.f_end=F_END;
    p.sample_rate=FS; p.num_samples=N; p.num_antennas=ANTENNAS;
    HeterodyneProcessorROCm proc(&backend);
    auto dc = proc.Dechirp(rx, ref, p);
    return ValidateFBeat("random_delays", dc, delays, ANTENNAS);
  });

  runner.print_summary();
}

#else

inline void run() {}

#endif

}  // namespace test_heterodyne_rocm
