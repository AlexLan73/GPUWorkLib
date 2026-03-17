#pragma once

/**
 * @file test_range_angle_basic.hpp
 * @brief Basic tests for RangeAngleProcessor (full pipeline)
 *
 * Tests:
 *   1. Construct + SetParams + Process with zero input (smoke test)
 *   2. Range estimation: synthetic LFM with delay τ → R = c*τ/2
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#if ENABLE_ROCM

#include "range_angle_processor.hpp"
#include "interface/i_backend.hpp"
#include "services/console_output.hpp"

#include <complex>
#include <vector>
#include <string>
#include <cmath>

namespace range_angle_basic_test {

// ─── Test 1: smoke ──────────────────────────────────────────────────────────

inline void run_smoke_test(drv_gpu_lib::IBackend* backend) {
  auto& con  = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = backend ? backend->GetDeviceIndex() : 0;

  con.Print(gpu_id, "RangeAngle", "[T1] Smoke: construct + SetParams + Process(zeros)");

  range_angle::RangeAngleProcessor proc(backend);

  range_angle::RangeAngleParams p;
  p.n_ant_az    = 4;
  p.n_ant_el    = 4;
  p.n_samples   = 512;
  p.f_start     = -5e6f;
  p.f_end       = +5e6f;
  p.sample_rate = 12e6f;
  p.carrier_freq    = 435e6f;
  p.antenna_spacing = 0.345f;
  proc.SetParams(p);

  const auto& got = proc.GetParams();
  con.Print(gpu_id, "RangeAngle",
      "[T1] nfft_range=" + std::to_string(got.nfft_range) +
      " n_range_bins=" + std::to_string(got.n_range_bins) +
      " range_res=" + std::to_string(got.range_res_m) + "m");

  size_t n = static_cast<size_t>(p.n_ant_az) * p.n_ant_el * p.n_samples;
  std::vector<std::complex<float>> iq(n, {0.f, 0.f});

  auto result = proc.Process(iq, /*download=*/false);

  bool ok = result.error_message.empty();
  con.Print(gpu_id, "RangeAngle",
      std::string("[T1] Smoke: ") + (ok ? "PASS" : "FAIL") +
      (ok ? "" : " err=" + result.error_message));
}

// ─── Test 2: range estimation ────────────────────────────────────────────────

inline void run_range_test(drv_gpu_lib::IBackend* backend) {
  auto& con  = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = backend ? backend->GetDeviceIndex() : 0;

  con.Print(gpu_id, "RangeAngle", "[T2] Range estimation: tau=0.5ms → R=75km");

  range_angle::RangeAngleParams p;
  p.n_ant_az    = 4;
  p.n_ant_el    = 4;
  p.n_samples   = 50000;
  p.f_start     = -5e6f;
  p.f_end       = +5e6f;
  p.sample_rate = 12e6f;
  p.carrier_freq    = 435e6f;
  p.antenna_spacing = 0.345f;

  range_angle::RangeAngleProcessor proc(backend);
  proc.SetParams(p);

  uint32_t N    = p.n_samples;
  float    fs   = p.sample_rate;
  float    tau  = 0.5e-3f;  // 0.5 ms delay
  float    B    = p.get_bandwidth();
  float    T    = p.get_duration();
  float    mu   = B / T;    // chirp rate Hz/s

  // Build delayed chirp rx = ref shifted by delay_samples
  uint32_t delay_samples = static_cast<uint32_t>(tau * fs);
  std::vector<std::complex<float>> rx_single(N, {0.f, 0.f});
  for (uint32_t i = delay_samples; i < N; ++i) {
    float t     = float(i - delay_samples) / fs;
    float phase = 3.14159265f * mu * t * t;
    rx_single[i] = {cosf(phase), sinf(phase)};
  }

  uint32_t n_ant = p.get_n_antennas();
  std::vector<std::complex<float>> data(n_ant * N);
  for (uint32_t a = 0; a < n_ant; ++a) {
    for (uint32_t i = 0; i < N; ++i) {
      data[a * N + i] = rx_single[i];
    }
  }

  auto result = proc.Process(data, /*download=*/true);

  if (!result.error_message.empty()) {
    con.Print(gpu_id, "RangeAngle",
        "[T2] FAIL: " + result.error_message);
    return;
  }

  if (result.targets.empty()) {
    con.Print(gpu_id, "RangeAngle", "[T2] FAIL: no targets found");
    return;
  }

  float R_expected = 3e8f * tau / 2.f;  // 75000 m
  float R_got      = result.targets[0].range_m;
  float err        = fabsf(R_got - R_expected);
  bool  ok         = (err < 1000.f);    // <1 km tolerance

  con.Print(gpu_id, "RangeAngle",
      "[T2] R_expected=" + std::to_string(R_expected) + "m"
      " R_got=" + std::to_string(R_got) + "m"
      " err=" + std::to_string(err) + "m"
      " [" + std::string(ok ? "PASS" : "FAIL") + "]");
}

// ─── Entry point ─────────────────────────────────────────────────────────────

inline void run(drv_gpu_lib::IBackend* backend) {
  auto& con  = drv_gpu_lib::ConsoleOutput::GetInstance();
  int gpu_id = backend ? backend->GetDeviceIndex() : 0;

  con.Print(gpu_id, "RangeAngle", "=== RangeAngle Basic Tests ===");

  if (!backend) {
    con.Print(gpu_id, "RangeAngle", "SKIP: backend is null");
    return;
  }

  run_smoke_test(backend);
  run_range_test(backend);

  con.Print(gpu_id, "RangeAngle", "=== RangeAngle Basic Tests Done ===");
}

// Legacy run() without args — called from all_test.hpp (backend passed below)
inline void run() {
  // NOTE: backend must be injected — see all_test.hpp
}

}  // namespace range_angle_basic_test

#endif  // ENABLE_ROCM
