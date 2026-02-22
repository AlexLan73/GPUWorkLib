#pragma once

/**
 * @file test_heterodyne_basic.hpp
 * @brief Basic heterodyne dechirp tests (GPU kernel verification)
 *
 * Test 1: Single antenna dechirp (delay=100us -> f_beat=300kHz)
 * Test 2: 5 antennas, linear delays [100,200,300,400,500] us
 * Test 3: dechirp_correct verification (peak should move to DC)
 *
 * Parameters: fs=12MHz, B=2MHz, N=8000, mu=3e9 Hz/s
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "heterodyne_dechirp.hpp"
#include "heterodyne_params.hpp"
#include "processors/heterodyne_processor_opencl.hpp"
#include "generators/lfm_conjugate_generator.hpp"
#include "generators/lfm_generator_analytical_delay.hpp"
#include "params/signal_request.hpp"
#include "params/system_sampling.hpp"

#include "spectrum_maxima_finder.h"

#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include "DrvGPU/services/console_output.hpp"

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <memory>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace heterodyne { namespace tests {

// ════════════════════════════════════════════════════════════════════════════
// Test parameters
// ════════════════════════════════════════════════════════════════════════════

constexpr float  FS         = 12e6f;
constexpr float  F_START    = 0.0f;
constexpr float  F_END      = 2e6f;
constexpr int    N          = 8000;
constexpr int    ANTENNAS   = 5;
constexpr float  BANDWIDTH  = F_END - F_START;  // 2 MHz
constexpr float  DURATION   = static_cast<float>(N) / FS;  // 666.67 us
constexpr float  MU         = BANDWIDTH / DURATION;  // 3e9 Hz/s

// Linear delays [us] -> f_beat = mu * tau
// All delays must be < chirp duration T=666us
// f_beat(100us)=300kHz, f_beat(500us)=1.5MHz — all < fs/2=6MHz
const std::vector<float> DELAYS_LINEAR_US = {100.f, 200.f, 300.f, 400.f, 500.f};

// Expected beat frequencies [Hz]
// f_beat = mu * tau = 3e9 * delay_s
// 100us -> 300kHz, 200us -> 600kHz, 300us -> 900kHz, 400us -> 1.2MHz, 500us -> 1.5MHz

constexpr float F_BEAT_TOL_HZ = 5000.f;  // +/- 5 kHz tolerance (plan requirement)

// ════════════════════════════════════════════════════════════════════════════
// Helper: generate delayed LFM rx data and flatten
// ════════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateRxFlat(
    drv_gpu_lib::IBackend* backend,
    const std::vector<float>& delays_us) {

  signal_gen::LfmParams lfm_p;
  lfm_p.f_start = F_START;
  lfm_p.f_end = F_END;
  lfm_p.amplitude = 1.0;
  lfm_p.complex_iq = true;

  signal_gen::SystemSampling sys;
  sys.fs = FS;
  sys.length = N;

  signal_gen::LfmGeneratorAnalyticalDelay gen(backend, lfm_p);
  gen.SetSampling(sys);
  gen.SetDelays(delays_us);

  auto cpu_2d = gen.GenerateToCpu();

  // Flatten [antennas][samples] -> [antennas * samples]
  size_t total = delays_us.size() * N;
  std::vector<std::complex<float>> flat(total);
  for (size_t ant = 0; ant < delays_us.size(); ++ant) {
    for (int n = 0; n < N; ++n) {
      flat[ant * N + n] = cpu_2d[ant][n];
    }
  }
  return flat;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: Single antenna dechirp
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_single_antenna() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  if (!con.IsRunning()) con.Start();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 1: Single antenna dechirp (delay=100us)");

  try {
    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->Initialize(0);

    // Generate single antenna rx with delay=100us
    std::vector<float> delay = {100.f};
    auto rx_flat = GenerateRxFlat(backend.get(), delay);

    // Setup params for 1 antenna
    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = 1;

    drv_gpu_lib::HeterodyneDechirp het(backend.get());
    het.SetParams(params);
    auto result = het.Process(rx_flat);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: " + result.error_message);
      return;
    }

    float expected_f_beat = MU * 100e-6f;  // 300 kHz
    float actual_f_beat = result.antennas[0].f_beat_hz;
    float error = std::abs(actual_f_beat - expected_f_beat);

    con.Print(gpu_id, "Heterodyne", "    f_beat expected: "
        + std::to_string(expected_f_beat) + " Hz");
    con.Print(gpu_id, "Heterodyne", "    f_beat actual:   "
        + std::to_string(actual_f_beat) + " Hz");
    con.Print(gpu_id, "Heterodyne", "    error:           "
        + std::to_string(error) + " Hz");
    con.Print(gpu_id, "Heterodyne", "    SNR:             "
        + std::to_string(result.antennas[0].peak_snr_db) + " dB");

    bool passed = (error < F_BEAT_TOL_HZ);
    con.Print(gpu_id, "Heterodyne",
        passed ? "    RESULT: PASSED" : "    RESULT: FAILED");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: 5 antennas, linear delays
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_5_antennas_linear() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 2: 5 antennas, linear delays");

  try {
    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->Initialize(0);

    auto rx_flat = GenerateRxFlat(backend.get(), DELAYS_LINEAR_US);

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    drv_gpu_lib::HeterodyneDechirp het(backend.get());
    het.SetParams(params);
    auto result = het.Process(rx_flat);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: " + result.error_message);
      return;
    }

    con.Print(gpu_id, "Heterodyne",
        "    Ant | Delay us | f_beat Hz   | R m       | SNR dB");
    con.Print(gpu_id, "Heterodyne",
        "    ----|----------|-------------|-----------|-------");

    bool all_passed = true;
    for (int ant = 0; ant < ANTENNAS; ++ant) {
      float delay_us = DELAYS_LINEAR_US[ant];
      float expected_f_beat = MU * delay_us * 1e-6f;
      float actual_f_beat = result.antennas[ant].f_beat_hz;
      float error = std::abs(actual_f_beat - expected_f_beat);

      char buf[256];
      snprintf(buf, sizeof(buf),
          "    %3d | %7.0f  | %11.0f | %9.2f | %6.1f  %s",
          ant, delay_us, actual_f_beat,
          result.antennas[ant].range_m,
          result.antennas[ant].peak_snr_db,
          (error < F_BEAT_TOL_HZ) ? "OK" : "FAIL");
      con.Print(gpu_id, "Heterodyne", buf);

      if (error >= F_BEAT_TOL_HZ) all_passed = false;
    }

    con.Print(gpu_id, "Heterodyne",
        all_passed ? "    RESULT: ALL PASSED" : "    RESULT: SOME FAILED");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: dechirp_correct — verify peak moves to DC
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_correction() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 3: Dechirp correction (peak -> DC)");

  try {
    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->Initialize(0);

    auto rx_flat = GenerateRxFlat(backend.get(), DELAYS_LINEAR_US);

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    // Step 1: Generate ref
    signal_gen::LfmParams lfm_p;
    lfm_p.f_start = F_START;
    lfm_p.f_end = F_END;
    lfm_p.amplitude = 1.0;
    lfm_p.complex_iq = true;

    signal_gen::SystemSampling sys;
    sys.fs = FS;
    sys.length = N;

    signal_gen::LfmConjugateGenerator conj_gen(backend.get(), lfm_p);
    conj_gen.SetSampling(sys);
    auto ref = conj_gen.GenerateToCpu();

    // Step 2: Dechirp on GPU
    drv_gpu_lib::HeterodyneProcessorOpenCL proc(backend.get());
    auto dc_data = proc.Dechirp(rx_flat, ref, params);

    // Step 3: Process to find f_beat
    drv_gpu_lib::HeterodyneDechirp het(backend.get());
    het.SetParams(params);
    auto result = het.Process(rx_flat);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: " + result.error_message);
      return;
    }

    // Step 4: Correct using found f_beat
    std::vector<float> f_beats(ANTENNAS);
    for (int i = 0; i < ANTENNAS; ++i) {
      f_beats[i] = result.antennas[i].f_beat_hz;
    }

    auto corrected = proc.Correct(dc_data, f_beats, params);

    // Step 5: FFT corrected data via SpectrumMaximaFinder, check peak near DC
    antenna_fft::SpectrumMaximaFinder finder(backend.get());

    antenna_fft::InputData<std::vector<std::complex<float>>> smf_input;
    smf_input.antenna_count = ANTENNAS;
    smf_input.n_point = N;
    smf_input.data = corrected;
    smf_input.repeat_count = 1;
    smf_input.sample_rate = FS;

    auto spec_results = finder.Process(smf_input,
        antenna_fft::PeakSearchMode::ONE_PEAK,
        antenna_fft::DriverType::OPENCL);

    bool all_dc = true;
    for (int ant = 0; ant < ANTENNAS; ++ant) {
      uint32_t peak_bin = spec_results[ant].interpolated.index;

      con.Print(gpu_id, "Heterodyne",
          "    Ant " + std::to_string(ant) + ": peak at bin "
          + std::to_string(peak_bin) + " (expect ~0)");

      if (peak_bin > 3) all_dc = false;
    }

    con.Print(gpu_id, "Heterodyne",
        all_dc ? "    RESULT: PASSED (peaks at DC)"
               : "    RESULT: FAILED (peaks NOT at DC)");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: Random delays (seed=42), 5 antennas, delays [10..600] us
// ════════════════════════════════════════════════════════════════════════════

inline void run_test_random_delays() {
  int gpu_id = 0;
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  con.Print(gpu_id, "Heterodyne", "");
  con.Print(gpu_id, "Heterodyne", "  Test 6: Random delays (seed=42)");

  try {
    auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
    backend->Initialize(0);

    // Generate random delays in [10, 500] us with seed=42
    // All must be < chirp duration T=666.67 us, and f_beat < fs/2
    // With B=2MHz, 500us -> f_beat=1.5MHz, matching plan's target range
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(10.0f, 500.0f);
    std::vector<float> delays_us(ANTENNAS);
    for (int i = 0; i < ANTENNAS; ++i) {
      delays_us[i] = dist(rng);
    }

    auto rx_flat = GenerateRxFlat(backend.get(), delays_us);

    drv_gpu_lib::HeterodyneParams params;
    params.f_start = F_START;
    params.f_end = F_END;
    params.sample_rate = FS;
    params.num_samples = N;
    params.num_antennas = ANTENNAS;

    drv_gpu_lib::HeterodyneDechirp het(backend.get());
    het.SetParams(params);
    auto result = het.Process(rx_flat);

    if (!result.success) {
      con.Print(gpu_id, "Heterodyne", "    FAIL: " + result.error_message);
      return;
    }

    con.Print(gpu_id, "Heterodyne",
        "    Ant | Delay us | f_beat Hz   | Expected Hz | Error Hz | SNR dB");
    con.Print(gpu_id, "Heterodyne",
        "    ----|----------|-------------|-------------|----------|-------");

    bool all_passed = true;
    for (int ant = 0; ant < ANTENNAS; ++ant) {
      float expected_f = MU * delays_us[ant] * 1e-6f;
      float actual_f = result.antennas[ant].f_beat_hz;
      float f_err = std::abs(actual_f - expected_f);

      char buf[256];
      snprintf(buf, sizeof(buf),
          "    %3d | %7.1f  | %11.0f | %11.0f | %8.0f | %6.1f  %s",
          ant, delays_us[ant], actual_f, expected_f, f_err,
          result.antennas[ant].peak_snr_db,
          (f_err < F_BEAT_TOL_HZ) ? "OK" : "FAIL");
      con.Print(gpu_id, "Heterodyne", buf);

      if (f_err >= F_BEAT_TOL_HZ) all_passed = false;
    }

    con.Print(gpu_id, "Heterodyne",
        all_passed ? "    RESULT: ALL PASSED" : "    RESULT: SOME FAILED");

  } catch (const std::exception& e) {
    con.Print(gpu_id, "Heterodyne", "    EXCEPTION: " + std::string(e.what()));
  }
}

}} // namespace heterodyne::tests
