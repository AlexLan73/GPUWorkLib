#pragma once

/**
 * @file test_signal_generators_rocm_basic.hpp
 * @brief ROCm тесты: CW, LFM, Noise, LfmConjugate генераторы (GPU vs CPU)
 *
 * Переписано с OpenCL test_signal_generators.hpp на ROCm.
 * Паттерн: GPU generate → hipMemcpyDtoH → сравнить с CPU reference.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22 (rewrite from OpenCL → ROCm)
 */

#if ENABLE_ROCM

#include "generators/cw_generator_rocm.hpp"
#include "generators/lfm_generator_rocm.hpp"
#include "generators/noise_generator_rocm.hpp"
#include "generators/lfm_conjugate_generator_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_signal_generators_rocm_basic {

// ════════════════════════════════════════════════════════════════════════════
// Utilities
// ════════════════════════════════════════════════════════════════════════════

inline float MaxError(const std::complex<float>* a, const std::complex<float>* b, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::abs(a[i] - b[i]);
        if (d > max_err) max_err = d;
    }
    return max_err;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: CW GPU vs CPU
// ════════════════════════════════════════════════════════════════════════════

inline bool TestCwGpuVsCpu(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen ROCm 1] CW: GPU vs CPU...\n";

    signal_gen::CwParams cw;
    cw.f0 = 250.0;
    cw.amplitude = 1.5;
    cw.phase = 0.3;

    signal_gen::SystemSampling sys{4000.0, 4096};

    signal_gen::CwGeneratorROCm gen(backend);
    auto cpu_data = gen.GenerateToCpu(sys, cw, 1);

    auto gpu_result = gen.GenerateToGpu(sys, cw, 1);
    std::vector<std::complex<float>> gpu_data(sys.length);
    hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                    sys.length * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    float max_err = MaxError(gpu_data.data(), cpu_data.data(), sys.length);
    bool pass = max_err < 1e-3f;

    std::cout << "    Max error: " << std::scientific << std::setprecision(2) << max_err << "\n";
    std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: CW multi-beam with freq_step
// ════════════════════════════════════════════════════════════════════════════

inline bool TestCwMultiBeam(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen ROCm 2] CW multi-beam (8 beams, freq_step)...\n";

    signal_gen::CwParams cw;
    cw.f0 = 100.0;
    cw.freq_step = 50.0;
    cw.amplitude = 1.0;

    signal_gen::SystemSampling sys{2000.0, 2048};
    uint32_t beam_count = 8;

    signal_gen::CwGeneratorROCm gen(backend);
    auto gpu_result = gen.GenerateToGpu(sys, cw, beam_count);

    size_t total = static_cast<size_t>(beam_count) * sys.length;
    std::vector<std::complex<float>> gpu_data(total);
    hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                    total * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    bool all_pass = true;
    for (uint32_t b = 0; b < beam_count; ++b) {
        float expected_freq = static_cast<float>(cw.f0 + b * cw.freq_step);
        float fs = static_cast<float>(sys.fs);

        std::vector<std::complex<float>> cpu_beam(sys.length);
        for (size_t i = 0; i < sys.length; ++i) {
            float t = static_cast<float>(i) / fs;
            float phase = 2.0f * static_cast<float>(M_PI) * expected_freq * t
                        + static_cast<float>(cw.phase);
            cpu_beam[i] = {cw.amplitude * std::cos(phase), cw.amplitude * std::sin(phase)};
        }

        float err = MaxError(gpu_data.data() + b * sys.length, cpu_beam.data(), sys.length);
        bool pass = err < 1e-3f;
        if (!pass) all_pass = false;

        std::cout << "    Beam " << b << " (f=" << std::setprecision(0)
                  << expected_freq << " Hz): err=" << std::scientific
                  << std::setprecision(2) << err << std::fixed
                  << " " << (pass ? "OK" : "FAIL") << "\n";
    }

    std::cout << "    " << (all_pass ? "PASS" : "FAIL") << "\n";
    return all_pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: LFM GPU vs CPU
// ════════════════════════════════════════════════════════════════════════════

inline bool TestLfmGpuVsCpu(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen ROCm 3] LFM: GPU vs CPU...\n";

    signal_gen::LfmParams lfm;
    lfm.f_start = 100.0;
    lfm.f_end = 500.0;

    signal_gen::SystemSampling sys{4000.0, 4096};

    signal_gen::LfmGeneratorROCm gen(backend);
    auto cpu_data = gen.GenerateToCpu(sys, lfm, 1);

    auto gpu_result = gen.GenerateToGpu(sys, lfm, 1);
    std::vector<std::complex<float>> gpu_data(sys.length);
    hipMemcpyDtoH(gpu_data.data(), gpu_result.data,
                    sys.length * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    float max_err = MaxError(gpu_data.data(), cpu_data.data(), sys.length);
    bool pass = max_err < 1e-3f;

    std::cout << "    Max error: " << std::scientific << std::setprecision(2) << max_err << "\n";
    std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: Noise statistical test
// ════════════════════════════════════════════════════════════════════════════

inline bool TestNoiseStatistics(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen ROCm 4] Noise: statistical test...\n";

    signal_gen::NoiseParams np;
    np.power = 1.0;
    np.seed = 42;

    signal_gen::SystemSampling sys{4000.0, 8192};

    signal_gen::NoiseGeneratorROCm gen(backend);
    auto gpu_result = gen.GenerateToGpu(sys, np, 1);

    std::vector<std::complex<float>> data(sys.length);
    hipMemcpyDtoH(data.data(), gpu_result.data,
                    sys.length * sizeof(std::complex<float>));
    hipFree(gpu_result.data);

    // Compute mean and variance
    double sum_re = 0, sum_im = 0, sum_sq = 0;
    for (size_t i = 0; i < sys.length; ++i) {
        sum_re += data[i].real();
        sum_im += data[i].imag();
        sum_sq += std::norm(data[i]);
    }
    double mean_re = sum_re / sys.length;
    double mean_im = sum_im / sys.length;
    double mean_power = sum_sq / sys.length;

    // Mean should be close to 0, power close to np.power
    bool mean_ok = std::abs(mean_re) < 0.1 && std::abs(mean_im) < 0.1;
    bool power_ok = std::abs(mean_power - np.power) < 0.3 * np.power;  // 30% tolerance

    std::cout << "    Mean: (" << std::setprecision(4) << mean_re << ", " << mean_im << ")\n";
    std::cout << "    Power: " << std::setprecision(4) << mean_power
              << " (expected " << np.power << ")\n";
    std::cout << "    " << (mean_ok && power_ok ? "PASS" : "FAIL") << "\n";

    return mean_ok && power_ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: LfmConjugate GPU vs CPU
// ════════════════════════════════════════════════════════════════════════════

inline bool TestLfmConjugateGpuVsCpu(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen ROCm 5] LfmConjugate: GPU vs CPU...\n";

    signal_gen::LfmParams lfm;
    lfm.f_start = 100.0;
    lfm.f_end = 500.0;

    signal_gen::SystemSampling sys{4000.0, 4096};

    signal_gen::LfmConjugateGeneratorROCm gen(backend, lfm);
    gen.SetSampling(sys);

    auto cpu_data = gen.GenerateToCpu();
    void* gpu_ptr = gen.GenerateToGpu();

    std::vector<std::complex<float>> gpu_data(sys.length);
    hipMemcpyDtoH(gpu_data.data(), gpu_ptr,
                    sys.length * sizeof(std::complex<float>));
    hipFree(gpu_ptr);

    float max_err = MaxError(gpu_data.data(), cpu_data.data(), sys.length);
    bool pass = max_err < 1e-3f;

    // Also verify conjugate property: phase should be negative
    float t1 = 1.0f / static_cast<float>(sys.fs);
    float phase1 = std::arg(gpu_data[1]);
    bool neg_phase = phase1 < 0.0f;  // conjugate → negative phase

    std::cout << "    Max error: " << std::scientific << std::setprecision(2) << max_err << "\n";
    std::cout << std::fixed << "    Phase[1] = " << phase1 << " (should be < 0)\n";
    std::cout << "    " << (pass && neg_phase ? "PASS" : "FAIL") << "\n";

    return pass && neg_phase;
}

// ════════════════════════════════════════════════════════════════════════════
// Runner
// ════════════════════════════════════════════════════════════════════════════

inline void run() {
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    if (!con.IsRunning()) con.Start();

    con.Print(0, "SigGen[ROCm]", "════════════════════════════════════════════");
    con.Print(0, "SigGen[ROCm]", " Signal Generators ROCm Basic Tests");
    con.Print(0, "SigGen[ROCm]", "════════════════════════════════════════════");

    drv_gpu_lib::ROCmBackend backend;
    backend.Initialize(0);

    int passed = 0, total = 0;

    auto check = [&](bool ok) { total++; if (ok) passed++; };

    check(TestCwGpuVsCpu(&backend));
    check(TestCwMultiBeam(&backend));
    check(TestLfmGpuVsCpu(&backend));
    check(TestNoiseStatistics(&backend));
    check(TestLfmConjugateGpuVsCpu(&backend));

    std::cout << "\n  Signal Generators ROCm: " << passed << "/" << total << " passed\n";
    con.Print(0, "SigGen[ROCm]", "════════════════════════════════════════════");
}

}  // namespace test_signal_generators_rocm_basic

#endif  // ENABLE_ROCM
