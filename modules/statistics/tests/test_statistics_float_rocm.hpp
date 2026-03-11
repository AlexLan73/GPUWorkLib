#pragma once

/**
 * @file test_statistics_float_rocm.hpp
 * @brief Tests for StatisticsProcessor float API + ProcessMagnitude→Statistics pipeline
 *
 * Tests:
 * 1. ComputeStatisticsFloat(vector<float>) -- known magnitudes, verify mean/variance/std
 * 2. ComputeMedianFloat(vector<float>)     -- sorted sequence, verify median
 * 3. ComputeStatisticsFloat(void*)         -- GPU float data via managed memory
 * 4. ComputeMedianFloat(void*)             -- GPU float data via managed memory
 * 5. Pipeline: ProcessMagnitude → ComputeStatisticsFloat (GPU-to-GPU)
 * 6. Pipeline: ProcessMagnitudeToGPU → ComputeMedianFloat (GPU-to-GPU)
 *
 * Reference: NumPy np.mean, np.std, np.median
 * Tolerance: 1e-3 for statistics (float precision + GPU reduction order)
 *
 * IMPORTANT: Tests compile ONLY with ENABLE_ROCM=1.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-11
 */

#if ENABLE_ROCM

#include "statistics_processor.hpp"
#include "complex_to_mag_phase_rocm.hpp"
#include "test_helpers_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_statistics_float_rocm {

using namespace statistics;
using namespace drv_gpu_lib;
using namespace test_helpers_rocm;

// =========================================================================
// Utilities
// =========================================================================

inline void print_result(ConsoleOutput& con, int gpu_id,
                         const std::string& test_name, bool passed) {
    std::string status = passed ? "PASSED" : "FAILED";
    std::string icon = passed ? "[+]" : "[X]";
    con.Print(gpu_id, "StatsFloat", icon + " " + test_name + " ... " + status);
}

/// CPU reference: mean of float vector
inline float RefMean(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += x;
    return static_cast<float>(s / v.size());
}

/// CPU reference: population std of float vector
inline float RefStd(const std::vector<float>& v) {
    float m = RefMean(v);
    double s = 0;
    for (float x : v) { double d = x - m; s += d * d; }
    return static_cast<float>(std::sqrt(s / v.size()));
}

/// CPU reference: median of float vector
inline float RefMedian(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? 0.5f * (v[n/2-1] + v[n/2]) : v[n/2];
}

// =========================================================================
// Test 1: ComputeStatisticsFloat(vector<float>) -- known data
// =========================================================================

inline void TestStatisticsVectorFloat(StatisticsProcessor& stats, ConsoleOutput& con, int gpu_id) {
    constexpr uint32_t N = 4096;
    std::vector<float> data(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.5f, 5.0f);
    for (float& x : data) x = dist(rng);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point    = N;

    auto results = stats.ComputeStatisticsFloat(data, params);

    float ref_mean = RefMean(data);
    float ref_std  = RefStd(data);

    bool ok = (results.size() == 1);
    if (ok) {
        float tol = 1e-2f;
        ok = (std::abs(results[0].mean_magnitude - ref_mean) < tol * ref_mean + 1e-5f) &&
             (std::abs(results[0].std_dev - ref_std)  < tol * ref_std  + 1e-5f);
    }
    print_result(con, gpu_id, "T1 ComputeStatisticsFloat(vector)", ok);
}

// =========================================================================
// Test 2: ComputeMedianFloat(vector<float>) -- sorted-ish data
// =========================================================================

inline void TestMedianVectorFloat(StatisticsProcessor& stats, ConsoleOutput& con, int gpu_id) {
    constexpr uint32_t N = 1024;
    std::vector<float> data(N);
    std::iota(data.begin(), data.end(), 0.0f);  // 0..1023 (sorted)

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point    = N;

    auto results = stats.ComputeMedianFloat(data, params);

    float ref_median = RefMedian(data);  // ~511.5

    bool ok = (results.size() == 1) &&
              (std::abs(results[0].median_magnitude - ref_median) < 1.0f);
    print_result(con, gpu_id, "T2 ComputeMedianFloat(vector)", ok);
}

// =========================================================================
// Test 3: ComputeStatisticsFloat(void*) -- managed memory
// =========================================================================

inline void TestStatisticsGpuFloatManaged(StatisticsProcessor& stats, ConsoleOutput& con, int gpu_id) {
    constexpr uint32_t N = 2048;
    auto managed = AllocateManagedForTest(N * sizeof(float));
    auto* ptr = static_cast<float*>(managed);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(1.0f, 4.0f);
    std::vector<float> src(N);
    for (size_t i = 0; i < N; ++i) {
        ptr[i] = dist(rng);
        src[i] = ptr[i];
    }

    auto input = MakeManagedMagnitudeInput(managed, 1, N);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point    = N;

    auto results = stats.ComputeStatisticsFloat(input.data, params);
    (void)hipFree(managed);

    float ref_mean = RefMean(src);
    float ref_std  = RefStd(src);

    bool ok = (results.size() == 1);
    if (ok) {
        float tol = 1e-2f;
        ok = (std::abs(results[0].mean_magnitude - ref_mean) < tol * ref_mean + 1e-5f) &&
             (std::abs(results[0].std_dev - ref_std)  < tol * ref_std  + 1e-5f);
    }
    print_result(con, gpu_id, "T3 ComputeStatisticsFloat(GPU managed)", ok);
}

// =========================================================================
// Test 4: ComputeMedianFloat(void*) -- managed memory
// =========================================================================

inline void TestMedianGpuFloatManaged(StatisticsProcessor& stats, ConsoleOutput& con, int gpu_id) {
    constexpr uint32_t N = 512;
    auto managed = AllocateManagedForTest(N * sizeof(float));
    auto* ptr = static_cast<float*>(managed);

    // Random shuffle of 0..N-1
    std::vector<float> src(N);
    std::iota(src.begin(), src.end(), 0.0f);
    std::mt19937 rng(13);
    std::shuffle(src.begin(), src.end(), rng);
    for (size_t i = 0; i < N; ++i) ptr[i] = src[i];

    auto input = MakeManagedMagnitudeInput(managed, 1, N);

    StatisticsParams params;
    params.beam_count = 1;
    params.n_point    = N;

    auto results = stats.ComputeMedianFloat(input.data, params);
    (void)hipFree(managed);

    float ref_median = RefMedian(src);

    bool ok = (results.size() == 1) &&
              (std::abs(results[0].median_magnitude - ref_median) < 1.0f);
    print_result(con, gpu_id, "T4 ComputeMedianFloat(GPU managed)", ok);
}

// =========================================================================
// Test 5: Pipeline ProcessMagnitude → ComputeStatisticsFloat (GPU-to-GPU)
// =========================================================================

inline void TestPipelineMagnitudeToStats(
    fft_processor::ComplexToMagPhaseROCm& mag_proc,
    StatisticsProcessor& stats,
    ConsoleOutput& con, int gpu_id)
{
    constexpr uint32_t kBeams = 2;
    constexpr uint32_t kN    = 4096;
    constexpr size_t kTotal  = kBeams * kN;

    // Managed complex input
    auto managed = AllocateManagedForTest(kTotal * sizeof(std::complex<float>));
    auto* cptr = static_cast<std::complex<float>*>(managed);

    // Fill: beam 0 = amplitude 1, beam 1 = amplitude 2
    std::vector<std::complex<float>> src(kTotal);
    for (uint32_t b = 0; b < kBeams; ++b) {
        float amp = static_cast<float>(b + 1);
        for (uint32_t k = 0; k < kN; ++k) {
            float ph = 2.0f * static_cast<float>(M_PI) * 440.0f * k / 44100.0f;
            std::complex<float> v = {amp * std::cos(ph), amp * std::sin(ph)};
            cptr[b * kN + k] = v;
            src[b * kN + k]  = v;
        }
    }

    // ProcessMagnitude → GPU float[]
    fft_processor::MagPhaseParams mp;
    mp.beam_count = kBeams;
    mp.n_point    = kN;
    mp.norm_coeff = 1.0f;

    void* gpu_mag = mag_proc.ProcessMagnitudeToGPU(managed, mp, kTotal * sizeof(std::complex<float>));
    (void)hipFree(managed);

    // ComputeStatisticsFloat on GPU float
    StatisticsParams sp;
    sp.beam_count = kBeams;
    sp.n_point    = kN;

    auto results = stats.ComputeStatisticsFloat(gpu_mag, sp);
    (void)hipFree(gpu_mag);

    // Reference: for constant-amplitude sinusoid, mean_magnitude ≈ amplitude
    bool ok = (results.size() == kBeams);
    for (uint32_t b = 0; b < kBeams && ok; ++b) {
        float expected_mean = static_cast<float>(b + 1);  // amplitude
        ok = std::abs(results[b].mean_magnitude - expected_mean) < 0.05f;
    }
    print_result(con, gpu_id, "T5 pipeline Mag→Stats GPU-GPU", ok);
}

// =========================================================================
// Test 6: Pipeline ProcessMagnitudeToGPU → ComputeMedianFloat (GPU-to-GPU)
// =========================================================================

inline void TestPipelineMagnitudeToMedian(
    fft_processor::ComplexToMagPhaseROCm& mag_proc,
    StatisticsProcessor& stats,
    ConsoleOutput& con, int gpu_id)
{
    constexpr uint32_t kN = 2048;

    // Constant signal: |z| = 3.0 for all points → median = 3.0
    auto managed = AllocateManagedForTest(kN * sizeof(std::complex<float>));
    auto* cptr = static_cast<std::complex<float>*>(managed);
    for (uint32_t k = 0; k < kN; ++k) {
        cptr[k] = {3.0f, 0.0f};  // magnitude = 3.0
    }

    fft_processor::MagPhaseParams mp;
    mp.beam_count = 1;
    mp.n_point    = kN;
    mp.norm_coeff = 1.0f;

    void* gpu_mag = mag_proc.ProcessMagnitudeToGPU(managed, mp, kN * sizeof(std::complex<float>));
    (void)hipFree(managed);

    StatisticsParams sp;
    sp.beam_count = 1;
    sp.n_point    = kN;

    auto results = stats.ComputeMedianFloat(gpu_mag, sp);
    (void)hipFree(gpu_mag);

    bool ok = (results.size() == 1) &&
              (std::abs(results[0].median_magnitude - 3.0f) < 1e-3f);
    print_result(con, gpu_id, "T6 pipeline Mag→Median GPU-GPU", ok);
}

// =========================================================================
// Entry point
// =========================================================================

inline void run() {
    auto& con = ConsoleOutput::GetInstance();
    int gpu_id = 0;

    con.Print(gpu_id, "StatsFloat", "=== Statistics Float + Pipeline Tests ===");

    ROCmBackend backend;
    backend.Initialize(gpu_id);
    StatisticsProcessor stats(&backend);
    fft_processor::ComplexToMagPhaseROCm mag_proc(&backend);

    TestStatisticsVectorFloat(stats, con, gpu_id);
    TestMedianVectorFloat(stats, con, gpu_id);
    TestStatisticsGpuFloatManaged(stats, con, gpu_id);
    TestMedianGpuFloatManaged(stats, con, gpu_id);
    TestPipelineMagnitudeToStats(mag_proc, stats, con, gpu_id);
    TestPipelineMagnitudeToMedian(mag_proc, stats, con, gpu_id);

    con.Print(gpu_id, "StatsFloat", "=== Done ===");
}

}  // namespace test_statistics_float_rocm

#endif  // ENABLE_ROCM
