#pragma once

/**
 * @file test_process_magnitude_rocm.hpp
 * @brief Tests for ComplexToMagPhaseROCm::ProcessMagnitude / ProcessMagnitudeToGPU
 *
 * Tests:
 * 1. ProcessMagnitude -- GPU void* input, no normalization (norm_coeff=1)
 * 2. ProcessMagnitude -- managed memory (hipMallocManaged), norm_coeff=-1 (÷n_point)
 * 3. ProcessMagnitude -- norm_coeff=0 (same as 1), edge: zero signal
 * 4. ProcessMagnitudeToGPU -- output stays on GPU, verify via hipMemcpy
 * 5. ProcessMagnitude multi-beam -- 4 beams × 4096 points (managed)
 *
 * Reference: magnitude = sqrt(re^2 + im^2), then × inv_n
 * Tolerance: 1e-4 relative (fast sqrt intrinsic precision)
 *
 * IMPORTANT: Tests compile ONLY with ENABLE_ROCM=1.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-11
 */

#if ENABLE_ROCM

#include "complex_to_mag_phase_rocm.hpp"
#include "test_helpers_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_process_magnitude_rocm {

using namespace fft_processor;
using namespace drv_gpu_lib;
using namespace test_helpers_rocm;

// =========================================================================
// Utilities
// =========================================================================

inline void print_result(ConsoleOutput& con, int gpu_id,
                         const std::string& test_name, bool passed) {
    std::string status = passed ? "PASSED" : "FAILED";
    std::string icon = passed ? "[+]" : "[X]";
    con.Print(gpu_id, "ProcMag", icon + " " + test_name + " ... " + status);
}

/// Generate complex sinusoid of given amplitude
inline std::vector<std::complex<float>> MakeSinusoid(
    size_t n, float amplitude = 1.0f, float freq_hz = 1000.0f, float sample_rate = 44100.0f)
{
    std::vector<std::complex<float>> data(n);
    for (size_t i = 0; i < n; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float ph = 2.0f * static_cast<float>(M_PI) * freq_hz * t;
        data[i] = {amplitude * std::cos(ph), amplitude * std::sin(ph)};
    }
    return data;
}

/// Compute reference magnitudes * inv_n on CPU
inline std::vector<float> RefMagnitudes(
    const std::vector<std::complex<float>>& data, float inv_n)
{
    std::vector<float> ref(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        ref[i] = std::abs(data[i]) * inv_n;
    }
    return ref;
}

inline bool AllClose(const std::vector<float>& a, const std::vector<float>& b,
                     float atol = 1e-4f)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > atol + 1e-6f * std::abs(b[i])) return false;
    }
    return true;
}

// =========================================================================
// Test 1: GPU void* input, norm_coeff=1 (no normalization)
// =========================================================================

inline void TestGpuInputNoNorm(ComplexToMagPhaseROCm& proc, ConsoleOutput& con, int gpu_id) {
    constexpr uint32_t N = 4096;
    auto src = MakeSinusoid(N, 2.5f);

    // Upload to GPU
    void* gpu_ptr = nullptr;
    (void)hipMalloc(&gpu_ptr, N * sizeof(std::complex<float>));
    (void)hipMemcpy(gpu_ptr, src.data(), N * sizeof(std::complex<float>), hipMemcpyHostToDevice);

    MagPhaseParams params;
    params.beam_count = 1;
    params.n_point    = N;
    params.norm_coeff = 1.0f;  // no normalization

    auto results = proc.ProcessMagnitude(gpu_ptr, params,
                                          N * sizeof(std::complex<float>));
    (void)hipFree(gpu_ptr);

    bool ok = (results.size() == 1 && results[0].n_point == N);
    if (ok) {
        auto ref = RefMagnitudes(src, 1.0f);
        ok = AllClose(results[0].magnitude, ref);
    }
    print_result(con, gpu_id, "T1 GPU-in norm=1", ok);
}

// =========================================================================
// Test 2: Managed memory (hipMallocManaged) + norm_coeff=-1 (÷n_point)
// =========================================================================

inline void TestManagedNormByN(ComplexToMagPhaseROCm& proc, ConsoleOutput& con, int gpu_id) {
    constexpr uint32_t N = 2048;
    auto managed = AllocateManagedForTest(N * sizeof(std::complex<float>));
    auto* ptr = static_cast<std::complex<float>*>(managed);
    auto src = MakeSinusoid(N, 3.0f);
    for (size_t i = 0; i < N; ++i) ptr[i] = src[i];

    auto input = MakeManagedInput(managed, 1, N);

    MagPhaseParams params;
    params.beam_count = 1;
    params.n_point    = N;
    params.norm_coeff = -1.0f;  // inv_n = 1/N

    auto results = proc.ProcessMagnitude(input.data, params, input.gpu_memory_bytes);
    (void)hipFree(managed);

    float inv_n = 1.0f / static_cast<float>(N);
    bool ok = (results.size() == 1);
    if (ok) {
        auto ref = RefMagnitudes(src, inv_n);
        ok = AllClose(results[0].magnitude, ref);
    }
    print_result(con, gpu_id, "T2 managed norm=-1 (÷N)", ok);
}

// =========================================================================
// Test 3: norm_coeff=0 (skip → inv_n=1), zero signal
// =========================================================================

inline void TestNormZeroAndZeroSignal(ComplexToMagPhaseROCm& proc, ConsoleOutput& con, int gpu_id) {
    constexpr uint32_t N = 512;
    // Zero signal
    std::vector<std::complex<float>> zeros(N, {0.0f, 0.0f});

    void* gpu_ptr = nullptr;
    (void)hipMalloc(&gpu_ptr, N * sizeof(std::complex<float>));
    (void)hipMemcpy(gpu_ptr, zeros.data(), N * sizeof(std::complex<float>), hipMemcpyHostToDevice);

    MagPhaseParams params;
    params.beam_count = 1;
    params.n_point    = N;
    params.norm_coeff = 0.0f;  // skip → inv_n=1

    auto results = proc.ProcessMagnitude(gpu_ptr, params);
    (void)hipFree(gpu_ptr);

    bool ok = (results.size() == 1);
    if (ok) {
        float sum = 0.0f;
        for (float v : results[0].magnitude) sum += std::abs(v);
        ok = (sum < 1e-6f);  // all zeros
    }
    print_result(con, gpu_id, "T3 norm=0 zero-signal", ok);
}

// =========================================================================
// Test 4: ProcessMagnitudeToGPU — stays on GPU, verify via hipMemcpy
// =========================================================================

inline void TestProcessMagnitudeToGPU(ComplexToMagPhaseROCm& proc, ConsoleOutput& con, int gpu_id) {
    constexpr uint32_t N = 1024;
    auto src = MakeSinusoid(N, 1.0f);

    void* gpu_in = nullptr;
    (void)hipMalloc(&gpu_in, N * sizeof(std::complex<float>));
    (void)hipMemcpy(gpu_in, src.data(), N * sizeof(std::complex<float>), hipMemcpyHostToDevice);

    MagPhaseParams params;
    params.beam_count = 1;
    params.n_point    = N;
    params.norm_coeff = 1.0f;

    void* gpu_out = proc.ProcessMagnitudeToGPU(gpu_in, params);  // CALLER OWNS
    (void)hipFree(gpu_in);

    // Read back
    std::vector<float> result(N);
    (void)hipMemcpy(result.data(), gpu_out, N * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipFree(gpu_out);

    auto ref = RefMagnitudes(src, 1.0f);
    bool ok = AllClose(result, ref);
    print_result(con, gpu_id, "T4 ProcessMagnitudeToGPU", ok);
}

// =========================================================================
// Test 5: Multi-beam managed (4 beams × 4096 points)
// =========================================================================

inline void TestMultiBeamManaged(ComplexToMagPhaseROCm& proc, ConsoleOutput& con, int gpu_id) {
    constexpr uint32_t kBeams = 4;
    constexpr uint32_t kN    = 4096;
    constexpr size_t kTotal  = kBeams * kN;

    auto managed = AllocateManagedForTest(kTotal * sizeof(std::complex<float>));
    auto* ptr = static_cast<std::complex<float>*>(managed);

    // Fill each beam with different amplitude
    std::vector<std::complex<float>> src(kTotal);
    for (uint32_t b = 0; b < kBeams; ++b) {
        float amp = static_cast<float>(b + 1) * 0.5f;
        auto beam = MakeSinusoid(kN, amp);
        for (uint32_t k = 0; k < kN; ++k) {
            ptr[b * kN + k] = beam[k];
            src[b * kN + k] = beam[k];
        }
    }

    auto input = MakeManagedInput(managed, kBeams, kN);

    MagPhaseParams params;
    params.beam_count = kBeams;
    params.n_point    = kN;
    params.norm_coeff = -1.0f;  // ÷n_point

    auto results = proc.ProcessMagnitude(input.data, params, input.gpu_memory_bytes);
    (void)hipFree(managed);

    bool ok = (results.size() == kBeams);
    float inv_n = 1.0f / static_cast<float>(kN);
    for (uint32_t b = 0; b < kBeams && ok; ++b) {
        ok = (results[b].beam_id == b && results[b].n_point == kN);
        if (ok) {
            std::vector<std::complex<float>> beam_src(src.begin() + b * kN,
                                                       src.begin() + (b + 1) * kN);
            auto ref = RefMagnitudes(beam_src, inv_n);
            ok = AllClose(results[b].magnitude, ref);
        }
    }
    print_result(con, gpu_id, "T5 multi-beam 4×4096 managed", ok);
}

// =========================================================================
// Test 6: ProcessMagnitudeToBuffer — zero allocations, caller's buffer
//         Compare result with ProcessMagnitudeToGPU + hipMemcpy
// =========================================================================

inline void TestProcessMagnitudeToBuffer(ComplexToMagPhaseROCm& proc, ConsoleOutput& con, int gpu_id) {
    constexpr uint32_t kBeams = 2;
    constexpr uint32_t N = 2048;
    constexpr size_t kTotal = static_cast<size_t>(kBeams) * N;

    auto src = MakeSinusoid(kTotal, 1.5f);

    // Input on GPU
    void* gpu_in = nullptr;
    (void)hipMalloc(&gpu_in, kTotal * sizeof(std::complex<float>));
    (void)hipMemcpy(gpu_in, src.data(), kTotal * sizeof(std::complex<float>), hipMemcpyHostToDevice);

    // Pre-allocated output buffer (caller owns, no alloc inside ProcessMagnitudeToBuffer)
    void* gpu_out = nullptr;
    (void)hipMalloc(&gpu_out, kTotal * sizeof(float));

    MagPhaseParams params;
    params.beam_count = kBeams;
    params.n_point    = N;
    params.norm_coeff = 0.0f;  // inv_n = 1

    // Call ProcessMagnitudeToBuffer (zero allocations)
    proc.ProcessMagnitudeToBuffer(gpu_in, gpu_out, params);

    // Reference: ProcessMagnitudeToGPU (allocates internally, then hipFree)
    void* gpu_ref = proc.ProcessMagnitudeToGPU(gpu_in, params);

    // Compare both GPU outputs
    std::vector<float> result_buf(kTotal);
    std::vector<float> result_ref(kTotal);
    (void)hipMemcpy(result_buf.data(), gpu_out, kTotal * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipMemcpy(result_ref.data(), gpu_ref, kTotal * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(gpu_in);
    (void)hipFree(gpu_out);
    (void)hipFree(gpu_ref);

    bool ok = AllClose(result_buf, result_ref);
    print_result(con, gpu_id, "T6 ProcessMagnitudeToBuffer 2×2048", ok);
}

// =========================================================================
// Entry point
// =========================================================================

inline void run() {
    auto& con = ConsoleOutput::GetInstance();
    int gpu_id = 0;

    con.Print(gpu_id, "ProcMag", "=== ProcessMagnitude ROCm Tests ===");

    ROCmBackend backend;
    backend.Initialize(gpu_id);
    ComplexToMagPhaseROCm proc(&backend);

    TestGpuInputNoNorm(proc, con, gpu_id);
    TestManagedNormByN(proc, con, gpu_id);
    TestNormZeroAndZeroSignal(proc, con, gpu_id);
    TestProcessMagnitudeToGPU(proc, con, gpu_id);
    TestMultiBeamManaged(proc, con, gpu_id);
    TestProcessMagnitudeToBuffer(proc, con, gpu_id);

    con.Print(gpu_id, "ProcMag", "=== Done ===");
}

}  // namespace test_process_magnitude_rocm

#endif  // ENABLE_ROCM
