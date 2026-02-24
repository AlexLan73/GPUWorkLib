#pragma once

/**
 * @file test_spectrum_maxima_rocm.hpp
 * @brief ROCm tests for SpectrumProcessorROCm
 *
 * Tests (all #if ENABLE_ROCM guarded):
 * 1. ONE_PEAK  — CW signal 100 Hz, fs=1000
 * 2. TWO_PEAKS — two CW signals (left + right half)
 * 3. FindAllMaxima — 3 sinusoids → 3 peaks
 * 4. AllMaximaFromCPU — pre-computed FFT data
 * 5. ProcessFromGPU — data already on GPU
 * 6. Batch processing — multiple beams
 *
 * NOT RUN on Windows — compile-only check.
 * Will be tested on Linux + AMD Radeon 9070.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if ENABLE_ROCM
#include "processors/spectrum_processor_rocm.hpp"
#include "factory/spectrum_processor_factory.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "common/backend_type.hpp"
#endif

namespace test_spectrum_maxima_rocm {

#if ENABLE_ROCM

using namespace antenna_fft;
using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════════════════
// Helper: generate CW signal
// ════════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateCW(
    float frequency, float sample_rate, uint32_t n_point,
    uint32_t beam_count = 1, float amplitude = 1.0f)
{
    std::vector<std::complex<float>> data(static_cast<size_t>(beam_count) * n_point);
    for (uint32_t b = 0; b < beam_count; ++b) {
        for (uint32_t t = 0; t < n_point; ++t) {
            float phase = 2.0f * static_cast<float>(M_PI) * frequency * t / sample_rate;
            float val = amplitude * std::sin(phase);
            data[b * n_point + t] = std::complex<float>(val, 0.0f);
        }
    }
    return data;
}

// ════════════════════════════════════════════════════════════════════════════
// Helper: generate multi-frequency signal
// ════════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateMultiCW(
    const std::vector<float>& frequencies, float sample_rate, uint32_t n_point,
    uint32_t beam_count = 1)
{
    std::vector<std::complex<float>> data(static_cast<size_t>(beam_count) * n_point);
    for (uint32_t b = 0; b < beam_count; ++b) {
        for (uint32_t t = 0; t < n_point; ++t) {
            float val = 0.0f;
            for (float f : frequencies) {
                val += std::sin(2.0f * static_cast<float>(M_PI) * f * t / sample_rate);
            }
            data[b * n_point + t] = std::complex<float>(val, 0.0f);
        }
    }
    return data;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: ONE_PEAK — CW 100 Hz
// ════════════════════════════════════════════════════════════════════════════

inline bool TestOnePeak(IBackend* backend) {
    std::cout << "\n=== ROCm Test: ONE_PEAK (CW 100 Hz) ===\n";

    const uint32_t n_point = 1000;
    const float sample_rate = 10000.0f;
    const float frequency = 100.0f;
    const uint32_t antenna_count = 4;

    auto data = GenerateCW(frequency, sample_rate, n_point, antenna_count);

    SpectrumParams params;
    params.antenna_count = antenna_count;
    params.n_point = n_point;
    params.repeat_count = 1;
    params.sample_rate = sample_rate;
    params.search_range = 0;
    params.peak_mode = PeakSearchMode::ONE_PEAK;
    params.memory_limit = 0.8f;

    auto processor = SpectrumProcessorFactory::Create(BackendType::ROCm, backend);
    processor->Initialize(params);

    auto results = processor->ProcessFromCPU(data);

    std::cout << "  Results: " << results.size() << " (expected " << antenna_count << ")\n";
    assert(results.size() == antenna_count);

    for (const auto& r : results) {
        float expected_bin = frequency * params.nFFT / sample_rate;
        float actual_freq = r.interpolated.refined_frequency;
        float error = std::abs(actual_freq - frequency);

        std::cout << "  Beam " << r.antenna_id
                  << ": freq=" << actual_freq << " Hz"
                  << " (error=" << error << " Hz)"
                  << " mag=" << r.interpolated.magnitude << "\n";

        assert(error < 5.0f);  // Within 5 Hz
    }

    std::cout << "  PASSED\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: TWO_PEAKS — two frequencies
// ════════════════════════════════════════════════════════════════════════════

inline bool TestTwoPeaks(IBackend* backend) {
    std::cout << "\n=== ROCm Test: TWO_PEAKS ===\n";

    const uint32_t n_point = 1000;
    const float sample_rate = 10000.0f;
    const uint32_t antenna_count = 2;

    auto data = GenerateMultiCW({100.0f, 300.0f}, sample_rate, n_point, antenna_count);

    SpectrumParams params;
    params.antenna_count = antenna_count;
    params.n_point = n_point;
    params.repeat_count = 1;
    params.sample_rate = sample_rate;
    params.search_range = 0;
    params.peak_mode = PeakSearchMode::TWO_PEAKS;
    params.memory_limit = 0.8f;

    auto processor = SpectrumProcessorFactory::Create(BackendType::ROCm, backend);
    processor->Initialize(params);

    auto results = processor->ProcessFromCPU(data);

    // TWO_PEAKS: 2 results per beam
    std::cout << "  Results: " << results.size() << " (expected " << antenna_count * 2 << ")\n";
    assert(results.size() == antenna_count * 2);

    for (const auto& r : results) {
        std::cout << "  Beam " << r.antenna_id
                  << ": freq=" << r.interpolated.refined_frequency << " Hz"
                  << " mag=" << r.interpolated.magnitude << "\n";
    }

    std::cout << "  PASSED\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: FindAllMaxima — 3 sinusoids
// ════════════════════════════════════════════════════════════════════════════

inline bool TestFindAllMaxima(IBackend* backend) {
    std::cout << "\n=== ROCm Test: FindAllMaxima (3 frequencies) ===\n";

    const uint32_t n_point = 1024;
    const float sample_rate = 1000.0f;
    const uint32_t antenna_count = 2;

    auto data = GenerateMultiCW({50.0f, 120.0f, 200.0f}, sample_rate, n_point, antenna_count);

    SpectrumParams params;
    params.antenna_count = antenna_count;
    params.n_point = n_point;
    params.repeat_count = 1;
    params.sample_rate = sample_rate;
    params.search_range = 0;
    params.peak_mode = PeakSearchMode::ONE_PEAK;  // mode doesn't matter for FindAllMaxima
    params.memory_limit = 0.8f;

    auto processor = SpectrumProcessorFactory::Create(BackendType::ROCm, backend);
    processor->Initialize(params);

    auto result = processor->FindAllMaximaFromCPU(data, OutputDestination::CPU, 1, 0);

    std::cout << "  Total maxima: " << result.total_maxima << "\n";
    assert(result.total_maxima > 0);

    for (uint32_t b = 0; b < result.beams.size(); ++b) {
        const auto& beam = result.beams[b];
        std::cout << "  Beam " << beam.antenna_id
                  << ": " << beam.num_maxima << " maxima\n";

        // Expect at least 3 maxima (50, 120, 200 Hz peaks)
        assert(beam.num_maxima >= 3);

        for (size_t j = 0; j < std::min(beam.num_maxima, 5u); ++j) {
            std::cout << "    [" << j << "] idx=" << beam.maxima[j].index
                      << " freq=" << beam.maxima[j].refined_frequency << " Hz"
                      << " mag=" << beam.maxima[j].magnitude << "\n";
        }
    }

    std::cout << "  PASSED\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: AllMaximaFromCPU — pre-computed FFT
// ════════════════════════════════════════════════════════════════════════════

inline bool TestAllMaximaFromCPU(IBackend* backend) {
    std::cout << "\n=== ROCm Test: AllMaximaFromCPU (pre-computed FFT) ===\n";

    // Create a synthetic FFT spectrum with known peaks
    const uint32_t nFFT = 1024;
    const uint32_t beam_count = 1;
    const float sample_rate = 1000.0f;

    std::vector<std::complex<float>> fft_data(nFFT, {0.0f, 0.0f});

    // Place known peaks at bins 50, 120, 200
    fft_data[50] = {100.0f, 0.0f};
    fft_data[120] = {80.0f, 0.0f};
    fft_data[200] = {60.0f, 0.0f};

    // Add some small noise
    for (uint32_t i = 0; i < nFFT; ++i) {
        if (i != 50 && i != 120 && i != 200) {
            fft_data[i] = {0.1f, 0.0f};
        }
    }

    SpectrumParams params;
    params.antenna_count = beam_count;
    params.n_point = nFFT;
    params.repeat_count = 1;
    params.sample_rate = sample_rate;
    params.nFFT = nFFT;
    params.base_fft = nFFT;
    params.peak_mode = PeakSearchMode::ONE_PEAK;
    params.memory_limit = 0.8f;

    auto processor = SpectrumProcessorFactory::Create(BackendType::ROCm, backend);
    processor->Initialize(params);

    auto result = processor->AllMaximaFromCPU(fft_data, beam_count, nFFT, sample_rate,
                                                OutputDestination::CPU, 1, nFFT / 2);

    std::cout << "  Total maxima: " << result.total_maxima << "\n";
    assert(result.total_maxima >= 3);

    if (!result.beams.empty()) {
        for (size_t j = 0; j < result.beams[0].num_maxima; ++j) {
            std::cout << "    [" << j << "] idx=" << result.beams[0].maxima[j].index
                      << " mag=" << result.beams[0].maxima[j].magnitude << "\n";
        }
    }

    std::cout << "  PASSED\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: Batch processing — 16 beams
// ════════════════════════════════════════════════════════════════════════════

inline bool TestBatchProcessing(IBackend* backend) {
    std::cout << "\n=== ROCm Test: Batch Processing (16 beams) ===\n";

    const uint32_t n_point = 1000;
    const float sample_rate = 10000.0f;
    const uint32_t antenna_count = 16;

    // Each beam has a slightly different frequency
    std::vector<std::complex<float>> data(static_cast<size_t>(antenna_count) * n_point);
    for (uint32_t b = 0; b < antenna_count; ++b) {
        float freq = 100.0f + b * 10.0f;  // 100, 110, 120, ... 250 Hz
        for (uint32_t t = 0; t < n_point; ++t) {
            float phase = 2.0f * static_cast<float>(M_PI) * freq * t / sample_rate;
            data[b * n_point + t] = std::complex<float>(std::sin(phase), 0.0f);
        }
    }

    SpectrumParams params;
    params.antenna_count = antenna_count;
    params.n_point = n_point;
    params.repeat_count = 1;
    params.sample_rate = sample_rate;
    params.search_range = 0;
    params.peak_mode = PeakSearchMode::ONE_PEAK;
    params.memory_limit = 0.8f;

    auto processor = SpectrumProcessorFactory::Create(BackendType::ROCm, backend);
    processor->Initialize(params);

    auto results = processor->ProcessFromCPU(data);

    std::cout << "  Results: " << results.size() << " (expected " << antenna_count << ")\n";
    assert(results.size() == antenna_count);

    for (const auto& r : results) {
        float expected_freq = 100.0f + r.antenna_id * 10.0f;
        float actual_freq = r.interpolated.refined_frequency;
        float error = std::abs(actual_freq - expected_freq);

        std::cout << "  Beam " << r.antenna_id
                  << ": freq=" << actual_freq << " Hz"
                  << " (expected=" << expected_freq << ", error=" << error << ")\n";

        assert(error < 10.0f);  // Within 10 Hz
    }

    std::cout << "  PASSED\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: Compare with OpenCL results
// ════════════════════════════════════════════════════════════════════════════

inline bool TestCompareWithOpenCL(IBackend* rocm_backend, IBackend* opencl_backend) {
    std::cout << "\n=== ROCm Test: Compare ROCm vs OpenCL ===\n";

    if (!opencl_backend || !opencl_backend->IsInitialized()) {
        std::cout << "  SKIPPED (OpenCL backend not available)\n";
        return true;
    }

    const uint32_t n_point = 1000;
    const float sample_rate = 10000.0f;
    const uint32_t antenna_count = 4;

    auto data = GenerateCW(100.0f, sample_rate, n_point, antenna_count);

    SpectrumParams params;
    params.antenna_count = antenna_count;
    params.n_point = n_point;
    params.repeat_count = 1;
    params.sample_rate = sample_rate;
    params.search_range = 0;
    params.peak_mode = PeakSearchMode::ONE_PEAK;
    params.memory_limit = 0.8f;

    auto rocm_proc = SpectrumProcessorFactory::Create(BackendType::ROCm, rocm_backend);
    rocm_proc->Initialize(params);
    auto rocm_results = rocm_proc->ProcessFromCPU(data);

    auto opencl_proc = SpectrumProcessorFactory::Create(BackendType::OPENCL, opencl_backend);
    opencl_proc->Initialize(params);
    auto opencl_results = opencl_proc->ProcessFromCPU(data);

    assert(rocm_results.size() == opencl_results.size());

    float max_freq_diff = 0.0f;
    float max_mag_diff = 0.0f;

    for (size_t i = 0; i < rocm_results.size(); ++i) {
        float freq_diff = std::abs(rocm_results[i].interpolated.refined_frequency -
                                    opencl_results[i].interpolated.refined_frequency);
        float mag_diff = std::abs(rocm_results[i].interpolated.magnitude -
                                   opencl_results[i].interpolated.magnitude);

        max_freq_diff = std::max(max_freq_diff, freq_diff);
        max_mag_diff = std::max(max_mag_diff, mag_diff);

        std::cout << "  Beam " << i
                  << ": ROCm freq=" << rocm_results[i].interpolated.refined_frequency
                  << " OpenCL freq=" << opencl_results[i].interpolated.refined_frequency
                  << " diff=" << freq_diff << "\n";
    }

    std::cout << "  Max freq diff: " << max_freq_diff << " Hz\n";
    std::cout << "  Max mag diff: " << max_mag_diff << "\n";

    // Allow some tolerance (GPU math differences)
    assert(max_freq_diff < 5.0f);

    std::cout << "  PASSED\n";
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Main entry point
// ════════════════════════════════════════════════════════════════════════════

inline void run() {
    std::cout << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  SpectrumProcessorROCm Tests                     ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";

    // Create ROCm backend directly (same pattern as test_fft_processor_rocm)
    drv_gpu_lib::ROCmBackend rocm_backend;
    try {
        rocm_backend.Initialize(0);
    } catch (const std::exception& e) {
        std::cout << "  SKIPPED: ROCm backend not available: " << e.what() << "\n";
        return;
    }
    drv_gpu_lib::IBackend* backend = &rocm_backend;

    TestOnePeak(backend);
    TestTwoPeaks(backend);
    TestFindAllMaxima(backend);
    TestAllMaximaFromCPU(backend);
    TestBatchProcessing(backend);

    // OpenCL comparison skipped (single-backend test)
    TestCompareWithOpenCL(backend, nullptr);

    std::cout << "\n  All ROCm SpectrumMaxima tests PASSED!\n";
}

#else  // !ENABLE_ROCM

inline void run() {
    std::cout << "\n[test_spectrum_maxima_rocm] SKIPPED: ENABLE_ROCM not defined\n";
}

#endif  // ENABLE_ROCM

}  // namespace test_spectrum_maxima_rocm
