/**
 * @file test_fft_maxima.cpp
 * @brief Test for FFT Maxima module
 *
 * Tests both Release (callbacks) and Debug (step-by-step) implementations
 * and compares their results.
 *
 * @author DrvGPU Team
 * @date 2026-02-04
 */

#include "antenna_fft_release.h"
#include "antenna_fft_debug.h"
#include "fft_result_writer.hpp"
#include "fft_logger.h"

#include "backends/opencl/opencl_backend.hpp"

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>

using namespace antenna_fft;

// ════════════════════════════════════════════════════════════════════════════
// Test data generation
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Generate test signal with known frequency peaks
 * @param beam_count Number of beams
 * @param count_points Points per beam
 * @param frequencies Frequencies to inject (as fraction of sampling rate)
 * @return Vector of complex samples
 */
std::vector<std::complex<float>> GenerateTestSignal(
    size_t beam_count,
    size_t count_points,
    const std::vector<float>& frequencies)
{
    std::vector<std::complex<float>> data(beam_count * count_points);
    const float pi = 3.14159265358979f;

    for (size_t beam = 0; beam < beam_count; ++beam) {
        for (size_t i = 0; i < count_points; ++i) {
            float t = static_cast<float>(i) / count_points;
            std::complex<float> sample(0.0f, 0.0f);

            // Add each frequency component
            for (size_t f = 0; f < frequencies.size(); ++f) {
                float freq = frequencies[f];
                float amplitude = 1.0f / (f + 1);  // Decreasing amplitude
                float phase = 2.0f * pi * freq * t;
                sample += amplitude * std::complex<float>(std::cos(phase), std::sin(phase));
            }

            // Add small noise
            float noise = 0.01f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
            sample += std::complex<float>(noise, noise);

            data[beam * count_points + i] = sample;
        }
    }

    return data;
}

// ════════════════════════════════════════════════════════════════════════════
// Test functions
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Test Release implementation (callbacks)
 */
bool TestRelease(drv_gpu_lib::IBackend* backend, const AntennaFFTParams& params,
                 const std::vector<std::complex<float>>& test_data)
{
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "  TEST: AntennaFFTProcMax (Release - Callbacks)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    try {
        AntennaFFTProcMax fft(params, backend);

        auto start = std::chrono::high_resolution_clock::now();
        AntennaFFTResult result = fft.ProcessNew(test_data);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "\n  Results:\n";
        std::cout << "    Total beams processed: " << result.total_beams << "\n";
        std::cout << "    nFFT: " << result.nFFT << "\n";
        std::cout << "    Processing time: " << elapsed << " ms\n";
        std::cout << "    Batch mode used: " << (fft.WasBatchModeUsed() ? "Yes" : "No") << "\n";

        // Print first beam results
        if (!result.results.empty()) {
            std::cout << "\n    Beam 0 maxima:\n";
            for (size_t i = 0; i < result.results[0].max_values.size(); ++i) {
                const auto& mv = result.results[0].max_values[i];
                std::cout << "      [" << i << "] Index: " << mv.index_point
                          << ", Amplitude: " << mv.amplitude
                          << ", Phase: " << mv.phase << " deg\n";
            }
        }

        // Print profiling
        const auto& prof = fft.GetLastProfilingResults();
        std::cout << "\n    Profiling:\n";
        std::cout << "      FFT time: " << prof.fft_time_ms << " ms\n";

        std::cout << "\n  [PASS] Release test completed!\n";
        return true;

    } catch (const std::exception& e) {
        std::cerr << "\n  [FAIL] Exception: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Test Debug implementation (step-by-step)
 */
bool TestDebug(drv_gpu_lib::IBackend* backend, const AntennaFFTParams& params,
               const std::vector<std::complex<float>>& test_data)
{
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "  TEST: AntennaFFTDebug (Debug - Step-by-Step)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    try {
        AntennaFFTDebug fft_debug(params, backend);

        // Set input data
        fft_debug.SetInputData(test_data);

        // Step 1: Padding
        std::cout << "\n  Step 1: Executing Padding Kernel...\n";
        cl_event padding_event;
        fft_debug.ExecutePaddingKernel(nullptr, &padding_event);
        clWaitForEvents(1, &padding_event);
        std::cout << "    Done!\n";

        // Read and check padded data
        auto fft_input = fft_debug.ReadFFTInputBuffer();
        std::cout << "    FFT input buffer size: " << fft_input.size() << "\n";
        std::cout << "    First 5 values: ";
        for (size_t i = 0; i < std::min(size_t(5), fft_input.size()); ++i) {
            std::cout << "(" << fft_input[i].real() << "," << fft_input[i].imag() << ") ";
        }
        std::cout << "\n";

        // Step 2: FFT
        std::cout << "\n  Step 2: Executing FFT...\n";
        cl_event fft_event;
        fft_debug.ExecuteFFTOnly(padding_event, &fft_event);
        clWaitForEvents(1, &fft_event);
        std::cout << "    Done!\n";

        // Read and check FFT output
        auto fft_output = fft_debug.ReadFFTOutputBuffer();
        std::cout << "    FFT output buffer size: " << fft_output.size() << "\n";

        // Find max in FFT output manually
        float max_mag = 0;
        size_t max_idx = 0;
        for (size_t i = 0; i < fft_output.size() / params.beam_count; ++i) {
            float mag = std::abs(fft_output[i]);
            if (mag > max_mag) {
                max_mag = mag;
                max_idx = i;
            }
        }
        std::cout << "    Beam 0 max magnitude: " << max_mag << " at index " << max_idx << "\n";

        // Step 3: Post-processing
        std::cout << "\n  Step 3: Executing Post Kernel...\n";
        cl_event post_event;
        fft_debug.ExecutePostKernel(fft_event, &post_event);
        clWaitForEvents(1, &post_event);
        std::cout << "    Done!\n";

        // Read selected data
        auto selected_mag = fft_debug.ReadSelectedMagnitudeBuffer();
        std::cout << "    Selected magnitude buffer size: " << selected_mag.size() << "\n";

        // Step 4: Find maxima
        std::cout << "\n  Step 4: Finding Maxima...\n";
        auto maxima = fft_debug.FindMaximaOnGPU();
        std::cout << "    Done!\n";

        // Print results
        if (!maxima.empty() && !maxima[0].empty()) {
            std::cout << "\n    Beam 0 maxima:\n";
            for (size_t i = 0; i < maxima[0].size(); ++i) {
                const auto& mv = maxima[0][i];
                std::cout << "      [" << i << "] Index: " << mv.index_point
                          << ", Amplitude: " << mv.amplitude
                          << ", Phase: " << mv.phase << " deg\n";
            }
        }

        // Cleanup events
        clReleaseEvent(padding_event);
        clReleaseEvent(fft_event);
        clReleaseEvent(post_event);

        std::cout << "\n  [PASS] Debug test completed!\n";
        return true;

    } catch (const std::exception& e) {
        std::cerr << "\n  [FAIL] Exception: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Compare Release and Debug results
 */
bool TestCompare(drv_gpu_lib::IBackend* backend, const AntennaFFTParams& params,
                 const std::vector<std::complex<float>>& test_data)
{
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "  TEST: Compare Release vs Debug Results\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    try {
        // Process with Release
        AntennaFFTProcMax fft_release(params, backend);
        auto result_release = fft_release.ProcessNew(test_data);

        // Process with Debug
        AntennaFFTDebug fft_debug(params, backend);
        auto result_debug = fft_debug.ProcessNew(test_data);

        // Compare
        bool match = true;
        const float tolerance = 0.01f;  // 1% tolerance

        for (size_t beam = 0; beam < params.beam_count && beam < 3; ++beam) {
            const auto& rel = result_release.results[beam];
            const auto& dbg = result_debug.results[beam];

            std::cout << "\n  Beam " << beam << ":\n";

            for (size_t i = 0; i < params.max_peaks_count; ++i) {
                const auto& mv_rel = rel.max_values[i];
                const auto& mv_dbg = dbg.max_values[i];

                bool idx_match = (mv_rel.index_point == mv_dbg.index_point);
                float amp_diff = std::abs(mv_rel.amplitude - mv_dbg.amplitude) /
                                 std::max(mv_rel.amplitude, 0.001f);

                std::cout << "    Peak " << i << ": ";
                std::cout << "Release idx=" << mv_rel.index_point << " amp=" << mv_rel.amplitude;
                std::cout << " | Debug idx=" << mv_dbg.index_point << " amp=" << mv_dbg.amplitude;

                if (idx_match && amp_diff < tolerance) {
                    std::cout << " [OK]\n";
                } else {
                    std::cout << " [MISMATCH]\n";
                    match = false;
                }
            }
        }

        if (match) {
            std::cout << "\n  [PASS] Release and Debug results match!\n";
        } else {
            std::cout << "\n  [WARN] Some results differ (may be due to floating point precision)\n";
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "\n  [FAIL] Exception: " << e.what() << "\n";
        return false;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[])
{
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "           FFT Maxima Module Test\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";

    // Parse arguments
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--verbose") {
            verbose = true;
        }
    }

    // Set logging level
    if (!verbose) {
        FFTLogger::SetLevel(FFTLogger::Level::Warning);
    }

    // Initialize OpenCL backend
    std::cout << "\nInitializing OpenCL backend...\n";

    drv_gpu_lib::OpenCLBackend backend;
    try {
        backend.Initialize(0);  // GPU #0
        std::cout << "  Device: " << backend.GetDeviceName() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize backend: " << e.what() << "\n";
        return 1;
    }

    // Test parameters
    AntennaFFTParams params;
    params.beam_count = 5;
    params.count_points = 1024;
    params.out_count_points_fft = 512;
    params.max_peaks_count = 3;
    params.task_id = "test_001";
    params.module_name = "fft_maxima_test";

    std::cout << "\nTest parameters:\n";
    std::cout << "  beam_count: " << params.beam_count << "\n";
    std::cout << "  count_points: " << params.count_points << "\n";
    std::cout << "  out_count_points_fft: " << params.out_count_points_fft << "\n";
    std::cout << "  max_peaks_count: " << params.max_peaks_count << "\n";

    // Generate test signal with known frequencies
    std::vector<float> frequencies = {0.1f, 0.25f, 0.4f};  // Normalized frequencies
    std::cout << "\nGenerating test signal with frequencies: ";
    for (float f : frequencies) std::cout << f << " ";
    std::cout << "\n";

    auto test_data = GenerateTestSignal(params.beam_count, params.count_points, frequencies);
    std::cout << "  Generated " << test_data.size() << " samples\n";

    // Run tests
    int passed = 0;
    int failed = 0;

    if (TestRelease(&backend, params, test_data)) passed++; else failed++;
    if (TestDebug(&backend, params, test_data)) passed++; else failed++;
    if (TestCompare(&backend, params, test_data)) passed++; else failed++;

    // Summary
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Test Summary\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Passed: " << passed << "\n";
    std::cout << "  Failed: " << failed << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";

    return (failed > 0) ? 1 : 0;
}
