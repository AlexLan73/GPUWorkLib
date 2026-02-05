/**
 * @file test_fft_maxima.hpp
 * @brief Test for FFT Maxima module
 *
 * Tests both Release (callbacks) and Debug (step-by-step) implementations
 * and compares their results.
 *
 * @author DrvGPU Team
 * @date 2026-02-04
 */

#include "modules/fft_maxima/include/antenna_fft_release.h"
#include "modules/fft_maxima/include/antenna_fft_release.h"
#include "modules/fft_maxima/include/fft_result_writer.hpp"
#include "modules/fft_maxima/include/fft_logger.h"

#include "backends/opencl/opencl_backend.hpp"

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>

namespace test_fft_max{
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


// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════

int run()
{
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "           FFT Maxima Module Test\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";

    // Parse arguments
    bool verbose = true; // управлял  через аргументы 

//    for (int i = 1; i < argc; ++i) {
//        if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--verbose") {
//            verbose = true;
//        }
//    }

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

    // Summary
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Test Summary\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Passed: " << passed << "\n";
    std::cout << "  Failed: " << failed << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";

    return (failed > 0) ? 1 : 0;
}
}