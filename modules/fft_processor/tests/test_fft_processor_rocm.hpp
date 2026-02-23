#pragma once

/**
 * @file test_fft_processor_rocm.hpp
 * @brief Tests for FFTProcessorROCm -- hipFFT-based FFT processing
 *
 * Tests:
 * 1. Single beam, known tone -> peak at expected bin
 * 2. Multi-beam batch -> validate each beam peak
 * 3. MagPhase mode -> mag/phase consistency with complex
 * 4. MagPhaseFreq mode -> frequency array check
 * 5. GPU input (void*) -> complex output
 *
 * IMPORTANT: Tests compile ONLY with ENABLE_ROCM=1.
 * On Windows (no ROCm) this file is completely skipped.
 * Run only on Linux with AMD GPU and ROCm SDK.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "fft_processor_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <string>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_processor_rocm {

using namespace fft_processor;
using namespace drv_gpu_lib;

// =========================================================================
// Utilities
// =========================================================================

inline void print_result(ConsoleOutput& con, int gpu_id,
                         const std::string& test_name, bool passed) {
    std::string status = passed ? "PASSED" : "FAILED";
    std::string icon = passed ? "[+]" : "[X]";
    con.Print(gpu_id, "FFT ROCm", icon + " " + test_name + " ... " + status);
}

inline std::vector<std::complex<float>> GenerateSinusoid(
    float freq, float sample_rate, size_t n_point, float amplitude = 1.0f)
{
    std::vector<std::complex<float>> data(n_point);
    for (size_t i = 0; i < n_point; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
        data[i] = std::complex<float>(amplitude * std::cos(phase),
                                       amplitude * std::sin(phase));
    }
    return data;
}

inline std::vector<std::complex<float>> GenerateMultiBeam(
    size_t beam_count, size_t n_point, float sample_rate,
    float base_freq, float freq_step)
{
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (size_t b = 0; b < beam_count; ++b) {
        float freq = base_freq + b * freq_step;
        for (size_t i = 0; i < n_point; ++i) {
            float t = static_cast<float>(i) / sample_rate;
            float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
            data[b * n_point + i] = std::complex<float>(std::cos(phase), std::sin(phase));
        }
    }
    return data;
}

inline size_t FindPeakBin(const std::vector<float>& magnitude, size_t search_range) {
    size_t peak_idx = 0;
    float peak_val = 0.0f;
    size_t limit = std::min(magnitude.size(), search_range);
    for (size_t i = 0; i < limit; ++i) {
        if (magnitude[i] > peak_val) {
            peak_val = magnitude[i];
            peak_idx = i;
        }
    }
    return peak_idx;
}

inline size_t FindPeakBinComplex(const std::vector<std::complex<float>>& spectrum, size_t search_range) {
    size_t peak_idx = 0;
    float peak_val = 0.0f;
    size_t limit = std::min(spectrum.size(), search_range);
    for (size_t i = 0; i < limit; ++i) {
        float mag = std::abs(spectrum[i]);
        if (mag > peak_val) {
            peak_val = mag;
            peak_idx = i;
        }
    }
    return peak_idx;
}

// =========================================================================
// Test 1: Single beam complex -- known tone at 100 Hz
// =========================================================================

inline bool test_single_beam_complex(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        FFTProcessorROCm fft(&backend);

        const float freq = 100.0f;
        const float sample_rate = 1000.0f;
        const uint32_t n_point = 1024;

        auto data = GenerateSinusoid(freq, sample_rate, n_point);

        FFTProcessorParams params;
        params.beam_count = 1;
        params.n_point = n_point;
        params.sample_rate = sample_rate;
        params.output_mode = FFTOutputMode::COMPLEX;

        auto results = fft.ProcessComplex(data, params);

        bool ok = (results.size() == 1);
        if (ok) {
            uint32_t nFFT = results[0].nFFT;
            size_t expected_bin = static_cast<size_t>(freq * nFFT / sample_rate);
            size_t peak_bin = FindPeakBinComplex(results[0].spectrum, nFFT / 2);

            ok = (peak_bin == expected_bin);

            con.Print(gpu_id, "FFT ROCm",
                      "  nFFT=" + std::to_string(nFFT) +
                      ", expected_bin=" + std::to_string(expected_bin) +
                      ", peak_bin=" + std::to_string(peak_bin));
        }

        print_result(con, gpu_id, "SingleBeamComplex (100 Hz)", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "FFT ROCm", "[X] SingleBeamComplex EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Test 2: Multi-beam batch -- different frequencies
// =========================================================================

inline bool test_multi_beam_batch(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        FFTProcessorROCm fft(&backend);

        const uint32_t beam_count = 8;
        const uint32_t n_point = 1024;
        const float sample_rate = 1000.0f;
        const float base_freq = 50.0f;
        const float freq_step = 25.0f;

        auto data = GenerateMultiBeam(beam_count, n_point, sample_rate, base_freq, freq_step);

        FFTProcessorParams params;
        params.beam_count = beam_count;
        params.n_point = n_point;
        params.sample_rate = sample_rate;
        params.output_mode = FFTOutputMode::COMPLEX;

        auto results = fft.ProcessComplex(data, params);

        bool ok = (results.size() == beam_count);
        for (uint32_t b = 0; b < beam_count && ok; ++b) {
            float freq = base_freq + b * freq_step;
            uint32_t nFFT = results[b].nFFT;
            size_t expected_bin = static_cast<size_t>(freq * nFFT / sample_rate);
            size_t peak_bin = FindPeakBinComplex(results[b].spectrum, nFFT / 2);

            if (peak_bin != expected_bin) {
                con.Print(gpu_id, "FFT ROCm",
                          "  Beam " + std::to_string(b) + ": expected bin " +
                          std::to_string(expected_bin) + ", got " + std::to_string(peak_bin));
                ok = false;
            }
        }

        print_result(con, gpu_id, "MultiBeamBatch (8 beams)", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "FFT ROCm", "[X] MultiBeamBatch EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Test 3: MagPhase mode -- consistency with complex output
// =========================================================================

inline bool test_mag_phase_consistency(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        const float freq = 200.0f;
        const float sample_rate = 1000.0f;
        const uint32_t n_point = 512;

        auto data = GenerateSinusoid(freq, sample_rate, n_point);

        FFTProcessorParams params;
        params.beam_count = 1;
        params.n_point = n_point;
        params.sample_rate = sample_rate;

        // Get complex result
        FFTProcessorROCm fft_complex(&backend);
        params.output_mode = FFTOutputMode::COMPLEX;
        auto complex_results = fft_complex.ProcessComplex(data, params);

        // Get mag/phase result
        FFTProcessorROCm fft_magphase(&backend);
        params.output_mode = FFTOutputMode::MAGNITUDE_PHASE;
        auto mp_results = fft_magphase.ProcessMagPhase(data, params);

        bool ok = (complex_results.size() == 1) && (mp_results.size() == 1);
        if (ok) {
            uint32_t nFFT = complex_results[0].nFFT;
            float max_mag_err = 0.0f;
            float max_phase_err = 0.0f;

            for (uint32_t k = 0; k < nFFT; ++k) {
                float cpu_mag = std::abs(complex_results[0].spectrum[k]);
                float cpu_phase = std::arg(complex_results[0].spectrum[k]);
                float gpu_mag = mp_results[0].magnitude[k];
                float gpu_phase = mp_results[0].phase[k];

                float mag_err = std::fabs(cpu_mag - gpu_mag);
                float phase_err = std::fabs(cpu_phase - gpu_phase);

                // Handle phase wrapping
                if (phase_err > static_cast<float>(M_PI)) {
                    phase_err = 2.0f * static_cast<float>(M_PI) - phase_err;
                }

                max_mag_err = std::max(max_mag_err, mag_err);
                // Only check phase where magnitude is significant
                if (cpu_mag > 1e-3f) {
                    max_phase_err = std::max(max_phase_err, phase_err);
                }
            }

            ok = (max_mag_err < 1e-2f) && (max_phase_err < 1e-2f);
            con.Print(gpu_id, "FFT ROCm",
                      "  max_mag_err=" + std::to_string(max_mag_err) +
                      ", max_phase_err=" + std::to_string(max_phase_err));
        }

        print_result(con, gpu_id, "MagPhase Consistency", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "FFT ROCm", "[X] MagPhase Consistency EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Test 4: MagPhaseFreq -- frequency array validation
// =========================================================================

inline bool test_mag_phase_freq(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        FFTProcessorROCm fft(&backend);

        const float freq = 150.0f;
        const float sample_rate = 1000.0f;
        const uint32_t n_point = 1024;

        auto data = GenerateSinusoid(freq, sample_rate, n_point);

        FFTProcessorParams params;
        params.beam_count = 1;
        params.n_point = n_point;
        params.sample_rate = sample_rate;
        params.output_mode = FFTOutputMode::MAGNITUDE_PHASE_FREQ;

        auto results = fft.ProcessMagPhase(data, params);

        bool ok = (results.size() == 1);
        if (ok) {
            ok = !results[0].frequency.empty();
            if (ok) {
                uint32_t nFFT = results[0].nFFT;
                float freq_step = sample_rate / static_cast<float>(nFFT);

                // Verify frequency array
                for (uint32_t k = 0; k < nFFT && ok; ++k) {
                    float expected_freq = static_cast<float>(k) * freq_step;
                    if (std::fabs(results[0].frequency[k] - expected_freq) > 1e-4f) {
                        ok = false;
                    }
                }

                // Verify peak at target frequency
                size_t peak_bin = FindPeakBin(results[0].magnitude, nFFT / 2);
                float peak_freq = results[0].frequency[peak_bin];
                float freq_error = std::fabs(peak_freq - freq);

                con.Print(gpu_id, "FFT ROCm",
                          "  peak_freq=" + std::to_string(peak_freq) +
                          " Hz, error=" + std::to_string(freq_error) + " Hz");

                ok = ok && (freq_error < freq_step);  // within one bin
            }
        }

        print_result(con, gpu_id, "MagPhaseFreq (freq array)", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "FFT ROCm", "[X] MagPhaseFreq EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Test 5: GPU input -- void* device pointer
// =========================================================================

inline bool test_gpu_input(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        FFTProcessorROCm fft(&backend);

        const float freq = 100.0f;
        const float sample_rate = 1000.0f;
        const uint32_t n_point = 1024;

        auto data = GenerateSinusoid(freq, sample_rate, n_point);

        // Upload data to GPU manually
        size_t data_size = data.size() * sizeof(std::complex<float>);
        void* gpu_data = backend.Allocate(data_size);
        backend.MemcpyHostToDevice(gpu_data, data.data(), data_size);

        FFTProcessorParams params;
        params.beam_count = 1;
        params.n_point = n_point;
        params.sample_rate = sample_rate;
        params.output_mode = FFTOutputMode::COMPLEX;

        auto results = fft.ProcessComplex(gpu_data, params, data_size);

        backend.Free(gpu_data);

        bool ok = (results.size() == 1);
        if (ok) {
            uint32_t nFFT = results[0].nFFT;
            size_t expected_bin = static_cast<size_t>(freq * nFFT / sample_rate);
            size_t peak_bin = FindPeakBinComplex(results[0].spectrum, nFFT / 2);

            ok = (peak_bin == expected_bin);
            con.Print(gpu_id, "FFT ROCm",
                      "  GPU input: expected_bin=" + std::to_string(expected_bin) +
                      ", peak_bin=" + std::to_string(peak_bin));
        }

        print_result(con, gpu_id, "GPU Input (void*)", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "FFT ROCm", "[X] GPU Input EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Main test runner
// =========================================================================

inline void run() {
    ConsoleOutput con;
    int gpu_id = 0;

    con.Print(gpu_id, "FFT ROCm", "");
    con.Print(gpu_id, "FFT ROCm", "============================================");
    con.Print(gpu_id, "FFT ROCm", "  FFTProcessorROCm Tests (hipFFT)");
    con.Print(gpu_id, "FFT ROCm", "============================================");

    // Check for ROCm devices
    int device_count = ROCmCore::GetAvailableDeviceCount();
    con.Print(gpu_id, "FFT ROCm", "Available ROCm devices: " + std::to_string(device_count));

    if (device_count == 0) {
        con.Print(gpu_id, "FFT ROCm", "[!] No ROCm devices found -- skipping tests");
        return;
    }

    int passed = 0;
    int total = 5;

    if (test_single_beam_complex(con, gpu_id)) ++passed;
    if (test_multi_beam_batch(con, gpu_id)) ++passed;
    if (test_mag_phase_consistency(con, gpu_id)) ++passed;
    if (test_mag_phase_freq(con, gpu_id)) ++passed;
    if (test_gpu_input(con, gpu_id)) ++passed;

    con.Print(gpu_id, "FFT ROCm", "");
    con.Print(gpu_id, "FFT ROCm", "Results: " + std::to_string(passed) + "/" +
                                    std::to_string(total) + " passed");
    con.Print(gpu_id, "FFT ROCm", "============================================");
    con.Print(gpu_id, "FFT ROCm", "");
}

}  // namespace test_fft_processor_rocm

#endif  // ENABLE_ROCM
