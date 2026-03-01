#pragma once

/**
 * @file test_complex_to_mag_phase_rocm.hpp
 * @brief Tests for ComplexToMagPhaseROCm -- direct complex->mag+phase conversion
 *
 * Tests:
 * 1. Single beam, CPU -> CPU, sinusoid -> verify mag/phase vs std::abs/std::arg
 * 2. Multi-beam batch, CPU -> CPU, different amplitudes
 * 3. GPU input (void*) -> CPU output
 * 4. CPU -> GPU output (ProcessToGPU), verify interleaved data
 * 5. GPU -> GPU output (full GPU pipeline)
 * 6. Accuracy: edge cases (zero, pure real, pure imaginary, large/small values)
 *
 * IMPORTANT: Tests compile ONLY with ENABLE_ROCM=1.
 * On Windows (no ROCm) this file is completely skipped.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include "complex_to_mag_phase_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/console_output.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_complex_to_mag_phase_rocm {

using namespace fft_processor;
using namespace drv_gpu_lib;

// =========================================================================
// Utilities
// =========================================================================

inline void print_result(ConsoleOutput& con, int gpu_id,
                         const std::string& test_name, bool passed) {
    std::string status = passed ? "PASSED" : "FAILED";
    std::string icon = passed ? "[+]" : "[X]";
    con.Print(gpu_id, "C2MP ROCm", icon + " " + test_name + " ... " + status);
}

/// Generate complex sinusoid (IQ) for testing
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

/// Generate multi-beam data with varying amplitudes
inline std::vector<std::complex<float>> GenerateMultiBeamVaryingAmplitude(
    size_t beam_count, size_t n_point, float sample_rate,
    float freq, float base_amplitude, float amp_step)
{
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (size_t b = 0; b < beam_count; ++b) {
        float amplitude = base_amplitude + b * amp_step;
        for (size_t i = 0; i < n_point; ++i) {
            float t = static_cast<float>(i) / sample_rate;
            float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
            data[b * n_point + i] = std::complex<float>(
                amplitude * std::cos(phase),
                amplitude * std::sin(phase));
        }
    }
    return data;
}

// =========================================================================
// Test 1: Single beam, CPU -> CPU (sinusoid)
// =========================================================================

inline bool test_single_beam_cpu(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        ComplexToMagPhaseROCm converter(&backend);

        const float freq = 100.0f;
        const float sample_rate = 1000.0f;
        const uint32_t n_point = 4096;
        const float amplitude = 2.5f;

        auto data = GenerateSinusoid(freq, sample_rate, n_point, amplitude);

        MagPhaseParams params;
        params.beam_count = 1;
        params.n_point = n_point;

        auto results = converter.Process(data, params);

        bool ok = (results.size() == 1);
        if (ok) {
            ok = (results[0].n_point == n_point);
        }

        // Compare GPU results with CPU reference
        float max_mag_err = 0.0f;
        float max_phase_err = 0.0f;

        if (ok) {
            for (uint32_t k = 0; k < n_point; ++k) {
                float cpu_mag = std::abs(data[k]);
                float cpu_phase = std::arg(data[k]);
                float gpu_mag = results[0].magnitude[k];
                float gpu_phase = results[0].phase[k];

                float mag_err = std::fabs(cpu_mag - gpu_mag);
                float phase_err = std::fabs(cpu_phase - gpu_phase);

                // Handle phase wrapping
                if (phase_err > static_cast<float>(M_PI)) {
                    phase_err = 2.0f * static_cast<float>(M_PI) - phase_err;
                }

                max_mag_err = std::max(max_mag_err, mag_err);
                if (cpu_mag > 1e-6f) {
                    max_phase_err = std::max(max_phase_err, phase_err);
                }
            }

            ok = (max_mag_err < 1e-3f) && (max_phase_err < 1e-3f);
            con.Print(gpu_id, "C2MP ROCm",
                      "  max_mag_err=" + std::to_string(max_mag_err) +
                      ", max_phase_err=" + std::to_string(max_phase_err));
        }

        print_result(con, gpu_id, "SingleBeam CPU->CPU (sinusoid)", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "C2MP ROCm", "[X] SingleBeam EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Test 2: Multi-beam batch, CPU -> CPU
// =========================================================================

inline bool test_multi_beam_cpu(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        ComplexToMagPhaseROCm converter(&backend);

        const uint32_t beam_count = 8;
        const uint32_t n_point = 4096;
        const float sample_rate = 12000.0f;
        const float freq = 500.0f;
        const float base_amp = 0.5f;
        const float amp_step = 0.5f;

        auto data = GenerateMultiBeamVaryingAmplitude(
            beam_count, n_point, sample_rate, freq, base_amp, amp_step);

        MagPhaseParams params;
        params.beam_count = beam_count;
        params.n_point = n_point;

        auto results = converter.Process(data, params);

        bool ok = (results.size() == beam_count);

        float max_mag_err = 0.0f;
        for (uint32_t b = 0; b < beam_count && ok; ++b) {
            ok = (results[b].beam_id == b);
            ok = ok && (results[b].n_point == n_point);

            float expected_amp = base_amp + b * amp_step;
            for (uint32_t k = 0; k < n_point && ok; ++k) {
                size_t idx = b * n_point + k;
                float cpu_mag = std::abs(data[idx]);
                float gpu_mag = results[b].magnitude[k];

                float mag_err = std::fabs(cpu_mag - gpu_mag);
                max_mag_err = std::max(max_mag_err, mag_err);
            }
        }

        ok = ok && (max_mag_err < 1e-3f);
        con.Print(gpu_id, "C2MP ROCm",
                  "  " + std::to_string(beam_count) + " beams, max_mag_err=" +
                  std::to_string(max_mag_err));

        print_result(con, gpu_id, "MultiBeam CPU->CPU (8 beams)", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "C2MP ROCm", "[X] MultiBeam EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Test 3: GPU input (void*) -> CPU output
// =========================================================================

inline bool test_gpu_input(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        ComplexToMagPhaseROCm converter(&backend);

        const uint32_t n_point = 2048;
        const float freq = 200.0f;
        const float sample_rate = 1000.0f;

        auto data = GenerateSinusoid(freq, sample_rate, n_point);

        // Upload data to GPU manually (external context)
        size_t data_size = data.size() * sizeof(std::complex<float>);
        void* gpu_data = backend.Allocate(data_size);
        backend.MemcpyHostToDevice(gpu_data, data.data(), data_size);

        MagPhaseParams params;
        params.beam_count = 1;
        params.n_point = n_point;

        auto results = converter.Process(gpu_data, params, data_size);

        backend.Free(gpu_data);

        bool ok = (results.size() == 1) && (results[0].n_point == n_point);

        float max_mag_err = 0.0f;
        if (ok) {
            for (uint32_t k = 0; k < n_point; ++k) {
                float cpu_mag = std::abs(data[k]);
                float gpu_mag = results[0].magnitude[k];
                float mag_err = std::fabs(cpu_mag - gpu_mag);
                max_mag_err = std::max(max_mag_err, mag_err);
            }
            ok = (max_mag_err < 1e-3f);
        }

        con.Print(gpu_id, "C2MP ROCm",
                  "  GPU input: max_mag_err=" + std::to_string(max_mag_err));

        print_result(con, gpu_id, "GPU Input (void*) -> CPU", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "C2MP ROCm", "[X] GPU Input EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Test 4: CPU -> GPU output (ProcessToGPU)
// =========================================================================

inline bool test_process_to_gpu_cpu(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        ComplexToMagPhaseROCm converter(&backend);

        const uint32_t n_point = 1024;
        const float freq = 300.0f;
        const float sample_rate = 2000.0f;
        const float amplitude = 3.0f;

        auto data = GenerateSinusoid(freq, sample_rate, n_point, amplitude);

        MagPhaseParams params;
        params.beam_count = 1;
        params.n_point = n_point;

        // Process CPU -> GPU
        void* gpu_output = converter.ProcessToGPU(data, params);

        bool ok = (gpu_output != nullptr);

        if (ok) {
            // Read back interleaved data to verify
            size_t output_size = n_point * 2 * sizeof(float);
            std::vector<float> raw(n_point * 2);
            backend.MemcpyDeviceToHost(raw.data(), gpu_output, output_size);

            float max_mag_err = 0.0f;
            float max_phase_err = 0.0f;

            for (uint32_t k = 0; k < n_point; ++k) {
                float gpu_mag   = raw[k * 2    ];  // .x = mag
                float gpu_phase = raw[k * 2 + 1];  // .y = phase
                float cpu_mag   = std::abs(data[k]);
                float cpu_phase = std::arg(data[k]);

                float mag_err = std::fabs(cpu_mag - gpu_mag);
                float phase_err = std::fabs(cpu_phase - gpu_phase);
                if (phase_err > static_cast<float>(M_PI)) {
                    phase_err = 2.0f * static_cast<float>(M_PI) - phase_err;
                }

                max_mag_err = std::max(max_mag_err, mag_err);
                if (cpu_mag > 1e-6f) {
                    max_phase_err = std::max(max_phase_err, phase_err);
                }
            }

            ok = (max_mag_err < 1e-3f) && (max_phase_err < 1e-3f);
            con.Print(gpu_id, "C2MP ROCm",
                      "  ProcessToGPU(CPU): max_mag_err=" + std::to_string(max_mag_err) +
                      ", max_phase_err=" + std::to_string(max_phase_err));

            // IMPORTANT: caller must free
            backend.Free(gpu_output);
        }

        print_result(con, gpu_id, "CPU -> GPU (ProcessToGPU)", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "C2MP ROCm", "[X] ProcessToGPU(CPU) EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Test 5: GPU -> GPU output (full GPU pipeline)
// =========================================================================

inline bool test_process_to_gpu_gpu(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        ComplexToMagPhaseROCm converter(&backend);

        const uint32_t beam_count = 4;
        const uint32_t n_point = 2048;
        const float freq = 150.0f;
        const float sample_rate = 1000.0f;

        auto data = GenerateMultiBeamVaryingAmplitude(
            beam_count, n_point, sample_rate, freq, 1.0f, 1.0f);

        // Upload data to GPU (external context)
        size_t data_size = data.size() * sizeof(std::complex<float>);
        void* gpu_input = backend.Allocate(data_size);
        backend.MemcpyHostToDevice(gpu_input, data.data(), data_size);

        MagPhaseParams params;
        params.beam_count = beam_count;
        params.n_point = n_point;

        // GPU -> GPU: zero-copy path
        void* gpu_output = converter.ProcessToGPU(gpu_input, params, data_size);

        bool ok = (gpu_output != nullptr);

        if (ok) {
            // Read back and verify
            size_t total = beam_count * n_point;
            std::vector<float> raw(total * 2);
            backend.MemcpyDeviceToHost(raw.data(), gpu_output, total * 2 * sizeof(float));

            float max_mag_err = 0.0f;
            for (size_t i = 0; i < total; ++i) {
                float gpu_mag = raw[i * 2];  // .x = mag
                float cpu_mag = std::abs(data[i]);
                float mag_err = std::fabs(cpu_mag - gpu_mag);
                max_mag_err = std::max(max_mag_err, mag_err);
            }

            ok = (max_mag_err < 1e-3f);
            con.Print(gpu_id, "C2MP ROCm",
                      "  GPU->GPU: " + std::to_string(beam_count) + " beams, max_mag_err=" +
                      std::to_string(max_mag_err));

            backend.Free(gpu_output);
        }

        backend.Free(gpu_input);

        print_result(con, gpu_id, "GPU -> GPU (ProcessToGPU)", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "C2MP ROCm", "[X] ProcessToGPU(GPU) EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Test 6: Accuracy -- edge cases
// =========================================================================

inline bool test_accuracy(ConsoleOutput& con, int gpu_id) {
    try {
        ROCmBackend backend;
        backend.Initialize(gpu_id);

        ComplexToMagPhaseROCm converter(&backend);

        // Edge case data: zero, pure real, pure imaginary, large, small, negative
        std::vector<std::complex<float>> data = {
            {0.0f, 0.0f},        // zero
            {1.0f, 0.0f},        // pure real positive
            {-1.0f, 0.0f},       // pure real negative
            {0.0f, 1.0f},        // pure imaginary positive
            {0.0f, -1.0f},       // pure imaginary negative
            {1.0f, 1.0f},        // 45 degrees
            {-1.0f, 1.0f},       // 135 degrees
            {-1.0f, -1.0f},      // -135 degrees
            {1.0f, -1.0f},       // -45 degrees
            {1000.0f, 2000.0f},  // large values
            {1e-6f, 1e-6f},      // small values
            {3.0f, 4.0f},        // 3-4-5 triangle
            {0.0f, 0.0f},       // padding to make n_point reasonable
            {5.0f, 12.0f},       // 5-12-13 triangle
            {1.0f, 0.0001f},     // nearly real
            {0.0001f, 1.0f},     // nearly imaginary
        };

        const uint32_t n_point = static_cast<uint32_t>(data.size());

        MagPhaseParams params;
        params.beam_count = 1;
        params.n_point = n_point;

        auto results = converter.Process(data, params);

        bool ok = (results.size() == 1) && (results[0].n_point == n_point);

        float max_mag_err = 0.0f;
        float max_phase_err = 0.0f;

        if (ok) {
            for (uint32_t k = 0; k < n_point; ++k) {
                float cpu_mag = std::abs(data[k]);
                float cpu_phase = std::arg(data[k]);
                float gpu_mag = results[0].magnitude[k];
                float gpu_phase = results[0].phase[k];

                float mag_err = std::fabs(cpu_mag - gpu_mag);
                float phase_err = std::fabs(cpu_phase - gpu_phase);

                // Handle phase wrapping
                if (phase_err > static_cast<float>(M_PI)) {
                    phase_err = 2.0f * static_cast<float>(M_PI) - phase_err;
                }

                max_mag_err = std::max(max_mag_err, mag_err);
                // Only check phase where magnitude is significant
                if (cpu_mag > 1e-5f) {
                    max_phase_err = std::max(max_phase_err, phase_err);
                }
            }

            // Verify specific known values
            // data[11] = {3, 4} -> mag = 5.0, phase = atan2(4,3) = 0.9273
            float mag_3_4_err = std::fabs(results[0].magnitude[11] - 5.0f);
            float phase_3_4_err = std::fabs(results[0].phase[11] - std::atan2(4.0f, 3.0f));

            con.Print(gpu_id, "C2MP ROCm",
                      "  3-4-5: mag_err=" + std::to_string(mag_3_4_err) +
                      ", phase_err=" + std::to_string(phase_3_4_err));

            ok = (max_mag_err < 1e-2f) && (max_phase_err < 1e-2f);
            con.Print(gpu_id, "C2MP ROCm",
                      "  max_mag_err=" + std::to_string(max_mag_err) +
                      ", max_phase_err=" + std::to_string(max_phase_err));
        }

        print_result(con, gpu_id, "Accuracy (edge cases)", ok);
        return ok;
    } catch (const std::exception& e) {
        con.Print(gpu_id, "C2MP ROCm", "[X] Accuracy EXCEPTION: " + std::string(e.what()));
        return false;
    }
}

// =========================================================================
// Main test runner
// =========================================================================

inline void run() {
    auto& con = ConsoleOutput::GetInstance();
    con.Start();
    int gpu_id = 0;

    con.Print(gpu_id, "C2MP ROCm", "");
    con.Print(gpu_id, "C2MP ROCm", "============================================");
    con.Print(gpu_id, "C2MP ROCm", "  ComplexToMagPhaseROCm Tests");
    con.Print(gpu_id, "C2MP ROCm", "============================================");

    // Check for ROCm devices
    int device_count = ROCmCore::GetAvailableDeviceCount();
    con.Print(gpu_id, "C2MP ROCm", "Available ROCm devices: " + std::to_string(device_count));

    if (device_count == 0) {
        con.Print(gpu_id, "C2MP ROCm", "[!] No ROCm devices found -- skipping tests");
        return;
    }

    int passed = 0;
    int total = 6;

    if (test_single_beam_cpu(con, gpu_id)) ++passed;
    if (test_multi_beam_cpu(con, gpu_id)) ++passed;
    if (test_gpu_input(con, gpu_id)) ++passed;
    if (test_process_to_gpu_cpu(con, gpu_id)) ++passed;
    if (test_process_to_gpu_gpu(con, gpu_id)) ++passed;
    if (test_accuracy(con, gpu_id)) ++passed;

    con.Print(gpu_id, "C2MP ROCm", "");
    con.Print(gpu_id, "C2MP ROCm", "Results: " + std::to_string(passed) + "/" +
                                    std::to_string(total) + " passed");
    con.Print(gpu_id, "C2MP ROCm", "============================================");
    con.Print(gpu_id, "C2MP ROCm", "");
}

}  // namespace test_complex_to_mag_phase_rocm

#endif  // ENABLE_ROCM
