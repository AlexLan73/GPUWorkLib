#pragma once

/**
 * @file test_fft_processor.hpp
 * @brief Тесты FFTProcessor — проверка FFT с разными режимами вывода
 *
 * Тесты:
 * 1. Один луч, known tone → проверка пика на частоте
 * 2. N лучей, batch → проверка каждого луча
 * 3. COMPLEX → magnitude+phase → сравнение с CPU reference
 * 4. MAGNITUDE_PHASE_FREQ → проверка freq = bin * fs / nFFT
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "fft_processor.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <string>
#include <iomanip>
#include <memory>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_processor {

// ════════════════════════════════════════════════════════════════════════════
// Утилиты
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Генерировать синусоиду на CPU
 */
inline std::vector<std::complex<float>> GenerateSinusoid(
    float freq, float sample_rate, size_t n_point, float amplitude = 1.0f)
{
    std::vector<std::complex<float>> data(n_point);
    for (size_t i = 0; i < n_point; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
        data[i] = std::complex<float>(amplitude * std::cos(phase), amplitude * std::sin(phase));
    }
    return data;
}

/**
 * @brief Генерировать несколько лучей с разными частотами
 */
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

/**
 * @brief Найти индекс максимальной магнитуды
 */
inline size_t FindPeakBin(const std::vector<float>& magnitude, size_t search_range) {
    size_t peak_idx = 0;
    float peak_val = 0.0f;
    for (size_t i = 0; i < search_range && i < magnitude.size(); ++i) {
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
    for (size_t i = 0; i < search_range && i < spectrum.size(); ++i) {
        float mag = std::abs(spectrum[i]);
        if (mag > peak_val) {
            peak_val = mag;
            peak_idx = i;
        }
    }
    return peak_idx;
}

// ════════════════════════════════════════════════════════════════════════════
// Тесты
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Test 1: Один луч, COMPLEX output — проверка пика
 */
inline bool TestSingleBeamComplex(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [Test 1] Single beam, COMPLEX output...\n";

    const float freq = 100.0f;
    const float sample_rate = 1000.0f;
    const size_t n_point = 1024;

    auto data = GenerateSinusoid(freq, sample_rate, n_point);

    fft_processor::FFTProcessor fft(backend);
    fft_processor::FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = static_cast<uint32_t>(n_point);
    params.sample_rate = sample_rate;
    params.repeat_count = 1;

    auto results = fft.ProcessComplex(data, params);

    if (results.size() != 1) {
        std::cerr << "    FAIL: expected 1 result, got " << results.size() << "\n";
        return false;
    }

    const auto& r = results[0];
    size_t peak_bin = FindPeakBinComplex(r.spectrum, r.nFFT / 2);
    float detected_freq = static_cast<float>(peak_bin) * sample_rate / r.nFFT;

    float error = std::abs(detected_freq - freq);
    bool pass = error < (sample_rate / r.nFFT);  // Ошибка < 1 бин

    std::cout << "    nFFT: " << r.nFFT << "\n";
    std::cout << "    Peak bin: " << peak_bin << "\n";
    std::cout << "    Detected freq: " << detected_freq << " Hz (expected " << freq << " Hz)\n";
    std::cout << "    Error: " << error << " Hz\n";
    std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/**
 * @brief Test 2: Multi-beam, MAGNITUDE_PHASE_FREQ — проверка частот
 */
inline bool TestMultiBeamMagPhaseFreq(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [Test 2] Multi-beam, MAGNITUDE_PHASE_FREQ output...\n";

    const size_t beam_count = 8;
    const size_t n_point = 2048;
    const float sample_rate = 10000.0f;
    const float base_freq = 500.0f;
    const float freq_step = 100.0f;

    auto data = GenerateMultiBeam(beam_count, n_point, sample_rate, base_freq, freq_step);

    fft_processor::FFTProcessor fft(backend);
    fft_processor::FFTProcessorParams params;
    params.beam_count = static_cast<uint32_t>(beam_count);
    params.n_point = static_cast<uint32_t>(n_point);
    params.sample_rate = sample_rate;
    params.output_mode = fft_processor::FFTOutputMode::MAGNITUDE_PHASE_FREQ;
    params.repeat_count = 1;

    auto results = fft.ProcessMagPhase(data, params);

    if (results.size() != beam_count) {
        std::cerr << "    FAIL: expected " << beam_count << " results, got " << results.size() << "\n";
        return false;
    }

    bool all_pass = true;
    float freq_resolution = sample_rate / results[0].nFFT;

    for (size_t b = 0; b < beam_count; ++b) {
        const auto& r = results[b];
        float expected_freq = base_freq + b * freq_step;

        // Найти пик
        size_t peak_bin = FindPeakBin(r.magnitude, r.nFFT / 2);
        float detected_freq = r.frequency[peak_bin];
        float error = std::abs(detected_freq - expected_freq);

        bool pass = error < freq_resolution;
        if (!pass) all_pass = false;

        std::cout << "    Beam " << b << ": expected " << std::fixed << std::setprecision(1)
                  << expected_freq << " Hz, detected " << detected_freq << " Hz"
                  << " (error=" << error << " Hz) " << (pass ? "OK" : "FAIL") << "\n";
    }

    // Проверить что frequency массив заполнен правильно
    const auto& r0 = results[0];
    if (r0.frequency.empty()) {
        std::cerr << "    FAIL: frequency array is empty\n";
        return false;
    }
    float expected_freq_step = sample_rate / r0.nFFT;
    float actual_freq_step = r0.frequency[1] - r0.frequency[0];
    if (std::abs(actual_freq_step - expected_freq_step) > 1e-3f) {
        std::cerr << "    FAIL: freq step mismatch: " << actual_freq_step
                  << " vs expected " << expected_freq_step << "\n";
        return false;
    }

    std::cout << "    " << (all_pass ? "PASS" : "FAIL") << "\n";
    return all_pass;
}

/**
 * @brief Test 3: MAGNITUDE_PHASE vs COMPLEX — сравнение
 */
inline bool TestMagPhaseVsComplex(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [Test 3] MAGNITUDE_PHASE vs COMPLEX consistency...\n";

    const float freq = 250.0f;
    const float sample_rate = 4000.0f;
    const size_t n_point = 512;

    auto data = GenerateSinusoid(freq, sample_rate, n_point);

    fft_processor::FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = static_cast<uint32_t>(n_point);
    params.sample_rate = sample_rate;
    params.repeat_count = 1;

    // Complex output
    fft_processor::FFTProcessor fft1(backend);
    auto complex_results = fft1.ProcessComplex(data, params);

    // MagPhase output
    params.output_mode = fft_processor::FFTOutputMode::MAGNITUDE_PHASE;
    fft_processor::FFTProcessor fft2(backend);
    auto mag_phase_results = fft2.ProcessMagPhase(data, params);

    if (complex_results.size() != 1 || mag_phase_results.size() != 1) {
        std::cerr << "    FAIL: wrong result count\n";
        return false;
    }

    const auto& cr = complex_results[0];
    const auto& mr = mag_phase_results[0];

    if (cr.nFFT != mr.nFFT) {
        std::cerr << "    FAIL: nFFT mismatch\n";
        return false;
    }

    // Сравнить magnitude из complex с magnitude из mag_phase
    float max_mag_error = 0.0f;
    float max_phase_error = 0.0f;

    for (uint32_t k = 0; k < cr.nFFT; ++k) {
        float ref_mag = std::abs(cr.spectrum[k]);
        float ref_phase = std::arg(cr.spectrum[k]);

        float mag_err = std::abs(ref_mag - mr.magnitude[k]);
        max_mag_error = std::max(max_mag_error, mag_err);

        // Фазу сравниваем только при значимой магнитуде
        if (ref_mag > 0.01f) {
            // Нормализация разницы фаз в [-pi, pi]
            float phase_diff = ref_phase - mr.phase[k];
            while (phase_diff > static_cast<float>(M_PI)) phase_diff -= 2.0f * static_cast<float>(M_PI);
            while (phase_diff < -static_cast<float>(M_PI)) phase_diff += 2.0f * static_cast<float>(M_PI);
            max_phase_error = std::max(max_phase_error, std::abs(phase_diff));
        }
    }

    bool pass = (max_mag_error < 1e-3f) && (max_phase_error < 1e-3f);

    std::cout << "    Max magnitude error: " << max_mag_error << "\n";
    std::cout << "    Max phase error: " << max_phase_error << " rad\n";
    std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/**
 * @brief Test 4: Profiling data check
 */
inline bool TestProfiling(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [Test 4] Profiling data...\n";

    auto data = GenerateSinusoid(100.0f, 1000.0f, 1024);

    fft_processor::FFTProcessor fft(backend);
    fft_processor::FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = 1024;
    params.sample_rate = 1000.0f;

    fft.ProcessComplex(data, params);
    auto prof = fft.GetProfilingData();

    bool pass = (prof.total_time_ms > 0.0) && (prof.fft_time_ms > 0.0);

    std::cout << "    Upload: " << prof.upload_time_ms << " ms\n";
    std::cout << "    FFT: " << prof.fft_time_ms << " ms\n";
    std::cout << "    Download: " << prof.download_time_ms << " ms\n";
    std::cout << "    Total: " << prof.total_time_ms << " ms\n";
    std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Runner
// ════════════════════════════════════════════════════════════════════════════

inline int run() {
    std::cout << "\n════════════════════════════════════════════════════════════\n";
    std::cout << "  FFTProcessor Tests\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    try {
        // Создаём OpenCL контекст
        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clGetPlatformIDs failed: " + std::to_string(err));

        cl_device_id device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clGetDeviceIDs failed: " + std::to_string(err));

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateContext failed: " + std::to_string(err));

#ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
#else
        cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
        if (err != CL_SUCCESS) {
            clReleaseContext(context);
            throw std::runtime_error("clCreateCommandQueue failed: " + std::to_string(err));
        }

        char dev_name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
        std::cout << "  GPU: " << dev_name << "\n";

        // Создаём DrvGPU backend
        auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
        backend->InitializeFromExternalContext(context, device, queue);

        // Запуск тестов
        int passed = 0;
        int total = 0;

        auto runTest = [&](bool result) {
            total++;
            if (result) passed++;
        };

        runTest(TestSingleBeamComplex(backend.get()));
        runTest(TestMultiBeamMagPhaseFreq(backend.get()));
        runTest(TestMagPhaseVsComplex(backend.get()));
        runTest(TestProfiling(backend.get()));

        std::cout << "\n════════════════════════════════════════════════════════════\n";
        std::cout << "  Results: " << passed << "/" << total << " tests passed\n";
        std::cout << "════════════════════════════════════════════════════════════\n\n";

        // Cleanup
        backend.reset();
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return (passed == total) ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "  FATAL: " << e.what() << "\n";
        return 1;
    }
}

} // namespace test_fft_processor
