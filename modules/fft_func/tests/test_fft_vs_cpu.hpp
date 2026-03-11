#pragma once

/**
 * @file test_fft_vs_cpu.hpp
 * @brief GPU FFT (FFTProcessor) vs CPU FFT (pocketfft) — поэлементное сравнение
 *
 * Тесты:
 * 1. Single beam: GPU complex spectrum vs CPU complex spectrum
 * 2. Multi-beam: GPU vs CPU для каждого луча
 * 3. Magnitude/Phase: GPU kernel vs CPU std::abs/std::arg
 * 4. Large FFT: проверка на больших данных (65536 точек)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "fft_processor.hpp"
#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include "../../../third_party/pocketfft/pocketfft_hdronly.h"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_vs_cpu {

// ════════════════════════════════════════════════════════════════════════════
// Утилиты
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief CPU FFT через pocketfft (reference)
 */
inline void ComputeCpuFFT(
    const std::complex<float>* input, size_t n_point,
    std::complex<float>* output, size_t nFFT)
{
    // Zero-padded input
    std::vector<std::complex<float>> padded(nFFT, {0.0f, 0.0f});
    std::copy(input, input + n_point, padded.begin());

    pocketfft::shape_t shape{nFFT};
    pocketfft::shape_t axes{0};
    pocketfft::stride_t stride{sizeof(std::complex<float>)};

    pocketfft::c2c(shape, stride, stride, axes, pocketfft::FORWARD,
                   padded.data(), output, 1.0f, 1);
}

/**
 * @brief Генерировать многочастотный сигнал
 */
inline std::vector<std::complex<float>> GenerateMultiTone(
    float sample_rate, size_t n_point,
    const std::vector<float>& freqs,
    const std::vector<float>& amplitudes)
{
    std::vector<std::complex<float>> data(n_point, {0.0f, 0.0f});
    for (size_t f = 0; f < freqs.size(); ++f) {
        float amp = (f < amplitudes.size()) ? amplitudes[f] : 1.0f;
        for (size_t i = 0; i < n_point; ++i) {
            float t = static_cast<float>(i) / sample_rate;
            float phase = 2.0f * static_cast<float>(M_PI) * freqs[f] * t;
            data[i] += std::complex<float>(amp * std::cos(phase), amp * std::sin(phase));
        }
    }
    return data;
}

/**
 * @brief Вычислить максимальную относительную ошибку спектра
 * @return {max_error, avg_error, rmse}
 */
struct SpectrumError {
    double max_error;
    double avg_error;
    double rmse;
    size_t max_error_bin;
};

inline SpectrumError CompareSpectra(
    const std::complex<float>* gpu, const std::complex<float>* cpu,
    size_t nFFT)
{
    SpectrumError err{0.0, 0.0, 0.0, 0};

    // Найти максимальную амплитуду CPU для нормализации
    float max_amp = 0.0f;
    for (size_t i = 0; i < nFFT; ++i) {
        float a = std::abs(cpu[i]);
        if (a > max_amp) max_amp = a;
    }
    if (max_amp < 1e-10f) max_amp = 1.0f;

    double sum_err = 0.0;
    double sum_sq = 0.0;

    for (size_t i = 0; i < nFFT; ++i) {
        double diff_re = static_cast<double>(gpu[i].real()) - static_cast<double>(cpu[i].real());
        double diff_im = static_cast<double>(gpu[i].imag()) - static_cast<double>(cpu[i].imag());
        double abs_err = std::sqrt(diff_re * diff_re + diff_im * diff_im);
        double rel_err = abs_err / max_amp;

        if (rel_err > err.max_error) {
            err.max_error = rel_err;
            err.max_error_bin = i;
        }
        sum_err += rel_err;
        sum_sq += rel_err * rel_err;
    }

    err.avg_error = sum_err / nFFT;
    err.rmse = std::sqrt(sum_sq / nFFT);
    return err;
}

// ════════════════════════════════════════════════════════════════════════════
// Тесты
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Test 1: Single tone — GPU vs CPU spectrum (element-wise)
 */
inline bool TestSingleToneVsCpu(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [VS-CPU 1] Single tone, element-wise comparison...\n";

    const float freq = 137.5f;
    const float sample_rate = 1000.0f;
    const size_t n_point = 1024;

    auto data = GenerateMultiTone(sample_rate, n_point, {freq}, {1.0f});

    // GPU FFT
    fft_processor::FFTProcessor fft(backend);
    fft_processor::FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = static_cast<uint32_t>(n_point);
    params.sample_rate = sample_rate;
    params.repeat_count = 1;

    auto gpu_results = fft.ProcessComplex(data, params);
    if (gpu_results.empty()) {
        std::cerr << "    FAIL: no GPU results\n";
        return false;
    }

    uint32_t nFFT = gpu_results[0].nFFT;

    // CPU FFT (с таким же padding)
    std::vector<std::complex<float>> cpu_spectrum(nFFT);
    ComputeCpuFFT(data.data(), n_point, cpu_spectrum.data(), nFFT);

    // Сравнение
    auto err = CompareSpectra(gpu_results[0].spectrum.data(), cpu_spectrum.data(), nFFT);

    bool pass = (err.max_error < 1e-4);

    std::cout << "    nFFT: " << nFFT << "\n";
    std::cout << "    Max relative error: " << std::scientific << std::setprecision(2)
              << err.max_error << " (bin " << err.max_error_bin << ")\n";
    std::cout << "    Avg relative error: " << err.avg_error << "\n";
    std::cout << "    RMSE:               " << err.rmse << "\n";
    std::cout << std::fixed;
    std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/**
 * @brief Test 2: Multi-tone signal — проверка что все частоты сохранены
 */
inline bool TestMultiToneVsCpu(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [VS-CPU 2] Multi-tone (3 frequencies)...\n";

    const float sample_rate = 4000.0f;
    const size_t n_point = 2048;
    std::vector<float> freqs = {100.0f, 500.0f, 1200.0f};
    std::vector<float> amps = {1.0f, 0.5f, 0.25f};

    auto data = GenerateMultiTone(sample_rate, n_point, freqs, amps);

    // GPU
    fft_processor::FFTProcessor fft(backend);
    fft_processor::FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = static_cast<uint32_t>(n_point);
    params.sample_rate = sample_rate;

    auto gpu_results = fft.ProcessComplex(data, params);
    uint32_t nFFT = gpu_results[0].nFFT;

    // CPU
    std::vector<std::complex<float>> cpu_spectrum(nFFT);
    ComputeCpuFFT(data.data(), n_point, cpu_spectrum.data(), nFFT);

    auto err = CompareSpectra(gpu_results[0].spectrum.data(), cpu_spectrum.data(), nFFT);

    bool pass = (err.max_error < 1e-4);

    std::cout << "    nFFT: " << nFFT << "\n";
    std::cout << "    Max relative error: " << std::scientific << std::setprecision(2)
              << err.max_error << "\n";
    std::cout << "    RMSE:               " << err.rmse << "\n";
    std::cout << std::fixed;

    // Проверить пики на всех 3 частотах
    float freq_res = sample_rate / nFFT;
    for (size_t f = 0; f < freqs.size(); ++f) {
        size_t expected_bin = static_cast<size_t>(std::round(freqs[f] / freq_res));

        float gpu_mag = std::abs(gpu_results[0].spectrum[expected_bin]);
        float cpu_mag = std::abs(cpu_spectrum[expected_bin]);
        float peak_diff = std::abs(gpu_mag - cpu_mag) / std::max(cpu_mag, 1e-10f);

        std::cout << "    f=" << std::setprecision(0) << freqs[f] << " Hz (bin "
                  << expected_bin << "): GPU=" << std::setprecision(2)
                  << gpu_mag << " CPU=" << cpu_mag
                  << " diff=" << std::scientific << peak_diff << std::fixed << "\n";
    }

    std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/**
 * @brief Test 3: Multi-beam — каждый луч GPU vs CPU
 */
inline bool TestMultiBeamVsCpu(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [VS-CPU 3] Multi-beam (4 beams), per-beam comparison...\n";

    const size_t beam_count = 4;
    const size_t n_point = 512;
    const float sample_rate = 2000.0f;
    const float base_freq = 200.0f;
    const float freq_step = 150.0f;

    // Собрать данные для beam_count лучей
    std::vector<std::complex<float>> all_data(beam_count * n_point);
    for (size_t b = 0; b < beam_count; ++b) {
        float freq = base_freq + b * freq_step;
        for (size_t i = 0; i < n_point; ++i) {
            float t = static_cast<float>(i) / sample_rate;
            float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
            all_data[b * n_point + i] = {std::cos(phase), std::sin(phase)};
        }
    }

    // GPU: все 4 луча за раз
    fft_processor::FFTProcessor fft(backend);
    fft_processor::FFTProcessorParams params;
    params.beam_count = static_cast<uint32_t>(beam_count);
    params.n_point = static_cast<uint32_t>(n_point);
    params.sample_rate = sample_rate;

    auto gpu_results = fft.ProcessComplex(all_data, params);
    if (gpu_results.size() != beam_count) {
        std::cerr << "    FAIL: expected " << beam_count << " results\n";
        return false;
    }

    uint32_t nFFT = gpu_results[0].nFFT;
    bool all_pass = true;
    double worst_error = 0.0;

    for (size_t b = 0; b < beam_count; ++b) {
        // CPU FFT для этого луча
        std::vector<std::complex<float>> cpu_spectrum(nFFT);
        ComputeCpuFFT(all_data.data() + b * n_point, n_point, cpu_spectrum.data(), nFFT);

        auto err = CompareSpectra(gpu_results[b].spectrum.data(), cpu_spectrum.data(), nFFT);
        worst_error = std::max(worst_error, err.max_error);

        bool pass = (err.max_error < 1e-4);
        if (!pass) all_pass = false;

        float freq = base_freq + b * freq_step;
        std::cout << "    Beam " << b << " (f=" << std::setprecision(0) << freq
                  << " Hz): max_err=" << std::scientific << std::setprecision(2)
                  << err.max_error << std::fixed
                  << " " << (pass ? "OK" : "FAIL") << "\n";
    }

    std::cout << "    Worst error: " << std::scientific << std::setprecision(2) << worst_error << std::fixed << "\n";
    std::cout << "    " << (all_pass ? "PASS" : "FAIL") << "\n";
    return all_pass;
}

/**
 * @brief Test 4: Large FFT (65536 points) — stress test accuracy
 */
inline bool TestLargeFFTVsCpu(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [VS-CPU 4] Large FFT (65536 points)...\n";

    const float sample_rate = 48000.0f;
    const size_t n_point = 65536;
    std::vector<float> freqs = {440.0f, 1000.0f, 5000.0f, 12000.0f};
    std::vector<float> amps = {1.0f, 0.7f, 0.3f, 0.1f};

    auto data = GenerateMultiTone(sample_rate, n_point, freqs, amps);

    // GPU
    fft_processor::FFTProcessor fft(backend);
    fft_processor::FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = static_cast<uint32_t>(n_point);
    params.sample_rate = sample_rate;

    auto gpu_results = fft.ProcessComplex(data, params);
    uint32_t nFFT = gpu_results[0].nFFT;

    // CPU
    std::vector<std::complex<float>> cpu_spectrum(nFFT);
    ComputeCpuFFT(data.data(), n_point, cpu_spectrum.data(), nFFT);

    auto err = CompareSpectra(gpu_results[0].spectrum.data(), cpu_spectrum.data(), nFFT);

    // Для больших FFT допустимая ошибка чуть выше из-за float32 accumulation
    bool pass = (err.max_error < 1e-3);

    std::cout << "    nFFT: " << nFFT << "\n";
    std::cout << "    Max relative error: " << std::scientific << std::setprecision(2)
              << err.max_error << " (bin " << err.max_error_bin << ")\n";
    std::cout << "    RMSE:               " << err.rmse << "\n";
    std::cout << std::fixed;

    // Profiling
    auto prof = fft.GetProfilingData();
    std::cout << "    GPU time: upload=" << std::setprecision(2) << prof.upload_time_ms
              << " fft=" << prof.fft_time_ms
              << " download=" << prof.download_time_ms
              << " total=" << prof.total_time_ms << " ms\n";

    std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/**
 * @brief Test 5: MagPhase GPU kernel vs CPU std::abs/std::arg
 */
inline bool TestMagPhaseVsCpuReference(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [VS-CPU 5] Magnitude/Phase: GPU kernel vs CPU...\n";

    const float sample_rate = 2000.0f;
    const size_t n_point = 1024;

    auto data = GenerateMultiTone(sample_rate, n_point, {250.0f, 750.0f}, {1.0f, 0.5f});

    // GPU: MagPhase mode
    fft_processor::FFTProcessor fft(backend);
    fft_processor::FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = static_cast<uint32_t>(n_point);
    params.sample_rate = sample_rate;
    params.output_mode = fft_processor::FFTOutputMode::MAGNITUDE_PHASE;

    auto gpu_mp = fft.ProcessMagPhase(data, params);
    uint32_t nFFT = gpu_mp[0].nFFT;

    // CPU: FFT + manual mag/phase
    std::vector<std::complex<float>> cpu_spectrum(nFFT);
    ComputeCpuFFT(data.data(), n_point, cpu_spectrum.data(), nFFT);

    float max_mag_err = 0.0f;
    float max_phase_err = 0.0f;

    for (uint32_t k = 0; k < nFFT; ++k) {
        float cpu_mag = std::abs(cpu_spectrum[k]);
        float cpu_phase = std::arg(cpu_spectrum[k]);

        float mag_diff = std::abs(gpu_mp[0].magnitude[k] - cpu_mag);
        max_mag_err = std::max(max_mag_err, mag_diff);

        if (cpu_mag > 0.01f) {
            float phase_diff = gpu_mp[0].phase[k] - cpu_phase;
            while (phase_diff > static_cast<float>(M_PI)) phase_diff -= 2.0f * static_cast<float>(M_PI);
            while (phase_diff < -static_cast<float>(M_PI)) phase_diff += 2.0f * static_cast<float>(M_PI);
            max_phase_err = std::max(max_phase_err, std::abs(phase_diff));
        }
    }

    bool pass = (max_mag_err < 1e-3f) && (max_phase_err < 1e-3f);

    std::cout << "    Max magnitude error: " << std::scientific << std::setprecision(2) << max_mag_err << "\n";
    std::cout << "    Max phase error:     " << max_phase_err << " rad\n";
    std::cout << std::fixed;
    std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Runner
// ════════════════════════════════════════════════════════════════════════════

inline int run() {
    std::cout << "\n════════════════════════════════════════════════════════════\n";
    std::cout << "  FFTProcessor vs CPU Reference (pocketfft)\n";
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

        auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
        backend->InitializeFromExternalContext(context, device, queue);

        // Запуск тестов
        int passed = 0;
        int total = 0;

        auto runTest = [&](bool result) {
            total++;
            if (result) passed++;
        };

        runTest(TestSingleToneVsCpu(backend.get()));
        runTest(TestMultiToneVsCpu(backend.get()));
        runTest(TestMultiBeamVsCpu(backend.get()));
        runTest(TestLargeFFTVsCpu(backend.get()));
        runTest(TestMagPhaseVsCpuReference(backend.get()));

        std::cout << "\n════════════════════════════════════════════════════════════\n";
        std::cout << "  VS-CPU Results: " << passed << "/" << total << " tests passed\n";
        std::cout << "════════════════════════════════════════════════════════════\n\n";

        backend.reset();
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return (passed == total) ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "  FATAL: " << e.what() << "\n";
        return 1;
    }
}

} // namespace test_fft_vs_cpu
