#pragma once

/**
 * @file test_signal_generators.hpp
 * @brief Тесты модуля Signal Generators
 *
 * 1. CW: GPU vs CPU comparison
 * 2. CW multi-beam: freq_step
 * 3. LFM: GPU vs CPU comparison, chirp rate check
 * 4. Noise: statistical test (mean ~ 0, variance ~ power)
 * 5. Integration: CW -> FFTProcessor -> check frequency
 * 6. Factory: create by SignalKind
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-13
 */

#include "signal_service.hpp"
#include "signal_generator_factory.hpp"
#include "generators/cw_generator.hpp"
#include "generators/lfm_generator.hpp"
#include "generators/noise_generator.hpp"
#if ENABLE_CLFFT
#include "fft_processor.hpp"
#endif
#include "DrvGPU/backends/opencl/opencl_backend.hpp"

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_signal_generators {

// ════════════════════════════════════════════════════════════════════════════
// Утилиты
// ════════════════════════════════════════════════════════════════════════════

inline float MaxError(const std::complex<float>* a, const std::complex<float>* b, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::abs(a[i] - b[i]);
        if (d > max_err) max_err = d;
    }
    return max_err;
}

inline size_t FindPeakBin(const std::vector<std::complex<float>>& spectrum, size_t half) {
    size_t peak = 0;
    float peak_val = 0.0f;
    for (size_t i = 0; i < half && i < spectrum.size(); ++i) {
        float m = std::abs(spectrum[i]);
        if (m > peak_val) { peak_val = m; peak = i; }
    }
    return peak;
}

// ════════════════════════════════════════════════════════════════════════════
// Тесты
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Test 1: CW GPU vs CPU
 */
inline bool TestCwGpuVsCpu(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen 1] CW: GPU vs CPU...\n";

    signal_gen::CwParams cw;
    cw.f0 = 250.0;
    cw.amplitude = 1.5;
    cw.phase = 0.3;

    signal_gen::SystemSampling sys{4000.0, 4096};

    // CPU
    signal_gen::CwGenerator gen(backend, cw);
    std::vector<std::complex<float>> cpu_data(sys.length);
    gen.GenerateToCpu(sys, cpu_data.data(), cpu_data.size());

    // GPU -> read back
    cl_mem gpu_buf = gen.GenerateToGpu(sys, 1);
    std::vector<std::complex<float>> gpu_data(sys.length);
    cl_command_queue q = static_cast<cl_command_queue>(backend->GetNativeQueue());
    clEnqueueReadBuffer(q, gpu_buf, CL_TRUE, 0,
                        sys.length * sizeof(std::complex<float>),
                        gpu_data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(gpu_buf);

    float max_err = MaxError(gpu_data.data(), cpu_data.data(), sys.length);
    bool pass = max_err < 1e-3f;  // GPU fast-relaxed-math sin/cos precision

    std::cout << "    Max error: " << std::scientific << std::setprecision(2) << max_err << "\n";
    std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/**
 * @brief Test 2: CW multi-beam with freq_step
 */
inline bool TestCwMultiBeam(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen 2] CW multi-beam (8 beams, freq_step)...\n";

    signal_gen::CwParams cw;
    cw.f0 = 100.0;
    cw.freq_step = 50.0;
    cw.amplitude = 1.0;

    signal_gen::SystemSampling sys{2000.0, 2048};
    size_t beam_count = 8;

    signal_gen::CwGenerator gen(backend, cw);
    cl_mem gpu_buf = gen.GenerateToGpu(sys, beam_count);

    // Read back all beams
    size_t total = beam_count * sys.length;
    std::vector<std::complex<float>> gpu_data(total);
    cl_command_queue q = static_cast<cl_command_queue>(backend->GetNativeQueue());
    clEnqueueReadBuffer(q, gpu_buf, CL_TRUE, 0,
                        total * sizeof(std::complex<float>),
                        gpu_data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(gpu_buf);

    // Verify each beam frequency vs CPU
    bool all_pass = true;
    for (size_t b = 0; b < beam_count; ++b) {
        float expected_freq = static_cast<float>(cw.f0 + b * cw.freq_step);
        float fs = static_cast<float>(sys.fs);

        // CPU reference for this beam
        std::vector<std::complex<float>> cpu_beam(sys.length);
        for (size_t i = 0; i < sys.length; ++i) {
            float t = static_cast<float>(i) / fs;
            float phase = 2.0f * static_cast<float>(M_PI) * expected_freq * t;
            cpu_beam[i] = {std::cos(phase), std::sin(phase)};
        }

        float err = MaxError(gpu_data.data() + b * sys.length, cpu_beam.data(), sys.length);
        bool pass = err < 1e-3f;  // GPU fast-relaxed-math
        if (!pass) all_pass = false;

        std::cout << "    Beam " << b << " (f=" << std::setprecision(0)
                  << expected_freq << " Hz): err=" << std::scientific
                  << std::setprecision(2) << err << std::fixed
                  << " " << (pass ? "OK" : "FAIL") << "\n";
    }

    std::cout << "    " << (all_pass ? "PASS" : "FAIL") << "\n";
    return all_pass;
}

/**
 * @brief Test 3: LFM GPU vs CPU
 */
inline bool TestLfmGpuVsCpu(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen 3] LFM: GPU vs CPU...\n";

    signal_gen::LfmParams lfm;
    lfm.f_start = 100.0;
    lfm.f_end = 500.0;

    signal_gen::SystemSampling sys{4000.0, 4096};

    signal_gen::LfmGenerator gen(backend, lfm);

    // CPU
    std::vector<std::complex<float>> cpu_data(sys.length);
    gen.GenerateToCpu(sys, cpu_data.data(), cpu_data.size());

    // GPU
    cl_mem gpu_buf = gen.GenerateToGpu(sys, 1);
    std::vector<std::complex<float>> gpu_data(sys.length);
    cl_command_queue q = static_cast<cl_command_queue>(backend->GetNativeQueue());
    clEnqueueReadBuffer(q, gpu_buf, CL_TRUE, 0,
                        sys.length * sizeof(std::complex<float>),
                        gpu_data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(gpu_buf);

    float max_err = MaxError(gpu_data.data(), cpu_data.data(), sys.length);
    bool pass = max_err < 1e-3f;  // GPU fast-relaxed-math sin/cos precision

    std::cout << "    Chirp: " << lfm.f_start << " -> " << lfm.f_end << " Hz\n";
    std::cout << "    Max error: " << std::scientific << std::setprecision(2) << max_err << "\n";
    std::cout << std::fixed << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/**
 * @brief Test 4: Noise statistical check
 */
inline bool TestNoiseStatistics(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen 4] Noise: statistical check...\n";

    signal_gen::NoiseParams noise;
    noise.type = signal_gen::NoiseType::GAUSSIAN;
    noise.power = 2.0;
    noise.seed = 42;

    signal_gen::SystemSampling sys{1000.0, 100000};

    signal_gen::NoiseGenerator gen(backend, noise);
    cl_mem gpu_buf = gen.GenerateToGpu(sys, 1);

    std::vector<std::complex<float>> data(sys.length);
    cl_command_queue q = static_cast<cl_command_queue>(backend->GetNativeQueue());
    clEnqueueReadBuffer(q, gpu_buf, CL_TRUE, 0,
                        sys.length * sizeof(std::complex<float>),
                        data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(gpu_buf);

    // Mean
    double sum_re = 0, sum_im = 0;
    for (auto& d : data) { sum_re += d.real(); sum_im += d.imag(); }
    double mean_re = sum_re / data.size();
    double mean_im = sum_im / data.size();

    // Variance
    double var_re = 0, var_im = 0;
    for (auto& d : data) {
        var_re += (d.real() - mean_re) * (d.real() - mean_re);
        var_im += (d.imag() - mean_im) * (d.imag() - mean_im);
    }
    var_re /= data.size();
    var_im /= data.size();

    // Expected: mean ~ 0, variance ~ power
    bool mean_ok = (std::abs(mean_re) < 0.1) && (std::abs(mean_im) < 0.1);
    // Box-Muller with std_dev=sqrt(power) gives variance = power
    bool var_ok = (std::abs(var_re - noise.power) < 0.2) && (std::abs(var_im - noise.power) < 0.2);

    bool pass = mean_ok && var_ok;

    std::cout << "    N=" << sys.length << ", expected power=" << noise.power << "\n";
    std::cout << "    Mean: re=" << std::setprecision(4) << mean_re
              << " im=" << mean_im << " (expected ~0)\n";
    std::cout << "    Variance: re=" << var_re << " im=" << var_im
              << " (expected ~" << noise.power << ")\n";
    std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/**
 * @brief Test 5: Integration — CW -> FFTProcessor -> check frequency
 * Only available when clFFT is enabled (ENABLE_CLFFT=1).
 */
#if ENABLE_CLFFT
inline bool TestCwFftIntegration(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen 5] Integration: CW -> FFT -> freq check...\n";

    const double f0 = 300.0;
    const double fs = 4000.0;
    const size_t n_point = 4096;

    // Generate CW on GPU
    signal_gen::SignalService service(backend);
    signal_gen::CwParams cw;
    cw.f0 = f0;

    cl_mem gpu_data = service.GenerateGpu(cw, {fs, n_point}, 1);

    // Read to CPU for FFTProcessor
    std::vector<std::complex<float>> cpu_data(n_point);
    cl_command_queue q = static_cast<cl_command_queue>(backend->GetNativeQueue());
    clEnqueueReadBuffer(q, gpu_data, CL_TRUE, 0,
                        n_point * sizeof(std::complex<float>),
                        cpu_data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(gpu_data);

    // FFT
    fft_processor::FFTProcessor fft(backend);
    fft_processor::FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = static_cast<uint32_t>(n_point);
    params.sample_rate = static_cast<float>(fs);

    auto results = fft.ProcessComplex(cpu_data, params);

    uint32_t nFFT = results[0].nFFT;
    size_t peak_bin = FindPeakBin(results[0].spectrum, nFFT / 2);
    float detected_freq = static_cast<float>(peak_bin) * static_cast<float>(fs) / nFFT;
    float error = std::abs(detected_freq - static_cast<float>(f0));
    float tolerance = static_cast<float>(fs) / nFFT;  // 1 bin

    bool pass = error < tolerance;

    std::cout << "    CW f0=" << f0 << " Hz, nFFT=" << nFFT << "\n";
    std::cout << "    Detected: " << detected_freq << " Hz, error=" << error << " Hz\n";
    std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}
#endif  // ENABLE_CLFFT

/**
 * @brief Test 6: Factory creates correct types
 */
inline bool TestFactory(drv_gpu_lib::IBackend* backend) {
    std::cout << "\n  [SigGen 6] Factory: create by SignalKind...\n";

    auto cw = signal_gen::SignalGeneratorFactory::CreateCw(backend, signal_gen::CwParams{});
    auto lfm = signal_gen::SignalGeneratorFactory::CreateLfm(backend, signal_gen::LfmParams{});
    auto noise = signal_gen::SignalGeneratorFactory::CreateNoise(backend, signal_gen::NoiseParams{});

    bool pass = (cw->Kind() == signal_gen::SignalKind::CW)
             && (lfm->Kind() == signal_gen::SignalKind::LFM)
             && (noise->Kind() == signal_gen::SignalKind::NOISE);

    // Test via SignalRequest
    signal_gen::SignalRequest req;
    req.kind = signal_gen::SignalKind::CW;
    req.params = signal_gen::CwParams{};
    auto gen = signal_gen::SignalGeneratorFactory::Create(backend, req);
    pass = pass && (gen->Kind() == signal_gen::SignalKind::CW);

    std::cout << "    CW: " << (cw->Kind() == signal_gen::SignalKind::CW ? "OK" : "FAIL") << "\n";
    std::cout << "    LFM: " << (lfm->Kind() == signal_gen::SignalKind::LFM ? "OK" : "FAIL") << "\n";
    std::cout << "    Noise: " << (noise->Kind() == signal_gen::SignalKind::NOISE ? "OK" : "FAIL") << "\n";
    std::cout << "    " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// Runner
// ════════════════════════════════════════════════════════════════════════════

inline int run() {
    std::cout << "\n════════════════════════════════════════════════════════════\n";
    std::cout << "  Signal Generators Tests\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    try {
        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clGetPlatformIDs failed");

        cl_device_id device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clGetDeviceIDs failed");

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateContext failed");

#ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
#else
        cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
        if (err != CL_SUCCESS) {
            clReleaseContext(context);
            throw std::runtime_error("clCreateCommandQueue failed");
        }

        char dev_name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
        std::cout << "  GPU: " << dev_name << "\n";

        auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
        backend->InitializeFromExternalContext(context, device, queue);

        int passed = 0, total = 0;
        auto runTest = [&](bool result) { total++; if (result) passed++; };

        runTest(TestCwGpuVsCpu(backend.get()));
        runTest(TestCwMultiBeam(backend.get()));
        runTest(TestLfmGpuVsCpu(backend.get()));
        runTest(TestNoiseStatistics(backend.get()));
#if ENABLE_CLFFT
        runTest(TestCwFftIntegration(backend.get()));
#endif
        runTest(TestFactory(backend.get()));

        std::cout << "\n════════════════════════════════════════════════════════════\n";
        std::cout << "  SigGen Results: " << passed << "/" << total << " tests passed\n";
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

} // namespace test_signal_generators
