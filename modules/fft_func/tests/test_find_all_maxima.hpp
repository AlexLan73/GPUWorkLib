#pragma once
/**
 * @file test_find_all_maxima.hpp
 * @brief Тесты для FindAllMaxima — поиск ВСЕХ локальных максимумов
 *
 * Тесты:
 * 1. Синтетический спектр с известными пиками — FFT данные на GPU (старый API)
 * 2. Несколько лучей (parallel beam-aware scan) — FFT данные на GPU
 * 3. OutputDestination::GPU
 * 4. Full pipeline из CPU данных (InputData<vector>) — новый API
 * 5. Full pipeline из GPU данных (InputData<cl_mem>) — новый API
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-14
 */

#include "spectrum_maxima_finder.h"
#include "drv_gpu.hpp"
#include "common/backend_type.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_find_all_maxima {

using namespace antenna_fft;
using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════════════════
// CPU Reference: поиск всех локальных максимумов (аналог SciPy find_peaks)
// ════════════════════════════════════════════════════════════════════════════
inline std::vector<uint32_t> CpuFindAllMaxima(
    const std::vector<float>& magnitudes,
    uint32_t search_start,
    uint32_t search_end)
{
    std::vector<uint32_t> maxima;
    for (uint32_t i = search_start; i < search_end; ++i) {
        if (magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1]) {
            maxima.push_back(i);
        }
    }
    return maxima;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU FFT (naive DFT для тестовых размеров)
// ════════════════════════════════════════════════════════════════════════════
inline std::vector<std::complex<float>> CpuDFT(
    const std::vector<std::complex<float>>& input, size_t nFFT)
{
    std::vector<std::complex<float>> output(nFFT, {0.0f, 0.0f});
    size_t N = input.size();

    for (size_t k = 0; k < nFFT; ++k) {
        for (size_t n = 0; n < N; ++n) {
            float angle = -2.0f * static_cast<float>(M_PI) * k * n / nFFT;
            output[k] += input[n] * std::complex<float>(std::cos(angle), std::sin(angle));
        }
    }
    return output;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 1: Синтетический спектр с 3 синусоидами → 3 пика (старый API)
// ════════════════════════════════════════════════════════════════════════════
inline bool TestThreePeaks(IBackend* backend) {
    std::cout << "\n═══ Test FindAllMaxima: 3 синусоиды (FFT на GPU) ═══\n";

    const uint32_t n_point = 1024;
    const uint32_t nFFT = 1024;
    const float sample_rate = 1000.0f;

    // Генерируем сигнал с 3 частотами: 50, 120, 200 Hz
    std::vector<float> freqs = {50.0f, 120.0f, 200.0f};
    std::vector<std::complex<float>> signal(n_point);

    for (uint32_t t = 0; t < n_point; ++t) {
        float val = 0.0f;
        for (float f : freqs) {
            val += std::sin(2.0f * static_cast<float>(M_PI) * f * t / sample_rate);
        }
        signal[t] = std::complex<float>(val, 0.0f);
    }

    // CPU DFT для reference
    auto fft_result = CpuDFT(signal, nFFT);

    // Вычисляем magnitudes
    std::vector<float> magnitudes(nFFT);
    for (size_t i = 0; i < nFFT; ++i) {
        magnitudes[i] = std::abs(fft_result[i]);
    }

    // CPU reference: find all maxima
    auto cpu_maxima = CpuFindAllMaxima(magnitudes, 1, nFFT / 2);

    std::cout << "  CPU reference: " << cpu_maxima.size() << " максимумов\n";
    for (auto pos : cpu_maxima) {
        std::cout << "    bin=" << pos << " freq=" << (pos * sample_rate / nFFT)
                  << " Hz, mag=" << magnitudes[pos] << "\n";
    }

    // Загружаем FFT результат на GPU
    cl_context ctx = static_cast<cl_context>(backend->GetNativeContext());
    cl_int err;

    cl_mem gpu_fft = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        nFFT * sizeof(std::complex<float>), fft_result.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "  FAIL: Failed to create GPU buffer: " << err << "\n";
        return false;
    }

    // GPU FindAllMaxima (старый API — FFT данные уже на GPU)
    SpectrumMaximaFinder finder(backend);
    auto result = finder.FindAllMaxima(gpu_fft, 1, nFFT, sample_rate, OutputDestination::CPU);

    clReleaseMemObject(gpu_fft);

    // Проверяем результат
    if (result.beams.size() != 1) {
        std::cerr << "  FAIL: Expected 1 beam, got " << result.beams.size() << "\n";
        return false;
    }

    auto& beam = result.beams[0];
    std::cout << "  GPU result: " << beam.num_maxima << " максимумов\n";
    for (uint32_t i = 0; i < beam.num_maxima && i < 10; ++i) {
        std::cout << "    bin=" << beam.maxima[i].index << " freq=" << beam.maxima[i].refined_frequency
                  << " Hz, mag=" << beam.maxima[i].magnitude << "\n";
    }

    // Проверяем что GPU нашёл все CPU максимумы
    bool all_found = true;
    for (auto cpu_pos : cpu_maxima) {
        bool found = false;
        for (uint32_t i = 0; i < beam.num_maxima; ++i) {
            if (beam.maxima[i].index == cpu_pos) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "  FAIL: CPU max at bin " << cpu_pos << " NOT found by GPU!\n";
            all_found = false;
        }
    }

    // Проверяем что 3 основных пика есть
    std::vector<float> expected_freqs = {50.0f, 120.0f, 200.0f};
    for (float ef : expected_freqs) {
        uint32_t expected_bin = static_cast<uint32_t>(ef * nFFT / sample_rate + 0.5f);
        bool found = false;
        for (uint32_t i = 0; i < beam.num_maxima; ++i) {
            if (std::abs(static_cast<int>(beam.maxima[i].index) - static_cast<int>(expected_bin)) <= 1) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "  FAIL: Expected peak at ~" << ef << " Hz (bin " << expected_bin << ") NOT found!\n";
            all_found = false;
        }
    }

    if (all_found) {
        std::cout << "  PASS: все максимумы найдены корректно\n";
    }

    return all_found;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 2: Несколько лучей (parallel beam-aware scan)
// ════════════════════════════════════════════════════════════════════════════
inline bool TestMultiBeam(IBackend* backend) {
    std::cout << "\n═══ Test FindAllMaxima: Multi-beam (5 лучей, parallel scan) ═══\n";

    const uint32_t beam_count = 5;
    const uint32_t n_point = 512;
    const uint32_t nFFT = 512;
    const float sample_rate = 1000.0f;

    // Каждый луч с разной частотой
    std::vector<float> beam_freqs = {30.0f, 60.0f, 100.0f, 150.0f, 200.0f};
    std::vector<std::complex<float>> all_fft(beam_count * nFFT);

    for (uint32_t b = 0; b < beam_count; ++b) {
        // Генерируем сигнал
        std::vector<std::complex<float>> signal(n_point);
        for (uint32_t t = 0; t < n_point; ++t) {
            float val = std::sin(2.0f * static_cast<float>(M_PI) * beam_freqs[b] * t / sample_rate);
            signal[t] = std::complex<float>(val, 0.0f);
        }

        // CPU DFT
        auto fft = CpuDFT(signal, nFFT);
        std::copy(fft.begin(), fft.end(), all_fft.begin() + b * nFFT);
    }

    // Загружаем на GPU
    cl_context ctx = static_cast<cl_context>(backend->GetNativeContext());
    cl_int err;

    cl_mem gpu_fft = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        all_fft.size() * sizeof(std::complex<float>), all_fft.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "  FAIL: Failed to create GPU buffer: " << err << "\n";
        return false;
    }

    // GPU FindAllMaxima
    SpectrumMaximaFinder finder(backend);
    auto result = finder.FindAllMaxima(gpu_fft, beam_count, nFFT, sample_rate);

    clReleaseMemObject(gpu_fft);

    // Проверяем
    if (result.beams.size() != beam_count) {
        std::cerr << "  FAIL: Expected " << beam_count << " beams, got " << result.beams.size() << "\n";
        return false;
    }

    bool all_ok = true;
    for (uint32_t b = 0; b < beam_count; ++b) {
        auto& beam = result.beams[b];
        uint32_t expected_bin = static_cast<uint32_t>(beam_freqs[b] * nFFT / sample_rate + 0.5f);

        bool found = false;
        for (uint32_t i = 0; i < beam.num_maxima; ++i) {
            if (std::abs(static_cast<int>(beam.maxima[i].index) - static_cast<int>(expected_bin)) <= 1) {
                found = true;
                break;
            }
        }

        std::cout << "  Beam " << b << " (" << beam_freqs[b] << " Hz): "
                  << beam.num_maxima << " maxima, expected_bin=" << expected_bin
                  << " -> " << (found ? "PASS" : "FAIL") << "\n";

        if (!found) all_ok = false;
    }

    if (all_ok)
        std::cout << "  PASS: все лучи обработаны корректно\n";

    return all_ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 3: OutputDestination::GPU
// ════════════════════════════════════════════════════════════════════════════
inline bool TestGpuOutput(IBackend* backend) {
    std::cout << "\n═══ Test FindAllMaxima: OutputDestination::GPU ═══\n";

    const uint32_t n_point = 256;
    const uint32_t nFFT = 256;
    const float sample_rate = 1000.0f;

    // Простой сигнал с одной частотой
    std::vector<std::complex<float>> signal(n_point);
    for (uint32_t t = 0; t < n_point; ++t) {
        float val = std::sin(2.0f * static_cast<float>(M_PI) * 100.0f * t / sample_rate);
        signal[t] = std::complex<float>(val, 0.0f);
    }

    auto fft = CpuDFT(signal, nFFT);

    cl_context ctx = static_cast<cl_context>(backend->GetNativeContext());
    cl_int err;

    cl_mem gpu_fft = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        nFFT * sizeof(std::complex<float>), fft.data(), &err);

    SpectrumMaximaFinder finder(backend);
    auto result = finder.FindAllMaxima(gpu_fft, 1, nFFT, sample_rate, OutputDestination::GPU);

    clReleaseMemObject(gpu_fft);

    bool ok = (result.gpu_maxima != nullptr) && (result.gpu_counts != nullptr) && (result.total_maxima > 0);

    if (ok) {
        std::cout << "  total_maxima: " << result.total_maxima << "\n";
        std::cout << "  gpu_maxima: " << (result.gpu_maxima ? "OK" : "NULL") << "\n";
        std::cout << "  gpu_counts: " << (result.gpu_counts ? "OK" : "NULL") << "\n";
        std::cout << "  PASS: GPU buffers returned\n";
    } else {
        std::cerr << "  FAIL\n";
    }

    // Освобождаем GPU буферы (наша ответственность)
    if (result.gpu_maxima) clReleaseMemObject(static_cast<cl_mem>(result.gpu_maxima));
    if (result.gpu_counts) clReleaseMemObject(static_cast<cl_mem>(result.gpu_counts));

    return ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 4: Full pipeline из CPU данных (InputData<vector>) — НОВЫЙ API
// ════════════════════════════════════════════════════════════════════════════
inline bool TestFullPipelineCPU(IBackend* backend) {
    std::cout << "\n═══ Test FindAllMaxima: Full Pipeline (CPU data, InputData<T>) ═══\n";

    const uint32_t n_point = 1024;
    const float sample_rate = 1000.0f;
    // repeat_count=1, n_point — степень двойки → nFFT = 1024
    const uint32_t nFFT = 1024;

    // Генерируем сигнал с 3 частотами: 80, 160, 300 Hz
    std::vector<float> freqs = {80.0f, 160.0f, 300.0f};
    std::vector<std::complex<float>> signal(n_point);

    for (uint32_t t = 0; t < n_point; ++t) {
        float val = 0.0f;
        for (float f : freqs) {
            val += std::sin(2.0f * static_cast<float>(M_PI) * f * t / sample_rate);
        }
        signal[t] = std::complex<float>(val, 0.0f);
    }

    // CPU reference: DFT → find maxima
    auto fft_ref = CpuDFT(signal, nFFT);
    std::vector<float> magnitudes(nFFT);
    for (size_t i = 0; i < nFFT; ++i) {
        magnitudes[i] = std::abs(fft_ref[i]);
    }
    auto cpu_maxima = CpuFindAllMaxima(magnitudes, 1, nFFT / 2);

    std::cout << "  CPU reference: " << cpu_maxima.size() << " максимумов\n";

    // GPU: Full pipeline через InputData<vector>
    SpectrumMaximaFinder finder(backend);

    InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = 1;
    input.n_point = n_point;
    input.data = signal;
    input.repeat_count = 1;
    input.sample_rate = sample_rate;

    auto result = finder.FindAllMaxima(input, OutputDestination::CPU);

    if (result.beams.size() != 1) {
        std::cerr << "  FAIL: Expected 1 beam, got " << result.beams.size() << "\n";
        return false;
    }

    auto& beam = result.beams[0];
    std::cout << "  GPU result: " << beam.num_maxima << " максимумов\n";

    // Проверяем основные пики
    bool all_found = true;
    for (float ef : freqs) {
        uint32_t expected_bin = static_cast<uint32_t>(ef * nFFT / sample_rate + 0.5f);
        bool found = false;
        for (uint32_t i = 0; i < beam.num_maxima; ++i) {
            if (std::abs(static_cast<int>(beam.maxima[i].index) - static_cast<int>(expected_bin)) <= 1) {
                found = true;
                break;
            }
        }
        std::cout << "    " << ef << " Hz (bin " << expected_bin << "): "
                  << (found ? "PASS" : "FAIL") << "\n";
        if (!found) all_found = false;
    }

    if (all_found) {
        std::cout << "  PASS: Full pipeline CPU работает\n";
    }
    return all_found;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 5: Full pipeline из GPU данных (InputData<cl_mem>) — НОВЫЙ API
// ════════════════════════════════════════════════════════════════════════════
inline bool TestFullPipelineGPU(IBackend* backend) {
    std::cout << "\n═══ Test FindAllMaxima: Full Pipeline (GPU data, InputData<cl_mem>) ═══\n";

    const uint32_t beam_count = 3;
    const uint32_t n_point = 512;
    const float sample_rate = 1000.0f;
    const uint32_t nFFT = 512;

    // Генерируем сигналы для 3 лучей
    std::vector<float> beam_freqs = {50.0f, 100.0f, 200.0f};
    std::vector<std::complex<float>> all_signals(beam_count * n_point);

    for (uint32_t b = 0; b < beam_count; ++b) {
        for (uint32_t t = 0; t < n_point; ++t) {
            float val = std::sin(2.0f * static_cast<float>(M_PI) * beam_freqs[b] * t / sample_rate);
            all_signals[b * n_point + t] = std::complex<float>(val, 0.0f);
        }
    }

    // Загружаем RAW сигнал на GPU (не FFT!)
    cl_context ctx = static_cast<cl_context>(backend->GetNativeContext());
    cl_int err;

    cl_mem gpu_signal = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        all_signals.size() * sizeof(std::complex<float>), all_signals.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "  FAIL: Failed to create GPU buffer: " << err << "\n";
        return false;
    }

    // GPU: Full pipeline через InputData<cl_mem>
    SpectrumMaximaFinder finder(backend);

    InputData<cl_mem> input;
    input.antenna_count = beam_count;
    input.n_point = n_point;
    input.data = gpu_signal;
    input.gpu_memory_bytes = all_signals.size() * sizeof(std::complex<float>);
    input.repeat_count = 1;
    input.sample_rate = sample_rate;

    auto result = finder.FindAllMaxima(input, OutputDestination::CPU);

    clReleaseMemObject(gpu_signal);

    if (result.beams.size() != beam_count) {
        std::cerr << "  FAIL: Expected " << beam_count << " beams, got " << result.beams.size() << "\n";
        return false;
    }

    bool all_ok = true;
    for (uint32_t b = 0; b < beam_count; ++b) {
        auto& beam = result.beams[b];
        uint32_t expected_bin = static_cast<uint32_t>(beam_freqs[b] * nFFT / sample_rate + 0.5f);

        bool found = false;
        for (uint32_t i = 0; i < beam.num_maxima; ++i) {
            if (std::abs(static_cast<int>(beam.maxima[i].index) - static_cast<int>(expected_bin)) <= 1) {
                found = true;
                break;
            }
        }

        std::cout << "  Beam " << b << " (" << beam_freqs[b] << " Hz): "
                  << beam.num_maxima << " maxima, expected_bin=" << expected_bin
                  << " -> " << (found ? "PASS" : "FAIL") << "\n";

        if (!found) all_ok = false;
    }

    if (all_ok)
        std::cout << "  PASS: Full pipeline GPU работает\n";

    return all_ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 6: AllMaxima<vector> — поиск максимумов в готовых FFT данных (CPU)
// ════════════════════════════════════════════════════════════════════════════
inline bool TestAllMaximaCPU(IBackend* backend) {
    std::cout << "\n═══ Test AllMaxima<vector>: CPU FFT data (без FFT) ═══\n";

    const uint32_t n_point = 1024;
    const uint32_t nFFT = 1024;
    const float sample_rate = 1000.0f;

    // Генерируем сигнал с 3 частотами
    std::vector<float> freqs = {50.0f, 120.0f, 200.0f};
    std::vector<std::complex<float>> signal(n_point);

    for (uint32_t t = 0; t < n_point; ++t) {
        float val = 0.0f;
        for (float f : freqs) {
            val += std::sin(2.0f * static_cast<float>(M_PI) * f * t / sample_rate);
        }
        signal[t] = std::complex<float>(val, 0.0f);
    }

    // CPU DFT → FFT данные
    auto fft_result = CpuDFT(signal, nFFT);

    // Вычисляем CPU reference magnitudes
    std::vector<float> magnitudes(nFFT);
    for (size_t i = 0; i < nFFT; ++i) {
        magnitudes[i] = std::abs(fft_result[i]);
    }
    auto cpu_maxima = CpuFindAllMaxima(magnitudes, 1, nFFT / 2);

    std::cout << "  CPU reference: " << cpu_maxima.size() << " максимумов\n";

    // GPU: AllMaxima<vector> — передаём FFT данные (НЕ сырой сигнал!)
    SpectrumMaximaFinder finder(backend);

    InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = 1;
    input.n_point = nFFT;  // n_point = nFFT (данные уже FFT!)
    input.data = fft_result;
    input.sample_rate = sample_rate;

    auto result = finder.AllMaxima(input, OutputDestination::CPU, DriverType::OPENCL);

    if (result.beams.size() != 1) {
        std::cerr << "  FAIL: Expected 1 beam, got " << result.beams.size() << "\n";
        return false;
    }

    auto& beam = result.beams[0];
    std::cout << "  GPU AllMaxima: " << beam.num_maxima << " максимумов\n";

    // Проверяем основные пики
    bool all_found = true;
    for (float ef : freqs) {
        uint32_t expected_bin = static_cast<uint32_t>(ef * nFFT / sample_rate + 0.5f);
        bool found = false;
        for (uint32_t i = 0; i < beam.num_maxima; ++i) {
            if (std::abs(static_cast<int>(beam.maxima[i].index) - static_cast<int>(expected_bin)) <= 1) {
                found = true;
                break;
            }
        }
        std::cout << "    " << ef << " Hz (bin " << expected_bin << "): "
                  << (found ? "PASS" : "FAIL") << "\n";
        if (!found) all_found = false;
    }

    if (all_found)
        std::cout << "  PASS: AllMaxima<vector> CPU работает\n";
    return all_found;
}

// ════════════════════════════════════════════════════════════════════════════
// Тест 7: AllMaxima<cl_mem> — поиск максимумов в готовых FFT данных (GPU)
// ════════════════════════════════════════════════════════════════════════════
inline bool TestAllMaximaGPU(IBackend* backend) {
    std::cout << "\n═══ Test AllMaxima<cl_mem>: GPU FFT data (без FFT) ═══\n";

    const uint32_t beam_count = 3;
    const uint32_t nFFT = 512;
    const float sample_rate = 1000.0f;

    // Генерируем FFT данные для 3 лучей
    std::vector<float> beam_freqs = {50.0f, 100.0f, 200.0f};
    std::vector<std::complex<float>> all_fft(beam_count * nFFT);

    for (uint32_t b = 0; b < beam_count; ++b) {
        std::vector<std::complex<float>> signal(nFFT);
        for (uint32_t t = 0; t < nFFT; ++t) {
            float val = std::sin(2.0f * static_cast<float>(M_PI) * beam_freqs[b] * t / sample_rate);
            signal[t] = std::complex<float>(val, 0.0f);
        }
        auto fft = CpuDFT(signal, nFFT);
        std::copy(fft.begin(), fft.end(), all_fft.begin() + b * nFFT);
    }

    // Загружаем FFT данные на GPU
    cl_context ctx = static_cast<cl_context>(backend->GetNativeContext());
    cl_int err;
    cl_mem gpu_fft = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        all_fft.size() * sizeof(std::complex<float>), all_fft.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "  FAIL: Failed to create GPU buffer: " << err << "\n";
        return false;
    }

    // GPU: AllMaxima<cl_mem> — FFT данные уже на GPU
    SpectrumMaximaFinder finder(backend);

    InputData<cl_mem> input;
    input.antenna_count = beam_count;
    input.n_point = nFFT;  // n_point = nFFT!
    input.data = gpu_fft;
    input.sample_rate = sample_rate;

    auto result = finder.AllMaxima(input, OutputDestination::CPU, DriverType::OPENCL);
    clReleaseMemObject(gpu_fft);

    if (result.beams.size() != beam_count) {
        std::cerr << "  FAIL: Expected " << beam_count << " beams, got " << result.beams.size() << "\n";
        return false;
    }

    bool all_ok = true;
    for (uint32_t b = 0; b < beam_count; ++b) {
        auto& beam = result.beams[b];
        uint32_t expected_bin = static_cast<uint32_t>(beam_freqs[b] * nFFT / sample_rate + 0.5f);

        bool found = false;
        for (uint32_t i = 0; i < beam.num_maxima; ++i) {
            if (std::abs(static_cast<int>(beam.maxima[i].index) - static_cast<int>(expected_bin)) <= 1) {
                found = true;
                break;
            }
        }
        std::cout << "  Beam " << b << " (" << beam_freqs[b] << " Hz): "
                  << beam.num_maxima << " maxima, expected_bin=" << expected_bin
                  << " -> " << (found ? "PASS" : "FAIL") << "\n";
        if (!found) all_ok = false;
    }

    if (all_ok)
        std::cout << "  PASS: AllMaxima<cl_mem> GPU работает\n";
    return all_ok;
}

// ════════════════════════════════════════════════════════════════════════════
// Run all tests (self-contained — создаёт свой backend)
// ════════════════════════════════════════════════════════════════════════════
inline int run() {
    std::cout << "\n+==================================================+\n";
    std::cout << "|   FindAllMaxima Tests                            |\n";
    std::cout << "+==================================================+\n";

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

        // Старый API (FFT данные на GPU)
        runTest(TestThreePeaks(backend.get()));
        runTest(TestMultiBeam(backend.get()));
        runTest(TestGpuOutput(backend.get()));

        // Новый API: FindAllMaxima<T> (Full pipeline, InputData<T>)
        runTest(TestFullPipelineCPU(backend.get()));
        runTest(TestFullPipelineGPU(backend.get()));

        // Новый API: AllMaxima<T> (FFT данные → detect → scan → compact, БЕЗ FFT)
        runTest(TestAllMaximaCPU(backend.get()));
        runTest(TestAllMaximaGPU(backend.get()));

        std::cout << "\n----------------------------------------\n";
        std::cout << "  FindAllMaxima: " << passed << "/" << total << " tests passed";
        if (passed == total)
            std::cout << " PASS\n";
        else
            std::cout << " FAIL\n";
        std::cout << "----------------------------------------\n\n";

        backend.reset();
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return (passed == total) ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "  FATAL: " << e.what() << "\n";
        return 1;
    }
}

} // namespace test_find_all_maxima
