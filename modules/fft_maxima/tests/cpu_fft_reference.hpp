#pragma once
/**
 * @file cpu_fft_reference.hpp
 * @brief CPU референсная реализация FFT + поиск максимума для валидации GPU результатов
 *
 * Использует PocketFFT (header-only) для вычисления спектра на CPU
 * и тот же алгоритм поиска максимума с параболической интерполяцией,
 * что и GPU kernel в fft_kernel_sources.hpp
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-08
 */

#include "../../../third_party/pocketfft/pocketfft_hdronly.h"
#include "../include/interface/antenna_fft_params.h"

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cpu_reference {

using namespace antenna_fft;

// ════════════════════════════════════════════════════════════════════════════
// Структура для CPU результата (аналог GPU MaxValue)
// ════════════════════════════════════════════════════════════════════════════

struct CPUMaxValue {
    uint32_t index;
    float real;
    float imag;
    float magnitude;
    float phase;
    float freq_offset;
    float refined_frequency;
};

struct CPUSpectrumResult {
    // ЛЕВЫЙ СПЕКТР (4 структуры)
    CPUMaxValue left_interpolated;   // [0] - левый перегиб (параболическая интерполяция)
    CPUMaxValue left_minus1;         // [1] - левая точка ind_l-1
    CPUMaxValue left_center;         // [2] - левая центральная точка (максимум ind_l)
    CPUMaxValue left_plus1;          // [3] - левая правая точка ind_l+1

    // ПРАВЫЙ СПЕКТР (4 структуры)
    CPUMaxValue right_interpolated;  // [4] - правый перегиб (параболическая интерполяция)
    CPUMaxValue right_minus1;        // [5] - правая точка ind_r-1
    CPUMaxValue right_center;        // [6] - правая центральная точка (максимум ind_r)
    CPUMaxValue right_plus1;         // [7] - правая правая точка ind_r+1
};

// ════════════════════════════════════════════════════════════════════════════
// CPU FFT через PocketFFT
// ════════════════════════════════════════════════════════════════════════════

inline void ComputeFFT_CPU(
    const std::vector<std::complex<float>>& input,
    std::vector<std::complex<float>>& output,
    size_t nFFT)
{
    if (input.size() < nFFT) {
        throw std::runtime_error("ComputeFFT_CPU: input size < nFFT");
    }

    output.resize(nFFT);

    // PocketFFT параметры
    pocketfft::shape_t shape{nFFT};
    pocketfft::shape_t axes{0};
    pocketfft::stride_t stride_in{sizeof(std::complex<float>)};
    pocketfft::stride_t stride_out{sizeof(std::complex<float>)};

    // Forward FFT (без нормализации, как в GPU)
    pocketfft::c2c(
        shape,
        stride_in, stride_out,
        axes,
        pocketfft::FORWARD,
        input.data(),
        output.data(),
        1.0f,  // no normalization
        1      // single thread
    );
}

// ════════════════════════════════════════════════════════════════════════════
// Поиск ДВУХ максимумов в краевых диапазонах + параболическая интерполяция
// ПРАВИЛЬНЫЙ алгоритм: ищем ОТДЕЛЬНО левый максимум и правый максимум!
// ════════════════════════════════════════════════════════════════════════════

inline CPUSpectrumResult FindMaximumWithInterpolation(
    const std::vector<std::complex<float>>& fft_output,
    uint32_t nFFT,
    uint32_t search_range,
    float sample_rate)
{
    CPUSpectrumResult result{};

    // ШАГ 1: Вычисляем half_range
    uint32_t half_range = search_range / 2;
    uint32_t range2_start = nFFT - half_range;
    float bin_width = sample_rate / static_cast<float>(nFFT);

    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 2: Ищем ЛЕВЫЙ максимум в диапазоне [0, half_range]
    // ════════════════════════════════════════════════════════════════════════

    float left_max_mag = -1.0f;
    uint32_t left_max_idx = 0;

    for (uint32_t i = 0; i < half_range; ++i) {
        float mag = std::abs(fft_output[i]);
        if (mag > left_max_mag) {
            left_max_mag = mag;
            left_max_idx = i;
        }
    }

    // Читаем 3 точки вокруг левого максимума
    uint32_t ind_l = left_max_idx;
    std::complex<float> l_center_val = fft_output[ind_l];
    float y_l_center = std::abs(l_center_val);

    std::complex<float> l_minus1_val{0.0f, 0.0f};
    float y_l_minus1 = 0.0f;
    bool has_l_minus1 = false;
    if (ind_l > 0 && (ind_l - 1) < half_range) {
        l_minus1_val = fft_output[ind_l - 1];
        y_l_minus1 = std::abs(l_minus1_val);
        has_l_minus1 = true;
    }

    std::complex<float> l_plus1_val{0.0f, 0.0f};
    float y_l_plus1 = 0.0f;
    bool has_l_plus1 = false;
    if (ind_l < nFFT - 1 && (ind_l + 1) < half_range) {
        l_plus1_val = fft_output[ind_l + 1];
        y_l_plus1 = std::abs(l_plus1_val);
        has_l_plus1 = true;
    }

    // Параболическая интерполяция для левого максимума
    float l_freq_offset = 0.0f;
    float l_refined_frequency = static_cast<float>(ind_l) * bin_width;

    if (has_l_minus1 && has_l_plus1) {
        float denom = y_l_minus1 - 2.0f * y_l_center + y_l_plus1;
        if (std::abs(denom) > 1e-10f) {
            float offset = 0.5f * (y_l_minus1 - y_l_plus1) / denom;
            offset = std::clamp(offset, -0.5f, 0.5f);
            l_freq_offset = offset;
            l_refined_frequency = (static_cast<float>(ind_l) + offset) * bin_width;
        }
    }

    // [0] - Левый перегиб (интерполяция)
    result.left_interpolated.index = ind_l;
    result.left_interpolated.real = l_center_val.real();
    result.left_interpolated.imag = l_center_val.imag();
    result.left_interpolated.magnitude = y_l_center;
    result.left_interpolated.phase = std::atan2(l_center_val.imag(), l_center_val.real()) * 57.29577951f;
    result.left_interpolated.freq_offset = l_freq_offset;
    result.left_interpolated.refined_frequency = l_refined_frequency;

    // [1] - ind_l-1
    if (has_l_minus1) {
        result.left_minus1.index = ind_l - 1;
        result.left_minus1.real = l_minus1_val.real();
        result.left_minus1.imag = l_minus1_val.imag();
        result.left_minus1.magnitude = y_l_minus1;
        result.left_minus1.phase = std::atan2(l_minus1_val.imag(), l_minus1_val.real()) * 57.29577951f;
        result.left_minus1.freq_offset = 0.0f;
        result.left_minus1.refined_frequency = static_cast<float>(ind_l - 1) * bin_width;
    } else {
        result.left_minus1 = CPUMaxValue{};
    }

    // [2] - ind_l (центр левого максимума)
    result.left_center.index = ind_l;
    result.left_center.real = l_center_val.real();
    result.left_center.imag = l_center_val.imag();
    result.left_center.magnitude = y_l_center;
    result.left_center.phase = std::atan2(l_center_val.imag(), l_center_val.real()) * 57.29577951f;
    result.left_center.freq_offset = 0.0f;
    result.left_center.refined_frequency = static_cast<float>(ind_l) * bin_width;

    // [3] - ind_l+1
    if (has_l_plus1) {
        result.left_plus1.index = ind_l + 1;
        result.left_plus1.real = l_plus1_val.real();
        result.left_plus1.imag = l_plus1_val.imag();
        result.left_plus1.magnitude = y_l_plus1;
        result.left_plus1.phase = std::atan2(l_plus1_val.imag(), l_plus1_val.real()) * 57.29577951f;
        result.left_plus1.freq_offset = 0.0f;
        result.left_plus1.refined_frequency = static_cast<float>(ind_l + 1) * bin_width;
    } else {
        result.left_plus1 = CPUMaxValue{};
    }

    // ════════════════════════════════════════════════════════════════════════
    // ШАГ 3: Ищем ПРАВЫЙ максимум в диапазоне [nFFT - half_range, nFFT]
    // ════════════════════════════════════════════════════════════════════════

    float right_max_mag = -1.0f;
    uint32_t right_max_idx = range2_start;

    for (uint32_t i = range2_start; i < nFFT; ++i) {
        float mag = std::abs(fft_output[i]);
        if (mag > right_max_mag) {
            right_max_mag = mag;
            right_max_idx = i;
        }
    }

    // Читаем 3 точки вокруг правого максимума
    uint32_t ind_r = right_max_idx;
    std::complex<float> r_center_val = fft_output[ind_r];
    float y_r_center = std::abs(r_center_val);

    std::complex<float> r_minus1_val{0.0f, 0.0f};
    float y_r_minus1 = 0.0f;
    bool has_r_minus1 = false;
    if (ind_r > 0 && (ind_r - 1) >= range2_start) {
        r_minus1_val = fft_output[ind_r - 1];
        y_r_minus1 = std::abs(r_minus1_val);
        has_r_minus1 = true;
    }

    std::complex<float> r_plus1_val{0.0f, 0.0f};
    float y_r_plus1 = 0.0f;
    bool has_r_plus1 = false;
    if (ind_r < nFFT - 1) {
        r_plus1_val = fft_output[ind_r + 1];
        y_r_plus1 = std::abs(r_plus1_val);
        has_r_plus1 = true;
    }

    // Параболическая интерполяция для правого максимума
    float r_freq_offset = 0.0f;
    float r_refined_frequency = static_cast<float>(ind_r) * bin_width;

    if (has_r_minus1 && has_r_plus1) {
        float denom = y_r_minus1 - 2.0f * y_r_center + y_r_plus1;
        if (std::abs(denom) > 1e-10f) {
            float offset = 0.5f * (y_r_minus1 - y_r_plus1) / denom;
            offset = std::clamp(offset, -0.5f, 0.5f);
            r_freq_offset = offset;
            r_refined_frequency = (static_cast<float>(ind_r) + offset) * bin_width;
        }
    }

    // [4] - Правый перегиб (интерполяция)
    result.right_interpolated.index = ind_r;
    result.right_interpolated.real = r_center_val.real();
    result.right_interpolated.imag = r_center_val.imag();
    result.right_interpolated.magnitude = y_r_center;
    result.right_interpolated.phase = std::atan2(r_center_val.imag(), r_center_val.real()) * 57.29577951f;
    result.right_interpolated.freq_offset = r_freq_offset;
    result.right_interpolated.refined_frequency = r_refined_frequency;

    // [5] - ind_r-1
    if (has_r_minus1) {
        result.right_minus1.index = ind_r - 1;
        result.right_minus1.real = r_minus1_val.real();
        result.right_minus1.imag = r_minus1_val.imag();
        result.right_minus1.magnitude = y_r_minus1;
        result.right_minus1.phase = std::atan2(r_minus1_val.imag(), r_minus1_val.real()) * 57.29577951f;
        result.right_minus1.freq_offset = 0.0f;
        result.right_minus1.refined_frequency = static_cast<float>(ind_r - 1) * bin_width;
    } else {
        result.right_minus1 = CPUMaxValue{};
    }

    // [6] - ind_r (центр правого максимума)
    result.right_center.index = ind_r;
    result.right_center.real = r_center_val.real();
    result.right_center.imag = r_center_val.imag();
    result.right_center.magnitude = y_r_center;
    result.right_center.phase = std::atan2(r_center_val.imag(), r_center_val.real()) * 57.29577951f;
    result.right_center.freq_offset = 0.0f;
    result.right_center.refined_frequency = static_cast<float>(ind_r) * bin_width;

    // [7] - ind_r+1
    if (has_r_plus1) {
        result.right_plus1.index = ind_r + 1;
        result.right_plus1.real = r_plus1_val.real();
        result.right_plus1.imag = r_plus1_val.imag();
        result.right_plus1.magnitude = y_r_plus1;
        result.right_plus1.phase = std::atan2(r_plus1_val.imag(), r_plus1_val.real()) * 57.29577951f;
        result.right_plus1.freq_offset = 0.0f;
        result.right_plus1.refined_frequency = static_cast<float>(ind_r + 1) * bin_width;
    } else {
        result.right_plus1 = CPUMaxValue{};
    }

    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Главная функция: CPU референс для одного луча (антенны)
// ════════════════════════════════════════════════════════════════════════════

inline CPUSpectrumResult ProcessSingleBeam_CPU(
    const std::vector<std::complex<float>>& input,
    uint32_t n_point,
    uint32_t nFFT,
    uint32_t search_range,
    float sample_rate,
    bool print_spectrum = false)
{
    if (input.size() < n_point) {
        throw std::runtime_error("ProcessSingleBeam_CPU: input size < n_point");
    }

    // 1. Padding: n_point → nFFT
    std::vector<std::complex<float>> padded_input(nFFT, {0.0f, 0.0f});
    std::copy(input.begin(), input.begin() + n_point, padded_input.begin());

    // 2. FFT
    std::vector<std::complex<float>> fft_output;
    ComputeFFT_CPU(padded_input, fft_output, nFFT);

    // 📊 Вывод спектра (если запрошен)
    if (print_spectrum) {
        uint32_t half_range = search_range / 2;
        uint32_t range2_start = nFFT - half_range;
        float bin_width = sample_rate / static_cast<float>(nFFT);

        std::cout << "\n  📊 CPU СПЕКТР (первые 10 точек левого диапазона):\n";
        for (uint32_t i = 0; i < std::min(10u, half_range); ++i) {
            float mag = std::abs(fft_output[i]);
            float freq = i * bin_width;
            std::cout << "    bin[" << std::setw(4) << i << "] = "
                      << std::fixed << std::setprecision(2) << std::setw(8) << mag
                      << "  @ " << std::setw(7) << std::setprecision(3) << freq << " Hz\n";
        }

        std::cout << "\n  📊 CPU СПЕКТР (последние 10 точек правого диапазона):\n";
        for (uint32_t i = std::max(range2_start, nFFT - 10); i < nFFT; ++i) {
            float mag = std::abs(fft_output[i]);
            float freq = i * bin_width;
            std::cout << "    bin[" << std::setw(4) << i << "] = "
                      << std::fixed << std::setprecision(2) << std::setw(8) << mag
                      << "  @ " << std::setw(7) << std::setprecision(3) << freq << " Hz\n";
        }
        std::cout << "\n";
    }

    // 3. Поиск максимума + параболическая интерполяция
    return FindMaximumWithInterpolation(fft_output, nFFT, search_range, sample_rate);
}

// ════════════════════════════════════════════════════════════════════════════
// Обработка всех лучей (антенн) на CPU
// ════════════════════════════════════════════════════════════════════════════

inline std::vector<CPUSpectrumResult> ProcessAllBeams_CPU(
    const std::vector<std::complex<float>>& input,
    uint32_t antenna_count,
    uint32_t n_point,
    uint32_t nFFT,
    uint32_t search_range,
    float sample_rate,
    bool print_spectrum = false)
{
    std::vector<CPUSpectrumResult> results;
    results.reserve(antenna_count);

    for (uint32_t antenna = 0; antenna < antenna_count; ++antenna) {
        // Извлекаем данные для одной антенны
        std::vector<std::complex<float>> beam_data(
            input.begin() + antenna * n_point,
            input.begin() + (antenna + 1) * n_point
        );

        // Вывод спектра только для первой антенны
        bool print_this = print_spectrum && (antenna == 0);
        if (print_this) {
            std::cout << "\n🔍 АНТЕНА 0 - CPU FFT СПЕКТР:\n";
        }

        // Обрабатываем луч на CPU
        auto result = ProcessSingleBeam_CPU(beam_data, n_point, nFFT, search_range, sample_rate, print_this);
        results.push_back(result);
    }

    return results;
}

} // namespace cpu_reference
