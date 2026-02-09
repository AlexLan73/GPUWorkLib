#pragma once
/**
 * @file cpu_fft_reference.hpp
 * @brief CPU референсная реализация FFT + поиск максимума для валидации GPU результатов
 *
 * Использует PocketFFT (header-only) для вычисления спектра на CPU
 * и тот же алгоритм поиска максимума с параболической интерполяцией,
 * что и GPU kernel в fft_kernel_sources.hpp
 *
 * Ref01: использует общие типы из interface/spectrum_maxima_types.h
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-08
 */

#include "../../../third_party/pocketfft/pocketfft_hdronly.h"
#include "interface/spectrum_maxima_types.h"

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cpu_reference {

using namespace antenna_fft;

// ════════════════════════════════════════════════════════════════════════════
// Результат: antenna_fft::CPUSpectrumResult (Ref02 — общий тип для CPU и GPU)
// ════════════════════════════════════════════════════════════════════════════
using CPUSpectrumResult = antenna_fft::CPUSpectrumResult;

// ════════════════════════════════════════════════════════════════════════════
// Вспомогательно: создать MaxValue с pad=0
// ════════════════════════════════════════════════════════════════════════════

inline MaxValue MakeMaxValue(uint32_t idx, float re, float im, float mag, float ph, float fo, float rf) {
    MaxValue m{};
    m.index = idx;
    m.real = re;
    m.imag = im;
    m.magnitude = mag;
    m.phase = ph;
    m.freq_offset = fo;
    m.refined_frequency = rf;
    m.pad = 0;
    return m;
}

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

    pocketfft::shape_t shape{nFFT};
    pocketfft::shape_t axes{0};
    pocketfft::stride_t stride_in{sizeof(std::complex<float>)};
    pocketfft::stride_t stride_out{sizeof(std::complex<float>)};

    pocketfft::c2c(
        shape, stride_in, stride_out, axes, pocketfft::FORWARD,
        input.data(), output.data(), 1.0f, 1);
}

// ════════════════════════════════════════════════════════════════════════════
// Следующая степень двойки
// ════════════════════════════════════════════════════════════════════════════

inline uint32_t NextPowerOf2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// ════════════════════════════════════════════════════════════════════════════
// Поиск ДВУХ максимумов в краевых диапазонах + параболическая интерполяция
// Возвращает CPUSpectrumResult с antenna_id в обоих SpectrumResult
// ════════════════════════════════════════════════════════════════════════════

inline CPUSpectrumResult FindMaximumWithInterpolation(
    const std::vector<std::complex<float>>& fft_output,
    uint32_t nFFT,
    uint32_t search_range,
    float sample_rate,
    uint32_t antenna_id)
{
    CPUSpectrumResult result{};

    uint32_t half_range = search_range / 2;
    uint32_t range2_start = nFFT - half_range;
    float bin_width = sample_rate / static_cast<float>(nFFT);

    // ═══ ЛЕВЫЙ максимум [0, half_range] ═══
    float left_max_mag = -1.0f;
    uint32_t left_max_idx = 0;
    for (uint32_t i = 0; i < half_range; ++i) {
        float mag = std::abs(fft_output[i]);
        if (mag > left_max_mag) {
            left_max_mag = mag;
            left_max_idx = i;
        }
    }

    uint32_t ind_l = left_max_idx;
    std::complex<float> l_center_val = fft_output[ind_l];
    float y_l_center = std::abs(l_center_val);

    std::complex<float> l_minus1_val{0.0f, 0.0f};
    float y_l_minus1 = 0.0f;
    bool has_l_minus1 = (ind_l > 0 && (ind_l - 1) < half_range);
    if (has_l_minus1) {
        l_minus1_val = fft_output[ind_l - 1];
        y_l_minus1 = std::abs(l_minus1_val);
    }

    std::complex<float> l_plus1_val{0.0f, 0.0f};
    float y_l_plus1 = 0.0f;
    bool has_l_plus1 = (ind_l < nFFT - 1 && (ind_l + 1) < half_range);
    if (has_l_plus1) {
        l_plus1_val = fft_output[ind_l + 1];
        y_l_plus1 = std::abs(l_plus1_val);
    }

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

    result.SpectrMax_left.antenna_id = antenna_id;
    result.SpectrMax_left.interpolated = MakeMaxValue(ind_l, l_center_val.real(), l_center_val.imag(),
        y_l_center, std::atan2(l_center_val.imag(), l_center_val.real()) * 57.29577951f,
        l_freq_offset, l_refined_frequency);
    result.SpectrMax_left.left_point = has_l_minus1
        ? MakeMaxValue(ind_l - 1, l_minus1_val.real(), l_minus1_val.imag(), y_l_minus1,
            std::atan2(l_minus1_val.imag(), l_minus1_val.real()) * 57.29577951f,
            0.0f, static_cast<float>(ind_l - 1) * bin_width)
        : MaxValue{};
    result.SpectrMax_left.center_point = MakeMaxValue(ind_l, l_center_val.real(), l_center_val.imag(),
        y_l_center, std::atan2(l_center_val.imag(), l_center_val.real()) * 57.29577951f,
        0.0f, static_cast<float>(ind_l) * bin_width);
    result.SpectrMax_left.right_point = has_l_plus1
        ? MakeMaxValue(ind_l + 1, l_plus1_val.real(), l_plus1_val.imag(), y_l_plus1,
            std::atan2(l_plus1_val.imag(), l_plus1_val.real()) * 57.29577951f,
            0.0f, static_cast<float>(ind_l + 1) * bin_width)
        : MaxValue{};

    // ═══ ПРАВЫЙ максимум [nFFT-half_range, nFFT] ═══
    float right_max_mag = -1.0f;
    uint32_t right_max_idx = range2_start;
    for (uint32_t i = range2_start; i < nFFT; ++i) {
        float mag = std::abs(fft_output[i]);
        if (mag > right_max_mag) {
            right_max_mag = mag;
            right_max_idx = i;
        }
    }

    uint32_t ind_r = right_max_idx;
    std::complex<float> r_center_val = fft_output[ind_r];
    float y_r_center = std::abs(r_center_val);

    std::complex<float> r_minus1_val{0.0f, 0.0f};
    float y_r_minus1 = 0.0f;
    bool has_r_minus1 = (ind_r > 0 && (ind_r - 1) >= range2_start);
    if (has_r_minus1) {
        r_minus1_val = fft_output[ind_r - 1];
        y_r_minus1 = std::abs(r_minus1_val);
    }

    std::complex<float> r_plus1_val{0.0f, 0.0f};
    float y_r_plus1 = 0.0f;
    bool has_r_plus1 = (ind_r < nFFT - 1);
    if (has_r_plus1) {
        r_plus1_val = fft_output[ind_r + 1];
        y_r_plus1 = std::abs(r_plus1_val);
    }

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

    result.SpectrMax_right.antenna_id = antenna_id;
    result.SpectrMax_right.interpolated = MakeMaxValue(ind_r, r_center_val.real(), r_center_val.imag(),
        y_r_center, std::atan2(r_center_val.imag(), r_center_val.real()) * 57.29577951f,
        r_freq_offset, r_refined_frequency);
    result.SpectrMax_right.left_point = has_r_minus1
        ? MakeMaxValue(ind_r - 1, r_minus1_val.real(), r_minus1_val.imag(), y_r_minus1,
            std::atan2(r_minus1_val.imag(), r_minus1_val.real()) * 57.29577951f,
            0.0f, static_cast<float>(ind_r - 1) * bin_width)
        : MaxValue{};
    result.SpectrMax_right.center_point = MakeMaxValue(ind_r, r_center_val.real(), r_center_val.imag(),
        y_r_center, std::atan2(r_center_val.imag(), r_center_val.real()) * 57.29577951f,
        0.0f, static_cast<float>(ind_r) * bin_width);
    result.SpectrMax_right.right_point = has_r_plus1
        ? MakeMaxValue(ind_r + 1, r_plus1_val.real(), r_plus1_val.imag(), y_r_plus1,
            std::atan2(r_plus1_val.imag(), r_plus1_val.real()) * 57.29577951f,
            0.0f, static_cast<float>(ind_r + 1) * bin_width)
        : MaxValue{};

    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Цепочка: padding → FFT → FindMaximumWithInterpolation (внутренняя)
// ════════════════════════════════════════════════════════════════════════════

inline CPUSpectrumResult ProcessSingleBeamCore(
    const std::vector<std::complex<float>>& beam_data,
    uint32_t n_point,
    uint32_t nFFT,
    uint32_t search_range,
    float sample_rate,
    uint32_t antenna_id)
{
    if (beam_data.size() < n_point) {
        throw std::runtime_error("ProcessSingleBeamCore: input size < n_point");
    }

    std::vector<std::complex<float>> padded_input(nFFT, {0.0f, 0.0f});
    std::copy(beam_data.begin(), beam_data.begin() + n_point, padded_input.begin());

    std::vector<std::complex<float>> fft_output;
    ComputeFFT_CPU(padded_input, fft_output, nFFT);

    return FindMaximumWithInterpolation(fft_output, nFFT, search_range, sample_rate, antenna_id);
}

// ════════════════════════════════════════════════════════════════════════════
// NewProcessAllBeams_CPU — главная функция (Ref01)
// Алгоритм как на GPU: vi → nextPow2 → repeat_count → FFT → поиск максимумов
// ════════════════════════════════════════════════════════════════════════════

inline std::map<int, CPUSpectrumResult> NewProcessAllBeams_CPU(
    const std::vector<std::complex<float>>& data,
    const SpectrumParams& params)
{
    std::map<int, CPUSpectrumResult> results;

    uint32_t nFFT = (params.nFFT > 0) ? params.nFFT : (NextPowerOf2(params.n_point) * params.repeat_count);
    uint32_t search_range = (params.search_range > 0) ? params.search_range : (nFFT / 4);

    for (uint32_t i = 0; i < params.antenna_count; ++i) {
        std::vector<std::complex<float>> beam_data(
            data.begin() + i * params.n_point,
            data.begin() + (i + 1) * params.n_point
        );

        auto cpu_result = ProcessSingleBeamCore(
            beam_data, params.n_point, nFFT, search_range,
            params.sample_rate, static_cast<uint32_t>(i));

        results[static_cast<int>(i)] = cpu_result;
    }

    return results;
}

} // namespace cpu_reference
