#pragma once

/**
 * @file spectrum_result_types.hpp
 * @brief Типы результатов поиска максимумов
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "interface/output_destination.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace antenna_fft {

using drv_gpu_lib::OutputDestination;

struct MaxValue {
    uint32_t index;
    float real, imag, magnitude, phase, freq_offset, refined_frequency;
    uint32_t pad;
};

struct SpectrumResult {
    uint32_t antenna_id;
    MaxValue interpolated, left_point, center_point, right_point;
};

struct CPUSpectrumResult {
    SpectrumResult SpectrMax_left, SpectrMax_right;
};

struct AllMaximaBeamResult {
    uint32_t antenna_id, num_maxima;
    std::vector<MaxValue> maxima;
};

struct AllMaximaResult {
    std::vector<AllMaximaBeamResult> beams;
    OutputDestination destination;
    void* gpu_maxima = nullptr;
    void* gpu_counts = nullptr;
    size_t total_maxima = 0;
    size_t gpu_bytes = 0;
    size_t TotalMaxima() const { return total_maxima; }
};

}  // namespace antenna_fft
