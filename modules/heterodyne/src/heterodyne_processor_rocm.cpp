/**
 * @file heterodyne_processor_rocm.cpp
 * @brief ROCm stub for heterodyne processor (not implemented)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "processors/heterodyne_processor_rocm.hpp"
#include <stdexcept>

namespace drv_gpu_lib {

HeterodyneProcessorROCm::HeterodyneProcessorROCm(IBackend* /*backend*/) {
  // ROCm not implemented yet
}

std::vector<std::complex<float>> HeterodyneProcessorROCm::Dechirp(
    const std::vector<std::complex<float>>& /*rx_data*/,
    const std::vector<std::complex<float>>& /*ref_data*/,
    const HeterodyneParams& /*params*/) {
  throw std::runtime_error("HeterodyneProcessorROCm::Dechirp: ROCm not implemented");
}

std::vector<std::complex<float>> HeterodyneProcessorROCm::Correct(
    const std::vector<std::complex<float>>& /*dc_data*/,
    const std::vector<float>& /*f_beat_hz*/,
    const HeterodyneParams& /*params*/) {
  throw std::runtime_error("HeterodyneProcessorROCm::Correct: ROCm not implemented");
}

std::vector<std::complex<float>> HeterodyneProcessorROCm::DechirpFromGPU(
    void* /*rx_cl_mem*/,
    const std::vector<std::complex<float>>& /*ref_data*/,
    const HeterodyneParams& /*params*/) {
  throw std::runtime_error("HeterodyneProcessorROCm::DechirpFromGPU: ROCm not implemented");
}

}  // namespace drv_gpu_lib
