/**
 * @file delayed_form_signal_generator_rocm.cpp
 * @brief ROCm pipeline: FormSignalROCm → LchFarrowROCm (delay + noise)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "generators/delayed_form_signal_generator_rocm.hpp"

#include <stdexcept>

namespace signal_gen {

DelayedFormSignalGeneratorROCm::DelayedFormSignalGeneratorROCm(
    drv_gpu_lib::IBackend* backend)
    : backend_(backend)
    , signal_gen_(backend)
    , lch_farrow_(backend) {
}

void DelayedFormSignalGeneratorROCm::SetParams(const FormParams& params) {
  params_ = params;

  // Clean signal: noise=0, delay handled by LchFarrow
  FormParams clean = params;
  clean.noise_amplitude = 0.0;
  signal_gen_.SetParams(clean);

  // Noise applied by LchFarrow after delay
  lch_farrow_.SetSampleRate(static_cast<float>(params.fs));
  if (params.noise_amplitude > 0.0) {
    lch_farrow_.SetNoise(static_cast<float>(params.noise_amplitude),
                          static_cast<float>(params.norm),
                          params.noise_seed);
  }
}

void DelayedFormSignalGeneratorROCm::SetDelays(const std::vector<float>& delay_us) {
  lch_farrow_.SetDelays(delay_us);
}

drv_gpu_lib::InputData<void*> DelayedFormSignalGeneratorROCm::GenerateInputData() {
  // Step 1: Generate clean signal on GPU
  auto clean_signal = signal_gen_.GenerateInputData();

  // Step 2: Apply fractional delay via LchFarrow
  auto delayed = lch_farrow_.Process(
      clean_signal.data, params_.antennas, params_.points);

  // Free clean signal (delay result is a new buffer)
  (void)hipFree(clean_signal.data);

  return delayed;
}

std::vector<std::vector<std::complex<float>>>
DelayedFormSignalGeneratorROCm::GenerateToCpu() {
  auto result = GenerateInputData();
  void* gpu_buf = result.data;

  size_t total = static_cast<size_t>(params_.antennas) * params_.points;
  std::vector<std::complex<float>> flat(total);
  hipMemcpyDtoH(flat.data(), gpu_buf, total * sizeof(std::complex<float>));
  (void)hipFree(gpu_buf);

  std::vector<std::vector<std::complex<float>>> out(params_.antennas);
  for (uint32_t a = 0; a < params_.antennas; ++a) {
    size_t off = static_cast<size_t>(a) * params_.points;
    out[a].assign(flat.begin() + off, flat.begin() + off + params_.points);
  }
  return out;
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
