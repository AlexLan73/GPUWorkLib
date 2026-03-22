/**
 * @file lfm_generator_analytical_delay_rocm.cpp
 * @brief ROCm implementation of LFM with analytical per-antenna delay
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "generators/lfm_generator_analytical_delay_rocm.hpp"
#include "kernels/lfm_kernels_rocm.hpp"
#include "rocm_profiling_helpers.hpp"

#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using fft_func_utils::MakeROCmDataFromEvents;

namespace signal_gen {

static const std::vector<std::string> kDelayKernelNames = {
  "generate_lfm_analytical_delay"
};

LfmGeneratorAnalyticalDelayROCm::LfmGeneratorAnalyticalDelayROCm(
    drv_gpu_lib::IBackend* backend, const LfmParams& params)
    : ctx_(backend, "LfmDelay", "modules/signal_generators/kernels")
    , params_(params) {
}

void LfmGeneratorAnalyticalDelayROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetLfmSource_rocm(), kDelayKernelNames);
  compiled_ = true;
}

drv_gpu_lib::InputData<void*> LfmGeneratorAnalyticalDelayROCm::GenerateToGpu(
    ROCmProfEvents* prof_events) {
  EnsureCompiled();

  uint32_t n_ant = GetAntennas();
  uint32_t n_point = static_cast<uint32_t>(system_.length);
  float sample_rate = static_cast<float>(system_.fs);
  float f_start = static_cast<float>(params_.f_start);
  float f_end = static_cast<float>(params_.f_end);
  float duration = static_cast<float>(n_point) / sample_rate;
  float chirp_rate = (f_end - f_start) / duration;
  float amplitude = static_cast<float>(params_.amplitude);

  size_t total = static_cast<size_t>(n_ant) * n_point;
  size_t buffer_size = total * sizeof(std::complex<float>);

  // Output buffer (caller owns)
  void* output = nullptr;
  hipError_t err = hipMalloc(&output, buffer_size);
  if (err != hipSuccess)
    throw std::runtime_error("LfmAnalyticalDelay: hipMalloc output failed");

  // Upload delays
  void* d_delays = nullptr;
  size_t delay_size = n_ant * sizeof(float);
  err = hipMalloc(&d_delays, delay_size);
  if (err != hipSuccess) {
    hipFree(output);
    throw std::runtime_error("LfmAnalyticalDelay: hipMalloc delays failed");
  }
  hipMemcpyHtoDAsync(d_delays, delay_us_.data(), delay_size, ctx_.stream());

  // Launch kernel
  unsigned int grid_x = (n_point + 255) / 256;
  unsigned int grid_y = n_ant;

  hipEvent_t ev_s = nullptr, ev_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_s); hipEventCreate(&ev_e);
    hipEventRecord(ev_s, ctx_.stream());
  }

  void* args[] = { &output, &d_delays, &n_ant, &n_point,
                    &sample_rate, &f_start, &chirp_rate, &amplitude };

  err = hipModuleLaunchKernel(
      ctx_.GetKernel("generate_lfm_analytical_delay"),
      grid_x, grid_y, 1,
      256, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess) {
    hipFree(output); hipFree(d_delays);
    throw std::runtime_error("LfmAnalyticalDelay: kernel launch failed");
  }

  if (prof_events) hipEventRecord(ev_e, ctx_.stream());
  hipStreamSynchronize(ctx_.stream());
  hipFree(d_delays);

  if (prof_events) {
    prof_events->push_back({"Kernel",
        MakeROCmDataFromEvents(ev_s, ev_e, 0, "generate_lfm_analytical_delay")});
  }

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = n_ant;
  result.n_point = n_point;
  result.data = output;
  result.gpu_memory_bytes = buffer_size;
  result.sample_rate = sample_rate;
  return result;
}

std::vector<std::vector<std::complex<float>>>
LfmGeneratorAnalyticalDelayROCm::GenerateToCpu() {
  uint32_t n_ant = GetAntennas();
  size_t n = system_.length;
  double fs = system_.fs;
  double duration = static_cast<double>(n) / fs;
  double mu = (params_.f_end - params_.f_start) / duration;

  std::vector<std::vector<std::complex<float>>> result(n_ant);
  for (uint32_t a = 0; a < n_ant; ++a) {
    result[a].resize(n);
    double tau = static_cast<double>(delay_us_[a]) * 1e-6;
    for (size_t i = 0; i < n; ++i) {
      double t = static_cast<double>(i) / fs;
      if (t < tau) {
        result[a][i] = {0.0f, 0.0f};
      } else {
        double tl = t - tau;
        double phase = M_PI * mu * tl * tl + 2.0 * M_PI * params_.f_start * tl;
        result[a][i] = std::complex<float>(
            static_cast<float>(params_.amplitude * std::cos(phase)),
            static_cast<float>(params_.amplitude * std::sin(phase)));
      }
    }
  }
  return result;
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
