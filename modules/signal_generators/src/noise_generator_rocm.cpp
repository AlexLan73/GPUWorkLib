/**
 * @file noise_generator_rocm.cpp
 * @brief NoiseGeneratorROCm implementation — Noise on GPU (ROCm/HIP)
 *
 * Philox-2x32-10 PRNG + Box-Muller (Gaussian) or uniform (White).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "generators/noise_generator_rocm.hpp"
#include "kernels/noise_kernels_rocm.hpp"
#include "services/console_output.hpp"

#include <stdexcept>
#include <cmath>

namespace signal_gen {

static const std::vector<std::string> kNoiseKernelNames = {
  "generate_noise_gaussian", "generate_noise_white"
};

static drv_gpu_lib::ROCmProfilingData MakeROCmData(
    hipEvent_t s, hipEvent_t e, uint32_t kind, const char* op) {
  hipEventSynchronize(e);
  float ms = 0.0f;
  hipEventElapsedTime(&ms, s, e);
  hipEventDestroy(s);
  hipEventDestroy(e);
  drv_gpu_lib::ROCmProfilingData d{};
  uint64_t ns = static_cast<uint64_t>(ms * 1e6f);
  d.start_ns = 0; d.end_ns = ns; d.complete_ns = ns;
  d.kind = kind; d.op_string = op;
  return d;
}

NoiseGeneratorROCm::NoiseGeneratorROCm(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "NoiseGen", "modules/signal_generators/kernels") {
}

void NoiseGeneratorROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetNoiseSource_rocm(), kNoiseKernelNames);
  compiled_ = true;
}

drv_gpu_lib::InputData<void*> NoiseGeneratorROCm::GenerateToGpu(
    const SystemSampling& system,
    const NoiseParams& params,
    uint32_t beam_count,
    ROCmProfEvents* prof_events) {

  EnsureCompiled();

  size_t total = static_cast<size_t>(beam_count) * system.length;
  size_t buffer_size = total * sizeof(std::complex<float>);

  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error("NoiseGeneratorROCm: hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  // Select kernel by noise type
  const char* kernel_name = (params.type == NoiseType::GAUSSIAN)
      ? "generate_noise_gaussian"
      : "generate_noise_white";
  hipFunction_t k = ctx_.GetKernel(kernel_name);

  // Seed: use provided or generate random
  uint32_t seed = (params.seed == 0)
      ? static_cast<uint32_t>(rng_())
      : static_cast<uint32_t>(params.seed);

  unsigned int total_u = static_cast<unsigned int>(total);
  float power_param = static_cast<float>(
      (params.type == NoiseType::GAUSSIAN) ? std::sqrt(params.power) : std::sqrt(params.power));

  void* args[] = { &output_ptr, &total_u, &power_param, &seed };

  // 1D grid (all samples in one dimension)
  unsigned int grid = (total_u + kBlockSize - 1) / kBlockSize;

  hipEvent_t ev_s = nullptr, ev_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_s); hipEventCreate(&ev_e);
    hipEventRecord(ev_s, ctx_.stream());
  }

  err = hipModuleLaunchKernel(k,
      grid, 1, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess) {
    if (ev_s) { hipEventDestroy(ev_s); hipEventDestroy(ev_e); }
    (void)hipFree(output_ptr);
    throw std::runtime_error("NoiseGeneratorROCm: kernel launch failed: " +
                              std::string(hipGetErrorString(err)));
  }

  if (prof_events) hipEventRecord(ev_e, ctx_.stream());
  hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"Kernel", MakeROCmData(ev_s, ev_e, 0, kernel_name)});
  }

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = beam_count;
  result.n_point = static_cast<uint32_t>(system.length);
  result.data = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

std::vector<std::complex<float>> NoiseGeneratorROCm::GenerateToCpu(
    const SystemSampling& system,
    const NoiseParams& params,
    uint32_t beam_count) {

  size_t total = static_cast<size_t>(beam_count) * system.length;
  std::vector<std::complex<float>> output(total);

  std::mt19937 gen(params.seed == 0 ? std::random_device{}() : static_cast<unsigned>(params.seed));

  if (params.type == NoiseType::GAUSSIAN) {
    float std_dev = static_cast<float>(std::sqrt(params.power));
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (size_t i = 0; i < total; ++i) {
      output[i] = std::complex<float>(dist(gen), dist(gen));
    }
  } else {
    float amp = static_cast<float>(std::sqrt(params.power));
    std::uniform_real_distribution<float> dist(-amp, amp);
    for (size_t i = 0; i < total; ++i) {
      output[i] = std::complex<float>(dist(gen), dist(gen));
    }
  }
  return output;
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
