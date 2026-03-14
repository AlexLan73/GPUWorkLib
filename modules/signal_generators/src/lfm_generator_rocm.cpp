/**
 * @file lfm_generator_rocm.cpp
 * @brief LfmGeneratorROCm implementation — LFM chirp on GPU (ROCm/HIP)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "generators/lfm_generator_rocm.hpp"
#include "kernels/lfm_kernels_rocm.hpp"
#include "services/console_output.hpp"

#include <stdexcept>
#include <cmath>

namespace signal_gen {

static const std::vector<std::string> kLfmKernelNames = {
  "generate_lfm", "generate_lfm_real"
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

LfmGeneratorROCm::LfmGeneratorROCm(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "LfmGen", "modules/signal_generators/kernels") {
}

void LfmGeneratorROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetLfmSource_rocm(), kLfmKernelNames);
  compiled_ = true;
}

drv_gpu_lib::InputData<void*> LfmGeneratorROCm::GenerateToGpu(
    const SystemSampling& system,
    const LfmParams& params,
    uint32_t beam_count,
    ROCmProfEvents* prof_events) {

  EnsureCompiled();

  size_t total = static_cast<size_t>(beam_count) * system.length;
  size_t buffer_size = total * sizeof(std::complex<float>);

  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error("LfmGeneratorROCm: hipMalloc failed: " +
                              std::string(hipGetErrorString(err)));
  }

  const char* kernel_name = params.complex_iq ? "generate_lfm" : "generate_lfm_real";
  hipFunction_t k = ctx_.GetKernel(kernel_name);

  double duration = static_cast<double>(system.length) / system.fs;
  float chirp_rate = static_cast<float>(params.GetChirpRate(duration));

  unsigned int bc = beam_count;
  unsigned int np = static_cast<unsigned int>(system.length);
  float fs   = static_cast<float>(system.fs);
  float f_st = static_cast<float>(params.f_start);
  float amp  = static_cast<float>(params.amplitude);

  void* args[] = { &output_ptr, &bc, &np, &fs, &f_st, &chirp_rate, &amp };

  unsigned int grid_x = (np + kBlockSize - 1) / kBlockSize;

  hipEvent_t ev_s = nullptr, ev_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_s); hipEventCreate(&ev_e);
    hipEventRecord(ev_s, ctx_.stream());
  }

  err = hipModuleLaunchKernel(k,
      grid_x, bc, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);
  if (err != hipSuccess) {
    if (ev_s) { hipEventDestroy(ev_s); hipEventDestroy(ev_e); }
    (void)hipFree(output_ptr);
    throw std::runtime_error("LfmGeneratorROCm: kernel launch failed: " +
                              std::string(hipGetErrorString(err)));
  }

  if (prof_events) hipEventRecord(ev_e, ctx_.stream());
  hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"Kernel", MakeROCmData(ev_s, ev_e, 0, "generate_lfm")});
  }

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = beam_count;
  result.n_point = static_cast<uint32_t>(system.length);
  result.data = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

std::vector<std::complex<float>> LfmGeneratorROCm::GenerateToCpu(
    const SystemSampling& system,
    const LfmParams& params,
    uint32_t beam_count) {

  size_t total = static_cast<size_t>(beam_count) * system.length;
  std::vector<std::complex<float>> output(total);
  double duration = static_cast<double>(system.length) / system.fs;
  double k = params.GetChirpRate(duration);

  for (uint32_t b = 0; b < beam_count; ++b) {
    for (size_t n = 0; n < system.length; ++n) {
      double t = static_cast<double>(n) / system.fs;
      double phase = M_PI * k * t * t + 2.0 * M_PI * params.f_start * t;
      size_t idx = b * system.length + n;
      if (params.complex_iq) {
        output[idx] = std::complex<float>(
            static_cast<float>(params.amplitude * std::cos(phase)),
            static_cast<float>(params.amplitude * std::sin(phase)));
      } else {
        output[idx] = std::complex<float>(
            static_cast<float>(params.amplitude * std::cos(phase)), 0.0f);
      }
    }
  }
  return output;
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
