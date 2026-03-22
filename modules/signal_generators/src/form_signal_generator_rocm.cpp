/**
 * @file form_signal_generator_rocm.cpp
 * @brief FormSignalGeneratorROCm — Ref03 implementation (GpuContext)
 *
 * Migrated from legacy hiprtc to Ref03 Unified Architecture.
 * GpuContext handles: kernel compilation, disk cache, arch detection.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (v1 legacy), 2026-03-22 (v2 Ref03)
 */

#if ENABLE_ROCM

#include "generators/form_signal_generator_rocm.hpp"
#include "kernels/form_signal_kernels_rocm.hpp"
#include "rocm_profiling_helpers.hpp"
#include "services/console_output.hpp"

#include <stdexcept>
#include <cmath>
#include <chrono>
#include <cstring>
#include <vector>

using fft_func_utils::MakeROCmDataFromEvents;

namespace signal_gen {

static const std::vector<std::string> kKernelNames = {
  "generate_form_signal"
};

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor / Move
// ════════════════════════════════════════════════════════════════════════════

FormSignalGeneratorROCm::FormSignalGeneratorROCm(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "FormSignal", "modules/signal_generators/kernels") {
}

FormSignalGeneratorROCm::~FormSignalGeneratorROCm() = default;

FormSignalGeneratorROCm::FormSignalGeneratorROCm(
    FormSignalGeneratorROCm&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , params_(other.params_)
    , compiled_(other.compiled_) {
  other.compiled_ = false;
}

FormSignalGeneratorROCm& FormSignalGeneratorROCm::operator=(
    FormSignalGeneratorROCm&& other) noexcept {
  if (this != &other) {
    ctx_ = std::move(other.ctx_);
    params_ = other.params_;
    compiled_ = other.compiled_;
    other.compiled_ = false;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Lazy compilation via GpuContext
// ════════════════════════════════════════════════════════════════════════════

void FormSignalGeneratorROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetFormSignalSource_rocm(), kKernelNames);
  compiled_ = true;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Generation
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<void*> FormSignalGeneratorROCm::GenerateInputData() {
  return GenerateInputData(nullptr);
}

drv_gpu_lib::InputData<void*>
FormSignalGeneratorROCm::GenerateInputData(ROCmProfEvents* prof_events) {
  EnsureCompiled();

  size_t total_points = GetTotalSamples();
  size_t buffer_size = total_points * sizeof(std::complex<float>);

  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "FormSignalGeneratorROCm: hipMalloc failed: " +
        std::string(hipGetErrorString(err)));
  }

  unsigned int ant = params_.antennas;
  unsigned int pts = params_.points;
  float dt = static_cast<float>(params_.GetDt());
  float ti = static_cast<float>(params_.GetDuration());
  float f0 = static_cast<float>(params_.f0);
  float amp = static_cast<float>(params_.amplitude);
  float an = static_cast<float>(params_.noise_amplitude);
  float phi = static_cast<float>(params_.phase);
  float fdev = static_cast<float>(params_.fdev);
  float norm_val = static_cast<float>(params_.norm);
  float tau_base = static_cast<float>(params_.tau_base);
  float tau_step = static_cast<float>(params_.tau_step);
  float tau_min = static_cast<float>(params_.tau_min);
  float tau_max = static_cast<float>(params_.tau_max);
  unsigned int tau_seed = params_.tau_seed;

  unsigned int noise_seed = params_.noise_seed;
  if (noise_seed == 0 && an > 0.0f) {
    noise_seed = static_cast<unsigned int>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
        & 0xFFFFFFFF);
  }

  unsigned int tau_mode = static_cast<unsigned int>(params_.GetTauMode());

  void* args[] = {
    &output_ptr, &ant, &pts, &dt, &ti, &f0, &amp, &an, &phi,
    &fdev, &norm_val, &tau_base, &tau_step, &tau_min, &tau_max,
    &tau_seed, &noise_seed, &tau_mode
  };

  unsigned int grid_x = static_cast<unsigned int>(
      (params_.points + kBlockSize - 1) / kBlockSize);
  unsigned int grid_y = params_.antennas;

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s);
    hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, ctx_.stream());
  }

  err = hipModuleLaunchKernel(
      ctx_.GetKernel("generate_form_signal"),
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);

  if (prof_events) {
    hipEventRecord(ev_k_e, ctx_.stream());
  }

  if (err != hipSuccess) {
    if (ev_k_s) { hipEventDestroy(ev_k_s); hipEventDestroy(ev_k_e); }
    (void)hipFree(output_ptr);
    throw std::runtime_error(
        "FormSignalGeneratorROCm: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  }

  (void)hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"Kernel",
        MakeROCmDataFromEvents(ev_k_s, ev_k_e, 0, "generate_form_signal")});
  }

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = params_.antennas;
  result.n_point       = params_.points;
  result.data          = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  result.sample_rate   = static_cast<float>(params_.fs);
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Generation (GPU generate → read back → split by channels)
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::vector<std::complex<float>>>
FormSignalGeneratorROCm::GenerateToCpu() {
  auto input = GenerateInputData();
  void* gpu_buf = input.data;

  size_t total = GetTotalSamples();
  std::vector<std::complex<float>> flat(total);

  hipError_t err = hipMemcpyDtoH(
      flat.data(), gpu_buf,
      total * sizeof(std::complex<float>));
  (void)hipFree(gpu_buf);

  if (err != hipSuccess) {
    throw std::runtime_error(
        "FormSignalGeneratorROCm::GenerateToCpu: hipMemcpyDtoH failed: " +
        std::string(hipGetErrorString(err)));
  }

  std::vector<std::vector<std::complex<float>>> result(params_.antennas);
  for (uint32_t a = 0; a < params_.antennas; ++a) {
    size_t offset = static_cast<size_t>(a) * params_.points;
    result[a].assign(
        flat.begin() + offset,
        flat.begin() + offset + params_.points);
  }

  return result;
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
