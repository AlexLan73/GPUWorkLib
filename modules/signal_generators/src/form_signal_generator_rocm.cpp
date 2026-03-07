/**
 * @file form_signal_generator_rocm.cpp
 * @brief FormSignalGeneratorROCm implementation - multi-channel getX on GPU (ROCm/HIP)
 *
 * Port of form_signal_generator.cpp (OpenCL) to HIP/ROCm.
 * Uses hiprtc for runtime kernel compilation, hipModuleLaunchKernel for dispatch.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "generators/form_signal_generator_rocm.hpp"
#include "kernels/form_signal_kernels_rocm.hpp"
#include "services/console_output.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include <stdexcept>
#include <cmath>
#include <chrono>
#include <cstring>
#include <vector>

namespace signal_gen {
namespace {

/// Helper: hipEvent → elapsed → ROCmProfilingData (уничтожает events)
drv_gpu_lib::ROCmProfilingData MakeROCmDataFromEvents(
    hipEvent_t ev_start, hipEvent_t ev_end,
    uint32_t kind = 0, const char* op = "")
{
  hipEventSynchronize(ev_end);
  float ms = 0.0f;
  hipEventElapsedTime(&ms, ev_start, ev_end);
  hipEventDestroy(ev_start);
  hipEventDestroy(ev_end);
  drv_gpu_lib::ROCmProfilingData d{};
  uint64_t ns = static_cast<uint64_t>(ms * 1e6f);
  d.start_ns = 0; d.end_ns = ns; d.complete_ns = ns;
  d.kind = kind; d.op_string = op;
  return d;
}

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

FormSignalGeneratorROCm::FormSignalGeneratorROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "FormSignalGeneratorROCm: backend is null or not initialized");
  }

  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
  if (!stream_) {
    throw std::runtime_error(
        "FormSignalGeneratorROCm: failed to get HIP stream from backend");
  }

  CompileKernel();
}

FormSignalGeneratorROCm::~FormSignalGeneratorROCm() {
  ReleaseGpuResources();
}

FormSignalGeneratorROCm::FormSignalGeneratorROCm(
    FormSignalGeneratorROCm&& other) noexcept
    : backend_(other.backend_)
    , stream_(other.stream_)
    , params_(other.params_)
    , module_(other.module_)
    , kernel_(other.kernel_)
    , kernel_compiled_(other.kernel_compiled_) {
  other.backend_ = nullptr;
  other.stream_ = nullptr;
  other.module_ = nullptr;
  other.kernel_ = nullptr;
  other.kernel_compiled_ = false;
}

FormSignalGeneratorROCm& FormSignalGeneratorROCm::operator=(
    FormSignalGeneratorROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    backend_ = other.backend_;
    stream_ = other.stream_;
    params_ = other.params_;
    module_ = other.module_;
    kernel_ = other.kernel_;
    kernel_compiled_ = other.kernel_compiled_;
    other.backend_ = nullptr;
    other.stream_ = nullptr;
    other.module_ = nullptr;
    other.kernel_ = nullptr;
    other.kernel_compiled_ = false;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Generation
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<void*> FormSignalGeneratorROCm::GenerateInputData() {
  return GenerateInputData(nullptr);
}

drv_gpu_lib::InputData<void*>
FormSignalGeneratorROCm::GenerateInputData(ROCmProfEvents* prof_events) {
  size_t total_points = GetTotalSamples();
  size_t buffer_size = total_points * sizeof(std::complex<float>);

  // Allocate output buffer
  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "FormSignalGeneratorROCm::GenerateInputData: hipMalloc failed: " +
        std::string(hipGetErrorString(err)));
  }

  // Prepare kernel arguments
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

  // hipModuleLaunchKernel args array: pointers to each argument
  void* args[] = {
    &output_ptr,
    &ant,
    &pts,
    &dt,
    &ti,
    &f0,
    &amp,
    &an,
    &phi,
    &fdev,
    &norm_val,
    &tau_base,
    &tau_step,
    &tau_min,
    &tau_max,
    &tau_seed,
    &noise_seed,
    &tau_mode
  };

  // 2D grid: gridX = sample blocks, gridY = antennas (eliminates div/mod)
  unsigned int grid_x = static_cast<unsigned int>(
      (params_.points + kBlockSize - 1) / kBlockSize);
  unsigned int grid_y = params_.antennas;

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s);
    hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, stream_);
  }

  err = hipModuleLaunchKernel(
      kernel_,
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, stream_,
      args, nullptr);

  if (prof_events) {
    hipEventRecord(ev_k_e, stream_);
  }

  if (err != hipSuccess) {
    if (ev_k_s) { hipEventDestroy(ev_k_s); hipEventDestroy(ev_k_e); }
    (void)hipFree(output_ptr);
    throw std::runtime_error(
        "FormSignalGeneratorROCm::GenerateInputData: hipModuleLaunchKernel failed: " +
        std::string(hipGetErrorString(err)));
  }

  (void)hipStreamSynchronize(stream_);

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

  // Split flat → vector<vector<complex>>
  std::vector<std::vector<std::complex<float>>> result(params_.antennas);
  for (uint32_t a = 0; a < params_.antennas; ++a) {
    size_t offset = static_cast<size_t>(a) * params_.points;
    result[a].assign(
        flat.begin() + offset,
        flat.begin() + offset + params_.points);
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Internals
// ════════════════════════════════════════════════════════════════════════════

void FormSignalGeneratorROCm::CompileKernel() {
  if (kernel_compiled_) return;

  const char* source = kernels::GetFormSignalSource_rocm();

  hiprtcProgram prog;
  hiprtcResult rtcResult = hiprtcCreateProgram(
      &prog, source, "form_signal_kernel.hip", 0, nullptr, nullptr);
  if (rtcResult != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        "FormSignalGeneratorROCm::CompileKernel: hiprtcCreateProgram failed: " +
        std::string(hiprtcGetErrorString(rtcResult)));
  }

  // Get target arch from backend for architecture-specific optimizations
  auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
  std::string arch = rocm_backend->GetCore().GetArchName();
  std::string arch_flag = "--offload-arch=" + arch;

  const char* options[] = { "-O3", arch_flag.c_str(), "-std=c++17" };
  rtcResult = hiprtcCompileProgram(prog, 3, options);
  if (rtcResult != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.PrintError(0, "FormSignal[ROCm]", "Kernel compile log:\n" + log);

    (void)hiprtcDestroyProgram(&prog);
    throw std::runtime_error(
        "FormSignalGeneratorROCm::CompileKernel: compilation failed");
  }

  size_t codeSize = 0;
  hiprtcGetCodeSize(prog, &codeSize);
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  (void)hiprtcDestroyProgram(&prog);

  hipError_t hipErr = hipModuleLoadData(&module_, code.data());
  if (hipErr != hipSuccess) {
    throw std::runtime_error(
        "FormSignalGeneratorROCm::CompileKernel: hipModuleLoadData failed: " +
        std::string(hipGetErrorString(hipErr)));
  }

  hipErr = hipModuleGetFunction(&kernel_, module_, "generate_form_signal");
  if (hipErr != hipSuccess) {
    throw std::runtime_error(
        "FormSignalGeneratorROCm::CompileKernel: hipModuleGetFunction(generate_form_signal) failed: " +
        std::string(hipGetErrorString(hipErr)));
  }

  kernel_compiled_ = true;

  drv_gpu_lib::ConsoleOutput::GetInstance().Print(0, "FormSignal[ROCm]",
      "HIP kernel compiled (generate_form_signal)");
}

void FormSignalGeneratorROCm::ReleaseGpuResources() {
  if (module_) {
    (void)hipModuleUnload(module_);
    module_ = nullptr;
    kernel_ = nullptr;
    kernel_compiled_ = false;
  }
}

}  // namespace signal_gen

#endif  // ENABLE_ROCM
