/**
 * @file diagonal_load_regularizer.cpp
 * @brief DiagonalLoadRegularizer — реализация (hiprtc compile + kernel launch)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16
 */

#if ENABLE_ROCM

#include "diagonal_load_regularizer.hpp"
#include "kernels/diagonal_load_kernel_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include <hip/hiprtc.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace vector_algebra {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor / Move
// ════════════════════════════════════════════════════════════════════════════

DiagonalLoadRegularizer::DiagonalLoadRegularizer(drv_gpu_lib::IBackend* backend) {
  if (!backend || !backend->IsInitialized()) {
    throw std::runtime_error(
        "DiagonalLoadRegularizer: backend is null or not initialized");
  }
  stream_ = static_cast<hipStream_t>(backend->GetNativeQueue());
  Compile(backend);
}

DiagonalLoadRegularizer::~DiagonalLoadRegularizer() {
  Release();
}

DiagonalLoadRegularizer::DiagonalLoadRegularizer(
    DiagonalLoadRegularizer&& other) noexcept
    : stream_(other.stream_)
    , module_(other.module_)
    , function_(other.function_) {
  other.stream_   = nullptr;
  other.module_   = nullptr;
  other.function_ = nullptr;
}

DiagonalLoadRegularizer& DiagonalLoadRegularizer::operator=(
    DiagonalLoadRegularizer&& other) noexcept {
  if (this != &other) {
    Release();
    stream_   = other.stream_;
    module_   = other.module_;
    function_ = other.function_;
    other.stream_   = nullptr;
    other.module_   = nullptr;
    other.function_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Compile — hiprtc компиляция kernel при создании объекта
// ════════════════════════════════════════════════════════════════════════════

void DiagonalLoadRegularizer::Compile(drv_gpu_lib::IBackend* backend) {
  const char* source = kernels::GetDiagonalLoadKernelSource();

  hiprtcProgram prog;
  hiprtcResult rtc = hiprtcCreateProgram(
      &prog, source, "diagonal_load.hip", 0, nullptr, nullptr);
  if (rtc != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        "DiagonalLoadRegularizer: hiprtcCreateProgram failed (" +
        std::to_string(static_cast<int>(rtc)) + ")");
  }

  // --offload-arch для ISA-оптимизаций (как в symmetrize и всех модулях)
  std::string arch_name;
  try {
    auto* rb = static_cast<drv_gpu_lib::ROCmBackend*>(backend);
    arch_name = rb->GetCore().GetArchName();
  } catch (...) {}
  std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);

  std::vector<const char*> opts = {"-O2", "-std=c++17"};
  if (!arch_flag.empty()) opts.push_back(arch_flag.c_str());

  rtc = hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
  if (rtc != HIPRTC_SUCCESS) {
    size_t log_size = 0;
    hiprtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    hiprtcGetProgramLog(prog, &log[0]);
    hiprtcDestroyProgram(&prog);
    throw std::runtime_error(
        "DiagonalLoadRegularizer: compilation failed:\n" + log);
  }

  size_t code_size = 0;
  hiprtcGetCodeSize(prog, &code_size);
  std::string code(code_size, '\0');
  hiprtcGetCode(prog, &code[0]);
  hiprtcDestroyProgram(&prog);

  hipModule_t mod = nullptr;
  hipError_t err = hipModuleLoadData(&mod, code.data());
  if (err != hipSuccess) {
    throw std::runtime_error(
        "DiagonalLoadRegularizer: hipModuleLoadData failed: " +
        std::string(hipGetErrorString(err)));
  }

  hipFunction_t func = nullptr;
  err = hipModuleGetFunction(&func, mod, "diagonal_load");
  if (err != hipSuccess) {
    hipModuleUnload(mod);
    throw std::runtime_error(
        "DiagonalLoadRegularizer: hipModuleGetFunction failed: " +
        std::string(hipGetErrorString(err)));
  }

  module_   = static_cast<void*>(mod);
  function_ = static_cast<void*>(func);

  // backend использован для arch_name выше; stream_ взят в конструкторе
}

// ════════════════════════════════════════════════════════════════════════════
// Apply — запуск kernel: A[i,i] += mu
// ════════════════════════════════════════════════════════════════════════════

void DiagonalLoadRegularizer::Apply(void* d_matrix, int n, float mu,
                                    hipStream_t stream) {
  if (mu == 0.0f) return;  // no-op: нечего прибавлять

  // Если caller передал stream — используем его (гарантирует порядок после
  // rocBLAS CGEMM на том же stream). Иначе — собственный stream из backend.
  hipStream_t target_stream = (stream != nullptr) ? stream : stream_;

  auto func = static_cast<hipFunction_t>(function_);
  auto un   = static_cast<unsigned int>(n);

  void* args[] = { &d_matrix, &mu, &un };
  hipError_t err = hipModuleLaunchKernel(
      func,
      (un + 255u) / 256u, 1, 1,  // grid
      256, 1, 1,                  // block
      0, target_stream,
      args, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error(
        "DiagonalLoadRegularizer::Apply: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Release
// ════════════════════════════════════════════════════════════════════════════

void DiagonalLoadRegularizer::Release() noexcept {
  if (module_) {
    hipModuleUnload(static_cast<hipModule_t>(module_));
    module_   = nullptr;
    function_ = nullptr;  // function_ — часть module_, не освобождать отдельно
  }
  stream_ = nullptr;
}

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
