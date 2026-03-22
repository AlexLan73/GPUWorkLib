/**
 * @file symmetrize_gpu_rocm.cpp
 * @brief CompileKernels + SymmetrizeGpuKernel (hiprtc + disk cache)
 *
 * Реализация GPU-пути симметризации: компиляция HIP kernel через hiprtc,
 * запуск через hipModuleLaunchKernel.
 *
 * Оптимизация: дисковый кеш HSACO через KernelCacheService.
 * Первый запуск: hiprtc compile + save. Последующие: load binary.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-26
 */

#if ENABLE_ROCM

#include "cholesky_inverter_rocm.hpp"
#include "kernels/symmetrize_kernel_sources_rocm.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include "services/console_output.hpp"
#include "services/kernel_cache_service.hpp"
#include "backends/rocm/rocm_backend.hpp"

namespace vector_algebra {

// ════════════════════════════════════════════════════════════════════════════
// LoadModuleFromBinary — загрузить hipModule из бинарного blob'а
// ════════════════════════════════════════════════════════════════════════════

static void LoadModuleAndFunction(const void* binary_data, size_t /*binary_size*/,
                                   void*& out_module, void*& out_kernel) {
  hipModule_t module = nullptr;
  hipError_t hip_err = hipModuleLoadData(&module, binary_data);
  if (hip_err != hipSuccess) {
    throw std::runtime_error(
        "CompileKernels: hipModuleLoadData failed: " +
        std::string(hipGetErrorString(hip_err)));
  }
  out_module = static_cast<void*>(module);

  hipFunction_t func = nullptr;
  hip_err =
      hipModuleGetFunction(&func, module, "symmetrize_upper_to_full");
  if (hip_err != hipSuccess) {
    throw std::runtime_error(
        "CompileKernels: hipModuleGetFunction failed: " +
        std::string(hipGetErrorString(hip_err)));
  }
  out_kernel = static_cast<void*>(func);
}

// ════════════════════════════════════════════════════════════════════════════
// CompileKernels — hiprtc compilation with disk cache
// ════════════════════════════════════════════════════════════════════════════

void CholeskyInverterROCm::CompileKernels() {
  if (kernels_compiled_) return;

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  constexpr const char* kKernelName = "symmetrize_upper_to_full";

  // ─── Try loading from disk cache ───────────────────────────────────────
  if (kernel_cache_) {
    auto entry = kernel_cache_->Load(kKernelName);
    if (entry && entry->has_binary()) {
      LoadModuleAndFunction(entry->binary.data(), entry->binary.size(),
                             sym_module_, sym_kernel_);
      kernels_compiled_ = true;
      con.Print(0, "VecAlg",
                "symmetrize kernel loaded from cache (HSACO)");
      return;
    }
  }

  // ─── Compile via hiprtc ────────────────────────────────────────────────
  const char* src = kernels::GetSymmetrizeKernelSource();

  hiprtcProgram prog;
  hiprtcResult rtc_err =
      hiprtcCreateProgram(&prog, src, "symmetrize.hip", 0, nullptr, nullptr);
  if (rtc_err != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        "CompileKernels: hiprtcCreateProgram failed: " +
        std::string(hiprtcGetErrorString(rtc_err)));
  }

  // Получить целевую архитектуру для ISA-оптимизаций
  std::string arch_name;
  try {
    auto* rocm = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
    arch_name = rocm->GetCore().GetArchName();
  } catch (...) {}
  std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);

  std::vector<const char*> options = {"-O3"};
  if (!arch_flag.empty()) options.push_back(arch_flag.c_str());

  rtc_err = hiprtcCompileProgram(prog,
      static_cast<int>(options.size()), options.data());
  if (rtc_err != HIPRTC_SUCCESS) {
    size_t log_size = 0;
    hiprtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    hiprtcGetProgramLog(prog, log.data());

    con.PrintError(0, "VectorAlgebra",
                   "symmetrize kernel compile log:\n" + log);

    (void)hiprtcDestroyProgram(&prog);
    throw std::runtime_error(
        "CompileKernels: hiprtcCompileProgram failed: " +
        std::string(hiprtcGetErrorString(rtc_err)));
  }

  // Get binary code (HSACO)
  size_t code_size = 0;
  hiprtcGetCodeSize(prog, &code_size);
  std::vector<char> code(code_size);
  hiprtcGetCode(prog, code.data());
  (void)hiprtcDestroyProgram(&prog);

  // Load module from compiled binary
  LoadModuleAndFunction(code.data(), code.size(),
                         sym_module_, sym_kernel_);
  kernels_compiled_ = true;

  con.Print(0, "VecAlg",
            "symmetrize kernel compiled (hiprtc, " +
            std::to_string(code_size) + " bytes HSACO)");

  // ─── Save to disk cache for next run ───────────────────────────────────
  if (kernel_cache_) {
    try {
      std::vector<uint8_t> binary(code.begin(), code.end());
      kernel_cache_->Save(kKernelName, src, binary,
                           "", "symmetrize_upper_to_full hiprtc kernel");
      con.Print(0, "VecAlg", "symmetrize kernel saved to cache");
    } catch (const std::exception& e) {
      con.Print(0, "VecAlg",
                "warning: cache save failed: " + std::string(e.what()));
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// SymmetrizeGpuKernel — single matrix
// ════════════════════════════════════════════════════════════════════════════

void CholeskyInverterROCm::SymmetrizeGpuKernel(void* d_matrix, int n,
                                                 void* stream) {
  CompileKernels();

  unsigned int un = static_cast<unsigned int>(n);
  dim3 block(16, 16);
  dim3 grid((un + 15) / 16, (un + 15) / 16);

  void* args[] = {&d_matrix, &un};

  hipError_t err = hipModuleLaunchKernel(
      static_cast<hipFunction_t>(sym_kernel_),
      grid.x, grid.y, 1,    // grid dimensions
      block.x, block.y, 1,  // block dimensions
      0,                     // shared memory
      static_cast<hipStream_t>(stream),
      args, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error(
        "SymmetrizeGpuKernel: launch failed: " +
        std::string(hipGetErrorString(err)));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// SymmetrizeGpuKernelBatched — цикл по матрицам
// ════════════════════════════════════════════════════════════════════════════

void CholeskyInverterROCm::SymmetrizeGpuKernelBatched(void* d_contiguous,
                                                        int n, int batch,
                                                        void* stream) {
  const size_t one_bytes =
      static_cast<size_t>(n) * n * sizeof(std::complex<float>);
  auto* base = static_cast<char*>(d_contiguous);

  for (int k = 0; k < batch; ++k) {
    void* ptr_k = base + static_cast<size_t>(k) * one_bytes;
    SymmetrizeGpuKernel(ptr_k, n, stream);
  }
}

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
