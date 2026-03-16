/**
 * @file gpu_context.cpp
 * @brief GpuContext implementation — kernel compilation, disk cache, shared buffers
 *
 * Part of Ref03 Unified Architecture (Layer 1).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "interface/gpu_context.hpp"
#include "interface/i_backend.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "services/kernel_cache_service.hpp"
#include "services/console_output.hpp"

#include <cstring>
#include <algorithm>

namespace drv_gpu_lib {

// ═════════════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ═════════════════════════════════════════════════════════════════════════════

GpuContext::GpuContext(IBackend* backend,
                       const char* module_name,
                       const std::string& cache_dir)
    : backend_(backend)
    , module_name_(module_name) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: backend is null or not initialized");
  }

  if (backend_->GetType() != BackendType::ROCm) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: requires ROCm backend");
  }

  // Get HIP stream from backend
  stream_ = static_cast<hipStream_t>(backend_->GetNativeQueue());
  if (!stream_) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: failed to get HIP stream");
  }

  // Determine GPU architecture and warp size
  try {
    auto* rocm_backend = static_cast<ROCmBackend*>(backend_);
    arch_name_ = rocm_backend->GetCore().GetArchName();
  } catch (...) {
    arch_name_ = "";
  }

  // CDNA / Vega (gfx900..gfx942) → 64, RDNA (gfx10xx, gfx11xx, gfx12xx) → 32
  warp_size_ = (arch_name_.find("gfx9") == 0) ? 64 : 32;

  // Disk cache for compiled HSACO
  if (!cache_dir.empty()) {
    kernel_cache_ = std::make_unique<KernelCacheService>(cache_dir, BackendType::ROCm);
  }
}

GpuContext::~GpuContext() {
  ReleaseShared();
  ReleaseModule();
}

GpuContext::GpuContext(GpuContext&& other) noexcept
    : backend_(other.backend_)
    , stream_(other.stream_)
    , module_name_(other.module_name_)
    , arch_name_(std::move(other.arch_name_))
    , warp_size_(other.warp_size_)
    , module_(other.module_)
    , kernels_(std::move(other.kernels_))
    , shared_(std::move(other.shared_))
    , kernel_cache_(std::move(other.kernel_cache_)) {
  other.backend_ = nullptr;
  other.stream_ = nullptr;
  other.module_ = nullptr;
}

GpuContext& GpuContext::operator=(GpuContext&& other) noexcept {
  if (this != &other) {
    ReleaseShared();
    ReleaseModule();

    backend_ = other.backend_;
    stream_ = other.stream_;
    module_name_ = other.module_name_;
    arch_name_ = std::move(other.arch_name_);
    warp_size_ = other.warp_size_;
    module_ = other.module_;
    kernels_ = std::move(other.kernels_);
    shared_ = std::move(other.shared_);
    kernel_cache_ = std::move(other.kernel_cache_);

    other.backend_ = nullptr;
    other.stream_ = nullptr;
    other.module_ = nullptr;
  }
  return *this;
}

// ═════════════════════════════════════════════════════════════════════════════
// Kernel Compilation
// ═════════════════════════════════════════════════════════════════════════════

void GpuContext::CompileModule(const char* source,
                               const std::vector<std::string>& kernel_names,
                               const std::vector<std::string>& extra_defines) {
  if (module_) return;  // already compiled

  auto& con = ConsoleOutput::GetInstance();
  const std::string cache_name = std::string(module_name_) + "_kernels";

  // Helper: extract all kernel functions from loaded module
  auto extractKernels = [&]() {
    for (const auto& name : kernel_names) {
      hipFunction_t func = nullptr;
      hipError_t err = hipModuleGetFunction(&func, module_, name.c_str());
      if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("GpuContext[") + module_name_ + "]: hipModuleGetFunction(" +
            name + ") failed: " + hipGetErrorString(err));
      }
      kernels_[name] = func;
    }
  };

  const int gpu_id = backend_->GetDeviceIndex();

  // ─── Try loading from disk cache ──────────────────────────────────────
  if (kernel_cache_) {
    auto entry = kernel_cache_->Load(cache_name);  // nullopt = cache miss
    if (entry && entry->has_binary()) {
      hipError_t err = hipModuleLoadData(&module_, entry->binary.data());
      if (err == hipSuccess) {
        extractKernels();
        con.Print(gpu_id, module_name_, "kernels loaded from cache (HSACO)");
        return;
      }
      // Cache might be stale (different arch) — fall through to compile
      if (module_) { hipModuleUnload(module_); module_ = nullptr; }
    }
  }

  // ─── Compile via hiprtc ───────────────────────────────────────────────
  hiprtcProgram prog;
  std::string filename = std::string(module_name_) + "_kernels.hip";
  hiprtcResult rtc = hiprtcCreateProgram(&prog, source, filename.c_str(),
                                          0, nullptr, nullptr);
  if (rtc != HIPRTC_SUCCESS) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: hiprtcCreateProgram failed: " +
        std::to_string(static_cast<int>(rtc)));
  }

  // Build compiler flags
  std::string warp_def  = "-DWARP_SIZE=" + std::to_string(warp_size_);
  std::string arch_flag = arch_name_.empty() ? "" : ("--offload-arch=" + arch_name_);

  std::vector<std::string> opts_storage = {"-O3", "-std=c++17", warp_def};
  for (const auto& def : extra_defines) {
    opts_storage.push_back(def);
  }
  if (!arch_flag.empty()) {
    opts_storage.push_back(arch_flag);
  }

  std::vector<const char*> opts;
  opts.reserve(opts_storage.size());
  for (const auto& o : opts_storage) {
    opts.push_back(o.c_str());
  }

  rtc = hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
  if (rtc != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);
    hiprtcDestroyProgram(&prog);
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: compilation failed:\n" + log);
  }

  // Extract compiled binary (HSACO)
  size_t code_size = 0;
  hiprtcGetCodeSize(prog, &code_size);
  std::vector<char> code(code_size);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  // Load module into GPU
  hipError_t hip_err = hipModuleLoadData(&module_, code.data());
  if (hip_err != hipSuccess) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: hipModuleLoadData failed: " +
        hipGetErrorString(hip_err));
  }

  // Extract kernel functions
  extractKernels();

  con.Print(gpu_id, module_name_,
            "kernels compiled (" + std::to_string(code_size) + " bytes HSACO" +
            (arch_name_.empty() ? "" : ", " + arch_name_) + ")");

  // ─── Save to disk cache ───────────────────────────────────────────────
  if (kernel_cache_) {
    try {
      std::vector<uint8_t> binary(code.begin(), code.end());
      kernel_cache_->Save(cache_name, source, binary, "",
                          std::string(module_name_) + " hiprtc kernels");
    } catch (...) {
      // Non-fatal: cache save failure doesn't stop execution
    }
  }
}

hipFunction_t GpuContext::GetKernel(const char* name) const {
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    throw std::runtime_error(
        std::string("GpuContext[") + module_name_ + "]: kernel '" + name +
        "' not found. CompileModule() not called or name misspelled.");
  }
  return it->second;
}

// ═════════════════════════════════════════════════════════════════════════════
// Private
// ═════════════════════════════════════════════════════════════════════════════

void GpuContext::ReleaseModule() {
  if (module_) {
    hipModuleUnload(module_);
    module_ = nullptr;
  }
  kernels_.clear();
}

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
