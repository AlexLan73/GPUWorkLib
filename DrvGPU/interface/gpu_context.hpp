#pragma once

/**
 * @file gpu_context.hpp
 * @brief GpuContext — per-module shared state for GPU operations
 *
 * Part of Ref03 Unified Architecture (Layer 1).
 *
 * Each module (StatisticsProcessor, FilterProcessor, etc.) creates its own
 * GpuContext. This provides:
 *   - backend + stream access (from IBackend)
 *   - Kernel compilation via hiprtc (one CompileModule call for ALL kernels)
 *   - Kernel lookup by name (GetKernel)
 *   - Shared GPU buffers (used by multiple Ops within the module)
 *   - Disk cache via KernelCacheService
 *   - WARP_SIZE determination by GPU architecture
 *
 * Thread safety: per-module instance → no shared mutable state between modules.
 * Operations within one module are sequential (same stream).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "services/buffer_set.hpp"
#include "services/console_output.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

// Forward declarations
namespace drv_gpu_lib {
class IBackend;
class KernelCacheService;
}

namespace drv_gpu_lib {

class GpuContext {
public:
  // ═══════════════════════════════════════════════════════════════════════
  // Shared buffer IDs — used by Ops to access module-wide GPU buffers.
  // Each module can define additional IDs starting after kSharedCount.
  // ═══════════════════════════════════════════════════════════════════════

  enum SharedBuf : size_t {
    kInput = 0,           ///< complex<float> input data
    kMagnitudes,          ///< float magnitudes (|z|)
    kResult,              ///< per-beam results (various types)
    kMediansCompact,      ///< float[beam_count] compact medians
    kSharedCount          ///< total count — used as BufferSet<N> template arg
  };

  // ═══════════════════════════════════════════════════════════════════════
  // Construction / Destruction
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Construct GpuContext for a module
   * @param backend Non-owning pointer to IBackend (must be ROCm, initialized)
   * @param module_name Human-readable name for logging (e.g. "Statistics")
   * @param cache_dir Disk cache directory for compiled HSACO (e.g. "modules/statistics/kernels")
   */
  GpuContext(IBackend* backend,
             const char* module_name,
             const std::string& cache_dir = "");

  ~GpuContext();

  // No copy
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

  // Move
  GpuContext(GpuContext&& other) noexcept;
  GpuContext& operator=(GpuContext&& other) noexcept;

  // ═══════════════════════════════════════════════════════════════════════
  // Immutable accessors (thread-safe reads)
  // ═══════════════════════════════════════════════════════════════════════

  IBackend* backend() const { return backend_; }
  hipStream_t stream() const { return stream_; }
  const char* module_name() const { return module_name_; }
  int warp_size() const { return warp_size_; }
  const std::string& arch_name() const { return arch_name_; }

  // ═══════════════════════════════════════════════════════════════════════
  // Kernel compilation (lazy, one-time per module)
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Compile all kernels for this module in one hiprtc call
   * @param source HIP C++ source (from kernels::GetXxxKernelSource())
   * @param kernel_names List of __global__ function names to extract
   * @param extra_defines Additional -D flags (e.g. "-DBLOCK_SIZE=256")
   *
   * Uses disk cache (KernelCacheService) when available.
   * Sets warp_size_ based on GPU architecture (gfx9* → 64, else → 32).
   * Idempotent: second call is a no-op.
   */
  void CompileModule(const char* source,
                     const std::vector<std::string>& kernel_names,
                     const std::vector<std::string>& extra_defines = {});

  /**
   * @brief Get compiled kernel function by name
   * @throws std::runtime_error if kernel not found or not compiled
   */
  hipFunction_t GetKernel(const char* name) const;

  /// True if CompileModule() has been called successfully
  bool IsCompiled() const { return module_ != nullptr; }

  // ═══════════════════════════════════════════════════════════════════════
  // Shared GPU buffers (module-wide, used by multiple Ops)
  // ═══════════════════════════════════════════════════════════════════════

  /**
   * @brief Get or allocate shared buffer
   * @param id SharedBuf enum value
   * @param bytes Required size in bytes
   * @return Device pointer (reused if existing buffer is large enough)
   */
  void* RequireShared(SharedBuf id, size_t bytes) {
    return shared_.Require(static_cast<size_t>(id), bytes);
  }

  /// Get existing shared buffer (no allocation, nullptr if not allocated)
  void* GetShared(SharedBuf id) const {
    return shared_.Get(static_cast<size_t>(id));
  }

  /// Release all shared buffers
  void ReleaseShared() { shared_.ReleaseAll(); }

private:
  // Backend (non-owning)
  IBackend* backend_ = nullptr;
  hipStream_t stream_ = nullptr;
  const char* module_name_ = "Unknown";

  // Architecture info (set in constructor)
  std::string arch_name_;
  int warp_size_ = 32;

  // Compiled kernels
  hipModule_t module_ = nullptr;
  std::unordered_map<std::string, hipFunction_t> kernels_;

  // Shared buffers
  BufferSet<kSharedCount> shared_;

  // Disk cache (optional)
  std::unique_ptr<KernelCacheService> kernel_cache_;

  /// Release compiled module
  void ReleaseModule();
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
