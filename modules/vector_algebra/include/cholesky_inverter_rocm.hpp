#pragma once
#if ENABLE_ROCM

/**
 * @file cholesky_inverter_rocm.hpp
 * @brief Инверсия эрмитовой положительно определённой матрицы (ROCm, POTRF+POTRI)
 *
 * Task_11 v2: два режима симметризации (Roundtrip / GpuKernel).
 *
 * Поддерживаемые входные форматы:
 *   - InputData<vector<complex<float>>>  — CPU вектор
 *   - InputData<void*>                   — ROCm device pointer
 *   - InputData<cl_mem>                  — OpenCL буфер (ZeroCopy)
 *
 * Результат: CholeskyResult (единый тип, void* d_data на GPU).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-26
 */

#include <complex>
#include <memory>
#include <vector>
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "vector_algebra_types.hpp"

namespace drv_gpu_lib { class KernelCacheService; }

namespace vector_algebra {

/**
 * @class CholeskyInverterROCm
 * @brief Инверсия эрмитовой положительно определённой матрицы (POTRF + POTRI).
 *
 * Два режима симметризации:
 *   - Roundtrip: Download → CPU sym → Upload
 *   - GpuKernel: HIP kernel in-place (hiprtc)
 *
 * Не копируемый, не перемещаемый (владеет rocBLAS handle + hipModule).
 *
 * @ingroup grp_vector_algebra
 */
class CholeskyInverterROCm {
public:
  /**
   * @brief Конструктор.
   * @param backend  ROCmBackend (или HybridBackend). Должен жить дольше объекта.
   * @param mode     Режим симметризации (по умолчанию GpuKernel).
   */
  explicit CholeskyInverterROCm(
      drv_gpu_lib::IBackend* backend,
      SymmetrizeMode mode = SymmetrizeMode::GpuKernel);

  ~CholeskyInverterROCm();

  CholeskyInverterROCm(const CholeskyInverterROCm&) = delete;
  CholeskyInverterROCm& operator=(const CholeskyInverterROCm&) = delete;

  /// Изменить режим симметризации
  void SetSymmetrizeMode(SymmetrizeMode mode);

  /// Текущий режим симметризации
  SymmetrizeMode GetSymmetrizeMode() const { return mode_; }

  /// Явная прекомпиляция hiprtc kernel (для warmup/benchmark).
  /// Вызывается автоматически в конструкторе при GpuKernel mode.
  void CompileKernels();

  /// Включить/выключить проверку info (POTRF/POTRI). Default: true.
  /// Для benchmark/production с гарантированно HPD матрицами — false.
  void SetCheckInfo(bool enabled) { check_info_ = enabled; }

  // ─── Одна матрица ─────────────────────────────────────────────────────

  /// CPU вектор → GPU → CholeskyResult
  CholeskyResult Invert(
      const drv_gpu_lib::InputData<std::vector<std::complex<float>>>& input,
      int n = 0);

  /// ROCm device pointer → CholeskyResult
  CholeskyResult Invert(
      const drv_gpu_lib::InputData<void*>& input,
      int n = 0);

#ifdef CL_VERSION_1_0
  /// OpenCL cl_mem (ZeroCopy) → CholeskyResult
  CholeskyResult Invert(
      const drv_gpu_lib::InputData<cl_mem>& input,
      int n = 0);
#endif

  // ─── Batched ──────────────────────────────────────────────────────────

  /// CPU batched → CholeskyResult
  CholeskyResult InvertBatch(
      const drv_gpu_lib::InputData<std::vector<std::complex<float>>>& input,
      int n);

  /// GPU batched → CholeskyResult
  CholeskyResult InvertBatch(
      const drv_gpu_lib::InputData<void*>& input,
      int n);

#ifdef CL_VERSION_1_0
  /// cl_mem batched (ZeroCopy) → CholeskyResult
  CholeskyResult InvertBatch(
      const drv_gpu_lib::InputData<cl_mem>& input,
      int n);
#endif

private:
  drv_gpu_lib::IBackend* backend_;
  void* handle_ = nullptr;    ///< rocblas_handle (opaque)
  SymmetrizeMode mode_;

  // ─── Предаллоцированный dev_info (Task_12: убираем hipMalloc/hipFree) ──
  void* d_info_ = nullptr;    ///< rocblas_int[2] на GPU (slot 0=potrf, 1=potri)

  // ─── hiprtc kernel state ──────────────────────────────────────────────
  void* sym_module_ = nullptr;   ///< hipModule_t
  void* sym_kernel_ = nullptr;   ///< hipFunction_t
  bool kernels_compiled_ = false;
  std::unique_ptr<drv_gpu_lib::KernelCacheService> kernel_cache_;

  // ─── Core GPU ops ─────────────────────────────────────────────────────

  /// POTRF: Cholesky decomposition A = U^H * U
  void CorePotrf(void* d_matrix, int n, void* stream);

  /// POTRI: Compute A^{-1} from U
  void CorePotri(void* d_matrix, int n, void* stream);

  /// POTRF batched (sequential per matrix, uses same handle)
  void CorePotrfBatched(void* d_contiguous, int n, int batch, void* stream);

  /// POTRI batched
  void CorePotriBatched(void* d_contiguous, int n, int batch, void* stream);

  // ─── Symmetrize: Roundtrip ────────────────────────────────────────────

  /// Download → CPU symmetrize → Upload (одна матрица)
  void SymmetrizeRoundtrip(void* d_matrix, int n);

  /// Download → CPU symmetrize → Upload (batched)
  void SymmetrizeRoundtripBatched(void* d_contiguous, int n, int batch);

  // ─── Symmetrize: GPU Kernel (в symmetrize_gpu_rocm.cpp) ──────────────

  /// HIP kernel in-place (одна матрица)
  void SymmetrizeGpuKernel(void* d_matrix, int n, void* stream);

  /// HIP kernel (batched — цикл по матрицам)
  void SymmetrizeGpuKernelBatched(void* d_contiguous, int n, int batch,
                                   void* stream);

  // ─── Утилиты ──────────────────────────────────────────────────────────

  /// Диспетчер симметризации: выбирает Roundtrip или GpuKernel
  void Symmetrize(void* d_matrix, int n, void* stream);

  /// Диспетчер batched
  void SymmetrizeBatched(void* d_contiguous, int n, int batch, void* stream);

  /// Проверить d_info_ после pipeline (отложенная проверка, одна синхронизация).
  /// Вызывается автоматически если check_info_ == true.
  void CheckInfo(const char* context);

  bool check_info_ = true;  ///< true=проверять info (safe), false=пропустить (benchmark)

  /// Вычислить n из n_point (sqrt) или вернуть n_hint
  int ResolveMatrixSize(uint32_t n_point, int n_hint) const;

  /// CPU-side symmetrize (используется в Roundtrip)
  static void SymmetrizeUpperToFull(std::complex<float>* data, int n);
};

}  // namespace vector_algebra

#else  // !ENABLE_ROCM — Windows stub

#include <complex>
#include <vector>
#include <stdexcept>
#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "vector_algebra_types.hpp"

namespace vector_algebra {

class CholeskyInverterROCm {
public:
  explicit CholeskyInverterROCm(drv_gpu_lib::IBackend*, SymmetrizeMode = SymmetrizeMode::GpuKernel) {}
  ~CholeskyInverterROCm() = default;

  CholeskyInverterROCm(const CholeskyInverterROCm&) = delete;
  CholeskyInverterROCm& operator=(const CholeskyInverterROCm&) = delete;

  void SetSymmetrizeMode(SymmetrizeMode) {}
  SymmetrizeMode GetSymmetrizeMode() const { return SymmetrizeMode::GpuKernel; }
  void CompileKernels() {}
  void SetCheckInfo(bool) {}

  CholeskyResult Invert(const drv_gpu_lib::InputData<std::vector<std::complex<float>>>&, int = 0) {
    throw std::runtime_error("CholeskyInverterROCm: ROCm not enabled");
  }
  CholeskyResult Invert(const drv_gpu_lib::InputData<void*>&, int = 0) {
    throw std::runtime_error("CholeskyInverterROCm: ROCm not enabled");
  }
  CholeskyResult InvertBatch(const drv_gpu_lib::InputData<std::vector<std::complex<float>>>&, int) {
    throw std::runtime_error("CholeskyInverterROCm: ROCm not enabled");
  }
  CholeskyResult InvertBatch(const drv_gpu_lib::InputData<void*>&, int) {
    throw std::runtime_error("CholeskyInverterROCm: ROCm not enabled");
  }
};

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
