#pragma once

/**
 * @file fm_correlator.hpp
 * @brief FMCorrelator facade — high-level API for FM correlation
 *
 * Delegates GPU work to FMCorrelatorProcessorROCm.
 * Provides M-sequence generation (CPU LFSR) and two PrepareReference modes:
 *   - External ref vector
 *   - Internal M-seq generation from params_.lfsr_seed
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#if ENABLE_ROCM

#include "fm_correlator_types.hpp"
#include "fm_correlator_processor_rocm.hpp"
#include "interface/i_backend.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace drv_gpu_lib {

class FMCorrelator {
public:
  explicit FMCorrelator(IBackend* backend);
  ~FMCorrelator() = default;

  FMCorrelator(const FMCorrelator&) = delete;
  FMCorrelator& operator=(const FMCorrelator&) = delete;

  void SetParams(const FMCorrelatorParams& params);

  /** M-sequence generation on CPU using params_.lfsr_seed */
  std::vector<float> GenerateMSequence() const;

  /** M-sequence generation with custom seed */
  std::vector<float> GenerateMSequence(uint32_t seed) const;

  /** Prepare reference from external vector */
  void PrepareReference(const std::vector<float>& ref);

  /** Prepare reference from internal M-seq generator (seed from params) */
  void PrepareReference();

  /** Full correlation pipeline */
  FMCorrelatorResult Process(const std::vector<float>& inp);

  /**
   * Test pattern: GPU-only generation of inputs via circshift(ref, s*shift_step).
   * ref must be prepared first. No H2D for input data.
   */
  FMCorrelatorResult RunTestPattern(int shift_step = 2);

  /** Auto-batching for large S */
  FMCorrelatorResult ProcessWithBatching(const std::vector<float>& inp,
                                          int total_signals);

  const FMCorrelatorParams& GetParams() const { return params_; }

private:
  IBackend* backend_;
  FMCorrelatorParams params_;
  std::unique_ptr<FMCorrelatorProcessorROCm> processor_;
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
