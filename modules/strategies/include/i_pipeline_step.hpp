#pragma once

/**
 * @file i_pipeline_step.hpp
 * @brief IPipelineStep — interface for pipeline steps + PipelineStepBase
 *
 * Ref03-C: Each pipeline stage implements IPipelineStep.
 * Steps get all resources from PipelineContext (not GpuContext directly).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "interface/i_gpu_operation.hpp"

namespace strategies {

struct PipelineContext;
struct AntennaProcessorConfig;

/// Interface for pipeline steps
class IPipelineStep : public drv_gpu_lib::IGpuOperation {
public:
  /// Execute this step using shared context
  virtual void Execute(PipelineContext& ctx) = 0;

  /// Return true if this step should run given current config
  virtual bool IsEnabled(const AntennaProcessorConfig& cfg) const = 0;
};

/// Convenience base: steps get context from PipelineContext, not GpuContext
class PipelineStepBase : public IPipelineStep {
public:
  // IGpuOperation defaults (pipeline steps don't use GpuContext directly)
  void Initialize(drv_gpu_lib::GpuContext&) override {}
  bool IsReady() const override { return true; }
  void Release() override {}
};

}  // namespace strategies

#endif  // ENABLE_ROCM
