#pragma once

/**
 * @file one_max_step.hpp
 * @brief OneMaxStep — per-beam one peak + 3-point parabolic interpolation
 *
 * Ref03-C Pipeline Step.
 * Kernel: one_max_no_phase
 * Reads: kBufMagnitudes, kBufSpectrum
 * Writes: kBufOneMaxResults → D2H → result->one_max
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "i_pipeline_step.hpp"
#include "pipeline_context.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>

namespace strategies {

class OneMaxStep : public PipelineStepBase {
public:
  const char* Name() const override { return "OneMaxParabola"; }

  bool IsEnabled(const AntennaProcessorConfig& cfg) const override {
    return cfg.scenario_mode == PostFftScenarioMode::ALL_REQUIRED ||
           cfg.scenario_mode == PostFftScenarioMode::ONE_MAX_PARABOLA;
  }

  /// Set stream for parallel execution (default: stream_debug3)
  void SetStream(hipStream_t s) { override_stream_ = s; }

  void Execute(PipelineContext& ctx) override {
    uint32_t n_ant = ctx.cfg->n_ant;
    float sr = ctx.cfg->sample_rate;
    hipStream_t stream = override_stream_ ? override_stream_ : ctx.stream_debug3;

    void* d_mag = ctx.buf(kBufMagnitudes);
    void* d_spec = ctx.buf(kBufSpectrum);
    void* d_results = ctx.buf(kBufOneMaxResults);

    void* args[] = { &d_mag, &d_spec, &d_results, &n_ant, &ctx.nFFT, &sr };

    hipError_t err = hipModuleLaunchKernel(
        ctx.gpu_ctx->GetKernel("one_max_no_phase"),
        1, n_ant, 1,
        kBlockSize, 1, 1,
        0, stream,
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("OneMaxStep: kernel launch failed");
    }

    // Sync + D2H copy
    hipStreamSynchronize(stream);
    ctx.result->one_max.resize(n_ant);
    hipMemcpy(ctx.result->one_max.data(), d_results,
              n_ant * sizeof(OneMaxParabolaLite), hipMemcpyDeviceToHost);
  }

private:
  hipStream_t override_stream_ = nullptr;
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace strategies

#endif  // ENABLE_ROCM
