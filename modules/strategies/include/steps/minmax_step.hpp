#pragma once

/**
 * @file minmax_step.hpp
 * @brief MinMaxStep — per-beam global MIN+MAX + dynamic range
 *
 * Ref03-C Pipeline Step.
 * Kernel: global_minmax
 * Reads: kBufMagnitudes
 * Writes: kBufMinMaxResults → D2H → result->minmax
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

class MinMaxStep : public PipelineStepBase {
public:
  const char* Name() const override { return "GlobalMinMax"; }

  bool IsEnabled(const AntennaProcessorConfig& cfg) const override {
    return cfg.scenario_mode == PostFftScenarioMode::ALL_REQUIRED ||
           cfg.scenario_mode == PostFftScenarioMode::GLOBAL_MINMAX;
  }

  /// Set stream for parallel execution (default: stream_debug3)
  void SetStream(hipStream_t s) { override_stream_ = s; }

  void Execute(PipelineContext& ctx) override {
    uint32_t n_ant = ctx.cfg->n_ant;
    float sr = ctx.cfg->sample_rate;
    hipStream_t stream = override_stream_ ? override_stream_ : ctx.stream_debug3;

    void* d_mag = ctx.buf(kBufMagnitudes);
    void* d_results = ctx.buf(kBufMinMaxResults);

    void* args[] = { &d_mag, &d_results, &n_ant, &ctx.nFFT, &sr };

    hipError_t err = hipModuleLaunchKernel(
        ctx.gpu_ctx->GetKernel("global_minmax"),
        1, n_ant, 1,
        kBlockSize, 1, 1,
        0, stream,
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MinMaxStep: kernel launch failed");
    }

    hipStreamSynchronize(stream);
    ctx.result->minmax.resize(n_ant);
    hipMemcpy(ctx.result->minmax.data(), d_results,
              n_ant * sizeof(MinMaxResult), hipMemcpyDeviceToHost);

    ctx.checkpoint->save_c3_minmax(ctx.result->minmax.data(), n_ant, ctx.gpu_id);
  }

private:
  hipStream_t override_stream_ = nullptr;
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace strategies

#endif  // ENABLE_ROCM
