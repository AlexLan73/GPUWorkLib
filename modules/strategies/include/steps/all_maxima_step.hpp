#pragma once

/**
 * @file all_maxima_step.hpp
 * @brief AllMaximaStep — find all local maxima via AllMaximaPipelineROCm
 *
 * Ref03-C Pipeline Step.
 * Delegates to AllMaximaPipelineROCm from fft_func module.
 * Reads: kBufMagnitudes, kBufSpectrum
 * Writes: result->all_maxima
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "i_pipeline_step.hpp"
#include "pipeline_context.hpp"

#include "all_maxima_pipeline_rocm.hpp"

namespace strategies {

class AllMaximaStep : public PipelineStepBase {
public:
  const char* Name() const override { return "AllMaxima"; }

  bool IsEnabled(const AntennaProcessorConfig& cfg) const override {
    return cfg.scenario_mode == PostFftScenarioMode::ALL_REQUIRED ||
           cfg.scenario_mode == PostFftScenarioMode::ALL_MAXIMA;
  }

  void Execute(PipelineContext& ctx) override {
    auto am_result = ctx.all_maxima_pipeline->Execute(
        ctx.buf(kBufMagnitudes),
        ctx.buf(kBufSpectrum),
        ctx.cfg->n_ant,
        ctx.nFFT,
        ctx.cfg->sample_rate,
        antenna_fft::OutputDestination::CPU,
        1,      // search_start (skip DC)
        0,      // search_end (0 = nFFT/2)
        ctx.cfg->maxima_limit);

    ctx.result->all_maxima = std::move(am_result.beams);
  }
};

}  // namespace strategies

#endif  // ENABLE_ROCM
