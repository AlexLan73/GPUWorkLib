#pragma once

/**
 * @file debug_stats_step.hpp
 * @brief DebugStatsStep — parameterized statistics at debug points
 *
 * Ref03-C Pipeline Step.
 * Three instances: PRE_INPUT, POST_GEMM, POST_FFT.
 * Each waits on appropriate event, computes stats, saves checkpoint.
 *
 * Cross-module: uses StatisticsProcessor (from statistics module).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "i_pipeline_step.hpp"
#include "pipeline_context.hpp"
#include "config/statistics_set.hpp"
#include "statistics_processor.hpp"

#include <hip/hip_runtime.h>

namespace strategies {

enum class DebugPoint : uint8_t { PRE_INPUT, POST_GEMM, POST_FFT };

class DebugStatsStep : public PipelineStepBase {
public:
  explicit DebugStatsStep(DebugPoint point) : point_(point) {}

  const char* Name() const override {
    switch (point_) {
      case DebugPoint::PRE_INPUT: return "DebugStats_PreInput";
      case DebugPoint::POST_GEMM: return "DebugStats_PostGEMM";
      case DebugPoint::POST_FFT:  return "DebugStats_PostFFT";
    }
    return "DebugStats_Unknown";
  }

  bool IsEnabled(const AntennaProcessorConfig& cfg) const override {
    switch (point_) {
      case DebugPoint::PRE_INPUT: return cfg.pre_input_stats != StatPreset::NONE;
      case DebugPoint::POST_GEMM: return cfg.post_gemm_stats != StatPreset::NONE;
      case DebugPoint::POST_FFT:  return cfg.post_fft_stats  != StatPreset::NONE;
    }
    return false;
  }

  void Execute(PipelineContext& ctx) override {
    statistics::StatisticsParams sp;
    sp.beam_count = ctx.cfg->n_ant;

    switch (point_) {
      case DebugPoint::PRE_INPUT:
        // No wait needed — d_S is ready
        sp.n_point = ctx.cfg->n_samples;
        ctx.result->pre_input_stats = ctx.stats_processor->ComputeStatistics(
            const_cast<void*>(ctx.d_S), sp);
        if (ctx.cfg->pre_input_stats & STAT_MEDIAN) {
          ctx.result->pre_input_medians = ctx.stats_processor->ComputeMedian(
              const_cast<void*>(ctx.d_S), sp);
        }
        ctx.checkpoint->save_c1_signal(ctx.d_S, ctx.cfg->n_ant,
                                        ctx.cfg->n_samples, ctx.cfg->sample_rate, ctx.gpu_id);
        break;

      case DebugPoint::POST_GEMM:
        // Wait for GEMM to complete
        hipStreamWaitEvent(ctx.stream_debug2, ctx.event_gemm_done, 0);
        sp.n_point = ctx.cfg->n_samples;
        ctx.result->post_gemm_stats = ctx.stats_processor->ComputeStatistics(
            ctx.buf(kBufX), sp);
        if (ctx.cfg->post_gemm_stats & STAT_MEDIAN) {
          ctx.result->post_gemm_medians = ctx.stats_processor->ComputeMedian(
              ctx.buf(kBufX), sp);
        }
        ctx.checkpoint->save_c2_data(ctx.buf(kBufX), ctx.cfg->n_ant,
                                      ctx.cfg->n_samples, ctx.cfg->sample_rate, ctx.gpu_id);
        break;

      case DebugPoint::POST_FFT:
        // Wait for FFT to complete
        hipStreamWaitEvent(ctx.stream_debug3, ctx.event_fft_done, 0);
        sp.n_point = ctx.nFFT;
        ctx.result->post_fft_stats = ctx.stats_processor->ComputeStatisticsFloat(
            ctx.buf(kBufMagnitudes), sp);
        if (ctx.cfg->post_fft_stats & STAT_MEDIAN) {
          ctx.result->post_fft_medians = ctx.stats_processor->ComputeMedianFloat(
              ctx.buf(kBufMagnitudes), sp);
        }
        ctx.checkpoint->save_c3_spectrum(ctx.buf(kBufSpectrum), ctx.cfg->n_ant,
                                          ctx.nFFT, ctx.gpu_id);
        break;
    }
  }

private:
  DebugPoint point_;
};

}  // namespace strategies

#endif  // ENABLE_ROCM
