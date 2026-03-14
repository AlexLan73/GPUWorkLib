#pragma once

/**
 * @file gemm_step.hpp
 * @brief GemmStep — hipBLAS CGEMM: X = W × S
 *
 * Ref03-C Pipeline Step.
 * Reads: d_S, d_W from PipelineContext
 * Writes: kBufX
 * Stream: stream_main (implicit via hipblas_handle binding)
 * Post: records event_gemm_done on stream_main
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "i_pipeline_step.hpp"
#include "pipeline_context.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdexcept>

namespace strategies {

class GemmStep : public PipelineStepBase {
public:
  const char* Name() const override { return "GEMM"; }
  bool IsEnabled(const AntennaProcessorConfig&) const override { return true; }

  void Execute(PipelineContext& ctx) override {
    const int M = static_cast<int>(ctx.cfg->n_samples);
    const int N = static_cast<int>(ctx.cfg->n_ant);
    const int K = static_cast<int>(ctx.cfg->n_ant);

    hipComplex alpha = {1.0f, 0.0f};
    hipComplex beta  = {0.0f, 0.0f};

    // Row-major trick: swap A/B in column-major hipBLAS
    hipblasStatus_t status = hipblasCgemm(
        ctx.hipblas_handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        M, N, K,
        &alpha,
        static_cast<const hipComplex*>(ctx.d_S), M,
        static_cast<const hipComplex*>(ctx.d_W), K,
        &beta,
        static_cast<hipComplex*>(ctx.buf(kBufX)), M);

    if (status != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("GemmStep: hipblasCgemm failed");
    }

    // Signal GEMM completion for dependent steps (debug22, window_fft)
    hipEventRecord(ctx.event_gemm_done, ctx.stream_main);
  }
};

}  // namespace strategies

#endif  // ENABLE_ROCM
