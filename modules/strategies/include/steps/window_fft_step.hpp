#pragma once

/**
 * @file window_fft_step.hpp
 * @brief WindowFftStep — Hamming window + zero-pad + batch FFT + magnitudes
 *
 * Ref03-C Pipeline Step.
 * Reads: kBufX, kBufHammingWindow
 * Writes: kBufFftInput, kBufSpectrum, kBufMagnitudes
 * Stream: stream_main
 * Post: records event_fft_done on stream_main
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "i_pipeline_step.hpp"
#include "pipeline_context.hpp"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <stdexcept>

// Forward declaration
namespace fft_processor { struct MagPhaseParams; }

namespace strategies {

class WindowFftStep : public PipelineStepBase {
public:
  const char* Name() const override { return "WindowFFT"; }
  bool IsEnabled(const AntennaProcessorConfig&) const override { return true; }

  void Execute(PipelineContext& ctx) override {
    uint32_t n_ant = ctx.cfg->n_ant;
    uint32_t n_samples = ctx.cfg->n_samples;
    uint32_t nFFT = ctx.nFFT;
    uint32_t total_fft = n_ant * nFFT;

    // 1. Zero entire FFT input buffer (P11 — eliminates if-else in kernel)
    hipMemsetAsync(ctx.buf(kBufFftInput), 0,
                   static_cast<size_t>(total_fft) * sizeof(float) * 2,
                   ctx.stream_main);

    // 2. Fused Hamming window + copy (P13+P10+P6)
    {
      unsigned int grid_x = (n_samples + kBlockSize - 1) / kBlockSize;
      unsigned int grid_y = n_ant;
      void* d_X = ctx.buf(kBufX);
      void* d_fft_in = ctx.buf(kBufFftInput);
      void* d_window = ctx.buf(kBufHammingWindow);

      void* args[] = { &d_X, &d_fft_in, &d_window, &n_ant, &n_samples, &nFFT };

      hipError_t err = hipModuleLaunchKernel(
          ctx.gpu_ctx->GetKernel("hamming_pad_fused"),
          grid_x, grid_y, 1,
          kBlockSize, 1, 1,
          0, ctx.stream_main,
          args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("WindowFftStep: hamming_pad_fused failed");
      }
    }

    // 3. Batch FFT: kBufFftInput → kBufSpectrum
    hipfftResult fft_result = hipfftExecC2C(
        ctx.fft_plan,
        static_cast<hipfftComplex*>(ctx.buf(kBufFftInput)),
        static_cast<hipfftComplex*>(ctx.buf(kBufSpectrum)),
        HIPFFT_FORWARD);
    if (fft_result != HIPFFT_SUCCESS) {
      throw std::runtime_error("WindowFftStep: hipfftExecC2C failed");
    }

    // 4. Compute magnitudes (via ComplexToMagPhaseROCm, zero-alloc)
    ctx.complex_to_mag->ProcessMagnitudeToBuffer(
        ctx.buf(kBufSpectrum), ctx.buf(kBufMagnitudes),
        {n_ant, nFFT, 0.0f});

    // Signal FFT completion for dependent steps (debug23, post-fft)
    hipEventRecord(ctx.event_fft_done, ctx.stream_main);
  }

private:
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace strategies

#endif  // ENABLE_ROCM
