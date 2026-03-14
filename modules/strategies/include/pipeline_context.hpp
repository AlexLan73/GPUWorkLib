#pragma once

/**
 * @file pipeline_context.hpp
 * @brief PipelineContext — shared state for all pipeline steps
 *
 * Ref03-C: Composition struct (non-owning). Aggregates references to
 * everything a step needs: GPU context, buffers, handles, config, result.
 *
 * Owned by Facade (AntennaProcessor_v1). Filled once in constructor,
 * d_S/d_W/result updated per process() call.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include "config/antenna_processor_config.hpp"
#include "result_types.hpp"
#include "interfaces/i_checkpoint_save.hpp"
#include "services/buffer_set.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipfft/hipfft.h>

// Forward declarations
namespace drv_gpu_lib   { class IBackend; class GpuContext; }
namespace statistics    { class StatisticsProcessor; }
namespace antenna_fft   { class AllMaximaPipelineROCm; }
namespace fft_processor { class ComplexToMagPhaseROCm; }

namespace strategies {

/// Buffer indices for the pipeline's BufferSet
enum PipelineBuf : size_t {
  kBufX = 0,            ///< [n_ant × n_samples] GEMM output
  kBufFftInput,         ///< [n_ant × nFFT] zero-padded for FFT
  kBufSpectrum,         ///< [n_ant × nFFT] FFT output (complex)
  kBufMagnitudes,       ///< [n_ant × nFFT] float |spectrum|
  kBufHammingWindow,    ///< [n_samples] precomputed Hamming window
  kBufOneMaxResults,    ///< [n_ant] OneMaxParabolaLite
  kBufMinMaxResults,    ///< [n_ant] MinMaxResult
  kBufCount
};

struct PipelineContext {
  // ── Immutable references (set once in constructor) ────────────────────
  drv_gpu_lib::IBackend* backend = nullptr;
  drv_gpu_lib::GpuContext* gpu_ctx = nullptr;
  const AntennaProcessorConfig* cfg = nullptr;

  // ── GPU handles (owned by Facade) ─────────────────────────────────────
  hipblasHandle_t hipblas_handle = nullptr;
  hipfftHandle    fft_plan = 0;

  // Streams
  hipStream_t stream_main   = nullptr;   ///< GEMM + Window+FFT
  hipStream_t stream_debug1 = nullptr;   ///< Debug 2.1 (stats on d_S)
  hipStream_t stream_debug2 = nullptr;   ///< Debug 2.2 (stats on d_X)
  hipStream_t stream_debug3 = nullptr;   ///< Debug 2.3 + post-FFT sequential
  hipStream_t stream_post_a = nullptr;   ///< Parallel post-FFT: OneMax
  hipStream_t stream_post_b = nullptr;   ///< Parallel post-FFT: AllMaxima
  hipStream_t stream_post_c = nullptr;   ///< Parallel post-FFT: MinMax

  // Events (inter-stream synchronization)
  hipEvent_t event_gemm_done = nullptr;
  hipEvent_t event_fft_done  = nullptr;
  hipEvent_t event_c1_done   = nullptr;
  hipEvent_t event_c2_done   = nullptr;

  // ── GPU buffers (owned by Facade via BufferSet) ───────────────────────
  drv_gpu_lib::BufferSet<kBufCount>* buffers = nullptr;

  // ── Sizes ─────────────────────────────────────────────────────────────
  uint32_t nFFT = 0;
  int gpu_id = 0;

  // ── Sub-processors (owned by Facade, non-owning pointers) ─────────────
  statistics::StatisticsProcessor*      stats_processor = nullptr;
  antenna_fft::AllMaximaPipelineROCm*   all_maxima_pipeline = nullptr;
  fft_processor::ComplexToMagPhaseROCm* complex_to_mag = nullptr;
  ICheckpointSave*                      checkpoint = nullptr;

  // ── Per-call inputs (set before Pipeline::Execute) ────────────────────
  const void* d_S = nullptr;   ///< Input signal (GPU)
  const void* d_W = nullptr;   ///< Weight matrix (GPU)

  // ── Accumulated result (steps write into this) ────────────────────────
  AntennaResult* result = nullptr;

  // ── Helper accessors ──────────────────────────────────────────────────
  void* buf(PipelineBuf id) const {
    return buffers->Get(static_cast<size_t>(id));
  }
};

}  // namespace strategies

#endif  // ENABLE_ROCM
