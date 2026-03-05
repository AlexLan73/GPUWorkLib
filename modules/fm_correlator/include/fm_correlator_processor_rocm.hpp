#pragma once

/**
 * @file fm_correlator_processor_rocm.hpp
 * @brief ROCm backend for FM Correlator (hipFFT + HIP kernels)
 *
 * GPU pipeline:
 *   Stream 0: H2D(ref) -> apply_cyclic_shifts -> C2C FFT(ref)
 *   Stream 1: H2D(inp) -> R2C FFT(inp)
 *   Sync both streams
 *   Stream 0: multiply_conj_fused -> C2R IFFT -> extract_magnitudes -> D2H
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#if ENABLE_ROCM

#include "fm_correlator_types.hpp"
#include "interface/i_backend.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hipfft/hipfft.h>

// ROCm-бэкенд FM-коррелятора: hipFFT + hiprtc-кернелы.
// Владеет GPU-буферами, hipFFT-планами и HIP-потоками.
// Порядок вызовов: SetParams() → PrepareReference() → Process()/RunTestPattern().
class FMCorrelatorProcessorROCm {
public:
  // Создаёт два HIP-потока для параллельной обработки ref и inp.
  // backend — не владеем.
  explicit FMCorrelatorProcessorROCm(IBackend* backend);
  ~FMCorrelatorProcessorROCm();

  FMCorrelatorProcessorROCm(const FMCorrelatorProcessorROCm&) = delete;
  FMCorrelatorProcessorROCm& operator=(const FMCorrelatorProcessorROCm&) = delete;

  // Применить параметры: compile (первый раз), free старые буферы/планы,
  // allocate под новые размеры. Сбрасывает ref_prepared_ — эталон потеряется.
  void SetParams(const FMCorrelatorParams& params);

  // H2D(ref) → apply_cyclic_shifts (GPU) → C2C FFT батча K на stream0.
  // После этого d_ref_complex_ содержит спектры всех K сдвигов.
  void PrepareReference(const std::vector<float>& ref);

  // H2D(inp) на stream1, R2C FFT на stream1; sync; RunCorrelationPipeline().
  FMCorrelatorResult Process(const std::vector<float>& inp);

  // GPU-генерация inp (circshift(ref, s*shift_step)) на stream1, R2C FFT;
  // sync; RunCorrelationPipeline(). Нет H2D для входных данных — только GPU.
  FMCorrelatorResult RunTestPattern(int shift_step);

  // Авто-батчинг: делит total_signals на батчи по BatchManager,
  // для каждого батча переконфигурирует num_signals и вызывает Process().
  FMCorrelatorResult ProcessWithBatching(const std::vector<float>& inp,
                                          int total_signals);

  bool IsReferencePrepared() const { return ref_prepared_; }

private:
  // Hiprtc: compile один раз лениво при первом SetParams(). Загружает module_
  // и извлекает 4 hipFunction_t по именам.
  void CompileKernels();
  void AllocateBuffers();
  void FreeBuffers();
  void CreatePlans();
  void DestroyPlans();
  // Вызывается из деструктора: DestroyPlans → FreeBuffers → unload module → destroy streams.
  void ReleaseAll();

  // Финальная часть пайплайна, общая для Process() и RunTestPattern():
  // multiply_conj_fused → C2R IFFT → extract_magnitudes → D2H peaks.
  FMCorrelatorResult RunCorrelationPipeline();

  IBackend* backend_;   ///< Не владеем — lifetime управляется снаружи
  FMCorrelatorParams params_;

  // stream0: ref-path (H2D ref, shifts, C2C FFT) + финальная корреляция
  // stream1: inp-path (H2D inp или gen_test_inputs, R2C FFT) — параллельно stream0
  // Синхронизируются перед multiply_conj_fused.
  hipStream_t stream0_ = nullptr;
  hipStream_t stream1_ = nullptr;

  // hipFFT-планы создаются в SetParams() и живут до следующего SetParams()/деструктора.
  // ВАЖНО: планы привязаны к потокам через hipfftSetStream() — без этого план
  // выполнится на default stream и нарушит параллелизм ref/inp.
  hipfftHandle plan_ref_  = 0;  ///< C2C Forward, batch=K (для эталонных сдвигов)
  hipfftHandle plan_inp_  = 0;  ///< R2C Forward, batch=S (для входных сигналов)
  hipfftHandle plan_corr_ = 0;  ///< C2R Inverse, batch=S*K (для корреляции)
  bool plans_created_ = false;

  // GPU-буферы. Все выделяются в AllocateBuffers() при SetParams().
  float*  d_ref_float_   = nullptr;  ///< [N] float  — исходный ref до apply_shifts
  void*   d_ref_complex_ = nullptr;  ///< [K × N] float2 — сдвиги + их C2C FFT in-place
  float*  d_inp_float_   = nullptr;  ///< [S × N] float  — входные сигналы (real)
  void*   d_inp_fft_     = nullptr;  ///< [S × (N/2+1)] float2 — R2C результат (hermitian!)
  void*   d_corr_fft_    = nullptr;  ///< [S × K × (N/2+1)] float2 — conj(ref)×inp спектры
  float*  d_corr_time_   = nullptr;  ///< [S × K × N] float — C2R IFFT (вещественный)
  float*  d_peaks_       = nullptr;  ///< [S × K × n_kg] float — финальные пики

  // Один hipModule содержит все 4 кернела, компилируется hiprtc один раз.
  hipModule_t   module_ = nullptr;
  hipFunction_t fn_apply_shifts_    = nullptr;  ///< float[N] → float2[K×N] с циклическими сдвигами
  hipFunction_t fn_multiply_conj_   = nullptr;  ///< conj(ref_fft[k]) × inp_fft[s]
  hipFunction_t fn_extract_mag_     = nullptr;  ///< |corr_time[j]| / N → peaks
  hipFunction_t fn_gen_test_inputs_ = nullptr;  ///< circshift(ref, s*step) → inp (тест без H2D)
  bool kernels_compiled_ = false;

  bool ref_prepared_ = false;       ///< true после успешного PrepareReference()
  bool buffers_allocated_ = false;
  int  current_batch_S_ = 0;        ///< S для которого выделены текущие буферы/планы

  // 256 threads/block — оптимум для RDNA/CDNA: кратно warp_size=64 (4 варпа),
  // достаточно для хорошего occupancy, не вызывает register spill для этих кернелов.
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace drv_gpu_lib

#endif  // ENABLE_ROCM
