#pragma once

/**
 * @file snr_estimator_op.hpp
 * @brief SnrEstimatorOp — Layer 5 Ref03 Op для SNR-estimator (SNR_05)
 *
 * Pipeline:
 *   1. gather_decimated           — вырезка подвыборки [n_ant_out × n_actual]
 *   2. ProcessMagnitudesToGPU     — pad(Hann) → FFT → |X|²
 *   3. peak_cfar                  — argmax + CA-CFAR per antenna
 *   4. MedianRadixSortOp::ExecuteFloat(1, n_ant_out) — медиана по антеннам
 *
 * Калибровано Python моделью (PyPanelAntennas/SNR/, P_correct=97.9%).
 *
 * Зависит от kernels:
 *   - gather_decimated       (из gather_decimated_kernel.hpp)
 *   - peak_cfar              (из peak_cfar_kernel.hpp)
 *   Оба компилируются facade'ом (StatisticsProcessor::EnsureCompiled)
 *   через конкатенацию source strings.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-04-09
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"
#include "interface/i_backend.hpp"
#include "statistics_types.hpp"
#include "operations/median_radix_sort_op.hpp"
#include "fft_processor_rocm.hpp"

#include <hip/hip_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <cstring>

namespace statistics {

/**
 * @brief SnrEstimatorOp — Layer 5 Op for full SNR CFAR pipeline.
 *
 * Lifecycle:
 *   1. op.SetupFft(backend)       — создать FFTProcessorROCm (один раз)
 *   2. op.Initialize(ctx)         — attach GpuContext (standard GpuKernelOp API)
 *   3. op.Execute(gpu_input, ..., config, result)  — выполнить pipeline
 *   4. op.Release()               — освободить ресурсы
 */
class SnrEstimatorOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "SnrEstimator"; }

  /**
   * @brief Setup FFT processor — вызывается однажды до Initialize.
   * @param fft_backend IBackend для FFTProcessorROCm (shared с facade)
   *
   * Создаёт свой FFTProcessorROCm. Это отдельный модуль со своим GpuContext
   * (своя компиляция FFT kernels) — НЕ конфликтует с нашим statistics ctx_.
   */
  void SetupFft(drv_gpu_lib::IBackend* fft_backend) {
    fft_processor_ = std::make_unique<fft_processor::FFTProcessorROCm>(fft_backend);
  }

  /**
   * @brief Execute full SNR pipeline: gather → FFT|X|² → CFAR → median.
   *
   * @param gpu_input       Complex<float>* [n_antennas × n_samples] (на GPU)
   * @param n_antennas      Число входных антенн
   * @param n_samples       Сэмплов на антенну
   * @param config          Validated SnrEstimationConfig
   * @param out_result      Result struct (populated by this method)
   */
  void Execute(void* gpu_input,
               uint32_t n_antennas, uint32_t n_samples,
               const SnrEstimationConfig& config,
               SnrEstimationResult& out_result) {
    config.Validate();

    if (!gpu_input) {
      throw std::invalid_argument("SnrEstimatorOp::Execute: gpu_input is null");
    }
    if (!fft_processor_) {
      throw std::runtime_error(
          "SnrEstimatorOp::Execute: SetupFft() не был вызван");
    }

    // 1. Compute auto-parameters (0 → default from snr_defaults)
    const uint32_t target_n_fft = (config.target_n_fft > 0)
        ? config.target_n_fft
        : snr_defaults::kTargetNFft;

    const uint32_t step_samples = (config.step_samples > 0)
        ? config.step_samples
        : CeilDiv(n_samples, target_n_fft);

    const uint32_t step_antennas = (config.step_antennas > 0)
        ? config.step_antennas
        : CeilDiv(n_antennas, snr_defaults::kTargetAntennasMedian);

    const uint32_t n_actual  = n_samples / step_samples;
    const uint32_t n_ant_out = CeilDiv(n_antennas, step_antennas);

    if (n_actual == 0 || n_ant_out == 0) {
      throw std::invalid_argument(
          "SnrEstimatorOp: degenerate sizes — n_actual=" +
          std::to_string(n_actual) + " n_ant_out=" + std::to_string(n_ant_out));
    }

    // Дополнительная проверка на фактическом n_actual после децимации
    if (2u * (config.guard_bins + config.ref_bins) + 1u >= n_actual) {
      throw std::invalid_argument(
          "SnrEstimatorOp: ref window (2*(guard+ref)+1) >= n_actual=" +
          std::to_string(n_actual));
    }

    // 2. Allocate shared buffers (slots из statistics::shared_buf)
    const size_t gather_bytes =
        (size_t)n_ant_out * (size_t)n_actual * sizeof(float) * 2;
    ctx_->RequireShared(shared_buf::kGatherOutput, gather_bytes);

    // nFFT pre-allocation: используем NextPowerOf2(n_actual) как оценку.
    // Фактический nFFT берём из fft_processor_->GetNFFT() после вызова.
    const uint32_t n_fft_est = NextPowerOf2(n_actual);
    const size_t mag_bytes =
        (size_t)n_ant_out * (size_t)n_fft_est * sizeof(float);
    ctx_->RequireShared(shared_buf::kFftMagSquared, mag_bytes);

    ctx_->RequireShared(shared_buf::kSnrPerAntenna,
                        (size_t)n_ant_out * sizeof(float));

    // 3. Stage 1: gather_decimated kernel
    ExecuteGather(gpu_input, n_antennas, n_samples,
                  step_antennas, step_samples, n_ant_out, n_actual);

    // 4. Stage 2: FFT → |X|² через FFTProcessorROCm (с Hann window!)
    fft_processor::FFTProcessorParams fft_params;
    fft_params.beam_count   = n_ant_out;
    fft_params.n_point      = n_actual;
    fft_params.repeat_count = 1;
    fft_params.sample_rate  = 1.0f;   // не используется — мы не читаем частоты

    // КРИТИЧНО: window=config.window (default Hann из snr_defaults).
    // Калибровано Python Эксп.5 — без Hann −27 dB bias от sinc sidelobes!
    fft_processor_->ProcessMagnitudesToGPU(
        ctx_->GetShared(shared_buf::kGatherOutput),
        ctx_->GetShared(shared_buf::kFftMagSquared),
        fft_params,
        /*squared=*/true,
        config.window);

    const uint32_t n_fft = fft_processor_->GetNFFT();

    // 5. Stage 3: peak_cfar kernel
    ExecutePeakCfar(n_ant_out, n_fft,
                    config.guard_bins, config.ref_bins,
                    config.search_full_spectrum);

    // 6. Stage 4: median по антеннам через MedianRadixSortOp::ExecuteFloat
    //    median_op_ читает из shared_buf::kMagnitudes, пишет в kMediansCompact.
    //    Копируем SNR-per-antenna → kMagnitudes (D2D, async).
    const size_t snr_bytes = (size_t)n_ant_out * sizeof(float);
    ctx_->RequireShared(shared_buf::kMagnitudes, snr_bytes);
    hipError_t err = hipMemcpyAsync(
        ctx_->GetShared(shared_buf::kMagnitudes),
        ctx_->GetShared(shared_buf::kSnrPerAntenna),
        snr_bytes,
        hipMemcpyDeviceToDevice,
        stream());
    if (err != hipSuccess) {
      throw std::runtime_error(
          "SnrEstimatorOp: D2D copy snr→magnitudes failed: " +
          std::string(hipGetErrorString(err)));
    }

    // Один "beam" с n_ant_out сэмплами → медиана по антеннам.
    median_op_.ExecuteFloat(/*beam_count=*/1, /*n_point=*/n_ant_out);

    // 7. D2H — читаем один float (медиана SNR_db)
    float median_snr_db = 0.0f;
    err = hipMemcpyAsync(
        &median_snr_db,
        ctx_->GetShared(shared_buf::kMediansCompact),
        sizeof(float),
        hipMemcpyDeviceToHost,
        stream());
    if (err != hipSuccess) {
      throw std::runtime_error(
          "SnrEstimatorOp: D2H median read failed: " +
          std::string(hipGetErrorString(err)));
    }
    hipStreamSynchronize(stream());

    // 8. Populate result struct (БЕЗ BranchType — его считает BranchSelector)
    out_result.snr_db_global       = median_snr_db;
    out_result.used_antennas       = n_ant_out;
    out_result.used_bins           = n_fft;
    out_result.actual_step_samples = step_samples;
    out_result.n_actual            = n_actual;
    out_result.snr_db_per_antenna.clear();  // опционально — пока не заполняем
  }

protected:
  /// Called after ctx_ set by Initialize(). Attach child Op.
  void OnInitialize() override {
    // median_op_ разделяет тот же GpuContext (stream, buffers).
    median_op_.Initialize(*ctx_);
  }

  void OnRelease() override {
    median_op_.Release();
    fft_processor_.reset();
  }

private:
  std::unique_ptr<fft_processor::FFTProcessorROCm> fft_processor_;
  MedianRadixSortOp median_op_;

  static uint32_t CeilDiv(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
  }

  /// NextPowerOf2 — совпадает с FFTProcessorROCm::NextPowerOf2 (fft_processor_rocm.cpp:559).
  /// Используется только для pre-allocation kFftMagSquared. Реальный nFFT
  /// берётся через fft_processor_->GetNFFT() после ProcessMagnitudesToGPU.
  static uint32_t NextPowerOf2(uint32_t n) {
    if (n == 0) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
  }

  void ExecuteGather(void* gpu_input,
                     uint32_t n_antennas, uint32_t n_samples,
                     uint32_t step_ant, uint32_t step_samp,
                     uint32_t n_ant_out, uint32_t n_samp_out)
  {
    // Launch: grid(ceil(n_ant_out/64), 1), block(64, 1)
    constexpr unsigned int kBlock = 64;
    unsigned int grid_x = (n_ant_out + kBlock - 1u) / kBlock;

    unsigned int ns     = n_samples;
    unsigned int nso    = n_samp_out;
    unsigned int sa     = step_ant;
    unsigned int ss     = step_samp;
    unsigned int nao    = n_ant_out;

    void* gather_out = ctx_->GetShared(shared_buf::kGatherOutput);

    void* args[] = { &gpu_input, &gather_out, &ns, &nso, &sa, &ss, &nao };

    hipError_t err = hipModuleLaunchKernel(
        kernel("gather_decimated"),
        grid_x, 1, 1,
        kBlock, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("SnrEstimatorOp gather_decimated: " +
                                std::string(hipGetErrorString(err)));
    }
  }

  void ExecutePeakCfar(uint32_t n_ant_out, uint32_t n_fft,
                       uint32_t guard_bins, uint32_t ref_bins,
                       bool search_full_spectrum)
  {
    // search_full_spectrum управляется через параметр nFFT:
    // true  → передаём полный nFFT (search over [0..nFFT))
    // false → передаём nFFT/2 (только [0..nFFT/2), положительные частоты)
    unsigned int search_nfft = search_full_spectrum ? n_fft : (n_fft / 2u);

    // Launch: grid(n_ant_out, 1, 1), block(256, 1, 1)
    // Один блок = одна антенна (см. peak_cfar_kernel.hpp)
    constexpr unsigned int kBlock = 256;
    unsigned int grid_x = n_ant_out;

    unsigned int snf = search_nfft;
    unsigned int gb  = guard_bins;
    unsigned int rb  = ref_bins;

    void* mag_sq = ctx_->GetShared(shared_buf::kFftMagSquared);
    void* snr_out = ctx_->GetShared(shared_buf::kSnrPerAntenna);

    void* args[] = { &mag_sq, &snr_out, &snf, &gb, &rb };

    hipError_t err = hipModuleLaunchKernel(
        kernel("peak_cfar"),
        grid_x, 1, 1,
        kBlock, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("SnrEstimatorOp peak_cfar: " +
                                std::string(hipGetErrorString(err)));
    }
  }
};

}  // namespace statistics

#endif  // ENABLE_ROCM
