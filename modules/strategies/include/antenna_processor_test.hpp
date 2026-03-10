#pragma once

/**
 * @file antenna_processor_test.hpp
 * @brief AntennaProcessorTest - step-by-step test wrapper over AntennaProcessor_v1
 *
 * Inherits from AntennaProcessor_v1 and exposes individual pipeline steps
 * for debugging, validation, and Python step-by-step testing.
 *
 * Each step_ method executes one pipeline stage and returns data to CPU
 * for comparison with NumPy/SciPy reference.
 *
 * @date 2026-03-07
 */

#include "antenna_processor_v1.hpp"
#include "weight_generator.hpp"

#include <complex>
#include <vector>

namespace strategies {

class AntennaProcessorTest : public AntennaProcessor_v1 {
public:
  using AntennaProcessor_v1::AntennaProcessor_v1;  // Inherit constructors

  // ========================================================================
  // Step-by-step API (Python-callable)
  // ========================================================================

  /**
   * @brief Step 0: Prepare input — store d_S, d_W pointers
   */
  void step_0_prepare_input(const void* d_S, const void* d_W) {
    d_S_ = d_S;
    d_W_ = d_W;
  }

  /**
   * @brief Step 1: Debug point 2.1 — stats on d_S
   * @return Statistics on input signal
   */
  AntennaResult step_1_debug_input() {
    AntennaResult result;
    do_debug_point_21(d_S_, result);
#if ENABLE_ROCM
    hipStreamSynchronize(nullptr);  // Ensure stats complete
#endif
    return result;
  }

  /**
   * @brief Step 2: GEMM — X = W * S
   * @return d_X copied to CPU [n_ant x n_samples] complex<float>
   */
  std::vector<std::complex<float>> step_2_gemm() {
    do_gemm(d_S_, d_W_);
#if ENABLE_ROCM
    hipDeviceSynchronize();
#endif
    return copy_buffer_to_cpu(get_d_X(),
        config().n_ant * config().n_samples);
  }

  /**
   * @brief Step 3: Debug point 2.2 — stats on d_X
   */
  AntennaResult step_3_debug_post_gemm() {
    AntennaResult result;
    do_debug_point_22(result);
#if ENABLE_ROCM
    hipDeviceSynchronize();
#endif
    return result;
  }

  /**
   * @brief Step 4: Window + FFT
   * @return d_spectrum copied to CPU [n_ant x nFFT] complex<float>
   */
  std::vector<std::complex<float>> step_4_window_fft() {
    do_window_fft();
#if ENABLE_ROCM
    hipDeviceSynchronize();
#endif
    return copy_buffer_to_cpu(get_d_spectrum(),
        config().n_ant * get_nFFT());
  }

  /**
   * @brief Step 5: Debug point 2.3 — stats on |spectrum|
   */
  AntennaResult step_5_debug_post_fft() {
    AntennaResult result;
    do_debug_point_23(result);
#if ENABLE_ROCM
    hipDeviceSynchronize();
#endif
    return result;
  }

  /**
   * @brief Step 6.1: OneMax + Parabola (no phase)
   */
  AntennaResult step_6_1_one_max_parabola() {
    auto saved = config().scenario_mode;
    const_cast<AntennaProcessorConfig&>(config()).scenario_mode =
        PostFftScenarioMode::ONE_MAX_PARABOLA;
    AntennaResult result;
    do_run_post_fft_scenarios(result);
    const_cast<AntennaProcessorConfig&>(config()).scenario_mode = saved;
    return result;
  }

  /**
   * @brief Step 6.2: AllMaxima
   */
  AntennaResult step_6_2_all_maxima() {
    auto saved = config().scenario_mode;
    const_cast<AntennaProcessorConfig&>(config()).scenario_mode =
        PostFftScenarioMode::ALL_MAXIMA;
    AntennaResult result;
    do_run_post_fft_scenarios(result);
    const_cast<AntennaProcessorConfig&>(config()).scenario_mode = saved;
    return result;
  }

  /**
   * @brief Step 6.3: GlobalMinMax
   */
  AntennaResult step_6_3_global_minmax() {
    auto saved = config().scenario_mode;
    const_cast<AntennaProcessorConfig&>(config()).scenario_mode =
        PostFftScenarioMode::GLOBAL_MINMAX;
    AntennaResult result;
    do_run_post_fft_scenarios(result);
    const_cast<AntennaProcessorConfig&>(config()).scenario_mode = saved;
    return result;
  }

  /**
   * @brief Full pipeline (all steps + all scenarios)
   */
  AntennaResult process_full() {
    return process(d_S_, d_W_);
  }

  /**
   * @brief Full pipeline using external weights loaded via set_external_weights()
   *
   * Requires prior call to set_external_weights().
   * d_S must be set via step_0_prepare_input or step_0_signal_only.
   */
  AntennaResult process_full_managed_w() {
    return process(d_S_, get_managed_weights_ptr());
  }

  /**
   * @brief Step 0 signal-only variant — uses pre-loaded managed weights
   *
   * Call after set_external_weights() to avoid re-uploading W on every frame.
   * Only updates d_S_; d_W_ is set to the internally managed GPU pointer.
   */
  void step_0_signal_only(const void* d_S) {
    d_S_ = d_S;
    d_W_ = get_managed_weights_ptr();
  }

  // Getters for test access
  uint32_t test_get_nFFT() const { return get_nFFT(); }

private:
  const void* d_S_ = nullptr;
  const void* d_W_ = nullptr;

  std::vector<std::complex<float>> copy_buffer_to_cpu(
      const void* d_buf, size_t num_complex_elements)
  {
    std::vector<std::complex<float>> host(num_complex_elements);
#if ENABLE_ROCM
    hipMemcpy(host.data(), d_buf,
              num_complex_elements * sizeof(std::complex<float>),
              hipMemcpyDeviceToHost);
#endif
    return host;
  }
};

}  // namespace strategies
