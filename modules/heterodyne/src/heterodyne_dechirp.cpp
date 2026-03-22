#define _USE_MATH_DEFINES
/**
 * @file heterodyne_dechirp.cpp
 * @brief Heterodyne dechirp LFM - facade implementation
 *
 * Full pipeline:
 *   1. LfmConjugateGenerator -> s_ref* = conj(LFM) [OPT-4: cached]
 *   2. processor_->Dechirp()  -> s_dc = s_rx * s_ref* [OPT-3: GPU ref when possible]
 *   3. FFTProcessor           -> spectrum per antenna
 *   4. CPU argmax + parabolic interpolation -> f_beat
 *   5. R = c*T*f_beat / (2*B)
 *   6. SNR = 20*log10(peak / noise_estimate)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "heterodyne_dechirp.hpp"
#include "processors/heterodyne_processor_rocm.hpp"

// Spectrum peak finding: FFT + OnePeak (parabolic interpolation) on GPU
#include <stdexcept>
#if ENABLE_ROCM
#include "factory/spectrum_processor_factory.hpp"
#endif

#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════════════

HeterodyneDechirp::HeterodyneDechirp(
    IBackend* backend, BackendType compute_backend)
    : backend_(backend)
    , compute_backend_(compute_backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "HeterodyneDechirp: backend is null or not initialized");
  }

  switch (compute_backend) {
    case BackendType::AUTO:
    case BackendType::ROCm:
      compute_backend_ = BackendType::ROCm;
      processor_ = std::make_unique<HeterodyneProcessorROCm>(backend_);
      break;
    case BackendType::OPENCL:
      throw std::runtime_error(
          "HeterodyneDechirp: OpenCL backend not supported on this branch. Use ROCm.");
    default:
      throw std::runtime_error("HeterodyneDechirp: unsupported backend type");
  }
}

// ════════════════════════════════════════════════════════════════════════════
// SetParams
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneDechirp::SetParams(const HeterodyneParams& params) {
  params_ = params;
  params_dirty_ = true;  // OPT-4: signal to rebuild conj_gen_
}

// CPU LFM conjugate fallback removed — now using LfmConjugateGeneratorROCm

// ════════════════════════════════════════════════════════════════════════════
// OPT-4: Lazy-init conjugate generator (rebuild only when params change)
// ════════════════════════════════════════════════════════════════════════════

void HeterodyneDechirp::EnsureConjugateGenerator() {
  if (!params_dirty_ && conj_gen_) return;

  signal_gen::LfmParams lfm_p;
  lfm_p.f_start = params_.f_start;
  lfm_p.f_end   = params_.f_end;
  lfm_p.amplitude = 1.0;
  lfm_p.complex_iq = true;

  signal_gen::SystemSampling sys;
  sys.fs = params_.sample_rate;
  sys.length = static_cast<size_t>(params_.num_samples);

  conj_gen_ = std::make_unique<signal_gen::LfmConjugateGeneratorROCm>(backend_, lfm_p);
  conj_gen_->SetSampling(sys);
  params_dirty_ = false;
}

// ════════════════════════════════════════════════════════════════════════════
// Process: full pipeline from CPU data
// ════════════════════════════════════════════════════════════════════════════

HeterodyneResult HeterodyneDechirp::Process(
    const std::vector<std::complex<float>>& rx_data) {

  try {
    EnsureConjugateGenerator();

    // Generate conjugate LFM reference via ROCm (or CPU fallback)
    auto ref_cpu = conj_gen_->GenerateToCpu();
    auto dc_data = processor_->Dechirp(rx_data, ref_cpu, params_);

    // Build result: FFT + peak finding + range + SNR
    last_result_ = BuildResult(dc_data, params_);
    return last_result_;

  } catch (const std::exception& e) {
    HeterodyneResult error_result;
    error_result.success = false;
    error_result.error_message = e.what();
    last_result_ = error_result;
    return error_result;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessExternal: pipeline from external GPU buffer (cl_mem or hipDeviceptr_t)
// ════════════════════════════════════════════════════════════════════════════

HeterodyneResult HeterodyneDechirp::ProcessExternal(
    void* rx_gpu_ptr, const HeterodyneParams& params) {

  try {
    // Update params_ if caller provides different params
    // (ProcessExternal receives params explicitly)
    if (params.f_start != params_.f_start || params.f_end != params_.f_end ||
        params.sample_rate != params_.sample_rate ||
        params.num_samples != params_.num_samples ||
        params.num_antennas != params_.num_antennas) {
      params_ = params;
      params_dirty_ = true;
    }

    // OPT-4: Reuse cached conj generator
    EnsureConjugateGenerator();

    auto ref_cpu = conj_gen_->GenerateToCpu();
    auto dc_data = processor_->DechirpFromGPU(rx_gpu_ptr, ref_cpu, params);

    // Build result
    last_result_ = BuildResult(dc_data, params);
    return last_result_;

  } catch (const std::exception& e) {
    HeterodyneResult error_result;
    error_result.success = false;
    error_result.error_message = e.what();
    last_result_ = error_result;
    return error_result;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// BuildResult: FFT + peak finding per antenna + SNR
// ════════════════════════════════════════════════════════════════════════════

HeterodyneResult HeterodyneDechirp::BuildResult(
    const std::vector<std::complex<float>>& dc_data,
    const HeterodyneParams& params) {

  HeterodyneResult result;
  std::vector<antenna_fft::SpectrumResult> spec_results;

#if ENABLE_ROCM
  if (compute_backend_ == BackendType::ROCm) {
    antenna_fft::SpectrumParams spec_params;
    spec_params.antenna_count = static_cast<uint32_t>(params.num_antennas);
    spec_params.n_point = static_cast<uint32_t>(params.num_samples);
    spec_params.repeat_count = 1;
    spec_params.sample_rate = params.sample_rate;
    spec_params.search_range = 5000;
    spec_params.peak_mode = antenna_fft::PeakSearchMode::ONE_PEAK;
    spec_params.memory_limit = 0.8f;

    auto processor = antenna_fft::SpectrumProcessorFactory::Create(
        BackendType::ROCm, backend_);
    processor->Initialize(spec_params);
    spec_results = processor->ProcessFromCPU(dc_data);
  } else
#endif
  {
    throw std::runtime_error(
        "HeterodyneDechirp::BuildResult: non-ROCm backend not supported "
        "on this branch. Use ROCm backend.");
  }

  float bandwidth = params.GetBandwidth();
  result.antennas.resize(params.num_antennas);

  for (int ant = 0; ant < params.num_antennas; ++ant) {
    auto& sr = spec_results[ant];
    float f_beat = sr.interpolated.refined_frequency;
    float refined_bin = static_cast<float>(sr.interpolated.index)
                        + sr.interpolated.freq_offset;
    float peak_mag = sr.interpolated.magnitude;

    float range = HeterodyneResult::CalcRange(
        f_beat, params.sample_rate, params.num_samples, bandwidth);

    // SNR computation: peak vs noise estimate from neighboring points
    float left_mag  = sr.left_point.magnitude;
    float right_mag = sr.right_point.magnitude;
    float noise_est = (left_mag + right_mag) * 0.5f;
    float snr_db = 0.0f;
    constexpr float kEpsilon = 1e-12f;
    if (noise_est > kEpsilon) {
      snr_db = 20.0f * std::log10(peak_mag / noise_est);
    } else if (peak_mag > kEpsilon) {
      snr_db = 100.0f;  // effectively infinite SNR
    }

    result.antennas[ant] = AntennaDechirpResult{
        ant,        // antenna_idx
        f_beat,     // f_beat_hz
        refined_bin,// f_beat_bin
        range,      // range_m
        peak_mag,   // peak_amplitude
        snr_db      // peak_snr_db
    };

    result.max_positions.push_back(refined_bin);
  }

  result.success = true;
  return result;
}

}  // namespace drv_gpu_lib
