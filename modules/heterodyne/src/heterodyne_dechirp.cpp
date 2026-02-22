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
#include "processors/heterodyne_processor_opencl.hpp"
#include "processors/heterodyne_processor_rocm.hpp"

// Spectrum peak finding: FFT + OnePeak (parabolic interpolation) on GPU
#include "spectrum_maxima_finder.h"

#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace drv_gpu_lib {

// ════════════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════════════

HeterodyneDechirp::HeterodyneDechirp(
    IBackend* backend, BackendType compute_backend)
    : backend_(backend) {

  if (!backend_ || !backend_->IsInitialized()) {
    throw std::runtime_error(
        "HeterodyneDechirp: backend is null or not initialized");
  }

  switch (compute_backend) {
    case BackendType::OPENCL:
    case BackendType::AUTO:
      processor_ = std::make_unique<HeterodyneProcessorOpenCL>(backend_);
      break;
    case BackendType::ROCm:
      processor_ = std::make_unique<HeterodyneProcessorROCm>(backend_);
      break;
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

  conj_gen_ = std::make_unique<signal_gen::LfmConjugateGenerator>(backend_, lfm_p);
  conj_gen_->SetSampling(sys);
  params_dirty_ = false;
}

// ════════════════════════════════════════════════════════════════════════════
// Process: full pipeline from CPU data
// ════════════════════════════════════════════════════════════════════════════

HeterodyneResult HeterodyneDechirp::Process(
    const std::vector<std::complex<float>>& rx_data) {

  try {
    // OPT-4: Reuse cached conj generator
    EnsureConjugateGenerator();

    // Generate ref on CPU, dechirp on GPU
    // (OPT-3 GPU-ref path is used in ProcessExternal() where rx is already on GPU)
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
// ProcessExternal: pipeline from external cl_mem buffer
// ════════════════════════════════════════════════════════════════════════════

HeterodyneResult HeterodyneDechirp::ProcessExternal(
    void* rx_cl_mem, const HeterodyneParams& params) {

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

    // OPT-3: Generate ref on GPU, dechirp both on GPU (no PCIe for ref)
    cl_mem ref_gpu = conj_gen_->GenerateToGpu();
    auto dc_data = processor_->DechirpWithGPURef(rx_cl_mem, ref_gpu, params);
    clReleaseMemObject(ref_gpu);

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

  // SpectrumMaximaFinder: FFT + OnePeak (parabolic interpolation) on GPU
  antenna_fft::SpectrumMaximaFinder finder(backend_);

  antenna_fft::InputData<std::vector<std::complex<float>>> input;
  input.antenna_count = static_cast<size_t>(params.num_antennas);
  input.n_point = static_cast<size_t>(params.num_samples);
  input.data = dc_data;
  input.repeat_count = 1;
  input.sample_rate = params.sample_rate;

  auto spec_results = finder.Process(input,
      antenna_fft::PeakSearchMode::ONE_PEAK,
      antenna_fft::DriverType::OPENCL);

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
