#pragma once

/**
 * @file py_strategies_rocm.hpp
 * @brief Python wrapper for AntennaProcessorTest (ROCm strategies module)
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Usage from Python:
 *   ctx = gpuworklib.ROCmGPUContext(0)
 *   proc = gpuworklib.AntennaProcessorTest(ctx, n_ant=5, n_samples=8000,
 *              sample_rate=12e6, signal_frequency_hz=2e6)
 *   proc.step_0_prepare_input(d_S_array, W_array)
 *   gemm = proc.step_2_gemm()
 *   spectrum = proc.step_4_window_fft()
 *   result = proc.step_6_1_one_max_parabola()
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-07
 */

#include "antenna_processor_test.hpp"
#include "weight_generator.hpp"
#include "config/antenna_processor_config.hpp"
#include "result_types.hpp"

#include <complex>
#include <vector>
#include <cstdint>

// ============================================================================
// PyAntennaProcessorTest — step-by-step antenna processor (ROCm)
// ============================================================================

class PyAntennaProcessorTest {
public:
  PyAntennaProcessorTest(
      ROCmGPUContext& ctx,
      uint32_t n_ant,
      uint32_t n_samples,
      float sample_rate,
      float signal_frequency_hz,
      bool debug_mode = true)
      : ctx_(ctx)
  {
    strategies::AntennaProcessorConfig cfg;
    cfg.n_ant               = n_ant;
    cfg.n_samples           = n_samples;
    cfg.sample_rate         = sample_rate;
    cfg.signal_frequency_hz = signal_frequency_hz;
    cfg.scenario_mode       = strategies::PostFftScenarioMode::ALL_REQUIRED;
    cfg.debug_mode          = debug_mode;

    proc_ = std::make_unique<strategies::AntennaProcessorTest>(ctx.backend(), cfg);
  }

  // Step 0: prepare input from numpy arrays
  void step_0_prepare_input(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> signal,
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> weights)
  {
    auto sig_info = signal.request();
    auto w_info   = weights.request();

    size_t sig_count = static_cast<size_t>(sig_info.size);
    size_t w_count   = static_cast<size_t>(w_info.size);

    // Upload signal to GPU
    if (d_S_) ctx_.backend()->Free(d_S_);
    d_S_ = ctx_.backend()->Allocate(sig_count * sizeof(std::complex<float>));
    hipMemcpy(d_S_, sig_info.ptr, sig_count * sizeof(std::complex<float>),
              hipMemcpyHostToDevice);

    // Upload weight matrix to GPU
    if (d_W_) ctx_.backend()->Free(d_W_);
    d_W_ = ctx_.backend()->Allocate(w_count * sizeof(std::complex<float>));
    hipMemcpy(d_W_, w_info.ptr, w_count * sizeof(std::complex<float>),
              hipMemcpyHostToDevice);

    proc_->step_0_prepare_input(d_S_, d_W_);
  }

  // Step 1: debug input stats
  py::dict step_1_debug_input() {
    strategies::AntennaResult r;
    {
      py::gil_scoped_release release;
      r = proc_->step_1_debug_input();
    }
    return stats_to_dict(r.pre_input_stats);
  }

  // Step 2: GEMM -> returns numpy [n_ant x n_samples] complex
  py::array_t<std::complex<float>> step_2_gemm() {
    std::vector<std::complex<float>> data;
    {
      py::gil_scoped_release release;
      data = proc_->step_2_gemm();
    }
    auto cfg = proc_->config();
    return vector_to_numpy_2d(std::move(data), cfg.n_ant, cfg.n_samples);
  }

  // Step 3: debug post-GEMM stats
  py::dict step_3_debug_post_gemm() {
    strategies::AntennaResult r;
    {
      py::gil_scoped_release release;
      r = proc_->step_3_debug_post_gemm();
    }
    return stats_to_dict(r.post_gemm_stats);
  }

  // Step 4: Window + FFT -> returns numpy [n_ant x nFFT] complex
  py::array_t<std::complex<float>> step_4_window_fft() {
    std::vector<std::complex<float>> data;
    {
      py::gil_scoped_release release;
      data = proc_->step_4_window_fft();
    }
    auto cfg = proc_->config();
    uint32_t nFFT = proc_->test_get_nFFT();
    return vector_to_numpy_2d(std::move(data), cfg.n_ant, nFFT);
  }

  // Step 5: debug post-FFT stats
  py::dict step_5_debug_post_fft() {
    strategies::AntennaResult r;
    {
      py::gil_scoped_release release;
      r = proc_->step_5_debug_post_fft();
    }
    return stats_to_dict(r.post_fft_stats);
  }

  // Step 6.1: OneMax + Parabola
  py::list step_6_1_one_max_parabola() {
    strategies::AntennaResult r;
    {
      py::gil_scoped_release release;
      r = proc_->step_6_1_one_max_parabola();
    }
    py::list out;
    for (const auto& om : r.one_max) {
      py::dict d;
      d["beam_id"]        = om.beam_id;
      d["bin_index"]      = om.bin_index;
      d["magnitude"]      = om.magnitude;
      d["freq_offset"]    = om.freq_offset;
      d["refined_freq_hz"] = om.refined_freq_hz;
      out.append(d);
    }
    return out;
  }

  // Step 6.2: AllMaxima
  py::list step_6_2_all_maxima() {
    strategies::AntennaResult r;
    {
      py::gil_scoped_release release;
      r = proc_->step_6_2_all_maxima();
    }
    py::list out;
    for (const auto& beam : r.all_maxima) {
      py::dict d;
      d["antenna_id"]  = beam.antenna_id;
      d["num_maxima"]  = beam.num_maxima;

      py::list maxima_list;
      for (const auto& mv : beam.maxima) {
        py::dict md;
        md["index"]              = mv.index;
        md["magnitude"]          = mv.magnitude;
        md["phase"]              = mv.phase;
        md["refined_frequency"]  = mv.refined_frequency;
        maxima_list.append(md);
      }
      d["maxima"] = maxima_list;
      out.append(d);
    }
    return out;
  }

  // Step 6.3: GlobalMinMax
  py::list step_6_3_global_minmax() {
    strategies::AntennaResult r;
    {
      py::gil_scoped_release release;
      r = proc_->step_6_3_global_minmax();
    }
    py::list out;
    for (const auto& mm : r.minmax) {
      py::dict d;
      d["beam_id"]          = mm.beam_id;
      d["min_magnitude"]    = mm.min_magnitude;
      d["min_bin"]          = mm.min_bin;
      d["min_frequency_hz"] = mm.min_frequency_hz;
      d["max_magnitude"]    = mm.max_magnitude;
      d["max_bin"]          = mm.max_bin;
      d["max_frequency_hz"] = mm.max_frequency_hz;
      d["dynamic_range_dB"] = mm.dynamic_range_dB;
      out.append(d);
    }
    return out;
  }

  // Upload external weight matrix to GPU (managed by AntennaProcessor_v1)
  // After this call, step_0_signal_only() uses the uploaded W automatically.
  void set_external_weights(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> W)
  {
    auto info = W.request();
    const std::complex<float>* ptr = static_cast<const std::complex<float>*>(info.ptr);
    size_t count = static_cast<size_t>(info.size);

    proc_->set_external_weights(std::vector<std::complex<float>>(ptr, ptr + count));
  }

  // Step 0 signal-only: uploads only signal, uses pre-loaded managed W.
  // Must call set_external_weights() before this.
  void step_0_signal_only(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> signal)
  {
    auto sig_info = signal.request();
    size_t sig_count = static_cast<size_t>(sig_info.size);

    if (d_S_) ctx_.backend()->Free(d_S_);
    d_S_ = ctx_.backend()->Allocate(sig_count * sizeof(std::complex<float>));
    hipMemcpy(d_S_, sig_info.ptr, sig_count * sizeof(std::complex<float>),
              hipMemcpyHostToDevice);

    proc_->step_0_signal_only(d_S_);
    // Note: d_W_ Python member is NOT updated — managed weight lifetime belongs to proc_
  }

  // Full pipeline using managed weights (after set_external_weights)
  py::dict process_full_managed_w() {
    strategies::AntennaResult r;
    {
      py::gil_scoped_release release;
      r = proc_->process_full_managed_w();
    }
    return build_full_result(r);
  }

  // Full pipeline
  py::dict process_full() {
    strategies::AntennaResult r;
    {
      py::gil_scoped_release release;
      r = proc_->process_full();
    }
    return build_full_result(r);
  }

  // Properties
  uint32_t nFFT() const { return proc_->test_get_nFFT(); }
  uint32_t n_ant() const { return proc_->config().n_ant; }
  uint32_t n_samples() const { return proc_->config().n_samples; }
  float sample_rate() const { return proc_->config().sample_rate; }

  ~PyAntennaProcessorTest() {
    if (d_S_) { ctx_.backend()->Free(d_S_); d_S_ = nullptr; }
    if (d_W_) { ctx_.backend()->Free(d_W_); d_W_ = nullptr; }
  }

private:
  ROCmGPUContext& ctx_;
  std::unique_ptr<strategies::AntennaProcessorTest> proc_;
  void* d_S_ = nullptr;
  void* d_W_ = nullptr;

  py::dict build_full_result(const strategies::AntennaResult& r) {
    py::dict result;
    result["total_ms"] = r.perf.total_ms;
    result["scenario_mode"] = static_cast<int>(r.scenario_mode);

    py::list one_max_list;
    for (const auto& om : r.one_max) {
      py::dict d;
      d["beam_id"]         = om.beam_id;
      d["bin_index"]       = om.bin_index;
      d["magnitude"]       = om.magnitude;
      d["refined_freq_hz"] = om.refined_freq_hz;
      one_max_list.append(d);
    }
    result["one_max"] = one_max_list;

    py::list minmax_list;
    for (const auto& mm : r.minmax) {
      py::dict d;
      d["beam_id"]          = mm.beam_id;
      d["min_magnitude"]    = mm.min_magnitude;
      d["max_magnitude"]    = mm.max_magnitude;
      d["dynamic_range_dB"] = mm.dynamic_range_dB;
      minmax_list.append(d);
    }
    result["minmax"] = minmax_list;
    return result;
  }

  py::dict stats_to_dict(const std::vector<statistics::StatisticsResult>& stats) {
    py::list beams;
    for (const auto& s : stats) {
      py::dict d;
      d["beam_id"]        = s.beam_id;
      d["mean_real"]      = s.mean.real();
      d["mean_imag"]      = s.mean.imag();
      d["mean_magnitude"] = s.mean_magnitude;
      d["variance"]       = s.variance;
      d["std_dev"]        = s.std_dev;
      beams.append(d);
    }
    py::dict result;
    result["stats"] = beams;
    result["beam_count"] = static_cast<int>(stats.size());
    return result;
  }
};

// ============================================================================
// PyWeightGenerator — static methods for W matrix generation
// ============================================================================

class PyWeightGenerator {
public:
  static py::array_t<std::complex<float>> generate_delay_and_sum(
      uint32_t n_ant, double f0, double tau_base, double tau_step)
  {
    strategies::WeightParams wp;
    wp.n_ant    = n_ant;
    wp.f0       = f0;
    wp.tau_base = tau_base;
    wp.tau_step = tau_step;

    auto W = strategies::WeightGenerator::generate_delay_and_sum(wp);
    return vector_to_numpy_2d(std::move(W), n_ant, n_ant);
  }
};

// ============================================================================
// register_strategies_rocm — call from PYBIND11_MODULE
// ============================================================================

inline void register_strategies_rocm(py::module_& m) {
  py::class_<PyAntennaProcessorTest>(m, "AntennaProcessorTest",
      "Step-by-step antenna array processor (ROCm).\n\n"
      "Pipeline: GEMM -> Window+FFT -> post-FFT scenarios\n\n"
      "Usage:\n"
      "  ctx = gpuworklib.ROCmGPUContext(0)\n"
      "  proc = gpuworklib.AntennaProcessorTest(ctx, n_ant=5, n_samples=8000,\n"
      "             sample_rate=12e6, signal_frequency_hz=2e6)\n"
      "  proc.step_0_prepare_input(signal, weights)\n"
      "  gemm = proc.step_2_gemm()\n"
      "  spectrum = proc.step_4_window_fft()\n"
      "  peaks = proc.step_6_1_one_max_parabola()")
      .def(py::init<ROCmGPUContext&, uint32_t, uint32_t, float, float, bool>(),
           py::arg("ctx"),
           py::arg("n_ant") = 5,
           py::arg("n_samples") = 8000,
           py::arg("sample_rate") = 12e6f,
           py::arg("signal_frequency_hz") = 2e6f,
           py::arg("debug_mode") = true,
           "Create antenna processor.\n\n"
           "Args:\n"
           "  ctx: ROCmGPUContext\n"
           "  n_ant: antenna count (default 5)\n"
           "  n_samples: samples per antenna (default 8000)\n"
           "  sample_rate: sampling rate Hz (default 12 MHz)\n"
           "  signal_frequency_hz: center freq Hz (default 2 MHz)\n"
           "  debug_mode: enable debug stats (default True)")

      .def("step_0_prepare_input", &PyAntennaProcessorTest::step_0_prepare_input,
           py::arg("signal"), py::arg("weights"),
           "Upload signal [n_ant x n_samples] and weights [n_ant x n_ant] to GPU")

      .def("step_1_debug_input", &PyAntennaProcessorTest::step_1_debug_input,
           "Debug stats on input signal d_S")

      .def("step_2_gemm", &PyAntennaProcessorTest::step_2_gemm,
           "GEMM: X = W * S. Returns numpy [n_ant x n_samples] complex64")

      .def("step_3_debug_post_gemm", &PyAntennaProcessorTest::step_3_debug_post_gemm,
           "Debug stats on post-GEMM d_X")

      .def("step_4_window_fft", &PyAntennaProcessorTest::step_4_window_fft,
           "Hamming window + FFT. Returns numpy [n_ant x nFFT] complex64")

      .def("step_5_debug_post_fft", &PyAntennaProcessorTest::step_5_debug_post_fft,
           "Debug stats on |spectrum|")

      .def("step_6_1_one_max_parabola", &PyAntennaProcessorTest::step_6_1_one_max_parabola,
           "OneMax + parabolic interpolation per beam.\n"
           "Returns list[dict] with bin_index, magnitude, refined_freq_hz")

      .def("step_6_2_all_maxima", &PyAntennaProcessorTest::step_6_2_all_maxima,
           "All local maxima per beam (limit=1000).\n"
           "Returns list[dict] with antenna_id, num_maxima, maxima[]")

      .def("step_6_3_global_minmax", &PyAntennaProcessorTest::step_6_3_global_minmax,
           "Global min/max per beam.\n"
           "Returns list[dict] with min/max magnitude, frequency, dynamic_range_dB")

      .def("process_full", &PyAntennaProcessorTest::process_full,
           "Run full pipeline (all steps + all scenarios).\n"
           "Returns dict with total_ms, one_max, minmax")

      .def("set_external_weights", &PyAntennaProcessorTest::set_external_weights,
           py::arg("W"),
           "Upload external weight matrix to GPU.\n\n"
           "Args:\n"
           "  W: numpy array (n_ant, n_ant) complex64\n\n"
           "After this call, use step_0_signal_only() to update only the signal\n"
           "without re-uploading W on every frame.")

      .def("step_0_signal_only", &PyAntennaProcessorTest::step_0_signal_only,
           py::arg("signal"),
           "Upload signal to GPU and use pre-loaded managed W.\n\n"
           "Must call set_external_weights() first.\n"
           "Args:\n"
           "  signal: numpy array (n_ant, n_samples) complex64")

      .def("process_full_managed_w", &PyAntennaProcessorTest::process_full_managed_w,
           "Run full pipeline using pre-loaded managed weights.\n\n"
           "Requires prior set_external_weights() call.\n"
           "Returns dict with total_ms, one_max, minmax")

      .def_property_readonly("nFFT", &PyAntennaProcessorTest::nFFT)
      .def_property_readonly("n_ant", &PyAntennaProcessorTest::n_ant)
      .def_property_readonly("n_samples", &PyAntennaProcessorTest::n_samples)
      .def_property_readonly("sample_rate", &PyAntennaProcessorTest::sample_rate)

      .def("__repr__", [](const PyAntennaProcessorTest& self) {
          return "<AntennaProcessorTest n_ant=" + std::to_string(self.n_ant()) +
                 " n_samples=" + std::to_string(self.n_samples()) +
                 " nFFT=" + std::to_string(self.nFFT()) + ">";
      });

  // WeightGenerator static methods
  m.def("generate_delay_and_sum_weights",
        &PyWeightGenerator::generate_delay_and_sum,
        py::arg("n_ant"), py::arg("f0"),
        py::arg("tau_base") = 0.0, py::arg("tau_step") = 100e-6,
        "Generate delay-and-sum weight matrix [n_ant x n_ant] complex64.\n\n"
        "W[beam][ant] = exp(-j*2*pi*f0*tau_ant) / sqrt(N_ant)\n\n"
        "Args:\n"
        "  n_ant: number of antennas\n"
        "  f0: center frequency (Hz)\n"
        "  tau_base: base delay (s, default 0)\n"
        "  tau_step: delay step per antenna (s, default 100us)\n\n"
        "Returns:\n"
        "  numpy.ndarray complex64 (n_ant, n_ant)");
}
