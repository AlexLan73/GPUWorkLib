#pragma once

/**
 * @file py_lfm_analytical_delay_rocm.hpp
 * @brief Python wrapper for LfmGeneratorAnalyticalDelayROCm
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * s(t) = A * exp(j * [pi*mu*(t-tau)^2 + 2*pi*f_start*(t-tau)])
 * where tau = delay_us[antenna] * 1e-6
 * output = 0 when t < tau
 *
 * Usage:
 *   gen = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=1e6, f_end=2e6)
 *   gen.set_sampling(fs=12e6, length=4096)
 *   gen.set_delays([0.0, 0.5, 1.0])
 *   gpu_data = gen.generate_gpu()   # numpy complex64 (3, 4096)
 *   cpu_data = gen.generate_cpu()   # numpy complex64 (3, 4096)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-23
 */

#include "generators/lfm_generator_analytical_delay_rocm.hpp"

// ============================================================================
// PyLfmAnalyticalDelayROCm — pythonic wrapper
// ============================================================================

class PyLfmAnalyticalDelayROCm {
public:
  PyLfmAnalyticalDelayROCm(ROCmGPUContext& ctx,
                            double f_start = 100.0,
                            double f_end = 500.0,
                            double amplitude = 1.0)
      : ctx_(ctx)
  {
    signal_gen::LfmParams params;
    params.f_start = f_start;
    params.f_end = f_end;
    params.amplitude = amplitude;
    gen_ = std::make_unique<signal_gen::LfmGeneratorAnalyticalDelayROCm>(
        ctx.backend(), params);
  }

  // -- Parameters -----------------------------------------------------------

  void set_sampling(double fs, size_t length) {
    signal_gen::SystemSampling sys;
    sys.fs = fs;
    sys.length = length;
    gen_->SetSampling(sys);
  }

  void set_delays(const std::vector<float>& delay_us) {
    gen_->SetDelays(delay_us);
  }

  void set_params(double f_start, double f_end, double amplitude = 1.0) {
    signal_gen::LfmParams p;
    p.f_start = f_start;
    p.f_end = f_end;
    p.amplitude = amplitude;
    gen_->SetParams(p);
  }

  // -- Generation -----------------------------------------------------------

  py::array_t<std::complex<float>> generate_gpu() {
    drv_gpu_lib::InputData<void*> result;
    {
      py::gil_scoped_release release;
      result = gen_->GenerateToGpu();
    }

    uint32_t n_ant = gen_->GetAntennas();
    uint32_t n_point = static_cast<uint32_t>(gen_->GetSampling().length);
    size_t total = static_cast<size_t>(n_ant) * n_point;

    std::vector<std::complex<float>> data(total);
    ctx_.backend()->MemcpyDeviceToHost(data.data(), result.data,
                                        total * sizeof(std::complex<float>));
    ctx_.backend()->Free(result.data);

    if (n_ant <= 1)
      return vector_to_numpy(std::move(data));
    return vector_to_numpy_2d(std::move(data), n_ant, n_point);
  }

  py::array_t<std::complex<float>> generate_cpu() {
    std::vector<std::vector<std::complex<float>>> data;
    {
      py::gil_scoped_release release;
      data = gen_->GenerateToCpu();
    }

    uint32_t n_ant = gen_->GetAntennas();
    uint32_t n_point = static_cast<uint32_t>(gen_->GetSampling().length);

    std::vector<std::complex<float>> flat;
    flat.reserve(static_cast<size_t>(n_ant) * n_point);
    for (auto& row : data)
      flat.insert(flat.end(), row.begin(), row.end());

    if (n_ant <= 1)
      return vector_to_numpy(std::move(flat));
    return vector_to_numpy_2d(std::move(flat), n_ant, n_point);
  }

  // -- Getters --------------------------------------------------------------

  uint32_t get_antennas() const { return gen_->GetAntennas(); }
  size_t   get_length()   const { return gen_->GetSampling().length; }
  double   get_fs()       const { return gen_->GetSampling().fs; }

  py::dict get_params_dict() const {
    const auto& p = gen_->GetParams();
    py::dict d;
    d["f_start"]   = p.f_start;
    d["f_end"]     = p.f_end;
    d["amplitude"] = p.amplitude;
    return d;
  }

  py::list get_delays() const {
    py::list result;
    for (float d : gen_->GetDelays()) result.append(d);
    return result;
  }

private:
  ROCmGPUContext& ctx_;
  std::unique_ptr<signal_gen::LfmGeneratorAnalyticalDelayROCm> gen_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_lfm_analytical_delay_rocm(py::module& m) {
  py::class_<PyLfmAnalyticalDelayROCm>(m, "LfmAnalyticalDelayROCm",
      "LFM generator with analytical per-antenna delay (ROCm/HIP).\n\n"
      "Signal formula:\n"
      "  s(t) = A * exp(j * [pi*mu*(t-tau)^2 + 2*pi*f_start*(t-tau)])\n"
      "  output = 0 when t < tau\n\n"
      "Usage:\n"
      "  gen = gpuworklib.LfmAnalyticalDelayROCm(ctx, f_start=1e6, f_end=2e6)\n"
      "  gen.set_sampling(fs=12e6, length=4096)\n"
      "  gen.set_delays([0.0, 0.5, 1.0])\n"
      "  gpu = gen.generate_gpu()  # numpy complex64\n")
      .def(py::init<ROCmGPUContext&, double, double, double>(),
           py::arg("ctx"),
           py::arg("f_start") = 100.0,
           py::arg("f_end") = 500.0,
           py::arg("amplitude") = 1.0,
           "Create LfmAnalyticalDelayROCm bound to ROCm GPU context.")

      .def("set_sampling", &PyLfmAnalyticalDelayROCm::set_sampling,
           py::arg("fs"), py::arg("length"),
           "Set sampling parameters (Hz and number of points).")

      .def("set_delays", &PyLfmAnalyticalDelayROCm::set_delays,
           py::arg("delay_us"),
           "Set per-antenna delays in microseconds.")

      .def("set_params", &PyLfmAnalyticalDelayROCm::set_params,
           py::arg("f_start"), py::arg("f_end"),
           py::arg("amplitude") = 1.0,
           "Set LFM parameters.")

      .def("generate_gpu", &PyLfmAnalyticalDelayROCm::generate_gpu,
           "Generate on GPU and return to CPU.\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64:\n"
           "    shape (length,) if 1 antenna\n"
           "    shape (antennas, length) if multiple\n\n"
           "Note: First call compiles HIP kernel (hiprtc).")

      .def("generate_cpu", &PyLfmAnalyticalDelayROCm::generate_cpu,
           "Generate CPU reference (double precision internally).\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64: same shape as generate_gpu()")

      .def("get_params", &PyLfmAnalyticalDelayROCm::get_params_dict,
           "Get LFM parameters as dict.")

      .def_property_readonly("antennas", &PyLfmAnalyticalDelayROCm::get_antennas,
           "Number of channels/antennas.")
      .def_property_readonly("length", &PyLfmAnalyticalDelayROCm::get_length,
           "Samples per channel.")
      .def_property_readonly("fs", &PyLfmAnalyticalDelayROCm::get_fs,
           "Sample rate Hz.")
      .def_property_readonly("delays", &PyLfmAnalyticalDelayROCm::get_delays,
           "Per-antenna delays in microseconds.")

      .def("__repr__", [](const PyLfmAnalyticalDelayROCm& self) {
          return "<LfmAnalyticalDelayROCm antennas=" +
                 std::to_string(self.get_antennas()) +
                 " length=" + std::to_string(self.get_length()) + ">";
      });
}
