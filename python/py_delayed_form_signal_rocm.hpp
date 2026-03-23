#pragma once

/**
 * @file py_delayed_form_signal_rocm.hpp
 * @brief Python wrapper for DelayedFormSignalGeneratorROCm (Farrow 48x5 delay)
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Pipeline:
 *   1. FormSignalGeneratorROCm -> clean signal on GPU
 *   2. LchFarrowROCm -> fractional delay (Lagrange 48x5) per-antenna
 *   3. Optional noise via LchFarrowROCm::SetNoise()
 *
 * Usage:
 *   gen = gpuworklib.DelayedFormSignalGeneratorROCm(ctx)
 *   gen.set_params(fs=1e6, antennas=8, points=4096, f0=50e3)
 *   gen.set_delays([0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5])
 *   signal = gen.generate()   # numpy complex64 (8, 4096)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-23
 */

#include "generators/delayed_form_signal_generator_rocm.hpp"

// ============================================================================
// PyDelayedFormSignalGeneratorROCm — pythonic wrapper
// ============================================================================

class PyDelayedFormSignalGeneratorROCm {
public:
  explicit PyDelayedFormSignalGeneratorROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), gen_(ctx.backend()) {}

  // -- Parameters -----------------------------------------------------------

  void set_params(
      double   fs              = 1e6,
      uint32_t antennas        = 1,
      uint32_t points          = 4096,
      double   f0              = 0.0,
      double   amplitude       = 1.0,
      double   noise_amplitude = 0.0,
      double   phase           = 0.0,
      double   fdev            = 0.0,
      double   norm            = 0.7071067811865476,
      uint32_t noise_seed      = 0)
  {
    signal_gen::FormParams p;
    p.fs              = fs;
    p.antennas        = antennas;
    p.points          = points;
    p.f0              = f0;
    p.amplitude       = amplitude;
    p.noise_amplitude = noise_amplitude;
    p.phase           = phase;
    p.fdev            = fdev;
    p.norm            = norm;
    p.noise_seed      = noise_seed;
    gen_.SetParams(p);
  }

  void set_delays(const std::vector<float>& delay_us) {
    gen_.SetDelays(delay_us);
  }

  void load_matrix(const std::string& json_path) {
    gen_.LoadMatrix(json_path);
  }

  // -- Generation -----------------------------------------------------------

  py::array_t<std::complex<float>> generate() {
    std::vector<std::vector<std::complex<float>>> data;
    {
      py::gil_scoped_release release;
      data = gen_.GenerateToCpu();
    }

    uint32_t n_ant    = gen_.GetAntennas();
    uint32_t n_points = gen_.GetPoints();

    // Flatten [antenna][sample] -> contiguous 1D
    std::vector<std::complex<float>> flat;
    flat.reserve(static_cast<size_t>(n_ant) * n_points);
    for (auto& row : data)
      flat.insert(flat.end(), row.begin(), row.end());

    if (n_ant <= 1)
      return vector_to_numpy(std::move(flat));
    return vector_to_numpy_2d(std::move(flat), n_ant, n_points);
  }

  // -- Getters --------------------------------------------------------------

  uint32_t get_antennas() const { return gen_.GetAntennas(); }
  uint32_t get_points()   const { return gen_.GetPoints(); }
  double   get_fs()       const { return gen_.GetParams().fs; }

  py::dict get_params_dict() const {
    const auto& p = gen_.GetParams();
    py::dict d;
    d["fs"]              = p.fs;
    d["antennas"]        = p.antennas;
    d["points"]          = p.points;
    d["f0"]              = p.f0;
    d["amplitude"]       = p.amplitude;
    d["noise_amplitude"] = p.noise_amplitude;
    d["phase"]           = p.phase;
    d["fdev"]            = p.fdev;
    d["norm"]            = p.norm;
    d["noise_seed"]      = p.noise_seed;
    return d;
  }

  py::list get_delays() const {
    py::list result;
    for (float d : gen_.GetDelays()) result.append(d);
    return result;
  }

private:
  ROCmGPUContext& ctx_;
  signal_gen::DelayedFormSignalGeneratorROCm gen_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_delayed_form_signal_rocm(py::module& m) {
  py::class_<PyDelayedFormSignalGeneratorROCm>(m, "DelayedFormSignalGeneratorROCm",
      "Delayed signal generator with Farrow 48x5 fractional delay (ROCm/HIP).\n\n"
      "Pipeline:\n"
      "  1. FormSignalGeneratorROCm -> clean getX signal on GPU\n"
      "  2. LchFarrowROCm -> per-antenna fractional delay (Lagrange 48x5)\n"
      "  3. Optional noise added after delay\n\n"
      "Usage:\n"
      "  gen = gpuworklib.DelayedFormSignalGeneratorROCm(ctx)\n"
      "  gen.set_params(fs=1e6, antennas=8, points=4096, f0=50e3)\n"
      "  gen.set_delays([0.0, 1.5, 3.0, ...])\n"
      "  signal = gen.generate()  # numpy complex64 (8, 4096)\n")
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create DelayedFormSignalGeneratorROCm bound to ROCm GPU context.")

      .def("set_params", &PyDelayedFormSignalGeneratorROCm::set_params,
           py::arg("fs")              = 1e6,
           py::arg("antennas")        = 1,
           py::arg("points")          = 4096,
           py::arg("f0")              = 0.0,
           py::arg("amplitude")       = 1.0,
           py::arg("noise_amplitude") = 0.0,
           py::arg("phase")           = 0.0,
           py::arg("fdev")            = 0.0,
           py::arg("norm")            = 0.7071067811865476,
           py::arg("noise_seed")      = 0,
           "Set signal parameters.\n\n"
           "Args:\n"
           "  fs:              sample rate Hz (default 1e6)\n"
           "  antennas:        number of channels (default 1)\n"
           "  points:          samples per channel (default 4096)\n"
           "  f0:              center frequency Hz (default 0)\n"
           "  amplitude:       signal amplitude (default 1.0)\n"
           "  noise_amplitude: noise level (0 = no noise)\n"
           "  phase:           initial phase rad (default 0.0)\n"
           "  fdev:            frequency deviation Hz for LFM (0 = CW)\n"
           "  norm:            normalization factor (default 1/sqrt(2))\n"
           "  noise_seed:      PRNG seed (0 = random)")

      .def("set_delays", &PyDelayedFormSignalGeneratorROCm::set_delays,
           py::arg("delay_us"),
           "Set per-antenna delays in microseconds.\n\n"
           "Args:\n"
           "  delay_us: list of float delays, one per antenna")

      .def("load_matrix", &PyDelayedFormSignalGeneratorROCm::load_matrix,
           py::arg("json_path"),
           "Load Lagrange interpolation matrix from JSON.\n\n"
           "Default: built-in 48x5 matrix. Use this to override.")

      .def("generate", &PyDelayedFormSignalGeneratorROCm::generate,
           "Generate delayed signal on GPU and return to CPU.\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64:\n"
           "    - shape (points,) if antennas == 1\n"
           "    - shape (antennas, points) if antennas > 1\n\n"
           "Note: First call compiles HIP kernel (hiprtc).")

      .def("get_params", &PyDelayedFormSignalGeneratorROCm::get_params_dict,
           "Get current parameters as dict.")

      .def_property_readonly("antennas", &PyDelayedFormSignalGeneratorROCm::get_antennas,
           "Number of channels.")
      .def_property_readonly("points", &PyDelayedFormSignalGeneratorROCm::get_points,
           "Samples per channel.")
      .def_property_readonly("fs", &PyDelayedFormSignalGeneratorROCm::get_fs,
           "Sample rate Hz.")
      .def_property_readonly("delays", &PyDelayedFormSignalGeneratorROCm::get_delays,
           "Per-antenna delays in microseconds.")

      .def("__repr__", [](const PyDelayedFormSignalGeneratorROCm& self) {
          return "<DelayedFormSignalGeneratorROCm antennas=" +
                 std::to_string(self.get_antennas()) +
                 " points=" + std::to_string(self.get_points()) + ">";
      });
}
