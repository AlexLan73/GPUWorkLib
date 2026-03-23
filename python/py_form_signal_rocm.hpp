#pragma once

/**
 * @file py_form_signal_rocm.hpp
 * @brief Python wrapper for FormSignalGeneratorROCm (multi-channel getX, ROCm)
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Usage:
 *   gen = gpuworklib.FormSignalGeneratorROCm(ctx)
 *   gen.set_params(antennas=5, points=8000, fs=12e6, f0=2e6)
 *   signal = gen.generate()              # numpy complex64 (5, 8000)
 *   gen.set_params_from_string("f0=1e6,a=1.0,an=0.1,antennas=3,points=4096,fs=12e6")
 *   signal = gen.generate()              # numpy complex64 (3, 4096)
 *
 * Signal formula (getX):
 *   X = a * norm * exp(j*(2π*f0*t + π*fdev/ti*((t-ti/2)²) + phi))
 *       + an * norm * (randn + j*randn)
 *   X = 0  when t < 0 or t > ti - dt
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-10
 */

#include "generators/form_signal_generator_rocm.hpp"

// ============================================================================
// PyFormSignalGeneratorROCm — pythonic wrapper
// ============================================================================

// ROCm-версия мультиканального генератора сигналов FormSignalGenerator.
// Поддерживает: CW (fdev=0), LFM-чирп (fdev≠0), шум (an>0), задержки (tau_step).
// Два способа задания параметров: set_params() по именованным аргументам
// и set_params_from_string() через CSV-строку "f0=1e6,a=1.0,...".
// generate() возвращает numpy complex64: (antennas, points) — shape 2D всегда.
class PyFormSignalGeneratorROCm {
public:
  explicit PyFormSignalGeneratorROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), gen_(ctx.backend()) {}

  // ── Параметры ─────────────────────────────────────────────────────────────

  void set_params(
      uint32_t antennas     = 1,
      uint32_t points       = 4096,
      double   fs           = 12e6,
      double   f0           = 0.0,
      double   amplitude    = 1.0,
      double   phase        = 0.0,
      double   fdev         = 0.0,
      double   norm         = 0.7071067811865476,
      double   noise_amplitude = 0.0,
      uint32_t noise_seed   = 0,
      double   tau_base     = 0.0,
      double   tau_step     = 0.0,
      double   tau_min      = 0.0,
      double   tau_max      = 0.0,
      uint32_t tau_seed     = 12345)
  {
    signal_gen::FormParams p;
    p.antennas        = antennas;
    p.points          = points;
    p.fs              = fs;
    p.f0              = f0;
    p.amplitude       = amplitude;
    p.phase           = phase;
    p.fdev            = fdev;
    p.norm            = norm;
    p.noise_amplitude = noise_amplitude;
    p.noise_seed      = noise_seed;
    p.tau_base        = tau_base;
    p.tau_step        = tau_step;
    p.tau_min         = tau_min;
    p.tau_max         = tau_max;
    p.tau_seed        = tau_seed;
    gen_.SetParams(p);
  }

  void set_params_from_string(const std::string& params_str) {
    gen_.SetParamsFromString(params_str);
  }

  // ── Генерация ─────────────────────────────────────────────────────────────

  py::array_t<std::complex<float>> generate() {
    std::vector<std::vector<std::complex<float>>> data;
    {
      py::gil_scoped_release release;
      data = gen_.GenerateToCpu();
    }

    uint32_t n_ant    = gen_.GetAntennas();
    uint32_t n_points = gen_.GetPoints();

    // Сплющить [antenna][sample] → flat 1D, затем вернуть 2D
    std::vector<std::complex<float>> flat;
    flat.reserve(static_cast<size_t>(n_ant) * n_points);
    for (auto& row : data)
      flat.insert(flat.end(), row.begin(), row.end());

    if (n_ant <= 1)
      return vector_to_numpy(std::move(flat));
    return vector_to_numpy_2d(std::move(flat), n_ant, n_points);
  }

  // ── Геттеры ───────────────────────────────────────────────────────────────

  uint32_t get_antennas() const { return gen_.GetAntennas(); }
  uint32_t get_points()   const { return gen_.GetPoints(); }

  py::dict get_params_dict() const {
    const auto& p = gen_.GetParams();
    py::dict d;
    d["antennas"]          = p.antennas;
    d["points"]            = p.points;
    d["fs"]                = p.fs;
    d["f0"]                = p.f0;
    d["amplitude"]         = p.amplitude;
    d["phase"]             = p.phase;
    d["fdev"]              = p.fdev;
    d["norm"]              = p.norm;
    d["noise_amplitude"]   = p.noise_amplitude;
    d["noise_seed"]        = p.noise_seed;
    d["tau_base"]          = p.tau_base;
    d["tau_step"]          = p.tau_step;
    d["tau_min"]           = p.tau_min;
    d["tau_max"]           = p.tau_max;
    d["tau_seed"]          = p.tau_seed;
    return d;
  }

private:
  ROCmGPUContext& ctx_;
  signal_gen::FormSignalGeneratorROCm gen_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_form_signal_rocm(py::module& m) {
  py::class_<PyFormSignalGeneratorROCm>(m, "FormSignalGeneratorROCm",
      "Multi-channel getX signal generator on ROCm/HIP GPU.\n\n"
      "Signal formula:\n"
      "  X = a * norm * exp(j*(2π*f0*t + π*fdev/ti*((t-ti/2)²) + phi))\n"
      "      + an * norm * (randn + j*randn)\n"
      "  X = 0  when t < 0 or t > ti - dt\n\n"
      "Delay modes:\n"
      "  FIXED:  tau_step=0  — same delay for all antennas (tau_base)\n"
      "  LINEAR: tau_step≠0  — tau = tau_base + antenna_id * tau_step\n"
      "  RANDOM: tau_min≠tau_max — uniform random in [tau_min, tau_max]\n\n"
      "Usage:\n"
      "  gen = gpuworklib.FormSignalGeneratorROCm(ctx)\n"
      "  gen.set_params(antennas=5, points=8000, fs=12e6, f0=2e6)\n"
      "  signal = gen.generate()  # numpy complex64 (5, 8000)\n")
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create FormSignalGeneratorROCm bound to ROCm GPU context.")

      .def("set_params", &PyFormSignalGeneratorROCm::set_params,
           py::arg("antennas")          = 1,
           py::arg("points")            = 4096,
           py::arg("fs")               = 12e6,
           py::arg("f0")               = 0.0,
           py::arg("amplitude")        = 1.0,
           py::arg("phase")            = 0.0,
           py::arg("fdev")             = 0.0,
           py::arg("norm")             = 0.7071067811865476,
           py::arg("noise_amplitude")  = 0.0,
           py::arg("noise_seed")       = 0,
           py::arg("tau_base")         = 0.0,
           py::arg("tau_step")         = 0.0,
           py::arg("tau_min")          = 0.0,
           py::arg("tau_max")          = 0.0,
           py::arg("tau_seed")         = 12345,
           "Set generator parameters.\n\n"
           "Args:\n"
           "  antennas: number of channels (default 1)\n"
           "  points:   samples per channel (default 4096)\n"
           "  fs:       sample rate Hz (default 12e6)\n"
           "  f0:       center frequency Hz (default 0)\n"
           "  amplitude: signal amplitude a (default 1.0)\n"
           "  phase:    initial phase phi rad (default 0.0)\n"
           "  fdev:     frequency deviation Hz for LFM chirp (0 = CW)\n"
           "  norm:     normalization factor (default 1/sqrt(2))\n"
           "  noise_amplitude: noise level an (0 = no noise)\n"
           "  noise_seed: PRNG seed for noise (0 = random)\n"
           "  tau_base: base delay seconds (FIXED mode)\n"
           "  tau_step: delay step per antenna seconds (LINEAR mode)\n"
           "  tau_min/tau_max: delay range for RANDOM mode\n"
           "  tau_seed: PRNG seed for random delays")

      .def("set_params_from_string", &PyFormSignalGeneratorROCm::set_params_from_string,
           py::arg("params_str"),
           "Set parameters from CSV string.\n\n"
           "Format: 'key=value,key=value,...'\n"
           "Keys: fs, f0, a, an, phi, fdev, norm, tau, tau_step, tau_min,\n"
           "      tau_max, tau_seed, noise_seed, antennas, points\n\n"
           "Example: 'f0=1e6,a=1.0,an=0.1,antennas=5,points=8000,fs=12e6'")

      .def("generate", &PyFormSignalGeneratorROCm::generate,
           "Generate signal on GPU and return to CPU.\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64 shape (antennas, points)\n\n"
           "Note: First call compiles HIP kernel (hiprtc), subsequent calls reuse it.")

      .def("get_params", &PyFormSignalGeneratorROCm::get_params_dict,
           "Get current parameters as dict.")

      .def_property_readonly("antennas", &PyFormSignalGeneratorROCm::get_antennas,
           "Number of channels.")
      .def_property_readonly("points",   &PyFormSignalGeneratorROCm::get_points,
           "Samples per channel.")

      .def("__repr__", [](const PyFormSignalGeneratorROCm& self) {
          return "<FormSignalGeneratorROCm antennas=" +
                 std::to_string(self.get_antennas()) +
                 " points=" + std::to_string(self.get_points()) + ">";
      });
}
