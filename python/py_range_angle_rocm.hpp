#pragma once

/**
 * @file py_range_angle_rocm.hpp
 * @brief Python bindings for RangeAngleProcessor (ROCm)
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Exported symbols:
 *   RangeAnglePeakMode  — enum PeakSearchMode (TOP_1 / TOP_N)
 *   RangeAngleParams    — processing parameters struct
 *   TargetInfo          — single detected target
 *   RangeAngleResult    — full processing result (targets + optional power cube)
 *   RangeAngleProcessor — facade: set_params / process pipeline
 *
 * Usage from Python:
 *   p = gpuworklib.RangeAngleParams()
 *   p.n_ant_az = 8; p.n_ant_el = 8; p.n_samples = 50000
 *   p.f_start = -5e6; p.f_end = 5e6; p.sample_rate = 12e6
 *   proc = gpuworklib.RangeAngleProcessor(ctx)   # ctx: ROCmGPUContext
 *   proc.set_params(p)
 *   result = proc.process(iq_data_complex64, download_result=True)
 *   for t in result.targets:
 *       print(t.range_m, t.angle_az_deg, t.angle_el_deg)
 *   cube = result.power_cube_numpy()  # shape (n_range_bins, n_ant_az, n_ant_el)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-17
 */

#include "../modules/range_angle/include/range_angle_processor.hpp"
#include "../modules/range_angle/include/range_angle_params.hpp"
#include "../modules/range_angle/include/range_angle_types.hpp"

// ============================================================================
// PyRangeAngleProcessor — thin wrapper around RangeAngleProcessor
//   Accepts ROCmGPUContext& (consistent with other ROCm Python wrappers)
// ============================================================================

class PyRangeAngleProcessor {
public:
  explicit PyRangeAngleProcessor(ROCmGPUContext& ctx)
      : proc_(ctx.backend()) {}

  void set_params(const range_angle::RangeAngleParams& params) {
    proc_.SetParams(params);
  }

  range_angle::RangeAngleParams get_params() const {
    return proc_.GetParams();
  }

  range_angle::RangeAngleResult process(
      py::array_t<std::complex<float>,
                  py::array::c_style | py::array::forcecast> data,
      bool download_result = true)
  {
    auto buf = data.request();
    size_t total = 1;
    for (py::ssize_t d = 0; d < buf.ndim; ++d)
      total *= static_cast<size_t>(buf.shape[d]);

    const auto* ptr = static_cast<const std::complex<float>*>(buf.ptr);
    std::vector<std::complex<float>> vec(ptr, ptr + total);

    range_angle::RangeAngleResult result;
    {
      py::gil_scoped_release release;
      result = proc_.Process(vec, download_result);
    }
    return result;
  }

private:
  range_angle::RangeAngleProcessor proc_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_range_angle(py::module& m) {

  // ── PeakSearchMode enum ───────────────────────────────────────────────────

  py::enum_<range_angle::PeakSearchMode>(m, "RangeAnglePeakMode",
      "Peak search mode for RangeAngleProcessor.\n\n"
      "  TOP_1 — find single strongest target (fastest)\n"
      "  TOP_N — find N strongest targets (n_peaks parameter)")
    .value("TOP_1", range_angle::PeakSearchMode::kTop1,  "Find single strongest target")
    .value("TOP_N", range_angle::PeakSearchMode::kTopN,  "Find N strongest targets")
    .export_values();

  // ── RangeAngleParams ──────────────────────────────────────────────────────

  py::class_<range_angle::RangeAngleParams>(m, "RangeAngleParams",
      "LFM range-angle processing parameters.\n\n"
      "Antenna array:\n"
      "  n_ant_az          — number of azimuth antennas (default 16)\n"
      "  n_ant_el          — number of elevation antennas (default 16)\n\n"
      "Signal:\n"
      "  n_samples         — samples per antenna per CPI (default 1 300 000)\n"
      "  f_start           — LFM start frequency, Hz (default -5e6)\n"
      "  f_end             — LFM end frequency, Hz  (default +5e6)\n"
      "  sample_rate       — ADC sample rate, Hz    (default 12e6)\n\n"
      "Range FFT:\n"
      "  nfft_range        — FFT size; 0 = auto (next power-of-2 >= n_samples)\n\n"
      "Spatial:\n"
      "  antenna_spacing   — inter-element spacing, metres (default 0.345)\n"
      "  carrier_freq      — RF carrier frequency, Hz     (default 435e6)\n\n"
      "Peak search:\n"
      "  peak_mode         — RangeAnglePeakMode enum\n"
      "  n_peaks           — number of peaks to find (TOP_N mode)\n\n"
      "Computed (filled by set_params on processor):\n"
      "  n_range_bins      — effective range bins after FFT\n"
      "  range_res_m       — range resolution, metres\n\n"
      "Helpers:\n"
      "  get_n_antennas()  → n_ant_az * n_ant_el\n"
      "  get_bandwidth()   → f_end - f_start\n"
      "  get_duration()    → n_samples / sample_rate\n"
      "  get_chirp_rate()  → bandwidth / duration")
    .def(py::init<>(), "Create RangeAngleParams with default values")

    // Antenna array
    .def_readwrite("n_ant_az",        &range_angle::RangeAngleParams::n_ant_az,
                   "Number of azimuth antennas")
    .def_readwrite("n_ant_el",        &range_angle::RangeAngleParams::n_ant_el,
                   "Number of elevation antennas")

    // Signal parameters
    .def_readwrite("n_samples",       &range_angle::RangeAngleParams::n_samples,
                   "Samples per antenna per CPI")
    .def_readwrite("f_start",         &range_angle::RangeAngleParams::f_start,
                   "LFM start frequency, Hz")
    .def_readwrite("f_end",           &range_angle::RangeAngleParams::f_end,
                   "LFM end frequency, Hz")
    .def_readwrite("sample_rate",     &range_angle::RangeAngleParams::sample_rate,
                   "ADC sample rate, Hz")

    // Range FFT
    .def_readwrite("nfft_range",      &range_angle::RangeAngleParams::nfft_range,
                   "FFT size for range axis (0 = auto)")

    // Spatial
    .def_readwrite("carrier_freq",    &range_angle::RangeAngleParams::carrier_freq,
                   "RF carrier frequency, Hz")
    .def_readwrite("antenna_spacing", &range_angle::RangeAngleParams::antenna_spacing,
                   "Inter-element spacing, metres")

    // Peak search
    .def_readwrite("peak_mode",       &range_angle::RangeAngleParams::peak_mode,
                   "Peak search mode (RangeAnglePeakMode)")
    .def_readwrite("n_peaks",         &range_angle::RangeAngleParams::n_peaks,
                   "Number of peaks to find (TOP_N mode)")

    // Computed fields
    .def_readwrite("n_range_bins",    &range_angle::RangeAngleParams::n_range_bins,
                   "Effective range bins after FFT (filled by set_params)")
    .def_readwrite("range_res_m",     &range_angle::RangeAngleParams::range_res_m,
                   "Range resolution, metres (filled by set_params)")

    // Helper methods
    .def("get_n_antennas",  &range_angle::RangeAngleParams::get_n_antennas,
         "Returns n_ant_az * n_ant_el")
    .def("get_bandwidth",   &range_angle::RangeAngleParams::get_bandwidth,
         "Returns f_end - f_start, Hz")
    .def("get_duration",    &range_angle::RangeAngleParams::get_duration,
         "Returns n_samples / sample_rate, seconds")
    .def("get_chirp_rate",  &range_angle::RangeAngleParams::get_chirp_rate,
         "Returns bandwidth / duration, Hz/s")

    .def("__repr__", [](const range_angle::RangeAngleParams& p) {
        return "<RangeAngleParams"
               " n_ant=" + std::to_string(p.get_n_antennas()) +
               " n_samples=" + std::to_string(p.n_samples) +
               " B=" + std::to_string(static_cast<int>(p.get_bandwidth())) + "Hz"
               " fs=" + std::to_string(static_cast<int>(p.sample_rate)) + "Hz"
               ">";
    });

  // ── TargetInfo ────────────────────────────────────────────────────────────

  py::class_<range_angle::TargetInfo>(m, "TargetInfo",
      "Single detected target from RangeAngle processing.\n\n"
      "Fields:\n"
      "  range_m      — range in metres\n"
      "  angle_az_deg — azimuth angle, degrees\n"
      "  angle_el_deg — elevation angle, degrees\n"
      "  range_bin    — fractional range bin index\n"
      "  az_bin       — fractional azimuth bin index\n"
      "  el_bin       — fractional elevation bin index\n"
      "  power_db     — peak power, dB\n"
      "  snr_db       — estimated SNR, dB")
    .def(py::init<>())
    .def_readwrite("range_m",      &range_angle::TargetInfo::range_m,
                   "Range in metres")
    .def_readwrite("angle_az_deg", &range_angle::TargetInfo::angle_az_deg,
                   "Azimuth angle, degrees")
    .def_readwrite("angle_el_deg", &range_angle::TargetInfo::angle_el_deg,
                   "Elevation angle, degrees")
    .def_readwrite("range_bin",    &range_angle::TargetInfo::range_bin,
                   "Fractional range bin index")
    .def_readwrite("az_bin",       &range_angle::TargetInfo::az_bin,
                   "Fractional azimuth bin index")
    .def_readwrite("el_bin",       &range_angle::TargetInfo::el_bin,
                   "Fractional elevation bin index")
    .def_readwrite("power_db",     &range_angle::TargetInfo::power_db,
                   "Peak power, dB")
    .def_readwrite("snr_db",       &range_angle::TargetInfo::snr_db,
                   "Estimated SNR, dB")
    .def("__repr__", [](const range_angle::TargetInfo& t) {
        return "<TargetInfo"
               " R="    + std::to_string(t.range_m)      + "m"
               " az="   + std::to_string(t.angle_az_deg) + "deg"
               " el="   + std::to_string(t.angle_el_deg) + "deg"
               " pwr="  + std::to_string(t.power_db)     + "dB"
               " snr="  + std::to_string(t.snr_db)       + "dB"
               ">";
    });

  // ── RangeAngleResult ──────────────────────────────────────────────────────

  py::class_<range_angle::RangeAngleResult>(m, "RangeAngleResult",
      "Result of RangeAngleProcessor.process().\n\n"
      "Fields:\n"
      "  success       — True if processing succeeded\n"
      "  n_range_bins  — number of range bins\n"
      "  n_ant_az      — azimuth antenna count\n"
      "  n_ant_el      — elevation antenna count\n"
      "  targets       — list of TargetInfo objects\n"
      "  error_message — error description (if success=False)\n\n"
      "Methods:\n"
      "  power_cube_numpy() — returns 3-D float32 array\n"
      "                       shape: (n_range_bins, n_ant_az, n_ant_el)\n"
      "                       empty if download_result=False was passed to process()")
    .def_readonly("success",       &range_angle::RangeAngleResult::success,
                  "True if processing succeeded")
    .def_readonly("n_range_bins",  &range_angle::RangeAngleResult::n_range_bins,
                  "Number of range bins")
    .def_readonly("n_ant_az",      &range_angle::RangeAngleResult::n_ant_az,
                  "Azimuth antenna count")
    .def_readonly("n_ant_el",      &range_angle::RangeAngleResult::n_ant_el,
                  "Elevation antenna count")
    .def_readonly("targets",       &range_angle::RangeAngleResult::targets,
                  "List of TargetInfo detected targets")
    .def_readonly("error_message", &range_angle::RangeAngleResult::error_message,
                  "Error description (non-empty only when success=False)")

    .def("power_cube_numpy",
      [](const range_angle::RangeAngleResult& r) -> py::array_t<float> {
        if (r.power_cube.empty()) {
          return py::array_t<float>(std::vector<ssize_t>{0});
        }
        std::vector<ssize_t> shape = {
            static_cast<ssize_t>(r.n_range_bins),
            static_cast<ssize_t>(r.n_ant_az),
            static_cast<ssize_t>(r.n_ant_el)
        };
        py::array_t<float> arr(shape);
        std::copy(r.power_cube.begin(), r.power_cube.end(),
                  arr.mutable_data());
        return arr;
      },
      "Return power cube as numpy float32 array, shape (n_range_bins, n_ant_az, n_ant_el).\n"
      "Returns empty array if download_result=False was used in process().")

    .def("__repr__", [](const range_angle::RangeAngleResult& r) {
        return "<RangeAngleResult"
               " success=" + std::string(r.success ? "True" : "False") +
               " n_range_bins=" + std::to_string(r.n_range_bins) +
               " targets=" + std::to_string(r.targets.size()) +
               ">";
    });

  // ── RangeAngleProcessor (Python wrapper) ─────────────────────────────────

  py::class_<PyRangeAngleProcessor>(m, "RangeAngleProcessor",
      "LFM range-angle processing pipeline (ROCm/HIP).\n\n"
      "Pipeline stages:\n"
      "  1. DechirpWindowOp  — multiply by conj(LFM ref) + Hann window\n"
      "  2. RangeFftOp       — batched 1-D forward FFT along range axis (hipFFT)\n"
      "  3. TransposeOp      — reorder [n_ant x n_range_bins] -> [n_range_bins x n_ant]\n"
      "  4. BeamFftOp        — 2-D spatial FFT (azimuth x elevation) + fftshift + |.|^2\n"
      "  5. PeakSearchOp     — find peak(s) in power cube -> TargetInfo list\n\n"
      "Usage:\n"
      "  p = gpuworklib.RangeAngleParams()\n"
      "  p.n_ant_az = 8; p.n_ant_el = 8; p.n_samples = 50000\n"
      "  p.f_start = -5e6; p.f_end = 5e6; p.sample_rate = 12e6\n"
      "  proc = gpuworklib.RangeAngleProcessor(ctx)   # ctx: ROCmGPUContext\n"
      "  proc.set_params(p)\n"
      "  iq = np.zeros((n_ant * n_samples,), dtype=np.complex64)\n"
      "  result = proc.process(iq, download_result=True)\n"
      "  print(result.targets[0].range_m)")
    .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
         "Create RangeAngleProcessor bound to a ROCm GPU context.")

    .def("set_params",
         &PyRangeAngleProcessor::set_params,
         py::arg("params"),
         "Set processing parameters.\n\n"
         "Computes derived fields (n_range_bins, range_res_m, nfft_range).\n"
         "Must be called before process().\n\n"
         "Args:\n"
         "  params: RangeAngleParams")

    .def("get_params",
         &PyRangeAngleProcessor::get_params,
         "Return current RangeAngleParams (copy).")

    .def("process",
         &PyRangeAngleProcessor::process,
         py::arg("data"),
         py::arg("download_result") = true,
         "Run full range-angle processing pipeline on GPU.\n\n"
         "Args:\n"
         "  data            : numpy complex64, flat antenna-major layout\n"
         "                    size must equal n_ant_az * n_ant_el * n_samples\n"
         "  download_result : if True, copy power cube to result (default True)\n\n"
         "Returns:\n"
         "  RangeAngleResult with:\n"
         "    .success       — True on success\n"
         "    .targets       — list[TargetInfo]\n"
         "    .error_message — non-empty on failure\n"
         "  and optionally power cube via .power_cube_numpy()")

    .def("__repr__", [](const PyRangeAngleProcessor& proc) {
        const auto& p = proc.get_params();
        return "<RangeAngleProcessor"
               " n_ant=" + std::to_string(p.get_n_antennas()) +
               " B=" + std::to_string(static_cast<int>(p.get_bandwidth())) + "Hz"
               " fs=" + std::to_string(static_cast<int>(p.sample_rate)) + "Hz"
               ">";
    });
}
