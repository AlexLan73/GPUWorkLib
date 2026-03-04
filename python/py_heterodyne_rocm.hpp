#pragma once

/**
 * @file py_heterodyne_rocm.hpp
 * @brief Python wrapper for HeterodyneProcessorROCm
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Note: HeterodyneProcessorROCm is a lower-level processor than HeterodyneDechirp.
 * It exposes Dechirp (multiply-conjugate) and Correct (freq correction).
 *
 * Usage from Python:
 *   het = gpuworklib.HeterodyneROCm(ctx)
 *   het.set_params(f_start=0, f_end=2e6, sample_rate=12e6,
 *                  num_samples=8000, num_antennas=5)
 *   dc  = het.dechirp(rx_signal, ref_signal)
 *   out = het.correct(dc, f_beat_hz=[123.0, 456.0])
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-24
 */

#include "processors/heterodyne_processor_rocm.hpp"
#include "heterodyne_params.hpp"

// ============================================================================
// PyHeterodyneROCm — LFM heterodyne processor (ROCm)
// ============================================================================

// Низкоуровневый процессор гетеродина (ROCm). В отличие от HeterodyneDechirp,
// предоставляет отдельные операции без полного пайплайна:
//   Dechirp: поэлементное rx * conj(ref) — только перемножение
//   Correct: частотная коррекция exp(j*2pi*f_beat/fs * n) — сдвиг спектра
// Нужен когда FFT/поиск максимума делается снаружи (например, собственный Python код).
// params_ хранится отдельно от процессора — нет SetParams(), только поля структуры.
class PyHeterodyneROCm {
public:
  explicit PyHeterodyneROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), proc_(ctx.backend()) {}

  void set_params(float f_start, float f_end, float sample_rate,
                  int num_samples, int num_antennas) {
    params_.f_start      = f_start;
    params_.f_end        = f_end;
    params_.sample_rate  = sample_rate;
    params_.num_samples  = num_samples;
    params_.num_antennas = num_antennas;
  }

  py::array_t<std::complex<float>> dechirp(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> rx,
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> ref)
  {
    auto rx_buf  = rx.request();
    auto ref_buf = ref.request();

    size_t rx_total = 1;
    for (py::ssize_t d = 0; d < rx_buf.ndim; ++d)
      rx_total *= static_cast<size_t>(rx_buf.shape[d]);

    size_t ref_total = 1;
    for (py::ssize_t d = 0; d < ref_buf.ndim; ++d)
      ref_total *= static_cast<size_t>(ref_buf.shape[d]);

    auto* rx_ptr  = static_cast<std::complex<float>*>(rx_buf.ptr);
    auto* ref_ptr = static_cast<std::complex<float>*>(ref_buf.ptr);

    std::vector<std::complex<float>> rx_vec(rx_ptr,  rx_ptr  + rx_total);
    std::vector<std::complex<float>> ref_vec(ref_ptr, ref_ptr + ref_total);

    std::vector<std::complex<float>> result;
    {
      py::gil_scoped_release release;
      result = proc_.Dechirp(rx_vec, ref_vec, params_);
    }

    return vector_to_numpy(std::move(result));
  }

  py::array_t<std::complex<float>> correct(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> dc,
      py::list f_beat_list)
  {
    auto buf = dc.request();
    size_t total = 1;
    for (py::ssize_t d = 0; d < buf.ndim; ++d)
      total *= static_cast<size_t>(buf.shape[d]);

    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    std::vector<std::complex<float>> dc_vec(ptr, ptr + total);

    std::vector<float> f_beat;
    for (auto item : f_beat_list)
      f_beat.push_back(item.cast<float>());

    std::vector<std::complex<float>> result;
    {
      py::gil_scoped_release release;
      result = proc_.Correct(dc_vec, f_beat, params_);
    }

    return vector_to_numpy(std::move(result));
  }

  py::dict get_params() const {
    py::dict d;
    d["f_start"]      = params_.f_start;
    d["f_end"]        = params_.f_end;
    d["sample_rate"]  = params_.sample_rate;
    d["num_samples"]  = params_.num_samples;
    d["num_antennas"] = params_.num_antennas;
    d["bandwidth"]    = params_.GetBandwidth();
    d["duration"]     = params_.GetDuration();
    d["chirp_rate"]   = params_.GetChirpRate();
    return d;
  }

private:
  ROCmGPUContext& ctx_;
  drv_gpu_lib::HeterodyneProcessorROCm proc_;
  drv_gpu_lib::HeterodyneParams params_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_heterodyne_rocm(py::module& m) {
  py::class_<PyHeterodyneROCm>(m, "HeterodyneROCm",
      "LFM heterodyne processor (ROCm).\n\n"
      "Exposes Dechirp and Correct operations:\n"
      "  Dechirp: s_dc = rx * conj(ref)   (element-wise)\n"
      "  Correct: freq correction via exp(j * phase_step * n)\n\n"
      "Usage:\n"
      "  het = gpuworklib.HeterodyneROCm(ctx)\n"
      "  het.set_params(f_start=0, f_end=2e6, sample_rate=12e6,\n"
      "                 num_samples=8000, num_antennas=5)\n"
      "  dc  = het.dechirp(rx_signal, ref_signal)\n"
      "  out = het.correct(dc, [f_beat_hz])\n")
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create HeterodyneROCm bound to ROCm GPU context")

      .def("set_params", &PyHeterodyneROCm::set_params,
           py::arg("f_start"), py::arg("f_end"),
           py::arg("sample_rate"), py::arg("num_samples"),
           py::arg("num_antennas"),
           "Set LFM parameters.\n\n"
           "Args:\n"
           "  f_start: LFM start frequency (Hz)\n"
           "  f_end: LFM end frequency (Hz), B = f_end - f_start\n"
           "  sample_rate: sampling rate (Hz)\n"
           "  num_samples: samples per antenna (N)\n"
           "  num_antennas: number of antennas/channels")

      .def("dechirp", &PyHeterodyneROCm::dechirp,
           py::arg("rx"), py::arg("ref"),
           "Dechirp: s_dc = rx * conj(ref) on GPU.\n\n"
           "Args:\n"
           "  rx:  numpy complex64 (antennas*samples,)\n"
           "  ref: numpy complex64 (samples,) or (antennas*samples,)\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64: dechirped signal")

      .def("correct", &PyHeterodyneROCm::correct,
           py::arg("dc"), py::arg("f_beat_hz"),
           "Frequency correction on GPU.\n\n"
           "Args:\n"
           "  dc: numpy complex64 dechirped signal\n"
           "  f_beat_hz: list of beat frequencies per antenna [Hz]\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64: corrected signal")

      .def_property_readonly("params", &PyHeterodyneROCm::get_params,
           "LFM parameters as dict")

      .def("__repr__", [](const PyHeterodyneROCm& self) {
          auto p = self.get_params();
          return "<HeterodyneROCm B=" +
                 std::to_string(static_cast<int>(p["bandwidth"].cast<float>())) +
                 " Hz, N=" +
                 std::to_string(p["num_samples"].cast<int>()) + ">";
      });
}
