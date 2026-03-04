#pragma once

/**
 * @file py_heterodyne.hpp
 * @brief Python wrapper for HeterodyneDechirp (LFM dechirp pipeline)
 *
 * Include AFTER GPUContext and vector_to_numpy definitions.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-21
 */

#include "heterodyne_dechirp.hpp"
#include "heterodyne_params.hpp"

// ============================================================================
// PyHeterodyneDechirp — LFM dechirp pipeline wrapper
// ============================================================================

// Высокоуровневый пайплайн LFM dechirp: полный цикл от RX-данных до диапазона.
// Отличие от HeterodyneROCm (низкоуровневый): здесь внутри всё — генерация reference,
// dechirp, FFT, поиск максимума, вычисление f_beat и дальности R = c*T*f_beat/(2B).
// Два конструктора — OpenCL и ROCm — чтобы один класс работал в обоих бэкендах.
// ctx_ / rocm_ctx_ — не владеют контекстом (raw pointer), lifetime контекста
// должен превышать lifetime объекта!
class PyHeterodyneDechirp {
public:
  explicit PyHeterodyneDechirp(GPUContext& ctx)
      : ctx_(&ctx), het_(ctx.backend(), drv_gpu_lib::BackendType::OPENCL) {}

#if ENABLE_ROCM
  explicit PyHeterodyneDechirp(ROCmGPUContext& ctx)
      : rocm_ctx_(&ctx), het_(ctx.backend(), drv_gpu_lib::BackendType::ROCm) {}
#endif

  void set_params(float f_start, float f_end, float sample_rate,
                  int num_samples, int num_antennas) {
    drv_gpu_lib::HeterodyneParams p;
    p.f_start = f_start;
    p.f_end = f_end;
    p.sample_rate = sample_rate;
    p.num_samples = num_samples;
    p.num_antennas = num_antennas;
    het_.SetParams(p);
  }

  py::dict process(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> rx) {

    auto buf = rx.request();
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);

    auto& params = het_.GetParams();
    size_t expected = static_cast<size_t>(params.num_antennas) * params.num_samples;

    size_t total;
    if (buf.ndim == 2) {
      total = static_cast<size_t>(buf.shape[0]) * buf.shape[1];
    } else if (buf.ndim == 1) {
      total = static_cast<size_t>(buf.shape[0]);
    } else {
      throw std::invalid_argument("Input must be 1D or 2D");
    }

    if (total != expected) {
      throw std::invalid_argument(
          "rx size mismatch: expected " + std::to_string(expected)
          + ", got " + std::to_string(total));
    }

    std::vector<std::complex<float>> rx_vec(ptr, ptr + total);

    drv_gpu_lib::HeterodyneResult result;
    {
      py::gil_scoped_release release;
      result = het_.Process(rx_vec);
    }

    return build_dict(result);
  }

  // Обработка данных, уже находящихся на GPU (cl_mem адрес как целое число).
  // Используется когда данные пришли из другого модуля через output="gpu" (PyGPUBuffer).
  // Избегает лишнего round-trip CPU→GPU, который происходит в process().
  // gpu_buffer_ptr — целочисленный адрес cl_mem объекта (получить через gpu_buf.ptr).
  py::dict process_external(uintptr_t gpu_buffer_ptr) {
    void* cl_mem_ptr = reinterpret_cast<void*>(gpu_buffer_ptr);
    auto& params = het_.GetParams();

    drv_gpu_lib::HeterodyneResult result;
    {
      py::gil_scoped_release release;
      result = het_.ProcessExternal(cl_mem_ptr, params);
    }

    return build_dict(result);
  }

  py::dict get_params() const {
    auto& p = het_.GetParams();
    py::dict d;
    d["f_start"] = p.f_start;
    d["f_end"] = p.f_end;
    d["sample_rate"] = p.sample_rate;
    d["num_samples"] = p.num_samples;
    d["num_antennas"] = p.num_antennas;
    d["bandwidth"] = p.GetBandwidth();
    d["duration"] = p.GetDuration();
    d["chirp_rate"] = p.GetChirpRate();
    d["bin_width"] = p.GetBinWidth();
    return d;
  }

private:
  // Конвертирует C++ HeterodyneResult в Python dict — удобно для распаковки
  // в Python: result['antennas'][0]['range_m']. Каждая антенна — отдельный dict
  // с полным набором результатов (частота, дальность, SNR).
  py::dict build_dict(const drv_gpu_lib::HeterodyneResult& result) {
    py::dict out;
    out["success"] = result.success;
    out["error_message"] = result.error_message;

    py::list ant_list;
    for (auto& a : result.antennas) {
      py::dict ad;
      ad["antenna_idx"] = a.antenna_idx;
      ad["f_beat_hz"] = a.f_beat_hz;
      ad["f_beat_bin"] = a.f_beat_bin;
      ad["range_m"] = a.range_m;
      ad["peak_amplitude"] = a.peak_amplitude;
      ad["peak_snr_db"] = a.peak_snr_db;
      ant_list.append(ad);
    }
    out["antennas"] = ant_list;

    py::list maxpos;
    for (float v : result.max_positions) maxpos.append(v);
    out["max_positions"] = maxpos;

    return out;
  }

  GPUContext* ctx_ = nullptr;
#if ENABLE_ROCM
  ROCmGPUContext* rocm_ctx_ = nullptr;
#endif
  drv_gpu_lib::HeterodyneDechirp het_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_heterodyne(py::module& m) {
  py::class_<PyHeterodyneDechirp>(m, "HeterodyneDechirp",
      "LFM dechirp pipeline on GPU.\n\n"
      "Pipeline:\n"
      "  1. Generate conjugate LFM reference\n"
      "  2. Dechirp: s_dc = s_rx * conj(s_tx)\n"
      "  3. FFT -> find peak -> f_beat\n"
      "  4. Calculate range R = c*T*f_beat/(2B)\n"
      "  5. Calculate SNR\n\n"
      "Usage:\n"
      "  het = gpuworklib.HeterodyneDechirp(ctx)\n"
      "  het.set_params(f_start=0, f_end=2e6, sample_rate=12e6,\n"
      "                 num_samples=8000, num_antennas=5)\n"
      "  result = het.process(rx_data)\n"
      "  print(result['antennas'][0]['f_beat_hz'])\n")
      .def(py::init<GPUContext&>(), py::arg("ctx"),
           "Create HeterodyneDechirp bound to OpenCL GPU context")
#if ENABLE_ROCM
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create HeterodyneDechirp bound to ROCm GPU context (AMD)")
#endif

      .def("set_params", &PyHeterodyneDechirp::set_params,
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

      .def("process", &PyHeterodyneDechirp::process,
           py::arg("rx"),
           "Run dechirp pipeline on CPU data.\n\n"
           "Args:\n"
           "  rx: numpy complex64 (antennas*samples,) or (antennas, samples)\n\n"
           "Returns:\n"
           "  dict with keys: success, error_message, antennas (list of dicts),\n"
           "  max_positions. Each antenna dict has: f_beat_hz, range_m, peak_snr_db")

      .def("process_external", &PyHeterodyneDechirp::process_external,
           py::arg("gpu_buffer_ptr"),
           "Run dechirp on an existing GPU cl_mem buffer.\n\n"
           "Args:\n"
           "  gpu_buffer_ptr: integer address of cl_mem object\n\n"
           "Returns:\n"
           "  dict (same structure as process())")

      .def_property_readonly("params", &PyHeterodyneDechirp::get_params,
           "Current LFM parameters as dict")

      .def("__repr__", [](const PyHeterodyneDechirp& self) {
          auto p = self.get_params();
          return "<HeterodyneDechirp B=" +
                 std::to_string(static_cast<int>(p["bandwidth"].cast<float>())) +
                 " Hz, N=" +
                 std::to_string(p["num_samples"].cast<int>()) + ">";
      });
}
