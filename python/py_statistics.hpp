#pragma once

/**
 * @file py_statistics.hpp
 * @brief Python wrapper for StatisticsProcessor (ROCm)
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Usage from Python:
 *   proc = gpuworklib.StatisticsProcessor(ctx)
 *   results = proc.compute_statistics(data, beam_count=4)
 *   print(results[0]['mean_real'], results[0]['std_dev'])
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-24
 */

#include "statistics_processor.hpp"
#include "statistics_types.hpp"

// ============================================================================
// PyStatisticsProcessor — GPU statistics on complex signal data (ROCm)
// ============================================================================

// Вычисляет статистику по нескольким «лучам» (beam) параллельно на GPU.
// Данные организованы как flat array: [beam0_sample0, beam0_sample1, ..., beam1_sample0, ...]
// — то есть beam_count * n_point элементов, beam-major layout.
// Три метода: compute_mean (быстро, только среднее), compute_median (требует radix sort),
// compute_statistics (Welford — mean+variance за один проход, оптимально).
class PyStatisticsProcessor {
public:
  explicit PyStatisticsProcessor(ROCmGPUContext& ctx)
      : ctx_(ctx), proc_(ctx.backend()) {}

  py::list compute_mean(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data,
      uint32_t beam_count)
  {
    auto vec = to_vector(data, beam_count);
    uint32_t n_point = static_cast<uint32_t>(vec.size() / beam_count);

    statistics::StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;

    std::vector<statistics::MeanResult> results;
    {
      py::gil_scoped_release release;
      results = proc_.ComputeMean(vec, params);
    }

    py::list out;
    for (const auto& r : results) {
      py::dict d;
      d["beam_id"]   = r.beam_id;
      d["mean_real"] = r.mean.real();
      d["mean_imag"] = r.mean.imag();
      out.append(d);
    }
    return out;
  }

  py::list compute_median(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data,
      uint32_t beam_count)
  {
    auto vec = to_vector(data, beam_count);
    uint32_t n_point = static_cast<uint32_t>(vec.size() / beam_count);

    statistics::StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;

    std::vector<statistics::MedianResult> results;
    {
      py::gil_scoped_release release;
      results = proc_.ComputeMedian(vec, params);
    }

    py::list out;
    for (const auto& r : results) {
      py::dict d;
      d["beam_id"]          = r.beam_id;
      d["median_magnitude"] = r.median_magnitude;
      out.append(d);
    }
    return out;
  }

  py::list compute_statistics(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data,
      uint32_t beam_count)
  {
    auto vec = to_vector(data, beam_count);
    uint32_t n_point = static_cast<uint32_t>(vec.size() / beam_count);

    statistics::StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;

    std::vector<statistics::StatisticsResult> results;
    {
      py::gil_scoped_release release;
      results = proc_.ComputeStatistics(vec, params);
    }

    py::list out;
    for (const auto& r : results) {
      py::dict d;
      d["beam_id"]        = r.beam_id;
      d["mean_real"]      = r.mean.real();
      d["mean_imag"]      = r.mean.imag();
      d["variance"]       = r.variance;
      d["std_dev"]        = r.std_dev;
      d["mean_magnitude"] = r.mean_magnitude;
      out.append(d);
    }
    return out;
  }

  // ── Float magnitude API ──────────────────────────────────────────────────

  py::list compute_statistics_float(
      py::array_t<float, py::array::c_style | py::array::forcecast> data,
      uint32_t beam_count)
  {
    auto buf = data.request();
    if (buf.ndim == 2 && beam_count == 0)
      beam_count = static_cast<uint32_t>(buf.shape[0]);
    if (beam_count == 0) beam_count = 1;

    auto vec = to_float_vector(data, beam_count);
    uint32_t n_point = static_cast<uint32_t>(vec.size() / beam_count);

    statistics::StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;

    std::vector<statistics::StatisticsResult> results;
    {
      py::gil_scoped_release release;
      results = proc_.ComputeStatisticsFloat(vec, params);
    }

    py::list out;
    for (const auto& r : results) {
      py::dict d;
      d["beam_id"]        = r.beam_id;
      d["variance"]       = r.variance;
      d["std_dev"]        = r.std_dev;
      d["mean_magnitude"] = r.mean_magnitude;
      out.append(d);
    }
    return out;
  }

  py::list compute_median_float(
      py::array_t<float, py::array::c_style | py::array::forcecast> data,
      uint32_t beam_count)
  {
    auto buf = data.request();
    if (buf.ndim == 2 && beam_count == 0)
      beam_count = static_cast<uint32_t>(buf.shape[0]);
    if (beam_count == 0) beam_count = 1;

    auto vec = to_float_vector(data, beam_count);
    uint32_t n_point = static_cast<uint32_t>(vec.size() / beam_count);

    statistics::StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;

    std::vector<statistics::MedianResult> results;
    {
      py::gil_scoped_release release;
      results = proc_.ComputeMedianFloat(vec, params);
    }

    py::list out;
    for (const auto& r : results) {
      py::dict d;
      d["beam_id"]          = r.beam_id;
      d["median_magnitude"] = r.median_magnitude;
      out.append(d);
    }
    return out;
  }

private:
  // Конвертирует numpy (любой формы) в flat vector. StatisticsProcessor ожидает
  // данные в beam-major порядке: сначала все samples beam[0], потом beam[1]...
  // Если данные пришли как 2D numpy (beam_count, n_point) — порядок уже правильный
  // (C-contiguous), если 1D — пользователь сам отвечает за layout.
  static std::vector<std::complex<float>> to_vector(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data,
      uint32_t beam_count)
  {
    auto buf = data.request();
    size_t total = 1;
    for (py::ssize_t d = 0; d < buf.ndim; ++d)
      total *= static_cast<size_t>(buf.shape[d]);

    if (beam_count == 0 || total % beam_count != 0)
      throw std::invalid_argument(
          "Data size " + std::to_string(total) +
          " is not divisible by beam_count " + std::to_string(beam_count));

    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    return std::vector<std::complex<float>>(ptr, ptr + total);
  }

  static std::vector<float> to_float_vector(
      py::array_t<float, py::array::c_style | py::array::forcecast> data,
      uint32_t beam_count)
  {
    auto buf = data.request();
    size_t total = 1;
    for (py::ssize_t d = 0; d < buf.ndim; ++d)
      total *= static_cast<size_t>(buf.shape[d]);

    if (beam_count == 0 || total % beam_count != 0)
      throw std::invalid_argument(
          "Data size " + std::to_string(total) +
          " is not divisible by beam_count " + std::to_string(beam_count));

    auto* ptr = static_cast<float*>(buf.ptr);
    return std::vector<float>(ptr, ptr + total);
  }

  ROCmGPUContext& ctx_;
  statistics::StatisticsProcessor proc_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_statistics(py::module& m) {
  py::class_<PyStatisticsProcessor>(m, "StatisticsProcessor",
      "GPU statistics processor (ROCm).\n\n"
      "Computes per-beam statistics on complex float signal data.\n\n"
      "Methods:\n"
      "  compute_mean       - complex mean per beam\n"
      "  compute_median     - median of magnitudes per beam (radix sort)\n"
      "  compute_statistics - full stats (mean+variance+std) per beam\n\n"
      "Usage:\n"
      "  proc = gpuworklib.StatisticsProcessor(ctx)\n"
      "  results = proc.compute_statistics(data, beam_count=4)\n"
      "  print(results[0]['mean_real'], results[0]['std_dev'])\n")
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create StatisticsProcessor bound to ROCm GPU context")

      .def("compute_mean", &PyStatisticsProcessor::compute_mean,
           py::arg("data"), py::arg("beam_count") = 1,
           "Compute complex mean per beam.\n\n"
           "Args:\n"
           "  data: numpy complex64 (beam_count * n_point,) or (beam_count, n_point)\n"
           "  beam_count: number of beams (default 1)\n\n"
           "Returns:\n"
           "  list of dicts: [{'beam_id':int, 'mean_real':float, 'mean_imag':float}, ...]")

      .def("compute_median", &PyStatisticsProcessor::compute_median,
           py::arg("data"), py::arg("beam_count") = 1,
           "Compute median of magnitudes per beam (GPU radix sort).\n\n"
           "Args:\n"
           "  data: numpy complex64 (beam_count * n_point,) or (beam_count, n_point)\n"
           "  beam_count: number of beams (default 1)\n\n"
           "Returns:\n"
           "  list of dicts: [{'beam_id':int, 'median_magnitude':float}, ...]")

      .def("compute_statistics", &PyStatisticsProcessor::compute_statistics,
           py::arg("data"), py::arg("beam_count") = 1,
           "Compute full statistics per beam (single-pass Welford).\n\n"
           "Args:\n"
           "  data: numpy complex64 (beam_count * n_point,) or (beam_count, n_point)\n"
           "  beam_count: number of beams (default 1)\n\n"
           "Returns:\n"
           "  list of dicts per beam:\n"
           "    beam_id, mean_real, mean_imag, variance, std_dev, mean_magnitude")

      .def("compute_statistics_float", &PyStatisticsProcessor::compute_statistics_float,
           py::arg("data"), py::arg("beam_count") = 1,
           "Compute statistics on float magnitudes per beam.\n\n"
           "Args:\n"
           "  data: numpy float32 (beam_count * n_point,) or (beam_count, n_point)\n"
           "  beam_count: number of beams (default 1)\n\n"
           "Returns:\n"
           "  list of dicts per beam:\n"
           "    beam_id, variance, std_dev, mean_magnitude")

      .def("compute_median_float", &PyStatisticsProcessor::compute_median_float,
           py::arg("data"), py::arg("beam_count") = 1,
           "Compute median of float magnitudes per beam.\n\n"
           "Args:\n"
           "  data: numpy float32 (beam_count * n_point,) or (beam_count, n_point)\n"
           "  beam_count: number of beams (default 1)\n\n"
           "Returns:\n"
           "  list of dicts: [{'beam_id':int, 'median_magnitude':float}, ...]")

      .def("__repr__", [](const PyStatisticsProcessor&) {
          return "<StatisticsProcessor>";
      });
}
