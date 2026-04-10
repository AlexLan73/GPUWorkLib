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
#include "branch_selector.hpp"  // SNR_07

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

  // ── ComputeAll API ───────────────────────────────────────────────────────

  py::list compute_all(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data,
      uint32_t beam_count)
  {
    {
      auto buf_check = data.request();
      if (buf_check.ndim == 2 && beam_count == 0)
        beam_count = static_cast<uint32_t>(buf_check.shape[0]);
      if (beam_count == 0) beam_count = 1;
    }
    auto vec = to_vector(data, beam_count);
    uint32_t n_point = static_cast<uint32_t>(vec.size() / beam_count);

    statistics::StatisticsParams params;
    params.beam_count = beam_count;
    params.n_point    = n_point;

    std::vector<statistics::FullStatisticsResult> results;
    {
      py::gil_scoped_release release;
      results = proc_.ComputeAll(vec, params);
    }

    py::list out;
    for (const auto& r : results) {
      py::dict d;
      d["beam_id"]          = r.beam_id;
      d["mean_real"]        = r.mean.real();
      d["mean_imag"]        = r.mean.imag();
      d["variance"]         = r.variance;
      d["std_dev"]          = r.std_dev;
      d["mean_magnitude"]   = r.mean_magnitude;
      d["median_magnitude"] = r.median_magnitude;
      out.append(d);
    }
    return out;
  }

  py::list compute_all_float(
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

    std::vector<statistics::FullStatisticsResult> results;
    {
      py::gil_scoped_release release;
      results = proc_.ComputeAllFloat(vec, params);
    }

    py::list out;
    for (const auto& r : results) {
      py::dict d;
      d["beam_id"]          = r.beam_id;
      d["mean_real"]        = r.mean.real();   // always 0.0 for float path
      d["mean_imag"]        = r.mean.imag();   // always 0.0 for float path
      d["variance"]         = r.variance;
      d["std_dev"]          = r.std_dev;
      d["mean_magnitude"]   = r.mean_magnitude;
      d["median_magnitude"] = r.median_magnitude;
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

  // ==========================================================================
  // SNR_07: compute_snr_db (SNR-estimator, CA-CFAR)
  // ==========================================================================

  statistics::SnrEstimationResult compute_snr_db(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data_np,
      uint32_t n_antennas,
      uint32_t n_samples,
      const statistics::SnrEstimationConfig& config)
  {
    if (data_np.ndim() != 2) {
      throw std::invalid_argument("compute_snr_db: expected 2D numpy array");
    }
    if (static_cast<uint32_t>(data_np.shape(0)) != n_antennas ||
        static_cast<uint32_t>(data_np.shape(1)) != n_samples) {
      throw std::invalid_argument(
          "compute_snr_db: shape mismatch — got (" +
          std::to_string(data_np.shape(0)) + "," +
          std::to_string(data_np.shape(1)) + "), expected (" +
          std::to_string(n_antennas) + "," + std::to_string(n_samples) + ")");
    }

    // Numpy → flat std::vector<complex<float>> (C-order contiguous)
    auto buf = data_np.request();
    const std::complex<float>* ptr = static_cast<const std::complex<float>*>(buf.ptr);
    std::vector<std::complex<float>> data(ptr, ptr + data_np.size());

    statistics::SnrEstimationResult result;
    {
      py::gil_scoped_release release;
      result = proc_.ComputeSnrDb(data, n_antennas, n_samples, config);
    }
    return result;
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
// SNR-estimator types (SNR_07) — регистрация enum/struct для Python
// ============================================================================

inline void register_snr_types(py::module& m) {
  using namespace statistics;

  // ── BranchType enum ────────────────────────────────────────────────────
  py::enum_<BranchType>(m, "BranchType",
      "SNR branch category for Low/Mid/High processing paths.")
    .value("Low",  BranchType::Low)
    .value("Mid",  BranchType::Mid)
    .value("High", BranchType::High)
    .export_values();

  // ── BranchThresholds ───────────────────────────────────────────────────
  py::class_<BranchThresholds>(m, "BranchThresholds",
      "Пороги переключения branch'ей (калибровано Python Эксп.5).\n\n"
      "Defaults: low_to_mid=15 dB, mid_to_high=30 dB, hysteresis=2 dB.")
    .def(py::init<>())
    .def_readwrite("low_to_mid_db",  &BranchThresholds::low_to_mid_db)
    .def_readwrite("mid_to_high_db", &BranchThresholds::mid_to_high_db)
    .def_readwrite("hysteresis_db",  &BranchThresholds::hysteresis_db);

  // ── SnrEstimationConfig ────────────────────────────────────────────────
  py::class_<SnrEstimationConfig>(m, "SnrEstimationConfig",
      "Config для SNR-estimator (CA-CFAR). Все 0-поля → auto режим.\n\n"
      "target_n_fft=0 → 2048, step_samples=0 → auto, step_antennas=0 → auto.\n"
      "Default window=Hann (решает sinc sidelobes).")
    .def(py::init<>())
    .def_readwrite("target_n_fft",         &SnrEstimationConfig::target_n_fft)
    .def_readwrite("step_samples",         &SnrEstimationConfig::step_samples)
    .def_readwrite("step_antennas",        &SnrEstimationConfig::step_antennas)
    .def_readwrite("guard_bins",           &SnrEstimationConfig::guard_bins)
    .def_readwrite("ref_bins",             &SnrEstimationConfig::ref_bins)
    .def_readwrite("search_full_spectrum", &SnrEstimationConfig::search_full_spectrum)
    .def_readwrite("with_dechirp",         &SnrEstimationConfig::with_dechirp)
    .def_readwrite("thresholds",           &SnrEstimationConfig::thresholds)
    .def("validate", &SnrEstimationConfig::Validate,
         "Validate invariants, throws ValueError on error.");

  // ── SnrEstimationResult (readonly, БЕЗ BranchType!) ───────────────────
  py::class_<SnrEstimationResult>(m, "SnrEstimationResult",
      "Result SNR-estimator.\n\n"
      "NB: Нет поля branch — классификация через BranchSelector.")
    .def(py::init<>())
    .def_readonly("snr_db_global",       &SnrEstimationResult::snr_db_global)
    .def_readonly("snr_db_per_antenna",  &SnrEstimationResult::snr_db_per_antenna)
    .def_readonly("used_antennas",       &SnrEstimationResult::used_antennas)
    .def_readonly("used_bins",           &SnrEstimationResult::used_bins)
    .def_readonly("actual_step_samples", &SnrEstimationResult::actual_step_samples)
    .def_readonly("n_actual",            &SnrEstimationResult::n_actual);

  // ── BranchSelector (stateful, hysteresis) ─────────────────────────────
  py::class_<BranchSelector>(m, "BranchSelector",
      "Stateful branch selector с hysteresis.\n\n"
      "Один экземпляр на поток/pipeline. Держит state между Select() calls.")
    .def(py::init<>())
    .def("select",  &BranchSelector::Select,
         py::arg("snr_db"), py::arg("thresholds"),
         "Select branch c hysteresis, updates state. NaN/Inf → current.")
    .def("current", &BranchSelector::Current,
         "Get current branch без изменения state.")
    .def("reset",   &BranchSelector::Reset,
         py::arg("to") = BranchType::Low,
         "Reset state (start of new session).");
}

// ============================================================================
// Binding registration
// ============================================================================

inline void register_statistics(py::module& m) {
  // SNR types register first — нужны для .def("compute_snr_db", ...) ниже
  register_snr_types(m);

  py::class_<PyStatisticsProcessor>(m, "StatisticsProcessor",
      "GPU statistics processor (ROCm).\n\n"
      "Computes per-beam statistics on complex float signal data.\n\n"
      "Methods:\n"
      "  compute_mean         - complex mean per beam\n"
      "  compute_median       - median of magnitudes per beam (radix sort)\n"
      "  compute_statistics   - full stats (mean+variance+std) per beam\n"
      "  compute_all          - statistics + median in one GPU call (faster)\n"
      "  compute_all_float    - statistics + median on float magnitudes\n\n"
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

      .def("compute_all", &PyStatisticsProcessor::compute_all,
           py::arg("data"), py::arg("beam_count") = 1,
           "Compute full statistics + median per beam in one GPU call.\n\n"
           "Equivalent to compute_statistics() + compute_median() but eliminates\n"
           "double upload and double sync — faster for CPU data path.\n\n"
           "Args:\n"
           "  data: numpy complex64 (beam_count * n_point,) or (beam_count, n_point)\n"
           "  beam_count: number of beams (default 1)\n\n"
           "Returns:\n"
           "  list of dicts per beam:\n"
           "    beam_id, mean_real, mean_imag, variance, std_dev,\n"
           "    mean_magnitude, median_magnitude")

      .def("compute_all_float", &PyStatisticsProcessor::compute_all_float,
           py::arg("data"), py::arg("beam_count") = 1,
           "Compute statistics + median on float magnitudes per beam.\n\n"
           "Note: mean_real and mean_imag are always 0.0 for float input path.\n\n"
           "Args:\n"
           "  data: numpy float32 (beam_count * n_point,) or (beam_count, n_point)\n"
           "  beam_count: number of beams (default 1)\n\n"
           "Returns:\n"
           "  list of dicts per beam:\n"
           "    beam_id, mean_real(=0), mean_imag(=0), variance, std_dev,\n"
           "    mean_magnitude, median_magnitude")

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

      // ── SNR_07: compute_snr_db (CA-CFAR SNR-estimator) ───────────────
      .def("compute_snr_db", &PyStatisticsProcessor::compute_snr_db,
           py::arg("data"), py::arg("n_antennas"),
           py::arg("n_samples"), py::arg("config"),
           "Compute SNR (dB) from numpy complex64 array via CA-CFAR.\n\n"
           "Args:\n"
           "  data:       numpy complex64 2D array, shape (n_antennas, n_samples)\n"
           "  n_antennas: number of input antennas\n"
           "  n_samples:  samples per antenna\n"
           "  config:     SnrEstimationConfig (см. snr_defaults)\n\n"
           "Returns:\n"
           "  SnrEstimationResult — snr_db_global, used_antennas, used_bins, ...\n"
           "  NB: НЕТ поля branch — использовать BranchSelector для классификации.\n\n"
           "Pipeline: upload → gather → FFT(Hann)|X|² → CA-CFAR → median.\n"
           "Калибровано Python моделью (PyPanelAntennas/SNR/, P_correct=97.9%).")

      .def("__repr__", [](const PyStatisticsProcessor&) {
          return "<StatisticsProcessor>";
      });
}
