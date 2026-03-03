#pragma once

/**
 * @file py_fm_correlator_rocm.hpp
 * @brief Python wrapper for FMCorrelator (ROCm)
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Usage from Python:
 *   corr = gpuworklib.FMCorrelatorROCm(ctx)
 *   corr.set_params(fft_size=32768, num_shifts=32, num_signals=10)
 *   corr.prepare_reference()
 *   peaks = corr.run_test_pattern(shift_step=2)  # numpy [S, K, n_kg]
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-03
 */

#include "fm_correlator.hpp"
#include "fm_correlator_types.hpp"

// ════════════════════════════════════════════════════════════════════════════
// Helper: vector -> numpy 3D [d1, d2, d3]
// ════════════════════════════════════════════════════════════════════════════

template<typename T>
py::array_t<T> vector_to_numpy_3d(std::vector<T>&& data,
                                   size_t d1, size_t d2, size_t d3) {
  auto* vec = new std::vector<T>(std::move(data));
  auto capsule = py::capsule(vec, [](void* ptr) {
    delete static_cast<std::vector<T>*>(ptr);
  });
  std::vector<py::ssize_t> shape = {
    static_cast<py::ssize_t>(d1),
    static_cast<py::ssize_t>(d2),
    static_cast<py::ssize_t>(d3)
  };
  std::vector<py::ssize_t> strides = {
    static_cast<py::ssize_t>(d2 * d3 * sizeof(T)),
    static_cast<py::ssize_t>(d3 * sizeof(T)),
    static_cast<py::ssize_t>(sizeof(T))
  };
  return py::array_t<T>(shape, strides, vec->data(), capsule);
}

// ════════════════════════════════════════════════════════════════════════════
// PyFMCorrelatorROCm
// ════════════════════════════════════════════════════════════════════════════

class PyFMCorrelatorROCm {
public:
  explicit PyFMCorrelatorROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), correlator_(ctx.backend()) {}

  void set_params(int fft_size, int num_shifts, int num_signals,
                  int num_output_points, uint32_t polynomial, uint32_t seed) {
    params_.fft_size = static_cast<size_t>(fft_size);
    params_.num_shifts = num_shifts;
    params_.num_signals = num_signals;
    params_.num_output_points = num_output_points;
    params_.lfsr_polynomial = polynomial;
    params_.lfsr_seed = seed;
    correlator_.SetParams(params_);
  }

  py::array_t<float> generate_msequence(uint32_t seed) {
    auto seq = correlator_.GenerateMSequence(seed);
    return vector_to_numpy(std::move(seq));
  }

  void prepare_reference() {
    py::gil_scoped_release release;
    correlator_.PrepareReference();
  }

  void prepare_reference_from_data(
      py::array_t<float, py::array::c_style | py::array::forcecast> ref) {
    auto buf = ref.request();
    auto* ptr = static_cast<float*>(buf.ptr);
    std::vector<float> vec(ptr, ptr + buf.shape[0]);
    py::gil_scoped_release release;
    correlator_.PrepareReference(vec);
  }

  py::array_t<float> process(
      py::array_t<float, py::array::c_style | py::array::forcecast> input) {
    auto buf = input.request();
    size_t total = 1;
    for (int d = 0; d < buf.ndim; ++d)
      total *= static_cast<size_t>(buf.shape[d]);
    auto* ptr = static_cast<float*>(buf.ptr);
    std::vector<float> vec(ptr, ptr + total);

    drv_gpu_lib::FMCorrelatorResult result;
    {
      py::gil_scoped_release release;
      result = correlator_.Process(vec);
    }

    return vector_to_numpy_3d(std::move(result.peaks),
        static_cast<size_t>(result.num_signals),
        static_cast<size_t>(result.num_shifts),
        static_cast<size_t>(result.num_output_points));
  }

  py::array_t<float> run_test_pattern(int shift_step) {
    drv_gpu_lib::FMCorrelatorResult result;
    {
      py::gil_scoped_release release;
      result = correlator_.RunTestPattern(shift_step);
    }

    return vector_to_numpy_3d(std::move(result.peaks),
        static_cast<size_t>(result.num_signals),
        static_cast<size_t>(result.num_shifts),
        static_cast<size_t>(result.num_output_points));
  }

private:
  ROCmGPUContext& ctx_;
  drv_gpu_lib::FMCorrelator correlator_;
  drv_gpu_lib::FMCorrelatorParams params_;
};

// ════════════════════════════════════════════════════════════════════════════
// Binding registration
// ════════════════════════════════════════════════════════════════════════════

inline void register_fm_correlator_rocm(py::module& m) {
  py::class_<PyFMCorrelatorROCm>(m, "FMCorrelatorROCm",
      "FM Correlator — frequency-domain correlation for M-sequence phase modulation (ROCm).\n\n"
      "Pipeline: cyclic_shifts → C2C FFT → R2C FFT → multiply_conj → C2R IFFT → extract\n\n"
      "Usage:\n"
      "  corr = gpuworklib.FMCorrelatorROCm(ctx)\n"
      "  corr.set_params(fft_size=32768, num_shifts=32, num_signals=10)\n"
      "  corr.prepare_reference()\n"
      "  peaks = corr.run_test_pattern(shift_step=2)  # numpy [S, K, n_kg]\n")
    .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
         "Create FMCorrelatorROCm bound to ROCm GPU context")

    .def("set_params", &PyFMCorrelatorROCm::set_params,
         py::arg("fft_size") = 32768,
         py::arg("num_shifts") = 32,
         py::arg("num_signals") = 5,
         py::arg("num_output_points") = 2000,
         py::arg("polynomial") = 0x00400007u,
         py::arg("seed") = 0x12345678u,
         "Set correlator parameters.\n\n"
         "Args:\n"
         "  fft_size: FFT size (power of 2)\n"
         "  num_shifts: K — cyclic shifts of reference\n"
         "  num_signals: S — input signals\n"
         "  num_output_points: n_kg — first IFFT points to extract\n"
         "  polynomial: LFSR polynomial (default 0x00400007, primitive)\n"
         "  seed: LFSR initial seed (default 0x12345678)")

    .def("generate_msequence", &PyFMCorrelatorROCm::generate_msequence,
         py::arg("seed") = 0x12345678u,
         "Generate M-sequence on CPU.\n\n"
         "Returns:\n"
         "  numpy.ndarray float32 [N] with values {+1.0, -1.0}")

    .def("prepare_reference", &PyFMCorrelatorROCm::prepare_reference,
         "Generate M-sequence internally and upload to GPU.\n"
         "Uses seed from set_params().")

    .def("prepare_reference_from_data", &PyFMCorrelatorROCm::prepare_reference_from_data,
         py::arg("ref"),
         "Prepare reference from external numpy array.\n\n"
         "Args:\n"
         "  ref: numpy float32 [N]")

    .def("process", &PyFMCorrelatorROCm::process,
         py::arg("input_signals"),
         "Run correlation with external input signals.\n\n"
         "Args:\n"
         "  input_signals: numpy float32 [S, N] or [S*N]\n\n"
         "Returns:\n"
         "  numpy.ndarray float32 [S, K, n_kg] — correlation peaks")

    .def("run_test_pattern", &PyFMCorrelatorROCm::run_test_pattern,
         py::arg("shift_step") = 2,
         "Run test pattern — GPU generates inputs via circshift(ref, s*shift_step).\n"
         "No data transfer needed — only shift_step parameter.\n\n"
         "Args:\n"
         "  shift_step: cyclic shift step between signals\n\n"
         "Returns:\n"
         "  numpy.ndarray float32 [S, K, n_kg] — correlation peaks");
}
