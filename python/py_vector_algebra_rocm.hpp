#pragma once

/**
 * @file py_vector_algebra_rocm.hpp
 * @brief Python wrapper for CholeskyInverterROCm (Task_11 v2: SymmetrizeMode)
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Usage from Python:
 *   inverter = gpuworklib.CholeskyInverterROCm(ctx, gpuworklib.SymmetrizeMode.GpuKernel)
 *   A_inv = inverter.invert_cpu(A.flatten(), n=341)     # (n, n) ndarray
 *   results = inverter.invert_batch_cpu(flat, n=64, batch_count=4)  # (4, 64, 64)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-26
 */

#if ENABLE_ROCM

#include "cholesky_inverter_rocm.hpp"

// ════════════════════════════════════════════════════════════════════════════
// PyCholeskyInverterROCm — Python wrapper
// ════════════════════════════════════════════════════════════════════════════

// Инверсия эрмитовой положительно определённой матрицы на GPU через:
//   rocSOLVER POTRF — Cholesky разложение A = L*L^H (нижний треугольник)
//   rocSOLVER POTRI — инверсия A^-1 по разложению (заполняет нижний треугольник)
// После POTRI нижний треугольник содержит инверсию, верхний — мусор.
// Симметризация (копирование lower→upper) — два режима:
//   Roundtrip: скачать на CPU, симметризовать, загрузить обратно (~медленнее)
//   GpuKernel: HIP ядро in-place (~быстрее, рекомендуется)
// std::unique_ptr<> — нельзя копировать объект (ресурс GPU), только move.
class PyCholeskyInverterROCm {
public:
  PyCholeskyInverterROCm(ROCmGPUContext& ctx,
                          vector_algebra::SymmetrizeMode mode)
      : ctx_(ctx),
        inverter_(std::make_unique<vector_algebra::CholeskyInverterROCm>(
            ctx.backend(), mode)) {}

  /// Инверсия одной матрицы n×n (CPU → GPU → CPU)
  py::array_t<std::complex<float>> invert_cpu(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> matrix_flat,
      int n) {
    auto buf = matrix_flat.request();
    const size_t expected = static_cast<size_t>(n) * n;
    if (static_cast<size_t>(buf.size) != expected) {
      throw std::invalid_argument(
          "invert_cpu: matrix_flat.size=" + std::to_string(buf.size) +
          " != n*n=" + std::to_string(expected));
    }

    drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = 1;
    input.n_point = static_cast<uint32_t>(expected);
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    input.data.assign(ptr, ptr + expected);

    vector_algebra::CholeskyResult result;
    {
      py::gil_scoped_release release;
      result = inverter_->Invert(input, n);
    }

    // Download GPU → numpy
    auto vec = result.AsVector();
    return vector_to_numpy_2d(std::move(vec),
                               static_cast<size_t>(n),
                               static_cast<size_t>(n));
  }

  /// Batched инверсия (CPU → GPU → CPU)
  py::array_t<std::complex<float>> invert_batch_cpu(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> matrices_flat,
      int n,
      int batch_count) {
    auto buf = matrices_flat.request();
    const size_t expected = static_cast<size_t>(batch_count) * n * n;
    if (static_cast<size_t>(buf.size) != expected) {
      throw std::invalid_argument(
          "invert_batch_cpu: matrices_flat.size=" + std::to_string(buf.size) +
          " != batch_count*n*n=" + std::to_string(expected));
    }

    drv_gpu_lib::InputData<std::vector<std::complex<float>>> input;
    input.antenna_count = static_cast<uint32_t>(batch_count);
    input.n_point = static_cast<uint32_t>(n * n);
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    input.data.assign(ptr, ptr + expected);

    vector_algebra::CholeskyResult result;
    {
      py::gil_scoped_release release;
      result = inverter_->InvertBatch(input, n);
    }

    // Download GPU → numpy (batch, n, n) с zero-copy через capsule.
    // 3D массив нельзя создать через vector_to_numpy_2d — нужны явные strides.
    // capsule держит владение вектором до уничтожения numpy-массива (Python GC).
    auto vec = result.AsVector();
    auto* out_vec = new std::vector<std::complex<float>>(std::move(vec));
    auto capsule = py::capsule(out_vec, [](void* p) {
      delete static_cast<std::vector<std::complex<float>>*>(p);
    });

    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(batch_count),
        static_cast<py::ssize_t>(n),
        static_cast<py::ssize_t>(n)};
    std::vector<py::ssize_t> strides = {
        static_cast<py::ssize_t>(n * n * sizeof(std::complex<float>)),
        static_cast<py::ssize_t>(n * sizeof(std::complex<float>)),
        static_cast<py::ssize_t>(sizeof(std::complex<float>))};

    return py::array_t<std::complex<float>>(shape, strides,
                                             out_vec->data(), capsule);
  }

  void set_symmetrize_mode(vector_algebra::SymmetrizeMode mode) {
    inverter_->SetSymmetrizeMode(mode);
  }

  vector_algebra::SymmetrizeMode get_symmetrize_mode() const {
    return inverter_->GetSymmetrizeMode();
  }

private:
  ROCmGPUContext& ctx_;
  std::unique_ptr<vector_algebra::CholeskyInverterROCm> inverter_;
};

// ════════════════════════════════════════════════════════════════════════════
// Binding registration
// ════════════════════════════════════════════════════════════════════════════

inline void register_cholesky_inverter_rocm(py::module& m) {
  // SymmetrizeMode enum
  py::enum_<vector_algebra::SymmetrizeMode>(m, "SymmetrizeMode",
      "Режим симметризации после Cholesky POTRI.\n\n"
      "  Roundtrip — Download GPU → CPU sym → Upload\n"
      "  GpuKernel — HIP kernel in-place на GPU")
      .value("Roundtrip", vector_algebra::SymmetrizeMode::Roundtrip,
             "CPU symmetrize (Download → CPU → Upload)")
      .value("GpuKernel", vector_algebra::SymmetrizeMode::GpuKernel,
             "GPU kernel in-place (hiprtc)");

  // CholeskyInverterROCm class
  py::class_<PyCholeskyInverterROCm>(m, "CholeskyInverterROCm",
      "Инверсия эрмитовой положительно определённой матрицы (Cholesky, ROCm).\n\n"
      "Использует rocSOLVER: POTRF (Cholesky) + POTRI (инверсия).\n"
      "Два режима симметризации: Roundtrip (CPU) и GpuKernel (hiprtc).\n\n"
      "Usage:\n"
      "  ctx = gpuworklib.ROCmGPUContext(0)\n"
      "  inverter = gpuworklib.CholeskyInverterROCm(ctx, gpuworklib.SymmetrizeMode.GpuKernel)\n"
      "  A_inv = inverter.invert_cpu(A.flatten(), n=341)\n")
      .def(py::init<ROCmGPUContext&, vector_algebra::SymmetrizeMode>(),
           py::arg("ctx"),
           py::arg("mode") = vector_algebra::SymmetrizeMode::GpuKernel,
           "Create CholeskyInverterROCm bound to ROCm GPU context")

      .def("invert_cpu",
           &PyCholeskyInverterROCm::invert_cpu,
           py::arg("matrix_flat"), py::arg("n"),
           "Инверсия одной матрицы n×n (CPU данные).\n\n"
           "Args:\n"
           "  matrix_flat: numpy complex64, shape (n*n,) row-major\n"
           "  n: размер матрицы\n\n"
           "Returns:\n"
           "  np.ndarray complex64, shape (n, n)")

      .def("invert_batch_cpu",
           &PyCholeskyInverterROCm::invert_batch_cpu,
           py::arg("matrices_flat"), py::arg("n"), py::arg("batch_count"),
           "Batched инверсия нескольких матриц (CPU данные).\n\n"
           "Args:\n"
           "  matrices_flat: numpy complex64, shape (batch_count*n*n,)\n"
           "  n: размер каждой матрицы\n"
           "  batch_count: количество матриц\n\n"
           "Returns:\n"
           "  np.ndarray complex64, shape (batch_count, n, n)")

      .def("set_symmetrize_mode",
           &PyCholeskyInverterROCm::set_symmetrize_mode,
           py::arg("mode"), "Изменить режим симметризации")

      .def("get_symmetrize_mode",
           &PyCholeskyInverterROCm::get_symmetrize_mode,
           "Текущий режим симметризации")

      .def("__repr__", [](const PyCholeskyInverterROCm& self) {
        auto mode = self.get_symmetrize_mode();
        std::string mode_str = mode == vector_algebra::SymmetrizeMode::Roundtrip
                                   ? "Roundtrip"
                                   : "GpuKernel";
        return "<CholeskyInverterROCm (ROCm, POTRF+POTRI, " + mode_str + ")>";
      });
}

#endif  // ENABLE_ROCM
