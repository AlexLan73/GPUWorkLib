#pragma once

/**
 * @file py_lfm_analytical_delay.hpp
 * @brief Python wrapper for LfmGeneratorAnalyticalDelay
 *
 * Analytical LFM with per-antenna delay (no interpolation artifacts).
 * Used as reference for testing LchFarrow and beamforming.
 *
 * Include AFTER GPUContext and vector_to_numpy definitions.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "generators/lfm_generator_analytical_delay.hpp"

// ============================================================================
// PyLfmAnalyticalDelay — LFM with per-antenna analytical delay
// ============================================================================

// Генератор ЛЧМ сигнала с аналитической задержкой (без интерполяции).
// Задержка реализована сдвигом аргумента фазы: t_local = t - tau.
// При t < tau сигнал = 0 (сигнал ещё не пришёл на данную антенну).
// Два режима генерации:
//   generate_gpu(): float32 на GPU, быстро, результат как numpy после readback
//   generate_cpu(): double precision на CPU, медленно — эталон для сравнения с LchFarrow
// std::unique_ptr<> — LfmGeneratorAnalyticalDelay некопируем (содержит GPU буферы).
class PyLfmAnalyticalDelay {
public:
    PyLfmAnalyticalDelay(GPUContext& ctx, double f_start, double f_end,
                          double amplitude = 1.0, bool complex_iq = true)
        : ctx_(ctx)
    {
        signal_gen::LfmParams params;
        params.f_start = f_start;
        params.f_end = f_end;
        params.amplitude = amplitude;
        params.complex_iq = complex_iq;
        gen_ = std::make_unique<signal_gen::LfmGeneratorAnalyticalDelay>(
            ctx.backend(), params);
    }

    // Устанавливает частоту дискретизации и длину сигнала.
    // Вызывать ПОСЛЕ конструктора и ДО generate_gpu/generate_cpu.
    // Без этого вызова генератор выдаст 0 точек.
    void set_sampling(double fs, size_t length) {
        signal_gen::SystemSampling sys{ fs, length };
        gen_->SetSampling(sys);
    }

    // Задаёт задержки в МИКРОСЕКУНДАХ (не секундах!) для каждой антенны.
    // Число элементов delay_us = число антенн на выходе.
    // delay_us[i] = 0 → антенна без задержки (сигнал с отсчёта 0).
    // При t < tau_i → сигнал антенны i равен 0 (сигнал ещё не пришёл).
    void set_delays(const std::vector<float>& delay_us) {
        gen_->SetDelays(delay_us);
    }

    // Генерирует сигнал на GPU (float32) и возвращает результат как numpy.
    // GIL освобождается на время GPU-операции — Python-интерпретатор не блокируется.
    // Возвращает: 1D [N] для одной антенны, 2D [A×N] для нескольких (row-major).
    // result.data (cl_mem) освобождается внутри — caller НЕ должен вызывать clRelease.
    py::array_t<std::complex<float>> generate_gpu() {
        drv_gpu_lib::InputData<cl_mem> result;
        {
            py::gil_scoped_release release;  // GPU блокирует поток — отпускаем GIL, чтобы другие Python-потоки не зависали
            result = gen_->GenerateToGpu();
        }

        uint32_t antennas = result.antenna_count;
        uint32_t points = result.n_point;
        size_t total = static_cast<size_t>(antennas) * points;
        // Readback: result.data — cl_mem с данными на GPU, освобождаем после чтения.
        std::vector<std::complex<float>> data(total);
        clEnqueueReadBuffer(ctx_.queue(), result.data, CL_TRUE, 0,
                            total * sizeof(std::complex<float>),
                            data.data(), 0, nullptr, nullptr);
        clReleaseMemObject(result.data);

        // 1D для одной антенны удобнее в matplotlib/numpy чем shape (1, N)
        if (antennas <= 1) {
            return vector_to_numpy(std::move(data));
        }
        return vector_to_numpy_2d(std::move(data), antennas, points);
    }

    // Генерирует сигнал на CPU с double precision — эталон для тестов.
    // Медленнее GPU-версии, но точнее: нет ошибок float32, нет GPU roundtrip.
    // Используется в Python-тестах для сравнения с LchFarrow и FormSignalGenerator.
    py::array_t<std::complex<float>> generate_cpu() {
        std::vector<std::vector<std::complex<float>>> cpu_data;
        {
            py::gil_scoped_release release;  // CPU-генерация тоже может быть долгой при большом числе точек
            cpu_data = gen_->GenerateToCpu();
        }

        uint32_t antennas = static_cast<uint32_t>(cpu_data.size());
        if (antennas == 0) return py::array_t<std::complex<float>>();

        uint32_t points = static_cast<uint32_t>(cpu_data[0].size());

        if (antennas == 1) {
            return vector_to_numpy(std::move(cpu_data[0]));
        }

        // GenerateToCpu возвращает vec<vec<>> — объединяем в плоский буфер,
        // потому что numpy требует contiguous memory, а vector<vector> — нет.
        std::vector<std::complex<float>> flat;
        flat.reserve(static_cast<size_t>(antennas) * points);
        for (auto& ch : cpu_data) {
            flat.insert(flat.end(), ch.begin(), ch.end());
        }
        return vector_to_numpy_2d(std::move(flat), antennas, points);
    }

    // Число антенн = число задержек, заданных через set_delays().
    // Read-only: менять антенны можно только через новый вызов set_delays().
    uint32_t antennas() const { return gen_->GetAntennas(); }

    // Возвращает текущие задержки в микросекундах как Python list[float].
    // py::list, а не numpy — задержки это параметры конфигурации, не сигнальные данные.
    py::list get_delays() const {
        py::list result;
        for (float d : gen_->GetDelays()) result.append(d);
        return result;
    }

private:
    GPUContext& ctx_;  ///< Ссылка на контекст — не владеет, контекст должен жить дольше генератора
    std::unique_ptr<signal_gen::LfmGeneratorAnalyticalDelay> gen_;  ///< Некопируем: содержит cl_mem буферы на GPU
};

// ============================================================================
// Binding registration
// ============================================================================

// Регистрирует Python-класс LfmAnalyticalDelay в модуле gpuworklib.
// Вызывается из gpu_worklib_bindings.cpp при импорте библиотеки.
// Docstring-и методов — единственная документация доступная из Python (help()).
inline void register_lfm_analytical_delay(py::module& m) {
    py::class_<PyLfmAnalyticalDelay>(m, "LfmAnalyticalDelay",
        "LFM (chirp) generator with per-antenna analytical delay.\n\n"
        "Computes S(t) = A * exp(j * phase) where:\n"
        "  t_local = t - tau (delay in seconds)\n"
        "  phase = pi * chirp_rate * t_local^2 + 2*pi * f_start * t_local\n"
        "  S = 0 when t < tau (signal not arrived yet)\n\n"
        "Usage:\n"
        "  gen = gpuworklib.LfmAnalyticalDelay(ctx, f_start=1e6, f_end=2e6)\n"
        "  gen.set_sampling(fs=12e6, length=4096)\n"
        "  gen.set_delays([0.0, 0.1, 0.2, 0.5])\n"
        "  data = gen.generate_gpu()\n")
        .def(py::init<GPUContext&, double, double, double, bool>(),
             py::arg("ctx"), py::arg("f_start"), py::arg("f_end"),
             py::arg("amplitude") = 1.0, py::arg("complex_iq") = true,
             "Create LFM analytical delay generator.\n\n"
             "Args:\n"
             "  ctx: GPU context\n"
             "  f_start: start frequency (Hz)\n"
             "  f_end: end frequency (Hz)\n"
             "  amplitude: signal amplitude (default 1.0)\n"
             "  complex_iq: True for complex IQ, False for real-only")

        .def("set_sampling", &PyLfmAnalyticalDelay::set_sampling,
             py::arg("fs"), py::arg("length"),
             "Set sampling parameters (fs, length).")

        .def("set_delays", &PyLfmAnalyticalDelay::set_delays,
             py::arg("delay_us"),
             "Set per-antenna delays in microseconds.")

        .def("generate_gpu", &PyLfmAnalyticalDelay::generate_gpu,
             "Generate on GPU, return numpy array.")

        .def("generate_cpu", &PyLfmAnalyticalDelay::generate_cpu,
             "Generate on CPU (double precision reference).")

        .def_property_readonly("antennas", &PyLfmAnalyticalDelay::antennas)
        .def_property_readonly("delays", &PyLfmAnalyticalDelay::get_delays)

        .def("__repr__", [](const PyLfmAnalyticalDelay& self) {
            return "<LfmAnalyticalDelay antennas=" +
                   std::to_string(self.antennas()) + ">";
        });
}
