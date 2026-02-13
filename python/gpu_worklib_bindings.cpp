/**
 * @file gpu_worklib_bindings.cpp
 * @brief pybind11 bindings for GPUWorkLib
 *
 * Python API:
 *   ctx = gpuworklib.GPUContext(device_index=0)
 *   sig = gpuworklib.SignalGenerator(ctx)
 *   fft = gpuworklib.FFTProcessor(ctx)
 *
 *   cw = sig.generate_cw(freq=100, fs=4000, length=4096)
 *   spectrum = fft.process_complex(cw, sample_rate=4000)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-13
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <CL/cl.h>
#include <complex>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <cmath>

// GPUWorkLib headers
#include "backends/opencl/opencl_backend.hpp"
#include "interface/i_backend.hpp"
#include "fft_processor.hpp"
#include "fft_processor_types.hpp"
#include "signal_service.hpp"
#include "signal_generator_factory.hpp"
#include "generators/cw_generator.hpp"
#include "generators/lfm_generator.hpp"
#include "generators/noise_generator.hpp"
#include "generators/script_generator.hpp"
#include "params/signal_request.hpp"
#include "params/system_sampling.hpp"

namespace py = pybind11;

// ============================================================================
// Helper: zero-copy vector -> numpy via capsule
// ============================================================================

template<typename T>
py::array_t<T> vector_to_numpy(std::vector<T>&& data) {
    auto* vec = new std::vector<T>(std::move(data));
    auto capsule = py::capsule(vec, [](void* ptr) {
        delete static_cast<std::vector<T>*>(ptr);
    });
    std::vector<py::ssize_t> shape = { static_cast<py::ssize_t>(vec->size()) };
    std::vector<py::ssize_t> strides = { static_cast<py::ssize_t>(sizeof(T)) };
    return py::array_t<T>(shape, strides, vec->data(), capsule);
}

template<typename T>
py::array_t<T> vector_to_numpy_2d(std::vector<T>&& data, size_t rows, size_t cols) {
    auto* vec = new std::vector<T>(std::move(data));
    auto capsule = py::capsule(vec, [](void* ptr) {
        delete static_cast<std::vector<T>*>(ptr);
    });
    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(rows),
        static_cast<py::ssize_t>(cols)
    };
    std::vector<py::ssize_t> strides = {
        static_cast<py::ssize_t>(cols * sizeof(T)),
        static_cast<py::ssize_t>(sizeof(T))
    };
    return py::array_t<T>(shape, strides, vec->data(), capsule);
}

// ============================================================================
// GPUContext — wraps OpenCL context + backend
// ============================================================================

class GPUContext {
public:
    GPUContext(int device_index = 0) {
        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS)
            throw std::runtime_error("OpenCL: no platforms found (error " + std::to_string(err) + ")");

        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0)
            throw std::runtime_error("OpenCL: no GPU devices found");

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

        if (device_index < 0 || static_cast<cl_uint>(device_index) >= num_devices)
            throw std::out_of_range("device_index " + std::to_string(device_index) +
                                    " out of range [0, " + std::to_string(num_devices) + ")");

        device_ = devices[device_index];

        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("OpenCL: clCreateContext failed (" + std::to_string(err) + ")");

#ifdef CL_VERSION_2_0
        cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
#else
        queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
        if (err != CL_SUCCESS) {
            clReleaseContext(context_);
            throw std::runtime_error("OpenCL: clCreateCommandQueue failed (" + std::to_string(err) + ")");
        }

        // Get device name
        char name[256];
        clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        device_name_ = name;

        // Create backend
        backend_ = std::make_unique<drv_gpu_lib::OpenCLBackend>();
        backend_->InitializeFromExternalContext(context_, device_, queue_);
    }

    ~GPUContext() {
        backend_.reset();
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

    // No copy
    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    drv_gpu_lib::IBackend* backend() { return backend_.get(); }
    const std::string& device_name() const { return device_name_; }
    cl_command_queue queue() const { return queue_; }

private:
    cl_context context_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_command_queue queue_ = nullptr;
    std::string device_name_;
    std::unique_ptr<drv_gpu_lib::OpenCLBackend> backend_;
};

// ============================================================================
// PySignalGenerator — pythonic wrapper over signal_gen::SignalService
// ============================================================================

class PySignalGenerator {
public:
    explicit PySignalGenerator(GPUContext& ctx) : ctx_(ctx) {}

    // ── CW ──────────────────────────────────────────────────────────
    py::array_t<std::complex<float>> generate_cw(
        double freq, double fs, size_t length,
        double amplitude = 1.0, double phase = 0.0,
        size_t beam_count = 1, double freq_step = 0.0)
    {
        signal_gen::CwParams cw;
        cw.f0 = freq;
        cw.amplitude = amplitude;
        cw.phase = phase;
        cw.freq_step = freq_step;

        signal_gen::SystemSampling sys{ fs, length };

        if (beam_count <= 1) {
            signal_gen::SignalService service(ctx_.backend());
            auto data = service.GenerateCpu(cw, sys);
            return vector_to_numpy(std::move(data));
        } else {
            // Multi-beam: generate on GPU, read back
            signal_gen::CwGenerator gen(ctx_.backend(), cw);
            cl_mem gpu_buf = gen.GenerateToGpu(sys, beam_count);

            size_t total = beam_count * length;
            std::vector<std::complex<float>> data(total);
            clEnqueueReadBuffer(ctx_.queue(), gpu_buf, CL_TRUE, 0,
                                total * sizeof(std::complex<float>),
                                data.data(), 0, nullptr, nullptr);
            clReleaseMemObject(gpu_buf);

            return vector_to_numpy_2d(std::move(data), beam_count, length);
        }
    }

    // ── CW from string ─────────────────────────────────────────────
    py::array_t<std::complex<float>> generate_cw_from_string(
        const std::string& params_str, double fs, size_t length,
        size_t beam_count = 1)
    {
        // Format: "freq=100,amp=1.0,phase=0,freq_step=10"
        double freq = 100, amp = 1.0, phase = 0.0, freq_step = 0.0;
        parse_params(params_str, freq, amp, phase, freq_step);
        return generate_cw(freq, fs, length, amp, phase, beam_count, freq_step);
    }

    // ── LFM ─────────────────────────────────────────────────────────
    py::array_t<std::complex<float>> generate_lfm(
        double f_start, double f_end, double fs, size_t length,
        double amplitude = 1.0, size_t beam_count = 1)
    {
        signal_gen::LfmParams lfm;
        lfm.f_start = f_start;
        lfm.f_end = f_end;
        lfm.amplitude = amplitude;

        signal_gen::SystemSampling sys{ fs, length };

        if (beam_count <= 1) {
            signal_gen::SignalService service(ctx_.backend());
            auto data = service.GenerateCpu(lfm, sys);
            return vector_to_numpy(std::move(data));
        } else {
            signal_gen::LfmGenerator gen(ctx_.backend(), lfm);
            cl_mem gpu_buf = gen.GenerateToGpu(sys, beam_count);

            size_t total = beam_count * length;
            std::vector<std::complex<float>> data(total);
            clEnqueueReadBuffer(ctx_.queue(), gpu_buf, CL_TRUE, 0,
                                total * sizeof(std::complex<float>),
                                data.data(), 0, nullptr, nullptr);
            clReleaseMemObject(gpu_buf);

            return vector_to_numpy_2d(std::move(data), beam_count, length);
        }
    }

    // ── LFM from string ────────────────────────────────────────────
    py::array_t<std::complex<float>> generate_lfm_from_string(
        const std::string& params_str, double fs, size_t length,
        size_t beam_count = 1)
    {
        // Format: "f_start=100,f_end=500,amp=1.0"
        double f_start = 100, f_end = 500, amp = 1.0;
        parse_lfm_params(params_str, f_start, f_end, amp);
        return generate_lfm(f_start, f_end, fs, length, amp, beam_count);
    }

    // ── Noise ───────────────────────────────────────────────────────
    py::array_t<std::complex<float>> generate_noise(
        double fs, size_t length, double power = 1.0,
        const std::string& noise_type = "gaussian", uint64_t seed = 0)
    {
        signal_gen::NoiseParams noise;
        noise.power = power;
        noise.seed = seed;
        noise.type = (noise_type == "white")
            ? signal_gen::NoiseType::WHITE
            : signal_gen::NoiseType::GAUSSIAN;

        signal_gen::SystemSampling sys{ fs, length };
        signal_gen::SignalService service(ctx_.backend());
        auto data = service.GenerateCpu(noise, sys);
        return vector_to_numpy(std::move(data));
    }

    // ── Universal: generate from string ─────────────────────────────
    // Format: "type=cw,freq=100,amp=1.0" or "type=lfm,f_start=100,f_end=500"
    // or "type=noise,power=2.0"
    py::array_t<std::complex<float>> generate_from_string(
        const std::string& params_str, double fs, size_t length,
        size_t beam_count = 1)
    {
        // Parse type field
        std::string sig_type = "cw";
        auto tokens = split(params_str, ',');
        for (auto& tok : tokens) {
            auto kv = split(tok, '=');
            if (kv.size() == 2 && trim(kv[0]) == "type")
                sig_type = trim(kv[1]);
        }

        // Convert to lowercase
        for (auto& c : sig_type) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if (sig_type == "cw" || sig_type == "sin" || sig_type == "sinusoid") {
            return generate_cw_from_string(params_str, fs, length, beam_count);
        } else if (sig_type == "lfm" || sig_type == "chirp") {
            return generate_lfm_from_string(params_str, fs, length, beam_count);
        } else if (sig_type == "noise" || sig_type == "gaussian" || sig_type == "white") {
            double power = 1.0;
            uint64_t seed = 0;
            std::string noise_type_str = sig_type;
            for (auto& tok : tokens) {
                auto kv = split(tok, '=');
                if (kv.size() != 2) continue;
                auto key = trim(kv[0]);
                if (key == "power") power = std::stod(kv[1]);
                else if (key == "seed") seed = static_cast<uint64_t>(std::stoull(kv[1]));
                else if (key == "noise_type") noise_type_str = trim(kv[1]);
            }
            if (noise_type_str == "noise" || noise_type_str == "gaussian")
                noise_type_str = "gaussian";
            return generate_noise(fs, length, power, noise_type_str, seed);
        } else {
            throw std::invalid_argument("Unknown signal type: '" + sig_type +
                "'. Use: cw, sin, lfm, chirp, noise, gaussian, white");
        }
    }

private:
    GPUContext& ctx_;

    // Simple key=value parser
    static void parse_params(const std::string& s,
        double& freq, double& amp, double& phase, double& freq_step)
    {
        auto tokens = split(s, ',');
        for (auto& tok : tokens) {
            auto kv = split(tok, '=');
            if (kv.size() != 2) continue;
            auto key = trim(kv[0]);
            if (key == "type") continue;  // skip non-numeric
            try {
                double val = std::stod(kv[1]);
                if (key == "freq" || key == "f0") freq = val;
                else if (key == "amp" || key == "amplitude") amp = val;
                else if (key == "phase") phase = val;
                else if (key == "freq_step") freq_step = val;
            } catch (...) { /* skip non-numeric values */ }
        }
    }

    static void parse_lfm_params(const std::string& s,
        double& f_start, double& f_end, double& amp)
    {
        auto tokens = split(s, ',');
        for (auto& tok : tokens) {
            auto kv = split(tok, '=');
            if (kv.size() != 2) continue;
            auto key = trim(kv[0]);
            if (key == "type") continue;
            try {
                double val = std::stod(kv[1]);
                if (key == "f_start") f_start = val;
                else if (key == "f_end") f_end = val;
                else if (key == "amp" || key == "amplitude") amp = val;
            } catch (...) { /* skip non-numeric values */ }
        }
    }

    static std::vector<std::string> split(const std::string& s, char delim) {
        std::vector<std::string> result;
        std::string token;
        for (char c : s) {
            if (c == delim) { result.push_back(token); token.clear(); }
            else token += c;
        }
        if (!token.empty()) result.push_back(token);
        return result;
    }

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t");
        size_t end = s.find_last_not_of(" \t");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }
};

// ============================================================================
// PyFFTProcessor — pythonic wrapper over fft_processor::FFTProcessor
// ============================================================================

class PyFFTProcessor {
public:
    explicit PyFFTProcessor(GPUContext& ctx) : ctx_(ctx), fft_(ctx.backend()) {}

    // ── Complex FFT ──────────────────────────────────────────────────
    py::array_t<std::complex<float>> process_complex(
        py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data,
        float sample_rate,
        uint32_t beam_count = 0,
        uint32_t n_point = 0)
    {
        auto buf = data.request();
        auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
        size_t total_size = static_cast<size_t>(buf.size);

        // Auto-detect beam_count/n_point from shape
        if (buf.ndim == 2) {
            if (beam_count == 0) beam_count = static_cast<uint32_t>(buf.shape[0]);
            if (n_point == 0) n_point = static_cast<uint32_t>(buf.shape[1]);
        } else if (buf.ndim == 1) {
            if (beam_count == 0) beam_count = 1;
            if (n_point == 0) n_point = static_cast<uint32_t>(total_size / beam_count);
        }

        std::vector<std::complex<float>> vec(ptr, ptr + total_size);

        fft_processor::FFTProcessorParams params;
        params.beam_count = beam_count;
        params.n_point = n_point;
        params.sample_rate = sample_rate;
        params.output_mode = fft_processor::FFTOutputMode::COMPLEX;

        std::vector<fft_processor::FFTComplexResult> results;
        {
            py::gil_scoped_release release;
            results = fft_.ProcessComplex(vec, params);
        }

        if (results.size() == 1) {
            // Single beam: return 1D array
            return vector_to_numpy(std::move(results[0].spectrum));
        } else {
            // Multi-beam: return 2D array
            uint32_t nFFT = results[0].nFFT;
            std::vector<std::complex<float>> flat;
            flat.reserve(results.size() * nFFT);
            for (auto& r : results) {
                flat.insert(flat.end(), r.spectrum.begin(), r.spectrum.end());
            }
            return vector_to_numpy_2d(std::move(flat), results.size(), nFFT);
        }
    }

    // ── Magnitude + Phase FFT ────────────────────────────────────────
    py::dict process_mag_phase(
        py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data,
        float sample_rate,
        uint32_t beam_count = 0,
        uint32_t n_point = 0,
        bool include_freq = true)
    {
        auto buf = data.request();
        auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
        size_t total_size = static_cast<size_t>(buf.size);

        if (buf.ndim == 2) {
            if (beam_count == 0) beam_count = static_cast<uint32_t>(buf.shape[0]);
            if (n_point == 0) n_point = static_cast<uint32_t>(buf.shape[1]);
        } else {
            if (beam_count == 0) beam_count = 1;
            if (n_point == 0) n_point = static_cast<uint32_t>(total_size / beam_count);
        }

        std::vector<std::complex<float>> vec(ptr, ptr + total_size);

        fft_processor::FFTProcessorParams params;
        params.beam_count = beam_count;
        params.n_point = n_point;
        params.sample_rate = sample_rate;
        params.output_mode = include_freq
            ? fft_processor::FFTOutputMode::MAGNITUDE_PHASE_FREQ
            : fft_processor::FFTOutputMode::MAGNITUDE_PHASE;

        std::vector<fft_processor::FFTMagPhaseResult> results;
        {
            py::gil_scoped_release release;
            results = fft_.ProcessMagPhase(vec, params);
        }

        py::dict out;
        out["nFFT"] = results[0].nFFT;
        out["sample_rate"] = results[0].sample_rate;

        if (results.size() == 1) {
            out["magnitude"] = vector_to_numpy(std::move(results[0].magnitude));
            out["phase"] = vector_to_numpy(std::move(results[0].phase));
            if (include_freq && !results[0].frequency.empty())
                out["frequency"] = vector_to_numpy(std::move(results[0].frequency));
        } else {
            uint32_t nFFT = results[0].nFFT;
            std::vector<float> all_mag, all_phase, all_freq;
            all_mag.reserve(results.size() * nFFT);
            all_phase.reserve(results.size() * nFFT);
            if (include_freq) all_freq.reserve(results.size() * nFFT);

            for (auto& r : results) {
                all_mag.insert(all_mag.end(), r.magnitude.begin(), r.magnitude.end());
                all_phase.insert(all_phase.end(), r.phase.begin(), r.phase.end());
                if (include_freq && !r.frequency.empty())
                    all_freq.insert(all_freq.end(), r.frequency.begin(), r.frequency.end());
            }

            out["magnitude"] = vector_to_numpy_2d(std::move(all_mag), results.size(), nFFT);
            out["phase"] = vector_to_numpy_2d(std::move(all_phase), results.size(), nFFT);
            if (!all_freq.empty())
                out["frequency"] = vector_to_numpy_2d(std::move(all_freq), results.size(), nFFT);
        }

        return out;
    }

    // ── Profiling info ───────────────────────────────────────────────
    py::dict get_profiling() const {
        auto& p = fft_.GetProfilingData();
        py::dict d;
        d["upload_ms"] = p.upload_time_ms;
        d["fft_ms"] = p.fft_time_ms;
        d["post_processing_ms"] = p.post_processing_time_ms;
        d["download_ms"] = p.download_time_ms;
        d["total_ms"] = p.total_time_ms;
        return d;
    }

    uint32_t get_nfft() const { return fft_.GetNFFT(); }

private:
    GPUContext& ctx_;
    fft_processor::FFTProcessor fft_;
};

// ============================================================================
// PyScriptGenerator — text DSL -> OpenCL kernel compiler
// ============================================================================

class PyScriptGenerator {
public:
    explicit PyScriptGenerator(GPUContext& ctx) : ctx_(ctx), gen_(ctx.backend()) {}

    void load(const std::string& script_text) {
        py::gil_scoped_release release;
        gen_.LoadScript(script_text);
    }

    void load_file(const std::string& file_path) {
        py::gil_scoped_release release;
        gen_.LoadFile(file_path);
    }

    py::array_t<std::complex<float>> generate() {
        uint32_t antennas = gen_.GetAntennas();
        uint32_t points = gen_.GetPoints();

        cl_mem gpu_buf;
        {
            py::gil_scoped_release release;
            gpu_buf = gen_.Generate();
        }

        size_t total = static_cast<size_t>(antennas) * points;
        std::vector<std::complex<float>> data(total);
        clEnqueueReadBuffer(ctx_.queue(), gpu_buf, CL_TRUE, 0,
                            total * sizeof(std::complex<float>),
                            data.data(), 0, nullptr, nullptr);
        clReleaseMemObject(gpu_buf);

        if (antennas <= 1) {
            return vector_to_numpy(std::move(data));
        } else {
            return vector_to_numpy_2d(std::move(data), antennas, points);
        }
    }

    uint32_t antennas() const { return gen_.GetAntennas(); }
    uint32_t points() const { return gen_.GetPoints(); }
    std::string kernel_source() const { return gen_.GetKernelSource(); }
    bool is_ready() const { return gen_.IsReady(); }

private:
    GPUContext& ctx_;
    signal_gen::ScriptGenerator gen_;
};

// ============================================================================
// PYBIND11_MODULE — the entry point
// ============================================================================

PYBIND11_MODULE(gpuworklib, m) {
    m.doc() = "GPUWorkLib - GPU Signal Processing (OpenCL)\n\n"
              "Modules:\n"
              "  GPUContext        - GPU device management\n"
              "  SignalGenerator   - CW, LFM, Noise generation\n"
              "  ScriptGenerator   - Text DSL -> GPU kernel compiler\n"
              "  FFTProcessor      - FFT with various output modes\n";

    // ════════════════════════════════════════════════════════════════
    // GPUContext
    // ════════════════════════════════════════════════════════════════
    py::class_<GPUContext>(m, "GPUContext",
        "GPU context wrapping OpenCL device.\n\n"
        "Usage:\n"
        "  ctx = gpuworklib.GPUContext(device_index=0)\n"
        "  print(ctx.device_name)")
        .def(py::init<int>(), py::arg("device_index") = 0,
             "Create GPU context on given device index")

        .def_property_readonly("device_name", &GPUContext::device_name,
             "GPU device name (str)")

        .def("__repr__", [](const GPUContext& ctx) {
            return "<GPUContext device='" + ctx.device_name() + "'>";
        })

        .def("__enter__", [](GPUContext& self) -> GPUContext& { return self; },
             py::return_value_policy::reference)
        .def("__exit__", [](GPUContext&, py::object, py::object, py::object) {
            return false;
        });

    // ════════════════════════════════════════════════════════════════
    // SignalGenerator
    // ════════════════════════════════════════════════════════════════
    py::class_<PySignalGenerator>(m, "SignalGenerator",
        "GPU signal generator (CW, LFM, Noise).\n\n"
        "Usage:\n"
        "  sig = gpuworklib.SignalGenerator(ctx)\n"
        "  cw = sig.generate_cw(freq=100, fs=4000, length=4096)")
        .def(py::init<GPUContext&>(), py::arg("ctx"),
             "Create signal generator bound to GPU context")

        // CW
        .def("generate_cw", &PySignalGenerator::generate_cw,
             py::arg("freq"), py::arg("fs"), py::arg("length"),
             py::arg("amplitude") = 1.0, py::arg("phase") = 0.0,
             py::arg("beam_count") = 1, py::arg("freq_step") = 0.0,
             "Generate CW (sinusoidal) signal.\n\n"
             "Args:\n"
             "  freq: frequency (Hz)\n"
             "  fs: sample rate (Hz)\n"
             "  length: samples per beam\n"
             "  amplitude: signal amplitude (default 1.0)\n"
             "  phase: initial phase in radians (default 0.0)\n"
             "  beam_count: number of beams (default 1)\n"
             "  freq_step: frequency step between beams (Hz)\n\n"
             "Returns:\n"
             "  numpy.ndarray complex64: (length,) or (beam_count, length)")

        // CW from string
        .def("generate_cw_from_string", &PySignalGenerator::generate_cw_from_string,
             py::arg("params"), py::arg("fs"), py::arg("length"),
             py::arg("beam_count") = 1,
             "Generate CW from string parameters.\n"
             "Format: 'freq=100,amp=1.0,phase=0,freq_step=10'")

        // LFM
        .def("generate_lfm", &PySignalGenerator::generate_lfm,
             py::arg("f_start"), py::arg("f_end"), py::arg("fs"), py::arg("length"),
             py::arg("amplitude") = 1.0, py::arg("beam_count") = 1,
             "Generate LFM (chirp) signal.\n\n"
             "Args:\n"
             "  f_start: start frequency (Hz)\n"
             "  f_end: end frequency (Hz)\n"
             "  fs: sample rate (Hz)\n"
             "  length: samples per beam\n\n"
             "Returns:\n"
             "  numpy.ndarray complex64")

        // LFM from string
        .def("generate_lfm_from_string", &PySignalGenerator::generate_lfm_from_string,
             py::arg("params"), py::arg("fs"), py::arg("length"),
             py::arg("beam_count") = 1,
             "Generate LFM from string parameters.\n"
             "Format: 'f_start=100,f_end=500,amp=1.0'")

        // Noise
        .def("generate_noise", &PySignalGenerator::generate_noise,
             py::arg("fs"), py::arg("length"),
             py::arg("power") = 1.0, py::arg("noise_type") = "gaussian",
             py::arg("seed") = 0,
             "Generate noise signal.\n\n"
             "Args:\n"
             "  noise_type: 'gaussian' or 'white'\n"
             "  power: noise power/variance\n"
             "  seed: random seed (0 = auto)\n\n"
             "Returns:\n"
             "  numpy.ndarray complex64")

        // Universal: generate from string with signal type
        .def("generate", &PySignalGenerator::generate_from_string,
             py::arg("params"), py::arg("fs"), py::arg("length"),
             py::arg("beam_count") = 1,
             "Generate signal from string description.\n\n"
             "Format: 'type=cw,freq=100,amp=1.0,freq_step=10'\n"
             "        'type=lfm,f_start=100,f_end=500'\n"
             "        'type=noise,power=2.0'\n\n"
             "Supported types: cw, sin, lfm, chirp, noise, gaussian, white\n\n"
             "Args:\n"
             "  params: string with key=value pairs\n"
             "  fs: sample rate (Hz)\n"
             "  length: samples per beam\n"
             "  beam_count: number of beams\n\n"
             "Returns:\n"
             "  numpy.ndarray complex64");

    // ════════════════════════════════════════════════════════════════
    // FFTProcessor
    // ════════════════════════════════════════════════════════════════
    py::class_<PyFFTProcessor>(m, "FFTProcessor",
        "GPU FFT processor (clFFT backend).\n\n"
        "Usage:\n"
        "  fft = gpuworklib.FFTProcessor(ctx)\n"
        "  spectrum = fft.process_complex(signal, sample_rate=4000)")
        .def(py::init<GPUContext&>(), py::arg("ctx"),
             "Create FFT processor bound to GPU context")

        .def("process_complex", &PyFFTProcessor::process_complex,
             py::arg("data"), py::arg("sample_rate"),
             py::arg("beam_count") = 0, py::arg("n_point") = 0,
             "Compute FFT, return complex spectrum.\n\n"
             "Args:\n"
             "  data: numpy complex64 array, 1D (length,) or 2D (beams, length)\n"
             "  sample_rate: sampling rate (Hz)\n"
             "  beam_count: auto-detected from shape if 0\n"
             "  n_point: auto-detected from shape if 0\n\n"
             "Returns:\n"
             "  numpy.ndarray complex64: (nFFT,) or (beams, nFFT)")

        .def("process_mag_phase", &PyFFTProcessor::process_mag_phase,
             py::arg("data"), py::arg("sample_rate"),
             py::arg("beam_count") = 0, py::arg("n_point") = 0,
             py::arg("include_freq") = true,
             "Compute FFT, return magnitude + phase + frequency.\n\n"
             "Returns:\n"
             "  dict with keys: 'magnitude', 'phase', 'frequency', 'nFFT', 'sample_rate'")

        .def("get_profiling", &PyFFTProcessor::get_profiling,
             "Get GPU profiling data (dict)")

        .def_property_readonly("nfft", &PyFFTProcessor::get_nfft,
             "Last used nFFT value");

    // ════════════════════════════════════════════════════════════════
    // ScriptGenerator
    // ════════════════════════════════════════════════════════════════
    py::class_<PyScriptGenerator>(m, "ScriptGenerator",
        "Text DSL -> GPU kernel compiler.\n\n"
        "Define signal formulas in text that compile to OpenCL kernels.\n\n"
        "Format:\n"
        "  [Params]\n"
        "  ANTENNAS = 256\n"
        "  POINTS = 10000\n\n"
        "  [Defs]\n"
        "  var_A = 1.0 + (float)ID * 0.01\n"
        "  var_W = 0.1 + (float)ID * 0.0005\n\n"
        "  [Signal]\n"
        "  res = var_A * sin(var_W * (float)T);\n\n"
        "Built-in variables: ID (antenna index), T (sample index)")
        .def(py::init<GPUContext&>(), py::arg("ctx"),
             "Create script generator bound to GPU context")

        .def("load", &PyScriptGenerator::load,
             py::arg("script"),
             "Parse and compile script from string.\n\n"
             "Sections: [Params], [Defs], [Signal]\n"
             "Built-in variables: ID (antenna/beam index), T (sample index)")

        .def("load_file", &PyScriptGenerator::load_file,
             py::arg("file_path"),
             "Parse and compile script from file")

        .def("generate", &PyScriptGenerator::generate,
             "Execute compiled kernel and return numpy array.\n\n"
             "Returns:\n"
             "  numpy.ndarray complex64: (antennas, points) or (points,)")

        .def_property_readonly("antennas", &PyScriptGenerator::antennas,
             "Number of antennas/beams from [Params]")

        .def_property_readonly("points", &PyScriptGenerator::points,
             "Number of samples per antenna from [Params]")

        .def_property_readonly("kernel_source", &PyScriptGenerator::kernel_source,
             "Generated OpenCL kernel source (for debugging)")

        .def_property_readonly("is_ready", &PyScriptGenerator::is_ready,
             "True if script is loaded and compiled")

        .def("__repr__", [](const PyScriptGenerator& self) {
            if (self.is_ready()) {
                return "<ScriptGenerator antennas=" + std::to_string(self.antennas()) +
                       " points=" + std::to_string(self.points()) + " ready>";
            }
            return std::string("<ScriptGenerator not loaded>");
        });

    // ════════════════════════════════════════════════════════════════
    // Module-level utilities
    // ════════════════════════════════════════════════════════════════
    m.def("get_gpu_count", []() -> int {
        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) != CL_SUCCESS) return 0;
        cl_uint count = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
        return static_cast<int>(count);
    }, "Get number of available GPU devices");

    m.def("list_gpus", []() -> py::list {
        py::list result;
        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) != CL_SUCCESS) return result;
        cl_uint count = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
        if (count == 0) return result;

        std::vector<cl_device_id> devices(count);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr);

        for (cl_uint i = 0; i < count; ++i) {
            char name[256];
            clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, nullptr);
            cl_ulong mem = 0;
            clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);

            py::dict info;
            info["index"] = static_cast<int>(i);
            info["name"] = std::string(name);
            info["memory_mb"] = static_cast<int>(mem / (1024 * 1024));
            result.append(info);
        }
        return result;
    }, "List available GPU devices with info");
}
