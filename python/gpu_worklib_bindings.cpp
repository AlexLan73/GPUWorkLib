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
#include "generators/form_signal_generator.hpp"
#include "generators/form_script_generator.hpp"
#include "generators/delayed_form_signal_generator.hpp"
#include "params/signal_request.hpp"
#include "params/system_sampling.hpp"
#include "params/form_params.hpp"
#include "spectrum_maxima_finder.h"

// ============================================================================
// ROCm headers (Linux + AMD GPU only)
// ============================================================================

#if ENABLE_ROCM
#include "backends/rocm/rocm_backend.hpp"
#include "backends/hybrid/hybrid_backend.hpp"
#endif

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
// ROCmGPUContext — wraps ROCm backend (Linux + AMD GPU only)
// ============================================================================

#if ENABLE_ROCM

class ROCmGPUContext {
public:
  explicit ROCmGPUContext(int device_index = 0)
      : backend_(std::make_unique<drv_gpu_lib::ROCmBackend>()) {
    backend_->Initialize(device_index);
  }

  ~ROCmGPUContext() = default;

  // No copy
  ROCmGPUContext(const ROCmGPUContext&) = delete;
  ROCmGPUContext& operator=(const ROCmGPUContext&) = delete;

  drv_gpu_lib::IBackend* backend() { return backend_.get(); }
  std::string device_name() const { return backend_->GetDeviceName(); }
  int device_index() const { return backend_->GetDeviceIndex(); }

private:
  std::unique_ptr<drv_gpu_lib::ROCmBackend> backend_;
};

// ROCm Python wrappers (include AFTER ROCmGPUContext is defined)
#include "py_filters_rocm.hpp"
#include "py_lch_farrow_rocm.hpp"
#include "py_heterodyne_rocm.hpp"
#include "py_statistics.hpp"
#include "py_vector_algebra_rocm.hpp"

// ============================================================================
// HybridGPUContext — wraps HybridBackend (OpenCL + ROCm on one GPU)
// ============================================================================

class HybridGPUContext {
public:
  explicit HybridGPUContext(int device_index = 0)
      : backend_(std::make_unique<drv_gpu_lib::HybridBackend>()) {
    backend_->Initialize(device_index);
  }

  ~HybridGPUContext() = default;

  HybridGPUContext(const HybridGPUContext&) = delete;
  HybridGPUContext& operator=(const HybridGPUContext&) = delete;

  drv_gpu_lib::HybridBackend* backend() { return backend_.get(); }

  std::string opencl_device_name() const {
    auto* ocl = backend_->GetOpenCL();
    if (ocl && ocl->IsInitialized()) return ocl->GetDeviceName();
    return "Unknown";
  }

  std::string rocm_device_name() const {
    auto* rocm = backend_->GetROCm();
    if (rocm && rocm->IsInitialized()) return rocm->GetDeviceName();
    return "Unknown";
  }

  std::string device_name() const { return backend_->GetDeviceName(); }
  int device_index() const { return backend_->GetDeviceIndex(); }

  std::string zero_copy_method() const {
    auto method = backend_->GetBestZeroCopyMethod();
    return drv_gpu_lib::ZeroCopyMethodToString(method);
  }

  bool is_zero_copy_supported() const {
    return backend_->GetBestZeroCopyMethod() != drv_gpu_lib::ZeroCopyMethod::NONE;
  }

private:
  std::unique_ptr<drv_gpu_lib::HybridBackend> backend_;
};

#endif  // ENABLE_ROCM

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
        auto p = fft_.GetProfilingData();
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
// PyGPUBuffer — GPU buffer handle for output="gpu"
// ============================================================================

class PyGPUBuffer {
public:
    PyGPUBuffer(cl_mem mem, cl_command_queue queue,
                uint32_t antenna_count, uint32_t n_point)
        : mem_(mem)
        , antenna_count_(antenna_count)
        , n_point_(n_point)
    {
        if (queue) {
            clRetainCommandQueue(queue);
            queue_ = queue;
        } else {
            queue_ = nullptr;
        }
    }

    ~PyGPUBuffer() {
        if (mem_) clReleaseMemObject(mem_);
        mem_ = nullptr;
        if (queue_) clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }

    PyGPUBuffer(const PyGPUBuffer&) = delete;
    PyGPUBuffer& operator=(const PyGPUBuffer&) = delete;

    PyGPUBuffer(PyGPUBuffer&& other) noexcept
        : mem_(other.mem_)
        , queue_(other.queue_)
        , antenna_count_(other.antenna_count_)
        , n_point_(other.n_point_)
    {
        other.mem_ = nullptr;
        other.queue_ = nullptr;
    }

    PyGPUBuffer& operator=(PyGPUBuffer&& other) noexcept {
        if (this != &other) {
            if (mem_) clReleaseMemObject(mem_);
            if (queue_) clReleaseCommandQueue(queue_);
            mem_ = other.mem_;
            queue_ = other.queue_;
            antenna_count_ = other.antenna_count_;
            n_point_ = other.n_point_;
            other.mem_ = nullptr;
            other.queue_ = nullptr;
        }
        return *this;
    }

    py::array_t<std::complex<float>> read() {
        if (!mem_ || !queue_)
            throw std::runtime_error("PyGPUBuffer::read: buffer already released");
        size_t total = static_cast<size_t>(antenna_count_) * n_point_;
        std::vector<std::complex<float>> data(total);
        cl_int err = clEnqueueReadBuffer(queue_, mem_, CL_TRUE, 0,
                total * sizeof(std::complex<float>),
                data.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS)
            throw std::runtime_error("PyGPUBuffer::read: clEnqueueReadBuffer failed");
        if (antenna_count_ <= 1)
            return vector_to_numpy(std::move(data));
        return vector_to_numpy_2d(std::move(data), antenna_count_, n_point_);
    }

    void release() {
        if (mem_) {
            clReleaseMemObject(mem_);
            mem_ = nullptr;
        }
    }

    uint32_t antenna_count() const { return antenna_count_; }
    uint32_t n_point() const { return n_point_; }

    py::tuple shape() const {
        if (antenna_count_ <= 1)
            return py::make_tuple(static_cast<py::ssize_t>(n_point_));
        return py::make_tuple(
            static_cast<py::ssize_t>(antenna_count_),
            static_cast<py::ssize_t>(n_point_));
    }

    bool is_valid() const { return mem_ != nullptr; }

private:
    cl_mem mem_ = nullptr;
    cl_command_queue queue_ = nullptr;
    uint32_t antenna_count_ = 0;
    uint32_t n_point_ = 0;
};

// ============================================================================
// PyFormSignalGenerator — multi-channel getX formula on GPU
// ============================================================================

class PyFormSignalGenerator {
public:
    explicit PyFormSignalGenerator(GPUContext& ctx)
        : ctx_(ctx), gen_(ctx.backend()) {}

    /// Set parameters from FormParams fields
    void set_params(double fs, uint32_t antennas, uint32_t points,
                    double f0, double amplitude, double noise_amplitude,
                    double phase, double fdev, double norm,
                    double tau_base, double tau_step,
                    double tau_min, double tau_max,
                    uint32_t tau_seed, uint32_t noise_seed) {
        signal_gen::FormParams p;
        p.fs = fs;
        p.antennas = antennas;
        p.points = points;
        p.f0 = f0;
        p.amplitude = amplitude;
        p.noise_amplitude = noise_amplitude;
        p.phase = phase;
        p.fdev = fdev;
        p.norm = norm;
        p.tau_base = tau_base;
        p.tau_step = tau_step;
        p.tau_min = tau_min;
        p.tau_max = tau_max;
        p.tau_seed = tau_seed;
        p.noise_seed = noise_seed;
        gen_.SetParams(p);
    }

    /// Set parameters from string "f0=1e6,a=1.0,an=0.1,tau=0.001"
    void set_params_from_string(const std::string& params_str) {
        gen_.SetParamsFromString(params_str);
    }

    /// Generate signal: output="cpu" (default) → numpy, output="gpu" → GPUBuffer
    py::object generate(const std::string& output = "cpu") {
        drv_gpu_lib::InputData<cl_mem> input;
        {
            py::gil_scoped_release release;
            input = gen_.GenerateInputData();
        }

        if (output == "gpu") {
            PyGPUBuffer buf(input.data, ctx_.queue(),
                            input.antenna_count, input.n_point);
            return py::cast(std::move(buf));
        }

        uint32_t antennas = input.antenna_count;
        uint32_t points = input.n_point;
        size_t total = static_cast<size_t>(antennas) * points;
        std::vector<std::complex<float>> data(total);
        clEnqueueReadBuffer(ctx_.queue(), input.data, CL_TRUE, 0,
                            total * sizeof(std::complex<float>),
                            data.data(), 0, nullptr, nullptr);
        clReleaseMemObject(input.data);

        if (antennas <= 1) {
            return vector_to_numpy(std::move(data));
        } else {
            return vector_to_numpy_2d(std::move(data), antennas, points);
        }
    }

    uint32_t antennas() const { return gen_.GetAntennas(); }
    uint32_t points() const { return gen_.GetPoints(); }
    double fs() const { return gen_.GetParams().fs; }

private:
    GPUContext& ctx_;
    signal_gen::FormSignalGenerator gen_;
};

// ============================================================================
// PyFormScriptGenerator — DSL + on-disk kernel cache for getX
// ============================================================================

class PyFormScriptGenerator {
public:
    explicit PyFormScriptGenerator(GPUContext& ctx)
        : ctx_(ctx), gen_(ctx.backend()) {}

    /// Set parameters (same as FormSignalGenerator)
    void set_params(double fs, uint32_t antennas, uint32_t points,
                    double f0, double amplitude, double noise_amplitude,
                    double phase, double fdev, double norm,
                    double tau_base, double tau_step,
                    double tau_min, double tau_max,
                    uint32_t tau_seed, uint32_t noise_seed) {
        signal_gen::FormParams p;
        p.fs = fs;
        p.antennas = antennas;
        p.points = points;
        p.f0 = f0;
        p.amplitude = amplitude;
        p.noise_amplitude = noise_amplitude;
        p.phase = phase;
        p.fdev = fdev;
        p.norm = norm;
        p.tau_base = tau_base;
        p.tau_step = tau_step;
        p.tau_min = tau_min;
        p.tau_max = tau_max;
        p.tau_seed = tau_seed;
        p.noise_seed = noise_seed;
        gen_.SetParams(p);
    }

    void set_params_from_string(const std::string& params_str) {
        gen_.SetParamsFromString(params_str);
    }

    /// Compile kernel from current params
    void compile() {
        py::gil_scoped_release release;
        gen_.Compile();
    }

    /// Generate DSL script text
    std::string generate_script() const {
        return gen_.GenerateScript();
    }

    /// Generate OpenCL kernel source
    std::string generate_kernel_source() const {
        return gen_.GenerateKernelSource();
    }

    /// Save compiled kernel to disk
    void save_kernel(const std::string& name, const std::string& comment = "") {
        gen_.SaveKernel(name, comment);
    }

    /// Load kernel from disk by name
    void load_kernel(const std::string& name) {
        gen_.LoadKernel(name);
    }

    /// List available kernel names from manifest
    std::vector<std::string> list_kernels() const {
        return gen_.ListKernels();
    }

    /// Generate signal: output="cpu" (default) → numpy, output="gpu" → GPUBuffer
    py::object generate(const std::string& output = "cpu") {
        drv_gpu_lib::InputData<cl_mem> input;
        {
            py::gil_scoped_release release;
            input = gen_.GenerateInputData();
        }

        if (output == "gpu") {
            PyGPUBuffer buf(input.data, ctx_.queue(),
                            input.antenna_count, input.n_point);
            return py::cast(std::move(buf));
        }

        uint32_t antennas = input.antenna_count;
        uint32_t points = input.n_point;
        size_t total = static_cast<size_t>(antennas) * points;
        std::vector<std::complex<float>> data(total);
        clEnqueueReadBuffer(ctx_.queue(), input.data, CL_TRUE, 0,
                            total * sizeof(std::complex<float>),
                            data.data(), 0, nullptr, nullptr);
        clReleaseMemObject(input.data);

        if (antennas <= 1) {
            return vector_to_numpy(std::move(data));
        } else {
            return vector_to_numpy_2d(std::move(data), antennas, points);
        }
    }

    uint32_t antennas() const { return gen_.GetAntennas(); }
    uint32_t points() const { return gen_.GetPoints(); }
    double fs() const { return gen_.GetParams().fs; }
    bool is_ready() const { return gen_.IsReady(); }
    std::string kernel_source() const { return gen_.GetCurrentKernelSource(); }

    static std::string kernels_dir() {
        return signal_gen::FormScriptGenerator::GetKernelsDir();
    }
    static std::string kernels_bin_dir() {
        return signal_gen::FormScriptGenerator::GetKernelsBinDir();
    }

private:
    GPUContext& ctx_;
    signal_gen::FormScriptGenerator gen_;
};

// ============================================================================
// PyDelayedFormSignalGenerator — Farrow 48×5 fractional delay on GPU
// ============================================================================

class PyDelayedFormSignalGenerator {
public:
    explicit PyDelayedFormSignalGenerator(GPUContext& ctx)
        : ctx_(ctx), gen_(ctx.backend()) {}

    /// Set signal parameters (same as FormSignalGenerator)
    void set_params(double fs, uint32_t antennas, uint32_t points,
                    double f0, double amplitude, double noise_amplitude,
                    double phase, double fdev, double norm,
                    uint32_t noise_seed) {
        signal_gen::FormParams p;
        p.fs = fs;
        p.antennas = antennas;
        p.points = points;
        p.f0 = f0;
        p.amplitude = amplitude;
        p.noise_amplitude = noise_amplitude;
        p.phase = phase;
        p.fdev = fdev;
        p.norm = norm;
        p.noise_seed = noise_seed;
        gen_.SetParams(p);
    }

    /// Set per-antenna delays in microseconds
    void set_delays(const std::vector<float>& delay_us) {
        gen_.SetDelays(delay_us);
    }

    /// Load Lagrange matrix from JSON file (optional, built-in used by default)
    void load_matrix(const std::string& json_path) {
        gen_.LoadMatrix(json_path);
    }

    /// Generate signal: output="cpu" (default) → numpy, output="gpu" → GPUBuffer
    py::object generate(const std::string& output = "cpu") {
        drv_gpu_lib::InputData<cl_mem> input;
        {
            py::gil_scoped_release release;
            input = gen_.GenerateInputData();
        }

        if (output == "gpu") {
            PyGPUBuffer buf(input.data, ctx_.queue(),
                            input.antenna_count, input.n_point);
            return py::cast(std::move(buf));
        }

        uint32_t antennas = input.antenna_count;
        uint32_t points = input.n_point;
        size_t total = static_cast<size_t>(antennas) * points;
        std::vector<std::complex<float>> data(total);
        clEnqueueReadBuffer(ctx_.queue(), input.data, CL_TRUE, 0,
                            total * sizeof(std::complex<float>),
                            data.data(), 0, nullptr, nullptr);
        clReleaseMemObject(input.data);

        if (antennas <= 1) {
            return vector_to_numpy(std::move(data));
        } else {
            return vector_to_numpy_2d(std::move(data), antennas, points);
        }
    }

    uint32_t antennas() const { return gen_.GetAntennas(); }
    uint32_t points() const { return gen_.GetPoints(); }
    double fs() const { return gen_.GetParams().fs; }
    py::list get_delays() const {
        py::list result;
        for (float d : gen_.GetDelays()) result.append(d);
        return result;
    }

private:
    GPUContext& ctx_;
    signal_gen::DelayedFormSignalGenerator gen_;
};

// ============================================================================
// New module wrappers — separate files (one class per file)
// ============================================================================

#include "py_lfm_analytical_delay.hpp"
#include "py_lch_farrow.hpp"
#include "py_filters.hpp"
#include "py_heterodyne.hpp"

// ============================================================================
// PySpectrumMaximaFinder — find all local maxima in FFT spectrum
// ============================================================================

class PySpectrumMaximaFinder {
public:
    explicit PySpectrumMaximaFinder(GPUContext& ctx) : ctx_(ctx), finder_(ctx.backend()) {}

    /**
     * find_all_maxima(fft_data, sample_rate, ...)
     *
     * Input: numpy complex64 array (FFT result) — 1D (nFFT,) or 2D (beams, nFFT)
     * Returns: dict (1 beam) or list[dict] (multi-beam)
     *   Each dict: {"positions": np.uint32, "magnitudes": np.float32,
     *               "frequencies": np.float32, "num_maxima": int}
     */
    py::object find_all_maxima(
        py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> fft_data,
        float sample_rate,
        uint32_t beam_count = 0,
        uint32_t nFFT = 0,
        uint32_t search_start = 0,
        uint32_t search_end = 0)
    {
        auto buf = fft_data.request();
        auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
        size_t total_size = static_cast<size_t>(buf.size);

        // Auto-detect beam_count/nFFT from shape
        if (buf.ndim == 2) {
            if (beam_count == 0) beam_count = static_cast<uint32_t>(buf.shape[0]);
            if (nFFT == 0) nFFT = static_cast<uint32_t>(buf.shape[1]);
        } else if (buf.ndim == 1) {
            if (beam_count == 0) beam_count = 1;
            if (nFFT == 0) nFFT = static_cast<uint32_t>(total_size / beam_count);
        }

        if (nFFT == 0 || beam_count == 0)
            throw std::invalid_argument("Cannot determine nFFT/beam_count from input shape");

        // Upload to GPU
        cl_context cl_ctx = static_cast<cl_context>(ctx_.backend()->GetNativeContext());
        cl_int err;
        cl_mem gpu_fft = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            total_size * sizeof(std::complex<float>),
            const_cast<std::complex<float>*>(ptr), &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("Failed to upload FFT data to GPU (error " +
                                     std::to_string(err) + ")");

        antenna_fft::AllMaximaResult result;
        {
            py::gil_scoped_release release;
            result = finder_.FindAllMaxima(gpu_fft, beam_count, nFFT, sample_rate,
                antenna_fft::OutputDestination::CPU, search_start, search_end);
        }

        clReleaseMemObject(gpu_fft);

        // Convert to Python (extract from MaxValue for backward compatibility)
        if (beam_count == 1 && result.beams.size() == 1) {
            // Single beam: return dict
            auto& b = result.beams[0];
            py::dict out;
            std::vector<uint32_t> positions; positions.reserve(b.maxima.size());
            std::vector<float> magnitudes; magnitudes.reserve(b.maxima.size());
            std::vector<float> frequencies; frequencies.reserve(b.maxima.size());
            for (const auto& m : b.maxima) {
                positions.push_back(m.index);
                magnitudes.push_back(m.magnitude);
                frequencies.push_back(m.refined_frequency);
            }
            out["positions"] = vector_to_numpy(std::move(positions));
            out["magnitudes"] = vector_to_numpy(std::move(magnitudes));
            out["frequencies"] = vector_to_numpy(std::move(frequencies));
            out["num_maxima"] = b.num_maxima;
            return out;
        } else {
            // Multi-beam: return list of dicts
            py::list beams;
            for (auto& b : result.beams) {
                py::dict out;
                std::vector<uint32_t> positions; positions.reserve(b.maxima.size());
                std::vector<float> magnitudes; magnitudes.reserve(b.maxima.size());
                std::vector<float> frequencies; frequencies.reserve(b.maxima.size());
                for (const auto& m : b.maxima) {
                    positions.push_back(m.index);
                    magnitudes.push_back(m.magnitude);
                    frequencies.push_back(m.refined_frequency);
                }
                out["antenna_id"] = b.antenna_id;
                out["positions"] = vector_to_numpy(std::move(positions));
                out["magnitudes"] = vector_to_numpy(std::move(magnitudes));
                out["frequencies"] = vector_to_numpy(std::move(frequencies));
                out["num_maxima"] = b.num_maxima;
                beams.append(out);
            }
            return beams;
        }
    }

private:
    GPUContext& ctx_;
    antenna_fft::SpectrumMaximaFinder finder_;
};

// ============================================================================
// PYBIND11_MODULE — the entry point
// ============================================================================

PYBIND11_MODULE(gpuworklib, m) {
    m.doc() = "GPUWorkLib - GPU Signal Processing (OpenCL + ROCm)\n\n"
              "OpenCL classes:\n"
              "  GPUContext              - GPU device management (OpenCL)\n"
              "  SignalGenerator         - CW, LFM, Noise generation\n"
              "  ScriptGenerator         - Text DSL -> GPU kernel compiler\n"
              "  FormSignalGenerator     - Multi-channel getX formula (signal+noise+delay)\n"
              "  DelayedFormSignalGenerator - Farrow 48x5 fractional delay on GPU\n"
              "  LfmAnalyticalDelay      - LFM with per-antenna analytical delay\n"
              "  LchFarrow               - Standalone Lagrange fractional delay processor\n"
              "  FFTProcessor            - FFT with various output modes\n"
              "  SpectrumMaximaFinder    - Find all local maxima in FFT spectrum\n"
              "  FirFilter               - GPU FIR convolution filter\n"
              "  IirFilter               - GPU IIR biquad cascade filter\n"
              "  HeterodyneDechirp       - LFM dechirp pipeline\n\n"
              "ROCm classes (Linux + AMD GPU, ENABLE_ROCM=1):\n"
              "  ROCmGPUContext          - GPU device management (ROCm/HIP)\n"
              "  FirFilterROCm           - GPU FIR filter (ROCm)\n"
              "  IirFilterROCm           - GPU IIR biquad cascade (ROCm)\n"
              "  LchFarrowROCm           - Lagrange fractional delay (ROCm)\n"
              "  HeterodyneROCm          - LFM dechirp + correct (ROCm)\n"
              "  StatisticsProcessor     - mean/median/variance/std (ROCm)\n";

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
    // GPUBuffer (for output="gpu" from FormSignalGenerator / FormScriptGenerator)
    // ════════════════════════════════════════════════════════════════
    py::class_<PyGPUBuffer>(m, "GPUBuffer",
        "GPU buffer handle returned by generate(output='gpu').\n\n"
        "Use read() to copy data to numpy array. Buffer is automatically\n"
        "released when the object is garbage-collected.\n\n"
        "Usage:\n"
        "  buf = gen.generate(output='gpu')\n"
        "  data = buf.read()  # numpy array\n"
        "  print(buf.shape)   # (antennas, points)")
        .def("read", &PyGPUBuffer::read,
             "Copy GPU buffer to numpy array (complex64).\n\n"
             "Returns:\n"
             "  numpy.ndarray: (points,) or (antennas, points)")
        .def("release", &PyGPUBuffer::release,
             "Explicitly release GPU buffer (optional, auto-released on GC)")
        .def_property_readonly("shape", &PyGPUBuffer::shape,
             "Shape: (points,) or (antennas, points)")
        .def_property_readonly("antenna_count", &PyGPUBuffer::antenna_count)
        .def_property_readonly("n_point", &PyGPUBuffer::n_point)
        .def("is_valid", &PyGPUBuffer::is_valid,
             "True if buffer not yet released")
        .def("__repr__", [](const PyGPUBuffer& self) {
            return "<GPUBuffer antennas=" + std::to_string(self.antenna_count()) +
                   " points=" + std::to_string(self.n_point()) + ">";
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
    // FormSignalGenerator
    // ════════════════════════════════════════════════════════════════
    py::class_<PyFormSignalGenerator>(m, "FormSignalGenerator",
        "Multi-channel signal generator using getX formula (GPU).\n\n"
        "Formula:\n"
        "  X = a*norm*exp(j*(2pi*f0*t + pi*fdev/ti*((t-ti/2)^2) + phi))\n"
        "    + an*norm*(randn + j*randn)\n"
        "  X = 0  if t < 0 or t > ti - dt\n\n"
        "Features:\n"
        "  - Multi-channel (antennas) parallel generation\n"
        "  - Per-channel delay: FIXED / LINEAR / RANDOM\n"
        "  - Chirp support (fdev != 0)\n"
        "  - Philox-2x32 PRNG + Box-Muller noise\n\n"
        "Usage:\n"
        "  gen = gpuworklib.FormSignalGenerator(ctx)\n"
        "  gen.set_params_from_string('f0=1e6,a=1.0,an=0.1,antennas=8,points=4096')\n"
        "  data = gen.generate()  # numpy (8, 4096) complex64\n")
        .def(py::init<GPUContext&>(), py::arg("ctx"),
             "Create FormSignalGenerator bound to GPU context")

        .def("set_params", &PyFormSignalGenerator::set_params,
             py::arg("fs") = 12e6,
             py::arg("antennas") = 1,
             py::arg("points") = 4096,
             py::arg("f0") = 0.0,
             py::arg("amplitude") = 1.0,
             py::arg("noise_amplitude") = 0.0,
             py::arg("phase") = 0.0,
             py::arg("fdev") = 0.0,
             py::arg("norm") = 0.7071067811865476,
             py::arg("tau_base") = 0.0,
             py::arg("tau_step") = 0.0,
             py::arg("tau_min") = 0.0,
             py::arg("tau_max") = 0.0,
             py::arg("tau_seed") = 12345,
             py::arg("noise_seed") = 0,
             "Set signal parameters.\n\n"
             "Args:\n"
             "  fs: sample rate (Hz, default 12 MHz)\n"
             "  antennas: number of channels (default 1)\n"
             "  points: samples per channel (default 4096)\n"
             "  f0: center frequency (Hz)\n"
             "  amplitude: signal amplitude 'a' (default 1.0)\n"
             "  noise_amplitude: noise amplitude 'an' (default 0, no noise)\n"
             "  phase: initial phase in radians (default 0)\n"
             "  fdev: frequency deviation for chirp (default 0, pure CW)\n"
             "  norm: normalization factor (default 1/sqrt(2))\n"
             "  tau_base: base delay in seconds (default 0)\n"
             "  tau_step: delay step per channel (0 = FIXED mode)\n"
             "  tau_min: min delay for RANDOM mode\n"
             "  tau_max: max delay for RANDOM mode\n"
             "  tau_seed: seed for random delay (default 12345)\n"
             "  noise_seed: seed for noise PRNG (0 = auto)\n")

        .def("set_params_from_string", &PyFormSignalGenerator::set_params_from_string,
             py::arg("params"),
             "Set parameters from string.\n\n"
             "Format: 'f0=1e6,a=1.0,an=0.1,tau=0.001,antennas=8,points=4096'\n\n"
             "Keys: fs, f0, a, an, phi, fdev, norm, tau, tau_step,\n"
             "      tau_min, tau_max, tau_seed, noise_seed, antennas, points")

        .def("generate", &PyFormSignalGenerator::generate,
             py::arg("output") = "cpu",
             "Generate signal on GPU.\n\n"
             "Args:\n"
             "  output: 'cpu' (default) — return numpy array\n"
             "          'gpu' — return GPUBuffer (use .read() to get numpy)\n\n"
             "Returns:\n"
             "  output='cpu': numpy.ndarray complex64 (points,) or (antennas, points)\n"
             "  output='gpu': GPUBuffer with .read(), .shape")

        .def_property_readonly("antennas", &PyFormSignalGenerator::antennas,
             "Number of antennas/channels")

        .def_property_readonly("points", &PyFormSignalGenerator::points,
             "Samples per channel")

        .def_property_readonly("fs", &PyFormSignalGenerator::fs,
             "Sample rate (Hz)")

        .def("__repr__", [](const PyFormSignalGenerator& self) {
            return "<FormSignalGenerator antennas=" + std::to_string(self.antennas()) +
                   " points=" + std::to_string(self.points()) +
                   " fs=" + std::to_string(static_cast<int>(self.fs())) + ">";
        });

    // ════════════════════════════════════════════════════════════════
    // FormScriptGenerator (DSL + on-disk kernel cache)
    // ════════════════════════════════════════════════════════════════
    py::class_<PyFormScriptGenerator>(m, "FormScriptGenerator",
        "DSL-based signal generator with on-disk kernel cache.\n\n"
        "Extends FormSignalGenerator with:\n"
        "  - DSL text generation (human-readable script)\n"
        "  - OpenCL kernel source generation (with #define params)\n"
        "  - Save/load compiled kernels to disk\n"
        "  - Versioning: name collisions create _00, _01, ...\n"
        "  - manifest.json with kernel metadata\n\n"
        "Two modes:\n"
        "  1. From params: set_params() -> compile() -> generate()\n"
        "  2. From cache:  load_kernel('name') -> generate()\n\n"
        "Usage:\n"
        "  gen = gpuworklib.FormScriptGenerator(ctx)\n"
        "  gen.set_params(f0=1e6, antennas=8, points=4096)\n"
        "  gen.compile()\n"
        "  data = gen.generate()  # numpy (8, 4096) complex64\n"
        "  gen.save_kernel('my_signal', 'CW 1MHz 8ch')\n"
        "  gen.load_kernel('my_signal')  # fast binary load\n")
        .def(py::init<GPUContext&>(), py::arg("ctx"),
             "Create FormScriptGenerator bound to GPU context")

        .def("set_params", &PyFormScriptGenerator::set_params,
             py::arg("fs") = 12e6,
             py::arg("antennas") = 1,
             py::arg("points") = 4096,
             py::arg("f0") = 0.0,
             py::arg("amplitude") = 1.0,
             py::arg("noise_amplitude") = 0.0,
             py::arg("phase") = 0.0,
             py::arg("fdev") = 0.0,
             py::arg("norm") = 0.7071067811865476,
             py::arg("tau_base") = 0.0,
             py::arg("tau_step") = 0.0,
             py::arg("tau_min") = 0.0,
             py::arg("tau_max") = 0.0,
             py::arg("tau_seed") = 12345,
             py::arg("noise_seed") = 0,
             "Set signal parameters (same as FormSignalGenerator)")

        .def("set_params_from_string", &PyFormScriptGenerator::set_params_from_string,
             py::arg("params"),
             "Set parameters from string: 'f0=1e6,a=1.0,antennas=8'")

        .def("compile", &PyFormScriptGenerator::compile,
             "Compile OpenCL kernel from current parameters.\n"
             "Must be called before generate() (unless using load_kernel).")

        .def("generate_script", &PyFormScriptGenerator::generate_script,
             "Generate human-readable DSL script text.\n\n"
             "Returns:\n"
             "  str: DSL text with [Params], [Defs], [Signal] sections")

        .def("generate_kernel_source", &PyFormScriptGenerator::generate_kernel_source,
             "Generate full OpenCL kernel source with #define params.\n\n"
             "Returns:\n"
             "  str: OpenCL C source code")

        .def("save_kernel", &PyFormScriptGenerator::save_kernel,
             py::arg("name"), py::arg("comment") = "",
             "Save compiled kernel to disk.\n\n"
             "Creates: name.cl + bin/name_opencl.bin\n"
             "If name exists, old files renamed to name_00, name_01, ...\n\n"
             "Args:\n"
             "  name: kernel name (without extension)\n"
             "  comment: optional description for manifest.json")

        .def("load_kernel", &PyFormScriptGenerator::load_kernel,
             py::arg("name"),
             "Load kernel from disk by name.\n\n"
             "Priority: binary (fast) -> source (compile + save binary)\n\n"
             "Args:\n"
             "  name: kernel name (without extension)")

        .def("list_kernels", &PyFormScriptGenerator::list_kernels,
             "List available kernel names from manifest.json.\n\n"
             "Returns:\n"
             "  list[str]: kernel names")

        .def("generate", &PyFormScriptGenerator::generate,
             py::arg("output") = "cpu",
             "Generate signal on GPU.\n\n"
             "Args:\n"
             "  output: 'cpu' (default) — return numpy array\n"
             "          'gpu' — return GPUBuffer (use .read() to get numpy)\n\n"
             "Returns:\n"
             "  output='cpu': numpy.ndarray complex64 (points,) or (antennas, points)\n"
             "  output='gpu': GPUBuffer with .read(), .shape")

        .def_property_readonly("antennas", &PyFormScriptGenerator::antennas,
             "Number of antennas/channels")

        .def_property_readonly("points", &PyFormScriptGenerator::points,
             "Samples per channel")

        .def_property_readonly("fs", &PyFormScriptGenerator::fs,
             "Sample rate (Hz)")

        .def_property_readonly("is_ready", &PyFormScriptGenerator::is_ready,
             "True if kernel is compiled")

        .def_property_readonly("kernel_source", &PyFormScriptGenerator::kernel_source,
             "Current compiled OpenCL kernel source")

        .def_static("get_kernels_dir", &PyFormScriptGenerator::kernels_dir,
             "Path to kernels directory")

        .def_static("get_kernels_bin_dir", &PyFormScriptGenerator::kernels_bin_dir,
             "Path to compiled kernel binaries directory")

        .def("__repr__", [](const PyFormScriptGenerator& self) {
            if (self.is_ready()) {
                return "<FormScriptGenerator antennas=" + std::to_string(self.antennas()) +
                       " points=" + std::to_string(self.points()) + " ready>";
            }
            return std::string("<FormScriptGenerator not compiled>");
        });

    // ════════════════════════════════════════════════════════════════
    // DelayedFormSignalGenerator (Farrow 48×5 fractional delay)
    // ════════════════════════════════════════════════════════════════
    py::class_<PyDelayedFormSignalGenerator>(m, "DelayedFormSignalGenerator",
        "Multi-channel signal generator with fractional delay (Farrow 48x5).\n\n"
        "Pipeline:\n"
        "  1. Generate clean signal (getX formula, no noise)\n"
        "  2. Apply fractional delay: integer shift + 5-point Lagrange interpolation\n"
        "  3. Add noise (Philox + Box-Muller)\n\n"
        "Delay units: microseconds (float) per antenna.\n"
        "Matrix: 48 rows (fractional delay bins) x 5 columns (interpolation taps).\n\n"
        "Usage:\n"
        "  gen = gpuworklib.DelayedFormSignalGenerator(ctx)\n"
        "  gen.set_params(fs=12e6, f0=1e6, antennas=8, points=4096)\n"
        "  gen.set_delays([0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5])\n"
        "  data = gen.generate()  # numpy (8, 4096) complex64\n")
        .def(py::init<GPUContext&>(), py::arg("ctx"),
             "Create DelayedFormSignalGenerator bound to GPU context")

        .def("set_params", &PyDelayedFormSignalGenerator::set_params,
             py::arg("fs") = 12e6,
             py::arg("antennas") = 1,
             py::arg("points") = 4096,
             py::arg("f0") = 0.0,
             py::arg("amplitude") = 1.0,
             py::arg("noise_amplitude") = 0.0,
             py::arg("phase") = 0.0,
             py::arg("fdev") = 0.0,
             py::arg("norm") = 0.7071067811865476,
             py::arg("noise_seed") = 0,
             "Set signal parameters.\n\n"
             "Args:\n"
             "  fs: sample rate (Hz, default 12 MHz)\n"
             "  antennas: number of channels (default 1)\n"
             "  points: samples per channel (default 4096)\n"
             "  f0: center frequency (Hz)\n"
             "  amplitude: signal amplitude 'a' (default 1.0)\n"
             "  noise_amplitude: noise amplitude 'an' (added AFTER delay)\n"
             "  phase: initial phase in radians (default 0)\n"
             "  fdev: frequency deviation for chirp (default 0)\n"
             "  norm: normalization factor (default 1/sqrt(2))\n"
             "  noise_seed: seed for noise PRNG (0 = auto)\n")

        .def("set_delays", &PyDelayedFormSignalGenerator::set_delays,
             py::arg("delay_us"),
             "Set per-antenna delays in microseconds.\n\n"
             "Args:\n"
             "  delay_us: list of floats, one per antenna\n\n"
             "Example:\n"
             "  gen.set_delays([0.0, 1.5, 3.0, 4.5])  # 4 antennas")

        .def("load_matrix", &PyDelayedFormSignalGenerator::load_matrix,
             py::arg("json_path"),
             "Load Lagrange 48x5 matrix from JSON file.\n\n"
             "Optional: built-in matrix used by default.\n"
             "Format: { \"data\": [[row0], [row1], ...] }")

        .def("generate", &PyDelayedFormSignalGenerator::generate,
             py::arg("output") = "cpu",
             "Generate signal with fractional delay on GPU.\n\n"
             "Args:\n"
             "  output: 'cpu' (default) — numpy, 'gpu' — GPUBuffer\n\n"
             "Returns:\n"
             "  output='cpu': numpy.ndarray complex64 (points,) or (antennas, points)\n"
             "  output='gpu': GPUBuffer with .read(), .shape")

        .def_property_readonly("antennas", &PyDelayedFormSignalGenerator::antennas,
             "Number of antennas/channels")

        .def_property_readonly("points", &PyDelayedFormSignalGenerator::points,
             "Samples per channel")

        .def_property_readonly("fs", &PyDelayedFormSignalGenerator::fs,
             "Sample rate (Hz)")

        .def_property_readonly("delays", &PyDelayedFormSignalGenerator::get_delays,
             "Current delays (list of float, microseconds)")

        .def("__repr__", [](const PyDelayedFormSignalGenerator& self) {
            return "<DelayedFormSignalGenerator antennas=" + std::to_string(self.antennas()) +
                   " points=" + std::to_string(self.points()) +
                   " fs=" + std::to_string(static_cast<int>(self.fs())) + ">";
        });

    // ════════════════════════════════════════════════════════════════
    // SpectrumMaximaFinder
    // ════════════════════════════════════════════════════════════════
    py::class_<PySpectrumMaximaFinder>(m, "SpectrumMaximaFinder",
        "Find all local maxima in FFT spectrum (GPU accelerated).\n\n"
        "Pipeline: Detection -> Prefix Sum (Blelloch Scan) -> Compaction\n\n"
        "Usage:\n"
        "  finder = gpuworklib.SpectrumMaximaFinder(ctx)\n"
        "  result = finder.find_all_maxima(fft_data, sample_rate=1000)\n\n"
        "For single beam returns dict, for multi-beam returns list[dict].\n"
        "Each dict: {positions, magnitudes, frequencies, num_maxima}")
        .def(py::init<GPUContext&>(), py::arg("ctx"),
             "Create spectrum maxima finder bound to GPU context")

        .def("find_all_maxima", &PySpectrumMaximaFinder::find_all_maxima,
             py::arg("fft_data"), py::arg("sample_rate"),
             py::arg("beam_count") = 0, py::arg("nFFT") = 0,
             py::arg("search_start") = 0, py::arg("search_end") = 0,
             "Find ALL local maxima in FFT spectrum.\n\n"
             "Args:\n"
             "  fft_data: numpy complex64 array with FFT result\n"
             "    1D (nFFT,) for single beam, 2D (beams, nFFT) for multi-beam\n"
             "  sample_rate: sampling rate (Hz)\n"
             "  beam_count: auto-detected from shape if 0\n"
             "  nFFT: auto-detected from shape if 0\n"
             "  search_start: start bin (0 = default = 1, skip DC)\n"
             "  search_end: end bin (0 = default = nFFT/2)\n\n"
             "Returns:\n"
             "  Single beam: dict with keys:\n"
             "    positions (np.uint32), magnitudes (np.float32),\n"
             "    frequencies (np.float32), num_maxima (int)\n"
             "  Multi-beam: list[dict] with additional 'antenna_id' key\n\n"
             "Example:\n"
             "  import numpy as np\n"
             "  import gpuworklib\n\n"
             "  ctx = gpuworklib.GPUContext(0)\n"
             "  fft = gpuworklib.FFTProcessor(ctx)\n"
             "  finder = gpuworklib.SpectrumMaximaFinder(ctx)\n\n"
             "  signal = np.sin(2*np.pi*100*np.arange(1024)/1000).astype(np.complex64)\n"
             "  spectrum = fft.process_complex(signal, sample_rate=1000)\n"
             "  peaks = finder.find_all_maxima(spectrum, sample_rate=1000)\n"
             "  print(f'Found {peaks[\"num_maxima\"]} peaks')\n"
             "  print(f'Frequencies: {peaks[\"frequencies\"]} Hz')");

    // ════════════════════════════════════════════════════════════════
    // LfmAnalyticalDelay (see py_lfm_analytical_delay.hpp)
    // ════════════════════════════════════════════════════════════════
    register_lfm_analytical_delay(m);

    // ════════════════════════════════════════════════════════════════
    // LchFarrow (see py_lch_farrow.hpp)
    // ════════════════════════════════════════════════════════════════
    register_lch_farrow(m);

    // ════════════════════════════════════════════════════════════════
    // FirFilter + IirFilter (see py_filters.hpp)
    // ════════════════════════════════════════════════════════════════
    register_filters(m);

    // ════════════════════════════════════════════════════════════════
    // HeterodyneDechirp (see py_heterodyne.hpp)
    // ════════════════════════════════════════════════════════════════
    register_heterodyne(m);

#if ENABLE_ROCM
    // ════════════════════════════════════════════════════════════════
    // ROCm classes (Linux + AMD GPU only)
    // ════════════════════════════════════════════════════════════════

    // ROCmGPUContext
    py::class_<ROCmGPUContext>(m, "ROCmGPUContext",
        "ROCm GPU context wrapping AMD HIP device.\n\n"
        "Usage:\n"
        "  ctx = gpuworklib.ROCmGPUContext(device_index=0)\n"
        "  print(ctx.device_name)")
        .def(py::init<int>(), py::arg("device_index") = 0,
             "Create ROCm GPU context on given device index")
        .def_property_readonly("device_name", &ROCmGPUContext::device_name,
             "GPU device name (str)")
        .def_property_readonly("device_index", &ROCmGPUContext::device_index,
             "GPU device index (int)")
        .def("__repr__", [](const ROCmGPUContext& ctx) {
            return "<ROCmGPUContext device='" + ctx.device_name() + "'>";
        })
        .def("__enter__", [](ROCmGPUContext& self) -> ROCmGPUContext& { return self; },
             py::return_value_policy::reference)
        .def("__exit__", [](ROCmGPUContext&, py::object, py::object, py::object) {
            return false;
        });

    // FirFilterROCm (see py_filters_rocm.hpp)
    register_fir_filter_rocm(m);

    // IirFilterROCm (see py_filters_rocm.hpp)
    register_iir_filter_rocm(m);

    // LchFarrowROCm (see py_lch_farrow_rocm.hpp)
    register_lch_farrow_rocm(m);

    // HeterodyneROCm (see py_heterodyne_rocm.hpp)
    register_heterodyne_rocm(m);

    // StatisticsProcessor (see py_statistics.hpp)
    register_statistics(m);

    // CholeskyInverterROCm (see py_vector_algebra_rocm.hpp)
    register_cholesky_inverter_rocm(m);

    // HybridGPUContext
    py::class_<HybridGPUContext>(m, "HybridGPUContext",
        "Hybrid GPU context with OpenCL + ROCm on one GPU.\n\n"
        "Usage:\n"
        "  ctx = gpuworklib.HybridGPUContext(device_index=0)\n"
        "  print(ctx.opencl_device_name)\n"
        "  print(ctx.rocm_device_name)\n"
        "  print(ctx.zero_copy_method)\n"
        "  if ctx.is_zero_copy_supported:\n"
        "      print('ZeroCopy available!')")
        .def(py::init<int>(), py::arg("device_index") = 0,
             "Create Hybrid GPU context (OpenCL + ROCm) on given device index")
        .def_property_readonly("opencl_device_name", &HybridGPUContext::opencl_device_name,
             "OpenCL sub-backend device name (str)")
        .def_property_readonly("rocm_device_name", &HybridGPUContext::rocm_device_name,
             "ROCm sub-backend device name (str)")
        .def_property_readonly("device_name", &HybridGPUContext::device_name,
             "Combined device name (str)")
        .def_property_readonly("device_index", &HybridGPUContext::device_index,
             "GPU device index (int)")
        .def_property_readonly("zero_copy_method", &HybridGPUContext::zero_copy_method,
             "Best available ZeroCopy method: 'AMD GPU VA', 'DMA-BUF', 'SVM', or 'None'")
        .def_property_readonly("is_zero_copy_supported", &HybridGPUContext::is_zero_copy_supported,
             "True if any ZeroCopy method is available on this GPU")
        .def("__repr__", [](const HybridGPUContext& ctx) {
            return "<HybridGPUContext device='" + ctx.device_name() +
                   "' zero_copy='" + ctx.zero_copy_method() + "'>";
        })
        .def("__enter__", [](HybridGPUContext& self) -> HybridGPUContext& { return self; },
             py::return_value_policy::reference)
        .def("__exit__", [](HybridGPUContext&, py::object, py::object, py::object) {
            return false;
        });
#endif  // ENABLE_ROCM

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
