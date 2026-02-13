# GPUWorkLib - Detailed Implementation Plan
## Based on plan_2026_02_13.md + Deep Analysis

> **Date**: 2026-02-13
> **Author**: Kodo (AI Assistant)
> **Status**: DRAFT - Awaiting Alex's approval

---

## Architecture Overview

```
GPUWorkLib/
├── DrvGPU/                     # [EXISTS] GPU driver abstraction
│   ├── interface/i_backend.hpp  # IBackend (Bridge Pattern)
│   └── backends/opencl/         # OpenCLBackend
│
├── modules/
│   ├── fft_maxima/              # [EXISTS] SpectrumMaximaFinder
│   │   ├── include/             # spectrum_maxima_finder.h, types, input_data
│   │   ├── src/                 # AntennaFFTCore, AntennaFFTProcMax
│   │   ├── kernels/             # pre-callback, post-kernel (OnePeak, TwoPeaks)
│   │   └── tests/               # test_signal_generator.hpp (TEMPORARY!)
│   │
│   ├── signal_generators/       # [NEW] Task 1 - Signal Generators Module
│   │   ├── include/
│   │   │   ├── i_signal_generator.hpp       # Interface
│   │   │   ├── signal_service.hpp           # Facade (DI)
│   │   │   ├── signal_generator_factory.hpp # Factory
│   │   │   ├── generators/
│   │   │   │   ├── cw_generator.hpp         # CW (sin/cos/IQ)
│   │   │   │   ├── lfm_generator.hpp        # LFM (chirp)
│   │   │   │   └── noise_generator.hpp      # White/Gaussian noise
│   │   │   └── params/
│   │   │       ├── signal_request.hpp       # SignalRequest + variant params
│   │   │       └── system_sampling.hpp      # SystemSampling{fs, length}
│   │   ├── src/
│   │   │   ├── cw_generator.cpp
│   │   │   ├── lfm_generator.cpp
│   │   │   ├── noise_generator.cpp
│   │   │   └── signal_generator_factory.cpp
│   │   ├── kernels/
│   │   │   ├── cw_kernel.cl                 # GPU kernel: sinusoid generation
│   │   │   ├── lfm_kernel.cl                # GPU kernel: LFM (from LCH-Farrow01)
│   │   │   └── noise_kernel.cl              # GPU kernel: Philox PRNG + Box-Muller
│   │   ├── tests/
│   │   │   ├── test_cw_generator.hpp
│   │   │   ├── test_lfm_generator.hpp
│   │   │   ├── test_noise_generator.hpp
│   │   │   └── test_generator_fft_integration.hpp  # Generator -> FFT -> check freq
│   │   └── CMakeLists.txt
│   │
│   └── fft_processor/           # [NEW] Task 2 - FFT Processor Module
│       ├── include/
│       │   ├── fft_processor.hpp             # Main class
│       │   └── fft_processor_types.hpp       # FFTOutputMode, FFTProcessorParams
│       ├── src/
│       │   └── fft_processor.cpp
│       ├── kernels/
│       │   └── fft_post_processing.cl        # mag+phase+freq conversion
│       ├── tests/
│       │   ├── test_fft_processor.hpp
│       │   └── test_fft_vs_cpu.hpp           # GPU FFT vs CPU reference
│       └── CMakeLists.txt
│
├── python/                      # [NEW] Task 3 - Python Bindings
│   ├── CMakeLists.txt
│   ├── gpu_worklib_bindings.cpp  # pybind11 module
│   └── tests/
│       └── test_gpu_worklib.py   # pytest
│
└── third_party/
    ├── plog/                    # [EXISTS] Logger
    └── pybind11/                # [NEW] To install via submodule
```

---

## Task 0: Analysis of antenna_fft (0.5 day)

### Status: PARTIALLY DONE (in plan file)

### Remaining Work:
1. Create dependency graph: `MemoryBank/specs/antenna_fft_dependencies.md`
2. Document which parts of AntennaFFTCore/AntennaFFTProcMax can be reused by FFTProcessor

### Files:
- `modules/fft_maxima/src/antenna_fft_core.cpp` - base: OpenCL context, clFFT, batch
- `modules/fft_maxima/src/antenna_fft_release.cpp` - AntennaFFTProcMax with callbacks
- `modules/fft_maxima/include/spectrum_maxima_finder.h` - public API

### Key Finding:
AntennaFFTCore handles: clFFT plan creation, pre-callback, batch config.
FFTProcessor can REUSE the clFFT setup logic but with different post-processing.

---

## Task 1: Signal Generators Module (5-7 days)

### Architecture Decision: Strategy + Factory + DI via IBackend*

### Phase 1.1: Infrastructure (1 day)

#### 1.1.1 Create module structure
```
modules/signal_generators/
├── include/
│   ├── i_signal_generator.hpp
│   ├── signal_service.hpp
│   ├── signal_generator_factory.hpp
│   ├── generators/
│   │   ├── cw_generator.hpp
│   │   ├── lfm_generator.hpp
│   │   └── noise_generator.hpp
│   └── params/
│       ├── signal_request.hpp
│       └── system_sampling.hpp
├── src/
├── kernels/
├── tests/
└── CMakeLists.txt
```

#### 1.1.2 Base types and interfaces

```cpp
// system_sampling.hpp
struct SystemSampling {
    double fs;          // Sample rate (Hz)
    size_t length;      // Number of samples per beam
};

// signal_request.hpp
enum class SignalKind { CW, LFM, NOISE };

struct CwParams {
    double f0;              // Frequency (Hz)
    double phase = 0.0;     // Initial phase (rad)
    double amplitude = 1.0;
    bool complex_iq = true; // true = exp(j*phase), false = real only
};

struct LfmParams {
    double f_start;         // Start frequency (Hz)
    double f_end;           // End frequency (Hz)
    double duration;        // Pulse duration (s), 0 = use SystemSampling.length
    double amplitude = 1.0;
    bool complex_iq = true;
};

struct NoiseParams {
    enum class NoiseType { WHITE, GAUSSIAN };
    NoiseType type = NoiseType::GAUSSIAN;
    double power = 1.0;     // Noise power (variance for Gaussian)
    uint64_t seed = 0;      // 0 = random seed
};

struct SignalRequest {
    SignalKind kind;
    SystemSampling system;
    std::variant<CwParams, LfmParams, NoiseParams> params;
};

// i_signal_generator.hpp
class ISignalGenerator {
public:
    virtual ~ISignalGenerator() = default;

    /// Generate to CPU buffer (reference implementation)
    virtual void generateToCpu(
        const SystemSampling& system,
        std::complex<float>* out,
        size_t out_size) = 0;

    /// Generate to GPU buffer (returns cl_mem, caller must release)
    virtual cl_mem generateToGpu(
        const SystemSampling& system,
        size_t beam_count = 1) = 0;

    /// Get signal kind
    virtual SignalKind kind() const = 0;
};
```

#### 1.1.3 CMakeLists.txt for module

```cmake
add_library(signal_generators STATIC
    src/cw_generator.cpp
    src/lfm_generator.cpp
    src/noise_generator.cpp
    src/signal_generator_factory.cpp
)
target_link_libraries(signal_generators PUBLIC DrvGPU::drvgpu OpenCL::OpenCL)
add_library(GPUWorkLib::signal_generators ALIAS signal_generators)
```

### Phase 1.2: CW Generator (1 day)

**Source**: Migrate `test_signal_generator.hpp` from tests/

**GPU Kernel** (cw_kernel.cl):
```c
__kernel void generate_cw(
    __global float2* output,
    const uint beam_count,
    const uint n_point,
    const float sample_rate,
    const float base_freq,
    const float freq_step,   // For multi-beam: freq_i = base_freq + i * freq_step
    const float amplitude,
    const float initial_phase)
{
    const size_t gid = get_global_id(0);
    const size_t beam_id = gid / n_point;
    const size_t sample_id = gid % n_point;
    if (beam_id >= beam_count) return;

    const float freq = base_freq + (float)beam_id * freq_step;
    const float t = (float)sample_id / sample_rate;
    const float phase = 2.0f * M_PI_F * freq * t + initial_phase;

    output[gid] = (float2)(amplitude * cos(phase), amplitude * sin(phase));
}
```

**Key Difference from TestSignalGenerator**:
- Takes `IBackend*` instead of raw `cl_context/cl_queue`
- Supports configurable freq_step, amplitude, initial_phase
- CPU reference implementation for testing

### Phase 1.3: LFM Generator (1.5 days)

**Source**: Adapt from `LCH-Farrow01/src/GPU/generator_gpu_new.cpp`

**Key Changes**:
- Remove singleton `OpenCLComputeEngine::GetInstance()` -> use `IBackend*`
- Support batch generation (N beams with different delays)
- Keep 3 modes: basic, delayed, combined

**GPU Kernel** (lfm_kernel.cl) - adapted from LCH-Farrow01:
```c
// kernel_lfm_basic: s(t) = exp(j*pi*k*t^2) where k = (f_stop - f_start) / duration
// kernel_lfm_delayed: with fractional delay per beam
// kernel_lfm_combined: angle delay + time delay
```

### Phase 1.4: Noise Generator (1 day)

**Algorithm**: Philox-2x32 PRNG + Box-Muller Transform

**Why Philox**: Counter-based RNG, perfect for GPU:
- Each work item generates independent random numbers
- No shared state, fully parallel
- Deterministic with seed

**GPU Kernel** (noise_kernel.cl):
```c
// Philox key schedule
uint2 philox2x32(uint2 counter, uint key) {
    // Single round of Philox
    uint hi = mul_hi(counter.x, 0xD2511F53u);
    uint lo = counter.x * 0xD2511F53u;
    return (uint2)(hi ^ key ^ counter.y, lo);
}

__kernel void generate_noise_gaussian(
    __global float2* output,
    const uint total_points,
    const float power,
    const uint seed)
{
    uint gid = get_global_id(0);
    if (gid >= total_points) return;

    // Generate 2 uniform random numbers using Philox
    uint2 ctr = (uint2)(gid, seed);
    uint2 rnd = philox2x32(ctr, 0xCD9E8D57u);

    // Box-Muller transform: uniform -> Gaussian
    float u1 = (float)(rnd.x) / 4294967296.0f + 1e-10f;
    float u2 = (float)(rnd.y) / 4294967296.0f;
    float r = sqrt(-2.0f * log(u1)) * sqrt(power);
    float theta = 2.0f * M_PI_F * u2;

    output[gid] = (float2)(r * cos(theta), r * sin(theta));
}
```

### Phase 1.5: Factory + Service + Tests (1.5 days)

**SignalGeneratorFactory**:
```cpp
class SignalGeneratorFactory {
public:
    static std::unique_ptr<ISignalGenerator> create(
        SignalKind kind,
        drv_gpu_lib::IBackend* backend);
};
```

**SignalService** (facade):
```cpp
class SignalService {
public:
    SignalService(drv_gpu_lib::IBackend* backend);

    // CPU generation
    std::vector<std::complex<float>> generateCpu(const SignalRequest& request);

    // GPU generation (returns cl_mem)
    cl_mem generateGpu(const SignalRequest& request, size_t beam_count = 1);

private:
    drv_gpu_lib::IBackend* backend_;
};
```

**Tests**:
1. Unit: each generator CPU vs GPU comparison (tolerance < 1e-5)
2. Integration: Generator -> FFT -> check frequency matches CwParams.f0
3. Noise: statistical test (mean ~ 0, variance ~ power)

---

## Task 2: FFT Processor Module (2-3 days)

### Architecture: New class, reuses clFFT infrastructure

### Phase 2.1: Types and Interface (0.5 day)

```cpp
// fft_processor_types.hpp
enum class FFTOutputMode {
    COMPLEX,            // Return raw complex FFT spectrum
    MAGNITUDE_PHASE,    // Return |FFT|, phase(FFT)
    MAGNITUDE_PHASE_FREQ // Return |FFT|, phase, freq_hz (bin * fs / nFFT)
};

struct FFTProcessorParams {
    size_t beam_count;       // 1 or N
    size_t n_point;          // Input samples per beam
    float sample_rate;       // fs for freq calculation
    FFTOutputMode output_mode = FFTOutputMode::COMPLEX;
    uint32_t repeat_count = 1; // nFFT multiplier (1 = nextPow2(n_point))
    float memory_limit = 0.80f;
};

// Output structures
struct FFTComplexResult {
    uint32_t beam_id;
    std::vector<std::complex<float>> spectrum;  // nFFT complex values
};

struct FFTMagPhaseResult {
    uint32_t beam_id;
    std::vector<float> magnitude;   // nFFT magnitudes
    std::vector<float> phase;       // nFFT phases (radians)
    std::vector<float> frequency;   // nFFT freq values (Hz), only if MAGNITUDE_PHASE_FREQ
};
```

### Phase 2.2: FFTProcessor Class (1.5 days)

```cpp
class FFTProcessor {
public:
    explicit FFTProcessor(drv_gpu_lib::IBackend* backend);
    ~FFTProcessor();

    // Process from CPU data
    template<typename OutputT>
    std::vector<OutputT> Process(
        const std::vector<std::complex<float>>& data,
        const FFTProcessorParams& params);

    // Process from GPU data
    template<typename OutputT>
    std::vector<OutputT> Process(
        cl_mem gpu_data,
        const FFTProcessorParams& params);

private:
    // Reuse clFFT plan creation from AntennaFFTCore pattern
    void CreateFFTPlan(size_t nFFT, size_t batch_size);
    void ExecuteFFT(cl_mem input, cl_mem output, size_t batch);

    // Post-processing kernels
    void ConvertToMagnitudePhase(cl_mem fft_output, cl_mem mag, cl_mem phase, size_t n);
    void CalculateFrequencies(cl_mem freq_out, size_t nFFT, float fs);

    drv_gpu_lib::IBackend* backend_;
    clfftPlanHandle plan_ = 0;
    // ...
};
```

### Phase 2.3: Post-processing Kernel (0.5 day)

```c
__kernel void complex_to_mag_phase(
    __global const float2* fft_data,    // Complex FFT output
    __global float* magnitude,           // |FFT[i]|
    __global float* phase,               // arg(FFT[i])
    const uint nFFT)
{
    uint gid = get_global_id(0);
    if (gid >= nFFT) return;

    float2 z = fft_data[gid];
    magnitude[gid] = sqrt(z.x * z.x + z.y * z.y);
    phase[gid] = atan2(z.y, z.x);
}
```

### Phase 2.4: Tests (0.5 day)

1. 1 beam: GPU FFT vs CPU FFT (use FFTW or manual DFT for small N)
2. N beams: batch vs one-by-one comparison
3. Known tone: generate sin(2*pi*f0*t), FFT, check peak at f0

---

## Task 3: Python Bindings (3-4 days)

### Phase 3.1: Setup pybind11 (0.5 day)

```bash
# Add as git submodule
git submodule add https://github.com/pybind/pybind11.git third_party/pybind11
```

CMake:
```cmake
add_subdirectory(third_party/pybind11)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(gpu_worklib python/gpu_worklib_bindings.cpp)
target_link_libraries(gpu_worklib PRIVATE
    GPUWorkLib::signal_generators
    GPUWorkLib::fft_processor
    GPUWorkLib::fft_maxima
    DrvGPU::drvgpu
)
```

### Phase 3.2: Bindings Implementation (2 days)

```cpp
// gpu_worklib_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(gpu_worklib, m) {
    m.doc() = "GPUWorkLib - GPU Signal Processing";

    // GPU Context wrapper
    py::class_<GPUContext>(m, "GPUContext")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("device_name", &GPUContext::GetDeviceName);

    // Signal Generator
    py::class_<PySignalService>(m, "SignalGenerator")
        .def(py::init<GPUContext&>())
        .def("generate_cw", &PySignalService::GenerateCW,
             py::arg("freq"), py::arg("fs"), py::arg("length"),
             py::arg("amplitude") = 1.0, py::arg("beam_count") = 1)
        .def("generate_lfm", &PySignalService::GenerateLFM,
             py::arg("f_start"), py::arg("f_end"), py::arg("fs"),
             py::arg("length"), py::arg("beam_count") = 1)
        .def("generate_noise", &PySignalService::GenerateNoise,
             py::arg("fs"), py::arg("length"), py::arg("power") = 1.0);

    // FFT Processor
    py::class_<PyFFTProcessor>(m, "FFTProcessor")
        .def(py::init<GPUContext&>())
        .def("fft", &PyFFTProcessor::FFT,
             py::arg("signal"), py::arg("fs"));

    // Spectrum Maxima Finder
    py::class_<PySpectrumFinder>(m, "SpectrumMaximaFinder")
        .def(py::init<GPUContext&>())
        .def("find_maxima", &PySpectrumFinder::FindMaxima,
             py::arg("signal"), py::arg("fs"),
             py::arg("antenna_count"), py::arg("n_point"));
}
```

**Zero-copy pattern** (GPU -> NumPy):
```cpp
py::array_t<std::complex<float>> GenerateCW(...) {
    // 1. Generate on GPU -> cl_mem
    cl_mem gpu_data = service_.generateGpu(request, beam_count);

    // 2. Allocate NumPy array
    py::array_t<std::complex<float>> result({beam_count, length});
    auto buf = result.request();

    // 3. Read GPU -> NumPy buffer directly (zero-copy host side)
    clEnqueueReadBuffer(queue, gpu_data, CL_TRUE, 0,
                        beam_count * length * sizeof(std::complex<float>),
                        buf.ptr, 0, nullptr, nullptr);

    // 4. Release GPU buffer
    clReleaseMemObject(gpu_data);
    return result;
}
```

### Phase 3.3: Tests (0.5 day)

```python
# test_gpu_worklib.py
import gpu_worklib
import numpy as np

def test_cw_generation():
    ctx = gpu_worklib.GPUContext(0)
    gen = gpu_worklib.SignalGenerator(ctx)
    signal = gen.generate_cw(freq=100.0, fs=1000.0, length=10000)
    assert signal.dtype == np.complex64
    assert signal.shape == (10000,)

    # Verify frequency via FFT
    spectrum = np.fft.fft(signal)
    peak_bin = np.argmax(np.abs(spectrum[:len(spectrum)//2]))
    peak_freq = peak_bin * 1000.0 / len(spectrum)
    assert abs(peak_freq - 100.0) < 0.5

def test_fft_processor():
    ctx = gpu_worklib.GPUContext(0)
    gen = gpu_worklib.SignalGenerator(ctx)
    fft = gpu_worklib.FFTProcessor(ctx)

    signal = gen.generate_cw(freq=50.0, fs=1000.0, length=4096)
    spectrum = fft.fft(signal, fs=1000.0)
    assert spectrum.dtype == np.complex64
```

---

## Task 4: Fractional Delay Analysis (3-5 days) - LOW PRIORITY

### Phase 4.1: CPU Prototype (1 day)
- Implement Tretter's estimator (LSQ phase slope of beat signal)
- Validate against Cramer-Rao bound with Monte-Carlo

### Phase 4.2: GPU Kernel (1.5 days)
- Adapt kernel from `delay_methods_report.md`
- Phase unwrapping via differential phase (prefix-scan approach)
- Linear regression via parallel reduction

### Phase 4.3: Integration (1 day)
- Create `FractionalDelayEstimator` class
- Integration with signal_generators (LFM as reference + delayed as received)

### Phase 4.4: Tests (0.5 day)
- Known delay -> estimate -> compare with true value
- Multiple SNR levels -> RMSE vs CRB

---

## Execution Order

```
Week 1:
  Day 1-2: Task 0 (finish) + Task 1 Phase 1.1-1.2 (Infrastructure + CW)
  Day 3-4: Task 1 Phase 1.3 (LFM from LCH-Farrow01)
  Day 5:   Task 1 Phase 1.4-1.5 (Noise + Factory + Tests)

Week 2:
  Day 1-2: Task 2 (FFTProcessor complete)
  Day 3-4: Task 3 Phase 3.1-3.2 (pybind11 setup + bindings)
  Day 5:   Task 3 Phase 3.3 (Python tests + polishing)

Week 3 (optional):
  Day 1-3: Task 4 (Fractional Delay - if time permits)
```

---

## Critical Dependencies

```
Task 0 ──→ Task 2 (FFTProcessor reuses AntennaFFTCore patterns)
Task 1 ──→ Task 2 (Generators create test data for FFTProcessor)
Task 1 ──→ Task 3 (Python wraps generators)
Task 2 ──→ Task 3 (Python wraps FFTProcessor)
Task 1 ──→ Task 4 (LFM generator needed for fractional delay)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| clFFT plan reuse in FFTProcessor | Study AntennaFFTCore code carefully, extract common helpers |
| Philox PRNG on different GPU vendors | Fallback to simpler LCG if Philox has precision issues |
| pybind11 + OpenCL on Windows | Test early, MSVC may need specific settings |
| LFM kernel adaptation from LCH-Farrow01 | Keep original kernel as reference, adapt incrementally |

---

## Files to Modify (Existing)

| File | Change |
|------|--------|
| `CMakeLists.txt` (root) | Add `add_subdirectory(modules/signal_generators)`, `add_subdirectory(modules/fft_processor)`, `add_subdirectory(python)` |
| `src/CMakeLists.txt` | Link new modules |
| `modules/fft_maxima/tests/test_signal_generator.hpp` | Mark as deprecated, point to new module |

---

*Plan created by Kodo based on deep analysis of codebase, reference projects, and scientific literature*
