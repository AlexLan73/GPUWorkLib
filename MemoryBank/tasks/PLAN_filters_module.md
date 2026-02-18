# 📋 Plan: GPU Filters Module (FIR + IIR)

> **Статус**: ✅ Одобрен Alex (2026-02-18)
> **Автор**: Кодо (AI Assistant)
> **Для изучения**: Полный детальный план реализации

---

## Context
Создаём новый модуль `filters` для GPUWorkLib — GPU-фильтрацию комплексных сигналов по всем каналам параллельно.
Следуем архитектуре `fft_processor` (STATIC lib, inline kernels, BatchManager, ROCm stub).
Три эволюционных этапа:
1. **Stage 1**: scipy→GPU (Python генерирует коэффициенты, передаёт в C++ → GPU ядро)
2. **Stage 2**: строка→ядро (FormScriptGenerator-подобный механизм, кэш ядер)
3. **Stage 3**: Groq AI micro-agent (natural language → filter design → GPU → plot)

---

## Module Architecture

### Файловая структура
```
modules/filters/
├── CMakeLists.txt
├── include/
│   ├── filters/
│   │   ├── fir_filter.hpp          ← FIR процессор (OpenCL)
│   │   ├── iir_filter.hpp          ← IIR процессор (биквады, OpenCL)
│   │   ├── fir_filter_rocm.hpp     ← ROCm stub для FIR
│   │   └── iir_filter_rocm.hpp     ← ROCm stub для IIR
│   ├── kernels/
│   │   ├── fir_kernels.hpp         ← inline R"CL(...)CL" FIR ядра
│   │   └── iir_kernels.hpp         ← inline R"CL(...)CL" IIR ядра
│   └── types/
│       ├── filter_types.hpp        ← FilterResult, FilterConfig и т.д.
│       ├── filter_params.hpp       ← FirParams, IirParams, BiquadSection
│       └── filter_modes.hpp        ← enum FilterPrecision { Float32, Float64 }
├── src/
│   ├── fir_filter.cpp
│   └── iir_filter.cpp
└── tests/
    ├── all_test.hpp                ← вызывается из src/main.cpp
    ├── test_fir_basic.hpp          ← unit тесты FIR
    ├── test_iir_basic.hpp          ← unit тесты IIR
    └── README.md
```

---

## Key Types (types/filter_params.hpp)

```cpp
namespace filters {

/// Biquad section: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
///                      - a1*y[n-1] - a2*y[n-2]
struct BiquadSection {
    float b0, b1, b2;   ///< Feedforward coeffs
    float a1, a2;       ///< Feedback coeffs (a0 = 1, normalized)
};

struct FirParams {
    std::vector<float> coefficients;  ///< h[k], length = num_taps
    uint32_t channels = 1;
    uint32_t points   = 1024;
};

struct IirParams {
    std::vector<BiquadSection> sections;  ///< Cascade of biquads
    uint32_t channels = 1;
    uint32_t points   = 1024;
};

/// JSON config format:
/// { "type": "fir", "coefficients": [...] }
/// { "type": "iir", "sections": [{"b0":1,"b1":0,"b2":0,"a1":0,"a2":0}, ...] }
struct FilterConfig {
    std::string type;                    ///< "fir" or "iir"
    std::vector<float> coefficients;     ///< FIR coeffs
    std::vector<BiquadSection> sections; ///< IIR biquads
    static FilterConfig LoadJson(const std::string& path);
};

} // namespace filters
```

---

## FIR Kernel Design (kernels/fir_kernels.hpp)

**Алгоритм**: Direct-form FIR convolution.
Каждый work-item вычисляет один выходной отсчёт `(channel, sample_idx)`.

```opencl
// Each work-item: one output sample (channel, sample_idx)
// input/output: float2 (re, im), interleaved [channels * points]
// coeffs: __constant float[], length = num_taps

__kernel void fir_filter_cf32(
    __global const float2* input,
    __global       float2* output,
    __constant     float*  coeffs,
    const uint num_taps,
    const uint points)
{
    uint ch = get_global_id(0);   // channel index
    uint n  = get_global_id(1);   // sample index

    float2 acc = (float2)(0.0f, 0.0f);
    for (uint k = 0; k < num_taps; k++) {
        int idx = (int)n - (int)k;
        if (idx >= 0) {
            acc += coeffs[k] * input[ch * points + idx];
        }
    }
    output[ch * points + n] = acc;
}
```

**Глобальный размер**: `[channels, points]`
**Коэффициенты**: `__constant` память (broadcast cache, ≤64KB ≈ 16384 float)
**Ограничение**: при `num_taps > 16000` → автоматически `__global` память

---

## IIR Kernel Design (kernels/iir_kernels.hpp)

**Стратегия**: Каскад биквадратных секций.
- Каждая секция: data dependency по времени → последовательный цикл по samples
- Параллелизм: только по каналам
- N секций = N последовательных вызовов ядра

**Форма**: Direct Form II Transposed (числово стабильная)

```opencl
// Biquad Direct Form II Transposed
// ONE kernel call per biquad section, sequential sections
// Parallelism: across channels only

__kernel void iir_biquad_cf32(
    __global const float2* input,   // [channels * points] float2
    __global       float2* output,  // [channels * points] float2
    const float b0, const float b1, const float b2,
    const float a1, const float a2,
    const uint points)
{
    uint ch = get_global_id(0);  // one work-item per channel

    float2 w1 = (float2)(0.0f, 0.0f);
    float2 w2 = (float2)(0.0f, 0.0f);

    for (uint n = 0; n < points; n++) {
        float2 x = input[ch * points + n];
        float2 y = b0 * x + w1;
        w1 = b1 * x - a1 * y + w2;
        w2 = b2 * x - a2 * y;
        output[ch * points + n] = y;
    }
}
```

**Глобальный размер**: `[channels]`
**Секции**: output секции N → input секции N+1

---

## FirFilter Class (include/filters/fir_filter.hpp)

```cpp
namespace filters {

class FirFilter {
public:
    explicit FirFilter(drv_gpu_lib::IBackend* backend);
    ~FirFilter();

    // No copy, move OK
    FirFilter(const FirFilter&) = delete;
    FirFilter& operator=(const FirFilter&) = delete;
    FirFilter(FirFilter&&) noexcept;
    FirFilter& operator=(FirFilter&&) noexcept;

    /// Load from JSON: { "type":"fir", "coefficients":[...] }
    void LoadConfig(const std::string& json_path);

    /// Set coefficients directly (Stage 1: scipy → GPU)
    void SetCoefficients(const std::vector<float>& coeffs);

    /// Process GPU buffer [channels * points] complex float
    /// Returns: cl_mem (caller must clReleaseMemObject)
    /// input_buf NOT released by this method
    drv_gpu_lib::InputData<cl_mem> Process(
        cl_mem input_buf, uint32_t channels, uint32_t points);

    /// CPU reference (for validation)
    std::vector<std::complex<float>> ProcessCpu(
        const std::vector<std::complex<float>>& input,
        uint32_t channels, uint32_t points);

    // Getters
    uint32_t GetNumTaps() const;
    const std::vector<float>& GetCoefficients() const;

private:
    void CompileKernel();
    void UploadCoefficients();
    void ReleaseGpuResources();

    drv_gpu_lib::IBackend* backend_ = nullptr;
    std::vector<float> coefficients_;

    cl_context       context_  = nullptr;
    cl_command_queue queue_    = nullptr;
    cl_device_id     device_   = nullptr;
    cl_program       program_  = nullptr;
    cl_mem           coeff_buf_= nullptr;
};

} // namespace filters
```

---

## IirFilter Class (include/filters/iir_filter.hpp)

```cpp
namespace filters {

class IirFilter {
public:
    explicit IirFilter(drv_gpu_lib::IBackend* backend);
    ~IirFilter();

    // No copy, move OK
    IirFilter(const IirFilter&) = delete;
    IirFilter& operator=(const IirFilter&) = delete;

    /// Load from JSON: { "type":"iir", "sections":[{b0,b1,b2,a1,a2},...] }
    void LoadConfig(const std::string& json_path);

    /// Set biquad sections directly (Stage 1)
    void SetBiquadSections(const std::vector<BiquadSection>& sections);

    /// Process GPU buffer [channels * points] complex float
    drv_gpu_lib::InputData<cl_mem> Process(
        cl_mem input_buf, uint32_t channels, uint32_t points);

    /// CPU reference
    std::vector<std::complex<float>> ProcessCpu(
        const std::vector<std::complex<float>>& input,
        uint32_t channels, uint32_t points);

    uint32_t GetNumSections() const;

private:
    void CompileKernel();
    void ReleaseGpuResources();

    drv_gpu_lib::IBackend* backend_ = nullptr;
    std::vector<BiquadSection> sections_;

    cl_context       context_ = nullptr;
    cl_command_queue queue_   = nullptr;
    cl_device_id     device_  = nullptr;
    cl_program       program_ = nullptr;
};

} // namespace filters
```

---

## ROCm Stubs (паттерн из fft_maxima)

```cpp
// fir_filter_rocm.hpp
namespace filters {
class FirFilterROCm {
public:
    explicit FirFilterROCm(drv_gpu_lib::IBackend* backend) {}
    void LoadConfig(const std::string&) {
        throw std::runtime_error("ROCm not implemented");
    }
    void SetCoefficients(const std::vector<float>&) {
        throw std::runtime_error("ROCm not implemented");
    }
    drv_gpu_lib::InputData<cl_mem> Process(cl_mem, uint32_t, uint32_t) {
        throw std::runtime_error("ROCm not implemented");
    }
};
} // namespace filters

// iir_filter_rocm.hpp — аналогично
```

---

## CMakeLists.txt (modules/filters/)

```cmake
cmake_minimum_required(VERSION 3.20)
set(MODULE_NAME filters)
message(STATUS "Configuring module: ${MODULE_NAME}")

find_package(OpenCL REQUIRED)

add_library(${MODULE_NAME} STATIC
    src/fir_filter.cpp
    src/iir_filter.cpp
)

target_include_directories(${MODULE_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
    PRIVATE
        ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
        drvgpu
        OpenCL::OpenCL
)

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)
add_library(GPUWorkLib::${MODULE_NAME} ALIAS ${MODULE_NAME})

install(TARGETS ${MODULE_NAME} EXPORT GPUWorkLibTargets
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/${MODULE_NAME}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

message(STATUS "  ${MODULE_NAME}: configured")
message(STATUS "    - FIR: parallel per-sample, __constant coeffs")
message(STATUS "    - IIR: biquad cascade, parallel per-channel")
```

### Root CMakeLists.txt: добавить строку
```cmake
add_subdirectory(modules/filters)
```

---

## Python Bindings (python/py_filters.hpp)

Паттерн: аналогично `python/py_lch_farrow.hpp`

```cpp
#pragma once
/**
 * @file py_filters.hpp
 * @brief Python wrappers for FirFilter and IirFilter
 * Include AFTER GPUContext and vector_to_numpy definitions.
 */

#include "filters/fir_filter.hpp"
#include "filters/iir_filter.hpp"

// ============================================================================
// PyFirFilter
// ============================================================================
class PyFirFilter {
public:
    explicit PyFirFilter(GPUContext& ctx)
        : ctx_(ctx), fir_(ctx.backend()) {}

    void load_config(const std::string& path) { fir_.LoadConfig(path); }
    void set_coefficients(const std::vector<float>& c) { fir_.SetCoefficients(c); }

    py::array_t<std::complex<float>> process(
        py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> input)
    {
        auto buf = input.request();
        uint32_t channels, points;
        if (buf.ndim == 2) {
            channels = buf.shape[0]; points = buf.shape[1];
        } else {
            channels = 1; points = buf.shape[0];
        }
        size_t total = (size_t)channels * points;
        auto* ptr = static_cast<std::complex<float>*>(buf.ptr);

        // Upload to GPU
        cl_context cl_ctx = static_cast<cl_context>(ctx_.backend()->GetNativeContext());
        cl_int err;
        cl_mem input_buf = clCreateBuffer(cl_ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            total * sizeof(std::complex<float>),
            const_cast<std::complex<float>*>(ptr), &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("PyFirFilter: upload failed");

        drv_gpu_lib::InputData<cl_mem> result;
        { py::gil_scoped_release release;
          result = fir_.Process(input_buf, channels, points); }
        clReleaseMemObject(input_buf);

        // Readback
        std::vector<std::complex<float>> data(total);
        clEnqueueReadBuffer(ctx_.queue(), result.data, CL_TRUE, 0,
            total * sizeof(std::complex<float>), data.data(), 0, nullptr, nullptr);
        clReleaseMemObject(result.data);

        if (channels <= 1) return vector_to_numpy(std::move(data));
        return vector_to_numpy_2d(std::move(data), channels, points);
    }

    py::list get_coefficients() const {
        py::list result;
        for (float c : fir_.GetCoefficients()) result.append(c);
        return result;
    }
    uint32_t num_taps() const { return fir_.GetNumTaps(); }

private:
    GPUContext& ctx_;
    filters::FirFilter fir_;
};

// ============================================================================
// PyIirFilter (аналогично, с set_sections)
// ============================================================================
class PyIirFilter { /* ... аналогично ... */ };

// ============================================================================
// Binding registration
// ============================================================================
inline void register_filters(py::module& m) {
    py::class_<PyFirFilter>(m, "FirFilter",
        "GPU FIR filter (OpenCL).\n\n"
        "Usage:\n"
        "  fir = gpuworklib.FirFilter(ctx)\n"
        "  fir.set_coefficients(scipy.signal.firwin(64, 0.1).tolist())\n"
        "  result = fir.process(signal)\n")
        .def(py::init<GPUContext&>(), py::arg("ctx"))
        .def("load_config", &PyFirFilter::load_config, py::arg("json_path"),
             "Load FIR coefficients from JSON file.")
        .def("set_coefficients", &PyFirFilter::set_coefficients,
             py::arg("coefficients"), "Set FIR coefficients (list of float).")
        .def("process", &PyFirFilter::process, py::arg("input"),
             "Apply FIR filter on GPU. Input: complex64 (points,) or (channels, points).")
        .def_property_readonly("num_taps", &PyFirFilter::num_taps)
        .def_property_readonly("coefficients", &PyFirFilter::get_coefficients)
        .def("__repr__", [](const PyFirFilter& self) {
            return "<FirFilter num_taps=" + std::to_string(self.num_taps()) + ">";
        });

    // IirFilter binding — аналогично
}
```

### gpu_worklib_bindings.cpp: добавить
```cpp
#include "py_filters.hpp"
// ...в теле модуля:
register_filters(m);
// ...в module doc добавить: "FirFilter", "IirFilter"
```

### python/CMakeLists.txt: добавить
```cmake
# В target_link_libraries:
filters

# В target_include_directories:
${CMAKE_SOURCE_DIR}/modules/filters/include
```

---

## Tests

### modules/filters/tests/all_test.hpp
```cpp
#pragma once
#include "test_fir_basic.hpp"
// #include "test_iir_basic.hpp"  // раскомментировать после IIR

inline void RunAllFilterTests(drv_gpu_lib::IBackend* backend) {
    RunFirBasicTests(backend);
    // RunIirBasicTests(backend);
}
```

### test_fir_basic.hpp — Plan
- Сигнал: 8 каналов, 4096 отсчётов, CW 100Hz + CW 5000Hz, `fs = 50kHz`
- FIR lowpass: 64 тапа, fc=0.1 (normalized), Hamming (хардкодные scipy коэффициенты)
- GPU Process vs ProcessCpu → max error < 1e-4
- Вывод через `console_output`

### src/main.cpp: добавить
```cpp
#include "modules/filters/tests/all_test.hpp"
// ...
RunAllFilterTests(backend);
```

---

## JSON Config Format

### fir_lowpass.json
```json
{
  "type": "fir",
  "description": "Low-pass FIR, fc=0.1, 64 taps, Hamming",
  "sample_rate": 1000000,
  "coefficients": [0.0008, 0.0012, ..., 0.0012, 0.0008]
}
```

### iir_butterworth.json
```json
{
  "type": "iir",
  "description": "Butterworth 4th order low-pass, fc=0.1",
  "sample_rate": 1000000,
  "sections": [
    {"b0": 0.0675, "b1": 0.1349, "b2": 0.0675, "a1": -1.1430, "a2": 0.4128},
    {"b0": 1.0000, "b1": 2.0000, "b2": 1.0000, "a1": -1.5529, "a2": 0.6562}
  ]
}
```

---

## Python Tests

### Python_test/test_filters_stage1.py — scipy→GPU
```python
"""Stage 1: scipy generates coefficients → pass to GPU FIR/IIR → validate"""
import scipy.signal as sig
import gpuworklib as gw
import numpy as np
import matplotlib.pyplot as plt

ctx = gw.GPUContext(0)
fir = gw.FirFilter(ctx)

# Design FIR
taps = sig.firwin(64, cutoff=0.1)
fir.set_coefficients(taps.tolist())

# Multi-channel test signal: low freq + high freq
channels, points, fs = 8, 4096, 50000
t = np.arange(points) / fs
signal = np.zeros((channels, points), dtype=np.complex64)
for ch in range(channels):
    signal[ch] = (np.cos(2*np.pi*100*t) + np.cos(2*np.pi*5000*t)).astype(np.float32)

# GPU filter
result_gpu = fir.process(signal)

# Reference (numpy)
result_ref = np.array([np.convolve(signal[ch].real, taps, 'full')[:points]
                       for ch in range(channels)], dtype=np.complex64)

# Assert
err = np.max(np.abs(result_gpu.real - result_ref))
print(f"Max error GPU vs NumPy: {err:.2e}")
assert err < 1e-3, f"FIR GPU error too large: {err}"

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# ... plot signal before/after, frequency response, filter taps
plt.tight_layout()
plt.savefig("Results/JSON/test_filters_stage1.png")
plt.show()
```

### Python_test/test_filters_stage3_ai.py — Groq AI agent
```python
"""Stage 3: Natural language → Groq → scipy coeffs → GPU filter → plot"""
# Базируется на шаблоне: Python_test/test_ai_fir_demo.py
# MODE = "groq"  # использовать Groq API

# Запрос к Groq: "Design a low-pass FIR filter, cutoff 1kHz, sample rate 50kHz, 64 taps"
# Groq возвращает scipy код → exec → получаем taps
# → fir.set_coefficients(taps)
# → GPU filter
# → 4-panel matplotlib plot: signal before/after, freq response, pole-zero, taps
```

---

## Implementation Sequence

| # | Тип | Файл | Описание |
|---|-----|------|----------|
| 1 | C++ | `types/filter_params.hpp` | BiquadSection, FirParams, IirParams, FilterConfig |
| 2 | C++ | `types/filter_types.hpp` | FilterResult (если нужен) |
| 3 | C++ | `types/filter_modes.hpp` | enum FilterPrecision |
| 4 | C++ | `kernels/fir_kernels.hpp` | inline FIR OpenCL kernel |
| 5 | C++ | `filters/fir_filter.hpp` | FirFilter class declaration |
| 6 | C++ | `src/fir_filter.cpp` | FirFilter implementation |
| 7 | C++ | `filters/fir_filter_rocm.hpp` | ROCm stub |
| 8 | C++ | `tests/test_fir_basic.hpp` | FIR unit test |
| 9 | C++ | `tests/all_test.hpp` | Test entry point |
| 10 | C++ | `tests/README.md` | Test docs |
| 11 | C++ | `CMakeLists.txt` | Module build config |
| 12 | C++ | Root `CMakeLists.txt` | add_subdirectory(modules/filters) |
| 13 | C++ | `src/main.cpp` | RunAllFilterTests call |
| 14 | C++ | `kernels/iir_kernels.hpp` | inline IIR OpenCL kernel |
| 15 | C++ | `filters/iir_filter.hpp` | IirFilter class declaration |
| 16 | C++ | `src/iir_filter.cpp` | IirFilter implementation |
| 17 | C++ | `filters/iir_filter_rocm.hpp` | ROCm stub |
| 18 | C++ | `tests/test_iir_basic.hpp` | IIR unit test |
| 19 | Py | `python/py_filters.hpp` | PyFirFilter + PyIirFilter + register_filters |
| 20 | Py | `python/gpu_worklib_bindings.cpp` | #include + register call |
| 21 | Py | `python/CMakeLists.txt` | Link + include dirs |
| 22 | Py | `Python_test/test_filters_stage1.py` | scipy→GPU validation |
| 23 | Py | `Python_test/test_filters_stage3_ai.py` | Groq AI agent |
| 24 | MB | `MemoryBank/specs/filters_module.md` | Обновить спеку |

---

## Critical Files to Modify

| Файл | Действие |
|------|----------|
| `modules/filters/` | ⭐ Создать весь модуль |
| `CMakeLists.txt` (root) | Добавить `add_subdirectory(modules/filters)` |
| `src/main.cpp` | Добавить тест |
| `python/py_filters.hpp` | ⭐ Создать биндинг |
| `python/gpu_worklib_bindings.cpp` | Добавить include + register |
| `python/CMakeLists.txt` | Линковать filters |
| `MemoryBank/specs/Precpectiva/filters_module.md` | Обновить статус |

---

## Verification Checklist

- [ ] `cmake --build . --target filters` — OK (без ошибок)
- [ ] `main.exe` → `RunAllFilterTests` — GPU FIR vs CPU ref < 1e-4
- [ ] `pytest Python_test/test_filters_stage1.py` — PASS (scipy→GPU < 1e-3)
- [ ] IIR секция: 4-й порядок Butterworth, GPU vs scipy.sosfilt < 1e-4
- [ ] `python test_filters_stage3_ai.py` — Groq генерирует, GPU фильтрует, plot отображается

---

## Out of Scope (Post-MVP)

| Фича | Когда |
|------|-------|
| Overlap-Save/Overlap-Add | После стабильного FIR |
| Адаптивные LMS/NLMS/RLS | Отдельная задача |
| Полифазные фильтры/децимация | Отдельная задача |
| Stage 2: text→kernel кэш | После Stage 1 работает |
| ROCm полная реализация | После AMD GPU |

---

*План создан: 2026-02-18*
*Одобрен: 2026-02-18*
*Автор: Кодо (AI Assistant)*
