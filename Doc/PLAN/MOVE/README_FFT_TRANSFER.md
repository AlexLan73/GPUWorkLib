# FFT Pipeline Transfer Documentation

## Overview

This document describes the FFT processing pipeline for transfer to another project.
The code is proven to work and includes:
- **2 GPU kernels** (padding_kernel + post_kernel)
- **clFFT integration** for FFT computation
- **Batch processing** with automatic memory management
- **Event chain** for maximum GPU performance

---

## Architecture

```
INPUT SIGNAL (beam_count * count_points)
            |
            v
+---------------------------+
|   PADDING KERNEL          |  <- GPU Kernel #1
|   count_points -> nFFT    |
|   (zero-padding)          |
+---------------------------+
            |
            v (event_padding)
+---------------------------+
|   clFFT FORWARD           |  <- clFFT Library
|   nFFT-point FFT          |
|   (per beam, batched)     |
+---------------------------+
            |
            v (event_fft)
+---------------------------+
|   POST KERNEL (Unified)   |  <- GPU Kernel #2
|   - magnitude calculation |
|   - top-N maxima search   |
|   - phase calculation     |
|   - parabolic interpolation|
+---------------------------+
            |
            v (event_post)
+---------------------------+
|   READ RESULTS            |  <- CPU
|   MaxValue[] -> FFTResult |
+---------------------------+
```

---

## Key Files

### Core FFT Processor
| File | Description |
|------|-------------|
| `antenna_fft_proc_max.h` | Header with class declaration |
| `antenna_fft_proc_max.cpp` | Implementation (~1700 lines) |
| `antenna_fft_params.h` | Data structures (params, results) |

### OpenCL Manager Dependencies
| File | Description |
|------|-------------|
| `opencl_compute_engine.hpp` | OpenCL initialization singleton |
| `opencl_core.hpp` | Context, device, queue management |
| `command_queue_pool.hpp` | Queue pool for parallel execution |
| `gpu_memory_buffer.hpp` | GPU buffer abstraction |
| `kernel_program.hpp` | Kernel compilation helper |
| `memory_type.hpp` | Memory type enums |

---

## GPU Kernels (CRITICAL!)

### Kernel #1: padding_kernel

```opencl
__kernel void padding_kernel(
    __global const float2* input,    // Full input buffer (all beams)
    __global float2* output,         // Output: batch_beam_count * nFFT
    uint batch_beam_count,           // Beams in current batch
    uint count_points,               // Points per beam
    uint nFFT,                       // FFT size
    uint beam_offset                 // Beam offset for batch processing
)
```

**Purpose:**
- Copy `count_points` from input to output
- Zero-pad remaining `nFFT - count_points` elements
- Support batch processing via `beam_offset` (NO DATA COPY!)

### Kernel #2: post_kernel (Unified)

```opencl
typedef struct {
    uint index;
    float real;               // Real part
    float imag;               // Imaginary part
    float magnitude;
    float phase;
    float freq_offset;        // Parabolic interpolation offset
    float refined_frequency;  // Refined frequency in Hz
    uint pad;                 // Alignment to 32 bytes
} MaxValue;

__kernel void post_kernel(
    __global const float2* fft_output,     // FFT result
    __global MaxValue* maxima_output,      // Output maxima
    uint beam_count,
    uint nFFT,
    uint search_range,                     // Points to analyze
    uint max_peaks_count,                  // Number of peaks (3-5)
    float sample_rate                      // Sample rate (12 MHz default)
)
```

**Purpose:**
- Calculate magnitude: `sqrt(re^2 + im^2)`
- Find top-N maxima using parallel reduction
- Calculate phase: `atan2(im, re) * 180/PI`
- Parabolic interpolation for peak #0 frequency refinement

---

## Key Methods

### ProcessNew() - Auto Strategy Selection
```cpp
AntennaFFTResult ProcessNew(cl_mem input_signal);
```
- Estimates required memory
- If memory OK -> calls `Process()` (single batch)
- If memory insufficient -> calls `ProcessWithBatching()`

### ProcessWithBatching() - Batch Processing
```cpp
AntennaFFTResult ProcessWithBatching(cl_mem input_signal);
```
- Calculates batch size (~22% of beams)
- Creates/reuses cached buffers and FFT plan
- Processes batches sequentially with profiling

### ProcessBatch() - Single Batch
```cpp
std::vector<FFTResult> ProcessBatch(
    cl_mem input_signal,
    size_t start_beam,
    size_t num_beams,
    cl_command_queue batch_queue,
    cl_event* completion_event,
    BatchProfilingData* out_profiling);
```
- EVENT CHAIN: padding -> FFT -> post_kernel
- Reads MaxValue results directly from GPU
- No intermediate CPU transfers!

---

## nFFT Calculation

```cpp
size_t CalculateNFFT(size_t count_points) {
    // Round up to power of 2
    if (!IsPowerOf2(count_points)) {
        count_points = NextPowerOf2(count_points);
    }
    // Multiply by 2
    return count_points * 2;
}
```

Example: `count_points = 103680` -> `nFFT = 262144`

---

## Data Structures

### AntennaFFTParams (Input)
```cpp
struct AntennaFFTParams {
    size_t beam_count;           // Number of beams
    size_t count_points;         // Input points per beam
    size_t out_count_points_fft; // Output FFT points
    size_t max_peaks_count;      // Number of peaks (3-5)
    std::string task_id;         // Task identifier
    std::string module_name;     // Module name
};
```

### FFTMaxResult (Single Peak)
```cpp
struct FFTMaxResult {
    size_t index_point;  // Spectrum index
    float real;          // Re
    float imag;          // Im
    float amplitude;     // Magnitude
    float phase;         // Phase in degrees
};
```

### FFTResult (Single Beam)
```cpp
struct FFTResult {
    size_t v_fft;                          // FFT size
    std::vector<FFTMaxResult> max_values;  // Top-N peaks
    float freq_offset;                     // Parabolic offset
    float refined_frequency;               // Refined frequency Hz
    std::string task_id;
    std::string module_name;
};
```

---

## Usage Example

```cpp
// 1. Initialize OpenCL (once)
ManagerOpenCL::OpenCLComputeEngine::Initialize(ManagerOpenCL::DeviceType::GPU);

// 2. Create params
AntennaFFTParams params(
    5,       // beam_count
    103680,  // count_points
    1000,    // out_count_points_fft
    3        // max_peaks_count
);

// 3. Create processor
AntennaFFTProcMax processor(params);

// 4. Prepare input signal on GPU
cl_mem signal_gpu = ...; // Your signal data

// 5. Process
AntennaFFTResult result = processor.ProcessNew(signal_gpu);

// 6. Use results
for (const auto& beam_result : result.results) {
    for (const auto& peak : beam_result.max_values) {
        std::cout << "Peak at index " << peak.index_point
                  << ", amplitude=" << peak.amplitude
                  << ", phase=" << peak.phase << " deg\n";
    }
}
```

---

## Dependencies

- **OpenCL 1.2+**
- **clFFT library** (AMD clFFT)
- **C++14 or later**

---

## Performance Features

1. **Event Chain** - All GPU ops linked via events
2. **Buffer Caching** - Reuse buffers across calls
3. **FFT Plan Caching** - Reuse clFFT plans
4. **No Data Copy** - padding_kernel reads directly from input via offset
5. **Unified Post-Kernel** - Single kernel for magnitude + max + phase

---

*Last updated: 2026-02-04*
*Source: LCH-Farrow01/great-heyrovsky*
