# FFT Maxima Module

## Overview
High-performance FFT processing module with maxima search for antenna signal processing.

## Features
- clFFT with **pre/post callbacks** for zero-copy data processing
- Automatic batch processing for large datasets (gigabytes)
- Multi-GPU support via DrvGPU backend
- Parabolic interpolation for frequency refinement
- Built-in profiling

## Architecture

```
INPUT (CPU/GPU) → pre_callback → clFFT → post_callback → post_kernel → RESULTS
```

### Callbacks (clFFT integrated)
- **prepareDataPre**: Data preparation with zero-padding
- **processFFTPost**: fftshift + magnitude calculation

### Kernels
- **padding_kernel**: For batch processing (alternative to pre-callback)
- **post_kernel**: Unified kernel for maxima search + phase + interpolation

## Usage

```cpp
#include "antenna_fft_module.hpp"

// Initialize DrvGPU
DrvGPU gpu(BackendType::OPENCL, 0);
gpu.Initialize();

// Create module
auto module = std::make_shared<AntennaFFTModule>(&gpu.GetBackend());

// Set parameters
AntennaFFTParams params(5, 1024, 512, 3);  // 5 beams, 1024 points, 512 output, 3 peaks
module->SetParams(params);
module->Initialize();

// Process from CPU
std::vector<std::complex<float>> data(5 * 1024);
// ... fill data ...
auto result = module->ProcessFromCPU(data);

// Get results
for (const auto& beam : result.results) {
    std::cout << "Max amplitude: " << beam.max_values[0].amplitude << "\n";
}
```

## Dependencies
- DrvGPU (OpenCL backend)
- clFFT library
- OpenCL 1.2+

