# strategies module tests

## Test files

| File | Description | Namespace |
|------|-------------|-----------|
| `test_strategies_pipeline.hpp` | Full pipeline test: signal gen -> GEMM -> Window+FFT -> Step2.1/2.2/2.3 | `test_strategies` |
| `test_strategies_step_profiling.hpp` | GPU profiling per pipeline step (hipEvent + GPUProfiler) | `test_strategies_profiling` |
| `all_test.hpp` | Entry point, includes all test files | `test_strategies_all` |

## How to run

From `src/main.cpp`:
```cpp
#include "modules/strategies/tests/all_test.hpp"
test_strategies_all::run_all(backend);
```

## Pipeline test parameters

- 5 antennas, 8000 points
- fs = 12 MHz, f0 = 2 MHz
- Delay-and-sum matrix W (tau_step = 100 us)
- No noise (noise_amplitude = 0)

## Profiling test parameters

- 10 antennas, 16384 points
- fs = 12 MHz, f0 = 2 MHz
- Warmup: 10 iterations, Measurements: 20 iterations
- Timing: hipEvent (GPU hardware timer)
- Output: GPUProfiler.PrintReport() only (no disk export)

### Measured steps

| Step | What is measured |
|------|------------------|
| `step1_GEMM` | hipBLAS CGEMM (W * S) |
| `step2_WindowFFT` | hamming_pad_fused + hipFFT batch + compute_magnitudes |
| `step3_OneMax` | one_max_no_phase kernel (per-beam max + parabola) |
| `step4_MinMax` | global_minmax kernel (per-beam min+max) |
| `step5_FullProcess` | Full process() pipeline end-to-end |
