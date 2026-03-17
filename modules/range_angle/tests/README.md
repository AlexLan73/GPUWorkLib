# range_angle — Tests

## Overview

Tests for the `RangeAngleProcessor` module (ROCm-only, Ref03 architecture).

All test files use `#if ENABLE_ROCM` guards and are included via `all_test.hpp`.

## Files

| File | Description |
|------|-------------|
| `all_test.hpp` | Entry point — included from `src/main.cpp` |
| `test_range_angle_basic.hpp` | Basic pipeline test: construct, SetParams, Process stub |
| `test_range_angle_benchmark.hpp` | Benchmark: measure pipeline latency with GPUProfiler |

## Running

```bash
# Single module
./GPUWorkLib range_angle

# All modules
./GPUWorkLib all
```

## Status

Current state: **STUB** — all Op Execute() methods are empty (TODO).
Tests verify construction, parameter setting, and pipeline call without crash.

## TODO

- [ ] Implement `dechirp_window_kernel` (hiprtc)
- [ ] Implement `RangeFftOp::Execute` (hipfftExecC2C batched)
- [ ] Implement `TransposeOp::Execute` (tiled kernel)
- [ ] Implement `BeamFftOp::Execute` (2-D batched FFT + fftshift)
- [ ] Implement `PeakSearchOp::Execute` (GPU max reduction)
- [ ] Add Python tests in `Python_test/range_angle/`
