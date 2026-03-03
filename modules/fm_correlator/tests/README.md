# FM Correlator Tests

## Test Files

| File | Description |
|------|-------------|
| `all_test.hpp` | Test registry — entry point for all tests |
| `test_fm_msequence.hpp` | LFSR M-sequence generator validation |
| `test_fm_basic.hpp` | Functional tests: autocorrelation, pipeline, shift pattern |
| `test_fm_benchmark_rocm.hpp` | Performance benchmark (hipEvent GPU timing) |

## Test Details

### test_fm_msequence
- All values are `{+1.0, -1.0}`
- ~50% ones / ~50% minus ones
- Different seeds → different sequences
- Same seed → identical sequence

### test_fm_basic

| Test | Parameters | Verification |
|------|-----------|-------------|
| Autocorrelation | N=4096, K=1, S=1 | SNR > 10 (peak[0] >> noise) |
| Basic pipeline | N=1024, K=4, S=2 | Pipeline completes, correct output size |
| Full pipeline | N=32768, K=32, S=5 | Pipeline completes, 320000 floats |
| Shift pattern | N=4096, K=10, S=5 | Peak positions match expected, CPU≈GPU (atol=1e-4) |

### test_fm_benchmark_rocm
- Warmup: 3 iterations
- Measurement: 20 iterations with `hipEvent` (GPU hardware timer)
- Metrics: avg/min/max ms
- GPUProfiler: `SetGPUInfo → Start → Record → Stop → PrintReport → Export`
- Parametric sweep: fft_size × num_shifts × num_signals

## Running Tests

```bash
./GPUWorkLib fm_correlator
```

Or from `config/tests_order.txt`:
```
fm_correlator
```
