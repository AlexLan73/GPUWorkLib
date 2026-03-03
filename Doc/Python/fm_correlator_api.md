# FM Correlator — Python API

> Frequency-domain correlation for M-sequence phase modulation (ROCm)

## Quick Start

```python
import gpuworklib
import numpy as np

# Create context and correlator
ctx = gpuworklib.ROCmGPUContext(0)
corr = gpuworklib.FMCorrelatorROCm(ctx)

# Mode 1: Parameters only (no data transfer)
corr.set_params(fft_size=32768, num_shifts=32, num_signals=10)
corr.prepare_reference()
peaks = corr.run_test_pattern(shift_step=2)  # numpy [S, K, n_kg]

# Mode 2: External data
ref = corr.generate_msequence()
signals = np.stack([np.roll(ref, -s * 2) for s in range(10)])
corr.prepare_reference_from_data(ref)
peaks = corr.process(signals.astype(np.float32))
```

## API Reference

### Constructor

```python
corr = gpuworklib.FMCorrelatorROCm(ctx)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `ROCmGPUContext` | GPU context |

### Methods

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `set_params(...)` | fft_size, num_shifts, ... | void | Configure correlator |
| `generate_msequence(seed)` | uint32 | numpy [N] float | M-sequence for analysis |
| `prepare_reference()` | — | void | Internal M-seq generation + GPU upload |
| `prepare_reference_from_data(ref)` | numpy [N] float | void | External reference |
| `process(input_signals)` | numpy [S, N] float | numpy [S, K, n_kg] float | Correlate |
| `run_test_pattern(shift_step)` | int | numpy [S, K, n_kg] float | GPU-only test |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fft_size` | 32768 | FFT size (power of 2) |
| `num_shifts` | 32 | K — cyclic shifts |
| `num_signals` | 5 | S — input signals |
| `num_output_points` | 2000 | n_kg — first IFFT points |
| `polynomial` | 0x00400007 | LFSR polynomial (x^32+x^22+x^2+x+1, primitive) |
| `seed` | 0x12345678 | LFSR initial seed |

## Examples

### Autocorrelation

```python
corr.set_params(fft_size=4096, num_shifts=1, num_signals=1,
                num_output_points=200)
ref = corr.generate_msequence()
corr.prepare_reference_from_data(ref)
peaks = corr.process(ref)

snr = peaks[0, 0, 0] / np.max(peaks[0, 0, 1:])
print(f"Autocorrelation SNR: {snr:.1f}")  # > 10
```

### Shift Pattern Verification

```python
N, K, S = 4096, 10, 5
corr.set_params(fft_size=N, num_shifts=K, num_signals=S,
                num_output_points=200)
corr.prepare_reference()
peaks = corr.run_test_pattern(shift_step=2)

for s in range(S):
    for k in range(K):
        expected_pos = (s * 2 - k) % N
        if expected_pos < 200:
            assert np.argmax(peaks[s, k, :]) == expected_pos
```
