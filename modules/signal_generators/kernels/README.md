# Signal Generators — Kernels

OpenCL kernel files for the signal_generators module.

## Structure

```
kernels/
├── prng.cl                   # Shared PRNG: Philox-2x32-10 + Box-Muller
├── cw_kernel.cl              # CW (sinusoid) generator kernel
├── lfm_kernel.cl             # LFM (chirp) generator kernel
├── noise_kernel.cl           # Noise generator (Gaussian + White), requires prng.cl
├── form_signal.cl            # FormSignal (getX formula + noise), requires prng.cl
├── delayed_form_signal.cl    # Fractional delay (Lagrange 5-point + noise), requires prng.cl
├── lfm_analytical_delay.cl   # LFM with per-antenna analytical delay
├── bin/                      # Compiled kernel binaries (on-disk cache)
│   ├── <name>_opencl.bin     # OpenCL compiled binary
│   └── <name>_opencl_XX.bin  # Old versions (_00, _01, ...)
├── manifest.json             # Kernel index (names, comments, dates)
├── <name>.cl                 # Saved kernel source (FormScriptGenerator)
├── <name>_XX.cl              # Old versions
└── README.md                 # This file
```

## Loading Kernels

All kernels are loaded at runtime via `kernel_loader.hpp`:

```cpp
#include "kernel_loader.hpp"

// Simple kernel (no PRNG)
std::string src = signal_gen::LoadKernelFile("cw_kernel.cl");

// Kernel with PRNG (prng.cl prepended automatically)
std::string src = signal_gen::LoadKernelWithPrng("form_signal.cl");
```

Path is set by CMake via `SIGNAL_GENERATORS_KERNELS_DIR`.

## On-disk Kernel Cache

**FormScriptGenerator** supports saving/loading compiled kernels:

### Save
```cpp
gen.SaveKernel("my_signal", "CW 1MHz 8 channels");
// Creates: my_signal.cl + bin/my_signal_opencl.bin
// Updates: manifest.json
```

### Load
```cpp
gen.LoadKernel("my_signal");
// Loads binary (fast) or source (compile + cache binary)
```

### Versioning
When saving with an existing name, old files are renamed:
- `my_signal.cl` -> `my_signal_00.cl`
- `bin/my_signal_opencl.bin` -> `bin/my_signal_opencl_00.bin`

Next collision: `_01`, `_02`, etc.

### manifest.json
```json
{
  "version": 1,
  "kernels": [
    {
      "name": "my_signal",
      "comment": "CW 1MHz 8 channels",
      "created": "2026-02-17T14:30:00",
      "params": "fs=12000000,f0=1000000,a=1,an=0,antennas=8,points=4096",
      "backend": "opencl"
    }
  ]
}
```

## Python Usage

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
gen = gpuworklib.FormScriptGenerator(ctx)

gen.set_params(f0=1e6, antennas=8, points=4096)
gen.compile()
gen.save_kernel("my_signal", "CW 1MHz 8ch")

# Later: fast load
gen.load_kernel("my_signal")
data = gen.generate()

# List available kernels
print(gen.list_kernels())
```

---

*Updated: 2026-02-18*
