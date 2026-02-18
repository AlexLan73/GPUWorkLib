# Filters Module Tests

## test_fir_basic.hpp
- **Low-pass FIR**: 64 taps, fc=0.1 (normalized), Hamming window
- **Signal**: 8 channels, 4096 points, CW 100Hz + CW 5000Hz, fs=50kHz
- **Validation**: GPU `Process()` vs `ProcessCpu()`, max error < 1e-3
- **Coefficients**: Pre-computed from `scipy.signal.firwin(64, 0.1)`

## test_iir_basic.hpp
- **Butterworth 2nd order LP**: fc=0.1 (normalized), 1 biquad section
- **Signal**: Same as FIR test
- **Validation**: GPU `Process()` vs `ProcessCpu()`, max error < 1e-3
- **Coefficients**: Pre-computed from `scipy.signal.butter(2, 0.1, output='sos')`

## Running Tests

From `src/main.cpp`:
```cpp
#include "modules/filters/tests/all_test.hpp"
// ...
RunAllFilterTests(backend);
```
