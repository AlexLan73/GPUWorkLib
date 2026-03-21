---
name: run-gpu-tests
description: Runs GPU test suites (C++ and Python) for GPUWorkLib with GPU detection. Skips clFFT tests on AMD, ROCm tests on NVIDIA. Use when the user asks to run tests, execute test suite, or validate the project.
---

# Run GPU Tests

Orchestrates C++ and Python tests for GPUWorkLib. Respects GPU platform: AMD skips clFFT, NVIDIA skips ROCm.

## Поведение
- **Не спрашивать подтверждение** перед запуском тестов — выполнять сразу
- **Спрашивать** только при ошибке, проблеме или неясности

## Trigger

Apply when user says: run tests, run all tests, test module X, run agent tests, validate build.

**Не спрашивать подтверждения** перед запуском — выполнять сразу. Спрашивать только при ошибке или неясности.

## Workflow

### 1. Detect GPU (both methods)

**Build-time** (if CMake preset used):
- Check `build/CMakeCache.txt` for `TYPE_GPU` or `ENABLE_ROCM`
- AMD preset → ENABLE_ROCM=ON, skip clFFT-linked tests
- NVIDIA preset → ENABLE_ROCM=OFF, skip ROCm tests

**Run-time**:
```bash
# AMD if rocminfo succeeds
rocminfo 2>/dev/null | grep -q "Marketing" && GPU=AMD

# NVIDIA if nvidia-smi succeeds
nvidia-smi &>/dev/null && GPU=NVIDIA
```

### 2. What to run

| User intent | Command | Meaning |
|-------------|---------|---------|
| All tests | `./build/GPUWorkLib all` | All modules in order |
| One module | `./build/GPUWorkLib fft_processor` | Single module |
| From file | `./build/GPUWorkLib --file config/tests_to_run.txt` | Modules from file, order preserved |

### 3. Module order (all)

Read from `config/tests_order.txt` or use default:

1. drvgpu
2. fft_processor
3. statistics
4. vector_algebra
5. fft_maxima
6. filters
7. signal_generators
8. lch_farrow
9. heterodyne

### 4. Skip rules

| GPU | Skip |
|-----|------|
| AMD | C++ tests using clFFT (fft_processor OpenCL, fft_maxima OpenCL) |
| NVIDIA | C++ and Python tests using ROCm (statistics, vector_algebra, heterodyne ROCm, etc.) |

### 5. Execute

**One command** (C++ + Python):
```bash
./scripts/run_agent_tests.sh all
# or: ./scripts/run_agent_tests.sh fft_processor
# or: ./scripts/run_agent_tests.sh --file config/tests_to_run.txt
```

**Or step by step**:
```bash
cmake --build build --target GPUWorkLib
./build/GPUWorkLib all   # or module / --file path
python scripts/run_agent_tests.py all   # or module / --file path
```

### 6. Preserve artifacts

- C++ profiling: `Results/Profiler/` (from configGPU.json is_prof)
- Python plots: `Results/Plots/{module}/`
- Do not delete these during test run

## File format (--file)

One module per line, comments with #:

```
# Custom test order
drvgpu
fft_processor
statistics
```

## Module → C++ namespace

| config name | C++ namespace |
|-------------|---------------|
| drvgpu | drvgpu_all_test::run() |
| fft_processor | fft_processor_all_test::run() |
| statistics | statistics_all_test::run() |
| vector_algebra | vector_algebra_all_test::run() |
| fft_maxima | fft_maxima_all_test::run() |
| filters | filters_all_test::run() |
| signal_generators | signal_generators_all_test::run() |
| lch_farrow | lch_farrow_all_test::run() |
| heterodyne | heterodyne_all_test::run() |

## Python test paths

| module | path |
|--------|------|
| drvgpu | (no Python) |
| fft_processor | (no direct test dir) |
| fft_maxima | Python_test/fft_maxima/ |
| statistics | Python_test/statistics/ |
| vector_algebra | Python_test/vector_algebra/ |
| filters | Python_test/filters/ |
| signal_generators | Python_test/signal_generators/ |
| lch_farrow | Python_test/lch_farrow/ |
| heterodyne | Python_test/heterodyne/ |

Run: `python run_tests.py -m {module}/ -v`
