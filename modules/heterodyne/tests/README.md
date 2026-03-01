# Heterodyne Module Tests

## Test List

| # | File | Test | Description |
|---|------|------|-------------|
| 1 | test_heterodyne_basic.hpp | run_test_single_antenna | Single antenna dechirp, delay=100us, expect f_beat=300kHz |
| 2 | test_heterodyne_basic.hpp | run_test_5_antennas_linear | 5 antennas, delays [100,200,300,400,500] us |
| 3 | test_heterodyne_basic.hpp | run_test_correction | dechirp_correct kernel, verify peak at DC after correction |
| 4 | test_heterodyne_pipeline.hpp | run_test_full_pipeline | Full Process() pipeline, f_beat + range validation |
| 5 | test_heterodyne_pipeline.hpp | run_test_process_external | ProcessExternal with cl_mem, verify external buffer not freed |

## Parameters

```
fs       = 12 MHz
B        = 1 MHz (f_start=0, f_end=1e6)
N        = 4000 points
T        = 333.33 us
mu       = 3e9 Hz/s
antennas = 5
```

## Expected Results

| Antenna | Delay us | f_beat Hz | Bin (N=4096) | Range m |
|---------|----------|-----------|--------------|---------|
| 0       | 100      | 300,000   | ~102         | 50.0    |
| 1       | 200      | 600,000   | ~205         | 100.0   |
| 2       | 300      | 900,000   | ~307         | 150.0   |
| 3       | 400      | 1,200,000 | ~410         | 200.0   |
| 4       | 500      | 1,500,000 | ~512         | 250.0   |

## Tolerances

- f_beat: +/- 10 kHz
- After correction: peak at bin 0-3

---

## Benchmark Tests (GpuBenchmarkBase)

| # | File | Class | Method | Стадии профилирования |
|---|------|-------|--------|----------------------|
| B1 | heterodyne_benchmark.hpp | HeterodyneDechirpBenchmark | Dechirp() | Upload_Rx, Upload_Ref, Kernel_Multiply, Download |
| B2 | heterodyne_benchmark.hpp | HeterodyneCorrectBenchmark | Correct() | Upload_DC, Upload_PhaseStep, Kernel_Correct, Download |
| B3 | heterodyne_benchmark_rocm.hpp | HeterodyneDechirpBenchmarkROCm | Dechirp() | Upload_Rx, Upload_Ref, Kernel_Multiply, Download |
| B4 | heterodyne_benchmark_rocm.hpp | HeterodyneCorrectBenchmarkROCm | Correct() | Upload_DC, Upload_PhaseStep, Kernel_Correct, Download |

### Параметры бенчмарка

```
num_antennas = 5
num_samples  = 4000
sample_rate  = 12 MHz
n_warmup = 5,  n_runs = 20
```

### Результаты

```
Results/Profiler/GPU_00_Heterodyne/         ← OpenCL
Results/Profiler/GPU_00_Heterodyne_ROCm/    ← ROCm
```

### Запуск

Раскомментировать в `all_test.hpp`:
```cpp
// test_heterodyne_benchmark::run();
// test_heterodyne_benchmark_rocm::run();
```
