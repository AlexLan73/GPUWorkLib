# COMPLETED — Завершённые задачи

> **Обновлено**: 2026-02-23 (Task_09 added)
> История выполненных тасок — см. git log, Doc/, changelog/

---

*Фокус: ROCm*

## Task_00 — ROCmBackend + rocm_core (2026-02-23)
- ROCmBackend : IBackend (DrvGPU/backends/rocm/)
- rocm_core (hipDevice, hipCtx, hipStream)
- DrvGPU::CreateBackend(ROCm) — без throw
- test_rocm_backend.hpp (init, allocate, memcpy, sync)

## Task_002 — FFTProcessorROCm + StatisticsProcessorROCm (2026-02-23)
- FFTProcessorROCm: hipFFT + hiprtc kernels (pad, mag/phase)
- test_fft_processor_rocm.hpp (6 тестов)
- StatisticsProcessorROCm (ROCm only): mean, median, variance, std
  - HIP kernels: compute_magnitudes, mean_reduce, welford_stats
  - rocPRIM radix_sort для медианы
  - test_statistics_rocm.hpp (6 тестов)

## Task_03 — SpectrumProcessorROCm (2026-02-23)
- SpectrumProcessorROCm: hipFFT + hiprtc post-kernels
- AllMaximaPipelineROCm (find_peaks, parabolic_interp, refine)
- fft_kernel_sources_rocm.hpp, all_maxima_kernel_sources_rocm.hpp
- test_spectrum_maxima_rocm.hpp (6 тестов: ONE_PEAK, TWO_PEAKS, FindAllMaxima, Batch, vs OpenCL)

## Task_05 — LchFarrowROCm (2026-02-23)
- LchFarrowROCm: fractional delay processor (Lagrange 48x5) на HIP
- lch_farrow_kernels_rocm.hpp (Philox PRNG + lch_farrow_delay kernel)
- lch_farrow_rocm.hpp/cpp (Process, ProcessFromCPU, ProcessCpu)
- test_lch_farrow_rocm.hpp (4 теста: zero, integer, fractional, multi-antenna)
- CMakeLists.txt: условная компиляция ROCm

## Task_06 — FirFilterROCm + IirFilterROCm (2026-02-23)
- FirFilterROCm: HIP direct-form FIR convolution (hiprtc + hipModuleLaunchKernel)
  - fir_kernels_rocm.hpp (float2_t struct, 1D grid, extern "C" __global__)
  - fir_filter_rocm.hpp/cpp (Process, ProcessFromCPU, ProcessCpu)
- IirFilterROCm: HIP biquad cascade DFII-T (hiprtc + hipModuleLaunchKernel)
  - iir_kernels_rocm.hpp (float2_t struct, 1D grid one-thread-per-channel)
  - iir_filter_rocm.hpp/cpp (Process, ProcessFromCPU, ProcessCpu)
- test_filters_rocm.hpp (6 тестов: FIR basic/large/gpu_ptr, IIR basic/multi-sec/gpu_ptr)
- CMakeLists.txt: условная компиляция ROCm (ROCM_ENABLED + hip::host)
- all_test.hpp: включён ROCm test (закомментирован для Windows)
- Windows build: ✅ компилируется (stubs)

## Task_07 — FormSignalGeneratorROCm (2026-02-23)
- FormSignalGeneratorROCm: HIP port of multi-channel getX signal generator
  - form_signal_kernels_rocm.hpp (Philox PRNG + Box-Muller + getX formula)
  - form_signal_generator_rocm.hpp (full class with #if ENABLE_ROCM + stub)
  - form_signal_generator_rocm.cpp (hiprtc compile + hipModuleLaunchKernel)
- API: GenerateInputData() -> InputData<void*>, GenerateToCpu() -> vector<vector<complex>>
- test_form_signal_rocm.hpp (6 тестов: no_noise, window, multi_channel, noise_stats, chirp, gpu_ptr)
- CMakeLists.txt: условная компиляция ROCm (ROCM_ENABLED + hip::host + hiprtc)
- all_test.hpp: включён ROCm test (закомментирован для Windows)
- Windows build: ✅ компилируется (stubs)

## Task_08 — HeterodyneProcessorROCm (2026-02-23)
- HeterodyneProcessorROCm: HIP port of LFM dechirp processor
  - heterodyne_kernels_rocm.hpp (dechirp_multiply + dechirp_correct HIP kernels)
  - heterodyne_processor_rocm.hpp (full class with #if ENABLE_ROCM + stub)
  - heterodyne_processor_rocm.cpp (hiprtc compile + hipModuleLaunchKernel)
- API: Dechirp(), Correct(), DechirpFromGPU(), DechirpWithGPURef()
- Optimizations preserved: OPT-1 (kernel cache), OPT-2 (buffer reuse), OPT-3 (GPU ref), OPT-5 (1D), OPT-6 (precompute phase)
- HeterodyneDechirp facade updated: BackendType::ROCm support, ProcessExternal ROCm path
- test_heterodyne_rocm.hpp (6 тестов: single_ant, 5_antennas, correction, full_pipeline, gpu_ptr, random_delays)
- CMakeLists.txt: условная компиляция ROCm (ROCM_ENABLED + hip::host + hiprtc)
- all_test.hpp: включён ROCm test (закомментирован для Windows)
- Windows build: ✅ компилируется (stubs)

## Task_09 — ZeroCopy + HybridBackend (2026-02-23)
- **Пункт 11: ZeroCopy Bridge (OpenCL ↔ ROCm)**
  - `DrvGPU/backends/opencl/opencl_export.hpp` — ExportClBufferToFd(), ExportClBufferToGpuVA(), DetectBestZeroCopyMethod()
    - Поддержка 3 методов: DMA-BUF, AMD GPU VA, SVM
    - ZeroCopyMethod enum + авто-определение лучшего метода
  - `DrvGPU/backends/rocm/zero_copy_bridge.hpp` — ZeroCopyBridge class (#if ENABLE_ROCM + Windows stub)
  - `DrvGPU/backends/rocm/zero_copy_bridge.cpp` — ImportFromDmaBuf(), ImportFromGpuVA(), ImportFromOpenCl() (universal)
  - `DrvGPU/tests/test_zero_copy.hpp` — 6 тестов (detect, export, import, data_integrity, lifecycle)
- **Пункт 12: HybridBackend (OPENCLandROCm)**
  - `DrvGPU/backends/hybrid/hybrid_backend.hpp` — HybridBackend : IBackend (#if ENABLE_ROCM + Windows stub)
    - Variant A (wrapper): OpenCLBackend + ROCmBackend
    - GetOpenCL() / GetROCm() — доступ к sub-backends
    - CreateZeroCopyBridge() — авто-ZeroCopy
    - SyncBeforeZeroCopy() / SyncAfterZeroCopy()
  - `DrvGPU/backends/hybrid/hybrid_backend.cpp` — полная реализация
  - `DrvGPU/tests/test_hybrid_backend.hpp` — 6 тестов (init, device_info, cl_alloc, rocm_alloc, zero_copy, sync)
- **Интеграция (частичная — доделать в следующей сессии!)**
  - ✅ `DrvGPU/src/drv_gpu.cpp` — CreateBackend(OPENCLandROCm) → HybridBackend
  - ⚠️ `DrvGPU/tests/all_test.hpp` — НЕ обновлён
  - ⚠️ `DrvGPU/CMakeLists.txt` — НЕ обновлён
