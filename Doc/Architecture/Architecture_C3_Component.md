# C3 — Component Diagram

> **Project**: GPUWorkLib
> **Date**: 2026-02-23
> **Reference**: [c4model.com](https://c4model.com)
> **Level**: 3 (Component) — компоненты внутри каждого контейнера

---

## 1. DrvGPU — Component Diagram

```
┌─────────────────────────────── DrvGPU ──────────────────────────────────┐
│                                                                          │
│  ┌──────────────────────── Interface Layer ─────────────────────────┐   │
│  │  ┌─────────────┐  ┌───────────────────┐  ┌──────────────────┐   │   │
│  │  │ IBackend    │  │ IComputeModule    │  │ IMemoryBuffer    │   │   │
│  │  │ (abstract)  │  │ (abstract)        │  │ (abstract)       │   │   │
│  │  └──────┬──────┘  └───────────────────┘  └──────────────────┘   │   │
│  └─────────┼────────────────────────────────────────────────────────┘   │
│            │ implements                                                  │
│  ┌─────────┼──────────────── Backend Layer ─────────────────────────┐   │
│  │         ▼                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │ OpenCL       │  │ ROCm         │  │ Hybrid               │   │   │
│  │  │ Backend      │  │ Backend      │  │ Backend              │   │   │
│  │  │              │  │              │  │ (OpenCL + ROCm       │   │   │
│  │  │ OpenCLCore   │  │ ROCmCore     │  │  fallback)           │   │   │
│  │  │ CmdQueuePool │  │ ZeroCopy-    │  │                      │   │   │
│  │  │ Profiling    │  │ Bridge       │  │                      │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────── Memory Layer ────────────────────────────┐   │
│  │  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐     │   │
│  │  │ MemoryManager │  │ GPUBuffer<T> │  │ SVMBuffer        │     │   │
│  │  │               │  │ (RAII)       │  │ (shared virtual)  │     │   │
│  │  │ Allocate()    │  │ Write()      │  │                   │     │   │
│  │  │ Free()        │  │ Read()       │  │                   │     │   │
│  │  │ Statistics()  │  │ GetPtr()     │  │                   │     │   │
│  │  └───────────────┘  └──────────────┘  └──────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────── Services Layer ──────────────────────────┐   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐    │   │
│  │  │ GPUProfiler   │  │ ConsoleOutput │  │ BatchManager     │    │   │
│  │  │ (Singleton)   │  │ (Singleton)   │  │ (static util)    │    │   │
│  │  │               │  │               │  │                   │    │   │
│  │  │ Record()      │  │ Print()       │  │ CalcOptBatch()   │    │   │
│  │  │ PrintReport() │  │ PrintError()  │  │ CreateBatches()  │    │   │
│  │  │ ExportJSON()  │  │ PrintWarning()│  │                   │    │   │
│  │  │ ExportMD()    │  │ PrintDebug()  │  │                   │    │   │
│  │  │ SetGPUInfo()  │  │               │  │                   │    │   │
│  │  └───────────────┘  └───────────────┘  └──────────────────┘    │   │
│  │                                                                 │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐    │   │
│  │  │ KernelCache   │  │ FilterConfig  │  │ ServiceManager   │    │   │
│  │  │ Service       │  │ Service       │  │                   │    │   │
│  │  │               │  │               │  │ Register()        │    │   │
│  │  │ SaveBinary()  │  │ LoadJSON()    │  │ GetService<T>()   │    │   │
│  │  │ LoadBinary()  │  │ GetCoeffs()   │  │ StartAll()        │    │   │
│  │  └───────────────┘  └───────────────┘  └──────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────── Infra ──────────────┐  ┌─────── Config ───────────┐     │
│  │  ┌─────────┐  ┌──────────────┐  │  │  ┌──────────────────┐   │     │
│  │  │ Logger  │  │ Module       │  │  │  │ GPUConfig        │   │     │
│  │  │ (plog)  │  │ Registry     │  │  │  │ (configGPU.json) │   │     │
│  │  │         │  │              │  │  │  │                   │   │     │
│  │  │ Per-GPU │  │ Register()   │  │  │  │ device_index     │   │     │
│  │  │ logfiles│  │ GetModule()  │  │  │  │ backend_type     │   │     │
│  │  └─────────┘  └──────────────┘  │  │  │ memory_limit     │   │     │
│  └──────────────────────────────────┘  │  └──────────────────┘   │     │
│                                         └─────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Signal Generators — Components

```
┌──────────────────── Signal Generators ──────────────────────┐
│                                                               │
│  ┌─────────────────── Interface ─────────────────────┐       │
│  │  ISignalGenerator                                  │       │
│  │  ├── GenerateToCpu(system, out, size)              │       │
│  │  ├── GenerateToGpu(system, beam_count) → cl_mem    │       │
│  │  └── Kind() → SignalKind                           │       │
│  └────────────────────┬──────────────────────────────┘       │
│                       │ implements                            │
│   ┌───────────────────┼────────────────────────────┐         │
│   │                   │                            │         │
│   ▼                   ▼                            ▼         │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │CwGenerator │  │LfmGenerator│  │NoiseGenerator        │   │
│  │            │  │            │  │                      │   │
│  │ f0, phase  │  │ f_start    │  │ type: WHITE/GAUSSIAN │   │
│  │ amplitude  │  │ f_end      │  │ power, seed          │   │
│  │ freq_step  │  │ amplitude  │  │ (Philox + Box-Muller)│   │
│  └────────────┘  └────────────┘  └──────────────────────┘   │
│                                                               │
│  ┌─────────────────┐  ┌──────────────────────────────────┐   │
│  │FormSignal       │  │LfmConjugateGenerator             │   │
│  │Generator        │  │                                   │   │
│  │                 │  │ Генерация conj(LFM)               │   │
│  │ DSL Script →    │  │ для HeterodyneDechirp             │   │
│  │ OpenCL kernel   │  │ (кешируется в Heterodyne)         │   │
│  └─────────────────┘  └──────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ SignalGeneratorFactory (static)                       │    │
│  │                                                       │    │
│  │ Create(backend, request) → unique_ptr<ISignalGen>     │    │
│  │ CreateCw / CreateLfm / CreateNoise / CreateForm       │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ SignalService (CPU reference)                         │    │
│  │                                                       │    │
│  │ GenerateCpu(params, system) → vector<complex<float>>  │    │
│  │ GenerateToGpu(backend, params, system) → cl_mem       │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌────────────────────── Data Types ────────────────────┐    │
│  │ SignalKind: CW | LFM | NOISE | FORM_SIGNAL           │    │
│  │ SystemSampling: { sample_rate, length }               │    │
│  │ SignalRequest: { kind, system, variant<Params...> }   │    │
│  │ CwParams / LfmParams / NoiseParams / FormParams       │    │
│  └──────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. FFT Processor — Components

```
┌───────────────────── FFT Processor ──────────────────────────┐
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ FFTProcessor                                          │     │
│  │                                                       │     │
│  │ ProcessComplex(data, params) → vector<ComplexResult>   │     │
│  │ ProcessComplex(cl_mem, params) → vector<ComplexResult> │     │
│  │ ProcessMagPhase(data, params) → vector<MagPhaseResult>│     │
│  │ ProcessMagPhase(cl_mem, params) → vector<MagPhaseResult│     │
│  │                                                       │     │
│  │ Internal GPU Buffers:                                  │     │
│  │   pre_callback_userdata_ (32B header + data)          │     │
│  │   fft_input_  (nFFT * batch)                          │     │
│  │   fft_output_ (FFT result)                            │     │
│  │   mag_output_ (Magnitude)                             │     │
│  │   phase_output_ (Phase)                               │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  ┌────────────────── Data Types ─────────────────────────┐    │
│  │ FFTOutputMode: COMPLEX | MAG_PHASE | MAG_PHASE_FREQ   │    │
│  │ FFTProcessorParams:                                    │    │
│  │   { beam_count, n_point, sample_rate,                  │    │
│  │     output_mode, use_padding }                         │    │
│  │ FFTComplexResult: { spectrum: vector<complex<float>> } │    │
│  │ FFTMagPhaseResult:                                     │    │
│  │   { magnitude, phase, frequency_hz: vector<float> }    │    │
│  └────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌──────────────── GPU Pipeline ─────────────────────────┐    │
│  │  Input → [Pre-callback kernel] → [clFFT] →            │    │
│  │          → [Post-process kernel: Mag/Phase] → Output   │    │
│  └────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. FFT Maxima (SpectrumMaximaFinder) — Components

```
┌───────────────── FFT Maxima ──────────────────────────────────┐
│                                                                 │
│  ┌────────────────── Interface ──────────────────────────┐     │
│  │ ISpectrumProcessor                                     │     │
│  │  ├── Initialize(params)                                │     │
│  │  ├── ProcessFromCPU(data) → vector<SpectrumResult>     │     │
│  │  ├── ProcessFromGPU(gpu_data, ...) → vector<Result>    │     │
│  │  ├── ProcessBatch(data, start, count) → vector<Result> │     │
│  │  ├── FindAllMaximaFromCPU(...) → AllMaximaResult       │     │
│  │  └── FindAllMaximaFromGPUPipeline(...) → AllMaxResult   │     │
│  └────────────────────┬──────────────────────────────────┘     │
│                       │ implements                               │
│       ┌───────────────┼────────────────────┐                    │
│       ▼                                    ▼                    │
│  ┌──────────────────────┐  ┌──────────────────────────────┐    │
│  │ SpectrumProcessor    │  │ SpectrumProcessorROCm        │    │
│  │ OpenCL               │  │                              │    │
│  │                      │  │ hipFFT + ROCm kernels        │    │
│  │ clFFT + OpenCL       │  │                              │    │
│  │ kernels              │  │                              │    │
│  └──────────────────────┘  └──────────────────────────────┘    │
│                                                                 │
│  ┌────────────── AllMaxima Pipeline ─────────────────────┐     │
│  │  Stage 1: FFT (clFFT / hipFFT)                         │     │
│  │  Stage 2: Magnitude computation (GPU kernel)           │     │
│  │  Stage 3: Peak detection — threshold scan (GPU)        │     │
│  │  Stage 4: Peak compaction — stream compact (GPU)       │     │
│  │  Stage 5: Result readback to CPU                       │     │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────── Data Types ─────────────────────────┐     │
│  │ SpectrumMode: ONE_PEAK | TWO_PEAKS | ALL_MAXIMA        │     │
│  │ SpectrumParams: { antenna_count, n_point, nFFT,        │     │
│  │                   sample_rate, mode }                   │     │
│  │ SpectrumResult: { antenna_idx, peak_freq_hz, peak_bin, │     │
│  │                   peak_amplitude, peak_snr_db,          │     │
│  │                   second_peak_* (for TWO_PEAKS) }       │     │
│  │ AllMaximaResult: { peaks[], num_peaks, runtime_ms }     │     │
│  │ OutputDestination: CPU | GPU                            │     │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Filters — Components

```
┌───────────────────── Filters ──────────────────────────────────┐
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ FirFilter                                             │       │
│  │                                                       │       │
│  │ LoadConfig(json_path)                                 │       │
│  │ SetCoefficients(coeffs)                               │       │
│  │ Process(cl_mem, channels, points) → InputData<cl_mem> │       │
│  │ ProcessCpu(input, ch, pts) → vector<complex<float>>   │       │
│  │ GetNumTaps() → uint32_t                               │       │
│  │                                                       │       │
│  │ ⚡ Kernel: __constant (≤16000 taps)                   │       │
│  │           __global   (>16000 taps, fallback)          │       │
│  │ Formula: y[ch][n] = Σ h[k] * x[ch][n-k]              │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ IirFilter                                             │       │
│  │                                                       │       │
│  │ SetCoefficients(a_coeffs, b_coeffs)                   │       │
│  │ Process(cl_mem, channels, points) → InputData<cl_mem> │       │
│  │                                                       │       │
│  │ Formula: y[n] = Σ b[k]*x[n-k] - Σ a[k]*y[n-k]       │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────── ROCm variants ───────────────────┐       │
│  │ FirFilterROCm / IirFilterROCm                         │       │
│  │ (HIP kernels, same API)                               │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Heterodyne (LFM Dechirp) — Components

```
┌───────────────────── Heterodyne ──────────────────────────────────┐
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ HeterodyneDechirp (Facade)                                │      │
│  │                                                           │      │
│  │ SetParams(params)                                         │      │
│  │ Process(rx_data) → HeterodyneResult                       │      │
│  │ ProcessExternal(rx_gpu_ptr, params) → HeterodyneResult    │      │
│  │                                                           │      │
│  │ ┌─── Internal Pipeline ──────────────────────────────┐   │      │
│  │ │ 1. Generate conj(LFM) reference (cached, OPT-4)   │   │      │
│  │ │ 2. Dechirp multiply: s_dc = s_rx * conj(s_tx)     │   │      │
│  │ │ 3. FFT → spectrum                                  │   │      │
│  │ │ 4. Peak search → f_beat                            │   │      │
│  │ │ 5. Range = c * T * f_beat / (2 * B)                │   │      │
│  │ └────────────────────────────────────────────────────┘   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  ┌────────────────── Interface ──────────────────────────────┐     │
│  │ IHeterodyneProcessor                                       │     │
│  │  ├── Initialize(backend, params)                           │     │
│  │  ├── Dechirp(rx_buf, ref_buf, ...) → cl_mem                │     │
│  │  └── Cleanup()                                             │     │
│  └────────────────────┬──────────────────────────────────────┘     │
│                       │ implements                                   │
│       ┌───────────────┼─────────────────────┐                       │
│       ▼                                     ▼                       │
│  ┌──────────────────────┐  ┌──────────────────────────┐            │
│  │ HeterodyneProcessor  │  │ HeterodyneProcessorROCm  │            │
│  │ OpenCL               │  │                          │            │
│  └──────────────────────┘  └──────────────────────────┘            │
│                                                                     │
│  ┌────────────────── Data Types ──────────────────────────────┐    │
│  │ HeterodyneParams:                                           │    │
│  │   { f_start, f_end, sample_rate, num_samples, num_antennas  │    │
│  │     GetBandwidth(), GetDuration(), GetChirpRate() }         │    │
│  │ AntennaDechirpResult:                                       │    │
│  │   { antenna_idx, f_beat_hz, f_beat_bin,                     │    │
│  │     range_m, peak_amplitude, peak_snr_db }                  │    │
│  │ HeterodyneResult:                                           │    │
│  │   { success, antennas[], max_positions[], error_message }   │    │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. LCH Farrow (Fractional Delay) — Components

```
┌───────────────────── LCH Farrow ───────────────────────────────┐
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ LchFarrow                                             │       │
│  │                                                       │       │
│  │ SetDelays(delay_us[])                                 │       │
│  │ SetSampleRate(sample_rate)                            │       │
│  │ SetNoise(amplitude, norm_val, seed)                   │       │
│  │ LoadMatrix(json_path)                                 │       │
│  │ Process(cl_mem, antennas, points) → InputData<cl_mem> │       │
│  │ ProcessCpu(input, antennas, pts) → vector<vector<>>   │       │
│  │                                                       │       │
│  │ Algorithm: Lagrange 5-point polynomial interpolation  │       │
│  │ Matrix: 48 × 5 pre-computed coefficients              │       │
│  │ Per-antenna independent fractional delay              │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────── ROCm variant ────────────────────────┐       │
│  │ LchFarrowROCm (HIP kernels, same API)                │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Python Bindings — Components

```
┌───────────────── Python Bindings ──────────────────────────────┐
│                                                                  │
│  gpu_worklib.pyd (pybind11 module)                               │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────────────────────┐      │
│  │ GPUContext       │  │ PySignalGenerator                │      │
│  │                  │  │                                  │      │
│  │ __init__(idx=0)  │  │ generate_cw(freq, fs, len, ...) │      │
│  │ backend → ptr    │  │ generate_lfm(f0, f1, fs, ...)   │      │
│  │ queue → ptr      │  │ generate_noise(type, power, ..) │      │
│  │ device_name      │  │                                  │      │
│  └─────────────────┘  │ Returns: np.ndarray[complex64]   │      │
│                        └──────────────────────────────────┘      │
│                                                                  │
│  ┌─────────────────────┐  ┌──────────────────────────────┐      │
│  │ PyFFTProcessor       │  │ PyHeterodyneDechirp          │      │
│  │                      │  │                              │      │
│  │ process_complex()    │  │ set_params(f0, f1, fs, ...)  │      │
│  │ process_magphase()   │  │ process(rx_data) → dict      │      │
│  │                      │  │                              │      │
│  │ Returns: list[dict]  │  │ Returns: dict with           │      │
│  │  { spectrum, mag,    │  │  { success, antennas[],      │      │
│  │    phase, freq_hz }  │  │    f_beat, range_m, ... }    │      │
│  └─────────────────────┘  └──────────────────────────────┘      │
│                                                                  │
│  ┌─────────────────────┐  ┌──────────────────────────────┐      │
│  │ PyFilters            │  │ PyLchFarrow                  │      │
│  │                      │  │                              │      │
│  │ fir_process()        │  │ set_delays()                 │      │
│  │ iir_process()        │  │ process()                    │      │
│  │ load_config()        │  │ load_matrix()                │      │
│  └─────────────────────┘  └──────────────────────────────┘      │
│                                                                  │
│  Data Flow: np.ndarray → cl_mem → GPU compute → cl_mem → np     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. Сводная таблица всех компонентов

| Контейнер | Компонент | Тип | Файлы |
|-----------|-----------|-----|-------|
| **DrvGPU** | IBackend | Interface | `interface/i_backend.hpp` |
| | OpenCLBackend | Class | `backends/opencl/opencl_backend.*` |
| | ROCmBackend | Class | `backends/rocm/rocm_backend.*` |
| | HybridBackend | Class | `backends/hybrid/hybrid_backend.*` |
| | ZeroCopyBridge | Class | `backends/rocm/zero_copy_bridge.*` |
| | OpenCLCore | Class | `backends/opencl/opencl_core.*` |
| | CommandQueuePool | Class | `backends/opencl/command_queue_pool.*` |
| | MemoryManager | Class | `memory/memory_manager.*` |
| | GPUBuffer\<T\> | Template | `memory/gpu_buffer.hpp` |
| | GPUProfiler | Singleton | `services/gpu_profiler.*` |
| | ConsoleOutput | Singleton | `services/console_output.*` |
| | BatchManager | Static | `services/batch_manager.*` |
| | KernelCacheService | Service | `services/kernel_cache_service.*` |
| | FilterConfigService | Service | `services/filter_config_service.*` |
| | ModuleRegistry | Class | `include/module_registry.*` |
| | Logger | plog wrapper | `logger/logger.*` |
| | GPUConfig | Class | `config/gpu_config.*` |
| **SigGen** | ISignalGenerator | Interface | `include/generators/i_signal_generator.hpp` |
| | CwGenerator | Strategy | `include/generators/cw_generator.hpp` |
| | LfmGenerator | Strategy | `include/generators/lfm_generator.hpp` |
| | NoiseGenerator | Strategy | `include/generators/noise_generator.hpp` |
| | FormSignalGenerator | Strategy | `include/generators/form_signal_generator.hpp` |
| | LfmConjugateGenerator | Strategy | `include/generators/lfm_conjugate_generator.hpp` |
| | SignalGeneratorFactory | Factory | `include/signal_generator_factory.hpp` |
| **FFT** | FFTProcessor | Class | `include/fft_processor.hpp` |
| **Maxima** | ISpectrumProcessor | Interface | `include/processors/i_spectrum_processor.hpp` |
| | SpectrumProcessorOpenCL | Strategy | `include/processors/spectrum_processor_opencl.hpp` |
| | SpectrumProcessorROCm | Strategy | `include/processors/spectrum_processor_rocm.hpp` |
| **Filters** | FirFilter | Class | `include/filters/fir_filter.hpp` |
| | IirFilter | Class | `include/filters/iir_filter.hpp` |
| **Heterodyne** | HeterodyneDechirp | Facade | `include/heterodyne_dechirp.hpp` |
| | IHeterodyneProcessor | Interface | `include/processors/i_heterodyne_processor.hpp` |
| | HeterodyneProcessorOpenCL | Strategy | `include/processors/heterodyne_processor_opencl.hpp` |
| | HeterodyneProcessorROCm | Strategy | `include/processors/heterodyne_processor_rocm.hpp` |
| **Farrow** | LchFarrow | Class | `include/lch_farrow.hpp` |
| **Python** | GPUContext | Wrapper | `python/gpu_worklib_bindings.cpp` |
| | PySignalGenerator | Wrapper | `python/gpu_worklib_bindings.cpp` |
| | PyFFTProcessor | Wrapper | `python/gpu_worklib_bindings.cpp` |
| | PyHeterodyneDechirp | Wrapper | `python/py_heterodyne.hpp` |
| | PyFilters | Wrapper | `python/py_filters.hpp` |
| | PyLchFarrow | Wrapper | `python/py_lch_farrow.hpp` |

---

*Предыдущий уровень: [C2 — Container Diagram](Architecture_C2_Container.md)*
*Следующий уровень: [C4 — Code Diagram](Architecture_C4_Code.md)*
