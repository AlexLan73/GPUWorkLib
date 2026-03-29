# C3 — Component Diagram

> **Project**: GPUWorkLib
> **Date**: 2026-03-28
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

## 3. fft_func — Components

```
┌───────────────────────── fft_func ──────────────────────────────────┐
│                                                                       │
│  ┌──────────────── FFT Processing (ROCm) ─────────────────────┐     │
│  │ FFTProcessorROCm (Facade)                                    │     │
│  │                                                              │     │
│  │ Initialize(params)                                           │     │
│  │ ProcessComplex(data, params) → vector<ComplexResult>          │     │
│  │ ProcessMagPhase(data, params) → vector<FFTMagPhaseResult>     │     │
│  │                                                              │     │
│  │ ComplexToMagPhaseROCm — HIP kernel: |z|, atan2(im,re)       │     │
│  │                                                              │     │
│  │ Internal GPU Buffers:                                        │     │
│  │   fft_input_  (nFFT * batch)                                 │     │
│  │   fft_output_ (FFT result)                                   │     │
│  │   mag_output_ (Magnitude)                                    │     │
│  │   phase_output_ (Phase)                                      │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌──────────────── Spectrum Maxima (ROCm) ────────────────────┐      │
│  │ SpectrumProcessorROCm / AllMaximaPipelineROCm               │      │
│  │                                                              │      │
│  │ Initialize(params)                                           │      │
│  │ ProcessFromCPU(data) → vector<SpectrumResult>                │      │
│  │ ProcessFromGPU(gpu_data, ...) → vector<Result>               │      │
│  │ FindAllMaximaFromCPU(...) → AllMaximaResult                  │      │
│  │ FindAllMaximaFromGPUPipeline(...) → AllMaximaResult          │      │
│  │                                                              │      │
│  │ AllMaxima Pipeline:                                          │      │
│  │   Stage 1: FFT (hipFFT)                                      │      │
│  │   Stage 2: Magnitude computation (HIP kernel)                │      │
│  │   Stage 3: Peak detection — threshold scan (GPU)             │      │
│  │   Stage 4: Peak compaction — stream compact (GPU)            │      │
│  │   Stage 5: Result readback to CPU                            │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                       │
│  ┌──────────────── Operations (Ref03 Layer 5) ────────────────┐      │
│  │ PadDataOp     — zero-pad input to nFFT (BufferSet<2>)       │      │
│  │ MagPhaseOp    — magnitude + phase extraction (BufferSet<3>) │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                       │
│  ┌────────────────── Data Types ──────────────────────────────┐      │
│  │ FFTProcessorParams:                                         │      │
│  │   { beam_count, n_point, sample_rate,                       │      │
│  │     output_mode, use_padding }                              │      │
│  │ FFTMagPhaseResult:                                          │      │
│  │   { magnitude, phase, frequency_hz: vector<float> }         │      │
│  │ SpectrumParams: { antenna_count, n_point, nFFT,             │      │
│  │                   sample_rate, mode }                        │      │
│  │ MaxValue: { bin, amplitude, frequency_hz }                  │      │
│  │ SpectrumMode: ONE_PEAK | TWO_PEAKS | ALL_MAXIMA             │      │
│  │ AllMaximaResult: { peaks[], num_peaks, runtime_ms }         │      │
│  │ OutputDestination: CPU | GPU                                │      │
│  └────────────────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 4. Filters — Components

```
┌───────────────────── Filters ──────────────────────────────────┐
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ FirFilter                          [OpenCL]           │       │
│  │                                                       │       │
│  │ LoadConfig(json_path)                                 │       │
│  │ SetCoefficients(coeffs)                               │       │
│  │ Process(cl_mem, channels, points) → InputData<cl_mem> │       │
│  │ ProcessCpu(input, ch, pts) → vector<complex<float>>   │       │
│  │                                                       │       │
│  │ ⚡ Kernel: __constant (≤16000 taps)                   │       │
│  │           __global   (>16000 taps, fallback)          │       │
│  │ Formula: y[ch][n] = Σ h[k] * x[ch][n-k]              │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ IirFilter                          [OpenCL]           │       │
│  │                                                       │       │
│  │ SetBiquadSections(sections)                           │       │
│  │ Process(cl_mem, channels, points) → InputData<cl_mem> │       │
│  │                                                       │       │
│  │ Formula: biquad DFII-T cascade                        │       │
│  │ y[n] = Σ b[k]*x[n-k] - Σ a[k]*y[n-k]                │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────── ROCm variants ───────────────────┐       │
│  │ FirFilterROCm / IirFilterROCm      [HIP + hiprtc]     │       │
│  │ (same API as OpenCL variants)                         │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌────────────── ROCm-only: Adaptive Filters ───────────┐       │
│  │ MovingAverageFilterROCm                               │       │
│  │   SetParams(type: SMA|EMA|MMA|DEMA|TEMA, N)           │       │
│  │   Process(cl_mem, channels, points) → cl_mem          │       │
│  │   1 thread per channel, efficient at ≥64 channels     │       │
│  │                                                       │       │
│  │ KalmanFilterROCm                                      │       │
│  │   SetParams(Q, R, x0, P0)                             │       │
│  │   1D scalar Kalman (Re/Im независимо)                 │       │
│  │                                                       │       │
│  │ KaufmanFilterROCm (KAMA)                              │       │
│  │   SetParams(er_period, fast_period, slow_period)      │       │
│  │   Adaptive Moving Average по Efficiency Ratio         │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Heterodyne (LFM Dechirp) — Components

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

## 6. LCH Farrow (Fractional Delay) — Components

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

## 7. FM Correlator — Components

```
┌───────────────────── FM Correlator ───────────────────────────┐
│  Backend: ROCm-only (hipFFT R2C/C2R + HIP kernels)             │
│                                                                 │
│  ┌─────────────────────────── API ──────────────────────┐      │
│  │ FMCorrelator                                          │      │
│  │                                                       │      │
│  │ SetParams(FMCorrelatorParams)                         │      │
│  │ GenerateMSequence([seed]) → vector<float>             │      │
│  │ PrepareReference([ref_signal])   // Step 1 (once)     │      │
│  │ Process(input_signals) → FMCorrelatorResult           │      │
│  │ RunTestPattern(shift_step) → FMCorrelatorResult       │      │
│  │ Step1_ReferenceFFT() / Step2_InputFFT() / Step3_Corr()│      │
│  └──────────────────────────┬────────────────────────────┘      │
│                             │                                    │
│  ┌──────────────────────────┴───────────────────────────┐       │
│  │ HIP Kernels (3 production + 1 utility)               │       │
│  │                                                       │       │
│  │ apply_cyclic_shifts   — float→float2 + K shifts      │       │
│  │ multiply_conj_fused   — conj(ref_fft) × inp_fft      │       │
│  │                          (conj inline, N/2+1 points)  │       │
│  │ extract_magnitudes    — |corr_time| / N, n_kg points  │       │
│  │ generate_test_inputs  — circshift(ref, s*step) on GPU │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌────────────────── hipFFT Plans ───────────────────────┐      │
│  │ plan_ref:  C2C Forward, batch=K  (in-place, ref_fft)  │      │
│  │ plan_inp:  R2C Forward, batch=S  (float → N/2+1 cplx) │      │
│  │ plan_corr: C2R Inverse, batch=S×K (cplx → float)     │      │
│  │ Созданы при SetParams(), хранятся постоянно           │      │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────── Data Types ───────────────────────┐      │
│  │ FMCorrelatorParams:                                   │      │
│  │   { fft_size=32768, num_shifts=32, num_signals=5,     │      │
│  │     num_output_points=2000,                           │      │
│  │     lfsr_polynomial=0xB8000000, lfsr_seed=0x1 }       │      │
│  │ FMCorrelatorResult:                                   │      │
│  │   { peaks[S×K×n_kg], at(signal,shift,point) }        │      │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Strategies — Components

```
┌──────────────────── Strategies ──────────────────────────┐
│                                                            │
│  ┌─────────── Facade ──────────────────┐                  │
│  │ AntennaProcessor_v1                  │                  │
│  │                                      │                  │
│  │ Initialize(config) → void            │                  │
│  │ Process(signal, weights) → Result    │                  │
│  │ 4 HIP streams, hipBLAS CGEMM        │                  │
│  └──────────────────────────────────────┘                  │
│                                                            │
│  ┌─────── Pipeline Steps (Ref03) ──────────────────────┐  │
│  │ GemmStep          — hipBLAS CGEMM: X = W·S           │  │
│  │ WindowFftStep     — Hamming window + hipFFT          │  │
│  │ DebugStatsStep    — StatisticsProcessor checkpoint   │  │
│  │ OneMaxStep        — Parabolic interpolation          │  │
│  │ AllMaximaStep     — Stream compaction                │  │
│  │ MinMaxStep        — Global min/max + DR(dB)          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────── Config / Types ──────────────────────────────┐  │
│  │ AntennaProcessorConfig  — n_ant, n_samples, fs       │  │
│  │ PostFftScenarioMode     — ALL_REQUIRED/ONE_MAX/...   │  │
│  │ AntennaResult           — stats, maxima, minmax      │  │
│  │ WeightGenerator         — delay-and-sum weights      │  │
│  │ PipelineBuilder         — Builder pattern            │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 9. Capon — Components

```
┌──────────────────── Capon (MVDR) ────────────────────────┐
│                                                            │
│  ┌─────────── Facade (Ref03 Layer 6) ──────────────────┐  │
│  │ CaponProcessor                                       │  │
│  │                                                      │  │
│  │ Initialize(params) → void                            │  │
│  │ ComputeRelief(Y, U) → CaponReliefResult             │  │
│  │ AdaptiveBeamform(Y, U) → CaponBeamResult             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────── Operations (Ref03 Layer 5) ──────────────────┐  │
│  │ CovarianceMatrixOp  — R = YY^H/N + μI (rocBLAS)     │  │
│  │ CaponInvertOp       — R^-1 via CholeskyInverterROCm  │  │
│  │ ComputeWeightsOp    — W = R^-1·U (rocBLAS CGEMM)    │  │
│  │ CaponReliefOp       — z[m] = 1/Re(u^H·R^-1·u)      │  │
│  │ AdaptBeamformOp     — Y_out = W^H·Y (rocBLAS)       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────── Types ───────────────────────────────────────┐  │
│  │ CaponParams          — n_channels, n_samples, n_dir  │  │
│  │ CaponReliefResult    — vector<float> relief[M]       │  │
│  │ CaponBeamResult      — vector<complex> output[M×N]   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 10. Range Angle — Components

```
┌──────────────── Range Angle (3D) ────────────────────────┐
│                                                            │
│  ┌─────────── Facade (Ref03 Layer 6) ──────────────────┐  │
│  │ RangeAngleProcessor                                  │  │
│  │                                                      │  │
│  │ Initialize(params) → void                            │  │
│  │ Process(signal) → RangeAngleResult                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────── Operations (Ref03 Layer 5) ──────────────────┐  │
│  │ DechirpWindowOp  — conj(ref) × rx + Hamming          │  │
│  │ RangeFftOp       — batched hipFFT per antenna        │  │
│  │ TransposeOp      — rearrange [range × az × el]       │  │
│  │ BeamFftOp        — 2D spatial FFT + fftshift          │  │
│  │ PeakSearchOp     — 3D max reduction → TargetInfo     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────── Types ───────────────────────────────────────┐  │
│  │ RangeAngleParams  — n_ant_az/el, n_samples, f_start  │  │
│  │ TargetInfo        — range_m, az/el_deg, power_db     │  │
│  │ RangeAngleResult  — targets[], power_cube             │  │
│  │ PeakSearchMode    — TOP_1, TOP_N                      │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 11. Python Bindings — Components

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

## 12. Сводная таблица всех компонентов

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
| **fft_func** | FFTProcessorROCm | Facade | `include/fft_processor_rocm.hpp` |
| | ComplexToMagPhaseROCm | Class | `include/complex_to_mag_phase_rocm.hpp` |
| | PadDataOp | GpuKernelOp | `include/operations/pad_data_op.hpp` |
| | MagPhaseOp | GpuKernelOp | `include/operations/mag_phase_op.hpp` |
| | ISpectrumProcessor | Interface | `include/processors/i_spectrum_processor.hpp` |
| | SpectrumProcessorROCm | Strategy | `include/processors/spectrum_processor_rocm.hpp` |
| | AllMaximaPipelineROCm | Class | `include/processors/all_maxima_pipeline_rocm.hpp` |
| **Filters** | FirFilter | Class | `include/filters/fir_filter.hpp` |
| | IirFilter | Class | `include/filters/iir_filter.hpp` |
| | FirFilterROCm | Class | `include/filters/fir_filter_rocm.hpp` |
| | IirFilterROCm | Class | `include/filters/iir_filter_rocm.hpp` |
| | MovingAverageFilterROCm | Class | `include/filters/moving_average_filter_rocm.hpp` |
| | KalmanFilterROCm | Class | `include/filters/kalman_filter_rocm.hpp` |
| | KaufmanFilterROCm | Class | `include/filters/kaufman_filter_rocm.hpp` |
| **Heterodyne** | HeterodyneDechirp | Facade | `include/heterodyne_dechirp.hpp` |
| | IHeterodyneProcessor | Interface | `include/processors/i_heterodyne_processor.hpp` |
| | HeterodyneProcessorOpenCL | Strategy | `include/processors/heterodyne_processor_opencl.hpp` |
| | HeterodyneProcessorROCm | Strategy | `include/processors/heterodyne_processor_rocm.hpp` |
| **Farrow** | LchFarrow | Class | `include/lch_farrow.hpp` |
| **FMCorr** | FMCorrelator | Class | `include/fm_correlator.hpp` |
| | FMCorrelatorParams | Struct | `include/fm_correlator.hpp` |
| | FMCorrelatorResult | Struct | `include/fm_correlator.hpp` |
| **Strategies** | AntennaProcessor_v1 | Facade | `include/antenna_processor_v1.hpp` |
| | AntennaProcessorConfig | Config | `include/config/antenna_processor_config.hpp` |
| | WeightGenerator | Static | `include/weight_generator.hpp` |
| | PipelineBuilder | Builder | `include/pipeline_builder.hpp` |
| | GemmStep/WindowFftStep/... | Steps | `include/steps/*.hpp` |
| **Capon** | CaponProcessor | Facade | `include/capon_processor.hpp` |
| | CovarianceMatrixOp | GpuKernelOp | `include/operations/covariance_matrix_op.hpp` |
| | CaponInvertOp | Op | `include/operations/capon_invert_op.hpp` |
| | ComputeWeightsOp | GpuKernelOp | `include/operations/compute_weights_op.hpp` |
| | CaponReliefOp | GpuKernelOp | `include/operations/capon_relief_op.hpp` |
| | AdaptBeamformOp | GpuKernelOp | `include/operations/adapt_beam_op.hpp` |
| **RangeAngle** | RangeAngleProcessor | Facade | `include/range_angle_processor.hpp` |
| | DechirpWindowOp | GpuKernelOp | `include/operations/dechirp_window_op.hpp` |
| | RangeFftOp | GpuKernelOp | `include/operations/range_fft_op.hpp` |
| | TransposeOp | GpuKernelOp | `include/operations/transpose_op.hpp` |
| | BeamFftOp | GpuKernelOp | `include/operations/beam_fft_op.hpp` |
| | PeakSearchOp | GpuKernelOp | `include/operations/peak_search_op.hpp` |
| **Python** | GPUContext | Wrapper | `python/gpu_worklib_bindings.cpp` |
| | PySignalGenerator | Wrapper | `python/gpu_worklib_bindings.cpp` |
| | PyFFTProcessor | Wrapper | `python/gpu_worklib_bindings.cpp` |
| | PyHeterodyneDechirp | Wrapper | `python/py_heterodyne.hpp` |
| | PyFilters | Wrapper | `python/py_filters.hpp` |
| | PyLchFarrow | Wrapper | `python/py_lch_farrow.hpp` |
| | PyFMCorrelatorROCm | Wrapper | `python/py_fm_correlator_rocm.hpp` |
| | PyAntennaProcessor | Wrapper | `python/py_antenna_processor.hpp` |
| | PyCaponProcessor | Wrapper | `python/py_capon_processor.hpp` |
| | PyRangeAngleProcessor | Wrapper | `python/py_range_angle_processor.hpp` |

---

*Предыдущий уровень: [C2 — Container Diagram](Architecture_C2_Container.md)*
*Следующий уровень: [C4 — Code Diagram](Architecture_C4_Code.md)*
