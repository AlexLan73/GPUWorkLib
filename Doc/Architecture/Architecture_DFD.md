# DFD — Data Flow Diagram

> **Project**: GPUWorkLib
> **Date**: 2026-03-28
> **Notation**: Gane-Sarson (процессы = прямоугольники с закруглёнными углами)

---

## Level 0 — Context DFD

Общий вид: откуда данные приходят, куда уходят.

```
 ┌───────────────┐                                    ┌───────────────┐
 │  User App     │                                    │  Results       │
 │  (C++ / Py)   │                                    │  (CPU / File)  │
 └───────┬───────┘                                    └───────▲───────┘
         │                                                    │
         │  SignalParams, RxData,                              │  SpectrumResults,
         │  FilterCoeffs, Delays                               │  HeterodyneResult,
         │                                                    │  FilteredData
         ▼                                                    │
 ╔═══════════════════════════════════════════════════════════════════╗
 ║                                                                   ║
 ║                        GPUWorkLib                                 ║
 ║                                                                   ║
 ║    [Params] ──→ [GPU Processing] ──→ [Results]                    ║
 ║                                                                   ║
 ╚═══════════════════════════════════════════════════════════════════╝
         │                                                    ▲
         ▼                                                    │
 ┌───────────────┐                                    ┌───────────────┐
 │  GPU Memory   │ ◄──── cl_mem / hipDeviceptr ─────▶ │  GPU Hardware  │
 │  (Buffers)    │                                    │  (Compute)     │
 └───────────────┘                                    └───────────────┘
```

---

## Level 1 — Main Processes DFD

Детализация до отдельных модулей-процессов.

```
                         ┌─────────────────────┐
                         │  User Application   │
                         └──┬───┬───┬───┬───┬──┘
                            │   │   │   │   │
              CwParams ─────┘   │   │   │   └───── delay_us[]
              LfmParams ────────┘   │   │          sample_rate
              NoiseParams ──────────┘   │
              RxData (complex[]) ───────┘
                            │
         ┌──────────────────┼──────────────────────────────┐
         ▼                  ▼                              ▼
╭─────────────────╮  ╭─────────────────╮         ╭──────────────────╮
│  P1: Signal     │  │  P4: Filters    │         │  P6: LCH Farrow  │
│  Generators     │  │  (FIR / IIR)    │         │  (Frac Delay)    │
│                 │  │                 │         │                  │
│ CW, LFM, Noise │  │ y=Σh[k]*x[n-k] │         │ Lagrange 5-pt    │
│ Form, Conjugate │  │                 │         │ 48×5 matrix      │
╰────────┬────────╯  ╰────────┬────────╯         ╰────────┬─────────╯
         │                    │                            │
         │ cl_mem             │ cl_mem                     │ cl_mem
         │ (complex IQ)      │ (filtered)                 │ (delayed)
         ▼                    ▼                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    D1: GPU Buffer Store                     │
    │                    (cl_mem / hipDeviceptr_t)                │
    └────────┬─────────────────────┬──────────────────────────────┘
             │                     │
             │ complex IQ          │ complex IQ (from any source)
             ▼                     ▼
╭─────────────────╮         ╭──────────────────────╮
│  P2: fft_func   │         │  P5: Heterodyne      │
│  (merged FFT +  │         │  Dechirp             │
│   Maxima)       │         │                      │
│                 │         │ 1. GenConj(LFM)      │
│ clFFT / hipFFT  │         │ 2. Multiply          │
│ Complex,        │         │ 3. FFT               │
│ MagPhase modes  │         │ 4. PeakFind          │
│ ONE_PEAK        │         │ 5. Range calc         │
│ TWO_PEAKS       │         ╰──────────┬───────────╯
│ ALL_MAXIMA      │                    │
╰────────┬────────╯                    │ HeterodyneResult
         │                             │ (f_beat, range_m)
         │ SpectrumResult[]            │
         │ AllMaximaResult             │
         │ FFTComplexResult            ▼
         │                       ┌─────────────┐
         │                       │  Results    │
         └──────────────────────▶│  (User App) │
                                 │             │
                                 └─────────────┘
```

---

## Level 2 — Detailed Pipelines

### Pipeline A: Signal Generation → fft_func (FFT + Peak Detection)

```
  CwParams / LfmParams                      vector<SpectrumResult>
  SystemSampling                             AllMaximaResult
       │                                          ▲
       ▼                                          │
╭──────────────╮    cl_mem     ╭──────────────────────────────────────╮
│ P1.1: Create │──(IQ data)──▶│            fft_func module            │
│ Generator    │              │                                      │
│ (Factory)    │              │ ╭──────────────╮  ╭──────────────╮   │
╰──────────────╯              │ │ P2.1: Pad &  │  │ P2.4: Peak   │   │
                              │ │ Pre-callback  │  │ Detection    │   │
                              │ │ (GPU kernel)  │  │ Scan         │   │
                              │ ╰──────┬────────╯  │ (GPU kernel) │   │
                              │        │           ╰──────┬───────╯   │
                              │ FFT input (padded)        │           │
                              │        │           peak candidates    │
                              │        ▼                  │           │
                              │ ╭──────────────╮  ╭──────┴───────╮   │
                              │ │ P2.2: clFFT  │  │ P2.5: Stream │   │
                              │ │ Transform    │  │ Compact      │   │
                              │ │ (Fwd C2C)    │  │ (GPU kernel) │   │
                              │ ╰──────┬────────╯ ╰──────┬───────╯   │
                              │        │          compacted peaks     │
                              │ spectrum (complex)       │           │
                              │        │                 ▼           │
                              │        ▼          ╭──────────────╮   │
                              │ ╭──────────────╮  │ P2.6: Read-  │   │
                              │ │ P2.3: Post-  │──│ back to CPU  │───│──▶
                              │ │ process      │  │              │   │
                              │ │ (Mag/Phase)  │  │ → vector<>   │   │
                              │ ╰──────────────╯  ╰──────────────╯   │
                              ╰──────────────────────────────────────╯
```

### Pipeline B: Heterodyne LFM Dechirp

```
  HeterodyneParams                           HeterodyneResult
  rx_data (complex[])                        (f_beat, range_m per antenna)
       │                                          ▲
       ▼                                          │
╭──────────────╮                            ╭──────────────╮
│ P5.1: Upload │   cl_mem (rx)              │ P5.6: Build  │
│ RX to GPU    │──────────────┐             │ Result       │
╰──────────────╯              │             ╰──────▲───────╯
                              │                    │
╭──────────────╮              │             peak freq + SNR
│ P5.2: GenConj│  cl_mem      │                    │
│ LFM Ref      │──(conj_ref)──┤             ╭──────┴───────╮
│ (cached OPT4)│              │             │ P5.5: Maxima │
╰──────────────╯              │             │ Find         │
                              ▼             │ (ISpectrum-  │
                       ╭──────────────╮     │  Processor)  │
                       │ P5.3: Dechirp│     ╰──────▲───────╯
                       │ Multiply     │            │
                       │              │      spectrum (complex)
                       │ s_dc[i] =    │            │
                       │ rx[i]*conj[i]│     ╭──────┴───────╮
                       ╰──────┬───────╯     │ P5.4: FFT    │
                              │             │ (FFTProcessor)│
                       cl_mem (dechirped)   │              │
                              │             ╰──────▲───────╯
                              └────────────────────┘
```

### Pipeline C: FIR Filter Processing

```
  filter_coeffs[]                         InputData<cl_mem>
  input cl_mem (complex, ch × pts)        (filtered output)
       │                                       ▲
       ▼                                       │
╭──────────────╮   cl_mem      ╭──────────────╮│
│ P4.1: Upload │──(coeff_buf)─▶│ P4.3: FIR    ││
│ Coefficients │              │ Convolution   ││
│ to GPU       │              │ (GPU kernel)  │┘
│              │              │               │
│ __constant   │   cl_mem     │ y[ch][n] =    │
│ or __global  │──(input_buf)▶│ Σh[k]*x[n-k] │
│ (>16K taps)  │              │               │
╰──────────────╯              ╰───────────────╯
```

### Pipeline D: LCH Farrow Fractional Delay

```
  delay_us[], matrix[48][5]              InputData<cl_mem>
  input cl_mem (complex, ant × pts)      (delayed output)
       │                                       ▲
       ▼                                       │
╭──────────────╮   cl_mem      ╭──────────────╮│
│ P6.1: Upload │──(matrix_buf)▶│ P6.3: Farrow │┘
│ Matrix +     │              │ Interpolate  │
│ Delays       │   cl_mem     │ (GPU kernel) │
│ to GPU       │──(delay_buf)▶│              │
╰──────────────╯              │ 5-point      │
                   cl_mem     │ Lagrange     │
            ──────(input_buf)▶│ per-antenna  │
                              ╰──────────────╯
```

### Pipeline E: Strategies Beamforming

```
  Signal[N_ant × M]                          AntennaResult
  Weights[N_ant × K]                         (pre_stats, post_gemm_stats,
       │                                      post_fft_stats, scenario results)
       ▼                                          ▲
╭──────────────╮                                  │
│ P7.1: Pre-   │  pre_stats                       │
│ Statistics   │──(mean, std, power)──┐           │
│ (Statistics- │                      │           │
│  Processor)  │                      │           │
╰──────┬───────╯                      │           │
       │ signal                       │           │
       ▼                              │           │
╭──────────────╮                      │           │
│ P7.2: CGEMM  │  X = W · S          │           │
│ (hipBLAS)    │──────────────────────┤           │
╰──────┬───────╯                      │           │
       │ X (transformed)              │           │
       ▼                              │           │
╭──────────────╮  post_gemm_stats     │           │
│ P7.3: Post-  │──(mean, std, power)──┤           │
│ GEMM Stats   │                      │           │
│ (Statistics) │                      │           │
╰──────┬───────╯                      │           │
       │ X                            │           │
       ▼                              │           │
╭──────────────╮                      │           │
│ P7.4: Hamming│  spectrum            │           │
│ + batched FFT│──────────┐           │           │
│ (hipFFT)     │          │           │           │
╰──────────────╯          │           │           │
                          ▼           │           │
                   ╭──────────────╮   │           │
                   │ P7.5: Post-  │   │           │
                   │ FFT Stats    │───┤           │
                   │ (Statistics) │   │           │
                   ╰──────┬───────╯   │           │
                          │           │           │
                          ▼           │           │
                   ╭──────────────╮   │           │
                   │ P7.6: PostFFT│───┘           │
                   │ Scenario     │───────────────┘
                   │ (OneMax /    │
                   │  AllMaxima / │
                   │  MinMax)     │
                   ╰──────────────╯
```

### Pipeline F: Capon MVDR

```
  Y[P×N] (antenna data)                     CaponReliefResult / CaponBeamResult
  U[P×M] (steering vectors)                 (relief[M] or Y_out[M×N])
       │                                          ▲
       ▼                                          │
╭──────────────╮                                  │
│ P8.1: Covari-│  R = YY^H/N + μI               │
│ ance Matrix  │──(rocBLAS CGEMM)──┐             │
│ (CovMatrix-  │                   │             │
│  Op)         │                   │             │
╰──────────────╯                   │             │
                                   ▼             │
                            ╭──────────────╮     │
                            │ P8.2: Cholesky│    │
                            │ Invert       │     │
                            │ (POTRF +     │     │
                            │  POTRI +     │     │
                            │  symmetrize) │     │
                            ╰──────┬───────╯     │
                                   │             │
                                   │ R^-1        │
                                   ▼             │
                            ╭──────────────╮     │
                            │ P8.3: Compute│     │
                            │ Weights or   │─────┘
                            │ Relief       │
                            │              │
                            │ Relief:      │
                            │  z[m] = 1/   │
                            │  Re(u^H·R^-1·u)
                            │              │
                            │ Beamform:    │
                            │  W = R^-1·U  │
                            │  Y_out=W^H·Y │
                            ╰──────────────╯
```

### Pipeline G: Range Angle 3D Processing

```
  rx[n_ant × n_samples]                     RangeAngleResult
  (FMCW radar data)                         (TargetInfo[], power_cube)
       │                                          ▲
       ▼                                          │
╭──────────────╮                                  │
│ P9.1: Dechirp│  beat tones                     │
│ Window       │──(rx × conj(ref)                │
│ + Hamming    │   + window)──┐                   │
╰──────────────╯              │                   │
                              ▼                   │
                       ╭──────────────╮           │
                       │ P9.2: Range  │           │
                       │ FFT          │           │
                       │ (batched     │           │
                       │  hipFFT per  │           │
                       │  antenna)    │           │
                       ╰──────┬───────╯           │
                              │                   │
                       range spectrum              │
                       per antenna                │
                              │                   │
                              ▼                   │
                       ╭──────────────╮           │
                       │ P9.3: Trans- │           │
                       │ pose         │           │
                       │ rearrange to │           │
                       │ [range×az×el]│           │
                       ╰──────┬───────╯           │
                              │                   │
                              ▼                   │
                       ╭──────────────╮           │
                       │ P9.4: Beam   │           │
                       │ FFT 2D       │           │
                       │ (spatial FFT │           │
                       │  + fftshift) │           │
                       │ → power cube │           │
                       ╰──────┬───────╯           │
                              │                   │
                       power cube                 │
                       [range×az×el]              │
                              │                   │
                              ▼                   │
                       ╭──────────────╮           │
                       │ P9.5: Peak   │───────────┘
                       │ Search       │
                       │ (3D max      │
                       │  reduction)  │
                       │ → TargetInfo │
                       ╰──────────────╯
```

---

## Level 2 — DrvGPU Internal Data Flow

```
 ┌─────────────┐           ┌─────────────────┐
 │ Module code │           │ configGPU.json  │
 │ (any module)│           │                 │
 └──────┬──────┘           └────────┬────────┘
        │                          │
        │ IBackend* calls          │ load
        ▼                          ▼
 ╭──────────────╮          ╭──────────────╮
 │  Backend     │◄─────────│ GPUConfig    │
 │  Dispatch    │  config  │ (JSON parse) │
 │              │          ╰──────────────╯
 │ OpenCL or    │
 │ ROCm or      │
 │ Hybrid       │
 ╰───┬──────┬───╯
     │      │
     │      │ timing events
     │      ▼
     │  ╭──────────────╮     ┌──────────────────┐
     │  │ GPUProfiler  │────▶│ Results/Profiler/ │
     │  │ (async queue)│     │ (.json, .md)      │
     │  ╰──────────────╯     └──────────────────┘
     │
     │ memory ops
     ▼
 ╭──────────────╮     ┌──────────────────┐
 │ Memory       │     │ GPU VRAM         │
 │ Manager      │────▶│ (cl_mem /        │
 │              │     │  hipDeviceptr)   │
 │ alloc/free   │     └──────────────────┘
 │ track stats  │
 ╰──────┬───────╯
        │
        │ log messages
        ▼
 ╭──────────────╮     ┌──────────────────┐
 │ ConsoleOutput│────▶│ stdout           │
 │ (async queue)│     │ [HH:MM:SS] [GPU] │
 ╰──────────────╯     └──────────────────┘
        │
        │ per-GPU log
        ▼
 ╭──────────────╮     ┌──────────────────┐
 │ Logger       │────▶│ Logs/DRVGPU_XX/  │
 │ (plog)       │     │ YYYY-MM-DD/*.log │
 ╰──────────────╯     └──────────────────┘
```

---

## Сводная таблица потоков данных

| # | Поток | Источник | Назначение | Тип данных | Среда |
|---|-------|----------|-----------|------------|-------|
| D1 | Signal params | User | SignalGenerator | CwParams/LfmParams/etc. | CPU |
| D2 | Generated IQ | SignalGenerator | GPU Buffer | `cl_mem` (complex\<float\>) | GPU |
| D3 | FFT input | GPU Buffer | fft_func | `cl_mem` | GPU |
| D4 | Spectrum | fft_func (FFT) | GPU Buffer | `cl_mem` (complex/mag+phase) | GPU |
| D5 | Spectrum | fft_func (FFT) | fft_func (Maxima) | `cl_mem` | GPU |
| D6 | Peak results | fft_func (Maxima) | User | `SpectrumResult[]` | CPU |
| D7 | RX data | User | Heterodyne | `vector<complex<float>>` | CPU→GPU |
| D8 | Conj(LFM) ref | SignalGen | Heterodyne | `cl_mem` (cached) | GPU |
| D9 | Dechirped | Heterodyne mul | FFT stage | `cl_mem` | GPU |
| D10 | Beat freq | fft_func (Maxima) | Heterodyne | `float` (f_beat_hz) | CPU |
| D11 | Filter coeffs | User/JSON | FIR/IIR | `vector<float>` → `cl_mem` | CPU→GPU |
| D12 | Filtered data | Filters | GPU Buffer | `cl_mem` | GPU |
| D13 | Delay params | User/JSON | LchFarrow | `vector<float>` | CPU→GPU |
| D14 | Delayed data | LchFarrow | GPU Buffer | `cl_mem` | GPU |
| D15 | Profiling | Any module | GPUProfiler | `ProfilingMessage` | Async queue |
| D16 | Log messages | Any module | ConsoleOutput | `ConsoleMessage` | Async queue |
| D17 | Kernel binary | KernelCache | Filesystem | `.bin` files | Disk |
| D18 | Signal + Weights | User | Strategies | `complex<float>*` × 2 | CPU→GPU |
| D19 | AntennaResult | Strategies | User | stats + scenario results | CPU |
| D20 | Y + Steering | User | Capon | `complex<float>*` × 2 | CPU→GPU |
| D21 | Relief / Beam | Capon | User | `float[]` or `complex[]` | CPU |
| D22 | FMCW RX data | User | RangeAngle | `complex<float>*` | CPU→GPU |
| D23 | Power cube | RangeAngle (BeamFFT) | PeakSearch | `float[]` 3D | GPU |
| D24 | TargetInfo[] | RangeAngle | User | range/angle/power | CPU |

---

*Last updated: 2026-03-28*

*Предыдущий документ: [C4 — Code Diagram](Architecture_C4_Code.md)*
*Следующий документ: [Sequence Diagrams](Architecture_Seq.md)*
