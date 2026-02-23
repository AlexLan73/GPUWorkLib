# DFD — Data Flow Diagram

> **Project**: GPUWorkLib
> **Date**: 2026-02-23
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
│  P2: FFT        │         │  P5: Heterodyne      │
│  Processor      │         │  Dechirp             │
│                 │         │                      │
│ clFFT / hipFFT  │         │ 1. GenConj(LFM)      │
│ Complex,        │         │ 2. Multiply          │
│ MagPhase modes  │         │ 3. FFT               │
╰────────┬────────╯         │ 4. PeakFind          │
         │                  │ 5. Range calc         │
         │ spectrum         ╰──────────┬───────────╯
         │ (complex/mag+phase)         │
         ▼                             │ HeterodyneResult
╭─────────────────╮                    │ (f_beat, range_m)
│  P3: FFT Maxima │                    │
│  (Spectrum-     │                    │
│   MaximaFinder) │                    │
│                 │                    │
│ ONE_PEAK        │                    │
│ TWO_PEAKS       │                    │
│ ALL_MAXIMA      │                    ▼
╰────────┬────────╯              ┌─────────────┐
         │                       │  Results    │
         │ SpectrumResult[]      │  (User App) │
         │ AllMaximaResult       │             │
         └──────────────────────▶│             │
                                 └─────────────┘
```

---

## Level 2 — Detailed Pipelines

### Pipeline A: Signal Generation → FFT → Peak Detection

```
  CwParams / LfmParams                      vector<SpectrumResult>
  SystemSampling                             AllMaximaResult
       │                                          ▲
       ▼                                          │
╭──────────────╮    cl_mem     ╭──────────────╮   │    ╭──────────────╮
│ P1.1: Create │──(IQ data)──▶│ P2.1: Pad &  │   │    │ P3.1: Peak   │
│ Generator    │              │ Pre-callback  │   │    │ Detection    │
│ (Factory)    │              │ (GPU kernel)  │   │    │ Scan         │
╰──────────────╯              ╰──────┬────────╯   │    │ (GPU kernel) │
                                     │            │    ╰──────┬───────╯
                              FFT input (padded)  │           │
                                     │            │    peak candidates
                                     ▼            │           │
                              ╭──────────────╮    │    ╭──────┴───────╮
                              │ P2.2: clFFT  │    │    │ P3.2: Stream │
                              │ Transform    │    │    │ Compact      │
                              │ (Fwd C2C)    │    │    │ (GPU kernel) │
                              ╰──────┬────────╯   │    ╰──────┬───────╯
                                     │            │           │
                              spectrum (complex)  │    compacted peaks
                                     │            │           │
                                     ▼            │           ▼
                              ╭──────────────╮    │    ╭──────────────╮
                              │ P2.3: Post-  │────┘    │ P3.3: Read-  │
                              │ process      │         │ back to CPU  │
                              │ (Mag/Phase   │         │              │
                              │  kernel)     │         │ → vector<>   │
                              ╰──────────────╯         ╰──────────────╯
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
| D3 | FFT input | GPU Buffer | FFTProcessor | `cl_mem` | GPU |
| D4 | Spectrum | FFTProcessor | GPU Buffer | `cl_mem` (complex/mag+phase) | GPU |
| D5 | Spectrum | GPU Buffer | FFTMaxima | `cl_mem` | GPU |
| D6 | Peak results | FFTMaxima | User | `SpectrumResult[]` | CPU |
| D7 | RX data | User | Heterodyne | `vector<complex<float>>` | CPU→GPU |
| D8 | Conj(LFM) ref | SignalGen | Heterodyne | `cl_mem` (cached) | GPU |
| D9 | Dechirped | Heterodyne mul | FFT stage | `cl_mem` | GPU |
| D10 | Beat freq | Maxima | Heterodyne | `float` (f_beat_hz) | CPU |
| D11 | Filter coeffs | User/JSON | FIR/IIR | `vector<float>` → `cl_mem` | CPU→GPU |
| D12 | Filtered data | Filters | GPU Buffer | `cl_mem` | GPU |
| D13 | Delay params | User/JSON | LchFarrow | `vector<float>` | CPU→GPU |
| D14 | Delayed data | LchFarrow | GPU Buffer | `cl_mem` | GPU |
| D15 | Profiling | Any module | GPUProfiler | `ProfilingMessage` | Async queue |
| D16 | Log messages | Any module | ConsoleOutput | `ConsoleMessage` | Async queue |
| D17 | Kernel binary | KernelCache | Filesystem | `.bin` files | Disk |

---

*Предыдущий документ: [C4 — Code Diagram](Architecture_C4_Code.md)*
*Следующий документ: [Sequence Diagrams](Architecture_Seq.md)*
