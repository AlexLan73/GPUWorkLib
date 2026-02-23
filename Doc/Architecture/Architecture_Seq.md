# Seq — Sequence Diagrams

> **Project**: GPUWorkLib
> **Date**: 2026-02-23
> **Notation**: UML Sequence Diagrams (ASCII + PlantUML)

---

## Seq-1: DrvGPU Initialization

Инициализация одного GPU-устройства.

```
 User App          DrvGPU          GPUConfig       OpenCLBackend      OpenCLCore     ConsoleOutput   Logger
    │                 │                │                 │                │                │            │
    │ DrvGPU()        │                │                 │                │                │            │
    ├────────────────▶│                │                 │                │                │            │
    │                 │ LoadConfig()   │                 │                │                │            │
    │                 ├───────────────▶│                 │                │                │            │
    │                 │ ◄──── config ──┤                 │                │                │            │
    │                 │                │                 │                │                │            │
    │ Initialize()    │                │                 │                │                │            │
    ├────────────────▶│                │                 │                │                │            │
    │                 │  new OpenCLBackend()             │                │                │            │
    │                 ├────────────────────────────────▶│                │                │            │
    │                 │                │                 │                │                │            │
    │                 │  Initialize(device_index)        │                │                │            │
    │                 ├────────────────────────────────▶│                │                │            │
    │                 │                │                 │ InitCore()     │                │            │
    │                 │                │                 ├───────────────▶│                │            │
    │                 │                │                 │                │ clGetPlatform  │            │
    │                 │                │                 │                │ clCreateCtx    │            │
    │                 │                │                 │                │ clCreateQueue  │            │
    │                 │                │                 │  ◄── OK ──────┤                │            │
    │                 │                │                 │                │                │            │
    │                 │                │                 │ InitQueuePool()│                │            │
    │                 │                │                 ├───────────────▶│                │            │
    │                 │  ◄──────── initialized ─────────┤                │                │            │
    │                 │                │                 │                │                │            │
    │                 │  new MemoryManager(backend)      │                │                │            │
    │                 │  new ModuleRegistry()            │                │                │            │
    │                 │                │                 │                │                │            │
    │                 │  Print("GPU initialized")        │                │                │            │
    │                 ├──────────────────────────────────────────────────────────────────▶│            │
    │                 │                │                 │                │                │            │
    │                 │  LOG_INFO("DrvGPU ready")        │                │                │            │
    │                 ├───────────────────────────────────────────────────────────────────────────────▶│
    │  ◄──── OK ──────┤                │                 │                │                │            │
    │                 │                │                 │                │                │            │
```

---

## Seq-2: Signal Generation → FFT → Peak Detection

Типичный pipeline: генерация CW сигнала → FFT → поиск максимума.

```
 User App       SigGenFactory    CwGenerator     FFTProcessor      clFFT       SpectrumProc    GPUProfiler
    │                │                │                │              │              │              │
    │ Create(CW)     │                │                │              │              │              │
    ├───────────────▶│                │                │              │              │              │
    │                │ new CwGen()    │                │              │              │              │
    │                ├───────────────▶│                │              │              │              │
    │ ◄── ptr ───────┤                │                │              │              │              │
    │                                 │                │              │              │              │
    │ GenerateToGpu(system, beams)    │                │              │              │              │
    ├────────────────────────────────▶│                │              │              │              │
    │                                 │ Allocate(N*beams*8B)          │              │              │
    │                                 │ clCreateKernel("cw_gen")     │              │              │
    │                                 │ clEnqueueNDRange             │              │              │
    │                                 │ Record(profiler)              │              │              │
    │                                 ├──────────────────────────────────────────────────────────▶│
    │ ◄── cl_mem ────────────────────┤                │              │              │              │
    │                                                  │              │              │              │
    │ ProcessComplex(cl_mem, params)                   │              │              │              │
    ├─────────────────────────────────────────────────▶│              │              │              │
    │                                                  │ pad_kernel   │              │              │
    │                                                  │ (pre-callback)              │              │
    │                                                  │              │              │              │
    │                                                  │ clfftEnqueue │              │              │
    │                                                  ├─────────────▶│              │              │
    │                                                  │ ◄── done ────┤              │              │
    │                                                  │              │              │              │
    │                                                  │ Record()     │              │              │
    │                                                  ├──────────────────────────────────────────▶│
    │ ◄── vector<FFTComplexResult> ───────────────────┤              │              │              │
    │                                                                 │              │              │
    │ ProcessFromGPU(spectrum_cl_mem, antennas, nFFT)                 │              │              │
    ├────────────────────────────────────────────────────────────────────────────────▶│              │
    │                                                                 │              │              │
    │                                                                 │  magnitude   │              │
    │                                                                 │  kernel      │              │
    │                                                                 │              │              │
    │                                                                 │  peak_detect │              │
    │                                                                 │  kernel      │              │
    │                                                                 │              │              │
    │                                                                 │  readback    │              │
    │                                                                 │              │ Record()     │
    │                                                                 │              ├─────────────▶│
    │ ◄── vector<SpectrumResult> ────────────────────────────────────────────────────┤              │
    │                                                                                               │
```

---

## Seq-3: Heterodyne LFM Dechirp Pipeline

Полный цикл дечирпирования.

```
 User App      HeterodyneDechirp   LfmConjGen      HeteroProc_OCL   FFTProcessor    SpectrumProc
    │                │                  │                │                │               │
    │ SetParams()    │                  │                │                │               │
    ├───────────────▶│                  │                │                │               │
    │                │ store params_    │                │                │               │
    │                │                  │                │                │               │
    │ Process(rx)    │                  │                │                │               │
    ├───────────────▶│                  │                │                │               │
    │                │                  │                │                │               │
    │                │ ┌── cache check ─┐               │                │               │
    │                │ │ conj_lfm_      │               │                │               │
    │                │ │ exists?        │               │                │               │
    │                │ └───┬────────────┘               │                │               │
    │                │     │ NO (first call)            │                │               │
    │                │     ▼                            │                │               │
    │                │ GenerateToGpu()                  │                │               │
    │                ├─────────────────▶│                │                │               │
    │                │                  │ Build kernel   │                │               │
    │                │                  │ Generate conj  │                │               │
    │                │ ◄─ cl_mem ──────┤                │                │               │
    │                │ cache_conj_lfm_ = cl_mem         │                │               │
    │                │                                  │                │               │
    │                │ Upload rx → GPU                  │                │               │
    │                │ MemcpyHostToDevice(rx_buf)       │                │               │
    │                │                                  │                │               │
    │                │ Dechirp(rx_buf, conj_buf)        │                │               │
    │                ├─────────────────────────────────▶│                │               │
    │                │                                  │ GPU kernel:    │               │
    │                │                                  │ dc[i] = rx[i]  │               │
    │                │                                  │   * conj[i]    │               │
    │                │ ◄── cl_mem (dechirped) ─────────┤                │               │
    │                │                                  │                │               │
    │                │ ProcessComplex(dechirped, params) │                │               │
    │                ├──────────────────────────────────────────────────▶│               │
    │                │                                  │                │ clFFT fwd     │
    │                │ ◄── spectrum ────────────────────────────────────┤               │
    │                │                                  │                │               │
    │                │ ProcessFromGPU(spectrum, ...)     │                │               │
    │                ├──────────────────────────────────────────────────────────────────▶│
    │                │                                  │                │               │
    │                │                                  │                │               │ peak detect
    │                │                                  │                │               │ magnitude
    │                │                                  │                │               │ scan+compact
    │                │ ◄── SpectrumResult[] ────────────────────────────────────────────┤
    │                │                                  │                │               │
    │                │ ┌── Range Calculation ──────────────────────┐    │               │
    │                │ │ for each antenna:                          │    │               │
    │                │ │   f_beat = result.peak_freq_hz             │    │               │
    │                │ │   range = c * T * f_beat / (2 * B)         │    │               │
    │                │ │   snr_db = result.peak_snr_db              │    │               │
    │                │ └────────────────────────────────────────────┘    │               │
    │                │                                  │                │               │
    │ ◄── HeterodyneResult ─────────────────────────┤                │               │
    │    { success=true,                              │                │               │
    │      antennas[ { f_beat, range_m, snr } ],      │                │               │
    │      max_positions[] }                           │                │               │
    │                                                  │                │               │
```

---

## Seq-4: Python API Usage

Типичный сценарий Python-пользователя.

```
 Python Script        GPUContext      PySignalGen     PyFFTProcessor   PyHeterodyne
    │                    │                │                │               │
    │ ctx = GPUContext(0)│                │                │               │
    ├───────────────────▶│                │                │               │
    │                    │ DrvGPU()       │                │               │
    │                    │ Initialize()   │                │               │
    │ ◄── ctx ──────────┤                │                │               │
    │                    │                │                │               │
    │ gen = SignalGenerator(ctx)          │                │               │
    ├───────────────────────────────────▶│                │               │
    │ ◄── gen ─────────────────────────┤                │               │
    │                                    │                │               │
    │ data = gen.generate_cw(            │                │               │
    │   freq=1000, fs=44100,             │                │               │
    │   length=4096, beams=8)            │                │               │
    ├───────────────────────────────────▶│                │               │
    │                    │                │ CwGenerator::  │               │
    │                    │                │ GenerateToGpu()│               │
    │                    │                │ ReadBack to    │               │
    │                    │                │ np.ndarray     │               │
    │ ◄── np.ndarray[complex64] ────────┤                │               │
    │    shape=(8, 4096)                  │                │               │
    │                                    │                │               │
    │ fft = FFTProcessor(ctx)            │                │               │
    ├────────────────────────────────────────────────────▶│               │
    │ ◄── fft ─────────────────────────────────────────┤               │
    │                                    │                │               │
    │ result = fft.process_complex(data, │                │               │
    │   beam_count=8, n_point=4096,      │                │               │
    │   sample_rate=44100.0)             │                │               │
    ├────────────────────────────────────────────────────▶│               │
    │                    │                │ np → cl_mem    │               │
    │                    │                │ FFTProcessor:: │               │
    │                    │                │ ProcessComplex │               │
    │                    │                │ cl_mem → np    │               │
    │ ◄── list[dict] ────────────────────────────────────┤               │
    │    [{'spectrum': np.ndarray}, ...]  │                │               │
    │                                    │                │               │
    │ het = HeterodyneDechirp(ctx)       │                │               │
    ├───────────────────────────────────────────────────────────────────▶│
    │                    │                │                │               │
    │ het.set_params(f_start=1e6,        │                │               │
    │   f_end=10e6, fs=50e6,             │                │               │
    │   num_samples=4096,                │                │               │
    │   num_antennas=4)                  │                │               │
    ├───────────────────────────────────────────────────────────────────▶│
    │                    │                │                │               │
    │ result = het.process(rx_data)       │                │               │
    ├───────────────────────────────────────────────────────────────────▶│
    │                    │                │                │               │
    │                    │                │                │  Full pipeline│
    │                    │                │                │  (Seq-3)      │
    │                    │                │                │               │
    │ ◄── dict ──────────────────────────────────────────────────────────┤
    │    {'success': True,                │                │               │
    │     'antennas': [                   │                │               │
    │       {'f_beat_hz': 500000.0,       │                │               │
    │        'range_m': 1500.0,           │                │               │
    │        'peak_snr_db': 45.2}, ...],  │                │               │
    │     'max_positions': [...]}         │                │               │
    │                                    │                │               │
```

---

## Seq-5: Multi-GPU Batch Processing

Обработка большого массива данных на нескольких GPU.

```
 User App         BatchManager       DrvGPU[0]        DrvGPU[1]        DrvGPU[N]
    │                  │                │                │                │
    │ CalcOptBatch()   │                │                │                │
    ├─────────────────▶│                │                │                │
    │                  │ GetFreeMemory()│                │                │
    │                  ├───────────────▶│                │                │
    │                  │ ◄── free_mem ──┤                │                │
    │                  │                │                │                │
    │                  │ batch_size = min(total,          │                │
    │                  │   free_mem * 0.7 / item_bytes)  │                │
    │ ◄── batch_size ──┤                │                │                │
    │                  │                │                │                │
    │ CreateBatches()  │                │                │                │
    ├─────────────────▶│                │                │                │
    │ ◄── BatchRange[] ┤                │                │                │
    │    [{0..1000},    │                │                │                │
    │     {1000..2000}, │                │                │                │
    │     {2000..2847}] │ (tail merged) │                │                │
    │                  │                │                │                │
    │ ┌── Parallel dispatch (per GPU) ─────────────────────────────────┐│
    │ │                │                │                │                ││
    │ │ Process(batch[0])               │                │                ││
    │ ├────────────────────────────────▶│                │                ││
    │ │                │                │                │                ││
    │ │ Process(batch[1])                                │                ││
    │ ├──────────────────────────────────────────────────▶│               ││
    │ │                │                │                │                ││
    │ │ Process(batch[2])                                                 ││
    │ ├──────────────────────────────────────────────────────────────────▶││
    │ │                │                │                │                ││
    │ │ ◄── results[0] ────────────────┤                │                ││
    │ │ ◄── results[1] ─────────────────────────────────┤                ││
    │ │ ◄── results[2] ──────────────────────────────────────────────────┤│
    │ └────────────────────────────────────────────────────────────────────┘
    │                  │                │                │                │
    │ Merge results    │                │                │                │
    │                  │                │                │                │
```

---

## Seq-6: Profiling & Reporting

```
 Module Code       GPUProfiler         AsyncQueue       File System
    │                  │                   │                │
    │ SetGPUInfo()     │                   │                │
    ├─────────────────▶│                   │                │
    │                  │ store info        │                │
    │                  │                   │                │
    │ profiler.Start() │                   │                │
    ├─────────────────▶│                   │                │
    │                  │ start worker      │                │
    │                  ├──────────────────▶│                │
    │                  │                   │ thread running │
    │                  │                   │                │
    │ Record(gpu_id,   │                   │                │
    │  module, event,  │                   │                │
    │  profiling_data) │                   │                │
    ├─────────────────▶│                   │                │
    │                  │ enqueue(msg)      │                │
    │                  ├──────────────────▶│                │
    │ ◄── return ──────┤  (non-blocking)   │                │
    │                  │                   │ process msg    │
    │                  │                   │ aggregate      │
    │  ... more Records ...                │                │
    │                  │                   │                │
    │ PrintReport()    │                   │                │
    ├─────────────────▶│                   │                │
    │                  │ format stats      │                │
    │                  │ → stdout          │                │
    │                  │                   │                │
    │ ExportJSON(path) │                   │                │
    ├─────────────────▶│                   │                │
    │                  │ serialize stats   │                │
    │                  ├──────────────────────────────────▶│
    │                  │                   │                │ write .json
    │                  │                   │                │
    │ ExportMarkdown() │                   │                │
    ├─────────────────▶│                   │                │
    │                  ├──────────────────────────────────▶│
    │                  │                   │                │ write .md
    │                  │                   │                │
    │ profiler.Stop()  │                   │                │
    ├─────────────────▶│                   │                │
    │                  │ stop worker       │                │
    │                  ├──────────────────▶│                │
    │                  │                   │ flush & exit   │
    │                  │                   │                │
```

---

## PlantUML — Seq-3 (Heterodyne)

```plantuml
@startuml Seq3_Heterodyne
title Seq-3: HeterodyneDechirp::Process()

participant "User App" as User
participant "HeterodyneDechirp" as HD
participant "LfmConjugateGen" as LCG
participant "HeteroProc_OCL" as HP
participant "FFTProcessor" as FFT
participant "ISpectrumProcessor" as SP

User -> HD: SetParams(params)
HD -> HD: store params_

User -> HD: Process(rx_data)
activate HD

alt cached_conj_lfm_ == nullptr
    HD -> LCG: GenerateToGpu(system, 1)
    activate LCG
    LCG --> HD: cl_mem (conj_ref)
    deactivate LCG
    HD -> HD: cached_conj_lfm_ = cl_mem
end

HD -> HD: Upload rx → GPU (MemcpyH2D)

HD -> HP: Dechirp(rx_buf, conj_buf, ...)
activate HP
note right: GPU kernel:\ndc[i] = rx[i] * conj[i]
HP --> HD: cl_mem (dechirped)
deactivate HP

HD -> FFT: ProcessComplex(dechirped, params)
activate FFT
note right: clFFT forward C2C
FFT --> HD: spectrum
deactivate FFT

HD -> SP: ProcessFromGPU(spectrum, ...)
activate SP
note right: Magnitude → Peak detect\n→ Stream compact → Readback
SP --> HD: vector<SpectrumResult>
deactivate SP

HD -> HD: Range calculation:\nR = c·T·f_beat / (2·B)

HD --> User: HeterodyneResult\n{success, antennas[], max_positions[]}
deactivate HD

@enduml
```

---

## PlantUML — Seq-4 (Python API)

```plantuml
@startuml Seq4_Python
title Seq-4: Python API Usage

actor "Python Script" as Py

participant "GPUContext" as Ctx
participant "PySignalGenerator" as SG
participant "PyFFTProcessor" as FFT
participant "PyHeterodyneDechirp" as HD

Py -> Ctx: GPUContext(device_index=0)
activate Ctx
Ctx -> Ctx: DrvGPU::Initialize()
Ctx --> Py: ctx
deactivate Ctx

Py -> SG: SignalGenerator(ctx)
SG --> Py: gen

Py -> SG: generate_cw(freq, fs, length, beams)
activate SG
SG -> SG: CwGenerator::GenerateToGpu()
SG -> SG: ReadBack → np.ndarray
SG --> Py: np.ndarray[complex64]\nshape=(beams, length)
deactivate SG

Py -> FFT: FFTProcessor(ctx)
FFT --> Py: fft

Py -> FFT: process_complex(data, beam_count, n_point, fs)
activate FFT
FFT -> FFT: np → cl_mem
FFT -> FFT: FFTProcessor::ProcessComplex()
FFT -> FFT: cl_mem → np
FFT --> Py: list[dict]\n[{spectrum: np.ndarray}, ...]
deactivate FFT

Py -> HD: HeterodyneDechirp(ctx)
HD --> Py: het

Py -> HD: set_params(f_start, f_end, fs, N, antennas)
Py -> HD: process(rx_data)
activate HD
HD -> HD: Full dechirp pipeline\n(see Seq-3)
HD --> Py: dict {success, antennas[], max_positions[]}
deactivate HD

@enduml
```

---

*Предыдущий документ: [DFD — Data Flow Diagram](Architecture_DFD.md)*
*Индекс: [Architecture Index](Architecture_INDEX.md)*
