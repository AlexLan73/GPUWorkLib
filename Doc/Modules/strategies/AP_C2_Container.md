# C2 — Container Diagram: AntennaProcessor Module
# GPUWorkLib — Antenna Array Processor

> **Project**: GPUWorkLib / AntennaProcessor
> **Date**: 2026-03-06
> **Reference**: [c4model.com](https://c4model.com)
> **Level**: 2 (Container) — контейнеры: GPU streams, компоненты, данные

---

## 1. Container Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              AntennaProcessor — Container View                                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

  EXTERNAL GPU PRODUCER                       GPU VRAM (AMD 9070: 16 ГБ / MI100: 32 ГБ)
  ──────────────────────                      ─────────────────────────────────────────────────────
  ┌─────────────────────┐                     ┌──────────────────────────────────────────────────┐
  │ d_S already on GPU  │                     │ d_S[N_ant × N_samples]    d_W[N_ant × N_ant]    │
  │ metadata (fs, N, ..)│────────────────────▶│ input signal               beamforming matrix     │
  └─────────────────────┘                     │                                                  │
                                              │ d_X[N_ant × N_samples]    d_spectrum[N_ant×nFFT]│
                                              │ GEMM output               shared FFT output      │
                                              └──────────────────────────────────────────────────┘
                                                               │
                         ┌─────────────────────────────────────┴──────────────────────────────────┐
                         │                                                                         │
             ┌───────────▼──────────────────┐                     ┌────────────────────────────────▼──────────────┐
             │ Stream 1: Debug/Stats 2.1     │                     │ Stream 2: Main Pipeline                      │
             │ (параллельно с Stream 2)      │                     │                                              │
             │ ───────────────────────────── │                     │ 1. GEMM: X = W × S                          │
             │ stats(d_S)                    │                     │ 2. Base block: Window + FFT -> d_spectrum   │
             │ save(d_S)                     │                     │ 3. запускает post-FFT consumers             │
             │ python(d_S or stats)          │                     │                                              │
             │ -> event_c1_done              │                     │ W кэшируется в L2                           │
             └───────────────────────────────┘                     └───────────────┬──────────────────────────────┘
                         │                                                         │
                         │                                      event_gemm_done    │ event_fft_done
                         │                                                         │
             ┌───────────▼──────────────────┐                     ┌────────────────▼──────────────────────────────┐
             │ Stream 3: Debug/Stats 2.2     │                     │ Stream 4: Post-FFT Consumers                │
             │ (параллельно с Window+FFT)    │                     │                                              │
             │ ───────────────────────────── │                     │ Step2.1: OneMax + Parabola (no phase)       │
             │ stats(d_X)                    │                     │ Step2.2: AllMaxima (limit=1000)             │
             │ save(d_X)                     │                     │ Step2.3: GlobalMinMax (limit=1000)          │
             │ python(d_X or stats)          │                     │ PostFFT stats over |spectrum|               │
             │ -> event_c2_done              │                     │ save/python for 2.3 debug                   │
             └───────────────────────────────┘                     └────────────────┬──────────────────────────────┘
                                                                                     │
                                                                                     ▼
                                                                       ┌──────────────────────────────┐
                                                                       │ AntennaResult                 │
                                                                       │ pre_input_stats               │
                                                                       │ post_gemm_stats               │
                                                                       │ post_fft_stats                │
                                                                       │ one_max / all_maxima / minmax │
                                                                       │ perf                          │
                                                                       └──────────────────────────────┘
```

---

## 2. HIP Events Flow

```
External GPU producer ─────► d_S ready
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
Stream 1 (2.1)   stats/save/python(d_S)        Stream 2 (Main)
                  ► event_c1_done               ├── hipblasCgemm ──► event_gemm_done
                                                ├── Window + FFT ──► event_fft_done
                                                └── shared d_spectrum

event_gemm_done ───────────────────────────────► Stream 3 (2.2)
                                                 └── stats/save/python(d_X)
                                                     ► event_c2_done

event_fft_done ────────────────────────────────► Stream 4 (2.3 + post-FFT)
                                                 ├── stats(|spectrum|)
                                                 ├── save/python(spectrum)
                                                 ├── Step2.1 one_max + parabola
                                                 ├── Step2.2 all_maxima
                                                 └── Step2.3 global min/max

Синхронизация (CPU side, перед сборкой AntennaResult):
  hipEventSynchronize(event_c1_done)
  hipEventSynchronize(event_c2_done)
  hipEventSynchronize(event_fft_done)
```

---

## 3. Данные в VRAM

| Буфер | Размер (256 × 1.2M) | Назначение |
|-------|---------------------|-----------|
| `d_S[N_ant × N_samples]` | 2.45 ГБ | Входной сигнал (READ-ONLY) |
| `d_W[N_ant × N_ant]` | 512 КБ | Матрица весов (READ-ONLY, fit L2!) |
| `d_X[N_ant × N_samples]` | 2.45 ГБ | GEMM output, вход для debug `2.2` и для `Window + FFT` |
| `d_hamming[N_samples]` | 4.8 МБ | Окно Хемминга (кешируется в L2!) |
| `d_spectrum[N_ant × nFFT]` | 4.92 ГБ | FFT output, общий вход для всех post-FFT consumers |
| **ИТОГО** | **≈ 10.3 ГБ** | Укладывается в 16 ГБ (9070) ✅ |

> ⚠️ При нехватке VRAM: `d_spectrum` можно переиспользовать батчами, но базовый вариант быстрее при одном вычислении FFT и повторном чтении общего спектра несколькими consumers. Для `AllMaxima` нужен дополнительный буфер флагов.

---

## 4. Конфигурация VRAM (схема расположения)

```
GPU VRAM [16 ГБ]:
┌───────────────────────────────────────────────────────────────────────────────┐
│  d_S      │ 2.45 ГБ │ [0x0000...0x9B000000]  input signal (READ-ONLY)       │
│  d_W      │ 512 КБ  │ [tiny, likely in L2 cache after first access]          │
│  d_X      │ 2.45 ГБ │ [0x9B000000...0x13600000] GEMM output (WRITE)         │
│  d_hamming│ 4.8 МБ  │ [tiny, definitely in L2 cache: 4.8MB << 32MB L2]      │
│  d_spectrum│ 4.92 ГБ │ [0x136... large!]  FFT output                        │
│  Branches │ < 1 МБ  │ MinMaxResult, MaxValue scratch buffers                 │
│  [FREE]   │ 5.7 ГБ  │ (reserved for OS/driver overhead + other modules)     │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. PlantUML: Container Diagram

```plantuml
@startuml AP_C2_Container
!include <C4/C4_Container>
LAYOUT_WITH_LEGEND()
title AntennaProcessor — C2: Container Diagram

Person(user, "C++ Engineer / Python Scientist", "Вызывает process(S, W)")

System_Boundary(ap_module, "strategies module") {

    Container(iap, "AntennaProcessor", "C++ Interface",
              "Abstract entry point\nprocess(S, W) → AntennaResult")

    Container(aap, "AntennaProcessor_v1", "C++ Class",
              "Main pipeline orchestrator\nManages HIP streams & events")

    ContainerDb(vram_s, "d_S [GPU VRAM]", "hipFloatComplex*",
                "Input signal: N_ant × N_samples\n2.45 ГБ")

    ContainerDb(vram_w, "d_W [GPU VRAM]", "hipFloatComplex*",
                "Weight matrix: N_ant × N_ant\n512 КБ (fits in L2!)")

    ContainerDb(vram_x, "d_X [GPU VRAM]", "hipFloatComplex*",
                "GEMM output: N_ant × N_samples\n2.45 ГБ")

    ContainerDb(vram_spec, "d_spectrum [GPU VRAM]", "hipFloatComplex*",
                "FFT output: N_ant × nFFT\n4.9 ГБ")

    Container(stream1, "Stream 1: Stats PRE", "HIP Stream",
              "welford_fused(d_S)\nradix_sort → medians\nParallel with GEMM")

    Container(stream2, "Stream 2: Main", "HIP Stream",
              "GEMM → Hamming → FFT\n→ fold_mirror → Branch")

    Container(stream3, "Stream 3: Stats POST", "HIP Stream",
              "welford_fused(d_X)\nParallel with Hamming+FFT")

    Container(branch, "IBranchStrategy", "C++ Strategy",
              "Step2.1: OneMax+Parabola\nStep2.2: AllMaxima\nStep2.3: GlobalMinMax")

    Container(chk, "ICheckpointSave", "C++ Null Object",
              "NullCheckpointSave (production)\nCheckpointSave (debug)")
}

Container_Ext(drv, "DrvGPU", "C++ (HIP backend)",
              "hipBLAS, hipFFT, streams, events\nGPUProfiler, ConsoleOutput")

Container_Ext(stats_proc, "StatisticsProcessor", "C++ Module",
              "welford_fused, radix_sort\nextract_medians kernels")

Container_Ext(fs, "Logs/GPU_XX/antenna_processor/", "File System",
              "Binary checkpoint files\nC1..C4_*.bin, meta.json")

Rel(user, iap, "process(S, W)", "C++ API or Python")
Rel(iap, aap, "implements")
Rel(aap, vram_s, "DMA load")
Rel(aap, vram_w, "DMA load")
Rel(aap, stream1, "stats PRE-GEMM")
Rel(aap, stream2, "GEMM+Hamming+FFT")
Rel(aap, stream3, "stats POST-GEMM")
Rel(stream1, stats_proc, "welford_fused, radix_sort")
Rel(stream2, vram_x, "writes GEMM output")
Rel(stream2, vram_spec, "writes FFT output")
Rel(stream2, branch, "execute(spectrum)")
Rel(aap, chk, "save checkpoints")
Rel(chk, fs, "writes binary files")
Rel(aap, drv, "GPU resources\n(streams, events, hipBLAS, hipFFT)")

SHOW_LEGEND()
@enduml
```

---

*Создано: 2026-03-06*
*Следующий уровень: [C3 — Component Diagram](AP_C3_Component.md)*
*Предыдущий уровень: [C1 — System Context](AP_C1_SystemContext.md)*
