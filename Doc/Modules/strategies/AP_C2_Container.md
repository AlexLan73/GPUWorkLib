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

  HOST RAM                                  GPU VRAM (AMD 9070: 16 ГБ / MI100: 32 ГБ)
  ──────────────────────                    ──────────────────────────────────────────────────────
  ┌─────────────────────┐                   ┌──────────────────────────────────────────────────┐
  │ S[N_ant × N_samples]│  Stream 0 (DMA)  │ d_S[N_ant × N_samples]    d_W[N_ant × N_ant]    │
  │ (cf32, 2.5 ГБ)      │──────────────────▶│ (cf32, 2.5 ГБ)            (cf32, 512 КБ)        │
  │ W[N_ant × N_ant]    │  hipMemcpyAsync  │                                                   │
  │ (cf32, 512 КБ)      │──────────────────▶│ d_hamming[N_samples]      d_X[N_ant × N_samples]│
  └─────────────────────┘                   │ (f32, 4.8 МБ, in L2!)    (cf32, 2.5 ГБ)        │
                                             │                                                   │
  hipEventRecord(event_data_ready)           │ d_spectrum[N_ant × nFFT]                        │
                                             │ (cf32, ≈4.9 ГБ)                                 │
                                             └──────────────────────────────────────────────────┘
                                                              │
                          ┌───────────────────────────────────┴────────────────────────────────┐
                          │                                                                     │
              ┌───────────▼──────────────────┐                    ┌────────────────────────────▼──────────────┐
              │   Stream 1: Statistics PRE    │                    │   Stream 2: Main Pipeline                  │
              │   (параллельно с Stream 2)    │                    │                                            │
              │   ─────────────────────────── │                    │   ┌──────────────────────────────────────┐ │
              │                               │                    │   │ GemmWrapper (hipBLAS Cgemm)           │ │
              │   StatisticsProcessor         │                    │   │ X = W × S                             │ │
              │   welford_fused(d_S):         │                    │   │ ≈ 13 мс (compute-bound)               │ │
              │   ┌─────────────────────────┐ │                    │   │ W (512 КБ) → из L2 кеша! ✅           │ │
              │   │ mean (complex)          │ │                    │   └──────────────────┬───────────────────┘ │
              │   │ variance of |z|         │ │                    │                      │ event_gemm_done      │
              │   │ std_dev of |z|          │ │                    │   ┌──────────────────▼───────────────────┐ │
              │   │ min |z| + idx           │ │                    │   │ HammingProcessor (apply_hamming.hip)  │ │
              │   │ max |z| + idx           │ │                    │   │ X[n] *= 0.54 - 0.46cos(2πn/N)        │ │
              │   └─────────────────────────┘ │                    │   │ d_hamming в L2 → overhead ≈ 0         │ │
              │   radix_sort + extract_medians│                    │   └──────────────────┬───────────────────┘ │
              │   → median of |z|            │                    │                      │                      │
              │                               │                    │   ┌──────────────────▼───────────────────┐ │
              │   Output: StatsResult[N_ant]  │                    │   │ hipFFT Batch (hipFFT)                  │ │
              │   (если pre_gemm_stats!=NONE) │                    │   │ N_ant × hipFFT_C2C(nFFT)              │ │
              │                               │                    │   │ nFFT = next_pow2(N_samples) ≈ 2.1M    │ │
              │   hipEventRecord              │                    │   │ ≈ 20 мс (TBD, бенчмарк нужен)        │ │
              │   (event_stats_done)          │                    │   └──────────────────┬───────────────────┘ │
              └───────────────────────────────┘                    │                      │ event_fft_done       │
                          │                                        │   ┌──────────────────▼───────────────────┐ │
                          │                                        │   │ fold_fft_mirror()                     │ │
                          │              ┌─────────────────────────│   │ bins k > nFFT/2 → negative freq        │ │
                          │              │ event_gemm_done          │   └──────────────────┬───────────────────┘ │
                          │              ▼                          │                      │                      │
              ┌───────────▼──────────────────┐                    │   ┌──────────────────▼───────────────────┐ │
              │   Stream 3: Statistics POST   │                    │   │ IBranchStrategy::execute()            │ │
              │   (параллельно с Ham+FFT)     │                    │   │   Branch 2: MinMaxBranchStrategy      │ │
              │                               │                    │   │   Branch 3: ParabolaBranchStrategy    │ │
              │   StatisticsProcessor         │                    │   │   Branch 4: AllMaximaBranchStrategy   │ │
              │   welford_fused(d_X):         │                    │   └──────────────────┬───────────────────┘ │
              │   ┌─────────────────────────┐ │                    │                      │                      │
              │   │ mean + median + ...     │ │                    │   ICheckpointSave::save_c3/c4()            │
              │   └─────────────────────────┘ │                    │                      │                      │
              │   → StatsResult[N_ant]        │                    └──────────────────────┼─────────────────────┘
              │   (если post_gemm_stats!=NONE)│                                           │
              │   hipEventRecord              │                                           │
              │   (event_spost_done)          │                                           │
              └───────────────────────────────┘                                           │
                                                                                          ▼
                                                                          ┌──────────────────────────┐
                                                                          │   AntennaResult           │
                                                                          │                            │
                                                                          │   .pre_stats[N_ant]        │
                                                                          │   .post_stats[N_ant]       │
                                                                          │   .minmax[N_ant] (B2)      │
                                                                          │   .peaks[N_ant]  (B3)      │
                                                                          │   .all_maxima[]  (B4)      │
                                                                          │   .perf (timing metrics)   │
                                                                          └──────────────────────────┘
```

---

## 2. HIP Events Flow

```
Stream 0 (DMA) ──────────────► event_data_ready
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                       ▼
Stream 1 (Stats)   ├── welford_fused(d_S)                Stream 2 (Main)
                    └── radix_sort                          ├── hipblasCgemm ──► event_gemm_done
                         ► event_stats_done                │                           │
                                                           │                           ├──────────► Stream 3 (SPost)
                                                           ├── apply_hamming           │              └── welford_fused(d_X)
                                                           ├── hipFFT                  │                   ► event_spost_done
                                                           │    ► event_fft_done       │
                                                           └── Branch + Checkpoint     │

Синхронизация (CPU side, перед сборкой AntennaResult):
  hipEventSynchronize(event_stats_done)
  hipEventSynchronize(event_spost_done)
  hipEventSynchronize(event_fft_done)
```

---

## 3. Данные в VRAM

| Буфер | Размер (256 × 1.2M) | Назначение |
|-------|---------------------|-----------|
| `d_S[N_ant × N_samples]` | 2.45 ГБ | Входной сигнал (READ-ONLY) |
| `d_W[N_ant × N_ant]` | 512 КБ | Матрица весов (READ-ONLY, fit L2!) |
| `d_X[N_ant × N_samples]` | 2.45 ГБ | GEMM output + Hamming in-place |
| `d_hamming[N_samples]` | 4.8 МБ | Окно Хемминга (кешируется в L2!) |
| `d_spectrum[N_ant × nFFT]` | 4.92 ГБ | FFT output |
| **ИТОГО** | **≈ 10.3 ГБ** | Укладывается в 16 ГБ (9070) ✅ |

> ⚠️ При нехватке VRAM: d_spectrum можно не хранить целиком — branch сразу читает и уничтожает. Для Branch 4 (all maxima) нужен дополнительный буфер флагов.

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
              "Branch 2: MinMax\nBranch 3: Parabola\nBranch 4: AllMaxima (internal)")

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
