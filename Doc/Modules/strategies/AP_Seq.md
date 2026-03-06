# Seq — Sequence & Timing Diagrams: AntennaProcessor
# GPUWorkLib — Antenna Array Processor

> **Project**: GPUWorkLib / AntennaProcessor
> **Date**: 2026-03-06
> **Notation**: UML Sequence (ASCII) + GPU Stream Timing

---

## Seq-1: Полный pipeline (Branch 3: Parabola)

Типичный сценарий: данные с сети → GPU → GEMM → FFT → один пик + парабола

```
 UserApp       AntennaProcessor_v1  Stream0(DMA) Stream1(Stats) Stream2(Main)  Stream3(SPost)    Result
    │                │                │             │               │              │               │
    │ create(cfg)    │                │             │               │              │               │
    ├───────────────▶│                │             │               │              │               │
    │                │ init_streams() │             │               │              │               │
    │                │ init_events()  │             │               │              │               │
    │                │ alloc_buffers()│             │               │              │               │
    │  OK            │                │             │               │              │               │
    │◀───────────────┤                │             │               │              │               │
    │                │                │             │               │              │               │
    │ process(S,W)   │                │             │               │              │               │
    ├───────────────▶│                │             │               │              │               │
    │                │ hipMemcpyAsync(d_S, S)       │               │              │               │
    │                ├───────────────▶│             │               │              │               │
    │                │ hipMemcpyAsync(d_W, W)       │               │              │               │
    │                ├───────────────▶│             │               │              │               │
    │                │ hipEventRecord(event_data_ready)             │              │               │
    │                ├───────────────▶│             │               │              │               │
    │                │                │             │               │              │               │
    │                │                │  ◄──────────event_data_ready (запуск Stats1 и Main2)────▶ │
    │                │                │             │               │              │               │
    │                │     [Stream 1: Statistics PRE-GEMM]          │              │               │
    │                │                │  welford_fused(d_S, N_ant, N_samples)     │               │
    │                │                │  ├── mean, std, var, min, max (per row)   │               │
    │                │                │  └── radix_sort → extract_medians         │               │
    │                │                │     event_stats_done ────────────────────▶│               │
    │                │                │                                            │               │
    │                │     [Stream 2: Main Pipeline]                               │               │
    │                │                │               hipblasCgemm:               │               │
    │                │                │               X = W × S    ───────────────▶               │
    │                │                │               (~13 мс для 256×1.2M)       │               │
    │                │                │               hipEventRecord(event_gemm_done)             │
    │                │                │               ◀─────event_gemm_done start Stream3         │
    │                │                │                                            │               │
    │                │     [Stream 3: Statistics POST-GEMM] (параллельно с Hamming+FFT)           │
    │                │                │                            welford_fused(d_X, ...)        │
    │                │                │                            event_spost_done ─────────────▶│
    │                │                │                                            │               │
    │                │                │               apply_hamming(d_X, d_hamm)  │               │
    │                │                │               (~2.6 мс)     ─────────────▶│               │
    │                │                │                                            │               │
    │                │                │               hipfftExecC2C(plan, d_X, d_spectrum)        │
    │                │                │               (N_ant FFTs × nFFT)         │               │
    │                │                │               hipEventRecord(event_fft_done)              │
    │                │                │                                            │               │
    │                │ hipEventSynchronize(event_stats_done)        │              │               │
    │                │ hipEventSynchronize(event_spost_done)        │              │               │
    │                │ hipEventSynchronize(event_fft_done)          │              │               │
    │                │                │                                            │               │
    │                │                │               strategy_->execute()        │               │
    │                │                │               (ParabolaBranchStrategy)    │               │
    │                │                │               post_kernel_one_peak        │               │
    │                │                │               → MaxValue[N_ant]           │               │
    │                │                │                                            │               │
    │                │ checkpoint_->save_c4_peak(peaks, N_ant)      │              │               │
    │                │                │                                            │               │
    │                │ build AntennaResult ─────────────────────────────────────────────────────▶│
    │  result        │                │             │               │              │               │
    │◀───────────────┤                │             │               │              │               │
    │                │                │             │               │              │               │
```

---

## Seq-2: GPU Stream Timing Diagram (параллелизм)

```
Время (мс) →    0    10    20    30    40    50    60
                │     │     │     │     │     │     │
Stream 0 (DMA): [===DMA S+W (78мс PCIe4, 2.5ГБ)=====]
                │     │     │     │     │     │     │
                       ▼ event_data_ready (~0мс, если данные уже в GPU!)
                │     │
Stream 1 (Stat):[==welford+sort==]►event_stats_done
                │     (~2.6мс)  │
                │               │ (ждёт пока Main завершит FFT)
                │               │
Stream 2 (Main):[===GEMM 13мс===][Ham 2.6мс][===FFT ≈20мс===]►event_fft_done
                │               │           │               │
                                ▼ event_gemm_done           │
Stream 3 (SPost):               [==stats POST 2.6мс==]►event_spost_done
                                │               │
                                                │
                                     Branch execute (~1мс)
                                                │
                                     ▼ AntennaResult ready

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

АНАЛИЗ ВРЕМЁН (256 антенн × 1,200,000 выборок, 9070 ~960 ГБ/с):

  Операция           │ Время    │ Ограничение  │ Примечание
  ───────────────────┼──────────┼──────────────┼─────────────────────────────
  DMA Host→GPU       │ 78 мс*   │ PCIe 4.0     │ *если данные на CPU RAM
  (если GPU→GPU)     │ 2.6 мс   │ 960 ГБ/с     │ если данные уже в VRAM
  GEMM (W×S)         │ 13 мс    │ Compute-bound│ 629 GFLOPs / 48 TFLOPS
  Hamming (apply)    │ 2.6 мс   │ BW-bound     │ 5 ГБ / 960 ГБ/с (parallel!)
  FFT batch (N×nFFT) │ ~20 мс** │ Compute+BW   │ **требует бенчмарка
  Stats PRE-GEMM     │ 2.6 мс   │ BW-bound     │ parallel с GEMM
  Stats POST-GEMM    │ 2.6 мс   │ BW-bound     │ parallel с Ham+FFT
  Branch 2 (minmax)  │ < 1 мс   │ Compute-light│ 256 блоков × 256 потоков
  Branch 3 (parabola)│ < 1 мс   │ Compute-light│ то же
  Branch 4 (all max) │ 2-5 мс   │ Compute+BW   │ Blelloch scan

  ИТОГО (без DMA, если данные в VRAM):
    GEMM (13) + Hamming(0, скрыт) + FFT(20) + Branch(1) ≈ 34 мс

  УЗКОЕ МЕСТО: FFT и GEMM!
  Оптимизация: hipFFT plan кешируется (создаётся один раз при init)
                W матрица маленькая (512 КБ) → fit в L2 кеш GPU!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

МАЛЫЙ ВАРИАНТ (3500 × 2500, 70 МБ):
  Все вычисления << 1 мс (данные влезают в L2 кеш GPU!)
```

---

## Seq-3: Смена Branch (без пересоздания объекта)

```
 UserApp       AntennaProcessor_v1    StrategyFactory
    │                │                       │
    │ process(S,W)   │                       │
    │ (mode=PARABOLA)│                       │
    ├───────────────▶│                       │
    │    result      │ (ParabolaBranchStrategy выполняется)
    │◀───────────────┤                       │
    │                │                       │
    │                │  // Заказчик хочет Branch 2
    │ set_branch(MINMAX)                     │
    ├───────────────▶│                       │
    │                │ strategy_.reset()     │
    │                │ strategy_ = new MinMaxBranchStrategy()
    │                │                       │
    │ process(S,W)   │                       │
    ├───────────────▶│                       │
    │    result      │ (MinMaxBranchStrategy выполняется)
    │◀───────────────┤                       │
    │                │                       │
    │                │  // Включить сохранение C4
    │ set_checkpoint_save(new CheckpointSave(save_cfg))
    ├───────────────▶│                       │
    │                │ checkpoint_.reset(new CheckpointSave(save_cfg))
    │                │                       │
    │ process(S,W)   │                       │
    ├───────────────▶│                       │
    │                │ ... pipeline ...      │
    │                │ checkpoint_->save_c4_peak(peaks)  // ← теперь сохраняется!
    │    result      │                       │
    │◀───────────────┤                       │
```

---

## Seq-4: Chunking для больших данных (> VRAM)

```
 UserApp       AntennaProcessor_v1   Stream0   Stream1    Stream2   WelfordAccum
    │                │               │          │          │            │
    │ process(S,W)   │               │          │          │            │
    ├───────────────▶│               │          │          │            │
    │                │  // Данные > VRAM → chunk_size = 65536 samples  │
    │                │  for chunk in [0, n_chunks):                     │
    │                │               │          │          │            │
    │                │  hipMemcpyAsync(chunk_S)             │            │
    │                ├──────────────▶│          │          │            │
    │                │               │          │          │            │
    │                │  welford_fused(chunk_S)  │          │            │
    │                │──────────────────────────▶          │            │
    │                │               │ acc.merge(chunk_stats)──────────▶│
    │                │               │                     │            │
    │                │  GEMM(chunk_S, W) → chunk_X         │            │
    │                │───────────────────────────────────▶ │            │
    │                │  hipFFT(chunk_X) → chunk_spectrum   │            │
    │                │───────────────────────────────────▶ │            │
    │                │  // Spectrum чанки накапливаются или сразу в Branch │
    │                │               │          │          │            │
    │                │  end for      │          │          │            │
    │                │               │          │          │            │
    │                │  // Финализируем Welford (variance из всех чанков)
    │                │  acc.finalize() ────────────────────────────────▶│
    │                │  ◀── final StatisticsResult[N_ant] ──────────────┤
    │                │               │          │          │            │
    │                │  // Медиана: P² online (при chunking) или HBM radix (если влезает)
    │                │               │          │          │            │
    │    result      │               │          │          │            │
    │◀───────────────┤               │          │          │            │

CHUNKING ПАРАМЕТРЫ:
  chunk_size       = 65536 сэмплов (256 МБ / чанк при N_ant=256)
  n_chunks         = ceil(N_samples / 65536) = 19 для 1.2M
  VRAM overhead    = 3 × chunk_size × N_ant × 8 = 3 × 256 МБ = 768 МБ
  Медиана при chunk: P²-онлайн (< 0.1% погрешность на 1.2M)
  ПРЕДУПРЕЖДЕНИЕ: при chunking FFT по всему N_samples = НЕВОЗМОЖНО
                  нужна постановка задачи: FFT делается по чанкам!
                  (только для статистики; FFT должен работать с полными данными)
```

---

## Seq-5: CheckpointSave (save_c1 = true)

```
 AntennaProcessor_v1    CheckpointSave     FileSystem        GPU
    │                    │                 │               │
    │                    │                 │               │
    │   [После DMA load] │                 │               │
    │                    │                 │               │
    │ save_c1_signal(d_S, N_ant, N_samples, fs, gpu_id)   │
    ├───────────────────▶│                 │               │
    │                    │ path = Logs/GPU_{gpu_id}/antenna_processor/
    │                    │         {date}/{time}/C1_signal.bin
    │                    │                 │               │
    │                    │ hipMemcpy(h_buf, d_S, 2.5 ГБ, D2H)
    │                    │────────────────────────────────▶│
    │                    │                 │               │
    │                    │ (блокирующая копия: ~78мс PCIe, ~2.6мс если GPU→shared)
    │                    │                 │               │
    │                    │ write_binary_file(path, header, h_buf)
    │                    ├────────────────▶│               │
    │                    │ ◄─── OK ────────┤               │
    │                    │                 │               │
    │  ◄─── OK ──────────┤                 │               │
    │                    │                 │               │
    │   [Продолжение pipeline...]          │               │
    │                    │                 │               │

ПРИМЕЧАНИЕ: save_c1_signal = БОЛЬШОЙ (2.5 ГБ)!
            Используется ТОЛЬКО для глубокой отладки.
            В production: cfg.save_cfg = nullptr → NullCheckpointSave::save_c1_signal() = {}

            save_c4_result = МАЛЕНЬКИЙ (12 КБ) → дёшево, можно включить всегда:
            cfg.c4_result = true  ← по умолчанию в CheckpointSaveConfig
```

---

## Seq-6: FFT Mirror Folding (Note #2)

```
 ParabolaBranchStrategy    fold_fft_mirror()    UserApp
        │                        │                 │
        │                        │                 │
        │ execute(d_spectrum) → MaxValue raw        │
        │  peak_bin = 1,500,000  │                 │
        │  nFFT    = 2,097,152   │                 │
        │  sample_rate = 1.2MHz  │                 │
        │                        │                 │
        │ AntennaProcessor_v1 calls: fold_fft_mirror(result)
        │────────────────────────▶                 │
        │                        │                 │
        │                  for each peak:          │
        │                  peak_bin = 1,500,000    │
        │                  nFFT/2   = 1,048,576    │
        │                  peak_bin > nFFT/2 → TRUE → это отрицательная частота! │
        │                                          │
        │                  freq_hz = (peak_bin - nFFT) * fs / nFFT
        │                          = (1,500,000 - 2,097,152) * 1,200,000 / 2,097,152
        │                          = -597,152 * 0.5721
        │                          = -341,601 Гц  (= -341.6 kHz)
        │                                          │
        │                  peak.frequency_hz = -341,601  ← отрицательная!
        │                  peak.bin_folded   = -597,152  ← бин в [-N/2, -1]
        │                                          │
        │                  ◄─── folded result ─────┤
        │                        │                 │
        ◄──── return to AntennaProcessor_v1 ──────────┤
                                 │                 │
                                 │ AntennaResult включает folded frequency!
                                 │────────────────▶│
                                 │                 │

ПРАВИЛО:
  Если full_spectrum_search = false: search_range = nFFT/2 → пики только в [0, nFFT/2)
  Если full_spectrum_search = true:  search_range = nFFT   → пики в [0, nFFT)
  В обоих случаях fold_fft_mirror() переводит бины к реальным частотам:
    bin в [0, nFFT/2]    → freq = bin * fs/nFFT       (положительные)
    bin в [nFFT/2+1, N-1] → freq = (bin-nFFT) * fs/nFFT (отрицательные)
```

---

*Создано: 2026-03-06*
*Связанные документы: [C3 Component](AP_C3_Component.md), [C4 Code](AP_C4_Code.md)*
