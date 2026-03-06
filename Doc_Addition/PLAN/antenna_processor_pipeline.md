# Antenna Array Processing Pipeline — Architecture Diagram
# AMD Radeon 9070 (RDNA4/gfx1201) & MI100 (CDNA/gfx908)
# Версия: v0.3 (ФИНАЛЬНАЯ архитектура подтверждена Alex 2026-03-06)

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║              ANTENNA ARRAY PROCESSING PIPELINE (GPU)                                   ║
║              AMD Radeon 9070 (RDNA4/gfx1201)  &  MI100 (CDNA/gfx908)                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ВХОДНЫЕ ДАННЫЕ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DataMatrix: N_ant × N_samples, complex<float>   (после LchFarrow — данные выровнены)
  Варианты:
    Малый:   256  × 1 200 000 × 8 байт ≈ 2.5 ГБ  → chunking нужен на 9070 (16 ГБ VRAM)
    Большой: 3500 × 2 500     × 8 байт ≈ 70 МБ   → всё помещается сразу

  WeightsMatrix: N_ant × N_samples, complex<float>  (та же размерность, что и данные)
    Динамическая: меняется и размером и коэффициентами
    При смене размера — перевыделение GPU-буфера

  DataFormatRegistry: читает описание форматов из config/data_formats/*.json
  config/data_formats/
  ├── antenna_raw_cf32.json        ← входные данные антенн
  ├── weights_matrix_cf32.json     ← матрица фазовых весов
  ├── beams_output_cf32.json       ← выходные лучи (после фазовой коррекции + Хемминг)
  ├── fft_output_cf32.json         ← результат FFT (частотная область)
  └── stats_output_f32.json        ← статистика по антеннам

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GPU PIPELINE (HIP Streams + Events)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

       HOST RAM
  ┌────────────────────────────────────────────┐
  │  DataMatrix[N_ant × N_samples]             │  cf32, после LchFarrow
  │  WeightsMatrix[N_ant × N_samples]          │  cf32, фазовая коррекция
  └────────────────┬───────────────────────────┘
                   │ HIP Stream 0: DMA Host→GPU
                   │ (если данные не влезают — chunking по N_samples)
                   ▼ event_data_ready
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │                            GPU VRAM                                                  │
  │  d_data[N_ant × N_samples]     d_weights[N_ant × N_samples]                         │
  │  d_hamming[N_samples]  ← вычисляется 1 РАЗ при старте потока, хранится в VRAM      │
  └────────────────────────────────┬───────────────────────────────────────────────────-─┘
                                   │
                ┌──────────────────┴─────────────────────┐
                │                                         │
                ▼                                         ▼
  ┌──────────────────────────────┐       ┌──────────────────────────────────────────────┐
  │  HIP Stream 1: STATISTICS    │       │  HIP Stream 2: BEAMFORMING PIPELINE          │
  │  ────────────────────────── │       │  ──────────────────────────────────────────── │
  │                              │       │                                                │
  │  Kernel: welford_fused       │       │  STEP A: Phase Correction (element-wise)      │
  │  Per antenna (per row):      │       │                                                │
  │  • mean (complex)            │       │  kernel: phase_correct                         │
  │  • variance of |z|           │       │  d_beams[i,j] = d_data[i,j] * d_weights[i,j] │
  │  • std_dev of |z|            │       │  (Hadamard product, N_ant × N_samples точек)  │
  │  • mean_magnitude            │       │                                                │
  │                              │       │  Grid: (N_ant, ceil(N_samples/256))           │
  │  Kernel: radix_sort +        │       │  Block: (256)                                  │
  │          extract_medians     │       │  BW-bound: очень быстро                        │
  │  • median of |z|             │       │                                                │
  │    [P²-приближение при       │       │           ↓ event_phase_done                   │
  │     streaming, точное при    │       │                                                │
  │     всё в VRAM]              │       │  STEP B: Hamming Window                        │
  │                              │       │                                                │
  │  Output (per antenna):       │       │  kernel: apply_hamming                         │
  │  • MeanResult[N_ant]         │       │  w[n] = 0.54 - 0.46·cos(2π·n/(N_samples-1))  │
  │  • StatisticsResult[N_ant]   │       │                                                │
  │  • MedianResult[N_ant]       │       │  d_beams[ant, n] *= d_hamming[n]              │
  │                              │       │  Grid: (N_ant, ceil(N_samples/256))            │
  │  Welford online:             │       │  Block: (256)                                  │
  │  при chunking — накапли-     │       │  ПАРАЛЛЕЛЬНО все N_ant лучей сразу             │
  │  вает через все чанки        │       │                                                │
  │                              │       │           ↓ event_hamming_done                 │
  └──────────┬───────────────────┘       │                                                │
             │                           │  STEP C: FFT (hipFFT / rocFFT)                 │
             │                           │                                                │
             │                           │  hipfftExecC2C batch:                          │
             │                           │  N_ant независимых FFT длиной N_samples        │
             │                           │  batch = N_ant                                 │
             │                           │  plan: HIPFFT_C2C, forward                     │
             │                           │                                                │
             │                           │  Output: d_spectrum[N_ant × N_samples]         │
             │                           │  (частотная область)                           │
             │                           └──────────────────┬─────────────────────────────┘
             │                                              │
             ▼                                              ▼
  ┌──────────────────────────┐          ┌──────────────────────────────────────────────┐
  │  STATS OUTPUT            │          │  SPECTRUM OUTPUT                             │
  │  N_ant × {               │          │  N_ant × N_samples (cf32, frequency domain)  │
  │    mean (cf32),          │          │  → запись через BinaryDataFormat             │
  │    variance (f32),       │          │  → fft_output_cf32.json описывает формат     │
  │    std_dev (f32),        │          └──────────────────────────────────────────────┘
  │    median (f32, P²)      │
  │  }                       │
  └──────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ИНИЦИАЛИЗАЦИЯ HAMMING (один раз при старте потока)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  При получении нового потока данных (возможно N_samples изменился):
  1. Вычислить вектор Хемминга на CPU или GPU (N_samples float)
     w[n] = 0.54 - 0.46 * cos(2π*n / (N_samples-1))
  2. Загрузить в d_hamming[N_samples] на GPU
  3. При изменении N_samples — перевыделить буфер

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ПАРАЛЛЕЛИЗМ и СИНХРОНИЗАЦИЯ (HIP Events)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Stream 0 (DMA) :  [=====Load Data + Weights=====] → event_data_ready
  Stream 1 (Stat):  .........[===Welford+Sort===]
  Stream 2 (Beam):  .........[=Phase=][=Hamming=][=========FFT=========]

  HIP Events:
  • event_data_ready    → запускает Stream1 и Stream2 одновременно
  • event_phase_done    → запускает Hamming (Stream2)
  • event_hamming_done  → запускает FFT (Stream2)
  • event_stats_done    → CPU читает статистику
  • event_fft_done      → CPU/следующий модуль читает спектр

  Chunking (если данные > VRAM):
  Chunk по N_samples: Welford накапливает онлайн через все чанки (суммирует count/mean/M2).
  Медиана точная при всём в VRAM; P²-приближение при chunking (~1% погрешность на 1.2M точках)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 СТРУКТУРА МОДУЛЯ (предлагаемая)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  modules/antenna_processor/
  ├── include/
  │   ├── data_matrix.hpp              # DataMatrix + Descriptor (гибкий размер)
  │   ├── data_format.hpp              # IDataFormat (abstract) + DataFormatRegistry
  │   ├── data_format_binary.hpp       # BinaryDataFormat (production)
  │   ├── data_format_json_bin.hpp     # JsonHeaderBinaryFormat (debug)
  │   ├── phase_corrector_rocm.hpp     # PhaseCorrectorROCm (element-wise multiply)
  │   ├── window_applicator.hpp        # WindowApplicator (Hamming, инит при старте)
  │   └── antenna_processor.hpp        # AntennaArrayProcessor (главный класс)
  ├── src/
  │   ├── data_matrix.cpp
  │   ├── data_format.cpp
  │   ├── data_format_binary.cpp
  │   ├── data_format_json_bin.cpp
  │   ├── phase_corrector_rocm.cpp
  │   ├── window_applicator.cpp
  │   └── antenna_processor.cpp
  ├── kernels/
  │   ├── phase_correct.hip            # HIP: element-wise complex multiply
  │   └── hamming_window.hip           # HIP: apply Hamming window to all beams
  └── tests/
      ├── all_test.hpp
      ├── test_data_matrix.hpp
      ├── test_phase_corrector.hpp
      ├── test_window_applicator.hpp
      └── README.md

  config/data_formats/
  ├── antenna_raw_cf32.json
  ├── weights_matrix_cf32.json
  ├── beams_output_cf32.json
  ├── fft_output_cf32.json
  └── stats_output_f32.json

  Переиспользуемые модули (уже есть в проекте):
  • StatisticsProcessor   → modules/statistics/
  • FFTProcessor (ROCm)   → modules/fft_processor/   (hipFFT batch)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Вариант: 256 × 1 200 000 cf32  (2.5 ГБ данных)

  AMD Radeon 9070 (RDNA4, ~960 ГБ/с):
  ├── Phase Correct (element-wise): 2.5 ГБ read + 2.5 ГБ write → ~5.2 мс (BW-bound)
  ├── Hamming Window:              2.5 ГБ read + 2.5 ГБ write → ~5.2 мс (BW-bound)
  ├── FFT (256 батчей × 1.2M):    hipFFT → ~TBD (зависит от длины FFT)
  ├── Statistics (Welford):        2.5 ГБ read → ~2.6 мс (параллельно с Phase Correct)
  └── Итого: ~10-15 мс (Phase+Hamming последовательно, Stats параллельно)

  Вариант: 3500 × 2500 cf32  (70 МБ данных) — всё в VRAM, очень быстро
  └── Итого: << 1 мс

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 МЕДИАНА — точность P²-алгоритма
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  P²-алгоритм (Jain & Chlamtac, 1985):
  • Онлайн, O(1) памяти на квантиль, одноходовой
  • Точность на 1.2M точек: погрешность < 0.1–1% от реального значения
  • Работает при streaming/chunking — не нужно хранить все данные

  Точная медиана (radix sort, как в Statistics):
  • Нужно: N_ant × N_samples × 4 байт (float модули) + буфер sort
  • 256 × 1.2M × 4 = ~1.2 ГБ → ещё 1.2 ГБ для sort = 2.4 ГБ → OK на 9070 (16 ГБ VRAM)
  • Рекомендация: использовать точную если всё помещается, P² при chunking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ОТКРЫТЫЕ ВОПРОСЫ v0.2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [OK] Порядок: Данные → Phase Correct (element-wise) → Hamming → FFT
  [OK] Stats параллельно с Phase Correct (оба читают входные данные)
  [OK] Данные уже после LchFarrow на входе
  [OK] Матрица весов: N_ant × N_samples, element-wise, динамическая (размер и коэффициенты)
  [OK] Hamming: инициализируется при старте потока по N_samples, перевыделяется при смене размера
  [OK] Каждая антенна обрабатывается НЕЗАВИСИМО: phase → hamming → FFT  (нет суммирования)
  [OK] hipFFT batch: N_ant параллельных FFT длиной N_samples
  [OK] Выход: spectrum[N_ant × N_samples] — частотная область, каждая строка = одна антенна
```

---
*Создано: 2026-03-06*
*Обновлено: 2026-03-06 v0.2 — после уточнений с Alex*
