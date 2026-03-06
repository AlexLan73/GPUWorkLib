# Antenna Array Processing Pipeline — Architecture Diagram
# AMD Radeon 9070 (RDNA4/gfx1201) & MI100 (CDNA/gfx908)
# Версия: v1.2 — добавлены: min/max pre-FFT fusion, post-FFT ветки 2/3 (все пики vs один+парабола)

!!!!!!!!!!!!!!!
неправтльно называется 
 modules/antenna_processor/
  ├── include/
  │   ├── data_matrix.hpp              # DataMatrix + Descriptor (гибкий размер)

долюно быть


 modules/strategies/
  ├── include/
  │   ├── data_matrix.hpp              # DataMatrix + Descriptor (гибкий размер)

это есть в Doc\Modules\strategies\AP_INDEX.md

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
    Малый:   256  × 1 200 000 × 8 байт ≈ 2.5 ГБ
    Большой: 3500 × 2 500     × 8 байт ≈ 70 МБ

  WeightsMatrix: N_ant × N_samples, complex<float>
    Динамическая: меняется и размером и коэффициентами
    При смене размера — перевыделение GPU-буфера

  DataFormatRegistry: читает описание форматов из config/data_formats/*.json
  config/data_formats/
  ├── antenna_raw_cf32.json        ← входные данные антенн
  ├── weights_matrix_cf32.json     ← матрица фазовых весов
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
  │  d_data[N_ant × N_samples]      d_weights[N_ant × N_samples]                        │
  │  d_hamming[N_samples]  ← вычисляется 1 РАЗ при старте потока / смене N_samples     │
  └────────────────────────────────┬────────────────────────────────────────────────────┘
                                   │
                                   -----------> Save 0 DATA ( d_data[N_ant × N_samples] ) + meta
                ┌──────────────────┴──────────────────────┐
                │                                          │
                ▼                                          ▼
  ┌───────────────────────────────┐       ┌─────────────────────────────────────────────┐
  │  HIP Stream 1: STATISTICS     │       │  HIP Stream 2: MAIN PIPELINE                │
  │  ─────────────────────────── │       │  ─────────────────────────────────────────── │
  │                               │       │                                               │
  │  Читает d_data напрямую       │       │  STEP A: Kernel Fusion                        │
  │  (не трогает d_weights)       │       │  ═══════════════════════════════             │
  │                               │       │  kernel: phase_correct_hamming               │
  │  welford_fused (per row):     │       │                                               │
  │  • mean (complex)             │       │  За ОДИН проход по памяти:                   │
  │  • variance of |z|            │       │  out[i,j] = data[i,j]                        │
  │  • std_dev of |z|             │       │           * weights[i,j]   ← phase correct   │
  │  • mean_magnitude             │       │           * hamming[j]     ← window (fused!) │
  │  • min |z| + idx  ← FUSED!   │       │                                               │
  │  • max |z| + idx  ← FUSED!   │       │  Читает: d_data  (2.5 ГБ)                    │
  │    (+0 доп. BW — уже чит.)   │       │          d_weights (2.5 ГБ)                   │
  │                               │       │          d_hamming (5 МБ, в L2 кэше)          │
  │  radix_sort + extract_medians │       │  Пишет:  d_out   (2.5 ГБ)                    │
  │  • median of |z|              │       │  Итого: ~5.7 мс на 9070 (960 ГБ/с)          │
  │  [точная если всё в VRAM,     │       │                                               │
  │   P² при chunking]            │       │  Grid:  (N_ant, ceil(N_samples / 256))        │
  │                               │       │  Block: (256)                                 │
  │  Output (per antenna):        │       │  Каждая антенна → независимый блок           │
  │  • MeanResult[N_ant]          │       │                                               │
  │  • StatisticsResult[N_ant]    │       │           ↓ event_fused_done                  │
  │  • MedianResult[N_ant]        │       │                                               │
  │  • MinMaxResult[N_ant] ←NEW   │       │  STEP B: FFT batch                            │
  │                               │       │  ═══════════════════════════════             │
  │  Welford online при chunking: │       │  hipfftExecC2C:                               │
  │  накапливает count/mean/M2    │       │  N_ant независимых FFT по N_samples точек    │
  │  через все чанки              │       │  batch = N_ant, plan = HIPFFT_C2C forward     │
  │                               │       │                                               │
  └──────────┬────────────────────┘       │  Output: d_spectrum[N_ant × N_samples]        │
             │ event_stats_done           │  (частотная область)                          │
             ▼                           └───────────────────────┬───────────────────────┘
  ┌──────────────────────────┐                                   │ event_fft_done
  │  STATS OUTPUT (PRE-FFT)  │                                   ▼
  │  N_ant × {               │           ┌───────────────────────────────────────────────┐
  │    mean (cf32),          │           │  d_spectrum[N_ant × N_samples]                │
  │    variance (f32),       │           │  (cf32, частотная область)                    │
  │    std_dev (f32),        │           └───────────────────────────────────────────────┘
  │    median (f32),         │
  │    min_mag + idx,  ←NEW  │
  │    max_mag + idx   ←NEW  │
  │  }                       │
  └──────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ИНИЦИАЛИЗАЦИЯ HAMMING (один раз при старте / смене N_samples)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Получить N_samples из нового потока данных
  2. Если N_samples изменился — перевыделить d_hamming на GPU
  3. Вычислить и загрузить:  w[n] = 0.54 - 0.46 * cos(2π·n / (N_samples-1))
  4. d_hamming остаётся в VRAM и переиспользуется для всех антенн

  Размер: N_samples × 4 байт (float) = 1.2M × 4 = 4.8 МБ → помещается в L2 кэш GPU!
  Эффект: d_hamming читается из кэша, не из VRAM → overhead ≈ 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ПАРАЛЛЕЛИЗМ и СИНХРОНИЗАЦИЯ (HIP Events)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Stream 0 (DMA) :  [=====Load Data + Weights=====]──► event_data_ready
  Stream 1 (Stat):  ..................[===Welford + radix_sort===]──► event_stats_done
  Stream 2 (Main):  ..................[=phase_correct_hamming=][=========FFT=========]──► event_fft_done

  Без kernel fusion (старая схема):
  Stream 2:  .....[=Phase=]──►[=Hamming=]──►[=====FFT=====]    ← 2 прохода по 2.5 ГБ

  С kernel fusion (новая схема):
  Stream 2:  .....[===PhaseHamming===]──►[=====FFT=====]        ← 1 проход по 2.5 ГБ
                  экономия: ~5.2 мс на 9070 (960 ГБ/с)

  HIP Events:
  • event_data_ready      → запускает Stream1 и Stream2 одновременно
  • event_fused_done      → запускает FFT (Stream2)
  • event_stats_done      → CPU читает статистику
  • event_fft_done        → CPU / следующий модуль читает спектр

  Chunking (если данные > VRAM):
  • Разбиваем по N_samples: chunk = 65536 сэмплов ≈ 256 МБ/чанк
  • Welford накапливает count/mean/M2 через все чанки → финальный variance в конце
  • Медиана: точная (radix sort) если всё в VRAM; P²-онлайн при chunking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 KERNEL: phase_correct_hamming (HIP, fusioned)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  __global__ void phase_correct_hamming(
      const float2_t* __restrict__ d_data,      // [N_ant × N_samples]
      const float2_t* __restrict__ d_weights,   // [N_ant × N_samples]
      const float*   __restrict__ d_hamming,    // [N_samples]  — из кэша
      float2_t*      __restrict__ d_out,        // [N_ant × N_samples]
      unsigned int N_ant,
      unsigned int N_samples)
  {
      unsigned int ant    = blockIdx.x;
      unsigned int sample = blockIdx.y * blockDim.x + threadIdx.x;
      if (ant >= N_ant || sample >= N_samples) return;

      unsigned int idx = ant * N_samples + sample;
      float2_t d = d_data[idx];
      float2_t w = d_weights[idx];
      float    h = d_hamming[sample];   // broadcast по антеннам, кэшируется

      // Комплексное умножение d * w, затем умножение на скаляр h
      float re = (d.x * w.x - d.y * w.y) * h;
      float im = (d.x * w.y + d.y * w.x) * h;
      d_out[idx] = {re, im};
  }

  Grid:  dim3(N_ant, ceil(N_samples / 256))
  Block: dim3(256)
  Shared memory: не нужна — чисто BW-bound операция

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 fft_maxima: добавление Hamming (минимальные изменения)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Текущий pipeline fft_maxima:
    pad_data → hipFFT → post_kernel (magnitude + max + парабола)

  Изменение: pad_data → apply_hamming_and_pad  (один новый параметр float* hamming)
    Если pos < count_points: out[gid] = input[src] * hamming[pos]  ← окно на реальные точки
    Если pos >= count_points: out[gid] = {0, 0}                     ← нули не трогаем!

  Нет нового прохода по памяти. Нет переписывания pipeline. Только один параметр.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 СТРУКТУРА МОДУЛЯ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  modules/antenna_processor/
  ├── include/
  │   ├── data_matrix.hpp              # DataMatrix + Descriptor (гибкий размер)
  │   ├── data_format.hpp              # IDataFormat (abstract) + DataFormatRegistry
  │   ├── data_format_binary.hpp       # BinaryDataFormat (production)
  │   ├── data_format_json_bin.hpp     # JsonHeaderBinaryFormat (debug)
  │   ├── phase_hamming_rocm.hpp       # PhaseCorrectorROCm (fused kernel)
  │   └── antenna_processor.hpp        # AntennaArrayProcessor (главный класс)
  ├── src/
  │   ├── data_matrix.cpp
  │   ├── data_format.cpp
  │   ├── data_format_binary.cpp
  │   ├── data_format_json_bin.cpp
  │   ├── phase_hamming_rocm.cpp
  │   └── antenna_processor.cpp
  ├── kernels/
  │   └── phase_hamming.hip            # HIP: fused phase_correct + hamming (один kernel)
  └── tests/
      ├── all_test.hpp
      ├── test_data_matrix.hpp
      ├── test_phase_hamming.hpp
      └── README.md

  config/data_formats/
  ├── antenna_raw_cf32.json
  ├── weights_matrix_cf32.json
  ├── fft_output_cf32.json
  └── stats_output_f32.json

  Переиспользуемые модули (уже есть в проекте):
  • StatisticsProcessor   → modules/statistics/
  • FFTProcessor (ROCm)   → modules/fft_processor/  (hipFFT batch)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 СТРУКТУРЫ: MinMaxResult, AllMaximaResult (новые)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  // PRE-FFT: результат min/max по модулю сигнала (временная область)
  // Вычисляется В WELFORD_FUSED — бесплатно (нет доп. проходов по памяти)
  struct MinMaxResult {
    uint32_t beam_id;
    float    min_magnitude;   // min |z| по антенне
    uint32_t min_index;       // позиция минимума в массиве samples
    float    max_magnitude;   // max |z| по антенне
    uint32_t max_index;       // позиция максимума в массиве samples
  };

  // POST-FFT: результат поиска ВСЕХ локальных максимумов спектра (ветка 2)
  // max_peaks_count — параметр конфигурации (например, 64 или 256)
  struct AllMaximaResult {
    uint32_t beam_id;
    uint32_t n_peaks;                      // кол-во найденных пиков
    MaxValue peaks[MAX_PEAKS_COUNT];       // локальные пики > CFAR threshold
    float    min_spec_magnitude;           // глобальный минимум спектра
    uint32_t min_spec_index;
    float    max_spec_magnitude;           // глобальный максимум спектра
    uint32_t max_spec_index;
  };

  // MaxValue — уже существует в fft_maxima (совместим):
  struct MaxValue {
    uint32_t index;
    float    real, imag;
    float    magnitude;
    float    phase;             // градусы
    float    freq_offset;       // [-0.5 .. +0.5] sub-bin (только для ветки 3)
    float    refined_frequency; // Гц
    uint32_t pad;
  };

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 KERNEL: minmax_spectrum (новый, для ветки 2 — post-FFT global min/max)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  __global__ void minmax_spectrum(
      const float2_t* __restrict__ spectrum,  // [N_ant × nFFT], cf32
      float*   __restrict__ out_min_mag,      // [N_ant]
      uint32_t* __restrict__ out_min_idx,     // [N_ant]
      float*   __restrict__ out_max_mag,      // [N_ant]
      uint32_t* __restrict__ out_max_idx,     // [N_ant]
      uint32_t N_ant,
      uint32_t nFFT,
      uint32_t search_range)                  // обычно nFFT/2 (односторонний спектр)
  {
      // Один блок / луч, 256 потоков
      // Каждый поток находит локальный min+max
      // Tree reduction в shared memory → записывает глобальный min+max

      __shared__ float  s_min_mag[256], s_max_mag[256];
      __shared__ uint32_t s_min_idx[256], s_max_idx[256];

      uint32_t beam = blockIdx.x;
      uint32_t lid  = threadIdx.x;
      // ... tree reduction, затем lid==0 пишет результат
  }

  Grid:  dim3(N_ant)         // один блок на луч
  Block: dim3(256)
  Используется: в ВЕТКЕ 2 (после detect_all_maxima)
  Примечание: min/max в ВЕТКЕ 2 — ищутся по спектру ПОСЛЕ FFT (частотная область)
              min/max в STATS (welford_fused) — по сигналу ДО FFT (временная область)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ (256 × 1 200 000 cf32, 9070 ~960 ГБ/с)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Без fusion:
  ├── Phase Correct: read 5 ГБ (data+weights) + write 2.5 ГБ → ~7.8 мс
  ├── Hamming:       read 2.5 ГБ + write 2.5 ГБ              → ~5.2 мс
  └── Итого kernel fusion экономит: ~5.2 мс

  С fusion (phase_correct_hamming):
  ├── Fused kernel:  read 5 ГБ (data+weights) + write 2.5 ГБ → ~7.8 мс  (hamming из L2!)
  ├── FFT batch:     256 × hipFFT(1.2M) → ~TBD
  ├── Statistics:    read 2.5 ГБ → ~2.6 мс  (параллельно с fused kernel)
  └── Итого: ~7.8 + FFT мс  (Stats скрыты за fused kernel)

  Малый вариант: 3500 × 2500 (70 МБ) → всё << 1 мс

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 РЕШЕНИЯ (подтверждены)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [OK] Порядок: data → phase_correct_hamming (fused) → hipFFT batch → spectrum
  [OK] Stats параллельно с fused kernel (оба читают d_data независимо)
  [OK] Данные уже после LchFarrow на входе
  [OK] Матрица весов: N_ant × N_samples, element-wise, динамическая
  [OK] Hamming: инициализируется при старте / смене N_samples; 4.8 МБ → в L2 кэше
  [OK] Kernel fusion: phase_correct + hamming → один kernel, один проход по памяти
  [OK] fft_maxima: pad_data → apply_hamming_and_pad (один параметр, нет переписывания)
  [OK] Медиана: ТОЧНАЯ (radix sort) — обязательно, т.к. используется CFAR
  [OK] Каждая антенна независима: нет суммирования между строками
  [OK] hipFFT batch: N_ant параллельных FFT длиной N_samples
  [OK] Хемминг: ТОЛЬКО перед FFT (DSP правило), реализован в kernel fusion с весами
  [OK] Min/Max PRE-FFT: fused в welford_fused — БЕСПЛАТНО (нет доп. прохода по памяти)
  [OK] Min/Max POST-FFT: отдельный kernel minmax_spectrum (1 блок/луч, tree reduction)
  [OK] Post-FFT ВЕТКА 2: все локальные пики + CFAR фильтр (detect_all_maxima) + global min/max
  [OK] Post-FFT ВЕТКА 3: один глобальный максимум + паrabola (sub-bin refinement)
  [OK] Ветки 2 и 3 ПЕРЕКЛЮЧАЕМЫЕ (switch/flag) — работает только одна за раз
  [OK] CFAR threshold = median * alpha (из Stream 1) → в ветку 2 как критерий пиков
  [OK] AllMaximaResult: peaks[], n_peaks, min_spec+idx, max_spec+idx (per antenna)
  [OK] MinMaxResult: min_mag+idx, max_mag+idx (per antenna, временная область)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 МЕДИАНА — математический анализ точности (почему точная, а не P²)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ПРИМЕНЕНИЕ: CFAR (Constant False Alarm Rate) — адаптивный порог обнаружения цели.
  РЕШЕНИЕ: нужна ТОЧНАЯ медиана через radix sort (уже есть в StatisticsProcessor).

  ── Математика: связь медианы с шумом ────────────────────────────────────────────────

  Шум: Re ~ N(0,σ²), Im ~ N(0,σ²)  →  |z| = sqrt(Re²+Im²) ~ Rayleigh:
    f(r) = (r/σ²) · exp(−r²/2σ²)

  Медиана Rayleigh:
    M = σ · √(2·ln2) = 1.1774·σ   →   σ_оценка = M_оценка / 1.1774

  Вывод: ошибка медианы ε напрямую переносится на ошибку оценки σ шума (тоже ε).

  ── Влияние ошибки медианы на CFAR ───────────────────────────────────────────────────

  Порог CFAR: T = α·σ_оценка,   Pfa = exp(−T²/2σ²)

  При ошибке медианы ε:
    T_оценка = T·(1+ε)   →   Pfa_факт = Pfa_цель ^ (1+ε)²

  ┌─────────────┬────────────────────────────┬──────────────────────────────────────────┐
  │ Pfa_цель    │ ε = +1% (порог завышен)    │ ε = −1% (порог занижен)                  │
  ├─────────────┼────────────────────────────┼──────────────────────────────────────────┤
  │ 10⁻³        │ 8.69×10⁻⁴  (−13%)         │ 1.15×10⁻³  (+15%)                        │
  │ 10⁻⁶        │ 7.57×10⁻⁷  (−24%)         │ 1.32×10⁻⁶  (+32%) ← ложных тревог в 1.3× │
  │ 10⁻⁸        │ 6.90×10⁻⁹  (−31%)         │ 1.45×10⁻⁸  (+45%) ← ложных тревог в 1.5× │
  └─────────────┴────────────────────────────┴──────────────────────────────────────────┘

  Важно: чем жёстче Pfa (важнее обнаружение), тем сильнее влияет ошибка медианы.
  При ε = −1%: порог занижен → система выдаёт в 1.3–1.5 РАЗА БОЛЬШЕ ложных тревог!

  Потери по SNR:
    ΔT_дБ = 20·log₁₀(1.01) = 0.086 дБ  ← пренебрежимо мало (стандартные потери 1–3 дБ)
    Но изменение Pfa — уже критично для CFAR.

  ── Реальная точность P² при N = 1 200 000 ───────────────────────────────────────────

  1% — консервативная (пессимистичная) оценка для P²-алгоритма.
  На N = 1.2M точек реальная погрешность P² для Rayleigh: ε < 0.05–0.1%

  При ε = 0.1% (реальная P² на 1.2M):
    Pfa_факт = (10⁻⁶)^(1.001²) = 9.72×10⁻⁷   → отклонение −2.8%  ← уже приемлемо

  НО: зачем рисковать, если точная медиана доступна и уже реализована?

  ── Решение: точная медиана (radix sort) всегда когда влезает в VRAM ─────────────────

  Память для точной медианы:
    256 × 1.2M × 4 байт (float модули) = 1.2 ГБ
    + буфер radix sort                  = 1.2 ГБ
    ─────────────────────────────────────────────
    Итого:                              ≈ 2.4 ГБ → на 9070 (16 ГБ VRAM) ✅
                                                  → на MI100 (32 ГБ HBM2) ✅

  Реализация: compute_median() из StatisticsProcessor — уже готова, использует rocPRIM.

  Правило выбора алгоритма медианы:
    Данные влезают в VRAM  →  точная медиана (radix sort, StatisticsProcessor)
    Принудительный chunking →  P²-онлайн (< 0.1% для мониторинга)
                                ВНИМАНИЕ: для CFAR при chunking нужен отдельный подход!

  ── Почему Хемминг только перед FFT (DSP правило) ────────────────────────────────────

  Оконная функция применяется к ВРЕМЕННОМУ сигналу перед преобразованием в частоту.
  Цель: подавить боковые лепестки спектра (spectral leakage).
  Применение ПОСЛЕ FFT к спектру — математически неверно и бессмысленно.

  Где в нашем kernel:
    out[ant, n] = data[ant, n]      ← сигнал (время)
                × weights[ant, n]   ← фазовая коррекция
                × hamming[n]        ← окно Хемминга (по времени!)
                ↓
              hipFFT → spectrum      ← частота

  Это верный порядок. Результирующий спектр имеет подавленные боковые лепестки,
  что критично для точного CFAR-порога (уменьшает ложные тревоги от утечки спектра).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 POST-FFT: ПЕРЕКЛЮЧАЕМЫЕ ВЕТКИ (после event_fft_done)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                     d_spectrum[N_ant × N_samples]
                                  │
              ┌───────────────────┴──────────────────┐
              │      ВЫБОР РЕЖИМА (switch/flag)       │
              └──────────┬──────────────────┬─────────┘
                         │                  │
                    ВЕТКА 2            ВЕТКА 3
              (переключаема с 3)  (переключаема с 2)
                         │                  │
                         ▼                  ▼
  ┌──────────────────────────────┐  ┌──────────────────────────────────────────────┐
  │  ВЕТКА 2:                    │  │  ВЕТКА 3:                                    │
  │  ВСЕ ЛОКАЛЬНЫЕ ПИКИ          │  │  ОДИН ГЛОБАЛЬНЫЙ МАКСИМУМ + ПАРАБОЛА        │
  │  + GLOBAL MIN/MAX            │  │  ─────────────────────────────────────────── │
  │  ─────────────────────────── │  │                                              │
  │                              │  │  Kernel: post_kernel_one_peak                │
  │  A) detect_all_maxima        │  │  (адаптирован из fft_maxima)                │
  │     (адаптирован из          │  │                                              │
  │      fft_maxima + CFAR):     │  │  1. Параллельная редукция (256 потоков/луч): │
  │                              │  │     tree reduction → global MAX per beam     │
  │     - |z| = mag(spectrum[i]) │  │  2. Парабола (3-точечная):                  │
  │     - 3-точечный тест:       │  │     offset = 0.5*(y_l - y_r) /              │
  │       mag[i] > mag[i-1] &&   │  │              (y_l - 2*y_c + y_r)            │
  │       mag[i] > mag[i+1]      │  │     clamp(offset, -0.5, +0.5)              │
  │     - CFAR фильтр:           │  │  3. refined_freq = (idx+offset) * BW        │
  │       mag > median * alpha   │  │     BW = sample_rate / nFFT                 │
  │       (median из Stream 1)   │  │                                              │
  │     - Blelloch prefix scan   │  │  Output per antenna:                         │
  │       (stream compaction)    │  │    MaxValue[N_ant] = {                       │
  │                              │  │      index,                                  │
  │  B) minmax_spectrum (новый): │  │      re, im,                                 │
  │     - 1 блок / луч           │  │      magnitude,                              │
  │     - 256 потоков            │  │      phase,        ← atan2(im,re)*180/π     │
  │     - tree reduction         │  │      freq_offset,  ← [-0.5 .. +0.5]         │
  │     - min + max + idx        │  │      refined_frequency ← Гц                 │
  │       за ОДИН проход по |z|  │  │    }                                         │
  │                              │  └──────────────────────────────────────────────┘
  │  Output per antenna:         │
  │    AllMaximaResult = {       │
  │      peaks[MAX_PEAKS],  ← локальные пики > CFAR │
  │      n_peaks,           ← кол-во найденных пиков │
  │      min_spec + idx,    ← глобальный минимум     │
  │      max_spec + idx     ← глобальный максимум    │
  │    }                         │
  └──────────────────────────────┘

  Примечание: результаты PRE-FFT (min_mag+idx, max_mag+idx из welford_fused)
  и POST-FFT (min_spec+idx, max_spec+idx из minmax_spectrum) — разные:
    PRE  — min/max по модулю сигнала во временной области (до окна/FFT)
    POST — min/max по модулю спектра в частотной области (после FFT)
```

---
*Создано: 2026-03-06*
*Обновлено: 2026-03-06 v0.6 — min/max fusion (pre-FFT), post-FFT ветки 2/3 (все пики vs один+парабола)*


### дополнения
1. Немного не так -2 ветка FFT Все глобальные мак/мин
2. Нудно получить статистику по сигналу  - обсудаем 
   2.1 средняя, медиана, std, дисперсия модет сразу добавить в этот поток и всех мак/мин- вроде логично
   2.2 повторить пункт 2.1 после перемножений матриц
3. Я не понял с Хемменгов мы говорили что от будет в FFT а в диаграме он находится в перемножении матриц
4. Заказчик хочет сохранять промежуточные данные 
  4.1 первую точку я пометил полсле получения данных из памяти GPU
  4.2 после реремножений матриц перед Хеммингом
  4.3 после FFT мах/мин
  4.4 после FFT один мах + парабала
 
5. продумай и предложи варианты реализации для рабочего варианта не нужны ветки отладуи с сохранением данных, может сделать как с профилированием (посмотри решение в .claude\worktrees\hungry-bohr\modules\fft_processor). Данные сохранять по принцепу Logs  

6. может сделать наборы из статистики к примеру!
  6.1 сразу все 2.1 
  6.2 средняя, медиана
  6.3 средняя, медиана, всех мак/мин
  6.4 std, дисперсия
  6.5 - потом еще добавиться