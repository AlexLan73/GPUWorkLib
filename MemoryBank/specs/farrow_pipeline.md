# 📝 Farrow Pipeline — Спецификация

> **Модуль**: `Python_test/strategies/`
> **Статус**: 🟡 WIP
> **Платформы**: Python (numpy) — CPU reference для сравнения с GPU
> **Автор**: Alex + Кодо
> **Создано**: 2026-03-08
> **Обновлено**: 2026-03-08

---

## 🎯 Назначение

Сравнение двух pipeline beamforming для ЛЧМ сигналов:

- **Pipeline A** (без Farrow): фазовая коррекция через W матрицу
- **Pipeline B** (с Farrow): временная коррекция через Lagrange interpolation

Pipeline B должен давать **лучшие результаты для ЛЧМ** (широкополосных) сигналов.

---

## 🏗️ Общая архитектура

```
┌──────────────────────────────────────────────────────────┐
│                  ScenarioBuilder                         │
│  ULAGeometry(n_ant, d_ant_m, c=3e8)                    │
│  + Targets (θ, f0, fdev, A)                            │
│  + Jammers (θ, f0, fdev, A)                            │
│  + Noise (σ)                                            │
│                                                          │
│  → S_raw [n_ant × n_samples] complex64                  │
│  → delays [n_ant] — физические задержки антенн (с)      │
└──────────────────────┬───────────────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
    Pipeline A                   Pipeline B
    (без Farrow)                 (с Farrow)
         │                            │
         │                   ┌────────┴────────┐
         │                   │  FarrowDelay     │
         │                   │  delays[n_ant]   │
         │                   │  Lagrange 48×5   │
         │                   │  → S_aligned     │
         │                   └────────┬─────────┘
         │                            │
    ┌────┴────┐              ┌────────┴────────┐
    │ W_phase │              │    W_sum        │
    │ (1/√N)· │              │   1/√N · I      │
    │ exp(-jφ)│              │ (когерентное    │
    │         │              │  суммирование)  │
    └────┬────┘              └────────┬────────┘
         │                            │
    ┌────┴────────────────────────────┴────┐
    │       Beamforming: X = W @ S         │
    │       Window (Hamming) + FFT         │
    │       Peak detection                 │
    └──────────────────┬───────────────────┘
                       │
              ┌────────┴────────┐
              │   Comparison    │
              │  • Peak mag     │
              │  • Freq accuracy│
              │  • SNR gain     │
              │  • BW resolution│
              └─────────────────┘
```

---

## 📊 Пошаговые диаграммы вызовов

### Pipeline A: `PipelineRunner.run_pipeline_a(scenario, steer_theta, steer_freq)`

```
┌─────────────────────────────────────────────────────────────┐
│ ВХОД: scenario = ScenarioBuilder.build()                    │
│   scenario['S']        → S_raw [n_ant × n_samples] complex │
│   scenario['array']    → ULAGeometry                        │
│   scenario['fs']       → sample_rate                        │
│   scenario['n_samples'] → n_samples                         │
│   steer_theta          → угол наведения (deg)               │
│   steer_freq           → частота для фазовой коррекции (Hz) │
└─────────────────────────────┬───────────────────────────────┘
                              │
              Step 0: Input   ▼
┌─────────────────────────────────────────────────────────────┐
│  stats_input = compute_matrix_stats(S_raw)                  │
│  → [ChannelStats × n_ant]                                   │
│  if save_input: → S_raw.npy                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
              Step 1: W_phase ▼
┌─────────────────────────────────────────────────────────────┐
│  delays = ULAGeometry.compute_delays(steer_theta)           │
│  W_phase[b][a] = (1/√N) · exp(-j·2π·steer_freq·τ_a)       │
│  → W [n_ant × n_ant] complex64                              │
│                                                              │
│  ⚠ Коррекция ТОЛЬКО на одной частоте steer_freq!            │
│    Для ЛЧМ → temporal smearing                              │
└─────────────────────────────┬───────────────────────────────┘
                              │
              Step 2: GEMM    ▼
┌─────────────────────────────────────────────────────────────┐
│  X = W_phase @ S_raw                                        │
│  → X_gemm [n_ant × n_samples] complex64                     │
│  stats_gemm = compute_matrix_stats(X)                       │
│  if save_gemm: → X_gemm.npy                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
         Step 3: Window+FFT   ▼
┌─────────────────────────────────────────────────────────────┐
│  nFFT = next_pow2(n_samples) * 2                            │
│  for each beam b:                                           │
│    X_padded[b, :n_samples] = X_gemm[b] * hamming(n_samples) │
│    spectrum[b] = FFT(X_padded[b])                           │
│    magnitudes[b] = |spectrum[b]|                             │
│  → spectrum [n_ant × nFFT] complex64                        │
│  → magnitudes [n_ant × nFFT] float32                        │
│  freq_axis = fftfreq(nFFT, 1/fs)                            │
│  stats_spectrum = compute_matrix_stats(magnitudes)           │
│  if save_spectrum: → spectrum.npy, magnitudes.npy            │
└─────────────────────────────┬───────────────────────────────┘
                              │
         Step 4: Peaks        ▼
┌─────────────────────────────────────────────────────────────┐
│  peaks = find_peaks_per_beam(magnitudes, freq_axis, n=5)    │
│  → List[List[PeakInfo]]                                     │
│  Каждый PeakInfo: beam_id, bin_index, freq_hz, magnitude    │
│  if save_stats: → stats.json                                │
│  if save_results: → results.json                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ВЫХОД           ▼
┌─────────────────────────────────────────────────────────────┐
│  PipelineResult:                                             │
│    .S_raw          [n_ant, n_samples]                        │
│    .W              [n_ant, n_ant]                             │
│    .X_gemm         [n_ant, n_samples]                        │
│    .spectrum       [n_ant, nFFT]                             │
│    .magnitudes     [n_ant, nFFT]                             │
│    .peaks          List[List[PeakInfo]]                       │
│    .stats_input    List[ChannelStats]                         │
│    .stats_gemm     List[ChannelStats]                         │
│    .stats_spectrum List[ChannelStats]                         │
│    .nFFT, .freq_axis                                         │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline B: `PipelineRunner.run_pipeline_b(scenario, steer_theta)`

```
┌─────────────────────────────────────────────────────────────┐
│ ВХОД: scenario = ScenarioBuilder.build()                    │
│   scenario['S']        → S_raw [n_ant × n_samples] complex │
│   scenario['array']    → ULAGeometry                        │
│   scenario['fs']       → sample_rate                        │
│   steer_theta          → угол наведения (deg)               │
│   ⚠ steer_freq НЕ НУЖЕН — Farrow делает временную коррекцию│
└─────────────────────────────┬───────────────────────────────┘
                              │
              Step 0: Input   ▼
┌─────────────────────────────────────────────────────────────┐
│  stats_input = compute_matrix_stats(S_raw)                  │
│  → [ChannelStats × n_ant]                                   │
│  if save_input: → S_raw.npy                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
         Step 0.5: FARROW     ▼
┌─────────────────────────────────────────────────────────────┐
│  farrow = FarrowDelay()    ← загружает lagrange_matrix_48x5 │
│  delays_s = ULAGeometry.compute_delays(steer_theta)         │
│                                                              │
│  S_aligned = farrow.compensate_seconds(S_raw, delays_s, fs) │
│    └── для каждой антенны:                                   │
│        delay_samples = -delays_s[ant] * fs                   │
│        int_delay = floor(delay_samples)                      │
│        frac = delay_samples - int_delay                      │
│        frac_idx = round(frac * 48) % 48                      │
│        coeffs[5] = matrix[frac_idx]                          │
│        out[n] = Σ coeffs[k] · in[n - int_delay - 2 + k]     │
│                                                              │
│  → S_aligned [n_ant × n_samples] complex64                   │
│  stats_aligned = compute_matrix_stats(S_aligned)             │
│  if save_aligned: → S_aligned.npy                            │
│                                                              │
│  ✅ После Farrow все антенны КОГЕРЕНТНЫ (задержки убраны)    │
└─────────────────────────────┬───────────────────────────────┘
                              │
              Step 1: W_sum   ▼
┌─────────────────────────────────────────────────────────────┐
│  W_sum[b][a] = 1/√N   (все элементы одинаковые!)            │
│  → W [n_ant × n_ant] complex64                               │
│                                                              │
│  ✅ Просто суммирование — без фазовых сдвигов               │
│     (задержки уже убраны Farrow)                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
              Step 2: GEMM    ▼
┌─────────────────────────────────────────────────────────────┐
│  X = W_sum @ S_aligned                                      │
│  → X_gemm [n_ant × n_samples] complex64                     │
│  stats_gemm = compute_matrix_stats(X)                       │
│  if save_gemm: → X_gemm.npy                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
         Step 3: Window+FFT   ▼
│       (ИДЕНТИЧНО Pipeline A — см. выше)                      │
                              │
         Step 4: Peaks        ▼
│       (ИДЕНТИЧНО Pipeline A — см. выше)                      │
                              │
              ВЫХОД           ▼
┌─────────────────────────────────────────────────────────────┐
│  PipelineResult:                                             │
│    (всё то же что Pipeline A, ПЛЮС:)                        │
│    .S_aligned      [n_ant, n_samples]  ← ТОЛЬКО Pipeline B  │
│    .stats_aligned  List[ChannelStats]  ← ТОЛЬКО Pipeline B  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔗 Диаграмма вызовов из тестов

```
test_farrow_pipeline.py
│
├── TestFarrowDelay (4 теста)
│   │
│   ├── test_farrow_identity
│   │   FarrowDelay() → .apply(signal[1×1000], [0.0])
│   │   проверка: |out| ≈ |in|
│   │
│   ├── test_farrow_integer_delay
│   │   FarrowDelay() → .apply(impulse[1×100], [5.0])
│   │   проверка: peak переместился на 5 отсчётов
│   │
│   ├── test_farrow_compensate
│   │   FarrowDelay() → .apply(signal, [3.7]) → .compensate(delayed, [3.7])
│   │   проверка: |restored| ≈ |original| (центр)
│   │
│   └── test_farrow_multi_antenna
│       FarrowDelay() → .apply(ones[4×200], [0,1,2,3])
│       проверка: ant[0] ≠ 0, ant[3][0:3] ≈ 0
│
├── TestPipelineBasic (4 теста)
│   │
│   │  _make_scenario(fdev, noise_sigma):
│   │    ULAGeometry(8, 0.05) → ScenarioBuilder(array, fs=12M, n=8000)
│   │      → .add_target(θ=30, f0=2M, fdev) → .build() → scenario dict
│   │
│   ├── test_cw_pipeline_a
│   │   scenario(fdev=0) → PipelineRunner()
│   │     → .run_pipeline_a(scenario, θ=30, freq=2M)
│   │   проверка: peak.freq ≈ 2 MHz
│   │
│   ├── test_cw_pipeline_b
│   │   scenario(fdev=0) → PipelineRunner()
│   │     → .run_pipeline_b(scenario, θ=30)
│   │   проверка: peak.freq ≈ 2 MHz, S_aligned ≠ None
│   │
│   ├── test_lfm_pipeline_a
│   │   scenario(fdev=1M) → .run_pipeline_a(...)
│   │   проверка: peak.magnitude > 0
│   │
│   └── test_lfm_pipeline_b
│       scenario(fdev=1M) → .run_pipeline_b(...)
│       проверка: peak.magnitude > 0, S_aligned.shape OK
│
├── TestPipelineComparison (3 теста) ← КЛЮЧЕВЫЕ
│   │
│   ├── test_cw_comparison
│   │   ScenarioBuilder → .build() → PipelineRunner()
│   │     → .run_pipeline_a(...) vs .run_pipeline_b(...)
│   │   проверка: freq_diff < 2·freq_res, mag_ratio ∈ [0.5, 2.0]
│   │
│   ├── test_lfm_comparison ← KEY TEST!
│   │   ScenarioBuilder(fdev=1M) → PipelineRunner()
│   │     → .run_pipeline_a vs .run_pipeline_b
│   │   проверка: energy_B(1..3 MHz) / energy_A(1..3 MHz) > 0.8
│   │
│   └── test_lfm_large_delay
│       ULAGeometry(d=0.5m!) → ScenarioBuilder(fdev=2M)
│       → A vs B, обе magnitude > 0
│
├── TestComplexScenarios (3 теста)
│   │
│   ├── test_multi_target_farrow
│   │   2 targets (θ=20/f0=2M + θ=45/f0=3.5M) + noise
│   │   → pipeline_b(steer=20) → peak ≈ 2 MHz
│   │
│   ├── test_jammer_scenario
│   │   target(θ=30) + jammer(θ=-20) + noise
│   │   → pipeline_b(steer=30) → peak.mag > 0
│   │
│   └── test_snr_improvement
│       target(A=1) + noise(σ=1) → pipeline_b
│       → peak / noise_floor > 3.0
│
└── TestStatsAndCheckpoints (5 тестов)
    │
    ├── test_stats_computed
    │   → pipeline_a + pipeline_b
    │   проверка: stats_input, stats_gemm, stats_spectrum ≠ None
    │             stats_aligned ≠ None (B only)
    │
    ├── test_stats_values
    │   проверка: power > 0, max ≥ min, n_samples = 8000
    │
    ├── test_pipeline_result_access
    │   проверка: S_raw, S_aligned, X_gemm, spectrum,
    │             magnitudes, W, freq_axis — все доступны
    │
    ├── test_save_to_disk
    │   PipelineRunner(output_dir=tmpdir) + PipelineConfig(save_all=True)
    │   проверка: S_raw.npy, S_aligned.npy, X_gemm.npy,
    │             spectrum.npy, stats.json, results.json
    │
    └── test_comparison_output + test_summary_strings
        runner.compare(A, B) → dict с magnitude_ratio_b_over_a
        result.peak_summary(), stats_summary() → строки
```

---

## 📂 Checkpoint файлы на диске

```
output_dir/
├── pipeline_a/
│   ├── S_raw.npy             [n_ant × n_samples] complex64
│   ├── X_gemm.npy            [n_ant × n_samples] complex64
│   ├── spectrum.npy          [n_ant × nFFT] complex64
│   ├── magnitudes.npy        [n_ant × nFFT] float32
│   ├── stats.json            { stats_input, stats_gemm, stats_spectrum }
│   └── results.json          { peaks: [[PeakInfo]] }
│
├── pipeline_b/
│   ├── S_raw.npy             (same)
│   ├── S_aligned.npy         [n_ant × n_samples] ← FARROW OUTPUT
│   ├── X_gemm.npy            (same)
│   ├── spectrum.npy          (same)
│   ├── magnitudes.npy        (same)
│   ├── stats.json            { + stats_aligned }
│   └── results.json          (same)
│
└── comparison.json           { magnitude_ratio_b_over_a, freq_diff_hz, ... }
```

---

## 🔬 Почему Pipeline B лучше для ЛЧМ

### Проблема Pipeline A

W_phase корректирует задержку через **фазовый сдвиг на одной частоте**:
```
W[b][a] = (1/√N) · exp(-j·2π·f0·τ_a)
```

Для CW (f = const) это **точно**. Но для ЛЧМ мгновенная частота меняется:
```
f(t) = f0 + (fdev/Ti)·t
```

Коррекция на f0 не компенсирует задержку на других частотах → **temporal smearing**.

### Решение Pipeline B

FarrowDelay применяет **ВРЕМЕННУЮ задержку** через Lagrange interpolation:
- Работает для **любого** сигнала (CW, ЛЧМ, шум)
- Точность: субсэмпловая (48 подразбиений дробной части)
- После Farrow все антенны **когерентны** → W просто суммирует

---

## 📐 Формулы

### Farrow Delay (Lagrange 48×5)

```
delay_samples = τ_ant · fs
int_delay = floor(delay_samples)
frac = delay_samples - int_delay
frac_idx = round(frac · 48) mod 48

coeffs[5] = lagrange_matrix[frac_idx]

output[n] = Σ_{k=0}^{4} coeffs[k] · input[n - int_delay - 2 + k]
```

### W_phase (Pipeline A)

```
W[b][a] = (1/√N) · exp(-j·2π·f0·τ_a)
τ_a = a · d · sin(θ_steer) / c
```

### W_sum (Pipeline B)

```
W[b][a] = 1/√N    (все элементы одинаковые)
```

---

## 📊 Метрики сравнения

| Метрика | Pipeline A | Pipeline B | Ожидание |
|---------|-----------|-----------|----------|
| Peak magnitude (CW) | ≈ равны | ≈ равны | Оба точны для CW |
| Peak magnitude (ЛЧМ) | Ниже | **Выше** | B лучше для chirp |
| Freq accuracy | ± freq_res | ± freq_res | B может быть точнее |
| SNR gain | ~10·log10(N) | ~10·log10(N) | Оба ~ √N |
| Sidelobe level | Зависит | Зависит | B чище |

---

## 📁 Файлы

| Файл | Описание |
|------|----------|
| `Python_test/strategies/scenario_builder.py` | Генератор сценариев с физикой ULA |
| `Python_test/strategies/farrow_delay.py` | Numpy Farrow (Lagrange 48×5) |
| `Python_test/strategies/pipeline_runner.py` | PipelineRunner (A/B + stats + checkpoints + compare) |
| `Python_test/strategies/test_farrow_pipeline.py` | 19 тестов Pipeline A vs B |
| `modules/lch_farrow/lagrange_matrix_48x5.json` | Матрица коэффициентов Лагранжа |
| `MemoryBank/specs/farrow_pipeline.md` | Эта спецификация |

---

## 🔗 Зависимости

- `scenario_builder.py` — генерация тестовых сигналов с физикой
- `modules/lch_farrow/lagrange_matrix_48x5.json` — коэффициенты Лагранжа
- `lch_farrow::LchFarrowROCm` — GPU версия (для будущей интеграции)

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-03-08 | Alex + Кодо | Создание спецификации |
| 2026-03-08 | Кодо | Добавлены пошаговые диаграммы вызовов, диаграмма тестов |
