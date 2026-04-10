# 📝 Farrow Pipeline — Спецификация

> **Модули**: `Python_test/strategies/` + `modules/lch_farrow/` + `modules/capon/`
> **Статус**: 🟢 Фаза 1 DONE | Фазы 2-4 в планах
> **Платформы**: Python reference (Фаза 1-2) → ROCm GPU (Фаза 3-4)
> **Автор**: Alex + Кодо
> **Создано**: 2026-03-08
> **Обновлено**: 2026-03-17
> **Связано**: [strategies/Full.md](Full.md) · [lch_farrow/Full.md](../lch_farrow/Full.md) · [capon/Full.md](../capon/Full.md)

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

---

## 🚀 Дорожная карта (Roadmap)

```
Фаза 1 (Python Reference)    ████████████████████  100% ✅
Фаза 2 (Визуализация)        ░░░░░░░░░░░░░░░░░░░░    0%
Фаза 3 (GPU — Pipeline C)    ░░░░░░░░░░░░░░░░░░░░    0%
Фаза 4 (Адаптивные — MVDR)   ████░░░░░░░░░░░░░░░░   20% 🔶 Python Reference
```

> ⚠️ **Обновление 2026-03-17**: Фаза 4 (MVDR/Capon) частично реализована!
> C++ GPU-модуль `modules/capon/` создан — MVDR алгоритм на ROCm, 4/4 тестов PASSED.
> Следующий шаг: Python биндинги `capon` → `CaponProcessor`, интеграция с `MVDRBeamformer` reference.

### Фаза 1: Python Reference ✅ ЗАВЕРШЕНО

| Компонент | Файл | Статус |
|-----------|------|--------|
| FarrowDelay (Lagrange 48×5) | `farrow_delay.py` | ✅ |
| Pipeline A (phase beamforming) | `pipeline_runner.py` | ✅ |
| Pipeline B (Farrow alignment) | `pipeline_runner.py` | ✅ |
| PipelineConfig + checkpoint'ы | `pipeline_runner.py` | ✅ |
| ChannelStats + compute_matrix_stats | `pipeline_runner.py` | ✅ |
| PeakInfo + find_peaks_per_beam | `pipeline_runner.py` | ✅ |
| PipelineResult (все поля) | `pipeline_runner.py` | ✅ |
| compare() + print_comparison() | `pipeline_runner.py` | ✅ |
| 19 тестов (unit + integration) | `test_farrow_pipeline.py` | ✅ |

### Фаза 2: Визуализация и анализ

**Цель**: Matplotlib графики для наглядного сравнения A vs B

#### TASK-FP-01: `plot_pipeline_comparison.py`

```
Файл: Python_test/strategies/plot_pipeline_comparison.py
Выход: Results/Plots/strategies/

Subplot layout (2×2):
┌─────────────────────┬─────────────────────┐
│  Спектр Beam 0      │  Диаграмма          │
│  A (синий)          │  направленности     │
│  B (оранжевый)      │  A vs B vs реальная │
│  fdev полоса выделена│                    │
├─────────────────────┼─────────────────────┤
│  S_raw[0] и         │  Энергия B/A        │
│  S_aligned[0]       │  по полосам         │
│  временны́е данные   │  (bar chart)        │
└─────────────────────┴─────────────────────┘
```

**API**:
```python
plot_pipeline_comparison(
    result_a: PipelineResult,
    result_b: PipelineResult,
    scenario: dict,
    title: str = "",
    save_path: str = None,   # Results/Plots/strategies/
    show: bool = True,
)
```

#### TASK-FP-02: `plot_beam_pattern.py`

```
Вход: W матрица [n_beams × n_ant], ULAGeometry, freq_hz
Выход: |W · a(θ)| vs θ от -90° до +90°
       стрелка на угол цели, угол помехи (если есть)

Формула:
  a(θ) = exp(-j·2π·freq·τ(θ))   — steering vector [n_ant]
  pattern(θ) = |W[0] · a(θ)|    — чувствительность beam 0
```

#### TASK-FP-03: Сводный отчёт (HTML/Markdown)

```python
runner.export_report(result_a, result_b, output_dir="Results/strategies/report/")
# → report.md с таблицей метрик + embedded PNG графики
```

### Фаза 3: Pipeline C — GPU реализация

**Цель**: AntennaProcessorROCm на GPU, верификация против Python reference

#### 3.1 Новый C++ модуль: `modules/strategies/`

```
modules/strategies/
├── AntennaProcessorROCm.h          ← публичный API
├── AntennaProcessorROCm.cpp        ← реализация (hipBLAS + hipFFT)
├── AntennaProcessorOpenCL.h        ← OpenCL backend (будущее)
├── tests/
│   ├── test_antenna_processor.hpp  ← C++ unit тесты
│   ├── test_antenna_benchmark.hpp  ← GPUProfiler benchmark
│   └── all_test.hpp
└── python/
    └── antenna_processor_bindings.cpp  ← pybind11
```

#### 3.2 C++ API план

```cpp
// modules/strategies/AntennaProcessorROCm.h

class AntennaProcessorROCm {
public:
    AntennaProcessorROCm(DrvGPU& ctx, int n_ant, int n_samples,
                         int n_fft_size = -1);  // -1 = auto: next_pow2*2

    // === ЗАГРУЗКА ДАННЫХ ===

    // CPU → GPU: входная матрица S [n_ant × n_samples] complex64
    void load_input(const std::complex<float>* S_host);

    // CPU → GPU: весовая матрица W [n_beams × n_ant] complex64
    void load_weights(const std::complex<float>* W_host,
                      int n_beams = -1);           // -1 = n_ant

    // CPU → GPU: задержки в отсчётах [n_ant] для Farrow
    void load_delays(const float* delays_samples);

    // === PIPELINE ШАГИ ===

    // Pipeline B: Farrow выравнивание через LchFarrowROCm
    // delays задаются через load_delays()
    void step_farrow_compensate();

    // GEMM: X = W @ S  (hipBLAS CGEMM)
    // Если Farrow был применён — X = W @ S_aligned
    void step_gemm();

    // Windowing + FFT (hipFFT) → spectrum [n_beams × nFFT]
    void step_fft();

    // Нахождение топ-N пиков на CPU (после transfer magnitudes)
    void step_find_peaks(int n_top = 5);

    // === РЕЗУЛЬТАТЫ (GPU → CPU) ===

    std::vector<float> get_magnitudes();      // [n_beams * nFFT]
    std::vector<float> get_peaks_freq_hz();   // [n_top] частоты пиков
    std::vector<float> get_peaks_mag();       // [n_top] магнитуды пиков
    std::vector<float> get_freq_axis();       // [nFFT] ось частот

    // === ПРОФИЛИРОВАНИЕ ===
    void set_gpu_info(const std::string& device_name,
                      const std::string& driver_ver);
    void export_profiling(const std::string& path);  // JSON/Markdown

private:
    DrvGPU& ctx_;
    LchFarrowROCm farrow_;
    GPUProfiler profiler_;
    // ... GPU буферы ...
};
```

#### 3.3 Pipeline C — Python wrapper

```python
# Python_test/strategies/pipeline_runner_gpu.py

class PipelineRunnerGPU:
    """GPU версия PipelineRunner через C++ AntennaProcessorTest bidings."""

    def __init__(self, gpu_ctx, output_dir=None):
        self.proc = AntennaProcessorTest(gpu_ctx, ...)
        self.output_dir = output_dir

    def run_pipeline_b_gpu(self, scenario, steer_theta) -> PipelineResult:
        """Выполнить Pipeline B на GPU, вернуть PipelineResult как у CPU."""
        S = scenario['S']
        delays = scenario['array'].compute_delays(steer_theta)
        delays_samples = delays * scenario['fs']

        W_sum = np.full((n_ant, n_ant), 1/np.sqrt(n_ant), dtype=np.complex64)

        self.proc.load_input(S)
        self.proc.load_weights(W_sum)
        self.proc.load_delays(delays_samples.astype(np.float32))
        self.proc.step_farrow_compensate()
        self.proc.step_gemm()
        self.proc.step_fft()
        self.proc.step_find_peaks()

        # Собрать результат в PipelineResult (для compare с CPU)
        return PipelineResult(
            pipeline_name="C (GPU)",
            magnitudes=self.proc.get_magnitudes(),
            peaks=...,
            freq_axis=self.proc.get_freq_axis(),
        )
```

#### 3.4 Верификационный тест

```python
# Python_test/strategies/test_gpu_verification.py

def test_pipeline_b_gpu_vs_python():
    """GPU Pipeline B == Python Pipeline B (в пределах float32 погрешности)."""
    scenario = make_single_target(n_ant=8, theta_deg=30, fdev_hz=1e6)

    # CPU reference
    runner_cpu = PipelineRunner()
    result_cpu = runner_cpu.run_pipeline_b(scenario, steer_theta=30)

    # GPU под тестом
    gpu_ctx = GPUContext()
    runner_gpu = PipelineRunnerGPU(gpu_ctx)
    result_gpu = runner_gpu.run_pipeline_b_gpu(scenario, steer_theta=30)

    # Сравнение пиков
    peak_cpu = result_cpu.peaks[0][0]
    peak_gpu = result_gpu.peaks[0][0]

    freq_res = FS / result_cpu.nFFT
    assert abs(peak_cpu.freq_hz - peak_gpu.freq_hz) < freq_res, \
        f"Freq mismatch: CPU={peak_cpu.freq_hz:.0f} GPU={peak_gpu.freq_hz:.0f}"
    assert abs(peak_cpu.magnitude - peak_gpu.magnitude) / peak_cpu.magnitude < 0.01, \
        f"Magnitude mismatch > 1%"
```

#### 3.5 GPU Profiling план

```
GPUProfiler отчёт по Pipeline B GPU:

Шаги профилирования:
  "Farrow"    ← LchFarrowROCm.Process()
  "GEMM"      ← hipBLAS CGEMM
  "Window"    ← кастомный HIP kernel (Hamming)
  "FFT"       ← hipFFT
  "Transfer"  ← GPU→CPU magnitudes

Ожидаемые результаты (8 антенн, N=8192):
  Farrow:   ~0.05 ms
  GEMM:     ~0.02 ms
  FFT:      ~0.03 ms
  Итого:    ~0.1 ms / блок данных
```

### Фаза 4: Адаптивные алгоритмы (MVDR / LCMV)

**Цель**: Pipeline D — оптимальный beamformer с подавлением помех

#### Теория MVDR

```
Проблема delay-and-sum (Pipeline A/B):
  Помеха из бокового лепестка не подавляется!

MVDR решение:
  R = (1/K) · S · S†              матрица ковариации [n_ant × n_ant]
  a(θ) = exp(-j·2π·f·τ(θ))        steering vector [n_ant]

  w = R⁻¹ · a / (a† · R⁻¹ · a)   MVDR weights [n_ant]

  Гарантия: w† · a(θ_target) = 1  (distortionless)
  Минимизация: E[|w† · S|²]       (подавление всего остального)

Выигрыш vs delay-and-sum:
  Помеха -20 ... -40 dB (зависит от SNR, числа антенн)
```

#### TASK-FP-10: MVDRBeamformer (Python reference)

```python
# Python_test/strategies/mvdr_beamformer.py

class MVDRBeamformer:
    def __init__(self, array: ULAGeometry, freq_hz: float,
                 diagonal_loading: float = 1e-4):
        """
        diagonal_loading: добавляется к R для стабильности инверсии
        R_loaded = R + diag_load * trace(R)/N * I
        """

    def fit(self, S: np.ndarray) -> 'MVDRBeamformer':
        """Оценить матрицу ковариации из данных S [n_ant × n_samples]."""
        K = S.shape[1]
        self.R = (S @ S.conj().T) / K + self.diag * np.eye(self.n_ant)
        self.R_inv = np.linalg.inv(self.R)
        return self

    def get_weights(self, steer_theta_deg: float) -> np.ndarray:
        """Вычислить MVDR веса для угла steer_theta_deg → [n_ant]."""
        a = self._steering_vector(steer_theta_deg)
        w = self.R_inv @ a
        return w / (a.conj() @ w)

    def beam_pattern(self, theta_range: np.ndarray) -> np.ndarray:
        """Диаграмма направленности MVDR vs угол → [len(theta_range)]."""
```

#### TASK-FP-11: Сравнительные тесты MVDR vs DAS

```python
def test_mvdr_jammer_suppression():
    """MVDR подавляет ЛЧМ помеху лучше delay-and-sum на ≥15 dB."""

    scenario = make_target_and_jammer(
        target_theta=30, jammer_theta=-20,
        target_f0=2e6, jammer_f0=2e6, jammer_amplitude=2.0
    )
    S = scenario['S']

    # Delay-and-sum
    runner = PipelineRunner()
    result_das = runner.run_pipeline_b(scenario, steer_theta=30)

    # MVDR
    mvdr = MVDRBeamformer(scenario['array'], freq_hz=2e6)
    mvdr.fit(S)
    w_mvdr = mvdr.get_weights(30)
    # ... pipeline с w_mvdr ...

    jammer_power_das = ...   # мощность в направлении -20°
    jammer_power_mvdr = ...
    suppression_db = 10*np.log10(jammer_power_das / jammer_power_mvdr)

    assert suppression_db > 15.0, f"MVDR suppression only {suppression_db:.1f} dB"
```

---

## 📋 Задачи (Backlog)

### P0 — Критические (блокируют показ результатов)

| ID | Задача | Оценка |
|----|--------|--------|
| TASK-FP-01 | `plot_pipeline_comparison.py` (matplotlib, 4 subplot) | 2-3 ч |
| TASK-FP-02 | `plot_beam_pattern.py` (A vs B диаграмма направленности) | 1-2 ч |

### P1 — Важные (GPU интеграция)

| ID | Задача | Оценка |
|----|--------|--------|
| TASK-FP-03 | `AntennaProcessorROCm` C++ класс (hipBLAS + hipFFT + LchFarrow) | 2-3 дня |
| TASK-FP-04 | pybind11 биндинг `AntennaProcessorTest` | 0.5 дня |
| TASK-FP-05 | `PipelineRunnerGPU` Python wrapper | 1 день |
| TASK-FP-06 | `test_gpu_verification.py` (GPU vs CPU ±1%) | 1 день |
| TASK-FP-07 | GPUProfiler отчёт по Pipeline B GPU | 0.5 дня |

### P2 — Расширения (адаптивные алгоритмы)

| ID | Задача | Оценка |
|----|--------|--------|
| TASK-FP-08 | Scan beamforming: sweep θ, сохранение 2D beam map | 1 день |
| TASK-FP-09 | Метрики: SINR, ширина луча, уровень боковых лепестков | 1 день |
| TASK-FP-10 | `MVDRBeamformer` Python reference | 1-2 дня |
| TASK-FP-11 | Тесты MVDR vs DAS (подавление помех) | 1 день |
| TASK-FP-12 | MVDR GPU (hipBLAS potrs/getrs для R⁻¹) | 3-5 дней |

---

## ✅ Критерии успеха по фазам

| Фаза | Критерий | Измерение |
|------|----------|-----------|
| **1** | 19 тестов `test_farrow_pipeline.py` pass | `python run_tests.py -v` — 0 failures |
| **2** | Графики сохраняются в `Results/Plots/strategies/` | Файлы есть, визуально корректны |
| **3 (GPU)** | Пики GPU == CPU ±1% по магнитуде | `test_gpu_verification.py` pass |
| **3 (GPU)** | Пики GPU == CPU ±freq_res по частоте | `test_gpu_verification.py` pass |
| **3 (GPU)** | Latency Pipeline B GPU < 1 ms (N=8192, 8 ant) | GPUProfiler report |
| **4 (MVDR)** | MVDR подавляет помеху ≥15 dB vs DAS | `test_mvdr_jammer_suppression` pass |
| **4 (MVDR)** | SINR(MVDR) > SINR(DAS) при jammer amp ≥ 1.0 | числовой тест |

---

## 📊 Ожидаемые производительные характеристики GPU (Фаза 3)

> Оценка для AMD GPU (RDNA4, gfx1201), 8 антенн, N=8192

| Операция | Ожидание | Библиотека |
|----------|----------|-----------|
| Farrow (8 каналов × 8192 samp) | ~0.05 ms | LchFarrowROCm |
| GEMM (8×8 @ 8×8192) complex | ~0.02 ms | hipBLAS CGEMM |
| FFT (8 × 16384) | ~0.03 ms | hipFFT |
| Total Pipeline B | **< 0.2 ms** | — |
| CPU→GPU transfer (S) | ~0.1 ms | DrvGPU |
| GPU→CPU transfer (magnitudes) | ~0.05 ms | DrvGPU |

---

## 🔮 Будущие направления (не в ближайших планах)

- **2D сканирование (θ, φ)** — URA, 3D beam pattern
- **Пространственно-временная обработка (STAP)** — Doppler + beamforming
- **ML-based beamforming** — нейросетевые веса
- **Когерентный MIMO** — несколько TX + RX решёток
- **Real-time визуализация** — Dear PyGui + streaming GPU данные
- **Интеграция с PyPanelAntennas** — визуализация поля через UDP

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-03-08 | Alex + Кодо | Создание спецификации |
| 2026-03-08 | Кодо | Добавлены пошаговые диаграммы вызовов, диаграмма тестов |
| 2026-03-08 | Кодо | Добавлены: дорожная карта (4 фазы), backlog (12 задач), C++ план AntennaProcessorROCm, MVDR теория + план, критерии успеха, GPU бенчмарки |
