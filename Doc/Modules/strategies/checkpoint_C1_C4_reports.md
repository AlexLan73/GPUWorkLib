# Checkpoint Reports C1–C4
# Antenna Array Processor — Save Points Analysis
# Версия: v2.0 — исправления: 3 ветки, FFT mirror, Logs/GPU_XX/, MaxValue reuse, ответы Q1-Q4

> **Контекст**: В архитектуре v0.7 (GEMM, квадратная W, 3 ветки):
> - S[N_ant × N_samples] — сырые данные антенн (входные, после LchFarrow)
> - W[N_ant × N_ant]     — матрица весов КВАДРАТНАЯ (beamforming коэффициенты, max 256×256)
> - X = W × S           — результат GEMM hipBLAS Cgemm (beamformed data, размер как S)
> - После GEMM: Hamming → FFT → **3 ветки**:
>   - **Step2.1**: один Global MAX + 3-point parabola (sub-bin частота, без фазы)
>   - **Step2.2**: ВСЕ локальные пики (CFAR)
>   - **Step2.3**: глобальные MAX/MIN по спектру
> - MaxValue: **переиспользуется из fft_maxima** (spectrum_result_types.hpp, 32 байта)
> - FFT fold: бины k > nFFT/2 → отрицательные частоты (fold_fft_mirror)
> - В старых примерах ниже legacy-обозначения `Branch 2/3/4` следует читать как `Step2.3 / Step2.1 / Step2.2`
>
> **Архитектурные документы**: `Doc/Architecture/AntennaProcessor/` (C1-C4 по C4 Model)

---

## Ответы на вопросы (из обсуждения)

| Вопрос | Ответ |
|--------|-------|
| **Q1**: C3 и C4 — переключаемые? | ✅ **ДА** — одна ветка за раз (по `branch_mode`) |
| **Q2**: PRE/POST stats — разные пресеты? | ✅ **ДА** — `pre_gemm_stats` и `post_gemm_stats` независимо |
| **Q3**: Полный спектр (4.9 ГБ) нужен? | ✅ **ДА, но только при тестировании** (`c3_spectrum=true`). Среднее/медиана — через отдельный kernel-вызов прямо из VRAM |
| **Q4**: detect_all_maxima — куда? | ✅ **Step2.2** — обязательный post-FFT сценарий |

---

## Содержание

1. [C1 — После загрузки в GPU](#c1)
2. [C2 — После GEMM, до Hamming](#c2)
3. [C3 — После FFT: Global Min/Max (Ветка 2)](#c3)
4. [C4 — После FFT: Один Max + Парабола (Ветка 3)](#c4)
-- добавить
5. [SaveOptions — Production pattern](#save-options)
6. [StatisticsSet — Пресеты статистики](#statistics-set)
7. [Формат файлов](#file-format)
8. [Сводная таблица](#summary)

---

## C1 — После загрузки данных в GPU {#c1}

```
Точка 4.1: "первую точку я пометил после получения данных из памяти GPU"
Место в pipeline: после DMA Host→GPU, event_data_ready
Ветка: общая (до разветвления Stream 1/2)

-- да данные должны попасть через сетевую карту на GPU
```

### Что сохраняется

```
┌─────────────────────────────────────────────────────────────────────────┐
│  C1 SAVE POINT                                                          │
│                                                                         │
│  A) Входные данные (сырой сигнал после LchFarrow):                      │
│     d_data[N_ant × N_samples]   cf32   READ-ONLY snapshot              │
│                                                                         │
│  B) Матрица весов (beamforming):                                        │
│     d_weights[N_ant × N_ant]    cf32   READ-ONLY snapshot              │
│     (квадратная! коэффициенты beamforming, max 256×256)                │
│                                                                         │
│  C) Метаданные:                                                         │
│     N_ant, N_samples, sample_rate, timestamp, gpu_id                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Размеры

| Данные | Формула | Малый вариант (256×1.2M) | Большой вариант (3500×2500) |
|--------|---------|--------------------------|------------------------------|
| S (сигнал) | N_ant × N_samples × 8 | **2.457 ГБ** | **70 МБ** |
| W (веса) | N_ant × N_ant × 8 | **512 КБ** | **196 МБ** |
| Метаданные | фиксированный | **~256 байт** | **~256 байт** |
| **Итого C1** | | **≈ 2.46 ГБ** | **≈ 266 МБ** |

> ⚠️ **Важно**: C1 сигнал — это БОЛЬШИЕ данные (2.5 ГБ)!
> Сохранять только при отладке или по запросу. W — маленькая (512 КБ), почти бесплатно.

### Что можно анализировать из C1

```
✅ Контроль целостности данных после DMA-переноса
✅ Уровни сигнала по антеннам (проверка неисправных антенн)
✅ Корреляция между антеннами (проверка фазировки)
✅ Проверка матрицы весов W (нет NaN/Inf, норма строк)
✅ Эталонное сравнение: С1 от software-эмулятора vs GPU-данные
✅ Аномалии: saturated samples, dropout, interference
```
!!! - Что можно анализировать из C1 - СУПЕР разверни и заложем это в Python анализ!!

### Python-анализ (что можно построить)

```python
import numpy as np
import matplotlib.pyplot as plt

S = load_c1_signal(path)  # [N_ant, N_samples] complex64
W = load_c1_weights(path) # [N_ant, N_ant] complex64

# 1. Мощность по антеннам
power = np.mean(np.abs(S)**2, axis=1)
plt.bar(range(N_ant), power)
plt.title("C1: Signal power per antenna")

# 2. Временной сигнал одной антенны
plt.plot(np.abs(S[0]))
plt.title("C1: Antenna 0 — magnitude vs time")

# 3. Матрица весов W (heatmap)
plt.imshow(np.abs(W), cmap='hot')
plt.title("C1: Weight matrix |W| (beamforming coefficients)")

# 4. Спектр (для быстрой проверки)
spectrum = np.fft.fft(S[0])
plt.plot(np.abs(spectrum[:N_samples//2]))
plt.title("C1: Quick FFT of antenna 0 (reference)")
```

---

## C2 — После GEMM, до Hamming {#c2}

```
Точка 4.2: "после перемножений матриц перед Хеммингом"
Место в pipeline: после hipBLAS Cgemm (X = S × W), перед apply_hamming
Ветка: Stream 2 (Main Pipeline)
```

### Что сохраняется

```
┌─────────────────────────────────────────────────────────────────────────┐
│  C2 SAVE POINT                                                          │
│                                                                         │
│  A) Результат beamforming (выход GEMM):                                 │
│     d_X[N_ant × N_samples]      cf32   сформированные лучи             │
│     (X = S × W — СТАНДАРТНОЕ умножение матриц)                         │
│                                                                         │
│  B) Статистика PRE-GEMM (по сырому сигналу S):                         │
│     StatsResult[N_ant] — уже посчитана в Stream 1                     │
│     Пресет: StatisticsSet (задаётся пользователем)                     │
│                                                                         │
│  C) Статистика POST-GEMM (по beamformed X):                            │
│     StatsResult[N_ant] — новый pass по X                               │
│     Пресет: StatisticsSet (тот же или другой)                          │
│                                                                         │
│  D) Метаданные:                                                         │
│     N_ant, N_samples, sample_rate, timestamp, gpu_id                   │
│     gemm_time_ms (время GEMM для диагностики)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Размеры

| Данные | Формула | Малый вариант (256×1.2M) | Большой вариант (3500×2500) |
|--------|---------|--------------------------|------------------------------|
| X (beamformed) | N_ant × N_samples × 8 | **2.457 ГБ** | **70 МБ** |
| Stats PRE-GEMM | N_ant × 56 байт | **14 КБ** | **196 КБ** |
| Stats POST-GEMM | N_ant × 56 байт | **14 КБ** | **196 КБ** |
| **Итого C2** | | **≈ 2.46 ГБ** | **≈ 70.4 МБ** |

> 💡 **Оптимизация**: Статистика (28 КБ) — почти бесплатно. X — большая как S.
> Если нужна только статистика (без X) — опция `stats_only = true`.

### Ключевые различия C1 vs C2

```
C1: S = raw antenna signals       — "что мы получили с антенн"
C2: X = S × W (beamformed)       — "что мы сформировали (лучи)"

Физический смысл:
  S[i, n] — сигнал i-й антенны в момент n
  X[k, n] — сигнал k-го луча в момент n  (суперпозиция всех антенн с весами W)
```

### Что можно анализировать из C2

```
✅ Усиление beamforming: power(X[k]) vs power(S[i]) — должно быть выше
✅ Пространственное разделение (isolation between beams)
✅ Проверка GEMM: X_cpu = S × W → compare with X_gpu
✅ Статистика PRE vs POST: как GEMM изменяет распределение сигнала
✅ Детекция дефектов матрицы W: если луч "слепой" (power ≈ 0)
✅ Динамический диапазон по лучам (нет overflow float32?)
✅ Hamming НЕ применялась → спектр имеет боковые лепестки → не для FFT-анализа!
```

### Python-анализ

```python
X = load_c2_data(path)            # [N_ant(=N_beams), N_samples] complex64
stats_pre = load_c2_stats_pre(path)   # StatsResult[N_ant]
stats_post = load_c2_stats_post(path) # StatsResult[N_ant]

# 1. Усиление beamforming по лучам
gain_dB = 10 * np.log10(stats_post.mean_magnitude / stats_pre.mean_magnitude)
plt.bar(range(N_ant), gain_dB)
plt.title("C2: Beamforming gain per beam [dB]")

# 2. Сравнение статистики PRE vs POST-GEMM
fig, axes = plt.subplots(2, 3)
for ax, field in zip(axes.flat, ['mean_mag', 'median', 'std', 'var', 'min_mag', 'max_mag']):
    ax.plot(stats_pre[field], label='PRE-GEMM')
    ax.plot(stats_post[field], label='POST-GEMM')
    ax.set_title(field)
    ax.legend()
plt.suptitle("C2: Statistics PRE vs POST-GEMM per beam")

# 3. Beam pattern (быстрая проверка)
# Для одного частотного бина — показать усиление в зависимости от индекса луча
f0_bin = int(f0 / sample_rate * N_samples)
beam_powers = np.abs(np.fft.fft(X, axis=1)[:, f0_bin])**2
plt.plot(range(N_ant), 10*np.log10(beam_powers))
plt.title("C2: Beam pattern at f0 [dB]")
```

---

## C3 — После FFT: Global Min/Max (Ветка 2) {#c3}

```
Точка 4.3: "после FFT мах/мин"
Место в pipeline: после hipFFT → minmax_spectrum
Ветка: POST-FFT, ВЕТКА 2 (переключаемая)
```

### Что сохраняется

```
┌─────────────────────────────────────────────────────────────────────────┐
│  C3 SAVE POINT (Branch 2: Global Min/Max)                              │
│                                                                         │
│  A) Основной результат:                                                 │
│     MinMaxResult[N_ant]          — глобальный min + max по спектру     │
│     (один min + один max на луч — GLOBAL, не локальные пики)          │
│                                                                         │
│  B) [Опционально] Полный спектр:                                        │
│     d_spectrum[N_ant × nFFT]     cf32   (БОЛЬШОЙ! 4.9 ГБ)             │
│     save_full_spectrum = false   ← по умолчанию НЕ сохраняем          │
│                                                                         │
│  C) [Опционально] Вариант заказчика — detect_all_maxima:               │
│     AllMaximaResult[N_ant]       — ВСЕ локальные пики > CFAR          │
│     (только если mode = DETECT_ALL_MAXIMA)                            │
│                                                                         │
│  D) Метаданные:                                                         │
│     N_ant, nFFT, sample_rate, timestamp                                │
│     fft_time_ms, minmax_time_ms                                        │
│     search_range (nFFT/2 для одностороннего спектра)                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Структура MinMaxResult (per beam)

```cpp
struct MinMaxResult {
    uint32_t beam_id;           // индекс луча (=антенны) [0..N_ant)
    // ── Минимум ──────────────────────────────────────────────────
    float    min_magnitude;     // min |z| по спектру этого луча
    uint32_t min_bin;           // частотный бин минимума [0..nFFT)
    float    min_frequency_hz;  // min_bin * sample_rate / nFFT
    // ── Максимум ─────────────────────────────────────────────────
    float    max_magnitude;     // max |z| по спектру этого луча
    uint32_t max_bin;           // частотный бин максимума [0..nFFT)
    float    max_frequency_hz;  // max_bin * sample_rate / nFFT
    // ── Производные ──────────────────────────────────────────────
    float    dynamic_range_dB;  // 20*log10(max_mag / max(min_mag, eps))
    uint32_t pad;               // выравнивание до 32 байт
};  // sizeof = 32 байта (точно, без padding)
```

### Размеры

| Данные | Формула | Малый вариант (256×1.2M) | Большой вариант (3500×2500) |
|--------|---------|--------------------------|------------------------------|
| MinMaxResult | N_ant × 32 байт | **8 КБ** | **112 КБ** |
| Полный спектр (опционально) | N_ant × nFFT × 8 | **≈ 4.9 ГБ** | **140 МБ** |
| AllMaximaResult (опционально) | N_ant × (64+8) × 32 | **≈ 2 МБ** | **≈ 29 МБ** |

> ✅ **C3 без спектра** — очень маленький (8 КБ)! Почти бесплатно.
> ⚠️ **C3 с полным спектром** — 4.9 ГБ, только для глубокой отладки.

### Ветка 2 vs Вариант заказчика

```
ВЕТКА 2 (основная):
  minmax_spectrum kernel:
  → один GLOBAL MAX + один GLOBAL MIN на луч
  → Kernel: 1 блок/луч, 256 потоков, tree reduction
  → Результат: MinMaxResult[N_ant]

ВАРИАНТ ЗАКАЗЧИКА (detect_all_maxima):
  detect_all_maxima kernel:
  → ВСЕ локальные пики > CFAR threshold (median * alpha)
  → Используется median из Stream 1 (уже подсчитана!)
  → Результат: AllMaximaResult[N_ant] — peaks[], n_peaks, min+max
  → Более сложный kernel (Blelloch prefix scan)
  → Включается флагом: mode = DETECT_ALL_MAXIMA
```

### Что можно анализировать из C3

```
✅ Динамический диапазон спектра по лучам (max/min ratio)
✅ Детекция явных целей: луч с max >> среднего = цель
✅ Шумовой пол по лучам (min_magnitude ≈ noise floor)
✅ Равномерность беамформинга: max_mag должны быть похожи для "пустого" поля
✅ Частота главного пика (max_frequency_hz) по лучам
✅ При detect_all_maxima: сколько пиков в каждом луче (n_peaks)
```

### Python-анализ

```python
results = load_c3_minmax(path)  # list of MinMaxResult

# 1. Динамический диапазон по лучам
plt.bar(range(N_ant), [r.dynamic_range_dB for r in results])
plt.title("C3: Dynamic range per beam [dB]")
plt.xlabel("Beam index"); plt.ylabel("Dynamic range [dB]")

# 2. Частота максимума по лучам
plt.plot([r.max_frequency_hz for r in results])
plt.title("C3: Peak frequency per beam [Hz]")

# 3. Если полный спектр сохранён:
spectrum = load_c3_spectrum(path)  # [N_ant, nFFT] complex64
# Waterfall plot (спектрограмма по лучам)
plt.imshow(20*np.log10(np.abs(spectrum[:, :nFFT//2])),
           aspect='auto', cmap='hot', origin='lower')
plt.title("C3: Spectrum waterfall (dB), beams × frequency")
plt.xlabel("Frequency bin"); plt.ylabel("Beam index")
```

---

## C4 — После FFT: Один Max + Парабола (Ветка 3) {#c4}

```
Точка 4.4: "после FFT один мах + парабола"
Место в pipeline: после hipFFT → post_kernel_one_peak (парабола)
Ветка: POST-FFT, ВЕТКА 3 (переключаемая)
```

### Что сохраняется

```
┌─────────────────────────────────────────────────────────────────────────┐
│  C4 SAVE POINT (Branch 3: One Max + Parabola)                          │
│                                                                         │
│  A) Основной результат (один пик на луч с sub-bin точностью):           │
│     MaxValue[N_ant]              — детальный результат поиска          │
│                                                                         │
│  B) Метаданные:                                                         │
│     N_ant, nFFT, sample_rate, timestamp                                │
│     fft_time_ms, peak_search_time_ms                                   │
│     search_range (nFFT/2 для одностороннего спектра)                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Структура MaxValue (per beam)

```cpp
struct MaxValue {
    uint32_t beam_id;           // индекс луча
    // ── Позиция максимума ─────────────────────────────────────────
    uint32_t peak_bin;          // частотный бин максимума (целый)
    float    peak_frequency_hz; // peak_bin × sample_rate / nFFT
    // ── Значение в максимуме ──────────────────────────────────────
    float    real;              // Re(spectrum[peak_bin])
    float    imag;              // Im(spectrum[peak_bin])
    float    magnitude;         // |spectrum[peak_bin]|
    float    phase_deg;         // atan2(imag, real) × 180/π [градусы]
    // ── Sub-bin уточнение (парабола) ─────────────────────────────
    float    freq_offset;       // [-0.5 .. +0.5] дробная часть бина
    float    refined_freq_hz;   // уточнённая частота = (peak_bin + freq_offset)
                                //                     × sample_rate / nFFT
    // ── Соседние точки для параболы ──────────────────────────────
    float    mag_left;          // |spectrum[peak_bin - 1]|
    float    mag_right;         // |spectrum[peak_bin + 1]|
    uint32_t pad;               // выравнивание
};  // sizeof = 48 байт

// Формула параболы:
//   y_l = mag[peak-1], y_c = mag[peak], y_r = mag[peak+1]
//   offset = 0.5 × (y_l - y_r) / (y_l - 2×y_c + y_r)
//   refined_freq = (peak_bin + offset) × sample_rate / nFFT
```

### Размеры

| Данные | Формула | Малый вариант (256×1.2M) | Большой вариант (3500×2500) |
|--------|---------|--------------------------|------------------------------|
| MaxValue | N_ant × 48 байт | **12 КБ** | **168 КБ** |
| **Итого C4** | | **≈ 12 КБ** | **≈ 168 КБ** |

> ✅ **C4 — самый компактный чекпоинт!** Всегда можно сохранять без накладных расходов.
> Это финальный результат обработки — именно то, что нужно заказчику.

### Что можно анализировать из C4

```
✅ Точность определения частоты (sub-bin refinement): refined vs peak bin
✅ Sub-bin offset plot: распределение offset по лучам (должно быть [-0.5..+0.5])
✅ Фаза по лучам: phase_deg[beam] → угловой профиль прихода сигнала
✅ Чувствительность: magnitude по лучам (слабый луч = далёкая цель)
✅ Сравнение C3 vs C4: max_bin из C3 ≈ peak_bin из C4 (проверка корректности)
✅ Временная серия: C4 timestamps → скорость изменения частоты цели
✅ Сравнение с эталоном (NumPy): проверка точности параболы
```

### Python-анализ

```python
results = load_c4_maxvalue(path)  # list of MaxValue

# 1. Частота по лучам (грубая vs уточнённая)
beams = range(N_ant)
plt.plot(beams, [r.peak_frequency_hz for r in results], 'b-', label='Integer bin')
plt.plot(beams, [r.refined_freq_hz for r in results], 'r-', label='Parabola refined')
plt.title("C4: Peak frequency per beam (integer vs refined)")
plt.legend()

# 2. Sub-bin offset (проверка параболы)
offsets = [r.freq_offset for r in results]
plt.hist(offsets, bins=50)
plt.title("C4: Sub-bin offset distribution (should be [-0.5..+0.5])")
plt.xlabel("freq_offset"); plt.axvline(0, color='r', linestyle='--')

# 3. Фазовый профиль по лучам
phases = [r.phase_deg for r in results]
plt.polar([np.deg2rad(p) for p in phases], range(N_ant), 'o')
plt.title("C4: Phase profile per beam")

# 4. Энергия по лучам (находим самый сильный луч)
magnitudes = [r.magnitude for r in results]
best_beam = np.argmax(magnitudes)
print(f"C4: Best beam = {best_beam}, freq = {results[best_beam].refined_freq_hz:.2f} Hz")
```

---

## SaveOptions — Production Pattern {#save-options}

```
Заметка 5: "продумай и предложи варианты реализации для рабочего варианта
не нужны ветки отладки с сохранением данных, может сделать как с профилированием
(посмотри решение в modules/fft_processor)"
```

### Решение: CollectOrRelease pattern (как в fft_processor)

```cpp
// ── SaveOptions: передаётся в process() как опциональный параметр ─────────

struct CheckpointSaveConfig {
    // Что сохранять
    bool   c1_signal  = false;   // 4.1: сырой сигнал S (БОЛЬШОЙ: 2.5 ГБ!)
    bool   c1_weights = false;   // 4.1: матрица весов W (маленькая: 512 КБ)
    bool   c2_data    = false;   // 4.2: GEMM output X (БОЛЬШОЙ: 2.5 ГБ!)
    bool   c2_stats   = false;   // 4.2: статистика PRE+POST GEMM (мелкая)
    bool   c3_result  = true;    // 4.3: MinMaxResult (МЕЛКАЯ: 8 КБ)
    bool   c3_spectrum = false;  // 4.3: полный спектр (БОЛЬШОЙ: 4.9 ГБ!)
    bool   c4_result  = true;    // 4.4: MaxValue (МЕЛКАЯ: 12 КБ)

    // Куда сохранять (исправлено Note #4: сначала GPU_XX, потом модуль!)
    std::string save_dir = "Logs/GPU_{gpu_id}/antenna_processor/";  // + YYYY-MM-DD/HH-MM-SS/

    // Формат
    bool   json_header = false;  // true=JSON+binary header (debug), false=binary only
};

// Использование в production:
AntennaProcessor proc(ctx);
proc.process(S, W, result);  // ← нет сохранения, нет оверхеда

// Использование в debug:
CheckpointSaveConfig save_cfg;
save_cfg.c1_weights = true;
save_cfg.c2_stats   = true;
save_cfg.c4_result  = true;
proc.process(S, W, result, &save_cfg);  // ← сохраняет только нужное
```

### Паттерн внутри AntennaProcessor

```cpp
// Аналог CollectOrRelease из fft_processor:
void MaybeSaveCheckpoint(const std::string& name,
                          const void* data, size_t size_bytes,
                          const CheckpointSaveConfig* cfg, bool flag) {
    if (cfg == nullptr || !flag) return;  // ← ZERO OVERHEAD в production

    // Сохранить в файл (асинхронно если нужно)
    auto path = MakeSavePath(cfg->save_dir, name);
    SaveBinary(path, data, size_bytes);
    LOG_INFO << "[CheckpointSave] " << name << " → " << path
             << " (" << size_bytes / 1024 << " КБ)";
}

// В pipeline:
void AntennaProcessor::process(const DataMatrix& S, const WeightsMatrix& W,
                                AntennaResult& out,
                                const CheckpointSaveConfig* save_cfg = nullptr) {
    // 1. DMA load
    load_to_gpu(S, W);

    // C1 save (опционально)
    MaybeSaveCheckpoint("C1_signal",  d_data,    data_bytes,  save_cfg, save_cfg && save_cfg->c1_signal);
    MaybeSaveCheckpoint("C1_weights", d_weights, weight_bytes, save_cfg, save_cfg && save_cfg->c1_weights);

    // 2. Statistics (Stream 1, parallel)
    run_statistics_stream();

    // 3. GEMM: X = S × W
    run_gemm();

    // C2 save (опционально)
    MaybeSaveCheckpoint("C2_data",  d_X, data_bytes, save_cfg, save_cfg && save_cfg->c2_data);
    if (save_cfg && save_cfg->c2_stats) save_statistics_results(save_cfg->save_dir);

    // 4. Hamming + FFT
    run_hamming_fft();

    // 5. Post-FFT branch
    if (mode_ == Mode::BRANCH_2_MINMAX) {
        run_minmax_spectrum();
        MaybeSaveCheckpoint("C3_minmax", d_minmax, minmax_bytes, save_cfg, save_cfg && save_cfg->c3_result);
        if (save_cfg && save_cfg->c3_spectrum)
            MaybeSaveCheckpoint("C3_spectrum", d_spectrum, spectrum_bytes, save_cfg, true);
    } else {
        run_peak_parabola();
        MaybeSaveCheckpoint("C4_peak", d_peak, peak_bytes, save_cfg, save_cfg && save_cfg->c4_result);
    }
}
```

### Именование файлов (по принципу Logs/)

```
Logs/
└── GPU_00/                        ← ← ← per-GPU (Note #4: сначала GPU!)
    └── antenna_processor/
        └── 2026-03-06/
            └── 14-32-05/              ← метка старта сессии
                ├── meta.json          ← общие параметры (N_ant, N_samples, mode, branch, ...)
                ├── C1_signal.bin      ← если c1_signal=true
                ├── C1_weights.bin     ← если c1_weights=true
                ├── C2_data.bin        ← если c2_data=true
                ├── C2_stats_pre.bin   ← если c2_stats=true
                ├── C2_stats_post.bin  ← если c2_stats=true
                ├── C3_minmax.bin      ← если c3_result=true (Branch 2: MIN+MAX)
                ├── C3_spectrum.bin    ← если c3_spectrum=true (тест: 4.9 ГБ!)
                └── C4_peak.bin        ← если c4_result=true (Branch 3: парабола)
```

---

## StatisticsSet — Пресеты статистики {#statistics-set}

```
Заметка 6: "может сделать наборы из статистики"
6.1 сразу все      6.2 средняя + медиана
6.3 средняя + медиана + мин/макс       6.4 std + дисперсия
```

### Предложение: bitmask enum

```cpp
// Поля статистики (bitmask):
enum class StatField : uint32_t {
    NONE     = 0,
    MEAN     = 1 << 0,   // 0b000001 — mean(Re), mean(Im), mean_magnitude
    MEDIAN   = 1 << 1,   // 0b000010 — median of |z| (radix sort / P²)
    STD      = 1 << 2,   // 0b000100 — std_dev of |z|
    VAR      = 1 << 3,   // 0b001000 — variance of |z|
    MIN      = 1 << 4,   // 0b010000 — min |z| + bin index
    MAX      = 1 << 5,   // 0b100000 — max |z| + bin index
};

using StatisticsSet = uint32_t;  // комбинация StatField через operator|

// ── Готовые пресеты (соответствуют заметке 6) ──────────────────────────
namespace StatPreset {
    constexpr StatisticsSet NONE         = 0;
    constexpr StatisticsSet P62_MEAN_MED = StatField::MEAN | StatField::MEDIAN;
    constexpr StatisticsSet P63_MED_MM   = StatField::MEAN | StatField::MEDIAN
                                         | StatField::MIN  | StatField::MAX;
    constexpr StatisticsSet P64_STD_VAR  = StatField::STD  | StatField::VAR;
    constexpr StatisticsSet P61_ALL      = StatField::MEAN | StatField::MEDIAN
                                         | StatField::STD  | StatField::VAR
                                         | StatField::MIN  | StatField::MAX;
    // 6.5 — зарезервировано для будущих полей
}

// Пример использования:
AntennaProcessorConfig cfg;
cfg.pre_gemm_stats  = StatPreset::P61_ALL;    // все поля до GEMM
cfg.post_gemm_stats = StatPreset::P62_MEAN_MED;  // только mean+median после
```

### Сравнение пресетов

```
Пресет  │ mean │ median │ std │ var │ min │ max │ Размер (256 антенн)
────────┼──────┼────────┼─────┼─────┼─────┼─────┼──────────────────────
6.1 ALL │  ✅  │  ✅    │  ✅ │  ✅ │  ✅ │  ✅ │  ~14 КБ  (все поля)
6.2     │  ✅  │  ✅    │  ❌ │  ❌ │  ❌ │  ❌ │   ~5 КБ
6.3     │  ✅  │  ✅    │  ❌ │  ❌ │  ✅ │  ✅ │   ~9 КБ
6.4     │  ❌  │  ❌    │  ✅ │  ✅ │  ❌ │  ❌ │   ~5 КБ
NONE    │  ❌  │  ❌    │  ❌ │  ❌ │  ❌ │  ❌ │    0 КБ  (пропуск)
────────┴──────┴────────┴─────┴─────┴─────┴─────┴──────────────────────
```

> 💡 **Если** StatisticsSet == NONE → welford_fused и radix_sort не запускаются → Stream 1 отключён → экономим GPU-время при ненужной статистике.

---

## Формат файлов {#file-format}

### Бинарный формат (production)

```
╔══════════════════════════════════════════════════════════════╗
║  Binary Checkpoint File Format (без JSON header)             ║
╠══════════════════════════════════════════════════════════════╣
║  [0..3]    : Magic = 0x43504B54 ("CPKT")                    ║
║  [4..7]    : Version = 1                                     ║
║  [8..11]   : Checkpoint ID (1=C1, 2=C2, 3=C3, 4=C4)        ║
║  [12..15]  : Data type (0=cf32, 1=f32, 2=u32, 3=mixed)      ║
║  [16..19]  : N_ant                                           ║
║  [20..23]  : N_samples / nFFT (зависит от чекпоинта)        ║
║  [24..31]  : sample_rate (double)                            ║
║  [32..39]  : timestamp (Unix time, double)                   ║
║  [40..43]  : gpu_id                                          ║
║  [44..47]  : data_size_bytes                                 ║
║  [48..   ] : binary data (little-endian)                     ║
╚══════════════════════════════════════════════════════════════╝
```

### JSON+binary формат (debug режим, json_header=true)

```
checkpoint_C4_peak.bin.json:
{
    "magic": "CPKT",
    "version": 1,
    "checkpoint_id": 4,
    "timestamp": "2026-03-06T14:32:05.123Z",
    "gpu_id": 0,
    "mode": "BRANCH_3_PARABOLA",
    "N_ant": 256,
    "nFFT": 2097152,
    "sample_rate": 1200000.0,
    "data_type": "MaxValue",
    "data_size_bytes": 12288,
    "fields": ["beam_id","peak_bin","peak_frequency_hz","real","imag",
               "magnitude","phase_deg","freq_offset","refined_freq_hz",
               "mag_left","mag_right","pad"]
}
checkpoint_C4_peak.bin:  ← binary data (12 КБ)
```

### Python читалка

```python
class CheckpointReader:
    MAGIC = b'CPKT'

    @staticmethod
    def read_c4_peak(path: str) -> np.ndarray:
        dtype = np.dtype([
            ('beam_id', np.uint32),
            ('peak_bin', np.uint32),
            ('peak_frequency_hz', np.float32),
            ('real', np.float32), ('imag', np.float32),
            ('magnitude', np.float32),
            ('phase_deg', np.float32),
            ('freq_offset', np.float32),
            ('refined_freq_hz', np.float32),
            ('mag_left', np.float32),
            ('mag_right', np.float32),
            ('pad', np.uint32),
        ])
        with open(path, 'rb') as f:
            header = f.read(48)   # 48-байтный header
            data = np.frombuffer(f.read(), dtype=dtype)
        return data

    @staticmethod
    def read_c1_signal(path: str) -> np.ndarray:
        with open(path, 'rb') as f:
            hdr = np.frombuffer(f.read(48), dtype=np.uint8)
            N_ant     = int.from_bytes(hdr[16:20], 'little')
            N_samples = int.from_bytes(hdr[20:24], 'little')
            data = np.frombuffer(f.read(), dtype=np.complex64)
        return data.reshape(N_ant, N_samples)
```

---

## Сводная таблица {#summary}

```
╔════╦══════════════════════════════╦═══════════╦══════════╦═════════════════════════════════╗
║ #  ║ Checkpoint                   ║ Размер    ║ По умол. ║ Когда нужен                     ║
╠════╬══════════════════════════════╬═══════════╬══════════╬═════════════════════════════════╣
║ C1 ║ S_signal (d_data)            ║ 2.45 ГБ   ║ false    ║ Проверка DMA / integrity        ║
║ C1 ║ W_weights (d_weights)        ║ 512 КБ    ║ false    ║ Проверка матрицы весов          ║
╠════╬══════════════════════════════╬═══════════╬══════════╬═════════════════════════════════╣
║ C2 ║ X_data (GEMM output)         ║ 2.45 ГБ   ║ false    ║ Проверка beamforming            ║
║ C2 ║ stats_pre (по S)             ║ 14 КБ     ║ false    ║ Анализ входного сигнала         ║
║ C2 ║ stats_post (по X)            ║ 14 КБ     ║ false    ║ Анализ после beamforming        ║
╠════╬══════════════════════════════╬═══════════╬══════════╬═════════════════════════════════╣
║ C3 ║ MinMaxResult (ветка 2)       ║ 8 КБ      ║ true     ║ Основной результат ветки 2      ║
║ C3 ║ d_spectrum (полный)          ║ 4.9 ГБ    ║ false    ║ Глубокий анализ спектра         ║
║ C3 ║ AllMaximaResult (заказчик)   ║ ~2 МБ     ║ false    ║ Все пики > CFAR (по запросу)    ║
╠════╬══════════════════════════════╬═══════════╬══════════╬═════════════════════════════════╣
║ C4 ║ MaxValue (ветка 3)           ║ 12 КБ     ║ true     ║ Финальный результат ветки 3     ║
╚════╩══════════════════════════════╩═══════════╩══════════╩═════════════════════════════════╝

Итого "всегда включено": C3 (8 КБ) + C4 (12 КБ) = 20 КБ → почти бесплатно ✅
Итого "максимальная отладка": ~10 ГБ — только при разработке/диагностике 🐛
```

### Вопросы — ЗАКРЫТЫ (ответы получены)

```
✅ Q1 ЗАКРЫТ: ветки переключаемые (одна за раз по mode)
   Branch 2 (MINMAX)   → MinMaxResult[N_ant]
   Branch 3 (PARABOLA) → MaxValue[N_ant]
   Branch 4 (ALL_MAXIMA) → AllMaximaBeamResult[N_ant]  ← ТОЛЬКО внутренние тесты

✅ Q2 ЗАКРЫТ: pre_gemm_stats и post_gemm_stats — независимые StatisticsSet пресеты
   Например: pre=P61_ALL, post=P62_MEAN_MED (или NONE для отключения)

✅ Q3 ЗАКРЫТ: c3_spectrum (4.9 ГБ) — ДА, нужен при тестировании
   Для среднего/медианы спектра → отдельный kernel из VRAM (не через CPU!)
   Streaming save не нужен — сохраняем блоком после hipEventSynchronize(event_fft_done)

✅ Q4 ЗАКРЫТ: detect_all_maxima = Branch 4 (отдельный BranchMode::ALL_MAXIMA)
   Используется для редких контрольных вызовов, не путать с Branch 2 (min/max)

✅ MaxValue (Note #3): ПЕРЕИСПОЛЬЗУЕТСЯ из fft_maxima (spectrum_result_types.hpp)
   Нет дублирования кода! beam_id хранится в обёртке AllMaximaBeamResult

✅ Logs path (Note #4): Logs/GPU_XX/antenna_processor/YYYY-MM-DD/HH-MM-SS/
```

---

*Создано: 2026-03-06*
*Версия: v1.0 — первичные отчёты, требуется обсуждение Q1-Q4*

### Дополнение
# 0 Название!
в файле .claude\worktrees\hungry-bohr\Doc_Addition\PLAN\antenna_processor_pipeline.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 СТРУКТУРА МОДУЛЯ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  modules/strategies/                    ← один модуль, много стратегий (Branch 2/3/4)
  ├── include/

Примечание: модуль = strategies; базовый класс = AntennaProcessor (antenna_processor.hpp) 

# 1
Сделай 3 ветки!!! ПОДЧЕРКМВАЮ! в разговоре заказчих хотел 3 вариант
3. [C3 — После FFT: Global Min/Max (Ветка 2)](#c3)
4. [C4 — После FFT: Один Max + Парабола (Ветка 3)]
хотел так 
1. [C3 — После FFT: один Global Min/Max (Ветка 2)](#c3)
2. [C4 — После FFT: Один Max + Парабола (Ветка 3)]
3. После FFT: все Global Min/Max - это отдельная ветка для внктреннего тестирования

# 2
учти что при FFT у нас пик может быть зеркальным с правой стороны нужно его пересчитать (перенести в лево) 

# 3
Структура MaxValue (per beam) - спользуем то что уже есть?

# 4 не правильно
Logs/
└── antenna_processor/
    └── 2026-03-06/
должно быть
Logs/
└── GPU_XX(номер)/
    └── antenna_processor/
        └── 2026-03-06/
# 5 ОТВЕТЫ
 Q1: C3 и C4  A. — ЭТО ПЕРЕКЛЮЧАЕМЫЕ ВЕТКИ? - Да в зависимости от режима работы используется одна из веток
 Q2: Статистика PRE-GEMM  A.
 Q3: Полный спектр (C3, 4.9 ГБ) — нужен заказчику? ДА -для проверки и только во время теста. Как будем считать среднее и медиану? как вариант сделать вызыв кернела из памяти и поститать эти параиетры
  Q4: detect_all_maxima (CFAR) — это ВЕТКА 2 или  - Это ветка для каких то редких вызовов для контроля 

# 6 когда я просил создвть С1-С4 я имел ввиду как в архитектуре .claude\worktrees\hungry-bohr\Doc\Architecture 
- что бы видеть все связи временные деаграммы - что классы реализуют SOLID, Шаблон GRASP, шаблон GoF. Потоки врасчетные времена - 
что бы по ним 
 - подродно написать Tasks
 - проверть сделанную работу
 - дабавить в отчет
 