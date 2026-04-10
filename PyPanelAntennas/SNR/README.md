# PyPanelAntennas/SNR — Python модель SNR-estimator

> Numpy/scipy модель для **калибровки параметров** C++ SNR-estimator до написания GPU кода.
>
> Связанный таск: [`MemoryBank/tasks/TASK_SNR_00_python_model.md`](../../MemoryBank/tasks/TASK_SNR_00_python_model.md)
> План: [`MemoryBank/specs/snr_estimator_statistics_plan.md`](../../MemoryBank/specs/snr_estimator_statistics_plan.md)

---

## ✅ Статус: КАЛИБРОВКА ЗАВЕРШЕНА (2026-04-09)

**Финальное решение для C++:**
| Параметр | Значение |
|---|---|
| Window function | **Hann** |
| CFAR estimator | **CA-CFAR (mean)** |
| guard_bins | **5** |
| ref_bins | **16** |
| target_n_fft | 2048 (default, гибкий) |
| **low_to_mid_db** | **15.0** |
| **mid_to_high_db** | **30.0** |
| hysteresis_db | 2.0 |
| **P_correct** | **97.9%** |

Эти значения пойдут в `snr_defaults::` namespace в `modules/statistics/include/statistics_types.hpp`.

---

## 📁 Структура

```
PyPanelAntennas/SNR/
├── cfar_estimator.py         — CA-CFAR + OS-CFAR + 5 окон (rect/hann/hamming/blackman/flattop)
├── lfm_signal_generator.py   — генератор CW / LFM / AWGN
├── dechirp_numpy.py          — numpy дечирп (умножение на conj(ref))
├── snr_estimator_model.py    — главный скрипт (5 экспериментов × 8 комбинаций)
├── quick_test.py             — smoke test для PyCharm отладки
├── debug_cfar_bias.py        — 🎨 график 1: обоснование выбора Hann + mean
├── debug_window_vs_guard.py  — 🎨 график 2: масштабирование Hann по N_actual
├── debug_peak_grows.py       — 🎨 график 3: НАРОДНАЯ версия — пик растёт с SNR
├── README.md                 — этот файл
├── plots/                    — PNG графики (9 штук)
└── results/                  — JSON результаты + exp5_thresholds.json (для C++)
```

## 📊 Графики в `plots/`

### Финальные (выбранная обработка Hann + mean)
1. **`debug_cfar_bias.png`** — почему выбрали Hann (sinc sidelobes, SNR vs guard)
2. **`debug_window_vs_guard.png`** — масштабирование Hann по N_actual (1024..8192)
3. **`debug_peak_grows.png`** — НАРОДНАЯ: серия спектров при растущем SNR_in

### Эксперименты (все 8 комбинаций)
4. **`exp0_comparison.png`** — сравнение 8 комбинаций (bar chart)
5. **`exp1_basic_curve.png`** — SNR_in → SNR_fft кривая + H0 артефакт
6. **`exp2_scaling.png`** — масштабирование по n_samples (2K..1.3M)
7. **`exp3_step_samples.png`** — влияние step_samples на точность
8. **`exp4_antennas.png`** — стабилизация медианы по антеннам
9. **`exp5_roc_thresholds.png`** — КАЛИБРОВКА порогов (ROC-like)

---

## 🚀 Быстрый старт

### Smoke test (PyCharm debug)
```bash
# Запустите quick_test.py в PyCharm с breakpoint'ами
python quick_test.py
```
Проверяет что все модули импортируются, генерация работает, CFAR даёт
разумный результат. Показывает 2 графика: спектр `|X|²` и входной сигнал.

### Полный прогон (все 5 экспериментов)
```bash
# Windows (дома)
"F:\Program Files (x86)\Python314\python.exe" snr_estimator_model.py

# Debian (работа)
python3 snr_estimator_model.py

# Быстрый прогон (меньше trials — для отладки)
python3 snr_estimator_model.py --quick

# Только один эксперимент
python3 snr_estimator_model.py --exp 5  # только калибровка порогов
```

---

## 🧪 Эксперименты

| # | Что проверяет | Главный вывод |
|---|---|---|
| **1** | Базовая кривая `SNR_in → SNR_fft` + H0 артефакт | Offset = `10·log10(N_actual)`, артефакт CFAR ≈ 8-10 dB |
| **2** | Масштабирование по `n_samples` (2K → 1.3M) | Устойчивость к размеру входа, auto-step правильно работает |
| **3** | Влияние `step_samples` (без децимации vs сильная) | "Колено" точность vs скорость |
| **4** | Стабилизация медианы по `N_ants` | Минимальное `N_ants` для `σ(median) < 1 dB` |
| **5** ⭐ | **Калибровка порогов branch** | `low_to_mid_db`, `mid_to_high_db` для C++ |

**Главный выход** — [`results/exp5_thresholds.json`](results/exp5_thresholds.json), содержит:
```json
{
  "low_to_mid_db": <калиброванное значение>,
  "mid_to_high_db": <калиброванное значение>,
  "hysteresis_db": 2.0,
  "p_correct": <P(правильного переключения)>
}
```
Эти значения пойдут в default'ы `BranchThresholds` в C++ при реализации SNR_05.

---

## 🎯 Ключевые параметры (повторяют C++ SnrEstimationConfig)

```python
from cfar_estimator import CfarConfig

cfg = CfarConfig(
    target_n_fft=0,       # 0 → auto = 2048 (default из snr_defaults)
    step_samples=0,       # 0 → auto = ceil(n_samples / target_n_fft)
    step_antennas=0,      # 0 → auto = ceil(n_ant / 50)
    guard_bins=3,
    ref_bins=8,
    search_full_spectrum=True,
)
```

### Pipeline sizes (как в C++ `SnrEstimatorOp`)

```python
from cfar_estimator import compute_pipeline_sizes

step, n_actual, n_fft = compute_pipeline_sizes(
    n_samples=1_300_000,
    target_n_fft=0,   # → 2048
    step_samples=0,    # → auto
)
# step = 635, n_actual = 2047, n_fft = 2048
```

---

## 🧮 Математика

### Coherent gain
```
SNR_fft_dB = SNR_in_dB + 10·log10(N_actual)
```
Где `N_actual` — число **ненулевых** сэмплов после децимации. Zero-padding **не добавляет** энергии — он только интерполирует спектр.

### CA-CFAR (square-law)
```
k_peak    = argmax(|X[k]|²)
P_noise   = mean(|X[k_ref]|²) по guard+ref окну с wraparound
SNR_fft_dB = 10·log10(|X_peak|² / P_noise)
```

### Артефакт H0 (только шум)
```
E[SNR_fft_dB | H0] ≈ 10·log10(ln(N_fft) + γ)
                   ≈ 8-10 dB  (для N_fft ∈ [1024, 8192])
```
Порог `low_to_mid_db` **обязан** быть > 10 dB, иначе чистый шум попадёт в Mid branch.

---

## 🐍 Для e2e тестов Python_test/statistics/

Класс-обёртка `CfarEstimator` совместим с Python bindings:

```python
from cfar_estimator import CfarEstimator

ref = CfarEstimator(target_n_fft=0, guard_bins=3, ref_bins=8)
snr_db = ref.estimate(signal_numpy)          # 1 антенна
snr_db_median = ref.estimate_batch(data_2d)  # [n_ant, n_samp]
```

Используется в [`Python_test/statistics/test_snr_estimator.py`](../../Python_test/statistics/test_snr_estimator.py) для сверки GPU результата с numpy эталоном.

---

## ⚠️ Известные ограничения модели

1. **numpy FFT != rocFFT** — результаты могут отличаться на уровне float32 округлений. Приемлемая толерантность: `|GPU - numpy| < 1 dB`.
2. **Нет параллелизма** — Python модель последовательная, на больших сценариях (1.3M samples × 256 антенн) работает **медленно** (минуты). Используйте `--quick` или `--exp N` для отладки.
3. **dtype = complex64** обязательно — complex128 даст чуть другие округления.

---

*Author: Кодо | Дата: 2026-04-09 | Привязка: TASK_SNR_00*
