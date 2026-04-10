# TASK SNR_00: Python модель SNR-estimator (5 экспериментов)

> **Дата**: 2026-04-09
> **Модуль**: `PyPanelAntennas/SNR/` (новая папка)
> **Приоритет**: High (блокирует C++ — нужны калиброванные пороги)
> **Статус**: BACKLOG
> **Зависимости**: — (первый таск)
> **Ревьюер**: Кодо
>
> 📐 **План**: [`../specs/snr_estimator_statistics_plan.md`](../specs/snr_estimator_statistics_plan.md) → **Часть 1**
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

Построить numpy/scipy модель SNR-estimator'а, провести 5 экспериментов, **откалибровать пороги** `low_to_mid_db` и `mid_to_high_db` (Эксп.5), проверить математику и найти оптимальные параметры (`step_samples`, `guard_bins`, `ref_bins`, `target_n_fft`) до C++.

---

## 📁 Файловая структура (создать)

```
PyPanelAntennas/SNR/
├── snr_estimator_model.py    — главный скрипт, запускает все 5 экспериментов
├── lfm_signal_generator.py   — генератор ЛЧМ + AWGN (complex float)
├── dechirp_numpy.py          — numpy гетеродин (дечирп LFM → CW)
├── cfar_estimator.py         — CA-CFAR SNR оценщик (основной класс)
├── plots/                    — PNG графики экспериментов
│   ├── exp1_basic_curve.png
│   ├── exp2_scaling.png
│   ├── exp3_step_samples.png
│   ├── exp4_antennas.png
│   └── exp5_roc_thresholds.png
└── results/                  — JSON с численными результатами
    ├── exp1_results.json
    ├── exp2_results.json
    ├── exp3_results.json
    ├── exp4_results.json
    └── exp5_thresholds.json   ← главный выход, калиброванные пороги
```

---

## 🧪 Эксперименты (подробно — в плане, L213-281)

| # | Цель | Переменная | Выход |
|---|------|------------|-------|
| 1 | Проверить теорию: `SNR_fft = SNR_in + 10·log10(N_actual)` | SNR_in ∈ {5..40} dB | Прямая с offset |
| 2 | Масштабирование n_samples 2K → 1.3M (лог. шаг, 11 точек) | `n_samples` | 3 кривые, σ(error) |
| 3 | Влияние `step_samples` на точность | step ∈ {1,2,4,8,16} | «колено» точность vs скорость |
| 4 | Стабилизация медианы по числу антенн | `N_ants_used` | Var(median) → min N_ants |
| 5 | **Калибровка порогов** через ROC-кривые | SNR_in шаг 1 dB | `low_to_mid_db`, `mid_to_high_db` при P_correct > 90% |

---

## 🔑 Ключевые параметры (default)

```python
target_n_fft   = 2048   # default из плана, гибкий
guard_bins     = 3
ref_bins       = 8
search_full_spectrum = True
step_antennas_auto_target = 50  # медиана по ~50 антеннам
```

---

## ✅ Critical math checks (обязательно проверить в Эксп.1)

1. **Coherent gain = `10·log10(N_actual)`** — НЕ `N_fft`! Zero-padding не добавляет энергии.
2. **Square-law distribution**: `|X[k]|² ~ Exp(N·σ²)` под H0 (проверить гистограммой)
3. **Артефакт CFAR** на чистом шуме: `E[SNR_fft_db_noise] ≈ 10·log10(ln(N_fft) + 0.577) ≈ 8-10 dB`
4. **Bias от sinc-боковых лепестков** при `N_actual << N_fft` (сильный zero-pad): проверить `guard_bins` зависимость

---

## ✅ Definition of Done

- [ ] 4 Python файла + 1 главный скрипт созданы
- [ ] Все 5 экспериментов запускаются без ошибок (Windows через `F:\Program Files (x86)\Python314\python.exe` или Debian `python3`)
- [ ] PNG графики сохранены в `plots/`
- [ ] JSON результаты сохранены в `results/`
- [ ] `exp5_thresholds.json` содержит финальные `low_to_mid_db` и `mid_to_high_db` при P_correct > 90%
- [ ] В Эксп.1 теория совпадает с экспериментом на ±2 dB (иначе искать ошибку в CFAR)
- [ ] Артефакт CFAR на чистом шуме измерен и задокументирован (Эксп.1 + отдельный прогон без сигнала)
- [ ] Кодо провёл ревью (математика + стиль + нет pytest)
- [ ] Статус в [TASK_SNR_INDEX.md](TASK_SNR_INDEX.md) обновлён на ✅ DONE

---

## 🚫 Запреты

- ❌ **НЕТ pytest** — только обычный `if __name__ == "__main__"` main
- ❌ НЕТ зависимостей от GPUWorkLib C++ кода (чистый numpy/scipy)
- ❌ НЕ использовать комплексный FFT из `scipy.fft.fftshift` для поиска пика — работать с `[0..N-1]` напрямую (как будет в C++)

---

## 📝 Заметки для исполнителя

1. **matplotlib backend:** использовать `matplotlib.use('Agg')` для работы без X-сервера на Debian
2. **Reproducibility:** все `np.random` должны использовать `seed=42` (или передаваемый параметр)
3. **Большие массивы (1.3M):** использовать `np.complex64` (не `complex128`), иначе memory issue
4. **Сравнение с теорией:** в каждом эксперименте печатать `measured vs theoretical` для дебага
5. **Названия графиков:** обязательно title/xlabel/ylabel/legend/grid — Кодо будет смотреть графики

---

## 🔗 Связанные таски

- **Следующий:** [TASK_SNR_01_types.md](TASK_SNR_01_types.md) — после калибровки порогов можно писать C++ типы
- **Параллельно:** [TASK_SNR_02_fft_func_squared.md](TASK_SNR_02_fft_func_squared.md) — fft_func расширение не зависит от Python модели

---

*Created 2026-04-09 | Кодо*
