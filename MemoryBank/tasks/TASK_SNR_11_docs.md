# TASK SNR_11: Документация SNR-estimator

> **Дата**: 2026-04-09
> **Модуль**: `Doc/Modules/statistics/` + `Doc/Python/statistics_api.md`
> **Приоритет**: Low (после успешных тестов)
> **Статус**: BACKLOG
> **Зависимости**: T1 (C++ тесты запущены и прошли) + T2 (Python e2e запущен и прошёл)
> **Ревьюер**: Кодо
>
> 📐 **План**: **Часть 5** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

Обновить документацию модуля `statistics` — добавить раздел SNR Estimator с архитектурой, формулами, примерами C++ и Python. **Только после** успешных тестов (понедельник).

---

## 📝 Файлы (дополнить)

| Файл | Что добавить |
|---|---|
| `Doc/Modules/statistics/Full.md` | Раздел «SNR Estimator»: архитектура pipeline, формулы CA-CFAR, выбор параметров, примеры |
| `Doc/Modules/statistics/API.md` | `ComputeSnrDb` сигнатуры, `SnrEstimationConfig`, `SnrEstimationResult`, `BranchSelector` |
| `Doc/Modules/statistics/Quick.md` | Пример 5-10 строк (C++ + Python) |
| `Doc/Python/statistics_api.md` | Python API: `compute_snr_db`, `BranchSelector`, numpy dtype |
| `modules/statistics/tests/README.md` | Описание 7 SNR тестов + benchmark |
| `PyPanelAntennas/SNR/README.md` | Описание 5 экспериментов Python модели (SNR_00) |

---

## 📝 Содержание (минимум)

### Full.md — раздел SNR Estimator

1. **Обзор**: зачем нужен (grub SNR для переключения branch)
2. **Архитектура pipeline**:
   ```
   gpu_data → gather_decimated → FFT |X|² → peak_cfar → median → snr_db_global
   ```
3. **Формулы**:
   - Coherent gain: `SNR_fft_dB = SNR_in_dB + 10·log10(N_actual)`
   - CA-CFAR: `SNR_fft_dB = 10·log10(|X_peak|² / mean(|X_ref|²))`
   - Артефакт под H0 (только шум): `≈ ln(N_fft) + γ` ≈ 8-10 dB
4. **Выбор параметров**:
   - `target_n_fft` default 2048 (гибкий — можно 1024/2048/4000/4096)
   - `step_samples` auto или manual
   - `step_antennas` auto → медиана по ≈50 антеннам
   - `guard_bins=3`, `ref_bins=8` — default
5. **Пример C++**:
   ```cpp
   statistics::StatisticsProcessor proc(backend);
   statistics::SnrEstimationConfig cfg;  // auto
   auto result = proc.ComputeSnrDb(data, n_ant, n_samp, cfg);

   statistics::BranchSelector selector;
   auto branch = selector.Select(result.snr_db_global, cfg.thresholds);
   ```
6. **Диаграмма Mermaid** (опционально — см. `Doc_Addition/Mermaid_DarkTheme_Guide.md`)

### API.md

Описать сигнатуры:
- `SnrEstimationResult ComputeSnrDb(CPU data, ...)` / `ComputeSnrDb(GPU data, ...)`
- Все поля `SnrEstimationConfig` с default'ами
- `BranchSelector.Select/Current/Reset`
- `BranchThresholds` (low_to_mid_db, mid_to_high_db, hysteresis_db)

### Quick.md — минимальный пример

```python
import numpy as np
from gpu_work_lib import StatisticsProcessor, SnrEstimationConfig, GPUContext

ctx = GPUContext()
proc = StatisticsProcessor(ctx)
data = np.random.randn(50, 5000).astype(np.complex64)  # dechirped data
cfg = SnrEstimationConfig()  # default target_n_fft=0 (auto 2048)

result = proc.compute_snr_db(data, 50, 5000, cfg)
print(f"SNR: {result.snr_db_global:.1f} dB, antennas: {result.used_antennas}")
```

### Python/statistics_api.md

Раздел «SNR Estimator»:
- numpy dtype требование: `complex64`, shape `(n_antennas, n_samples)`
- Все аттрибуты `SnrEstimationConfig` и `SnrEstimationResult`
- Пример с `BranchSelector`
- Важно: `result.branch` **НЕ существует** — использовать `BranchSelector.select(...)`

---

## ✅ Definition of Done

- [ ] Все 6 файлов дополнены разделом SNR Estimator
- [ ] Формулы корректны (coherent gain через `N_actual`, НЕ `N_fft`)
- [ ] Примеры C++ и Python компилируются/запускаются (ссылка на реальный тест)
- [ ] Упомянут `BranchSelector` как отдельный класс, НЕ внутри `StatisticsProcessor`
- [ ] Упомянут `MedianRadixSortOp::ExecuteFloat(1, n_ant_used)` как переиспользование
- [ ] В Full.md добавлена ссылка на план `MemoryBank/specs/snr_estimator_statistics_plan.md`
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ Формулы используют `N_actual` для coherent gain (НЕ `N_fft`)
- ✅ Описан `|X|²` (square-law), упомянут `complex_to_magnitude_squared`
- ✅ `SnrEstimationResult` описан БЕЗ поля `branch`
- ✅ `BranchSelector` описан как отдельный класс с примером использования
- ✅ Все имена lowercase: `target_n_fft`, `search_full_spectrum`, `n_actual`
- ✅ Пороги упомянуты как «калибруемые из Python модели» (см. SNR_00)
- ✅ Для Python — `dtype=complex64` обязательно (warning про `complex128`)
- ✅ Нет pytest в примерах

---

## 🚫 Запреты

- ❌ НЕ копировать текст из плана целиком — в документации ссылка на план
- ❌ НЕ создавать новые файлы документации — только дополнять существующие
- ❌ НЕ писать примеры с `result.branch` (его не существует)

---

## 🔗 Связанные таски

- **Требует:** все предыдущие (тесты должны пройти в понедельник)
- **Последний таск** в цепочке SNR

---

*Created 2026-04-09 | Кодо*
