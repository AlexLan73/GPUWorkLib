# SNR-estimator — Индекс тасков

> **Спецификация:** [`MemoryBank/specs/snr_estimator_statistics_plan.md`](../specs/snr_estimator_statistics_plan.md) (v4 — после ревью v6, блокеры закрыты)
> **Ревью v5:** [`MemoryBank/specs/snr_estimator_review_2026-04-09.md`](../specs/snr_estimator_review_2026-04-09.md) (архитектурные решения)
> **Ревью v6:** [`MemoryBank/specs/snr_estimator_review_v6_findings_2026-04-09.md`](../specs/snr_estimator_review_v6_findings_2026-04-09.md) (технический аудит кода, все закрыто)
> **Дата создания:** 2026-04-09
> **Автор индекса:** Кодо
> **Ревьюер всех тасков:** Кодо ⭐
> **Модуль:** `modules/statistics` (+ расширение `modules/fft_func`)

---

## 🎯 Цель

Реализовать быстрый **грубый SNR estimator** для переключения ветвей обработки (Low/Mid/High) на основе CA-CFAR после FFT. Работает на уже дечирпированных complex float данных прямо на GPU.

---

## 📅 Workflow

```
   СЕГОДНЯ (Windows)                          ПОНЕДЕЛЬНИК (Debian/ROCm)
  ┌──────────────────────────┐              ┌────────────────────────┐
  │  SNR_00 Python model     │              │  SNR_T1 C++ test run   │
  │  SNR_01 Types            │              │  SNR_T2 Python e2e run │
  │  SNR_02 fft_func squared │              │  SNR_T3 Benchmark      │
  │  SNR_03 gather kernel    │─────────────▶│  SNR_T4 Отладка        │
  │  SNR_04 ProcessMag2GPU   │              └────────────────────────┘
  │  SNR_05 SnrEstimatorOp   │
  │  SNR_06 Facade           │                  ПОСЛЕ ТЕСТОВ
  │  SNR_07 Python bindings  │              ┌────────────────────────┐
  │  SNR_08 C++ tests (код)  │              │  SNR_11 Документация   │
  │  SNR_09 Benchmark (код)  │              └────────────────────────┘
  │  SNR_10 Python e2e (код) │
  └──────────────────────────┘
```

> ⚠️ На Windows ветка `main` не собирается (ROCm only). Поэтому весь код пишем сегодня, **запускаем в понедельник** на Debian.

---

## 📋 Порядок выполнения

```
SNR_00 (python model) ─────────┐ (результаты нужны для калибровки порогов)
                                │
SNR_01 (types) ─────────────────┤
                                │
SNR_02 (fft_func squared) ──────┤
                                │
SNR_03 (gather kernel) ─────────┤
                                ▼
SNR_04 (ProcessMag2GPU) ──► SNR_05 (SnrEstimatorOp) ──► SNR_06 (Facade)
                                                             │
                                                             ▼
                                                      SNR_07 (Py bindings)
                                                             │
                                                             ▼
                              SNR_08 (C++ tests) ──► SNR_09 (Benchmark)
                                                             │
                                                             ▼
                                                      SNR_10 (Python e2e)
                                                             │
                                                             ▼
                                 [ПОНЕДЕЛЬНИК: запуск, отладка]
                                                             │
                                                             ▼
                                                      SNR_11 (Docs)
```

---

## 📁 Таски

| # | Файл | Что делает | Зависимости | Где пишется | Статус |
|---|------|-----------|-------------|-------------|--------|
| 00 | [TASK_SNR_00_python_model.md](TASK_SNR_00_python_model.md) | Python модель (5 экспериментов) — калибровка параметров и порогов | — | `PyPanelAntennas/SNR/` | ✅ **DONE** |
| 01 | [TASK_SNR_01_types.md](TASK_SNR_01_types.md) | Дополнить `statistics_types.hpp` (Config, Result, BranchType, `window`, shared_buf) | 02b (WindowType) | `modules/statistics/include/` | ✅ **CODE DONE** (compile Mon) |
| 02 | [TASK_SNR_02_fft_func_squared.md](TASK_SNR_02_fft_func_squared.md) | Новый kernel `complex_to_magnitude_squared` + параметр `squared` в `MagnitudeOp` | — | `modules/fft_func/` | ✅ **CODE DONE** (compile Mon) |
| **02b** | [TASK_SNR_02b_pad_data_windowed.md](TASK_SNR_02b_pad_data_windowed.md) | 🆕 `WindowType` enum + kernel `pad_data_windowed` + параметр `window` в `PadDataOp` | — | `modules/fft_func/` | ✅ **CODE DONE** (compile Mon) |
| 03 | [TASK_SNR_03_gather_kernel.md](TASK_SNR_03_gather_kernel.md) | HIP kernel `gather_decimated_kernel` (thread-per-antenna) | — | `modules/statistics/kernels/` | ✅ **CODE DONE** (compile Mon) |
| 04 | [TASK_SNR_04_fft_process_to_gpu.md](TASK_SNR_04_fft_process_to_gpu.md) | Новый метод `FFTProcessorROCm::ProcessMagnitudesToGPU(window)` | 02, 02b | `modules/fft_func/` | ✅ **CODE DONE** (compile Mon) |
| 05 | [TASK_SNR_05_snr_estimator_op.md](TASK_SNR_05_snr_estimator_op.md) | `SnrEstimatorOp` (Layer 5) + `peak_cfar_kernel` + `BranchSelector` | 01, 03, 04 | `modules/statistics/include/operations/` | ✅ **CODE DONE** (compile Mon) |
| 06 | [TASK_SNR_06_facade.md](TASK_SNR_06_facade.md) | `ComputeSnrDb` в `StatisticsProcessor` (Layer 6 Facade, stateless) | 05 | `modules/statistics/` | ✅ **CODE DONE** (compile Mon) |
| 07 | [TASK_SNR_07_python_bindings.md](TASK_SNR_07_python_bindings.md) | pybind11 экспорт: Config, Result, BranchSelector, compute_snr_db | 06 | `modules/statistics/python/` | ✅ **CODE DONE** (compile Mon) |
| 08 | [TASK_SNR_08_cpp_tests.md](TASK_SNR_08_cpp_tests.md) | C++ тесты (test_01..test_06b) + `snr_test_helpers.hpp` | 06 | `modules/statistics/tests/` | ✅ **CODE DONE** (run Mon) |
| 09 | [TASK_SNR_09_benchmark.md](TASK_SNR_09_benchmark.md) | `snr_estimator_benchmark.hpp` + runner (наследник `GpuBenchmarkBase`) | 06 | `modules/statistics/tests/` | ✅ **CODE DONE** (run Mon) |
| 10 | [TASK_SNR_10_python_e2e.md](TASK_SNR_10_python_e2e.md) | Python e2e: signal_generators → heterodyne → SNR → сверка с numpy | 07 | `Python_test/statistics/` | ✅ **CODE DONE** (run Mon) |
| 11 | [TASK_SNR_11_docs.md](TASK_SNR_11_docs.md) | Full.md / API.md / Quick.md / Python API | T1, T2 | `Doc/Modules/statistics/` | ⏳ BACKLOG |

---

## 🔑 Ключевые решения (не менять без обсуждения!)

1. **Size N_fft — гибкий**: `target_n_fft` параметр Config, default 2048. `N_actual` после децимации → `NextPowerOf2(N_actual)` автоматически.
2. **Square-law detector**: новый kernel `complex_to_magnitude_squared` (БЕЗ sqrt, ~7× быстрее), вызов через `MagnitudeOp::Execute(..., squared=true)`.
3. **🆕 Window function**: **Hann** (default) — решает проблему sinc sidelobes (без window −27 dB bias!). Расширение `PadDataOp` параметром `WindowType`, kernel `pad_data_windowed`. SNR_02b.
4. **FFT pipeline**: `FFTProcessorROCm::ProcessMagnitudesToGPU(squared, window)` (новый метод), НЕ `ISpectrumProcessor`.
5. **CFAR estimator**: CA-CFAR (mean) — калибровано в Python. OS-CFAR (median) даёт +1-3 dB но сложнее, отложен.
6. **🆕 guard=5, ref=16** (было 3/8) — калибровано для Hann window в Python Эксп.5.
7. **🆕 Пороги**: `low_to_mid=15 dB, mid_to_high=30 dB` — калиброваны в Python Эксп.5 (P_correct=97.9%).
8. **Median**: переиспользуем `MedianRadixSortOp::ExecuteFloat(1, n_ant_used)` — он именно для малых (<100K) массивов.
9. **Hysteresis**: отдельный класс `BranchSelector` — facade `StatisticsProcessor` остаётся stateless.
10. **Gather thread mapping**: один поток на антенну, sequential loop по samples (НЕ один поток на элемент).
11. **BatchManager**: НЕ используем, данные целиком + проверка памяти с `std::runtime_error`.
12. **Тесты**: пишем сегодня, запускаем в понедельник на Debian/AMD (нет AMD GPU под Windows).
13. **Профилирование**: только через `GPUProfiler` (`PrintReport`/`ExportMarkdown`/`ExportJSON`).
14. **Бенчмарки**: наследники `GpuBenchmarkBase`, namespace `test_snr_estimator`.

## 📊 Python калибровка (SNR_00 DONE)

Python модель в [`PyPanelAntennas/SNR/`](../../PyPanelAntennas/SNR/):
- 5 экспериментов × 8 комбинаций (rect/hann/hamming/blackman × mean/median)
- 3 финальных графика показывают что **Hann + mean — оптимальный компромисс**
- Калиброванные пороги сохранены в `results/exp5_thresholds.json`

**Результат:** `low_to_mid_db=15, mid_to_high_db=30, P_correct=97.9%` для Hann + mean.

---

## ⚠️ Важные замечания для всех исполнителей

### Стандарты GPUWorkLib (обязательно к соблюдению)
- **Код в файлах** согласно `CLAUDE.md`: корень репо, **НЕ** в `.claude/worktrees/`
- **Консольный вывод** — только через `drv_gpu_lib::ConsoleOutput::GetInstance()`
- **Профилирование** — только через `GPUProfiler`, НЕ вручную `GetStats() + cout`
- **НЕ вызывать `hipfftExecC2C` напрямую** — только через `FFTProcessorROCm`
- **Стиль**: Google C++ Style + 2 пробела, CamelCase классы, snake_case методы, `kConstant` константы
- **Один класс — один файл** (Op'ы в `operations/`, kernels в `kernels/`)
- **Тесты**: `.hpp` файлы в `{module}/tests/`, запуск через `all_test.hpp`
- **Python тесты**: БЕЗ `pytest`, только через `common/runner.py::TestRunner`

### Что будет проверять Кодо (ревьюер)
- Соответствие плану **v4** (`snr_estimator_statistics_plan.md`)
- Правильные имена: `target_n_fft`, `search_full_spectrum`, `n_actual`, `SnrEstimationConfig`, `SnrEstimationResult`
- `SnrEstimationResult` НЕ содержит `BranchType` (это `BranchSelector`)
- `MagnitudeOp::Execute(..., squared)` — default `false`
- `complex_to_magnitude_squared` добавлен в **обе** функции source
- Kernel `complex_to_magnitude_squared` БЕЗ `sqrt`/`sqrtf`/`__fsqrt_rn`
- Kernel **`gather_decimated`** (имя без суффикса `_kernel`!) — поток на антенну
- Kernel **`peak_cfar`** по разделу **2.2.7**: LDS argmax + parallel atomicAdd ref-sum + `fmaxf` + без `search_full_spectrum` параметра
- `BranchSelector::Select` — первая строка `if (!std::isfinite(snr_db)) return current_;`
- Переиспользование `MedianRadixSortOp::ExecuteFloat(1, n_ant_used)` — НЕ новый median kernel
- `hipfftExecC2C` — ТОЛЬКО через `FFTProcessorROCm`
- `StatisticsProcessor` остаётся stateless (hysteresis в `BranchSelector`)
- `StatisticsProcessor` хранит `backend_` + `snr_op_initialized_` (не `HasContext()`!)
- GPU memory check НЕ учитывает входные `n_antennas × n_samples` (double-count!)

---

## 📊 Определение готовности (Definition of Done) для всех тасков

Таск считается DONE когда:
- [ ] Код написан по плану и прошёл ревью Кодо
- [ ] Критерии ревью из шапки таска закрыты
- [ ] Нет упоминаний устаревших имён (`target_N_fft`, `search_left_right`)
- [ ] Нет нарушений стандартов GPUWorkLib
- [ ] Компилируется локально (где применимо — SNR_00 Python model)
- [ ] Статус в этом INDEX обновлён на ✅ DONE

Для тасков 08-10 (тесты) — `DONE = код написан`. Запуск и проверка прохождения в понедельник (отдельные T-таски будут созданы после).

---

*Created 2026-04-09 | Reviewed & updated 2026-04-09 (v4 plan sync) | Кодо*
