# IN PROGRESS — Текущие задачи

> **Обновлено**: 2026-03-10
> **Главный план**: `MemoryBank/tasks/MODULES_WORK_PLAN.md`
> **Фокус**: ✅ Все задачи MODULES_WORK_PLAN завершены — ждём GPU AMD 9070 для финального запуска

---

## ~~TASK-01~~ — DrvGPU: External Context тесты ✅ COMPLETED

**Статус**: ✅ 2026-03-09 — все тесты раскомментированы в `DrvGPU/tests/all_test.hpp`
- `test_drv_gpu_external.hpp` — `test_drv_gpu_external::run()`
- `test_rocm_external_context.hpp` — `test_rocm_external_context::run()`
- `test_hybrid_external_context.hpp` — `test_hybrid_external_context::run()`

---

## ~~TASK-04~~ — signal_generators: FormSignalROCm Python Binding ✅ COMPLETED

**Статус**: ✅ 2026-03-10
- `python/py_form_signal_rocm.hpp`: `PyFormSignalGeneratorROCm`
  - `set_params(antennas, points, fs, f0, amplitude, fdev, noise_amplitude, tau_step, ...)`
  - `set_params_from_string("f0=1e6,antennas=5,...")` — CSV парсинг
  - `generate()` → `numpy complex64 (antennas, points)`
  - `get_params()` → `dict` всех параметров
- `gpu_worklib_bindings.cpp`: `#include` + `register_form_signal_rocm(m)`
- `Python_test/signal_generators/test_form_signal_rocm.py`: 9 passed, 6 skipped (GPU)

---

## ~~TASK-03~~ — fft_maxima: Python Binding ✅ COMPLETED

**Статус**: ✅ 2026-03-10
- `python/py_spectrum_maxima_finder_rocm.hpp`: `PySpectrumMaximaFinderROCm`
  - `process()` — ONE_PEAK: один пик на луч с параболической интерполяцией
  - `find_all_maxima()` — ALL_MAXIMA из готового FFT спектра (без FFT)
  - `find_all_maxima_from_signal()` — ALL_MAXIMA из raw сигнала (с FFT)
  - Формат вывода совместим с OpenCL `SpectrumMaximaFinder`
- `gpu_worklib_bindings.cpp`: `#include` + `register_spectrum_maxima_finder_rocm(m)`
- `Python_test/fft_maxima/test_spectrum_maxima_finder_rocm.py`: 8 passed, 6 skipped (GPU)

---

## ~~TASK-02~~ — fft_processor: Python Binding ✅ COMPLETED

**Статус**: ✅ 2026-03-10
- `python/py_fft_processor_rocm.hpp`: `PyFFTProcessorROCm` — `process_complex`, `process_mag_phase`, `get_profiling`, `nfft`
- `gpu_worklib_bindings.cpp`: `#include "py_fft_processor_rocm.hpp"` + `register_fft_processor_rocm(m)`
- `Python_test/fft_processor/test_fft_processor_rocm.py`: 8 passed, 7 skipped (GPU)
- `Doc/Modules/fft_processor/API.md`: добавлена секция Python API + примеры

---

## ~~TASK-05~~ — fm_correlator: Тесты + API.md ✅ COMPLETED

**Статус**: ✅ 2026-03-10
- `modules/fm_correlator/tests/all_test.hpp`: раскомментированы 4 базовых теста
- `Python_test/fm_correlator/test_fm_correlator.py`: 8 passed, 4 skipped (GPU)
- `Doc/Modules/fm_correlator/API.md`: создан (C++ + Python API)

---

## ~~TASK-06~~ — strategies: Финализация ✅ COMPLETED (кроме GPU запуска)

**Статус**: ✅ Код/тесты/визуализация готовы (2026-03-10). GPU запуск — при следующей сессии.

### Сделано
- [x] **6.1** `set_external_weights(W)` C++: `AntennaProcessor_v1` + `release_buffers`
- [x] **6.1** `step_0_signal_only()` + `process_full_managed_w()` в `AntennaProcessorTest`
- [x] **6.1** Python binding: `set_external_weights`, `step_0_signal_only`, `process_full_managed_w`
- [x] **6.1** C++ тест `test_external_weights` добавлен в `test_strategies_pipeline.hpp`
- [x] **6.3** `plot_strategies_results.py` → 4 графика в `Results/Plots/strategies/`
- [ ] **6.2** Запуск на GPU AMD 9070 — ждёт железо

---

## ~~TASK-07~~ — ScenarioBuilder + Farrow Pipeline ✅ COMPLETED

**Статус**: ✅ 20/20 тестов пройдено (2026-03-10)
- `test_scenario_builder.py`: 17/17 PASSED
- `test_farrow_pipeline.py`: 20/20 PASSED (после фикса `test_farrow_compensate` — Runge's phenomenon для frac~0.7)

**Фикс**: integer delay round-trip вместо fractional (anti-causal нестабилен при больших frac)

---

## ⏳ Ожидает GPU AMD 9070

- `./GPUWorkLib strategies` → C++ тесты на реальном железе
- `./GPUWorkLib fm_correlator` → 4 GPU теста
- `./GPUWorkLib drvgpu` → external context тесты (ROCm + Hybrid)
- `pytest Python_test/...` GPU-тесты (все 6 ROCm-specific классов)

*Последнее обновление: 2026-03-10*
