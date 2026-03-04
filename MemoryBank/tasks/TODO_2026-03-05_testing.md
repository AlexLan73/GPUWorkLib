# TODO завтра (2026-03-05) — Тестирование на Linux + AMD GPU

> **Порядок**: сначала `modules/filters` → потом `modules/fm_correlator`

---

## ШАГ 1 — modules/filters (Task_20 / Task_21 / Task_22)

### Что включить в all_test.hpp перед тестом

Файл: `modules/filters/tests/all_test.hpp`

Раскомментировать вызовы (внутри `#if ENABLE_ROCM`):
```cpp
test_moving_average_rocm::run();   // Task_20: SMA/EMA/MMA/DEMA/TEMA
test_kalman_rocm::run();           // Task_21: 1D Kalman filter
test_kaufman_rocm::run();          // Task_22: KAMA (Kaufman adaptive MA)
```

### C++ тесты — что проверяют

| Файл | Тесты | Ожидание |
|------|-------|----------|
| `test_moving_average_rocm.hpp` | 6 тестов: EMA/SMA/MMA/DEMA/TEMA (GPU vs CPU) + step demo | max error < 1e-4f |
| `test_kalman_rocm.hpp` | 5 тестов: GPU vs CPU, convergence, channels, step, LFM radar demo | max error < 1e-4f |
| `test_kaufman_rocm.hpp` | 5 тестов: GPU vs CPU, trend(ER≈1), noise(ER≈0), adaptive transition, demo | max error < 1e-4f |

### Python тесты — ✅ СОЗДАНЫ (2026-03-04)

Файлы готовы, запустить после успешных C++ тестов:
```bash
python Python_test/filters/test_moving_average_rocm.py
python Python_test/filters/test_kalman_rocm.py
python Python_test/filters/test_kaufman_rocm.py
# или все разом:
pytest Python_test/filters/test_moving_average_rocm.py \
       Python_test/filters/test_kalman_rocm.py \
       Python_test/filters/test_kaufman_rocm.py -v
```

| Файл | Тестов | Что проверяет |
|------|--------|---------------|
| `test_moving_average_rocm.py` | 10 | EMA/SMA/MMA/DEMA/TEMA GPU vs numpy; impulse; 256-ch independence; step demo (CPU-only) |
| `test_kalman_rocm.py`         | 7  | GPU vs numpy; noise reduction SNR; step response; LFM radar 5-ant demo (CPU-only) |
| `test_kaufman_rocm.py`        | 8  | GPU vs numpy; trend(ER≈1); noise(ER≈0); trend→noise→trend adaptive; step demo (CPU-only) |

**⚠️ Требует Python bindings** для `MovingAverageFilterROCm`, `KalmanFilterROCm`, `KaufmanFilterROCm`
(биндинги ещё не написаны — нужно добавить в pybind11 модуль filters).

**API который ожидают тесты:**
```python
ma     = gpuworklib.MovingAverageFilterROCm(ctx)
ma.set_params("EMA", window_size)   # тип: "SMA"/"EMA"/"MMA"/"DEMA"/"TEMA"
ma.process(data)                    # 1D или 2D (ch, pts) → complex64
ma.is_ready() / ma.get_window_size() / ma.get_type()

kalman = gpuworklib.KalmanFilterROCm(ctx)
kalman.set_params(Q, R, x0=0.0, P0=25.0)
kalman.process(data)
kalman.is_ready() / kalman.get_params()  # → dict {Q, R, x0, P0}

kauf   = gpuworklib.KaufmanFilterROCm(ctx)
kauf.set_params(er_period=10, fast=2, slow=30)
kauf.process(data)
kauf.is_ready() / kauf.get_params()  # → dict {er_period, fast_period, slow_period}
```

**Демо-тесты (без GPU, запускаются всегда):**
- `test_step_response_demo` — SMA/EMA/MMA/DEMA/TEMA на ступеньке с таблицей
- `test_kalman_lfm_radar_demo` — 5 антенн LFM, FFT пик до/после Калмана
- `test_kaufman_step_demo` — KAMA: визуальное поведение на ступеньке и trend→noise→trend

---

## ШАГ 2 — modules/fm_correlator (step profiling)

### Что включить в all_test.hpp перед тестом

Файл: `modules/fm_correlator/tests/all_test.hpp`

Раскомментировать **поочерёдно**:

#### 2.1 — сначала тест step profiling (20 Record на шаг)
```cpp
fm_correlator::tests::run_step_profiling();
```
Отчёт: `Results/Profiler/fm_correlator/fm_step_profiling_*.md`
Ожидание: 2 группы событий (step1 N=20, step2 N=20)

#### 2.2 — затем avg summary (1 синтетическое событие на шаг)
```cpp
fm_correlator::tests::run_avg_summary();
```
Отчёт: `Results/Profiler/fm_correlator/fm_avg_summary_*.md`
Ожидание: compact summary `step1 avg = X ms, step2 avg = Y ms`

#### 2.3 — только после принятия 2.2
```cpp
fm_correlator::tests::run_combined();
```
Отчёт: `Results/Profiler/fm_correlator/fm_combined_*.md`
Ожидание: 3 события: all_time + step1 + step2

---

## Результаты — куда смотреть

```
Results/Profiler/
├── fm_correlator/
│   ├── fm_step_profiling_*.md    ← 2.1
│   ├── fm_avg_summary_*.md       ← 2.2
│   └── fm_combined_*.md          ← 2.3
```

---

## Если что-то не компилируется

| Модуль | Возможная проблема |
|--------|-------------------|
| MovingAverageFilterROCm | SMA: `ring[128]` — убедись что `window_size ≤ 128` |
| KalmanFilterROCm | `__frcp_rn()` — оптимизация HIP, доступна только на GPU |
| KaufmanFilterROCm | `%N` заменено на `if(++head>=N)head=0` — проверь логику |
| fm_correlator step | `GetTestBackend()` берётся из `test_fm_benchmark_rocm_all_time.hpp` |

---

*Создано: 2026-03-04 / Кодо*
