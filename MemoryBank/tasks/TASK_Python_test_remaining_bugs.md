# TASK: Python тесты — 5 нерешённых проблем

> ✅ **COMPLETED 2026-03-23** — Все 5 багов исправлены (коммит f04753f)

**Дата**: 2026-03-23
**Контекст**: После массового запуска `run_agent_tests.sh all` исправлены 17 проблем. Остались 5 сложных.
**C++ тесты**: Все проходят (12 модулей).
**Python тесты**: ✅ ALL PASS после фиксов.

---

## 1. 🔴 Segfault: `test_spectrum_maxima_finder_rocm.py` (exit 139)

**Файл теста**: `Python_test/fft_func/test_spectrum_maxima_finder_rocm.py`
**Биндинг**: `python/py_spectrum_maxima_finder_rocm.hpp`
**C++ класс**: `antenna_fft::SpectrumProcessorROCm` → `modules/fft_func/src/spectrum_processor_rocm.cpp`

**Что происходит**:
Процесс крашится с SIGSEGV при вызове GPU-методов (`process()`, `find_all_maxima()`).

**Подозрение**:
- Конструктор `SpectrumProcessorROCm` (строка 77-83) создаёт `AllMaximaPipelineROCm(stream, backend_)` — если stream или backend в невалидном состоянии → segfault
- Метод `AllMaximaFromCPU()` в биндинге вызывается без проверки инициализации

**Как воспроизвести**:
```bash
PYTHONPATH=build/debian-radeon9070/python python Python_test/fft_func/test_spectrum_maxima_finder_rocm.py
```

**Приоритет**: Критический (crash)

---

## 2. 🔴 Segfault: `test_matrix_csv_comparison.py` (exit 139)

**Файл теста**: `Python_test/vector_algebra/test_matrix_csv_comparison.py`
**Биндинг**: `python/py_vector_algebra_rocm.hpp`
**C++ класс**: `vector_algebra::CholeskyInverterROCm`

**Что происходит**:
Segfault при вызове `invert_cpu(R_inv.flatten(), n)` с CSV-данными.

**CSV файлы**:
- `modules/vector_algebra/tests/Data/R_inv_85.csv` (85x85)
- `modules/vector_algebra/tests/Data/R_85 (1).csv`
- `modules/vector_algebra/tests/Data/R_inv_341.csv` (341x341)
- `modules/vector_algebra/tests/Data/R_341 (1).csv`

**Подозрение**:
- Парсинг CSV: формат `a+bi` с запятыми — возможны edge-case ошибки парсинга
- `invert_cpu()` получает misaligned/wrong-shaped данные → OOB access
- Размер 341x341 — возможно, проблема с выделением GPU-памяти

**Как воспроизвести**:
```bash
PYTHONPATH=build/debian-radeon9070/python python Python_test/vector_algebra/test_matrix_csv_comparison.py
```

**Приоритет**: Критический (crash)

---

## 3. 🟡 LchFarrow: баг integer delay в multi-channel

**Файл теста**: `Python_test/lch_farrow/test_lch_farrow.py` (test 4: test_multi_antenna)
**Также**: `Python_test/signal_generators/test_delayed_form_signal.py` (test 3: test_multichannel_delay)
**C++ ядро**: `modules/lch_farrow/include/kernels/lch_farrow_kernels_rocm.hpp`
**C++ код**: `modules/lch_farrow/src/lch_farrow_rocm.cpp`

**Что происходит**:
В multi-channel режиме задержки кратные целым сэмплам (3.0, 6.0, 9.0) дают огромную ошибку (err=5.65), а дробные (1.5, 4.5, 7.5, 10.5) работают нормально (err<1e-3).

**Паттерн ошибки**:
```
ch0: delay=0.0us (0.0 samp) err=1.08e-04  ← OK
ch1: delay=1.5us (1.5 samp) err=1.53e-01  ← хуже, но ОК
ch2: delay=3.0us (3.0 samp) err=5.65       ← INTEGER — ОГРОМНАЯ ОШИБКА
ch3: delay=4.5us (4.5 samp) err=1.53e-01  ← дробная, ОК
ch4: delay=6.0us (6.0 samp) err=5.65       ← INTEGER — ОГРОМНАЯ ОШИБКА
ch5: delay=7.5us (7.5 samp) err=6.86e-04  ← дробная, ОК
ch6: delay=9.0us (9.0 samp) err=5.65       ← INTEGER — ОГРОМНАЯ ОШИБКА
ch7: delay=10.5us(10.5 samp) err=6.86e-04 ← дробная, ОК
```

**Ключевое**: Одиночная антенна с integer delay (test 1: 5.0 сэмплов) — РАБОТАЕТ! Баг только в multi-channel.

**Логика ядра** (lch_farrow_kernels_rocm.hpp строки ~115-117):
```cpp
int center = (int)floorf(read_pos);
float frac = read_pos - (float)center;
unsigned int row = ((unsigned int)(frac * 48.0f)) % 48u;
```

**Гипотеза**: В multi-channel mode данные для разных антенн лежат последовательно в одном буфере. Возможно, баг в вычислении `read_pos` для антенны > 0 — неправильное смещение индекса входного буфера. При integer delay `frac=0.0`, row=0, коэффициенты `[0, 1, 0, 0, 0]` (Dirac), и если `center` указывает в чужую антенну — данные берутся из неправильного места.

**Как воспроизвести**:
```bash
PYTHONPATH=build/debian-radeon9070/python python Python_test/lch_farrow/test_lch_farrow.py
# Падает test 4 (test_multi_antenna)
```

**Приоритет**: Высокий (2 теста зависят от этого)

---

## 4. 🟡 FormSignal: window test (test 3)

**Файл теста**: `Python_test/signal_generators/test_form_signal.py` (test_window)
**C++ ядро**: `modules/signal_generators/include/generators/form_signal_generator_rocm.hpp`

**Что происходит**:
Тест устанавливает `tau_base=-0.1` (сдвиг окна на -100 мс при fs=1000). Ожидается:
- Первые 100 сэмплов (t < 0) = ноль
- Сэмплы [110..500] = ненулевые (сигнал)

GPU возвращает: `zeros_first_100=100/100 OK, nonzeros_mid=0/390 FAIL` — весь сигнал нулевой!

**Формула окна (NumPy reference)**:
```python
t = np.arange(points) * dt + tau     # tau = -0.1
in_window = (t >= 0.0) & (t <= ti - dt)  # ti = 1.0
# t[100] = 0.1 * 0.001 * 100 + (-0.1) = 0.0 → in_window = True
# t[999] = 0.999 + (-0.1) = 0.899 < 0.999 → in_window = True
```

NumPy: сигнал есть в [100..999].

**Подозрение**:
- GPU-ядро FormSignalGeneratorROCm не учитывает `tau_base` при вычислении окна
- Или `tau_base` интерпретируется как параметр задержки антенны, а не как параметр окна (FormParams.tau_base)
- Формула getX в ядре: вероятно `t = n * dt` (без `+ tau`), поэтому при `tau_base=-0.1` окно `t >= 0 && t <= ti-dt` выполняется для всех сэмплов, но фазовый сдвиг другой

**Как воспроизвести**:
```bash
PYTHONPATH=build/debian-radeon9070/python python Python_test/signal_generators/test_form_signal.py
# Падает test 3 (test_window)
```

**Приоритет**: Высокий

---

## 5. 🟡 Нет биндинга: `LfmGeneratorAnalyticalDelayROCm`

**Файл теста**: `Python_test/signal_generators/test_lfm_analytical_delay.py`
**C++ класс ЕСТЬ**: `modules/signal_generators/include/generators/lfm_generator_analytical_delay_rocm.hpp`
**C++ impl ЕСТЬ**: `modules/signal_generators/src/lfm_generator_analytical_delay_rocm.cpp`
**Python биндинг**: НЕТ

**Что нужно сделать**:
Создать Python-обёртку по образцу `python/py_delayed_form_signal_rocm.hpp`.

**API ожидаемый тестом**:
```python
gen = gpuworklib.LfmAnalyticalDelay(ctx, f_start=f_start, f_end=f_end, amplitude=amplitude)
gen.set_sampling(fs=fs, length=length)
gen.set_delays([0.0])
gpu_delayed = gen.generate_gpu()
gpu_no_delay = gen.generate_cpu()  # без задержки
```

**Шаги**:
1. Прочитать C++ header — `lfm_generator_analytical_delay_rocm.hpp`
2. Создать `python/py_lfm_analytical_delay_rocm.hpp` (по образцу `py_delayed_form_signal_rocm.hpp`)
3. Зарегистрировать в `python/gpu_worklib_bindings.cpp` (include + `register_...()` в `#if ENABLE_ROCM` блок)
4. Обновить тест: `LfmAnalyticalDelay` → `LfmAnalyticalDelayROCm`, `GPUContext` → `ROCmGPUContext`

**Приоритет**: Высокий (простая задача, по аналогии с DelayedFormSignalGeneratorROCm)

---

## Общий приоритет

| # | Задача | Сложность | Зависимости |
|---|--------|-----------|-------------|
| 5 | Биндинг LfmAnalyticalDelay | Простая | Нет |
| 3 | LchFarrow integer delay | Средняя | Блокирует delayed_form_signal test3 + lch_farrow test4 |
| 4 | FormSignal window | Средняя | Нет |
| 1 | SpectrumMaxima segfault | Сложная | Нужен GDB/ASAN |
| 2 | Matrix CSV segfault | Сложная | Нужен GDB/ASAN |

**Рекомендация**: начать с #5 (самая простая), потом #3 (два теста зависят), потом #4, потом segfault'ы (#1, #2) с GDB.
