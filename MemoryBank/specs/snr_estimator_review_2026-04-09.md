# 📋 Code Review: SNR-estimator план — v5 финал

> **Объект:** [`snr_estimator_statistics_plan.md`](snr_estimator_statistics_plan.md)
> **Дата:** 2026-04-09 (v5 — все вопросы закрыты)
> **Ревьюер:** Кодо
> **Статус:** ✅ Все открытые вопросы согласованы с Alex. План готов к Python моделированию.

---

## ✅ Все согласованные решения

### 1. Размер FFT — гибкий, default 2048
- `target_N_fft` — параметр Config, **не догма 1024**
- Можно 1024/2048/4000/4096/3200/... — `PadDataOp` допадит до rocFFT размера
- Default: `kTargetNFft = 2048` (компромисс между разрешением и скоростью)
- Оба `target_N_fft=0` и `step_samples=0` → auto с target=2048
- Любой из них != 0 → другой вычисляется автоматически
- Цитата: *«нет догм привода к 1024 семпла! это может быть и 4000 а программа сама допишет нули»*

### 2. Gather через отдельный буфер (Q-2 = A)
- Отдельный промежуточный буфер `kGatherOutput[n_ant_used × N_actual × complex]`
- Thread mapping: **один поток на антенну, sequential loop по samples**
- L2 prefetcher видит линейный паттерн → эффективное чтение
- Память не проблема: 16 GB VRAM >> 80 KB – 2.8 MB результата
- Цитата: *«создадим еще один массив с данными и с ним работать — просто скопировать памяти хватает»*

### 3. `|X|²` — новый kernel `complex_to_magnitude_squared` (Q-1 + Q Голосование №2 = A)
- Добавить kernel `complex_to_magnitude_squared` в [complex_to_mag_phase_kernels_rocm.hpp](../../modules/fft_func/include/kernels/complex_to_mag_phase_kernels_rocm.hpp)
- **Два kernel в одном source**, компилируются вместе в один HSACO module
- Параметр `bool squared = false` в [MagnitudeOp::Execute](../../modules/fft_func/include/operations/magnitude_op.hpp)
- Default `squared=false` — существующий API не ломается
- Scope: **только ROCm** (main ветка), OpenCL не трогаем
- Цитата: *«Все)) пишем длинно и правильно complex_to_magnitude_squared»*

### 4. `FFTProcessorROCm::ProcessMagnitudesToGPU` — новый метод (Q-1 = A)
- Новый публичный метод в [fft_processor_rocm.hpp](../../modules/fft_func/include/fft_processor_rocm.hpp)
- Pipeline: `PadDataOp → hipfftExecC2C → MagnitudeOp(squared=true)`
- БЕЗ `ReadMagPhaseResults` — результат остаётся на GPU
- Caller владеет `gpu_out_magnitudes` (передаёт указатель)
- Переиспользует внутренний `BufferSet<4>` и LRU-2 plan cache
- Цитата: *«Вариант А + сделать kernel который оставляет данные на GPU»*

### 5. `BranchSelector` — отдельный класс (Q-3 = C)
- Facade `StatisticsProcessor` остаётся **stateless**
- `BranchSelector` хранит `prev_branch`, применяет hysteresis
- SOLID Single Responsibility
- Caller создаёт один `BranchSelector`, вызывает `Select(snr_db, thresholds)`
- `SnrEstimationResult` **БЕЗ** `BranchType` (убираем поле)
- Легко тестировать изолированно без GPU

### 6. `target_N_fft` Config — оба параметра (Q-4 = D)
- `target_N_fft = 0` и `step_samples = 0` в Config
- 0 → auto вычисление
- Default при обоих 0: `target_N_fft = 2048`
- Максимально гибко — caller может задать что хочет

### 7. Median переиспользуем существующий Op (Q-5 — признание ошибки)
- **НЕ пишем новый median kernel!**
- Переиспользуем [`MedianRadixSortOp::ExecuteFloat(1, n_ant_used)`](../../modules/statistics/include/operations/median_radix_sort_op.hpp)
- Он именно для **малых** данных (`n_point <= kHistogramThreshold = 100'000`)
- Комментарий в коде: *"Used when n_point <= kHistogramThreshold (small data — sort is faster)"*
- Работает для n_ant_used = 5..9000 — покрывает все наши сценарии
- Я была неправа, когда предложила CPU `nth_element` — ты прав, читать внимательнее

### 8. BatchManager НЕ нужен (Q-6)
- Данные помещаются целиком в 16 GB VRAM даже для сценария B (2.66 GB)
- Защита: `hipMemGetInfo` + проверка `required > free * 0.8` → `throw std::runtime_error`
- BatchManager остаётся на будущее, если придут данные >10 GB

### 9. Сценарии тестирования (Q-6 уточнение)
- **Py-Small** (Python model): 5 лучей × 1.3M complex = 50 MB
- **Сцен. A** (C++ стандарт): 2500 × 5000 = 100 MB
- **Сцен. B** (C++ большой): 256 × 1.3M = 2.66 GB
- **Сцен. C** (C++ огромный): 9000 (95×95) × 10 000 = 720 MB
- Все варианты помещаются в 16 GB VRAM

### 10. Python модель — все варианты (Q-7)
- Alex запускает через PyCharm с отладкой по шагам
- Кодо может тоже запускать через `F:\Program Files (x86)\Python314\python.exe`
- Скрипты работают и на Windows, и на Debian (venv)

### 11. Папка переименована `PyPanelAantenns → PyPanelAntennas` (Q-8)
- Исправлена опечатка (двойная `a`)
- Папка переименована через `git mv` (история сохранена)
- Обновлены все ссылки:
  - `Doc/PyPanelAantenns.md → Doc/PyPanelAntennas.md`
  - `Doc/INDEX.md` (2 места)
  - `Doc/Modules/strategies/Farrow_Pipeline.md`
  - `PyPanelAntennas/run_example.sh` (shebang комментарий)
  - `PyPanelAntennas/run_example_dock.sh`
  - `PyPanelAntennas/Examples/README.md`
  - `MemoryBank/specs/snr_estimator_statistics_plan.md`
- `PyPanelAntennas/SNR/` создана пустая подпапка для SNR модели

### 12. Имя kernel — `complex_to_magnitude_squared` (long)
- Консистентно с соседями (`complex_to_magnitude`, `complex_to_mag_phase`)
- Цитата: *«Все)) пишем длинно и правильно complex_to_magnitude_squared»*

### 13. Workflow — код сегодня, тесты в понедельник
- **Сегодня (Windows):** Python анализ + весь C++ код + Python bindings + код тестов (БЕЗ запуска)
- **Понедельник (Debian/AMD):** запуск тестов, отладка, профилирование
- **После тестов:** документация

### 14. Распределение ролей
- **Автор плана:** Кодо
- **Автор тасков:** другие помощники
- **Автор кода:** другие помощники
- **Старший ревьюер:** Кодо (проверка тасков и кода, согласованность с планом и стандартами)
- Исключение: Python моделирование может делать Кодо (исследовательская часть)

---

## 📌 Памятка C-3 — будущая оптимизация gather

> ⚠️ **Когда вернуться и ускорить `gather_decimated`**

**Триггеры для оптимизации:**
- Бенчмарк показывает что `gather_decimated` занимает > 10% общего времени SNR-estimator
- **И** частота вызовов `ComputeSnrDb` > 100 Hz

**Текущее решение (Вариант A — простое):**
- Отдельный kernel `gather_decimated_kernel` с thread mapping «один поток на антенну»
- Отдельный промежуточный буфер `kGatherOutput`
- Оценка времени: ~1-5 ms в сценарии B (зависит от n_ant_used)

**Вариант C — оптимизация через расширение `PadDataOp`:**

Вместо отдельного `gather_decimated_kernel` встроить decimation в существующий `PadDataOp`:

```cpp
// Расширить PadDataOp::Execute (modules/fft_func/include/operations/pad_data_op.hpp):
void Execute(void* input_buf, void* fft_input_buf,
             size_t beam_count, uint32_t n_point, uint32_t nFFT,
             uint32_t step_samples = 1,     // ← NEW: > 1 → decimation
             size_t   src_beam_stride = 0);  // ← NEW: для step_antennas

// Kernel pad_data внутри:
fft_input[ant*nFFT + s] = input[ant*src_beam_stride + s*step_samples];
// ... нули после N_actual
```

И `ProcessMagnitudesToGPU` принимает параметр `step_samples`:
```cpp
void ProcessMagnitudesToGPU(
    void* gpu_data,
    void* gpu_out_magnitudes,
    const FFTProcessorParams& params,
    bool squared = false,
    uint32_t step_samples = 1,           // ← NEW
    uint32_t step_antennas = 1,          // ← NEW
    size_t gpu_memory_bytes = 0,
    ROCmProfEvents* prof_events = nullptr);
```

**Что это даёт:**
- ✅ Убирает kernel `gather_decimated_kernel` полностью
- ✅ Убирает промежуточный буфер `kGatherOutput` (−80 KB ... −2.8 MB)
- ✅ Экономит +1 kernel launch (~5-10 µs)
- ✅ Убирает +1 memset для zero-pad хвоста
- ✅ Архитектурно чище — не плодим сущности
- ❌ Ломает invariant «PadDataOp работает с continuous input»

**Вариант D — Welch's method (радикально):**

Забыть про decimation вообще. Делать batched FFT по блокам фиксированного размера + усреднение power spectra:

```
Для n_samples = 1.3M, блок 1024:
  1269 блоков на антенну × 43 антенн = 54 567 FFT
  batched hipFFT обрабатывает все за ~5-10 ms
  Среднее power spectrum убирает variance оценки шума
```

- ✅ **Математически правильнее** для PSD estimation
- ✅ Нет проблемы coalescing gather вообще
- ✅ Variance оценки шума падает √(число блоков) → точнее CFAR
- ❌ Большая переделка плана
- ❌ Нет coherent gain — SNR bound'ы другие

**Решение:** Вариант A сейчас, C или D — когда подтвердится необходимость бенчмарком.

---

## 📌 Памятка — другие потенциальные оптимизации

### G-1. `kSnrPerAntenna` → `kMagnitudes` slot reuse
В `SnrEstimatorOp` `peak_cfar_kernel` может писать SNR_db прямо в `shared_buf::kMagnitudes` (переиспользуя слот), чтобы `MedianRadixSortOp::ExecuteFloat(1, n_ant_used)` мог читать напрямую без копии. Экономия ~1 µs на `hipMemcpyAsync`.

### G-2. Float32 precision для |X|²
Для очень больших `N_actual` (>16К) квадрат может переполняться float32 — рассмотреть нормализацию `/ n_point` перед peak search. Либо использовать `inv_n` параметр `MagnitudeOp`.

### G-3. CA-CFAR bias от zero-padding sinc-боковых лепестков
При `N_actual << N_fft` (сильный zero-padding) ref-окно CFAR попадает в боковые лепестки пика → P_noise_est завышен на 2-5 dB → SNR занижен. **Проверить в Python Эксп.1**. Смягчение: увеличить `guard_bins` до `max(3, nFFT/N_actual * 3)`.

### G-4. Windowing перед FFT
Применение Hann/Hamming окна перед FFT:
- Уменьшает боковые лепестки sinc (−40 dB вместо −13 dB)
- **Но** уменьшает coherent gain на 1.5-2 dB
- Компромисс — решать после Эксп.1 на Python

---

## 🗂 Что дальше

1. **Создать Python модель** в [PyPanelAntennas/SNR/](../../PyPanelAntennas/SNR/)
2. Прогнать 5 экспериментов из раздела «Часть 1 — Python-модель» плана
3. Откалибровать параметры и пороги
4. Только после этого — генерировать таски для C++ реализации
5. Кодо проверяет таски и код (ревью)
6. Понедельник — запуск тестов на Debian/AMD

---

*v5 финал — все 8 вопросов закрыты | план приведён в соответствие с решениями | готов к Python моделированию*
