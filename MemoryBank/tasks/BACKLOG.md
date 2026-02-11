# 📋 BACKLOG — Очередь задач

> **Обновлено**: 2026-02-11
> **Всего тем**: 4 (рефакторинг)

---

## 🔥 ПРИОРИТЕТ: Критический

### ✅ ТЕМА 4: DrvGPU Optimization Analysis (ЗАВЕРШЕНА!)

**Цель**: Проанализировать код DrvGPU и предложить оптимизации БЕЗ ухудшения функциональности

**Статус**: ✅ **Completed** (2026-02-10)
**Спецификация**: [specs/drvgpu_optimization.md](../specs/drvgpu_optimization.md)
**Отчёт**: [research/drvgpu_optimization_analysis.md](../research/drvgpu_optimization_analysis.md)

**Что сделано**:
- ✅ Проанализировано 42 заголовочных файла DrvGPU
- ✅ Использовано: `sequential-thinking` (16 шагов) + `context7` (Boost.Compute)
- ✅ Найдено 3 проблемы (1 стоит исправить, 2 оставить)
- ✅ Создан отчёт с рекомендациями

**Результат**:
- ✅ Отчёт в `research/drvgpu_optimization_analysis.md`
- ✅ **P1 (OpenCLBackend)** — ВЫПОЛНЕНО! (объединение с OpenCLBackendExternal)
- 📋 **P2 (BatchManager)** — ОТЛОЖЕНО (вернуться позже)
- ❌ **P3 (Буферы)** — НЕ ДЕЛАТЬ (type safety важнее)

**Вывод**: Архитектура DrvGPU отличная! ✅

---

## 🔥 ПРИОРИТЕТ: Высокий

### 🎯 ТЕМА 3: Kernel Refactoring — OnePeak & TwoPeaks

**Цель**: Разделить поиск пиков на два отдельных кернела (OnePeak и TwoPeaks)

**Статус**: 📋 Planned
**Спецификация**: [specs/kernel_refactoring.md](../specs/kernel_refactoring.md)
**Приоритет**: 🔥 Высокий (делаем СРАЗУ после анализа DrvGPU)

**Что делаем**:
1. ✅ Переименовать: `GetPostKernelSource()` → `GetPostKernelSource_TwoPeaks()`
2. ✅ Создать: `GetPostKernelSource_OnePeak()` (ищет 1 пик → 4 MaxValue)
3. ✅ Alex напишет алгоритм OnePeak в комментариях — Кодо проверит!
4. ✅ Исправить амплитуду правой стороны в TwoPeaks
5. ✅ Динамический выбор кернела по `PeakSearchMode`
6.  По умолчанию стоит выбор на один пик.
7.  Вывод профилирования (GPUProfiler)
8.  Вывод валидации (CPU vs GPU)
9.  * примечание к обоим кернел если пик приходится с право стороны преобразовать в правильные значения.

**РЕШЕНО**: ❌ НЕ ОБЪЕДИНЯТЬ в один кернел! Два отдельных → лучше производительность!

**📋 ТАСКИ**: Будут созданы при начале работы над темой

---

### 🎯 ТЕМА 2: Batch Processing — Large Data Testing

**Цель**: Протестировать fft_maxima на 256 лучей × 1300000 точек с batch processing

**Статус**: 📋 Planned
**Спецификация**: [specs/batch_large_data.md](../specs/batch_large_data.md)
**Приоритет**: 🔥 Высокий

**Что делаем**:
1. Интегрировать `BatchManager` в `SpectrumMaximaFinder` (универсальное решение для всех модулей!)
2. Добавить параметр `memory_limit` в `SpectrumParams`
3. Реализовать `ProcessBatch()` для пакетной обработки
4. Создать `test_large_batch.hpp` (256×1300000, новый API)
5. Вывод профилирования (GPUProfiler) и валидации (CPU vs GPU)

**Анализ** (задача 2.2 от Alex):
- Проверить использование `BatchManager` в `antenna_fft_core.cpp`
- Устранить дублирование логики (если найдено) — **ОТЛОЖЕНО** (после тестов)

---

#### 📋 ДЕТАЛЬНЫЕ ТАСКИ

**ФАЗА 1: АНАЛИЗ** (исследование текущего состояния)

- **T-BATCH-001**: Анализ использования BatchManager в antenna_fft_core.cpp
  - **Цель**: Проверить, дублирует ли AntennaFFTCore логику BatchManager
  - **Действия**: Прочитать `antenna_fft_core.cpp`, найти `CalculateBatchConfig`
  - **Результат**: Документировать дублирование в `research/batch_analysis.md`
  - **Приоритет**: Средний (не блокирует основную работу)

- **T-BATCH-002**: Анализ формулы памяти для SpectrumMaximaFinder
  - **Цель**: Определить формулу `bytes_per_antenna` для FFT
  - **Действия**: Изучить `AllocateBuffers()` в spectrum_maxima_finder.cpp
  - **Формула**: `32 + n_point×8 + 2×nFFT×8 + (4 или 8)×32`
  - **Результат**: Формула для передачи в `BatchManager::CalculateOptimalBatchSize`
  - **Приоритет**: Высокий (блокирует разработку)

---

**ФАЗА 2: ИНТЕГРАЦИЯ BatchManager В SpectrumMaximaFinder**

- **T-BATCH-003**: Добавить параметр `memory_limit` в `SpectrumParams`
  - **Файл**: `modules/fft_maxima/include/spectrum_maxima_finder.h`
  - **Действия**: Добавить поле `float memory_limit = 0.7f;` (70% по умолчанию)
  - **Обоснование**: Параметр извне (как требовал Alex)
  - **Приоритет**: Высокий

- **T-BATCH-004**: Реализовать метод `CalculateBytesPerAntenna()`
  - **Файл**: `modules/fft_maxima/src/spectrum_maxima_finder.cpp`
  - **Сигнатура**: `size_t CalculateBytesPerAntenna(int n_point, int nFFT, PeakSearchMode mode)`
  - **Формула**: Из T-BATCH-002
  - **Приоритет**: Высокий

- **T-BATCH-005**: Интегрировать `BatchManager::CalculateOptimalBatchSize`
  - **Файл**: `modules/fft_maxima/src/spectrum_maxima_finder.cpp`
  - **Где**: В начале метода `Process()`
  - **Действия**: Вызвать `BatchManager::CalculateOptimalBatchSize(backend_, antenna_count, bytes_per_antenna, params.memory_limit)`
  - **Приоритет**: Высокий

- **T-BATCH-006**: Интегрировать `BatchManager::CreateBatches`
  - **Файл**: `modules/fft_maxima/src/spectrum_maxima_finder.cpp`
  - **Действия**: Вызвать `BatchManager::CreateBatches(antenna_count, batch_size, 3, true)`
  - **Параметры**: `min_tail=3` (слияние [1..3] антенн)
  - **Приоритет**: Высокий

- **T-BATCH-007**: Добавить оптимизацию `AllItemsFit()`
  - **Цель**: Пропустить batch-логику, если все данные помещаются в память
  - **Действия**: Вызвать `BatchManager::AllItemsFit()` перед созданием батчей
  - **Результат**: Если `true` → обработать всё за раз (без цикла)
  - **Приоритет**: Средний (оптимизация)

- **T-BATCH-008**: Реализовать метод `ProcessBatch()`
  - **Файл**: `modules/fft_maxima/src/spectrum_maxima_finder.cpp`
  - **Сигнатура**: `std::vector<SpectrumResult> ProcessBatch(const InputData<T>& input, size_t start_antenna, size_t count, ...)`
  - **Действия**:
    - Перевыделить буферы под `count` антенн (вместо `antenna_count`)
    - Считать срез входных данных `[start_antenna ... start_antenna+count)`
    - FFT → post-kernel → read results
    - Вернуть результаты для этого batch
  - **Приоритет**: Критический (основная логика)

- **T-BATCH-009**: Реализовать цикл обработки батчей в `Process()`
  - **Файл**: `modules/fft_maxima/src/spectrum_maxima_finder.cpp`
  - **Действия**:
    ```cpp
    std::vector<SpectrumResult> all_results;
    for (const auto& batch : batches) {
        auto batch_results = ProcessBatch(input, batch.start, batch.count, params, mode);
        all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
    }
    return all_results;
    ```
  - **Приоритет**: Критический

- **T-BATCH-010**: Добавить логирование BatchManager::PrintBatchInfo()
  - **Цель**: Диагностика конфигурации батчей
  - **Действия**: Вызвать `BatchManager::PrintBatchInfo(batches, antenna_count)` после создания батчей
  - **Приоритет**: Низкий (полезно для отладки)

---

**ФАЗА 3: ТЕСТИРОВАНИЕ**

- **T-BATCH-011**: Создать `test_large_batch.hpp`
  - **Файл**: `modules/fft_maxima/tests/test_large_batch.hpp`
  - **Структура**: По образцу `test_spectrum_maxima.hpp`
  - **Namespace**: `test_large_batch`
  - **Приоритет**: Высокий

- **T-BATCH-012**: Реализовать генерацию тестовых данных (256×1300000)
  - **Файл**: `test_large_batch.hpp`
  - **Функция**: `GenerateLargeTestData()`
  - **Данные**: 256 антенн, 1300000 точек, синусоиды с разными частотами
  - **Приоритет**: Высокий

- **T-BATCH-013**: Интеграция с новым API (InputData<vector>, ProcessingParams)
  - **Файл**: `test_large_batch.hpp`
  - **Действия**: Использовать `InputData<vector<complex<float>>>` и `ProcessingParams`
  - **Режим**: `PeakSearchMode::ONE_PEAK` (**ВЕЗДЕ ищем ОДИН пик!**)
  - **Приоритет**: Высокий

- **T-BATCH-014**: Добавить вывод профилирования (GPUProfiler)
  - **Файл**: `test_large_batch.hpp`
  - **Действия**:
    - `profiler.Start()` перед обработкой
    - `profiler.PrintReport()` после обработки
    - `profiler.Stop()`
  - **Приоритет**: Средний

- **T-BATCH-015**: Реализовать валидацию CPU vs GPU (первые 10 лучей)
  - **Файл**: `test_large_batch.hpp`
  - **Действия**:
    - Обработать первые 10 антенн на CPU (через `cpu_fft_reference.hpp`)
    - Сравнить с результатами GPU
    - Допуск: `max_error < 1e-4`
  - **Обоснование**: Полная валидация 256 лучей на CPU будет ОЧЕНЬ медленной
  - **Приоритет**: Средний

- **T-BATCH-016**: Запустить тест и проверить результаты
  - **Действия**: Запустить `test_large_batch::run()`
  - **Проверка**: Нет OOM, корректные результаты, профилирование работает
  - **Приоритет**: Критический

---

**ФАЗА 4: ДОКУМЕНТАЦИЯ**

- **T-BATCH-017**: Документировать результаты в `research/`
  - **Файл**: `MemoryBank/research/batch_large_data_results.md`
  - **Содержание**:
    - Результаты тестирования (256×1300000)
    - Конфигурация батчей (сколько батчей, размер каждого)
    - Время обработки (из GPUProfiler)
    - Использование памяти
  - **Приоритет**: Средний

- **T-BATCH-018**: Обновить спецификацию `specs/fft_module.md`
  - **Действия**: Добавить информацию о batch processing в SpectrumMaximaFinder
  - **Приоритет**: Низкий

---

**ФАЗА 5: РЕФАКТОРИНГ AntennaFFTCore** (ОТЛОЖЕНО — после тестов!)

- **T-BATCH-019**: Рефакторинг AntennaFFTCore (использовать BatchManager)
  - **Статус**: ⏳ Отложено (сначала протестировать SpectrumMaximaFinder)
  - **Цель**: Заменить собственную batch-логику на `BatchManager`
  - **Действия**:
    - Заменить `CalculateBatchConfig` на `BatchManager::CalculateOptimalBatchSize`
    - Сделать `memory_usage_limit` конфигурируемым (по умолчанию 0.7)
    - Убрать `batch_size_ratio`
    - Использовать `CreateBatches` с `merge_small_tail=true`
  - **Приоритет**: Низкий (после проверки работы batch в SpectrumMaximaFinder)

---

**ЗАВИСИМОСТИ МЕЖДУ ТАСКАМИ**:

```
T-BATCH-002 (формула памяти)
    ↓
T-BATCH-003 (memory_limit в SpectrumParams)
    ↓
T-BATCH-004 (CalculateBytesPerAntenna)
    ↓
T-BATCH-005 (CalculateOptimalBatchSize)
    ↓
T-BATCH-006 (CreateBatches)
    ↓
T-BATCH-007 (AllItemsFit — опционально)
    ↓
T-BATCH-008 (ProcessBatch — КРИТИЧНО!)
    ↓
T-BATCH-009 (цикл обработки)
    ↓
T-BATCH-010 (PrintBatchInfo — опционально)
    ↓
T-BATCH-011..016 (тесты)
    ↓
T-BATCH-017..018 (документация)

T-BATCH-001 (анализ AntennaFFTCore) — параллельно, не блокирует
T-BATCH-019 (рефакторинг) — ПОСЛЕ всех тестов
```

---

**КРИТЕРИИ ГОТОВНОСТИ ТЕМЫ**:

- ✅ `SpectrumMaximaFinder` использует `BatchManager` (универсальное решение)
- ✅ `memory_limit` настраиваемый параметр в `SpectrumParams`
- ✅ `test_large_batch.hpp` создан и работает (256×1300000)
- ✅ Batch processing работает автоматически (нет OOM)
- ✅ Профилирование выводится корректно (GPUProfiler)
- ✅ Валидация проходит (первые 10 лучей, CPU vs GPU)
- ✅ Документация обновлена (`research/`, `specs/`)
- ✅ **ВЕЗДЕ ищем ОДИН пик** (`PeakSearchMode::ONE_PEAK`)
- ⏳ Рефакторинг `AntennaFFTCore` — ОТЛОЖЕН (следующий этап)

---

**ВСЕГО ТАСОК**: 19 (из них 16 активных, 1 отложена на потом)

---

### 🎯 ТЕМА 1: API Refactoring — SpectrumMaximaFinder

**Цель**: Создать универсальный API для приёма данных в любом формате (vector, cl_mem, SVM, и др.)

**Статус**: 📋 Planned
**Спецификация**: [specs/api_refactoring.md](../specs/api_refactoring.md)
**Приоритет**: 🔥 Критический (архитектурный рефакторинг!)

**Что делаем**:
1. ✅ Изменить конструктор: `SpectrumMaximaFinder(IBackend* backend)` — БЕЗ params!
2. ✅ Создать `InputData<T>` структуру (в `interface/`)
3. ✅ Создать `ProcessingParams` структуру (в `interface/`)
4. ✅ Создать `PeakSearchMode` enum (в `interface/`)
5. ✅ Реализовать шаблонный `Process<T>(input, params, mode)`
добавил `Process<T>(input, params, mode, typedriver)`  - typedriver работа с разными драверами + разная реализация под OpenCl & ROCm 
6. ✅ Поддержка типов: `vector`, `cl_mem`, `void*` (SVM), `GPUBuffer*`, `ExternalCLBufferAdapter*`
7. ✅ Динамическое создание FFT плана в `Process()`
8. ✅ Динамический выбор кернела (`ONE_PEAK` / `TWO_PEAKS`)

**ВАЖНО**:
- ❌ Полная замена API (breaking change)
- ✅ Обновить **ВСЕ** примеры на новый API
- ✅ По умолчанию: `PeakSearchMode::ONE_PEAK`

**СВЯЗИ**:
- Требует ТЕМА 3 (кернелы OnePeak и TwoPeaks)

**📋 ТАСКИ**: Будут созданы при начале работы над темой

---

## 📝 КАК РАБОТАТЬ С ТЕМАМИ

### 🎯 Начало работы над темой:

1. **Выбрать тему** (по приоритету или по решению Alex)
2. **Прочитать спецификацию** в `specs/`
3. **Создать таски** для темы (разбить на конкретные шаги)
4. **Добавить таски** в этот файл или в `IN_PROGRESS.md`
5. **Начать работу** над первым таском

### ✅ Завершение темы:

1. **Все таски** выполнены
2. **Тесты** проходят
3. **Документация** обновлена
4. **Переместить тему** в `COMPLETED.md`

---

## 🔗 СТАРЫЕ ЗАДАЧИ (до рефакторинга)

Эти задачи **НЕ АКТУАЛЬНЫ** — они будут решены в рамках новых тем!

| ID | Модуль | Задача | Статус |
|----|--------|--------|--------|
| T-001 | FFT | Добавить оконные функции (Hann, Hamming, Blackman) | ⚪ Отложено |
| T-002 | FFT | Реализовать Real-to-Complex FFT (R2C) | ⚪ Отложено |
| T-003 | DrvGPU | Добавить ROCm backend (rocFFT) | ⚪ Отложено |
| T-004 | FFT | Batch FFT (несколько сигналов) | ✅ Решается в ТЕМА 2 |
| T-005 | Filters | Создать FIR-фильтр на GPU | ⚪ Отложено |
| T-006 | Filters | Создать IIR-фильтр (биквадратные секции) | ⚪ Отложено |
| T-007 | Statistics | Базовые статистики (mean, std, variance) | ⚪ Отложено |
| T-008 | Heterodyne | Базовый NCO на GPU | ⚪ Отложено |
| T-009 | Heterodyne | MixDown / MixUp функции | ⚪ Отложено |
| T-010 | SignalSynth | Генератор синусоиды | ⚪ Отложено |
| T-011 | SignalSynth | Генератор шума (white, pink) | ⚪ Отложено |
| T-012 | SignalSynth | Chirp генератор | ⚪ Отложено |

---

## 📊 ПОРЯДОК РАБОТЫ

**Согласовано с Alex (2026-02-10)**:

```
1️⃣ ТЕМА 4 (DrvGPU Optimization) — ПЕРВАЯ! Анализ без кода
    ↓
2️⃣ ТЕМА 3 (Kernel Refactoring) — OnePeak & TwoPeaks кернелы
    ↓
3️⃣ ТЕМА 2 (Batch Large Data) — Тестирование больших данных
    ↓
4️⃣ ТЕМА 1 (API Refactoring) — Универсальный API (самая большая)
```

**Подход**: Последовательно (Вариант А) — по одной теме, обсуждаем, делаем, завершаем

---

*Последнее обновление: 2026-02-11*
*Автор: Кодо (AI Assistant)*
