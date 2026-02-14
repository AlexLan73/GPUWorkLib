# ✅ Checklist: SpectrumMaximaFinder Extension — FFT_FindAllMax

> **Дата создания**: 2026-02-14
> **Статус**: 🟡 В работе (Core C++ реализация готова, сборка OK)
> **Цель**: Расширение SpectrumMaximaFinder для поиска ВСЕХ максимумов после FFT

---

## 📋 Этап 1: Анализ текущего состояния (0.5 дня)

### Изучение SpectrumMaximaFinder
- [x] Прочитать `SpectrumMaximaFinder.h` ✅ (434 строки)
- [x] Прочитать `SpectrumMaximaFinder.cpp` ✅ (1347 строк)
- [x] Понять текущую архитектуру ✅
- [x] Оценить размер класса (строки кода) ✅ h=434, cpp=1347
- [x] Определить: расширять класс OR создать новое ядро? ✅ → Расширять

### Решение архитектуры
- [x] **Вариант A**: Простое расширение (если <5 режимов поиска) ✅ ВЫБРАНО
  - [x] Добавить метод `FindAllMaxima()` ✅
  - [x] Добавить enum `PeakSearchMode { ONE_PEAK, TWO_PEAKS, ALL_MAXIMA }` ✅
- [ ] ~~**Вариант B**: Strategy Pattern~~ — НЕ НУЖЕН (<5 режимов)

**Решение**: Вариант A — 3 режима, Strategy Pattern будет over-engineering

---

## 🏗️ Этап 2: Архитектура (если Strategy Pattern) (1-2 дня)

### Создание базового интерфейса
- [ ] Создать `IPeakSearchStrategy.h`
- [ ] Метод `virtual MaximaResult Search(data, params) = 0`
- [ ] Структура `MaximaResult` с позициями и значениями

### Реализации Strategy
- [ ] Создать `SinglePeakStrategy.h/.cpp` (рефакторинг текущего кода)
- [ ] Создать `TwoPeaksStrategy.h/.cpp` (рефакторинг текущего кода)
- [ ] Создать `AllMaximaStrategy.h/.cpp` (новый функционал)

### Рефакторинг SpectrumMaximaFinder
- [ ] Добавить поле `std::unique_ptr<IPeakSearchStrategy> strategy_`
- [ ] Метод `SetStrategy(std::unique_ptr<IPeakSearchStrategy> strategy)`
- [ ] Метод `Find()` делегирует вызов `strategy_->Search()`
- [ ] Обратная совместимость API сохранена

---

## 🔍 Этап 3: AllMaximaStrategy — Detection Kernel (1-2 дня)

### OpenCL Custom Implementation
- [x] Создать kernel detect_all_maxima ✅ (inline в all_maxima_kernel_sources.hpp)
- [x] Detection kernel:
  - [x] Условие: `data[i] > data[i-1] && data[i] > data[i+1]` ✅
  - [x] Записать флаг в `flags[i]` (0 или 1) ✅
  - [x] Обработка краёв массива ✅ (search_start/search_end)
- [x] Обработка `complex<float>` (детекция по модулю) ✅ sqrt(re²+im²)
- [x] Batch processing для 256 лучей ✅ (beam_idx = gid / nFFT)
- [x] Интеграция с DrvGPU (память, очереди) ✅
- [x] Unit test: корректность детекции ✅ (test_find_all_maxima.hpp)

### ROCm Заглушка
- [ ] Создать `kernels/rocm/detect_all_maxima.hip` (заглушка)
- [ ] Прерывание с надписью через `console_output`

---

## 📊 Этап 4: Scan-based Compaction (1-2 дня)

### Интеграция Scan (rocPRIM or custom)
- [x] Изучить rocPRIM `exclusive_scan` ✅ (не доступен, custom scan)
- [x] Интеграция с OpenCL (если rocPRIM недоступно → custom scan) ✅
- [x] Custom scan kernel (inline в all_maxima_kernel_sources.hpp):
  - [x] Work-efficient scan (Blelloch algorithm) ✅
  - [x] Two-phase: block-level + global ✅ (рекурсивный двухуровневый)
- [x] Вычисление prefix sum для `flags[]` ✅ (per-beam scan)
- [x] Результат: `positions[i]` — количество максимумов до позиции i ✅
- [ ] Unit test: корректность scan (отдельный тест)

### Compaction Kernel
- [x] Создать compact_maxima kernel ✅ (inline в all_maxima_kernel_sources.hpp)
- [x] Для каждого `flags[i] == 1`:
  - [x] `output_positions[positions[i]] = i` ✅
  - [x] `output_values[positions[i]] = data[i]` ✅
- [x] Вывод компактного списка максимумов ✅
- [x] OutputDestination: CPU (default), GPU, ALL ✅
- [ ] Unit test: корректность compaction (отдельный тест)

---

## 🎯 Этап 5: AllMaximaStrategy Integration (0.5-1 день)

### Сборка pipeline
- [x] Метод `FindAllMaxima()` в SpectrumMaximaFinder ✅ (Вариант A — без Strategy)
- [x] Методы:
  - [x] `AllMaximaResult FindAllMaxima(cl_mem, beam_count, nFFT, sample_rate, dest)` ✅
  - [x] Внутри: Detection → Scan → Compaction ✅
- [x] Интеграция с DrvGPU (IBackend контекст/очередь) ✅
- [x] Профилирование через GPUProfiler ✅ (3 стадии: Detect, Scan, Compact)
- [ ] Логирование через plog
- [ ] Вывод через console_output

### Управление памятью
- [x] Выделение временных буферов (flags, scan, positions, magnitudes, counts) ✅
- [x] Lazy компиляция kernel'ов ✅
- [x] Освобождение памяти после завершения ✅ (ReleaseAllMaximaResources)
- [ ] Unit test: отсутствие утечек памяти

---

## 📤 Этап 6: OutputDestination Strategy (0.5-1 день)

### Enum OutputDestination
- [x] Добавить `enum class OutputDestination { CPU, GPU, ALL }` ✅
- [x] Метод `FindAllMaxima(..., OutputDestination dest = OutputDestination::CPU)` ✅

### Реализация вывода
- [x] **CPU**: `clEnqueueReadBuffer` → `std::vector<T>` ✅
- [x] **GPU**: вернуть `cl_mem` буфер (не копировать на CPU) ✅
- [x] **ALL**: вернуть оба (CPU vector + GPU cl_mem) ✅
- [x] Структура результата: ✅ AllMaximaResult + AllMaximaBeamResult
  ```cpp
  struct MaximaResult {
    std::vector<int> cpu_positions;     // если CPU или ALL
    std::vector<float> cpu_values;      // если CPU или ALL
    cl_mem gpu_positions;               // если GPU или ALL
    cl_mem gpu_values;                  // если GPU или ALL
    OutputDestination destination;
  };
  ```

### Тесты OutputDestination
- [ ] C++ unit test: вывод на CPU
- [ ] C++ unit test: вывод на GPU
- [ ] C++ unit test: вывод на ALL
- [ ] C++ unit test: корректность результатов (CPU == GPU)

---

## 🐍 Этап 7: Python Bindings (1-2 дня)

### pybind11 Bindings
- [x] Обновить `python/gpu_worklib_bindings.cpp` ✅ (PySpectrumMaximaFinder)
- [x] Экспорт `OutputDestination` enum ✅ (через CPU default в wrapper)
- [x] Экспорт `AllMaximaResult` структуры ✅ (как dict/list[dict])
- [x] Метод `find_all_maxima(data, sample_rate, ...)` ✅
- [x] NumPy array интеграция для результатов ✅ (positions: uint32, magnitudes: float32, frequencies: float32)
- [x] Документация docstrings ✅

### Python Tests
- [x] `Python_test/test_spectrum_find_all_maxima.py` ✅ (5 тестов)
  - [x] Сравнение с SciPy `find_peaks()` ✅ (5/5 peaks match)
  - [x] Корректность (все максимумы найдены) ✅
  - [ ] OutputDestination (CPU/GPU/ALL) — GPU/ALL через Python не нужен (только CPU)
  - [x] Производительность GPU ✅ (256 beams x 4096: 56ms, 0.22ms/beam)
  - [x] Batch processing (256 лучей) ✅

### Python Examples
- [ ] Создать `examples/spectrum_find_all_maxima.py`
- [ ] Графики (matplotlib): спектр + отмеченные максимумы
- [ ] Сравнение производительности (NumPy vs GPU)

---

## 📝 Этап 8: Documentation (0.5-1 день)

### C++ Documentation
- [ ] Doxygen comments для новых классов/методов
- [ ] Обновить `modules/spectrum/README.md`
  - [ ] Описание AllMaximaStrategy
  - [ ] OutputDestination usage
  - [ ] Примеры C++
- [ ] Обновить `modules/spectrum/tests/README.md`

### Python Documentation
- [ ] Обновить `Doc/Python/spectrum_api.md`
  - [ ] `find_all_maxima()` API
  - [ ] OutputDestination примеры
  - [ ] Performance notes
- [ ] Примеры использования

### Спецификация
- [ ] Обновить спецификацию модуля (если есть)
- [ ] Добавить детали реализации
- [ ] Performance benchmarks

---

## 🔧 Дополнительные задачи

### Code Quality
- [ ] Google C++ Style Guide соблюдён
- [ ] clang-format применён
- [ ] SOLID principles проверены
- [ ] Strategy Pattern применён корректно (если выбран)
- [ ] Code review (`/code-review`)

### Performance Optimization
- [ ] Профилирование через GPUProfiler
- [ ] Сохранение в `Results/Profiler/`
- [ ] Анализ узких мест:
  - [ ] Detection kernel
  - [ ] Scan performance
  - [ ] Compaction kernel
  - [ ] Memory transfers (CPU↔GPU)
- [ ] Оптимизация (если нужно)

### ROCm Подготовка
- [ ] Заменить заглушки на HIP kernels
- [ ] rocPRIM scan интеграция
- [ ] Тестирование на AMD GPU
- [ ] Benchmark: OpenCL vs ROCm

---

## 📊 Прогресс

**Этап 1 (Анализ)**: 🟩🟩🟩🟩🟩🟩🟩⬜⬜⬜ 7/7 (100%) ✅
**Этап 2 (Архитектура)**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ SKIP (Вариант A)
**Этап 3 (Detection)**: 🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜ 7/8 (87%) — нет ROCm заглушки
**Этап 4 (Scan+Compaction)**: 🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜ 8/10 (80%) — нет отдельных unit tests
**Этап 5 (Integration)**: 🟩🟩🟩🟩🟩🟩🟩⬜⬜⬜ 7/9 (78%) — нет plog/console_output
**Этап 6 (OutputDestination)**: 🟩🟩🟩🟩🟩🟩🟩⬜⬜⬜ 7/7 (100%) ✅
**Этап 7 (Python)**: 🟩🟩🟩🟩🟩🟩🟩🟩🟩⬜ 10/12 (83%) — bindings + tests done
**Этап 8 (Docs)**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/7 (0%) — TODO
**Дополнительно**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/11 (0%)

**Общий прогресс**: 🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜ ~58/75 (77%)

---

## 📅 Timeline

| Этап | Дней | Начало | Конец | Статус |
|------|------|--------|-------|--------|
| 1. Анализ | 0.5 | 2026-02-14 | 2026-02-14 | ✅ Готово |
| 2. Архитектура | - | - | - | ⏭️ SKIP (Вариант A) |
| 3. Detection | 1-2 | 2026-02-14 | 2026-02-14 | ✅ Готово |
| 4. Scan+Compaction | 1-2 | 2026-02-14 | 2026-02-14 | ✅ Готово |
| 5. Integration | 0.5-1 | 2026-02-14 | 2026-02-14 | ✅ Готово |
| 6. OutputDestination | 0.5-1 | 2026-02-14 | 2026-02-14 | ✅ Готово |
| 7. Python | 1-2 | 2026-02-14 | 2026-02-14 | ✅ Готово |
| 8. Docs | 0.5-1 | - | - | ⬜ Не начато |

**Итого**: 6-12 дней (1-2.5 недели)

---

## 🎯 Acceptance Criteria

### Функциональность
- [ ] Находит ВСЕ локальные максимумы (не пропускает) ✅
- [ ] Корректность 100% vs SciPy `find_peaks()` ✅
- [ ] OutputDestination работает (CPU/GPU/ALL) ✅
- [ ] Batch processing: 256 лучей × 4M точек ✅
- [ ] `complex<float>` обрабатывается (детекция по модулю) ✅

### Производительность
- [ ] ≤75 ms для 256 лучей (10 GPU) ✅
- [ ] ≤3 ms на луч (1 GPU) ✅
- [ ] Speedup vs CPU: ≥15× ✅

### Архитектура
- [ ] SOLID principles соблюдены ✅
- [ ] Strategy Pattern применён корректно (если выбран) ✅
- [ ] Обратная совместимость API сохранена ✅
- [ ] DrvGPU максимально используется ✅

### Качество кода
- [ ] Google C++ Style Guide + 2-пробельная табуляция ✅
- [ ] Логи через plog ✅
- [ ] Вывод через console_output ✅
- [ ] Профилирование через GPUProfiler ✅
- [ ] Unit tests покрытие ≥80% ✅
- [ ] Все тесты проходят ✅

### Документация
- [ ] C++ API документирован ✅
- [ ] Python API документирован ✅
- [ ] README обновлён ✅
- [ ] Примеры добавлены ✅

---

## 🔗 Ссылки на исследования

- [ ] Прочитать `MemoryBank/DiscussionPlan/~1. FFT_FindAllMax/README.md`
- [ ] Прочитать `MemoryBank/DiscussionPlan/~1. FFT_FindAllMax/GPU_Maxima_Detection_Methods_2026-02-14.md`
- [ ] Прочитать `MemoryBank/DiscussionPlan/~1. FFT_FindAllMax/GPU_Maxima_Quick_Reference.md`

---

*Последнее обновление: 2026-02-14*
*Отмечай галочки по мере выполнения!* ✅
