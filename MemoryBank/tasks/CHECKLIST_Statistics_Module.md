# ✅ Checklist: Statistics Module Implementation

> **Дата создания**: 2026-02-14
> **Статус**: В работе
> **Цель**: Реализация Statistics модуля с галочками для отслеживания прогресса

---
# ВСЕ КЕРНЕЛ ДОЛЖНЫ ВЫЗЫВАТЬСЯ ПАПАЛЛЕЛЬНО ПО ВСЕМ ЛУЧАМ (АНТЕННАМ)

## 📋 Этап 1: Инфраструктура (1-2 дня)

### InputData Interface
- [ ] Создать `IInputData<T>` базовый интерфейс
- [ ] Реализовать `InputDataImpl<T>` с SoA/AoS layout
- [ ] Добавить enum `MemoryLayout { SOA, AOS }`
- [ ] Создать Factory method для удобного создания
- [ ] Unit tests для InputData
- [ ] Документация Doxygen

### Statistics Operation Interface
- [ ] Создать `IStatisticsOperation<T>` базовый интерфейс
- [ ] Добавить enum `StatOpType { MEAN, MEDIAN, STD, VARIANCE }`
- [ ] Добавить enum `DriverType { OPENCL, ROCM }`
- [ ] Добавить enum `ImplType { NATIVE, CUSTOM }`
- [ ] Добавить enum `OutputDestination { CPU, GPU, ALL }`
- [ ] Документация Doxygen

### Структура каталогов
- [ ] Создать `modules/statistics/`
- [ ] Создать `modules/statistics/include/`
- [ ] Создать `modules/statistics/src/`
- [ ] Создать `modules/statistics/kernels/opencl/`
- [ ] Создать `modules/statistics/kernels/rocm/` (заглушки)
- [ ] Создать `modules/statistics/tests/`
- [ ] Создать `modules/statistics/tests/cpp/`
- [ ] Создать `modules/statistics/tests/python/`

---

## 📊 Этап 2: Mean Operation (3-4 дня)

### OpenCL Custom Implementation
- [ ] Создать `MeanOperationOpenCLCustom.h`
- [ ] Создать `MeanOperationOpenCLCustom.cpp`
- [ ] Написать kernel `kernels/opencl/mean_reduction.cl`
  - [ ] Two-level hierarchical reduction
  - [ ] Shared memory optimization
  - [ ] Coalesced memory access
- [ ] Интеграция с DrvGPU (память, очереди, профилирование)
- [ ] Batch processing для 256 лучей
- [ ] Обработка `complex<float>` данных
- [ ] Вывод на CPU (по умолчанию)

### OpenCL Native Implementation
- [ ] Создать `MeanOperationOpenCLNative.h`
- [ ] Создать `MeanOperationOpenCLNative.cpp`
- [ ] Попробовать использовать готовую библиотеку (если есть)
- [ ] Fallback на custom если native недоступно

### ROCm Заглушки
- [ ] Создать `MeanOperationROCmCustom.h` (заглушка)
- [ ] Создать `MeanOperationROCmNative.h` (заглушка)
- [ ] Прерывание с надписью через `console_output`

### Тесты Mean
- [ ] C++ unit test: корректность vs CPU reference
- [ ] C++ unit test: performance benchmark
- [ ] C++ unit test: численная точность (≤1e-5)
- [ ] C++ unit test: batch processing (256 лучей)
- [ ] Python test: сравнение с `np.mean()`
- [ ] Python test: производительность GPU vs CPU
- [ ] Документация тестов `tests/README.md`

---

## 📈 Этап 3: Median Operation (3-5 дней)

### OpenCL Custom Implementation — Radix Sort
- [ ] Создать `MedianOperationOpenCLCustomRadix.h`
- [ ] Создать `MedianOperationOpenCLCustomRadix.cpp`
- [ ] Написать kernel `kernels/opencl/radix_sort.cl`
  - [ ] Radix sort для каждого луча
  - [ ] Выбор среднего элемента `sorted[N/2]`
- [ ] Интеграция с DrvGPU
- [ ] Batch processing для 256 лучей
- [ ] Обработка `complex<float>` (по модулю или по real/imag)
- [ ] Вывод на CPU

### OpenCL Custom Implementation — RadiK (будущее, оптимальный)
- [ ] Создать `MedianOperationOpenCLCustomRadiK.h`
- [ ] Создать `MedianOperationOpenCLCustomRadiK.cpp`
- [ ] Портировать RadiK algorithm с CUDA
- [ ] Написать kernel `kernels/opencl/radik_select.cl`
- [ ] Batch optimization
- [ ] Benchmark: сравнение с Radix Sort

### OpenCL Native Implementation
- [ ] Создать `MedianOperationOpenCLNative.h`
- [ ] Создать `MedianOperationOpenCLNative.cpp`
- [ ] Использовать готовую sort библиотеку (если есть)
- [ ] Fallback на custom Radix Sort

### ROCm Заглушки
- [ ] Создать `MedianOperationROCmCustom.h` (заглушка)
- [ ] Создать `MedianOperationROCmNative.h` (заглушка)
- [ ] Прерывание с надписью через `console_output`

### Тесты Median
- [ ] C++ unit test: корректность vs CPU reference
- [ ] C++ unit test: performance benchmark (Radix Sort)
- [ ] C++ unit test: performance benchmark (RadiK, если реализован)
- [ ] C++ unit test: сравнение Radix Sort vs RadiK
- [ ] C++ unit test: численная точность
- [ ] C++ unit test: batch processing (256 лучей)
- [ ] Python test: сравнение с `np.median()`
- [ ] Python test: производительность GPU vs CPU
- [ ] Документация тестов

### Ссылки на исследования Median
- [ ] Прочитать `MemoryBank/DiscussionPlan/~3. Median/README.md`
- [ ] Прочитать `MemoryBank/DiscussionPlan/~3. Median/2026-02-14_GPU_Median_Algorithms_Research.md`
- [ ] Прочитать `MemoryBank/DiscussionPlan/~3. Median/GPU_Median_Quick_Reference.md`

---

## 📉 Этап 4: Std Operation (2-3 дня)

### OpenCL Custom Implementation — Welford
- [ ] Создать `StdOperationOpenCLCustom.h`
- [ ] Создать `StdOperationOpenCLCustom.cpp`
- [ ] Написать kernel `kernels/opencl/welford_std.cl`
  - [ ] Welford's algorithm (numerically stable)
  - [ ] Single-pass computation
  - [ ] Вычисление mean + variance + std одновременно
- [ ] Интеграция с DrvGPU
- [ ] Batch processing для 256 лучей
- [ ] Обработка `complex<float>` (по модулю)
- [ ] Вывод на CPU

### OpenCL Native Implementation
- [ ] Создать `StdOperationOpenCLNative.h`
- [ ] Создать `StdOperationOpenCLNative.cpp`
- [ ] Использовать готовую библиотеку (если есть)
- [ ] Fallback на custom Welford

### ROCm Заглушки
- [ ] Создать `StdOperationROCmCustom.h` (заглушка)
- [ ] Создать `StdOperationROCmNative.h` (заглушка)
- [ ] Прерывание с надписью через `console_output`

### Тесты Std
- [ ] C++ unit test: корректность vs CPU reference
- [ ] C++ unit test: performance benchmark
- [ ] C++ unit test: численная устойчивость (Welford vs naive)
- [ ] C++ unit test: численная точность (≤1e-4)
- [ ] C++ unit test: batch processing (256 лучей)
- [ ] Python test: сравнение с `np.std()`
- [ ] Python test: производительность GPU vs CPU
- [ ] Документация тестов

### Ссылки на исследования Std
- [ ] Прочитать `MemoryBank/DiscussionPlan/~4. STD/README.md`
- [ ] Прочитать `MemoryBank/DiscussionPlan/~4. STD/GPU_STD_Research.md`

---

## 📊 Этап 5: Variance Operation (1-2 дня)

### OpenCL Custom Implementation — Welford
- [ ] Создать `VarianceOperationOpenCLCustom.h`
- [ ] Создать `VarianceOperationOpenCLCustom.cpp`
- [ ] Использовать тот же kernel `welford_std.cl` (без sqrt)
- [ ] Интеграция с DrvGPU
- [ ] Batch processing для 256 лучей
- [ ] Обработка `complex<float>` (по модулю)
- [ ] Вывод на CPU

### OpenCL Native Implementation
- [ ] Создать `VarianceOperationOpenCLNative.h`
- [ ] Создать `VarianceOperationOpenCLNative.cpp`
- [ ] Использовать готовую библиотеку (если есть)
- [ ] Fallback на custom Welford

### ROCm Заглушки
- [ ] Создать `VarianceOperationROCmCustom.h` (заглушка)
- [ ] Создать `VarianceOperationROCmNative.h` (заглушка)
- [ ] Прерывание с надписью через `console_output`

### Тесты Variance
- [ ] C++ unit test: корректность vs CPU reference
- [ ] C++ unit test: performance benchmark
- [ ] C++ unit test: численная устойчивость
- [ ] C++ unit test: численная точность (≤1e-4)
- [ ] C++ unit test: batch processing (256 лучей)
- [ ] Python test: сравнение с `np.var()`
- [ ] Python test: производительность GPU vs CPU
- [ ] Документация тестов

### Ссылки на исследования Variance
- [ ] Прочитать `MemoryBank/DiscussionPlan/~5. variance/README.md`
- [ ] Прочитать `MemoryBank/DiscussionPlan/~5. variance/GPU_Variance_Research.md`
- [ ] Прочитать `MemoryBank/DiscussionPlan/~5. variance/Quick_Reference.md`

---

## 🏭 Этап 6: Factory + Facade (1-2 дня)

### Factory Pattern
- [ ] Создать `StatisticsOperationFactory.h`
- [ ] Создать `StatisticsOperationFactory.cpp`
- [ ] Методы `CreateMean()`, `CreateMedian()`, `CreateStd()`, `CreateVariance()`
- [ ] Параметры: DriverType, ImplType
- [ ] Unit tests для Factory

### Facade Pattern
- [ ] Создать `StatisticsModule.h`
- [ ] Создать `StatisticsModule.cpp`
- [ ] Простой API для пользователя:
  - [ ] `ComputeMean(input)`
  - [ ] `ComputeMedian(input)`
  - [ ] `ComputeStd(input)`
  - [ ] `ComputeVariance(input)`
  - [ ] `ComputeAll(input)` — возвращает все 4 результата
- [ ] Интеграция с DrvGPU
- [ ] Кеширование планов и буферов
- [ ] Логирование через plog
- [ ] Профилирование через GPUProfiler
- [ ] Вывод через console_output

### Integration Tests
- [ ] C++ integration test: Mean + Std + Variance combo (Welford)
- [ ] C++ integration test: Все 4 операции последовательно
- [ ] C++ integration test: OutputDestination (CPU/GPU/ALL)
- [ ] C++ integration test: Batch processing (256 лучей)
- [ ] Performance comparison: раздельно vs combo

---

## 🐍 Этап 7: Python Bindings (2-3 дня)

### pybind11 Bindings
- [ ] Создать `python_bindings/statistics_module.cpp`
- [ ] Экспорт `InputData<T>`
- [ ] Экспорт `StatisticsModule`
- [ ] Экспорт enums (DriverType, ImplType, OutputDestination)
- [ ] Методы:
  - [ ] `compute_mean(data)`
  - [ ] `compute_median(data)`
  - [ ] `compute_std(data)`
  - [ ] `compute_variance(data)`
  - [ ] `compute_all(data)`
- [ ] NumPy array интеграция
- [ ] Документация docstrings

### Python Tests
- [ ] `Python_test/test_statistics_mean.py`
  - [ ] Сравнение с `np.mean()`
  - [ ] Точность ≤1e-5
  - [ ] Производительность GPU vs CPU
  - [ ] Batch processing (256 лучей)
- [ ] `Python_test/test_statistics_median.py`
  - [ ] Сравнение с `np.median()`
  - [ ] Точность
  - [ ] Производительность GPU vs CPU
  - [ ] Batch processing
- [ ] `Python_test/test_statistics_std.py`
  - [ ] Сравнение с `np.std()`
  - [ ] Точность ≤1e-4
  - [ ] Производительность GPU vs CPU
- [ ] `Python_test/test_statistics_variance.py`
  - [ ] Сравнение с `np.var()`
  - [ ] Точность ≤1e-4
  - [ ] Производительность GPU vs CPU
- [ ] `Python_test/test_statistics_combo.py`
  - [ ] Тест `compute_all()`
  - [ ] Сравнение с NumPy

### Python Examples
- [ ] Создать `examples/statistics_basic.py`
- [ ] Создать `examples/statistics_batch.py`
- [ ] Создать `examples/statistics_comparison.py` (NumPy vs GPU)
- [ ] Графики производительности (matplotlib)

---

## 📝 Этап 8: Documentation (1-2 дня)

### C++ Documentation
- [ ] Doxygen comments для всех классов
- [ ] Doxygen comments для всех методов
- [ ] Создать `modules/statistics/README.md`
  - [ ] Описание модуля
  - [ ] Архитектура (классы, паттерны)
  - [ ] Примеры использования C++
  - [ ] Performance benchmarks
- [ ] Создать `modules/statistics/tests/README.md`
  - [ ] Описание тестов
  - [ ] Как запускать
  - [ ] Acceptance criteria

### Python Documentation
- [ ] Создать `Doc/Python/statistics_api.md`
  - [ ] API reference
  - [ ] Примеры использования
  - [ ] Performance notes
  - [ ] NumPy comparison
- [ ] Обновить `Doc/Python/README.md` (общий индекс)

### Спецификация
- [ ] Обновить `MemoryBank/specs/statistics_module.md`
  - [ ] Добавить детали реализации
  - [ ] Добавить performance metrics
  - [ ] Добавить примеры кода

---

## 🔧 Дополнительные задачи

### Code Quality
- [ ] Google C++ Style Guide соблюдён (2-пробельная табуляция)
- [ ] clang-format применён ко всем файлам
- [ ] SOLID principles проверены
- [ ] GRASP principles проверены
- [ ] GoF patterns применены корректно
- [ ] Code review (можно использовать `/code-review`)

### Performance Optimization
- [ ] Профилирование через GPUProfiler
- [ ] Сохранение результатов в `Results/Profiler/`
- [ ] Анализ узких мест
- [ ] Оптимизация kernels (если нужно)
- [ ] Memory access patterns проверены

### ROCm Подготовка (когда придёт AMD GPU)
- [ ] Заменить заглушки на реальные реализации
- [ ] HIP kernels (портация с OpenCL)
- [ ] rocPRIM интеграция
- [ ] Тестирование на AMD GPU
- [ ] Benchmark: OpenCL vs ROCm

---

## 📊 Прогресс

**Этап 1 (Инфраструктура)**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/18 (0%)
**Этап 2 (Mean)**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/18 (0%)
**Этап 3 (Median)**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/24 (0%)
**Этап 4 (Std)**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/18 (0%)
**Этап 5 (Variance)**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/18 (0%)
**Этап 6 (Factory+Facade)**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/15 (0%)
**Этап 7 (Python)**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/18 (0%)
**Этап 8 (Docs)**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/8 (0%)
**Дополнительно**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/12 (0%)

**Общий прогресс**: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0/149 (0%)

---

## 📅 Timeline

| Этап | Дней | Начало | Конец | Статус |
|------|------|--------|-------|--------|
| 1. Инфраструктура | 1-2 | - | - | ⬜ Не начато |
| 2. Mean | 3-4 | - | - | ⬜ Не начато |
| 3. Median | 3-5 | - | - | ⬜ Не начато |
| 4. Std | 2-3 | - | - | ⬜ Не начато |
| 5. Variance | 1-2 | - | - | ⬜ Не начато |
| 6. Factory+Facade | 1-2 | - | - | ⬜ Не начато |
| 7. Python | 2-3 | - | - | ⬜ Не начато |
| 8. Docs | 1-2 | - | - | ⬜ Не начато |

**Итого**: 15-25 дней (3-5 недель)

---

## 🎯 Acceptance Criteria (финальная проверка)

### Функциональность
- [ ] Mean: точность ≤1e-5 vs NumPy ✅
- [ ] Median: точность 100% (exact median) ✅
- [ ] Std: точность ≤1e-4 vs NumPy ✅
- [ ] Variance: точность ≤1e-4 vs NumPy ✅
- [ ] Batch processing: 256 лучей × 4M точек ✅
- [ ] OutputDestination работает (CPU/GPU/ALL) ✅
- [ ] `complex<float>` обрабатывается корректно ✅

### Производительность
- [ ] Mean: ≤5 ms (256 лучей, 1 GPU) ✅
- [ ] Median: ≤60 ms Radix Sort OR ≤30 ms RadiK (256 лучей, 1 GPU) ✅
- [ ] Std: ≤15 ms (256 лучей, 1 GPU) ✅
- [ ] Variance: ≤15 ms (256 лучей, 1 GPU) ✅
- [ ] Speedup vs CPU: ≥15× для каждой операции ✅

### Архитектура
- [ ] SOLID principles соблюдены ✅
- [ ] GRASP principles соблюдены ✅
- [ ] GoF patterns применены (Strategy, Factory, Facade, Template Method) ✅
- [ ] InputData<T> с ООП наследованием ✅
- [ ] 4 реализации (2 OpenCL + 2 ROCm заглушки) ✅
- [ ] DrvGPU максимально используется (память, очереди, логи, профилирование) ✅

### Качество кода
- [ ] Google C++ Style Guide + 2-пробельная табуляция ✅
- [ ] Логи через plog (DrvGPU) ✅
- [ ] Вывод через console_output (DrvGPU) ✅
- [ ] Профилирование через GPUProfiler ✅
- [ ] Unit tests покрытие ≥80% ✅
- [ ] Все тесты проходят (C++ и Python) ✅

### Документация
- [ ] C++ API документирован (Doxygen) ✅
- [ ] Python API документирован (`Doc/Python/statistics_api.md`) ✅
- [ ] README в `modules/statistics/` ✅
- [ ] README в `modules/statistics/tests/` ✅
- [ ] Примеры использования (C++ и Python) ✅

---

*Последнее обновление: 2026-02-14*
*Отмечай галочки по мере выполнения!* ✅
