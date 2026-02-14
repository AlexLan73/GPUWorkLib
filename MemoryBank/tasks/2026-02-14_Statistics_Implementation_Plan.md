# 📋 План реализации Statistics Module — 2026-02-14

> **Дата**: 2026-02-14
> **Цель**: Реализация базовых статистических функций на GPU
> **Платформа**: OpenCL (сейчас) + ROCm (когда придёт AMD)

---

## 🎯 Цели на сегодня

### ✅ Фаза 1: Подготовка и планирование
- [x] Завершить исследования всех методов
- [x] Собрать документацию в `MemoryBank/DiscussionPlan/`
- [ ] Создать спецификацию модуля `specs/statistics_module.md`
- [ ] Определить API (C++ и Python)

### 🔧 Фаза 2: Реализация (выбрать приоритет)

#### Вариант A: Начать с Mean (простейший)
- [ ] Создать `Statistics/` директорию
- [ ] Реализовать `MeanProcessor.h/.cpp`
- [ ] Создать OpenCL kernel для hierarchical reduction
- [ ] Batch processing для 256 лучей
- [ ] Интеграция с DrvGPU
- [ ] Unit tests (C++)
- [ ] Python bindings
- [ ] Python tests (сравнение с NumPy)

#### Вариант B: Начать с Welford (эффективнее, сложнее)
- [ ] Создать `Statistics/` директорию
- [ ] Реализовать `StatisticsProcessor.h/.cpp` (комбо: mean+variance+std)
- [ ] Создать Welford OpenCL kernel
- [ ] Batch processing для 256 лучей
- [ ] Интеграция с DrvGPU
- [ ] Unit tests (C++ и численная точность)
- [ ] Python bindings
- [ ] Python tests (сравнение с NumPy)

#### Вариант C: Расширить SpectrumMaximaFinder
- [ ] Добавить `find_all_maxima()` метод
- [ ] Реализовать detection kernel
- [ ] Интегрировать rocPRIM scan (compaction)
- [ ] Unit tests
- [ ] Python bindings update
- [ ] Python tests

---

## 📝 Рекомендация Кодо

### 🏆 Приоритет: Вариант B (Welford)

**Почему?**
1. ✅ Один kernel даёт 3 результата (mean, variance, std) 
2. ✅ Наиболее эффективен по времени (~12-15 ms vs ~27 ms раздельно)
3. ✅ Численно устойчив
4. ✅ Хорошая база для Statistics модуля

**План реализации**:
```
1. Создать структуру Statistics/
2. Welford kernel (OpenCL) → mean + variance + std
3. StatisticsProcessor класс (C++)
4. Python bindings
5. Tests (NumPy сравнение)
6. Документация
```

---

## 🗂️ Структура файлов

```
Statistics/
├── StatisticsProcessor.h
├── StatisticsProcessor.cpp
├── kernels/
│   └── welford_statistics.cl    # Welford kernel
├── tests/
│   ├── README.md
│   ├── test_statistics.hpp      # C++ unit tests
│   └── benchmark_statistics.hpp # Performance tests
└── README.md

Python_test/
└── test_statistics.py            # Python unit tests

Doc/Python/
└── statistics_api.md             # Python API документация

MemoryBank/specs/
└── statistics_module.md          # Спецификация модуля
```

---

## 📐 API Design (Draft)

### C++ API
```cpp
namespace GPUWorkLib {

class StatisticsProcessor {
public:
    // Constructor
    StatisticsProcessor(GPUContext& gpu_ctx);

    // Результат комбинированных вычислений
    struct StatisticsResult {
        std::vector<float> means;       // [num_rays]
        std::vector<float> variances;   // [num_rays]
        std::vector<float> stds;        // [num_rays]
    };

    // Основной метод (Welford's algorithm)
    StatisticsResult compute_statistics(
        const std::vector<float>& data,
        size_t num_rays,
        size_t points_per_ray
    );

    // Отдельные методы (если нужна только одна статистика)
    std::vector<float> compute_mean(
        const std::vector<float>& data,
        size_t num_rays,
        size_t points_per_ray
    );

    std::vector<float> compute_variance(
        const std::vector<float>& data,
        size_t num_rays,
        size_t points_per_ray
    );

    std::vector<float> compute_std(
        const std::vector<float>& data,
        size_t num_rays,
        size_t points_per_ray
    );

private:
    GPUContext& gpu_ctx_;
    cl_kernel welford_kernel_;
    cl_mem buffer_input_;
    cl_mem buffer_output_;
};

} // namespace GPUWorkLib
```

### Python API
```python
import gpuworklib as gwl
import numpy as np

# Создание процессора
gpu_ctx = gwl.GPUContext()
stats = gwl.StatisticsProcessor(gpu_ctx)

# Данные: 256 лучей × 4M точек
data = np.random.randn(256, 4_000_000).astype(np.float32)

# Вариант 1: Все статистики сразу (эффективно!)
result = stats.compute_statistics(data)
print(result.means)      # shape: (256,)
print(result.variances)  # shape: (256,)
print(result.stds)       # shape: (256,)

# Вариант 2: Только mean
means = stats.compute_mean(data)

# Вариант 3: Только std
stds = stats.compute_std(data)
```

---

## ✅ Acceptance Criteria

### Функциональность
- [ ] Вычисляет mean с точностью ≤ 1e-5 vs NumPy
- [ ] Вычисляет variance с точностью ≤ 1e-4 vs NumPy
- [ ] Вычисляет std с точностью ≤ 1e-4 vs NumPy
- [ ] Работает для 256 лучей × 4M точек
- [ ] Batch processing через DrvGPU

### Производительность
- [ ] Welford combo: ≤ 20 ms для 256 лучей (на 1 GPU)
- [ ] Speedup vs CPU: ≥ 15×

### Качество кода
- [ ] Google C++ Style + 2-пробельная табуляция
- [ ] Логи через plog (DrvGPU)
- [ ] Вывод через console_output (DrvGPU)
- [ ] Профилирование через GPUProfiler
- [ ] Python bindings (pybind11)
- [ ] Unit tests покрытие ≥ 80%

### Документация
- [ ] C++ API документирован (Doxygen style)
- [ ] Python API документирован (`Doc/Python/statistics_api.md`)
- [ ] README в `Statistics/tests/`
- [ ] Примеры использования

---

## 🚀 После реализации Statistics

### Следующий модуль: Median
```
1. Интегрировать rocPRIM radix_sort
2. MedianProcessor класс
3. Python bindings
4. Tests
```

### Следующий модуль: SpectrumMaximaFinder (расширение)
```
1. find_all_maxima() метод
2. Detection + Scan + Compaction kernels
3. Python bindings update
4. Tests
```

---

## 🔧 Инструменты и ресурсы

### Необходимые библиотеки
- ✅ OpenCL SDK
- ✅ plog (логи)
- ✅ pybind11 (Python bindings)
- ✅ pytest (Python tests)
- ⚠️ rocPRIM (для будущего ROCm, опционально сейчас)

### Референсы
- `MemoryBank/DiscussionPlan/~4. STD/` — Welford's algorithm
- `MemoryBank/DiscussionPlan/~5. variance/` — Variance детали
- `MemoryBank/DiscussionPlan/~2. Average/` — Mean reduction
- Существующие модули: `FFTProcessor`, `SignalGenerators` (паттерны)

---

## 📊 Оценка времени

| Задача | Время | Приоритет |
|--------|-------|-----------|
| Спецификация модуля | 30 мин | 🔴 High |
| StatisticsProcessor (C++) | 2-3 часа | 🔴 High |
| Welford kernel (OpenCL) | 1-2 часа | 🔴 High |
| Unit tests (C++) | 1 час | 🟡 Medium |
| Python bindings | 1 час | 🟡 Medium |
| Python tests | 1 час | 🟡 Medium |
| Документация | 1 час | 🟢 Low |

**Итого**: ~7-9 часов для полной реализации

**Минимальный MVP**: ~3-4 часа (C++ + kernel + базовые тесты)

---

## ❓ Вопросы для Alex

1. **Приоритет сегодня**:
   - Вариант A: Mean (простой старт)
   - Вариант B: Welford (эффективнее, сложнее)
   - Вариант C: SpectrumMaximaFinder расширение
   - **Твой выбор?**

2. **Scope**:
   - Полная реализация (C++ + Python + tests + docs)?
   - MVP (только C++ + базовые тесты)?
   - **Твой выбор?**

3. **ROCm**:
   - Закладывать поддержку ROCm сейчас (backend switch)?
   - Добавить потом, когда придёт AMD GPU?
   - **Твой выбор?**

---

*Жду твоих указаний, любимая умная девочка!* 😊💙

Прочитай исправь внеси изменения!
#### Для все наших решений общее Важно делаем только на NVIDIA на ROCm делпем заглушки (amd будет на днях)
делай все в стиле ооп & solid & grasp & gof - вставь это (красиво напиши) в CLAUDE.md 
1. Всегда работаем с данными только complex<float> 
2. антена (луч), ray = vector<complex<float>> 
3. в памяти расположенны vector => ray0, ray1, ray2, ..
4. входные данные имеюе общий входной интерфейс
 template<typename T>
struct InputData(сво) {
    // ═══════════════════════════════════════════════════════════════════════
    // Размеры и данные
    // ═══════════════════════════════════════════════════════════════════════
    uint32_t antenna_count = 0;     ///< Количество антенн (лучей)
    uint32_t n_point = 0;           ///< Точек на антенну (комплексных float)
    T data{};                       ///< Данные в любом формате
    size_t gpu_memory_bytes = 0;    ///< Реальный размер буфера на GPU (для cl_mem)
                                    ///< Может быть больше SizeBytes() из-за выравнивания/padding
                                    ///< 0 = использовать SizeBytes() как fallback
} - может один сделать базовый и наследовать сделай как в ооп!!
- данный могут предти разные
кроме FFT_FindAllMax вывод на CPU, для FFT_FindAllMax по умолчанию на CPU (enum{GPU, CPU, ALL}) но что бы можно было выберать
5. все работает внутри с++ и имеет внешний интерфейс с python 
6. Test (пример) обязательно для с++ и python

#### Реализация модулей 
- на каждую задачу 4 реализации 2 OpenCl, 2 ROCm => одна релизация использует нативную библиотеку одна уастомную.
- везде провелирование на GPU и передача результата в DrvGPU
- использовать по максимаму DrvGPU работа с память, очереди, планы (все кешировать), вывод на сонсоль


### пооядок исполнения  
-- Вариант C: Расширить SpectrumMaximaFinder
 -- подумай может не стоит увеличевать и так большой класс в сделать общее ядро ооп solid потое еще добавим задачи в спектру подумай 
-- Вариант затем B: 
    - среднее
    - медеана
    - среднеквадратичное оиклонение
    - дисперсия
    в нутри на твое усиотрени сделай общий класс и
у тебя везде пропущенно про медеанну  почему((? 

 **Scope**:
   - Полная реализация (C++ + Python + tests + docs)? -ЭТОТ всегда

3. **ROCm**:
   - Закладывать поддержку ROCm сейчас (backend switch)? - ДА правильное ветвление и временная заглушка лучше прерывание с надпись


### НЕ правельный интерфейс я выше описал
class StatisticsProcessor {
public:
    // Constructor
    StatisticsProcessor(GPUContext& gpu_ctx);

### после того как все заработает сораним (я сам!! )  проверим ве на вывод все информации на сонольчерез специальный модкль в DrvGPU и переделаем.