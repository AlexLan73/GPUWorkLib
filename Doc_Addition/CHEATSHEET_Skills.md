# 🎯 Шпаргалка по Skills — feature-dev и code-review

> **Дата**: 2026-02-14
> **Автор**: Кодо (AI Assistant)

---

## 📚 Общая информация

**Skills** — это специализированные агенты Claude Code для выполнения сложных задач:
- `feature-dev` — разработка новых фич с глубоким анализом кодовой базы
- `code-review` — ревью кода и pull request'ов

---

## 🚀 feature-dev — Разработка фич

### Когда использовать
✅ Разработка новых модулей (Statistics, Heterodyne, Filters)
✅ Добавление сложного функционала (новые алгоритмы ЦОС)
✅ Рефакторинг с изменением архитектуры
✅ Интеграция с существующей кодовой базой

❌ Простые исправления (опечатки, мелкие баги)
❌ Одиночные функции без контекста

### Основные возможности
1. **Анализ кодовой базы** — изучает существующий код, паттерны, архитектуру
2. **Планирование архитектуры** — предлагает структуру новой фичи
3. **Пошаговая реализация** — разбивает задачу на этапы
4. **Интеграция** — учитывает существующие модули (DrvGPU, BatchManager и т.д.)
5. **Тестирование** — помогает написать тесты

### Как запустить

#### Вариант 1: Через команду в чате
```
/feature-dev Добавить модуль Statistics с функциями mean, std, variance на GPU
```

#### Вариант 2: Через описание задачи
```
Используя feature-dev, создай модуль Heterodyne с NCO и MixDown/MixUp
```

### Workflow feature-dev

```
┌─────────────────────────────────────────────────┐
│ 1. Анализ кодовой базы                          │
│    - Изучает существующие модули                │
│    - Находит паттерны и соглашения              │
│    - Понимает архитектуру DrvGPU                │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 2. Планирование архитектуры                     │
│    - Предлагает структуру файлов                │
│    - Определяет интерфейсы                      │
│    - Планирует интеграцию с DrvGPU              │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 3. Пошаговая реализация                         │
│    - Создаёт классы и заголовки                 │
│    - Пишет OpenCL kernels                       │
│    - Добавляет Python bindings (pybind11)       │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 4. Тестирование и документация                  │
│    - Создаёт C++ тесты (*.hpp)                  │
│    - Добавляет Python тесты                     │
│    - Обновляет документацию                     │
└─────────────────────────────────────────────────┘
```

### Примеры использования

#### Пример 1: Разработка модуля Statistics
```
/feature-dev Создай модуль Statistics со следующими функциями:
- GPU-вычисление mean (среднее)
- GPU-вычисление std (стандартное отклонение)
- GPU-вычисление variance (дисперсия)

Требования:
- Использовать DrvGPU контекст
- BatchManager для больших массивов
- Python биндинги (pybind11)
- Тесты на сравнение с NumPy
```

#### Пример 2: Добавление гетеродина
```
/feature-dev Реализуй модуль Heterodyne:
- NCO (Numerically Controlled Oscillator)
- MixDown (понижение частоты)
- MixUp (повышение частоты)

Архитектура:
- OpenCL kernels для всех операций
- Python API с примерами
- Интеграция с FFTProcessor
```

#### Пример 3: Рефакторинг существующего кода
```
/feature-dev Рефакторинг SpectrumMaximaFinder:
- Разделить на Strategy Pattern (OpenCL/ROCm)
- Улучшить производительность поиска максимума
- Добавить batch-обработку через DrvGPU
```

### Что делает feature-dev автоматически

✅ Анализирует `DrvGPU` и использует его API
✅ Следует Google C++ Style + 2-пробельная табуляция
✅ Создаёт отдельные файлы для классов
✅ Добавляет `plog` логирование через DrvGPU
✅ Использует `console_output` для вывода
✅ Документирует Python API в `Doc/Python/`
✅ Создаёт тесты в `{module}/tests/*.hpp`

### Советы по работе с feature-dev

1. **Чёткое ТЗ** — опиши что нужно, feature-dev сам разберётся как
2. **Указывай ограничения** — например, "только OpenCL, ROCm потом"
3. **Укажи интеграцию** — "использовать BatchManager из DrvGPU"
4. **Тестирование** — попроси сразу добавить тесты
5. **Не мешай процессу** — feature-dev работает пошагово, дай ему завершить

---

## 🔍 code-review — Ревью кода

### Когда использовать
✅ Проверка pull request перед merge
✅ Ревью нового модуля (Statistics, Heterodyne)
✅ Поиск багов и утечек памяти
✅ Проверка соответствия стандартам (Google C++ Style)
✅ Анализ безопасности (OpenCL буферы, race conditions)

❌ Простой синтаксический анализ (для этого есть clang-tidy)

### Основные возможности
1. **Анализ логики** — находит логические ошибки
2. **Безопасность** — проверяет утечки, race conditions
3. **Производительность** — выявляет узкие места
4. **Стиль кода** — проверка Google C++ Style
5. **Архитектурные замечания** — соответствие паттернам проекта

### Как запустить

#### Вариант 1: Ревью pull request
```
/code-review PR #123
```

#### Вариант 2: Ревью файлов
```
/code-review Statistics/StatisticsProcessor.h Statistics/StatisticsProcessor.cpp
```

#### Вариант 3: Ревью модуля
```
Сделай code-review модуля Heterodyne, проверь:
- Корректность работы с DrvGPU
- Утечки памяти OpenCL буферов
- Соответствие стилю кода
```

### Что проверяет code-review

#### 🐛 Баги и логические ошибки
```cpp
// ❌ Плохо — утечка памяти
cl_mem buffer = clCreateBuffer(...);
// забыли clReleaseMemObject(buffer)

// ❌ Плохо — race condition
for (int i = 0; i < 10; i++) {
  launch_kernel_async(i);  // без синхронизации
}

// ❌ Плохо — выход за границы
float* data = new float[1024];
data[1024] = 0.0f;  // index out of bounds
```

#### 🔒 Безопасность и ресурсы
```cpp
// ❌ Плохо — не проверяется статус
clEnqueueNDRangeKernel(...);  // может вернуть ошибку

// ✅ Хорошо — проверка ошибок
cl_int err = clEnqueueNDRangeKernel(...);
if (err != CL_SUCCESS) {
  throw std::runtime_error("Kernel execution failed");
}
```

#### 🎯 Производительность
```cpp
// ❌ Плохо — копирование в цикле
for (int i = 0; i < N; i++) {
  clEnqueueWriteBuffer(queue, buffer, CL_TRUE, ...);
}

// ✅ Хорошо — одна большая передача
clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, N * sizeof(float), data, ...);
```

#### 📐 Стиль кода (Google C++ Style)
```cpp
// ❌ Плохо — snake_case для класса
class gpu_processor { };

// ✅ Хорошо — CamelCase для класса
class GPUProcessor { };

// ❌ Плохо — CamelCase для метода
void ProcessData();

// ✅ Хорошо — snake_case для метода
void process_data();
```

#### 🏗️ Архитектурные проблемы
```cpp
// ❌ Плохо — создаём свой OpenCL контекст
cl_context ctx = clCreateContext(...);

// ✅ Хорошо — используем DrvGPU
auto& gpu_ctx = GPUContext::get_instance();

// ❌ Плохо — прямой cout
std::cout << "Result: " << value << std::endl;

// ✅ Хорошо — через DrvGPU console_output
gpu_ctx.console_output("Result: " + std::to_string(value));
```

### Формат отчёта code-review

```
🔍 Code Review Report
=====================

📁 Files reviewed:
- Statistics/StatisticsProcessor.h
- Statistics/StatisticsProcessor.cpp
- Statistics/kernels/statistics.cl

⚠️ Critical Issues (3)
-----------------------
1. [Memory Leak] Line 145: cl_mem buffer not released
2. [Race Condition] Line 230: Missing barrier in kernel
3. [Nullptr Dereference] Line 89: data pointer not checked

🟡 Warnings (5)
----------------
1. [Style] Line 12: Use snake_case for method names
2. [Performance] Line 167: Unnecessary host-device copy
3. [Architecture] Line 203: Should use DrvGPU console_output
4. [Documentation] Missing Python API documentation
5. [Testing] No unit tests for variance calculation

✅ Suggestions (2)
------------------
1. Consider using BatchManager for large arrays
2. Add profiling through GPUProfiler

📊 Summary
-----------
- Total issues: 10
- Critical: 3 (must fix before merge)
- Warnings: 5 (should fix)
- Suggestions: 2 (nice to have)

🎯 Recommendation: DO NOT MERGE until critical issues are resolved
```

### Примеры использования

#### Пример 1: Ревью перед коммитом
```
/code-review Heterodyne/HeterodyneProcessor.cpp

Проверь:
- Утечки памяти OpenCL буферов
- Корректность работы с DrvGPU
- Стиль кода (Google C++ Style)
```

#### Пример 2: Полное ревью модуля
```
/code-review Statistics/

Проведи полное ревью модуля Statistics:
- Архитектура (использование DrvGPU)
- Безопасность (OpenCL ресурсы)
- Производительность (оптимизация kernels)
- Тестирование (покрытие тестами)
- Документация (Python API)
```

#### Пример 3: Фокус на производительности
```
/code-review FFTProcessor.cpp --focus=performance

Найди узкие места:
- Лишние копирования данных
- Неоптимальные kernel launches
- Возможности использования BatchManager
```

### Советы по работе с code-review

1. **Конкретизируй фокус** — что именно проверить (баги/стиль/производительность)
2. **Укажи контекст** — например, "это для multi-GPU системы"
3. **Проси примеры** — code-review может показать как исправить
4. **Приоритизируй** — попроси сначала критичные проблемы
5. **Итеративно** — исправил критичное → повторное ревью

---

## 🔄 Комбинированный workflow

### Разработка новой фичи + ревью

```bash
# Шаг 1: Разработка через feature-dev
/feature-dev Создай модуль Statistics с mean, std, variance

# Шаг 2: Проверка через code-review
/code-review Statistics/

# Шаг 3: Исправление критичных проблем
# (code-review укажет что исправить)

# Шаг 4: Повторное ревью
/code-review Statistics/ --quick

# Шаг 5: Коммит
git add Statistics/
git commit -m "Add Statistics module with GPU mean/std/variance"
```

### Рефакторинг существующего кода

```bash
# Шаг 1: Анализ текущего состояния
/code-review SpectrumMaximaFinder/ --focus=architecture

# Шаг 2: Рефакторинг через feature-dev
/feature-dev Рефакторинг SpectrumMaximaFinder:
- Strategy Pattern для OpenCL/ROCm
- Улучшение производительности

# Шаг 3: Проверка после рефакторинга
/code-review SpectrumMaximaFinder/ --compare-with=main

# Шаг 4: Merge
git merge refactor-spectrum-finder
```

---

## 🎯 Когда использовать feature-dev vs code-review

| Задача | feature-dev | code-review |
|--------|-------------|-------------|
| Новый модуль | ✅ Да | ➡️ После реализации |
| Добавление функции | ✅ Да (если сложная) | ➡️ После реализации |
| Рефакторинг | ✅ Да | ➡️ До и после |
| Исправление бага | ❌ Нет (ручками) | ✅ Да (найти причину) |
| Проверка PR | ❌ Нет | ✅ Да |
| Оптимизация | 🔶 Может помочь | ✅ Да (найти узкие места) |
| Документация | 🔶 Может помочь | ✅ Да (проверить полноту) |

---

## 🚨 Важные замечания для GPUWorkLib

### feature-dev должен учитывать:
- ✅ Использовать только DrvGPU для работы с GPU
- ✅ Логи только через `plog` (DrvGPU)
- ✅ Вывод только через `console_output` (DrvGPU)
- ✅ Профилирование только через `GPUProfiler` (DrvGPU)
- ✅ Batch обработка через `BatchManager` (DrvGPU)
- ✅ Отдельные файлы для классов (исключение: интерфейсы)
- ✅ Python API документация в `Doc/Python/{module}_api.md`
- ✅ C++ тесты в `{module}/tests/*.hpp`
- ✅ Python тесты в `Python_test/test_*.py`

### code-review должен проверять:
- ✅ **НЕТ** прямого создания OpenCL контекста (только через DrvGPU)
- ✅ **НЕТ** `std::cout` / `printf` (только `console_output`)
- ✅ **НЕТ** ручного профилирования (только `GPUProfiler`)
- ✅ Все OpenCL буферы освобождаются (нет утечек)
- ✅ Google C++ Style + 2-пробельная табуляция
- ✅ Python биндинги для значимых API
- ✅ Тесты покрывают основной функционал

---

## 📝 Шаблоны команд

### feature-dev

```bash
# Базовый шаблон
/feature-dev [Описание задачи]

# С требованиями
/feature-dev [Задача]
Требования:
- [требование 1]
- [требование 2]

# С архитектурными указаниями
/feature-dev [Задача]
Архитектура:
- [класс 1]: [назначение]
- [класс 2]: [назначение]
Интеграция:
- Использовать [существующий модуль]
```

### code-review

```bash
# Базовый шаблон
/code-review [файлы или PR]

# С фокусом
/code-review [файлы] --focus=[bugs|performance|style|security]

# С критериями
/code-review [файлы]
Проверить:
- [критерий 1]
- [критерий 2]

# Сравнительное ревью
/code-review [файлы] --compare-with=[branch]
```

---

## 🎓 Итоговые советы

### Для feature-dev:
1. Давай чёткое ТЗ, feature-dev сам спланирует архитектуру
2. Указывай интеграцию с DrvGPU явно
3. Проси тесты и документацию сразу
4. Не мешай процессу, feature-dev работает пошагово

### Для code-review:
1. Используй перед каждым коммитом/PR
2. Фокусируйся на конкретных аспектах (баги/стиль/производительность)
3. Исправляй критичные проблемы в первую очередь
4. Делай повторное ревью после исправлений

### Общее:
- feature-dev → реализация
- code-review → проверка качества
- Используй их **вместе** для лучшего результата

---

*Последнее обновление: 2026-02-14*
*Автор: Кодо (AI Assistant)*
