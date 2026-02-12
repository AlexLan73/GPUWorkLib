# 📋 ПЛАН РЕФАКТОРИНГА API - ФИНАЛЬНАЯ ВЕРСИЯ

> **Дата**: 2026-02-12
> **Цель**: Упростить API до 3 параметров, объединив InputData + ProcessingParams
> **Автор**: Кодо (AI Assistant)
> **Согласовано с**: Alex

---

## 🎯 ЧТО ДЕЛАЕМ:

### ✅ Подтверждено Alex:

1. ✅ Исправить путаницу в enum'ах (PeakSearchMode → DriverType)
2. ✅ Добавить `memory_limit` в InputData
3. ✅ Примеры: `Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL)`
4. ✅ ProcessingParams оставить (пока не удалять)

---

## 📝 ЗАДАЧИ:

### **ЗАДАЧА 1**: Обновить `InputData<T>` структуру
**Файл**: `modules/fft_maxima/include/interface/spectrum_input_data.hpp`

**Было**:
```cpp
template<typename T>
struct InputData {
    uint32_t antenna_count = 0;
    uint32_t n_point = 0;
    T data{};
    size_t gpu_memory_bytes = 0;
};
```

**Станет**:
```cpp
template<typename T>
struct InputData {
    // Размеры данных
    uint32_t antenna_count = 0;
    uint32_t n_point = 0;
    T data{};
    size_t gpu_memory_bytes = 0;

    // Параметры обработки (бывший ProcessingParams)
    uint32_t repeat_count = 2;
    float sample_rate = 1000.0f;
    uint32_t search_range = 0;
    float memory_limit = 0.80f;  // ← ДЛЯ BATCH!
};
```

**Что добавить**:
- `repeat_count` — множитель FFT
- `sample_rate` — частота дискретизации
- `search_range` — диапазон поиска (0 = auto)
- `memory_limit` — лимит памяти для batch (0.80 = 80%)

---

### **ЗАДАЧА 2**: Исправить путаницу enum в спеке
**Файл**: `MemoryBank/specs/api_refactoring.md`

**Исправить строки 88-92**:
```cpp
// БЫЛО (НЕПРАВИЛЬНО):
enum class PeakSearchMode { - он уже есть
    AUTO,
    OPENCL
    ROCm
};

// СТАНЕТ:
enum class DriverType {  // ← Это DriverType, а не PeakSearchMode!
    AUTO,
    OPENCL,
    ROCM
};
```

**Пояснение**: Два разных enum'а:
- `PeakSearchMode { ONE_PEAK, TWO_PEAKS }` — сколько пиков искать
- `DriverType { AUTO, OPENCL, ROCM }` — какой backend использовать

---

### **ЗАДАЧА 3**: Добавить третий параметр DriverType в Process()
**Файлы**:
- `spectrum_maxima_finder.h` (объявление, строка ~180)
- `spectrum_maxima_finder.h` (реализация шаблона, строка ~350)

**Было**:
```cpp
template<typename T>
std::vector<SpectrumResult> Process(
    const InputData<T>& input,
    const ProcessingParams& proc_params,
    PeakSearchMode mode = PeakSearchMode::ONE_PEAK,
    DriverType driver = DriverType::AUTO);  // ← AUTO
```

**Станет**:
```cpp
template<typename T>
std::vector<SpectrumResult> Process(
    const InputData<T>& input,
    PeakSearchMode mode = PeakSearchMode::ONE_PEAK,
    DriverType driver = DriverType::ROCM);  // ← ROCM по умолчанию!
```

**Изменения внутри реализации**:
```cpp
template<typename T>
std::vector<SpectrumResult> SpectrumMaximaFinder::Process(
    const InputData<T>& input,
    PeakSearchMode mode,
    DriverType driver)
{
    // 1. Подготовить параметры из InputData
    PrepareParams(input.antenna_count, input.n_point,
                  input.repeat_count, input.sample_rate,
                  input.search_range, input.memory_limit, mode);

    // 2. Диспетчеризация по типу данных
    if constexpr (is_cpu_vector_v<T>) {
        if (!initialized_) {
            Initialize();
        }
        return ProcessFromCPU(input.data);
    }
    else if constexpr (std::is_same_v<T, cl_mem>) {
        return ProcessFromGPU(input.data, input.antenna_count, input.n_point,
                              input.gpu_memory_bytes);
    }
    // ...

    // driver пока не используется (будет для ROCm)
    (void)driver;
}
```

**Важно**: PrepareParams() нужно адаптировать под новую сигнатуру!

---

### **ЗАДАЧА 4**: Обновить примеры в документации
**Файлы**:
- `spectrum_maxima_finder.h` (примеры в комментариях, строки ~161-173)
- `MemoryBank/specs/api_refactoring.md` (примеры, строки ~126-186)

#### **ПРИМЕР 1: CPU данные**

**БЫЛО**:
```cpp
// CPU данные
InputData<std::vector<std::complex<float>>> input{256, 1300000, my_data};
ProcessingParams params{.repeat_count = 2, .sample_rate = 1000.0f};
auto results = finder.Process(input, params);
```

**СТАНЕТ**:
```cpp
// CPU данные
InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 256,
    .n_point = 1300000,
    .data = my_data,
    .repeat_count = 2,
    .sample_rate = 1000.0f
};
auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

#### **ПРИМЕР 2: GPU данные (cl_mem)**

**БЫЛО**:
```cpp
// GPU данные (cl_mem)
InputData<cl_mem> gpu_input{256, 1300000, my_cl_mem};
auto results2 = finder.Process(gpu_input, params);
```

**СТАНЕТ**:
```cpp
// GPU данные (cl_mem)
InputData<cl_mem> gpu_input{
    .antenna_count = 256,
    .n_point = 1300000,
    .data = my_cl_mem,
    .gpu_memory_bytes = actual_buffer_size,  // Если известен
    .repeat_count = 2,
    .sample_rate = 1000.0f
};
auto results = finder.Process(gpu_input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

**Важно**: В примерах использовать `DriverType::OPENCL` (как просил Alex)!

---

### **ЗАДАЧА 5**: Обновить test_large_batch.hpp
**Файл**: `modules/fft_maxima/tests/test_large_batch.hpp`

**Найти старый вызов** (строка ~288):
```cpp
auto gpu_results = finder.Process(input_data);
```

**Заменить на новый API**:
```cpp
// Создать InputData с параметрами
InputData<std::vector<std::complex<float>>> input{
    .antenna_count = params.antenna_count,
    .n_point = params.n_point,
    .data = input_data,
    .repeat_count = params.repeat_count,
    .sample_rate = params.sample_rate,
    .search_range = params.search_range,
    .memory_limit = params.memory_limit
};

// Новый API
auto gpu_results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

**Примечание**: Если в тесте есть переменная `params` типа `SpectrumParams`, нужно извлечь из неё значения.

---

### **ЗАДАЧА 5a**: Обновить test_spectrum_maxima.hpp
**Файл**: `modules/fft_maxima/tests/test_spectrum_maxima.hpp`

**Описание**: Перевести тест на новый API (аналогично test_large_batch.hpp)

**Что сделать**:
1. Найти все вызовы `finder.Process(...)` со старым API
2. Создать `InputData<std::vector<std::complex<float>>>` с полями:
   - antenna_count, n_point, data
   - repeat_count, sample_rate, search_range, memory_limit
3. Заменить вызов на: `finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL)`

**Пример**:
```cpp
// БЫЛО:
auto results = finder.Process(input_data);

// СТАНЕТ:
InputData<std::vector<std::complex<float>>> input{
    .antenna_count = antenna_count,
    .n_point = n_point,
    .data = input_data,
    .repeat_count = 2,
    .sample_rate = 1000.0f,
    .search_range = 0,
    .memory_limit = 0.80f
};
auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

---

### **ЗАДАЧА 6**: Удалить старый Process(vector<complex<float>>&)
**Файлы**:
- `spectrum_maxima_finder.h` (строки 135-141) - удалить объявление
- `spectrum_maxima_finder.cpp` (строки 279-343) - удалить реализацию

**Объявление для удаления**:
```cpp
// В .h файле (строки 135-141)
/**
 * @deprecated Используйте Process(InputData<T>, ProcessingParams)
 */
std::vector<SpectrumResult> Process(
    const std::vector<std::complex<float>>& input_data);
```

**Реализация для удаления**:
```cpp
// В .cpp файле (строки 279-343)
std::vector<SpectrumResult> SpectrumMaximaFinder::Process(
    const std::vector<std::complex<float>>& input_data) {
    // ... весь метод удалить
}
```

**Причина**: Больше не нужен, есть шаблонный `Process<T>()` с новым API.

**⚠️ Важно**: Удалять ТОЛЬКО ПОСЛЕ того как test_large_batch.hpp обновлён!

---

### **ЗАДАЧА 7**: Обновить PrepareParams()
**Файл**: `spectrum_maxima_finder.cpp`

**Текущая сигнатура**:
```cpp
void PrepareParams(uint32_t antenna_count, uint32_t n_point,
                   const ProcessingParams& proc_params, PeakSearchMode mode);
```

**Два варианта решения**:

#### **Вариант A**: Изменить сигнатуру PrepareParams
```cpp
void PrepareParams(
    uint32_t antenna_count,
    uint32_t n_point,
    uint32_t repeat_count,
    float sample_rate,
    uint32_t search_range,
    float memory_limit,
    PeakSearchMode mode);
```

И вызывать так:
```cpp
PrepareParams(input.antenna_count, input.n_point,
              input.repeat_count, input.sample_rate,
              input.search_range, input.memory_limit, mode);
```

#### **Вариант B**: Создать временный ProcessingParams (рекомендую!)
```cpp
// Внутри Process<T>() создаём временный ProcessingParams
ProcessingParams params{
    .repeat_count = input.repeat_count,
    .sample_rate = input.sample_rate,
    .search_range = input.search_range,
    .memory_limit = input.memory_limit
};
PrepareParams(input.antenna_count, input.n_point, params, mode);
```

**Плюс варианта B**: Не нужно менять PrepareParams() и внутреннюю логику!

---

## 🔥 ВАЖНЫЕ ДЕТАЛИ:

### 1. **ProcessingParams НЕ УДАЛЯЕМ**
Оставляем структуру для внутреннего использования. Пользователь её не видит, но внутри кода она удобна.

### 2. **Дефолт DriverType = ROCM**
В объявлении: `DriverType driver = DriverType::ROCM`
Но в примерах показываем: `DriverType::OPENCL`

### 3. **Старый Process() удалить ПОСЛЕ**
Сначала обновить test_large_batch.hpp, потом удалить старый метод.

### 4. **memory_limit обязательно!**
Без него не будет работать batch processing для больших данных.

### 5. **Designated initializers**
Использовать `.antenna_count = 256` для ясности кода.

### 6. **gpu_memory_bytes**
Уже есть в InputData, ничего добавлять не нужно.

---

## ✅ КРИТЕРИИ ГОТОВНОСТИ:

- [ ] InputData содержит все поля: antenna_count, n_point, data, gpu_memory_bytes, repeat_count, sample_rate, search_range, memory_limit
- [ ] Process() имеет сигнатуру: Process(InputData<T>, PeakSearchMode, DriverType)
- [ ] Дефолт DriverType::ROCM установлен
- [ ] Примеры в .h обновлены на новый API
- [ ] Примеры в api_refactoring.md обновлены
- [ ] test_large_batch.hpp использует новый API
- [ ] Старый Process(vector) удалён из .h и .cpp
- [ ] Путаница с enum исправлена в api_refactoring.md (строки 88-92)
- [ ] Код компилируется без ошибок
- [ ] Тесты проходят успешно

---

## 📊 ПОРЯДОК ВЫПОЛНЕНИЯ:

Рекомендуемая последовательность для минимизации ошибок:

1. **ЗАДАЧА 1** — Обновить InputData (добавить поля)
2. **ЗАДАЧА 2** — Исправить enum в спеке
3. **ЗАДАЧА 3** — Изменить Process() (добавить DriverType, убрать ProcessingParams)
4. **ЗАДАЧА 7** — Обновить PrepareParams() или создать временный ProcessingParams
5. **ЗАДАЧА 4** — Обновить примеры в документации
6. **ЗАДАЧА 5** — Обновить test_large_batch.hpp
7. **ЗАДАЧА 5a** — Обновить test_spectrum_maxima.hpp
8. **ЗАДАЧА 6** — Удалить старый Process(vector) ← ПОСЛЕДНИМ!

---

## 🔗 СВЯЗАННЫЕ ФАЙЛЫ:

- `MemoryBank/specs/api_refactoring.md` — основная спека
- `MemoryBank/specs/api_refactoring_code_review_tasks.md` — задачи по code review
- `modules/fft_maxima/include/interface/spectrum_input_data.hpp` — структуры данных
- `modules/fft_maxima/include/spectrum_maxima_finder.h` — основной заголовок
- `modules/fft_maxima/src/spectrum_maxima_finder.cpp` — реализация
- `modules/fft_maxima/tests/test_large_batch.hpp` — тест для обновления

---

*Последнее обновление: 2026-02-12*
*Автор: Кодо (AI Assistant)*
*Согласовано с: Alex*
