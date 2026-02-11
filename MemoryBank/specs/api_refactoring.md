# 🔄 API Refactoring — SpectrumMaximaFinder

> **Тема**: ТЕМА 1 (изначально была External OpenCL Integration)
> **Приоритет**: 🔥 Критический
> **Статус**: 📋 Planned
> **Дата создания**: 2026-02-10
> **Автор**: Кодо (AI Assistant)

---

## 🎯 ЦЕЛЬ

Создать **универсальный API** для `SpectrumMaximaFinder`, который принимает данные в **любом формате**:
- `vector<complex<float>>` (CPU)
- `cl_mem` (OpenCL буфер от заказчика)
- `void*` (SVM)
- `GPUBuffer<T>*` (наша обёртка)
- И другие форматы

**КЛЮЧЕВАЯ ИДЕЯ**: Параметры (antenna_count, n_point) приходят **ВМЕСТЕ С ДАННЫМИ**, а не в конструкторе!

---

## 🔥 ПРОБЛЕМА

**Текущая архитектура** (неправильная):
```cpp
// Конструктор — принимает params
SpectrumMaximaFinder(const SpectrumParams& params, IBackend* backend);

// Process — принимает только vector
std::vector<SpectrumResult> Process(const vector<complex<float>>& data);
```

**Недостатки**:
- ❌ Принимает только `vector<complex<float>>`
- ❌ Параметры фиксированы в конструкторе
- ❌ Нет поддержки `cl_mem` от заказчика
- ❌ Нет выбора алгоритма (1 или 2 вершины)
- ❌ Нельзя обработать данные с разными параметрами

---

## ✅ РЕШЕНИЕ

### 🏗️ Новая архитектура

```cpp
// ═══════════════════════════════════════════════════════════════════
// 1. КОНСТРУКТОР — ТОЛЬКО backend (БЕЗ params!)
// ═══════════════════════════════════════════════════════════════════
SpectrumMaximaFinder(IBackend* backend);

// ═══════════════════════════════════════════════════════════════════
// 2. СТРУКТУРА ВХОДНЫХ ДАННЫХ (шаблонная)
// Хранится в: modules/fft_maxima/include/interface/
// ═══════════════════════════════════════════════════════════════════
template<typename T>
struct InputData {
    int antenna_count;      // Количество антенн (лучей)
    int n_point;           // Точек на антенну (комплексных float)
    T data;                // Данные в любом формате!
};

// ═══════════════════════════════════════════════════════════════════
// 3. ПАРАМЕТРЫ ОБРАБОТКИ
// Хранится в: modules/fft_maxima/include/interface/
// ═══════════════════════════════════════════════════════════════════
struct ProcessingParams {
    int repeat_count = 4;        // Для nFFT = NextPowerOf2(n_point * repeat)
    float sample_rate = 1000.0f; // Частота дискретизации (Hz)
    int search_range = 0;        // Диапазон поиска (0 = auto = nFFT/2)
};

// ═══════════════════════════════════════════════════════════════════
// 4. РЕЖИМ ПОИСКА ПИКОВ (управление алгоритмом!)
// Хранится в: modules/fft_maxima/include/interface/
// ═══════════════════════════════════════════════════════════════════
enum class PeakSearchMode {
    ONE_PEAK,   // 1 вершина → GetPostKernelSource_OnePeak() → 4 MaxValue
    TWO_PEAKS   // 2 вершины → GetPostKernelSource_TwoPeaks() → 8 MaxValue
};

// ═══════════════════════════════════════════════════════════════════
// 5. УНИВЕРСАЛЬНЫЙ МЕТОД Process
// ═══════════════════════════════════════════════════════════════════
template<typename T>
std::vector<SpectrumResult> Process(
    const InputData<T>& input,
    const ProcessingParams& params,
    PeakSearchMode mode = PeakSearchMode::ONE_PEAK  // ← ПО УМОЛЧАНИЮ 1 ВЕРШИНА!
-- доработай - добавил  typedriver)`  - typedriver работа с разными драверами + разная реализация под OpenCl & ROCm 
  поставь по умолчанию ROCm
    
);
```

---

## 📦 ПОДДЕРЖИВАЕМЫЕ ТИПЫ ДАННЫХ (T)

| Тип | Описание | Местоположение |
|-----|----------|----------------|
| `std::vector<std::complex<float>>` | Данные на CPU | Стандартная библиотека |
| `cl_mem` | OpenCL буфер (от заказчика!) | OpenCL API |
| `void*` | SVM pointer | OpenCL 2.0+ |
| `GPUBuffer<std::complex<float>>*` | Наша RAII обёртка | `DrvGPU/memory/gpu_buffer.hpp` |
| `ExternalCLBufferAdapter<std::complex<float>>*` | Адаптер внешних буферов | `DrvGPU/memory/external_cl_buffer_adapter.hpp` |

---

## 📝 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

### ПРИМЕР 1: Данные на CPU (vector)

```cpp
#include "spectrum_maxima_finder.h"

SpectrumMaximaFinder finder(&backend);

// Генерация данных на CPU
std::vector<std::complex<float>> cpu_data = GenerateTestData(5, 100000);

// Создаём InputData
InputData<std::vector<std::complex<float>>> input;
input.antenna_count = 5;
input.n_point = 100000;
input.data = std::move(cpu_data);

// Параметры обработки
ProcessingParams params;
params.repeat_count = 4;
params.sample_rate = 1000.0f;

// Обработка с TWO_PEAKS
auto results = finder.Process(input, params, PeakSearchMode::TWO_PEAKS);
```

### ПРИМЕР 2: Данные на GPU (cl_mem от заказчика)

```cpp
SpectrumMaximaFinder finder(&backend);

// Заказчик передал cl_mem буфер
cl_mem external_buffer = /* ... буфер от заказчика ... */;

// Создаём InputData
InputData<cl_mem> input;
input.antenna_count = 256;
input.n_point = 1300000;
input.data = external_buffer;  // ← БЕЗ КОПИРОВАНИЯ!

ProcessingParams params;
params.repeat_count = 4;
params.sample_rate = 1000.0f;

// Обработка с ONE_PEAK (по умолчанию)
auto results = finder.Process(input, params);
```

### ПРИМЕР 3: SVM

```cpp
void* svm_ptr = /* ... SVM указатель ... */;

InputData<void*> input;
input.antenna_count = 10;
input.n_point = 50000;
input.data = svm_ptr;

auto results = finder.Process(input, params);
```

---

## 🔧 РЕАЛИЗАЦИЯ

### Шаг 1: Создать интерфейсные структуры

**Файл**: `modules/fft_maxima/include/interface/spectrum_input_data.hpp`

```cpp
#pragma once
#include <complex>
#include <vector>

namespace antenna_fft {

template<typename T>
struct InputData {
    int antenna_count;
    int n_point;
    T data;
};

struct ProcessingParams {
    int repeat_count = 4;
    float sample_rate = 1000.0f;
    int search_range = 0;
};

enum class PeakSearchMode {
    ONE_PEAK,
    TWO_PEAKS
};

} // namespace antenna_fft
```

### Шаг 2: Изменить конструктор

**До**:
```cpp
SpectrumMaximaFinder(const SpectrumParams& params, IBackend* backend);
```

**После**:
```cpp
SpectrumMaximaFinder(IBackend* backend);
```

### Шаг 3: Реализовать шаблонный Process()

```cpp
template<typename T>
std::vector<SpectrumResult> Process(
    const InputData<T>& input,
    const ProcessingParams& params,
    PeakSearchMode mode = PeakSearchMode::ONE_PEAK)
{
    // if constexpr для определения типа T
    if constexpr (std::is_same_v<T, std::vector<std::complex<float>>>) {
        return ProcessFromCPU(input, params, mode);
    }
    else if constexpr (std::is_same_v<T, cl_mem>) {
        return ProcessFromCLMem(input, params, mode);
    }
    else if constexpr (std::is_same_v<T, void*>) {
        return ProcessFromSVM(input, params, mode);
    }
    // ... другие типы
}
```

### Шаг 4: Динамическое создание FFT плана

```cpp
void PrepareFFTPlan(int n_point, int antenna_count, int repeat_count) {
    int nFFT = NextPowerOf2(n_point * repeat_count);

    // Если план уже создан с этими параметрами — переиспользуем
    if (fft_plan_ && current_nFFT_ == nFFT) {
        return;
    }

    // Создать новый план
    CreateFFTPlan(nFFT, antenna_count);
    current_nFFT_ = nFFT;
}
```

### Шаг 5: Динамический выбор кернела

```cpp
void SelectPostKernel(PeakSearchMode mode) {
    if (mode == PeakSearchMode::ONE_PEAK) {
        post_kernel_source_ = kernels::GetPostKernelSource_OnePeak();
        output_size_per_beam_ = 4;  // 4 MaxValue
    } else {
        post_kernel_source_ = kernels::GetPostKernelSource_TwoPeaks();
        output_size_per_beam_ = 8;  // 8 MaxValue
    }

    // Компиляция кернела
    CompilePostKernel();
}
```

---

## 📋 ЗАДАЧИ (создаются при начале работы)

При начале работы над ТЕМОЙ 1 будут созданы таски:

- `T-XXX`: Создать `spectrum_input_data.hpp` (интерфейсные структуры)
- `T-XXX`: Изменить конструктор `SpectrumMaximaFinder`
- `T-XXX`: Реализовать шаблонный `Process<T>()`
- `T-XXX`: Реализовать `ProcessFromCPU()`
- `T-XXX`: Реализовать `ProcessFromCLMem()`
- `T-XXX`: Реализовать `ProcessFromSVM()`
- `T-XXX`: Динамическое создание FFT плана
- `T-XXX`: Динамический выбор кернела по `mode`
- `T-XXX`: Обновить `test_spectrum_maxima.hpp` на новый API
- `T-XXX`: Создать `test_new_api_all_types.hpp` (все типы данных)
- `T-XXX`: Обновить документацию

---

## 🔗 СВЯЗИ С ДРУГИМИ ТЕМАМИ

- **ТЕМА 3** (Кернелы): Требует `GetPostKernelSource_OnePeak()` и `GetPostKernelSource_TwoPeaks()`
- **ТЕМА 2** (Batch processing): Использует новый API с `InputData<cl_mem>`

---

## ✅ КРИТЕРИИ ГОТОВНОСТИ

- ✅ Конструктор принимает только `IBackend*`
- ✅ `InputData<T>` поддерживает: `vector`, `cl_mem`, `void*`, `GPUBuffer*`, `ExternalCLBufferAdapter*`
- ✅ `Process()` работает со ВСЕМИ типами данных
- ✅ Динамический выбор кернела (`ONE_PEAK` / `TWO_PEAKS`)
- ✅ Все старые примеры обновлены на новый API
- ✅ Тесты проходят
- ✅ Документация обновлена

---

## 📚 СПРАВОЧНАЯ ИНФОРМАЦИЯ

### Про "clBuffer" (вопрос от Alex)

**Q**: Есть ли `clBuffer` в стандарте OpenCL?
**A**: ❌ НЕТ! В OpenCL есть только `cl_mem` (handle на memory object).

Когда заказчик говорит "clBuffer", он имеет в виду:
1. `cl_mem` созданный через `clCreateBuffer()`
2. Разговорное название для "OpenCL buffer"
3. Возможно wrapper-класс в своём коде

**Решение**: Поддерживаем `cl_mem` напрямую → покрывает ВСЕ случаи!

---

*Последнее обновление: 2026-02-10*
*Автор: Кодо (AI Assistant)*
