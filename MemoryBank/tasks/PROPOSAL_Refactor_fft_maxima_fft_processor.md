# Предложение: Рефакторинг fft_maxima + fft_processor

**Дата:** 2026-02-15  
**Автор:** Кодо (AI Assistant)

---

## 1. Удаление неиспользуемого и дублирующего кода

### 1.1 Неиспользуемое

| Файл/Класс | Использование | Рекомендация |
|------------|---------------|--------------|
| **antenna_fft_core.cpp/h** | Только test_fft_maxima, test_fft_svm, test_external_context_fft (в main закомментированы) | ⚠️ Удалить после миграции тестов на FFTProcessor/SpectrumMaximaFinder |
| **antenna_fft_release.cpp/h** | То же | ⚠️ Удалить вместе с antenna_fft_core |
| **TestSignalGenerator** (test_signal_generator.hpp) | test_gpu_generator_integration использует CwGenerator | ⚠️ Удалить — заменён на signal_gen::CwGenerator |
| **fft_batch_adapter.hpp** | Связан с AntennaFFTCore | Удалить при удалении antenna_fft |
| **fft_result_writer.hpp** | Использует AntennaFFTResult | Проверить — если только для legacy, удалить |
| **fft_plan_cache.hpp** | Упоминает AntennaFFTProcMax | Проверить |
| **antenna_fft_params.h** | AntennaFFTCore, AntennaFFTProcMax | Удалить при удалении antenna_fft |
| **fft_kernels copy.cl** | Копия fft_kernels.cl | ⚠️ Удалить дубликат |

### 1.2 Дублирующее

| Дубликат | Где | Рекомендация |
|---------|-----|--------------|
| **FFTProfilingData** | fft_processor_types.hpp | DrvGPU уже имеет profiling_types — унифицировать |
| **ProfilingData** | spectrum_maxima_types.h | То же |
| **memory_limit, sample_rate** | InputData, ProcessingParams, FFTProcessorParams, SpectrumParams | Унифицировать через InputData<T> |
| **PreCallbackHeader** | fft_processor.hpp, spectrum_processor_opencl | Вынести в общий заголовок |

---

## 2. DrvGPU/interface — общие интерфейсы

### 2.1 Переносим в DrvGPU/interface

**Файл:** `DrvGPU/interface/input_data.hpp` (новый)

```cpp
// Универсальный шаблон входных данных для ВСЕХ модулей:
// - fft_maxima (SpectrumMaximaFinder)
// - fft_processor (FFTProcessor)
// - statistics (будущий)
// - heterodyne (будущий)
// - fractional_delay (будущий)
// - vector_math (будущий)

template<typename T>
struct InputData {
    uint32_t antenna_count = 0;   // или beam_count
    uint32_t n_point = 0;
    T data{};
    size_t gpu_memory_bytes = 0;
    uint32_t repeat_count = 2;
    float sample_rate = 1000.0f;
    uint32_t search_range = 0;
    float memory_limit = 0.80f;
    // ... методы TotalPoints(), SizeBytes(), ActualGpuMemory()
};
```

**Примечание:** `DriverType` НЕ создаём — используем `drv_gpu_lib::BackendType` из `DrvGPU/common/backend_type.hpp`.

**Файл:** `DrvGPU/interface/output_destination.hpp` (новый)

```cpp
// Куда выводить результат — CPU, GPU или оба
enum class OutputDestination {
    CPU, GPU, ALL
};
```

### 2.2 Type traits — оставить или перенести?

**Рекомендация:** Перенести в `DrvGPU/interface/input_data_traits.hpp`

```cpp
// Для статистики, гетеродина, дробной задержки — данные приходят в том же формате:
// vector<complex<float>> или cl_mem. Разный смысл (спектр, сигнал, и т.д.).
// Type traits нужны для диспетчеризации в Process/Execute.

template<typename T> struct is_cpu_vector : std::false_type {};
template<> struct is_cpu_vector<std::vector<std::complex<float>>> : std::true_type {};

template<typename T> struct is_svm_pointer : std::false_type {};
template<> struct is_svm_pointer<void*> : std::true_type {};

// Добавить для будущих модулей:
// is_gpu_buffer<T> для cl_mem
```

---

## 3. Разделение spectrum_maxima_types.h

### 3.1 Группировка по смыслу

**Файл 1:** `fft_maxima/include/types/spectrum_modes.hpp`  
Содержит: enum'ы режимов

```cpp
// Режимы поиска пиков (только fft_maxima)
enum class PeakSearchMode { ONE_PEAK, TWO_PEAKS, ALL_MAXIMA };

// OutputDestination — перенести в DrvGPU/interface (используется везде)
```

**Файл 2:** `fft_maxima/include/types/spectrum_params.hpp`  
Содержит: параметры спектра

```cpp
struct SpectrumParams {
    uint32_t antenna_count, n_point, repeat_count, ...
    PeakSearchMode peak_mode;
    float memory_limit;
    size_t max_maxima_per_beam;
    // вычисляемые: nFFT, base_fft
};
```

**Файл 3:** `fft_maxima/include/types/spectrum_result_types.hpp`  
Содержит: типы результатов

```cpp
// Базовый пик (может использоваться в статистике, гетеродине)
struct MaxValue { index, real, imag, magnitude, phase, freq_offset, refined_frequency, pad; };

// Результат для одного пика (ONE_PEAK/TWO_PEAKS)
struct SpectrumResult { antenna_id, interpolated, left_point, center_point, right_point; };

// Два пика (левый/правый диапазон)
struct CPUSpectrumResult { SpectrMax_left, SpectrMax_right; };

// FindAllMaxima — один луч
struct AllMaximaBeamResult { antenna_id, num_maxima, vector<MaxValue> maxima; };

// FindAllMaxima — все лучи
struct AllMaximaResult { beams, destination, gpu_maxima, gpu_counts, total_maxima, gpu_bytes; };
```

**Файл 4:** `fft_maxima/include/types/spectrum_profiling.hpp`  
Содержит: профилирование

```cpp
// ⚠️ ИСПРАВИТЬ: ProfilingData должен быть в DrvGPU/services/profiling_types.hpp
// Использовать OpenCLProfilingData или обёртку — НЕ дублировать свой ProfilingData
```

### 3.2 Общий индекс

**Файл:** `fft_maxima/include/types/spectrum_types.hpp`  
Содержит: `#include` всех типов + namespace

```cpp
#pragma once
// Индекс всех типов fft_maxima
#include "spectrum_modes.hpp"
#include "spectrum_params.hpp"
#include "spectrum_result_types.hpp"
#include "spectrum_profiling.hpp"
```

### 3.3 Что может использоваться в других модулях

| Тип | fft_maxima | statistics | heterodyne | vector_math |
|-----|------------|------------|------------|-------------|
| InputData<T> | ✓ | ✓ | ✓ | ✓ → DrvGPU |
| OutputDestination | ✓ | ✓ | ✓ | ✓ → DrvGPU |
| BackendType (вместо DriverType) | ✓ | ✓ | ✓ | ✓ → DrvGPU/common |
| MaxValue | ✓ | ? | ? | ? |
| ProfilingData | ✓ | ✓ | ✓ | ✓ → DrvGPU? |

**MaxValue** — структура «максимум в спектре». Для статистики (mean, std) — другой формат. Для гетеродина — может быть полезен (частота, фаза). Оставить в fft_maxima, при необходимости — вынести позже.

---

## 4. Разделение fft_processor_types.hpp

Сейчас: 1 файл, 6 типов (FFTOutputMode, FFTProcessorParams, FFTBeamResult, FFTComplexResult, FFTMagPhaseResult, FFTProfilingData).

**Рекомендация:** Группировать

**Файл 1:** `fft_processor/include/types/fft_modes.hpp`  
Содержит: `FFTOutputMode`

**Файл 2:** `fft_processor/include/types/fft_params.hpp`  
Содержит: `FFTProcessorParams`

**Файл 3:** `fft_processor/include/types/fft_results.hpp`  
Содержит: `FFTBeamResult`, `FFTComplexResult`, `FFTMagPhaseResult`, `FFTProfilingData`

**Файл 4:** `fft_processor/include/types/fft_types.hpp`  
Содержит: индекс всех типов

---

## 5. Итоговая структура файлов

```
DrvGPU/interface/
├── input_data.hpp          # InputData<T> + ProcessingParams (deprecated)
├── input_data_traits.hpp   # is_cpu_vector, is_svm_pointer
├── output_destination.hpp  # OutputDestination
# driver_type: НЕ создаём — используем BackendType из DrvGPU/common/
└── i_backend.hpp
    ...

modules/fft_maxima/include/
├── interface/
│   ├── spectrum_input_data.hpp  # УДАЛИТЬ — заменить на #include DrvGPU/interface/input_data.hpp
│   └── spectrum_maxima_types.h  # УДАЛИТЬ — заменить на types/
├── types/
│   ├── spectrum_modes.hpp
│   ├── spectrum_params.hpp
│   ├── spectrum_result_types.hpp
│   ├── spectrum_profiling.hpp
│   └── spectrum_types.hpp
└── ...
```

---

## 6. Комментарии на русском

Добавить русские комментарии во все файлы:
- fft_maxima (include + src)
- fft_processor (include + src)
- DrvGPU/interface (новые файлы)

---

## 7. Вопросы для обсуждения

1. **Удаление antenna_fft** — тесты test_fft_maxima, test_fft_svm, test_external_context_fft используют AntennaFFTProcMax. Мигрировать их на FFTProcessor + SpectrumMaximaFinder или оставить legacy-тесты?

2. **ProcessingParams** — оставить как deprecated или полностью убрать, везде использовать InputData?

3. **ProfilingData** — в fft_maxima свой, в DrvGPU есть profiling_types. Унифицировать в один?

4. **antenna_count vs beam_count** — в разных модулях разные имена. InputData — унифицировать как `beam_count` или `antenna_count`?

---

*Прочитай, обсудим, внесём правки.*

---

## ИСПРАВЛЕНИЯ (по обратной связи 2026-02-15)

### 1.1 Неиспользуемое — НЕ удаляем
Все тесты не удаляем (закомментированы), вернёмся к этому отдельно. Оставляем antenna_fft, TestSignalGenerator, fft_batch_adapter и т.д.

### 1.2 Дублирующее — согласен
Унифицировать FFTProfilingData/ProfilingData, memory_limit, sample_rate, PreCallbackHeader.

### 2. DriverType — НЕ создавать новый
DrvGPU уже имеет `BackendType` (common/backend_type.hpp): OPENCL, ROCm, OPENCLandROCm, AUTO.
**Действие:** Заменить `antenna_fft::DriverType` на `drv_gpu_lib::BackendType` в fft_maxima. Новый driver_type.hpp не создавать.

### 2.2 Type traits — перенести
Перенести в `DrvGPU/interface/input_data_traits.hpp`.

### 3. ProfilingData — ОШИБКА, исправить
Профилирование должно быть в одном месте! DrvGPU/services/profiling_types.hpp содержит:
- ProfilingDataBase, OpenCLProfilingData, ROCmProfilingData
- GPUReportInfo

**Действие:** fft_maxima и fft_processor должны использовать типы из DrvGPU. Удалить свой ProfilingData/FFTProfilingData, использовать DrvGPU::OpenCLProfilingData или обёртку для upload/fft/download времени.

### 4. fft_processor types — структура и использование

| Тип | Где определён | Где используется |
|-----|---------------|------------------|
| **FFTOutputMode** | fft_processor_types.hpp | fft_processor.cpp, fft_processor.hpp, test_fft_processor.hpp, test_fft_vs_cpu.hpp, gpu_worklib_bindings.cpp |
| **FFTProcessorParams** | fft_processor_types.hpp | fft_processor.cpp, fft_processor.hpp, test_fft_processor.hpp, test_fft_vs_cpu.hpp, test_signal_generators.hpp, gpu_worklib_bindings.cpp |
| **FFTBeamResult** | fft_processor_types.hpp | Базовый класс для FFTComplexResult, FFTMagPhaseResult |
| **FFTComplexResult** | fft_processor_types.hpp | fft_processor.cpp, fft_processor.hpp, gpu_worklib_bindings.cpp |
| **FFTMagPhaseResult** | fft_processor_types.hpp | fft_processor.cpp, fft_processor.hpp, gpu_worklib_bindings.cpp |
| **FFTProfilingData** | fft_processor_types.hpp | fft_processor.cpp, fft_processor.hpp (GetProfilingData) |

**Разделение:** fft_modes.hpp, fft_params.hpp, fft_results.hpp, fft_types.hpp — как в предложении.

### 5. Ответы на вопросы

1. **antenna_fft** — ПОКА НЕ удаляем. Пометить в MemoryBank/specs что к этому нужно вернуться.
2. **ProcessingParams** — использовать InputData, убрать дублирование.
3. **ProfilingData** — использовать DrvGPU/services/profiling_types.hpp.
4. **antenna_count vs beam_count** — унифицировать в InputData (одно имя для всех модулей).

#### Дополнение: вызов тестов через all_test.hpp — ✅ **ВЫПОЛНЕНО** (2026-02-15)

#### Рефакторинг интерфейсов — ✅ **ВЫПОЛНЕНО** (2026-02-15)
- DrvGPU/interface: input_data.hpp, output_destination.hpp, input_data_traits.hpp
- spectrum_input_data.hpp — обёртка над DrvGPU
- DriverType → BackendType (alias для совместимости)
- ProfilingData/FFTProfilingData — остаются локальными в модулях (GPUProfiler использует OpenCLProfilingData)
- spectrum_maxima_types.h → types/ (spectrum_modes, params, result_types, profiling)
- fft_processor_types.hpp → types/ (fft_modes, params, results)

В глобальном main вызывать не тесты напрямую, а файл `all_test.hpp` каждого модуля:
- `modules/fft_maxima/tests/all_test.hpp`
- `modules/fft_processor/tests/all_test.hpp`
- `modules/signal_generators/tests/all_test.hpp`
- `DrvGPU/tests/all_test.hpp`

В каждом — перечень с комментариями. Потом подчистим и удалим ненужные.

### 5. Ответы!!!

1. **antenna_fft** — ПОКА НЕ удаляем. Пометить в MemoryBank/specs что к этому нужно вернуться. - ДА
2. **ProcessingParams** — использовать InputData, убрать дублирование. - ДА
3. **ProfilingData** — использовать DrvGPU/services/profiling_types.hpp. - ДА
4. **antenna_count vs beam_count** — унифицировать в InputData (одно имя для всех модулей). Да!
