# SpectrumMaximaFinder API Guide

> **Версия**: 2.0 (новый API)
> **Дата**: 2026-02-12
> **Автор**: Кодо (AI Assistant)

---

## Что это такое? (для начинающих)

**SpectrumMaximaFinder** — это класс для поиска частоты сигнала на GPU.

### Простая аналогия

Представьте эквалайзер в музыкальном плеере — он показывает какие частоты громче (басы, средние, высокие). SpectrumMaximaFinder делает то же самое, но находит **САМУЮ громкую частоту** с высокой точностью.

### Что делает?

1. Принимает сигнал (массив комплексных чисел)
2. Выполняет FFT (быстрое преобразование Фурье) на GPU
3. Находит максимум спектра
4. Уточняет частоту параболической интерполяцией
5. Возвращает частоту и амплитуду

### Быстрый старт (3 шага)

```cpp
// 1. Создать finder
DrvGPU gpu(BackendType::OPENCL, 0);
gpu.Initialize();
SpectrumMaximaFinder finder(&gpu.GetBackend());

// 2. Подготовить данные
InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 5,           // Количество антенн (каналов)
    .n_point = 100000,            // Точек на антенну
    .data = my_signal_data,       // Ваши данные
    .repeat_count = 2,            // Повторений для усреднения
    .sample_rate = 1000.0f        // Частота дискретизации (Гц)
};

// 3. Обработать
auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);

// Результат
for (const auto& r : results) {
    std::cout << "Антенна " << r.antenna_id
              << ": частота = " << r.interpolated.freq_hz << " Гц\n";
}
```

---

## Детальное описание API

### 1. Входные данные: `InputData<T>`

```cpp
template<typename T>
struct InputData {
    uint32_t antenna_count;      // Количество антенн/каналов
    uint32_t n_point;            // Точек на антенну
    T data;                      // Данные (тип зависит от T)
    size_t gpu_memory_bytes;     // Для cl_mem: размер буфера

    // Параметры обработки
    uint32_t repeat_count = 2;   // Повторений (padding)
    float sample_rate = 1000.0f; // Частота дискретизации
    uint32_t search_range = 0;   // Диапазон поиска (0 = весь спектр)
    float memory_limit = 0.80f;  // Лимит GPU памяти для batch
};
```

### Поддерживаемые типы T

| Тип | Описание | Когда использовать |
|-----|----------|-------------------|
| `std::vector<std::complex<float>>` | CPU данные | Данные в оперативной памяти |
| `cl_mem` | OpenCL буфер | Данные уже на GPU (zero-copy) |
| `void*` | SVM указатель | Shared Virtual Memory |

### 2. Режимы поиска: `PeakSearchMode`

```cpp
enum class PeakSearchMode {
    ONE_PEAK,    // 1 максимум → 4 MaxValue на луч
    TWO_PEAKS    // 2 максимума → 8 MaxValue на луч (левый + правый)
};
```

### 3. Выбор драйвера: `DriverType`

```cpp
enum class DriverType {
    AUTO,    // Автовыбор (ROCm если доступен, иначе OpenCL)
    OPENCL,  // Принудительно OpenCL
    ROCM     // Принудительно ROCm (планируется)
};
```

### 4. Результат: `SpectrumResult`

```cpp
struct SpectrumResult {
    uint32_t antenna_id;           // ID антенны (0, 1, 2...)

    // Сырые данные (индекс бина)
    MaxValue center_point;         // Центральный пик
    MaxValue left_point, right_point; // Соседние точки

    // Интерполированные значения
    struct {
        float freq_hz;             // Частота в Гц
        float amplitude;           // Амплитуда
        float bin_offset;          // Смещение от центра бина
    } interpolated;
};
```

---

## Примеры использования

### Пример 1: CPU данные + инициализация DrvGPU

```cpp
#include "drv_gpu.hpp"
#include "spectrum_maxima_finder.h"

using namespace drv_gpu_lib;
using namespace antenna_fft;

int main() {
    // Инициализация GPU
    DrvGPU gpu(BackendType::OPENCL, 0);
    gpu.Initialize();

    // Создать finder
    SpectrumMaximaFinder finder(&gpu.GetBackend());

    // Генерация тестовых данных (синусоида 10 Гц)
    std::vector<std::complex<float>> data(100000);
    float freq = 10.0f;
    float sample_rate = 1000.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        float t = i / sample_rate;
        data[i] = std::complex<float>(std::sin(2 * M_PI * freq * t), 0);
    }

    // Подготовить InputData
    InputData<std::vector<std::complex<float>>> input{
        .antenna_count = 1,
        .n_point = 100000,
        .data = std::move(data),
        .repeat_count = 2,
        .sample_rate = sample_rate
    };

    // Обработка
    auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);

    // Вывод
    std::cout << "Найденная частота: " << results[0].interpolated.freq_hz << " Гц\n";
    // Ожидаем: ~10.0 Гц

    return 0;
}
```

### Пример 2: Внешний OpenCL контекст (cl_mem)

```cpp
#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include "spectrum_maxima_finder.h"

// Предположим, у вас уже есть OpenCL контекст от другой библиотеки
cl_context ext_context = ...;
cl_device_id ext_device = ...;
cl_command_queue ext_queue = ...;
cl_mem ext_buffer = ...;  // Данные уже на GPU

// Создать backend из внешнего контекста
auto backend = drv_gpu_lib::OpenCLBackend::CreateFromExternalContext(
    ext_context, ext_device, ext_queue);

// Создать finder
antenna_fft::SpectrumMaximaFinder finder(backend.get());

// InputData с cl_mem
antenna_fft::InputData<cl_mem> input{
    .antenna_count = 256,
    .n_point = 1300000,
    .data = ext_buffer,
    .gpu_memory_bytes = 256 * 1300000 * sizeof(std::complex<float>),
    .repeat_count = 2,
    .sample_rate = 1000.0f
};

// Обработка (данные НЕ копируются — zero-copy!)
auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

### Пример 3: Batch processing (большие данные)

```cpp
// При обработке больших объёмов данные автоматически разбиваются на batch'и
// Управление через memory_limit

InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 256,        // 256 антенн
    .n_point = 1300000,          // 1.3M точек каждая
    .data = huge_data,           // ~2.5 GB данных
    .repeat_count = 2,
    .sample_rate = 1000.0f,
    .memory_limit = 0.80f        // Использовать 80% GPU памяти
};

// BatchManager автоматически разобьёт на ~5 batch'ей
auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);

// Результат: 256 SpectrumResult (по одному на антенну)
```

---

## Логирование и профилирование

### Логи

Формат пути: `Logs/DRVGPU_XX/YYYY-MM-DD/HH-MM-SS.log`

```
2026-02-12 17:07:11.244 INFO  [SpectrumMaxima] >>> Starting Process(): 256 antennas x 1300000 points
2026-02-12 17:07:12.199 INFO  [SpectrumMaxima] <<< Process() completed: 256 results in 955 ms
```

### Профилирование

Формат пути: `Results/Profiler/GPU_XX_Profiler/short_name_HH-MM-SS.md|json`

Включается в `configGPU.json`:
```json
{
    "gpus": [{
        "is_prof": true
    }]
}
```

Пример отчёта:
```
| Событие      | N | Выполн. | Всего  |
|--------------|---|---------|--------|
| FFT          | 5 | 13.2 ms | 66 ms  |
| PostKernel   | 5 |  1.6 ms |  8 ms  |
| --- ИТОГО ---| 20|         | 91 ms  |
```

---

## FAQ

**Q: Почему частота не точно совпадает с ожидаемой?**
A: FFT имеет дискретное разрешение `df = sample_rate / nFFT`. Параболическая интерполяция уточняет результат до ~0.01 бина.

**Q: Сколько памяти нужно?**
A: Примерно `antenna_count × nFFT × 16 байт` (complex float × 2 буфера). При нехватке включается batch processing.

**Q: Можно ли обрабатывать данные разного размера?**
A: Да, новый API позволяет передавать параметры в каждый вызов Process().

---

*Документация создана: 2026-02-12*
