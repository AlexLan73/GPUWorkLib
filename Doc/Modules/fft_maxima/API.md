# FFT Maxima — API Reference

**Namespace**: `antenna_fft`

---

## Типы данных

### PeakSearchMode

```cpp
enum class PeakSearchMode {
    ONE_PEAK,   // Один пик: сравнение левого и правого, выбор большего → 4 MaxValue
    TWO_PEAKS   // Два пика: независимо левый и правый → 8 MaxValue
};
```

### DriverType

```cpp
enum class DriverType {
    AUTO,       // Автоматический выбор
    OPENCL,     // OpenCL (clFFT)
    ROCM        // ROCm/HIP (hipFFT) — planned
};
```

### InputData\<T\>

Универсальная структура входных данных. Поддерживаемые типы `T`:
- `std::vector<std::complex<float>>` — CPU данные
- `cl_mem` — OpenCL буфер
- `void*` — SVM pointer

```cpp
template<typename T>
struct InputData {
    uint32_t antenna_count = 0;     // Количество антенн
    uint32_t n_point = 0;           // Точек на антенну
    T data{};                       // Данные
    size_t gpu_memory_bytes = 0;    // Размер GPU буфера (для cl_mem)

    uint32_t repeat_count = 2;      // Множитель FFT: nFFT = nextPow2(n_point) * repeat_count
    float sample_rate = 1000.0f;    // Частота дискретизации (Hz)
    uint32_t search_range = 0;      // Диапазон поиска (0 = auto = nFFT/4)
    float memory_limit = 0.80f;     // Доля GPU памяти для batch (0.0-1.0)

    size_t TotalPoints() const;
    size_t SizeBytes() const;
    size_t ActualGpuMemory() const;
};
```

### MaxValue

GPU-совместимая структура (32 bytes, aligned).

```cpp
struct MaxValue {
    uint32_t index;             // Индекс в FFT спектре
    float real;                 // Re компонента
    float imag;                 // Im компонента
    float magnitude;            // |magnitude| = sqrt(re^2 + im^2)
    float phase;                // Фаза в градусах
    float freq_offset;          // Параболическая поправка [-0.5, 0.5]
    float refined_frequency;    // Уточнённая частота (Hz)
    uint32_t pad;               // Padding (32 bytes total)
};
```

### SpectrumResult

Результат для одной антенны (один пик).

```cpp
struct SpectrumResult {
    uint32_t antenna_id;        // Номер антенны
    MaxValue interpolated;      // Параболическая интерполяция
    MaxValue left_point;        // Левая точка (index-1)
    MaxValue center_point;      // Центральная точка (максимум)
    MaxValue right_point;       // Правая точка (index+1)
};
```

### CPUSpectrumResult

Результат для одного луча: два пика (левый и правый диапазон спектра).

```cpp
struct CPUSpectrumResult {
    SpectrumResult SpectrMax_left;   // Максимум [0, half_range]
    SpectrumResult SpectrMax_right;  // Максимум [nFFT-half_range, nFFT]
};
```

### SpectrumParams

```cpp
struct SpectrumParams {
    uint32_t antenna_count = 5;         // Количество антенн (1-256)
    uint32_t n_point = 1000;            // Точек на антенну
    uint32_t repeat_count = 2;          // Множитель FFT
    float sample_rate = 1000.0f;        // Частота дискретизации (Hz)
    uint32_t search_range = 0;          // 0 = auto = nFFT/4
    PeakSearchMode peak_mode = PeakSearchMode::ONE_PEAK;
    float memory_limit = 0.80f;         // Доля GPU памяти

    // Вычисляемые
    uint32_t nFFT = 0;                  // nextPow2(n_point) * repeat_count
    uint32_t base_fft = 0;              // nextPow2(n_point)
};
```

### ProfilingData

```cpp
struct ProfilingData {
    double upload_time_ms = 0.0;
    double fft_time_ms = 0.0;
    double post_kernel_time_ms = 0.0;
    double download_time_ms = 0.0;
    double total_time_ms = 0.0;
};
```

---

## FFTPlanCache

**Файл**: `include/fft_plan_cache.hpp`

Кеш clFFT планов для повторного использования. Устраняет пересоздание планов при batch processing.

```cpp
class FFTPlanCache {
public:
    FFTPlanCache(cl_context context, cl_command_queue queue);

    clfftPlanHandle GetOrCreate(size_t nFFT, size_t batch_size);
    bool HasPlan(size_t nFFT, size_t batch_size) const;
    bool IsBaked(size_t nFFT, size_t batch_size) const;
    void MarkBaked(size_t nFFT, size_t batch_size);
    void Remove(size_t nFFT, size_t batch_size);
    void ClearAll();

    // Statistics
    size_t GetCacheSize() const;
    size_t GetTotalCreates() const;
    size_t GetTotalHits() const;
    double GetHitRatio() const;
    void PrintStats() const;
};
```

Кеш по ключу `(nFFT, batch_size)`. RAII — все планы освобождаются в деструкторе.

---

## FFTBatchAdapter

**Файл**: `include/fft_batch_adapter.hpp`

Адаптер для расчёта batch'ей через `DrvGPU::BatchManager`.

```cpp
class FFTBatchAdapter {
public:
    FFTBatchAdapter(const AntennaFFTParams& params, size_t nFFT);

    std::vector<BatchRange> CalculateBatches(
        IBackend* backend, size_t min_tail = 3, double mem_limit = 0.7) const;

    bool AllBeamsFit(IBackend* backend, double mem_limit = 0.7) const;
    size_t GetPerBeamMemory() const;
    size_t GetTotalRequiredMemory() const;
    void PrintMemoryInfo() const;
};
```

---

## FFTResultWriter

**Файл**: `include/fft_result_writer.hpp`

Вывод результатов на экран и в файлы (Markdown + JSON).

```cpp
class FFTResultWriter {
public:
    static void PrintResults(const AntennaFFTResult& result);
    static void PrintProfiling(const FFTProfilingResults& profiling);
    static std::string GetProfilingStats(const FFTProfilingResults& profiling);
    static void SaveResultsToFile(
        const AntennaFFTResult& result,
        const std::string& filepath,
        const FFTProfilingResults& profiling,
        const AntennaFFTParams& params,
        cl_command_queue queue = nullptr,
        cl_mem post_callback_userdata = nullptr);
};
```

---

## Параболическая интерполяция

Уточнение частоты до долей бина по 3 точкам: `(index-1, index, index+1)`.

```
                 ●  center (max)
                /|\
               / | \
              /  |  \
         ●   /   |   \   ●
    left  ──/────|────\── right
             freq_offset
             [-0.5, 0.5]

refined_frequency = (index + freq_offset) * sample_rate / nFFT
```

---

## Примеры

### Базовое использование

```cpp
antenna_fft::InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 8,
    .n_point = 4096,
    .data = generate_test_signal(),
    .repeat_count = 2,
    .sample_rate = 1000.0f
};

auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);

for (size_t i = 0; i < results.size(); ++i) {
    auto& r = results[i];
    printf("Antenna %u: %.4f Hz (mag=%.2f, phase=%.2f deg)\n",
           r.antenna_id, r.interpolated.refined_frequency,
           r.interpolated.magnitude, r.interpolated.phase);
}
```

### Batch processing (256 антенн x 1.3M точек)

```cpp
antenna_fft::InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 256,
    .n_point = 1300000,
    .data = huge_data,        // 256 * 1.3M * 8 bytes = 2.66 GB
    .repeat_count = 2,
    .sample_rate = 10000.0f,
    .memory_limit = 0.80f     // Использовать 80% свободной GPU памяти
};

// Автоматическое разбиение на batch'и
auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

---

*Обновлено: 2026-02-13*
