# GPUWorkLib — Full Reference

> Развёрнутое описание всех публичных классов, структур, перечислений и методов.
> Назначение: читатель должен понять **что есть**, **зачем нужно** и **как использовать**.
>
> **Date**: 2026-03-05 | **Quick Reference**: [Quick_Reference.md](Quick_Reference.md) | **Index**: [INDEX.md](INDEX.md)

---

## Содержание

1. [FFT Processor](#1-fft-processor)
2. [Statistics (OpenCL / ROCm)](#2-statistics-rocm)
3. [Vector Algebra (OpenCL / ROCm)](#3-vector-algebra-rocm)
4. [FFT Maxima](#4-fft-maxima)
5. [Filters](#5-filters)
6. [Signal Generators](#6-signal-generators)
7. [LCH Farrow](#7-lch-farrow)
8. [Heterodyne](#8-heterodyne)
9. [FM Correlator](#9-fm-correlator)
10. [DrvGPU — Core Driver](#10-drvgpu--core-driver)
11. [Python API](#11-python-api)

---

## 1. FFT Processor

> **Путь**: `modules/fft_processor/`
> **Документация**: [Modules/fft_processor/Full.md](Modules/fft_processor/Full.md)
> **Backend**: OpenCL (`clFFT`) / ROCm (`hipFFT`), выбор по флагу

Модуль вычисляет быстрое преобразование Фурье для одного или нескольких каналов (антенн/лучей) одновременно. Поддерживает два режима вывода: **Complex** (комплексный спектр) и **MagPhase** (амплитуда + фаза). На AMD GPU автоматически использует `hipFFT`, на NVIDIA — `clFFT`.

### Класс `FFTProcessor`

```
modules/fft_processor/include/fft_processor.hpp
```

**Конструктор**

```cpp
explicit FFTProcessor(drv_gpu_lib::IBackend* backend);
```

Принимает указатель на бэкенд GPU. Бэкенд получается через `DrvGPU::GetBackend()`.
Объект **move-only** — нельзя копировать.

---

**Структура параметров `FFTProcessorParams`**

| Поле | Тип | Описание |
|------|-----|----------|
| `nfft` | `uint32_t` | Размер FFT (степень 2: 64…16384) |
| `n_beams` | `uint32_t` | Количество каналов (антенн/лучей) |
| `sample_rate` | `float` | Частота дискретизации, Hz |
| `window` | `WindowType` | Оконная функция (см. ниже) |
| `normalize` | `bool` | Нормализовать амплитуду (÷ nfft) |

**Перечисление `WindowType`**

```cpp
enum class WindowType {
    NONE,       // без окна (прямоугольник)
    HANN,       // окно Ханна
    HAMMING,    // окно Хэмминга
    BLACKMAN,   // окно Блэкмана
    FLAT_TOP,   // flat-top окно (точная амплитуда)
};
```

---

**Методы обработки**

```cpp
// CPU → Complex вывод
std::vector<FFTComplexResult> ProcessComplex(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);
```

`data` — плоский вектор, layout: `[beam0_pt0, beam0_pt1, ..., beam1_pt0, ...]`, размер = `n_beams * nfft`.

```cpp
// GPU → Complex вывод
std::vector<FFTComplexResult> ProcessComplex(
    cl_mem gpu_data,
    const FFTProcessorParams& params,
    size_t gpu_memory_bytes = 0,
    std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);
```

Входной буфер уже находится на GPU (`cl_mem`). `gpu_memory_bytes` — размер буфера в байтах (0 = вычислять автоматически).

```cpp
// CPU / GPU → MagPhase вывод (аналогично)
std::vector<FFTMagPhaseResult> ProcessMagPhase(...);
```

---

**Структуры результатов**

```cpp
struct FFTComplexResult {
    uint32_t beam_id;
    std::vector<std::complex<float>> spectrum;  // nfft точек
};

struct FFTMagPhaseResult {
    uint32_t beam_id;
    std::vector<float> magnitude;  // nfft значений |z|
    std::vector<float> phase;      // nfft значений arg(z) [-π, π]
};
```

**Диагностика**

```cpp
FFTProfilingData GetProfilingData() const;  // данные профилирования последнего вызова
uint32_t GetNFFT() const;                   // текущий размер FFT
```

---

## 2. Statistics *(OpenCL / ROCm)*

> **Путь**: `modules/statistics/`
> **Документация**: [Modules/statistics/](Modules/statistics/)
> **Backend**: OpenCL (CPU reference) / ROCm (GPU, требует `ENABLE_ROCM=1`)

Вычисляет статистические характеристики комплексных сигналов. Использует алгоритм **Welford** для численно устойчивого однопроходного вычисления среднего и дисперсии. Медиана вычисляется через **radix sort** на GPU.

### Класс `StatisticsProcessor`

```
modules/statistics/include/statistics_processor.hpp
```

**Конструктор**

```cpp
explicit StatisticsProcessor(drv_gpu_lib::IBackend* backend);
```

Бэкенд может быть OpenCL (CPU reference) или ROCm (GPU). На OpenCL выполняется CPU версия алгоритма, на ROCm — GPU HIP ядро.

---

**Структура параметров `StatisticsParams`**

| Поле | Тип | Описание |
|------|-----|----------|
| `n_beams` | `uint32_t` | Количество каналов (антенн) |
| `n_points` | `uint32_t` | Точек на канал |

---

**Методы**

```cpp
// Комплексное среднее (real и imag усредняются отдельно)
std::vector<MeanResult> ComputeMean(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params);

// Медиана амплитуд |z[i]| (radix sort на GPU для ROCm, CPU для OpenCL)
std::vector<MedianResult> ComputeMedian(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params);

// Полная статистика за один проход (самый быстрый вариант)
std::vector<StatisticsResult> ComputeStatistics(
    const std::vector<std::complex<float>>& data,
    const StatisticsParams& params);

// GPU варианты (данные уже на GPU, void* = HIP device ptr)
std::vector<MeanResult>       ComputeMean(void* gpu_data, const StatisticsParams& params);
std::vector<MedianResult>     ComputeMedian(void* gpu_data, const StatisticsParams& params);
std::vector<StatisticsResult> ComputeStatistics(void* gpu_data, const StatisticsParams& params);
```

---

**Структуры результатов**

```cpp
struct MeanResult {
    uint32_t beam_id;
    std::complex<float> mean;       // комплексное среднее: mean_re + i·mean_im
};

struct MedianResult {
    uint32_t beam_id;
    float median_magnitude;         // медиана |z|
};

struct StatisticsResult {
    uint32_t beam_id;
    std::complex<float> mean;       // комплексное среднее
    float variance;                 // дисперсия амплитуд σ²(|z|)
    float std_dev;                  // стандартное отклонение σ(|z|)
    float mean_magnitude;           // среднее амплитуд E[|z|]
};
```

---

## 3. Vector Algebra *(OpenCL / ROCm)*

> **Путь**: `modules/vector_algebra/`
> **Документация**: [Doc/Python/vector_algebra_api.md](Python/vector_algebra_api.md)
> **Backend**: OpenCL (CPU reference) / ROCm (GPU `rocSOLVER`: `POTRF` + `POTRI`)

Инверсия эрмитовых положительно-определённых матриц через разложение Холецкого.
**Важно**: rocSOLVER работает в column-major формате. Для row-major данных используем `rocblas_fill_lower` (эквивалентно `fill_upper` при транспонировании).

### Перечисление `SymmetrizeMode`

```cpp
enum class SymmetrizeMode {
    Roundtrip,   // скачать на CPU → симметризовать → загрузить обратно
    GpuKernel,   // HIP kernel in-place (быстрее, рекомендуется)
};
```

### Класс `CholeskyInverterROCm`

```
modules/vector_algebra/include/cholesky_inverter_rocm.hpp
```

**Конструктор**

```cpp
explicit CholeskyInverterROCm(
    drv_gpu_lib::IBackend* backend,
    SymmetrizeMode mode = SymmetrizeMode::GpuKernel);
```

---

**Управление режимом**

```cpp
void SetSymmetrizeMode(SymmetrizeMode mode);
SymmetrizeMode GetSymmetrizeMode() const;
```

После `POTRI` матрица содержит только треугольник. `Symmetrize` заполняет вторую половину:
- `GpuKernel` — kernel `symmetrize_matrix` запускается на GPU (≈0.05 мс для 341×341)
- `Roundtrip` — D2H + CPU copy + H2D (≈0.5 мс, но нет hiprtc зависимости)

**Контроль проверок**

```cpp
void SetCheckInfo(bool enabled);   // false = не проверять info после POTRF/POTRI
                                   // (убирает синхронный hipMemcpy D2H, ускоряет benchmark)
void CompileKernels();             // явная прекомпиляция hiprtc kernels (опционально)
```

---

**Входные данные `InputData<T>`**

Универсальный шаблон для передачи данных из разных источников:

```cpp
// CPU вектор
InputData<std::vector<std::complex<float>>> input{data};

// HIP device ptr (ROCm)
InputData<void*> input{hip_ptr};

// OpenCL буфер
InputData<cl_mem> input{cl_buffer};
```

---

**Методы инверсии**

```cpp
// Одна матрица n×n
CholeskyResult Invert(const InputData<vector<complex<float>>>& input, int n = 0);
CholeskyResult Invert(const InputData<void*>& input, int n = 0);
CholeskyResult Invert(const InputData<cl_mem>& input, int n = 0);

// Batch: batch_count = data.size() / (n*n) матриц размером n×n
CholeskyResult InvertBatch(const InputData<vector<complex<float>>>& input, int n);
CholeskyResult InvertBatch(const InputData<void*>& input, int n);
CholeskyResult InvertBatch(const InputData<cl_mem>& input, int n);
```

---

**Структура `CholeskyResult`**

RAII-обёртка над HIP device ptr. **Move-only** — нельзя копировать.

```cpp
struct CholeskyResult {
    void*  d_data      = nullptr;   // HIP device ptr (owns memory, освобождается в деструкторе)
    IBackend* backend  = nullptr;   // для memcpy / free
    int matrix_size    = 0;         // n (размер стороны матрицы)
    int batch_count    = 0;         // количество матриц в batch

    // Скачать на CPU
    std::vector<std::complex<float>> AsVector() const;  // плоский, row-major
    std::vector<std::vector<std::complex<float>>> matrix() const;          // [n][n]
    std::vector<std::vector<std::vector<std::complex<float>>>> matrices() const; // [k][n][n]

    // Прямой доступ к GPU памяти
    void* AsHipPtr() const;
};
```

---

## 4. FFT Maxima

> **Путь**: `modules/fft_maxima/`
> **Документация**: [Modules/fft_maxima/Full.md](Modules/fft_maxima/Full.md)
> **Backend**: OpenCL / ROCm (стратегия выбирается через фабрику)

Ищет спектральные максимумы в FFT-выходе. Поддерживает два режима: **OnePeak** (один пик на канал с параболической интерполяцией) и **AllMaxima** (все пики выше порога, GPU-компактация).

### Фабрика `SpectrumProcessorFactory`

```cpp
// modules/fft_maxima/include/spectrum_processor_factory.hpp
static std::unique_ptr<ISpectrumProcessor> Create(
    drv_gpu_lib::BackendType backend_type,
    drv_gpu_lib::IBackend* backend);
```

Возвращает правильную реализацию в зависимости от `backend_type`:
- `OPENCL` → OpenCL реализация
- `ROCM` → HIP реализация

### Интерфейс `ISpectrumProcessor`

```
modules/fft_maxima/include/interface/i_spectrum_processor.hpp
```

**Инициализация**

```cpp
void Initialize(const SpectrumParams& params);
bool IsInitialized() const;
```

**Структура `SpectrumParams`**

| Поле | Тип | Описание |
|------|-----|----------|
| `nfft` | `uint32_t` | Размер FFT |
| `n_antennas` | `uint32_t` | Количество антенн |
| `sample_rate` | `float` | Частота дискретизации, Hz |
| `threshold_db` | `float` | Порог для AllMaxima (dB относительно пика) |

---

**Режим OnePeak (один главный пик)**

```cpp
// CPU вход
std::vector<SpectrumResult> ProcessFromCPU(
    const std::vector<std::complex<float>>& data);

// GPU вход
std::vector<SpectrumResult> ProcessFromGPU(
    void* gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes = 0);

// Batch-обработка (для больших данных)
std::vector<SpectrumResult> ProcessBatch(
    const std::vector<std::complex<float>>& batch_data,
    size_t start_antenna, size_t batch_antenna_count);

std::vector<SpectrumResult> ProcessBatchFromGPU(
    void* gpu_data, size_t src_offset_bytes,
    size_t start_antenna, size_t batch_antenna_count);
```

---

**Режим AllMaxima (все пики)**

```cpp
// FFT + поиск всех пиков (полный pipeline)
AllMaximaResult FindAllMaximaFromGPUPipeline(
    void* gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes,
    OutputDestination dest,    // CPU или GPU
    uint32_t search_start,     // начало диапазона поиска (бин)
    uint32_t search_end);      // конец диапазона (0 = весь спектр)

// Поиск пиков по готовым FFT данным (только поиск, без FFT)
AllMaximaResult FindAllMaxima(
    void* fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest = OutputDestination::CPU,
    uint32_t search_start = 0, uint32_t search_end = 0);

// Из CPU данных (FFT + поиск)
AllMaximaResult FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>& data,
    OutputDestination dest, uint32_t search_start, uint32_t search_end);
```

---

**Структуры результатов**

```cpp
struct MaxValue {
    uint32_t index;              // индекс бина в FFT
    float real, imag;            // комплексное значение
    float magnitude;             // |z|
    float phase;                 // arg(z)
    float freq_offset;           // смещение частоты (Hz) от 0
    float refined_frequency;     // уточнённая частота (Hz) после интерполяции
    uint32_t pad;                // выравнивание
};

struct SpectrumResult {
    uint32_t antenna_id;
    MaxValue interpolated;       // интерполированный пик
    MaxValue left_point;         // левая точка (index-1)
    MaxValue center_point;       // центральный пик
    MaxValue right_point;        // правая точка (index+1)
};

struct AllMaximaBeamResult {
    uint32_t antenna_id;
    uint32_t num_maxima;
    std::vector<MaxValue> maxima; // все найденные пики
};

struct AllMaximaResult {
    std::vector<AllMaximaBeamResult> beams;  // CPU результаты
    OutputDestination destination;
    void* gpu_maxima  = nullptr;             // GPU результаты (если dest=GPU)
    void* gpu_counts  = nullptr;
    size_t total_maxima = 0;
    size_t gpu_bytes    = 0;
};

enum class OutputDestination { CPU, GPU };
```

---

**Утилиты**

```cpp
void ReallocateBuffersForBatch(size_t batch_antenna_count);
size_t CalculateBytesPerAntenna() const;
void CompilePostKernel();
DriverType GetDriverType() const;
ProfilingData GetProfilingData() const;
```

---

## 5. Filters

> **Путь**: `modules/filters/`
> **Документация**: [Modules/filters/Full.md](Modules/filters/Full.md)
> **Backend**: OpenCL / ROCm
> **Статус**: реализован и протестирован (C++ тесты, Python тесты, бенчмарки).

Реализует два вида цифровых фильтров: **FIR** (КИХ) с прямой свёрткой и **IIR** (БИХ) в форме каскадных бикватных секций (Direct Form II Transposed).

### Класс `FirFilter`

```
modules/filters/include/filters/fir_filter.hpp
```

КИХ-фильтр (Finite Impulse Response). Реализует операцию свёртки `y[n] = Σ h[k]·x[n-k]` на GPU.

**Конструктор**

```cpp
explicit FirFilter(drv_gpu_lib::IBackend* backend);
```

**Конфигурация**

```cpp
// Прямая установка коэффициентов
void SetCoefficients(const std::vector<float>& coeffs);  // h[0], h[1], ..., h[N-1]

// Загрузка из JSON (поле "coefficients": [...])
void LoadConfig(const std::string& json_path);

bool IsReady() const;
uint32_t GetNumTaps() const;
const std::vector<float>& GetCoefficients() const;
```

**Обработка**

```cpp
// GPU: вход cl_mem → выход InputData<cl_mem>
drv_gpu_lib::InputData<cl_mem> Process(
    cl_mem input_buf,
    uint32_t channels,   // количество каналов
    uint32_t points,     // точек на канал
    std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);

// CPU reference (для проверки)
std::vector<std::complex<float>> ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points);
```

---

### Класс `IirFilter`

```
modules/filters/include/filters/iir_filter.hpp
```

БИХ-фильтр (Infinite Impulse Response) в форме каскадных бикватных секций.
Каждая секция реализует: `y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]`.

**Конструктор**

```cpp
explicit IirFilter(drv_gpu_lib::IBackend* backend);
```

**Структура секции `BiquadSection`**

```cpp
struct BiquadSection {
    float b0, b1, b2;   // числитель (нули)
    float a1, a2;        // знаменатель (полюсы), a0 = 1.0 подразумевается
};
```

**Конфигурация**

```cpp
void SetBiquadSections(const std::vector<BiquadSection>& sections);
void LoadConfig(const std::string& json_path);  // поле "sections": [{...}]

bool IsReady() const;
uint32_t GetNumSections() const;
const std::vector<BiquadSection>& GetSections() const;
```

**Обработка** — аналогично `FirFilter`:

```cpp
drv_gpu_lib::InputData<cl_mem> Process(cl_mem input_buf, uint32_t channels, uint32_t points, ...);
std::vector<std::complex<float>> ProcessCpu(const std::vector<std::complex<float>>& input, ...);
```

---

## 6. Signal Generators

> **Путь**: `modules/signal_generators/`
> **Документация**: [Modules/signal_generators/Full.md](Modules/signal_generators/Full.md)
> **Backend**: OpenCL / ROCm

Генерирует тестовые и рабочие сигналы на GPU. Все генераторы реализуют интерфейс `ISignalGenerator`. Используй фабрику `SignalGeneratorFactory` вместо прямого создания.

### Перечисление `SignalKind`

```cpp
enum class SignalKind { CW, LFM, NOISE, FORM_SIGNAL };
```

### Структура `SystemSampling`

Параметры системы дискретизации, общие для всех генераторов:

```cpp
struct SystemSampling {
    float sample_rate;    // частота дискретизации (Hz)
    uint32_t n_points;    // точек на луч
};
```

### Параметры генераторов

**`CwParams`** — непрерывная волна (CW)

| Поле | Тип | Описание |
|------|-----|----------|
| `f0` | `double` | Несущая частота (Hz) |
| `phase` | `double` | Начальная фаза (rad) |
| `amplitude` | `double` | Амплитуда (1.0 = единичная) |
| `complex_iq` | `bool` | Вывод I/Q (true) или real (false) |
| `freq_step` | `double` | Шаг частоты между лучами (Hz) |

**`LfmParams`** — линейная частотная модуляция

| Поле | Тип | Описание |
|------|-----|----------|
| `f_start` | `double` | Начальная частота (Hz) |
| `f_end` | `double` | Конечная частота (Hz) |
| `amplitude` | `double` | Амплитуда |
| `complex_iq` | `bool` | Вывод I/Q |

Вспомогательные методы: `GetChirpRate(double duration)`, `GetBandwidth()`.

**`NoiseParams`** — генератор шума

| Поле | Тип | Описание |
|------|-----|----------|
| `type` | `NoiseType` | `WHITE` или `GAUSSIAN` |
| `power` | `double` | Мощность шума |
| `seed` | `uint64_t` | Seed для PRNG (Philox) |

**`FormParams`** — сложная форма сигнала (getX DSL)

| Поле | Тип | Описание |
|------|-----|----------|
| `script` | `string` | DSL-скрипт (`"sin(2*PI*f0*t)"`) |
| `f0` | `double` | Параметр f0 для DSL |
| `amplitude` | `double` | Амплитуда |

### Интерфейс `ISignalGenerator`

```cpp
// Генерация на CPU (reference, медленно)
virtual void GenerateToCpu(
    const SystemSampling& system,
    std::complex<float>* out,
    size_t out_size) = 0;

// Генерация на GPU (production, быстро)
virtual cl_mem GenerateToGpu(
    const SystemSampling& system,
    size_t beam_count = 1) = 0;

virtual SignalKind Kind() const = 0;
```

### Фабрика `SignalGeneratorFactory`

```cpp
// Создание конкретного генератора
static std::unique_ptr<ISignalGenerator>     CreateCw(IBackend*, const CwParams&);
static std::unique_ptr<ISignalGenerator>     CreateLfm(IBackend*, const LfmParams&);
static std::unique_ptr<ISignalGenerator>     CreateNoise(IBackend*, const NoiseParams&);
static std::unique_ptr<FormSignalGenerator>  CreateForm(IBackend*, const FormParams&);
static std::unique_ptr<FormSignalGeneratorROCm> CreateFormROCm(IBackend*, const FormParams&);
static std::unique_ptr<FormScriptGenerator>  CreateFormScript(IBackend*, const FormParams&);

// Универсальный метод по запросу
static std::unique_ptr<ISignalGenerator> Create(IBackend*, const SignalRequest& request);
```

**`SignalRequest`** — универсальный запрос:

```cpp
struct SignalRequest {
    SignalKind kind;
    SystemSampling system;
    std::variant<CwParams, LfmParams, NoiseParams, FormParams> params;
};
```

---

## 7. LCH Farrow

> **Путь**: `modules/lch_farrow/`
> **Документация**: [Modules/lch_farrow/Full.md](Modules/lch_farrow/Full.md)
> **Backend**: OpenCL / ROCm

Реализует дробную задержку через интерполяцию Лагранжа 5-го порядка с матрицей коэффициентов 48×5 (48 фаз × 5 точек). Применяется для точного временного выравнивания сигналов от разных антенн.

### Класс `LchFarrow`

```
modules/lch_farrow/include/lch_farrow.hpp
```

**Конструктор**

```cpp
explicit LchFarrow(drv_gpu_lib::IBackend* backend);
```

**Конфигурация**

```cpp
// Частота дискретизации входного сигнала
void SetSampleRate(float sample_rate);

// Задержки в микросекундах для каждой антенны
// Размер вектора должен совпадать с n_antennas при обработке
void SetDelays(const std::vector<float>& delay_us);

// Добавить шум (опционально, для тестирования)
// amplitude — амплитуда шума
// norm_val — нормировочный коэффициент (по умолчанию 1/√2)
// noise_seed — seed для PRNG
void SetNoise(float noise_amplitude,
              float norm_val = 0.7071067811865476f,
              uint32_t noise_seed = 0);

// Загрузить матрицу коэффициентов 48×5 из JSON
void LoadMatrix(const std::string& json_path);

// Геттеры
const std::vector<float>& GetDelays() const;
float GetSampleRate() const;
```

**Обработка**

```cpp
// GPU обработка
// Возвращает InputData<cl_mem> — GPU буфер с задержанными сигналами
drv_gpu_lib::InputData<cl_mem> Process(
    cl_mem input_buf,
    uint32_t antennas,   // количество антенн
    uint32_t points,     // точек на антенну
    std::vector<std::pair<const char*, cl_event>>* prof_events = nullptr);

// CPU reference
// Возвращает vector[antenna][point]
std::vector<std::vector<std::complex<float>>> ProcessCpu(
    const std::vector<std::vector<std::complex<float>>>& input,
    uint32_t antennas, uint32_t points);
```

---

## 8. Heterodyne

> **Путь**: `modules/heterodyne/`
> **Документация**: [Modules/heterodyne/Full.md](Modules/heterodyne/Full.md)
> **Backend**: OpenCL / ROCm

Реализует **LFM Dechirp** (stretch processing) — алгоритм измерения дальности для ЛЧМ сигналов. Pipeline: генерация эталонного ЛЧМ → перемножение (гетеродинирование) → БПФ → поиск пика → вычисление дальности.

### Структура `HeterodyneParams`

```cpp
struct HeterodyneParams {
    float f_start      = 0.0f;    // начальная частота ЛЧМ (Hz)
    float f_end        = 1e6f;    // конечная частота ЛЧМ (Hz)
    float sample_rate  = 12e6f;   // частота дискретизации (Hz)
    int num_samples    = 4000;    // точек на антенну
    int num_antennas   = 5;       // количество антенн

    // Вспомогательные вычисления
    float GetBandwidth() const;   // f_end - f_start (Hz)
    float GetDuration() const;    // num_samples / sample_rate (с)
    float GetChirpRate() const;   // Bandwidth / Duration (Hz/с)
    float GetBinWidth() const;    // sample_rate / num_samples (Hz/бин)
};
```

### Структуры результатов

```cpp
struct AntennaDechirpResult {
    int   antenna_idx     = 0;      // индекс антенны
    float f_beat_hz       = 0.0f;   // beat-частота (Hz)
    float f_beat_bin      = 0.0f;   // beat-частота в бинах (дробная)
    float range_m         = 0.0f;   // дальность (м)
    float peak_amplitude  = 0.0f;   // амплитуда пика
    float peak_snr_db     = 0.0f;   // SNR в дБ
};

struct HeterodyneResult {
    bool success = false;
    std::vector<AntennaDechirpResult> antennas;
    std::vector<float> max_positions;    // позиции всех максимумов
    std::string error_message;

    // Статический метод перевода beat-частоты в дальность
    static float CalcRange(
        float f_beat,        // beat-частота (Hz)
        float sample_rate,   // частота дискретизации (Hz)
        int num_samples,     // точек на антенну
        float bandwidth);    // полоса сигнала (Hz)
};
```

### Класс `HeterodyneDechirp`

```
modules/heterodyne/include/heterodyne_dechirp.hpp
```

**Конструктор**

```cpp
explicit HeterodyneDechirp(
    IBackend* backend,
    BackendType compute_backend = BackendType::OPENCL);
```

`compute_backend` задаёт выбор вычислительного пути (OpenCL или ROCm).

**Методы**

```cpp
// Установить параметры ЛЧМ сигнала
void SetParams(const HeterodyneParams& params);
const HeterodyneParams& GetParams() const;

// Обработать CPU данные (vector, размер = num_antennas * num_samples)
HeterodyneResult Process(const std::vector<std::complex<float>>& rx_data);

// Обработать GPU данные (внешний буфер, params передаётся явно)
HeterodyneResult ProcessExternal(void* rx_gpu_ptr, const HeterodyneParams& params);

// Последний результат
const HeterodyneResult& GetLastResult() const;
```

---

## 9. FM Correlator

> **Путь**: `modules/fm_correlator/`
> **Документация**: [Modules/fm_correlator/Full.md](Modules/fm_correlator/Full.md)
> **Backend**: ROCm only — требует `ENABLE_ROCM=1`
> **Статус**: реализован и протестирован (C++ тесты, Python тесты, бенчмарки).

Вычисляет FM-корреляцию сигналов с M-sequence эталоном. M-последовательность генерируется на CPU через LFSR (Linear Feedback Shift Register). Pipeline: генерация M-seq → FFT эталона (GPU) → FFT входных сигналов → перемножение в частотной области → IFFT → поиск пиков.

### Структура `FMCorrelatorParams`

```cpp
struct FMCorrelatorParams {
    size_t   fft_size         = 32768;        // размер FFT
    int      num_shifts       = 32;           // количество сдвигов (K)
    int      num_signals      = 5;            // количество входных сигналов (S)
    int      num_output_points = 2000;        // точек в выходном массиве (n_kg)
    uint32_t lfsr_polynomial  = 0x00400007;   // полином LFSR: x^32+x^22+x^2+x+1
    uint32_t lfsr_seed        = 0x12345678;   // начальное состояние LFSR
};
```

### Структура `FMCorrelatorResult`

```cpp
struct FMCorrelatorResult {
    std::vector<float> peaks;   // [S * K * n_kg], row-major: [signal][shift][point]
    int num_signals       = 0;
    int num_shifts        = 0;
    int num_output_points = 0;

    // Удобный accessor
    float at(int signal, int shift, int point) const;
};
```

### Класс `FMCorrelator`

```
modules/fm_correlator/include/fm_correlator.hpp
```

**Конструктор**

```cpp
explicit FMCorrelator(drv_gpu_lib::IBackend* backend);  // backend — ROCm IBackend*
```

**Методы**

```cpp
// Установить параметры
void SetParams(const FMCorrelatorParams& params);
const FMCorrelatorParams& GetParams() const;

// Генерация M-sequence на CPU (LFSR)
std::vector<float> GenerateMSequence() const;           // seed из params
std::vector<float> GenerateMSequence(uint32_t seed) const; // custom seed

// Подготовить эталон (FFT на GPU)
void PrepareReference(const std::vector<float>& ref);  // из внешнего вектора
void PrepareReference();                                // из M-seq (seed из params)

// Корреляция входных данных с эталоном
FMCorrelatorResult Process(const std::vector<float>& inp);

// Тестовый паттерн: входные данные = circshift(ref, s * shift_step) на GPU
// ref должен быть подготовлен заранее. Не требует H2D для входных данных.
FMCorrelatorResult RunTestPattern(int shift_step = 2);

// Auto-batching для больших S
FMCorrelatorResult ProcessWithBatching(const std::vector<float>& inp, int total_signals);
```

---

## 10. DrvGPU — Core Driver

> **Путь**: `DrvGPU/`
> **Документация**: [DrvGPU/Architecture.md](DrvGPU/Architecture.md) · [DrvGPU/Classes.md](DrvGPU/Classes.md)

Ядро системы. Предоставляет единую абстракцию GPU (OpenCL и ROCm), управляет памятью, очередями команд, профилированием и логированием. Все модули получают доступ к GPU исключительно через `IBackend*`.

### Класс `DrvGPU`

```
DrvGPU/include/drv_gpu.hpp
```

Единственный GPU. **Move-only** — нельзя копировать.

```cpp
DrvGPU(BackendType backend_type, int device_index = 0);

// Информация
std::string GetDeviceName() const;
GPUDeviceInfo GetDeviceInfo() const;
int GetDeviceIndex() const;
BackendType GetBackendType() const;

// Доступ к бэкенду (для передачи в модули)
IBackend& GetBackend();

// Синхронизация
void Synchronize();
void Flush();

// Статистика
void PrintStatistics() const;
std::string GetStatistics() const;
void ResetStatistics();
```

### Класс `GPUManager`

```
DrvGPU/include/gpu_manager.hpp
```

Менеджер для Multi-GPU систем (до 10 GPU).

```cpp
GPUManager mgr;
mgr.InitializeAll(BackendType::OPENCL);
mgr.InitializeSpecific(BackendType::ROCM, {0, 1, 2});

// Доступ к GPU
DrvGPU& g = mgr.GetGPU(0);
DrvGPU& g = mgr.GetNextGPU();           // Round-Robin балансировка
DrvGPU& g = mgr.GetLeastLoadedGPU();    // по загрузке
size_t n  = mgr.GetGPUCount();
std::vector<DrvGPU*> all = mgr.GetAllGPUs();

// Для профайлера
GPUReportInfo info = mgr.GetGPUReportInfo(gpu_id);

// Статика
static int GetAvailableGPUCount(BackendType bt);
```

### Перечисление `BackendType`

```cpp
enum class BackendType {
    OPENCL,   // OpenCL (NVIDIA, AMD, Intel)
    ROCM,     // ROCm / HIP (AMD only)
    HYBRID,   // OpenCL + ROCm совместно
};
```

### Интерфейс `IBackend`

```
DrvGPU/interface/i_backend.hpp
```

Абстрактный интерфейс для всех бэкендов. Модули работают только через этот интерфейс.

```cpp
// Жизненный цикл
virtual void Initialize(int device_index) = 0;
virtual bool IsInitialized() const = 0;
virtual void Cleanup() = 0;

// Информация
virtual BackendType GetType() const = 0;
virtual std::string GetDeviceName() const = 0;
virtual GPUDeviceInfo GetDeviceInfo() const = 0;
virtual int GetDeviceIndex() const = 0;

// Нативные хэндлы (void* для межплатформенности)
virtual void* GetNativeContext() const = 0;   // cl_context / hipCtx_t
virtual void* GetNativeDevice()  const = 0;   // cl_device_id / hipDevice_t
virtual void* GetNativeQueue()   const = 0;   // cl_command_queue / hipStream_t

// Управление памятью
virtual void* Allocate(size_t size_bytes, unsigned int flags = 0) = 0;
virtual void  Free(void* ptr) = 0;
virtual void  MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) = 0;
virtual void  MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) = 0;
virtual void  MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) = 0;

// Синхронизация
virtual void Synchronize() = 0;
virtual void Flush() = 0;

// Возможности
virtual bool   SupportsSVM() const = 0;
virtual bool   SupportsDoublePrecision() const = 0;
virtual size_t GetMaxWorkGroupSize() const = 0;
virtual size_t GetGlobalMemorySize() const = 0;
virtual size_t GetFreeMemorySize() const = 0;
virtual size_t GetLocalMemorySize() const = 0;
```

### Сервис `GPUProfiler`

```
DrvGPU/services/gpu_profiler.hpp
```

Профилировщик GPU задач. **Обязательно** вызвать `SetGPUInfo()` до `Start()`.

```cpp
GPUProfiler profiler;

// ⚠️ Устанавливать GPU info ДО Start() — иначе в отчёте «Unknown»
profiler.SetGPUInfo(mgr.GetGPUReportInfo(gpu_id));
profiler.Start();

// ... GPU работа ...

profiler.Stop("stage_name");     // зафиксировать этап

// Вывод
profiler.PrintReport();          // в консоль через ConsoleOutput
profiler.ExportMarkdown("report.md");
profiler.ExportJSON("report.json");
```

**Запрещено**: вручную вызывать `GetStats()` и выводить через `con.Print` / `std::cout`. Только `PrintReport()` / `ExportMarkdown()` / `ExportJSON()`.

### Сервис `ConsoleOutput`

```
DrvGPU/services/console_output.hpp
```

Мультиgpu-безопасный вывод в консоль через async очередь.

```cpp
auto& con = gpu.GetBackend().GetConsoleOutput();

// ⚠️ Только 3-arg вариант (gpu_id, module_name, message)
con.Print(gpu_id, "FFTProcessor", "FFT completed");

// Запрещено: 1-arg Print или std::cout напрямую
```

### Интерфейс `IComputeModule`

```
DrvGPU/interface/i_compute_module.hpp
```

Базовый интерфейс для всех вычислительных модулей (опционально):

```cpp
virtual void Initialize() = 0;
virtual bool IsInitialized() const = 0;
virtual void Cleanup() = 0;
virtual std::string GetName() const = 0;
virtual std::string GetVersion() const = 0;
virtual IBackend* GetBackend() const = 0;
```

### Шаблон `InputData<T>`

```
DrvGPU/interface/input_data.hpp
```

Универсальная обёртка для передачи данных. `T` — одно из:
- `std::vector<complex<float>>` — CPU вектор
- `void*` — ROCm HIP device ptr
- `cl_mem` — OpenCL буфер

```cpp
InputData<std::vector<std::complex<float>>> input_cpu{cpu_vec};
InputData<void*>   input_hip{hip_ptr};
InputData<cl_mem>  input_cl{cl_buffer};
```

---

## 11. Python API

> **Документация**: [Python/](Python/) · [Python_test/Full.md](Python_test/Full.md)
> **Биндинги**: `python/gpu_worklib_bindings.cpp` (pybind11)

GPUWorkLib предоставляет полный Python API через **pybind11**. Каждый модуль обёрнут в Python класс с NumPy-совместимым интерфейсом.

### Сборка и установка

```bash
# Сборка с Python биндингами
cmake .. -DBUILD_PYTHON=ON -DENABLE_ROCM=ON -DCMAKE_PREFIX_PATH=/opt/rocm
make -j$(nproc)

# Запуск тестов
PYTHONPATH=build/python python run_tests.py -m 
```

### Как работает конвертация C++ → Python

Все C++ типы автоматически конвертируются через pybind11:

| C++ тип | Python тип | Примечание |
|---------|-----------|-----------|
| `std::vector<float>` | `numpy.ndarray` dtype=float32 | buffer protocol |
| `std::vector<complex<float>>` | `numpy.ndarray` dtype=complex64 | buffer protocol |
| `std::string` | `str` | UTF-8 |
| `struct { float ... }` | `dict` | через .def_readwrite |
| `enum class` | `IntEnum` | через py::enum_ |
| `unique_ptr<T>` | instance T | владение передаётся Python |
| `cl_mem` / `void*` | `GPUBuffer` | специальный handle-класс |

**Пример биндинга модуля** (`python/py_*.hpp`):

```cpp
// C++ класс
class StatisticsProcessor {
    StatisticsProcessor(IBackend* backend);
    std::vector<StatisticsResult> ComputeStatistics(...);
};

// Python биндинг (в py_statistics.hpp)
py::class_<PyStatisticsProcessor>(m, "StatisticsProcessor")
    .def(py::init<ROCmGPUContext&>())
    .def("compute_statistics", &PyStatisticsProcessor::compute_statistics,
         py::arg("data"), py::arg("n_beams"), py::arg("n_points"))
    .def_readonly("results", &PyStatisticsProcessor::last_results);
```

### Классы Python API

| Python класс | C++ класс | Файл биндинга | API документация |
|-------------|-----------|---------------|------------------|
| `GPUContext` | `DrvGPU` | `gpu_worklib_bindings.cpp` | — |
| `ROCmGPUContext` | `DrvGPU (ROCm)` | `gpu_worklib_bindings.cpp` | [Python/rocm_modules_api.md](Python/rocm_modules_api.md) |
| `SignalGenerator` | `ISignalGenerator` | `py_signal_gen.hpp` | [Python/signal_generators_api.md](Python/signal_generators_api.md) |
| `SpectrumMaximaFinder` | `ISpectrumProcessor` | `py_fft_maxima.hpp` | [Python/spectrum_maxima_api.md](Python/spectrum_maxima_api.md) |
| `FirFilter` | `FirFilter` | `py_filters.hpp` | [Python/rocm_modules_api.md](Python/rocm_modules_api.md) |
| `IirFilter` | `IirFilter` | `py_filters.hpp` | [Python/rocm_modules_api.md](Python/rocm_modules_api.md) |
| `FirFilterROCm` | `FirFilter (ROCm)` | `py_filters_rocm.hpp` | [Python/rocm_modules_api.md](Python/rocm_modules_api.md) |
| `IirFilterROCm` | `IirFilter (ROCm)` | `py_filters_rocm.hpp` | [Python/rocm_modules_api.md](Python/rocm_modules_api.md) |
| `LchFarrow` | `LchFarrow` | `py_lch_farrow.hpp` | [Python/lch_farrow_api.md](Python/lch_farrow_api.md) |
| `LchFarrowROCm` | `LchFarrow (ROCm)` | `py_lch_farrow_rocm.hpp` | [Python/rocm_modules_api.md](Python/rocm_modules_api.md) |
| `HeterodyneDechirp` | `HeterodyneDechirp` | `py_heterodyne.hpp` | [Python/rocm_modules_api.md](Python/rocm_modules_api.md) |
| `HeterodyneROCm` | `HeterodyneDechirp (ROCm)` | `py_heterodyne_rocm.hpp` | [Python/rocm_modules_api.md](Python/rocm_modules_api.md) |
| `StatisticsProcessor` | `StatisticsProcessor` | `py_statistics.hpp` | [Python/rocm_modules_api.md](Python/rocm_modules_api.md) |
| `CholeskyInverterROCm` | `CholeskyInverterROCm` | `py_vector_algebra_rocm.hpp` | [Python/vector_algebra_api.md](Python/vector_algebra_api.md) |
| `LfmAnalyticalDelay` | `LfmAnalyticalDelay` | `py_lfm_analytical_delay.hpp` | [Python/signal_generators_api.md](Python/signal_generators_api.md) |
| `FMCorrelatorROCm` | `FMCorrelator` | `py_fm_correlator_rocm.hpp` | [Python/fm_correlator_api.md](Python/fm_correlator_api.md) |

### Примеры Python использования

**OpenCL (NVIDIA / любой GPU)**

```python
import gpu_worklib as gw
import numpy as np

# Инициализация
ctx = gw.GPUContext(device_index=0)           # OpenCL backend

# Генератор сигнала
gen = gw.SignalGenerator(ctx)
gen.set_cw(f0=100e3, amplitude=1.0)
signal = gen.generate(sample_rate=12e6, n_points=4096, n_beams=8)
# signal — numpy.ndarray shape=(8, 4096) dtype=complex64

# FFT
fft = gw.FFTProcessor(ctx)
spectrum = fft.process_complex(signal, nfft=4096)
# spectrum — numpy.ndarray shape=(8, 4096) dtype=complex64

# Поиск пиков
finder = gw.SpectrumMaximaFinder(ctx)
finder.initialize(nfft=4096, n_antennas=8, sample_rate=12e6)
peaks = finder.find_all_maxima(spectrum)
# peaks — list of dicts: [{'antenna_id': 0, 'maxima': [{'freq': ..., 'magnitude': ...}]}]
```

**OpenCL (CPU reference, любой GPU)**

```python
import gpu_worklib as gw
import numpy as np

# OpenCL контекст
ctx = gw.GPUContext(device_index=0)

# Статистика (CPU reference версия)
stat = gw.StatisticsProcessor(ctx)
data = np.random.randn(8, 1024).astype(np.complex64)
result = stat.compute_statistics(data, n_beams=8, n_points=1024)
# result — list of dicts: [{'beam_id': 0, 'mean': ..., 'std_dev': ..., 'variance': ...}]
```

**ROCm (AMD GPU)**

```python
import gpu_worklib as gw
import numpy as np

# ROCm контекст
ctx = gw.ROCmGPUContext(device_index=0)

# Статистика (GPU версия на ROCm)
stat = gw.StatisticsProcessor(ctx)
data = np.random.randn(8, 1024).astype(np.complex64)
result = stat.compute_statistics(data, n_beams=8, n_points=1024)
# result — list of dicts: [{'beam_id': 0, 'mean': ..., 'std_dev': ..., 'variance': ...}]

# Cholesky инверсия
inv = gw.CholeskyInverterROCm(ctx)
matrix = np.eye(10, dtype=np.complex64)      # 10×10 единичная матрица
result = inv.invert_cpu(matrix.flatten().tolist(), n=10)
# result — numpy.ndarray shape=(10, 10) dtype=complex64

# Фильтры ROCm
fir = gw.FirFilterROCm(ctx)
fir.set_coefficients([0.1, 0.2, 0.4, 0.2, 0.1])
filtered = fir.process(signal, channels=8, points=1024)
```

### Python тесты

Все тесты в `Python_test/` с подпапками по модулям. Полное описание: [Python_test/Full.md](Python_test/Full.md)

| Директория | Модуль | Тестов | Примеры |
|-----------|--------|--------|---------|
| `signal_generators/` | SignalGenerator, FormSignal, LfmAnalytical | 3 файла | [test_form_signal.py](../Python_test/signal_generators/test_form_signal.py) |
| `fft_maxima/` | SpectrumMaximaFinder | 3 файла | [test_spectrum_find_all_maxima.py](../Python_test/fft_maxima/test_spectrum_find_all_maxima.py) |
| `filters/` | FirFilter, IirFilter (OpenCL + ROCm) | 6 файлов | [test_filters_stage1.py](../Python_test/filters/test_filters_stage1.py) |
| `heterodyne/` | HeterodyneDechirp, ROCm | 4 файла | [test_heterodyne.py](../Python_test/heterodyne/test_heterodyne.py) |
| `lch_farrow/` | LchFarrow (OpenCL + ROCm) | 2 файла | [test_lch_farrow.py](../Python_test/lch_farrow/test_lch_farrow.py) |
| `statistics/` | StatisticsProcessor (ROCm) | 1 файл | [test_statistics_rocm.py](../Python_test/statistics/test_statistics_rocm.py) |
| `vector_algebra/` | CholeskyInverterROCm | 2 файла | [test_cholesky_inverter_rocm.py](../Python_test/vector_algebra/test_cholesky_inverter_rocm.py) |
| `integration/` | Все модули совместно | 1 файл | [test_gpuworklib.py](../Python_test/integration/test_gpuworklib.py) |
| `fm_correlator/` | FMCorrelatorROCm | 1 файл | [test_fm_correlator_rocm.py](../Python_test/fm_correlator/test_fm_correlator_rocm.py) |

---

*See also: [Quick_Reference.md](Quick_Reference.md) · [INDEX.md](INDEX.md) · [Architecture/](Architecture/)*
*Last updated: 2026-03-05 | Maintained by: Кодо*
