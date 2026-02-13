# Signal Generators — API Reference

**Namespace**: `signal_gen`

---

## Типы данных

### SystemSampling

Параметры дискретизации, общие для всех генераторов.

```cpp
struct SystemSampling {
    double fs = 1000.0;    // Частота дискретизации (Hz)
    size_t length = 1024;  // Количество отсчётов на луч
};
```

### SignalKind

```cpp
enum class SignalKind { CW, LFM, NOISE };
```

### CwParams

Параметры CW (continuous wave) генератора.

```cpp
struct CwParams {
    double f0 = 100.0;           // Частота (Hz)
    double phase = 0.0;          // Начальная фаза (rad)
    double amplitude = 1.0;      // Амплитуда
    bool complex_iq = true;      // true = exp(j*phase), false = real only
    double freq_step = 0.0;      // Шаг частоты между лучами (Hz)
};
```

Для multi-beam: `freq_i = f0 + i * freq_step`

### LfmParams

Параметры LFM (chirp) генератора.

```cpp
struct LfmParams {
    double f_start = 100.0;      // Начальная частота (Hz)
    double f_end = 500.0;        // Конечная частота (Hz)
    double amplitude = 1.0;      // Амплитуда
    bool complex_iq = true;      // Комплексный выход

    double GetChirpRate(double duration) const;  // k = (f_end - f_start) / duration
};
```

### NoiseParams

Параметры генератора шума.

```cpp
enum class NoiseType { WHITE, GAUSSIAN };

struct NoiseParams {
    NoiseType type = NoiseType::GAUSSIAN;
    double power = 1.0;          // Мощность шума (дисперсия для Gaussian)
    uint64_t seed = 0;           // 0 = random seed
};
```

### SignalRequest

Единый запрос через `std::variant`.

```cpp
struct SignalRequest {
    SignalKind kind;
    SystemSampling system;
    std::variant<CwParams, LfmParams, NoiseParams> params;
};
```

---

## ISignalGenerator (Interface)

**Файл**: `include/i_signal_generator.hpp`

Абстрактный интерфейс, реализуемый CW, LFM, Noise генераторами.

| Метод | Описание |
|-------|----------|
| `GenerateToCpu(system, out, out_size)` | Генерация на CPU (reference) |
| `GenerateToGpu(system, beam_count) → cl_mem` | Генерация на GPU |
| `Kind() → SignalKind` | Тип сигнала |

```cpp
// CPU генерация
void GenerateToCpu(const SystemSampling& system,
                   std::complex<float>* out, size_t out_size);

// GPU генерация — вызывающий должен освободить cl_mem!
cl_mem GenerateToGpu(const SystemSampling& system, size_t beam_count = 1);
```

---

## CwGenerator

**Файл**: `include/generators/cw_generator.hpp`

Генерирует: `s(t) = A * exp(j * (2*pi*f*t + phase))`

```cpp
CwGenerator(IBackend* backend, const CwParams& params);

void SetParams(const CwParams& params);
const CwParams& GetParams() const;
```

---

## LfmGenerator

**Файл**: `include/generators/lfm_generator.hpp`

Генерирует: `s(t) = A * exp(j*pi*k*t^2 + j*2*pi*f_start*t)`

```cpp
LfmGenerator(IBackend* backend, const LfmParams& params);

void SetParams(const LfmParams& params);
const LfmParams& GetParams() const;
```

---

## NoiseGenerator

**Файл**: `include/generators/noise_generator.hpp`

GPU: Philox-2x32 PRNG + Box-Muller transform
CPU: `std::mt19937` + `std::normal_distribution`

```cpp
NoiseGenerator(IBackend* backend, const NoiseParams& params);

void SetParams(const NoiseParams& params);
const NoiseParams& GetParams() const;
```

---

## SignalGeneratorFactory

**Файл**: `include/signal_generator_factory.hpp`

Создаёт генераторы по типу сигнала.

| Метод | Возвращает |
|-------|------------|
| `CreateCw(backend, params)` | `unique_ptr<ISignalGenerator>` |
| `CreateLfm(backend, params)` | `unique_ptr<ISignalGenerator>` |
| `CreateNoise(backend, params)` | `unique_ptr<ISignalGenerator>` |
| `Create(backend, request)` | `unique_ptr<ISignalGenerator>` |

---

## SignalService (Facade)

**Файл**: `include/signal_service.hpp`

Единая точка входа для генерации сигналов.

```cpp
explicit SignalService(IBackend* backend);
```

### CPU генерация (1 луч)

```cpp
vector<complex<float>> GenerateCpu(const CwParams& params, const SystemSampling& system);
vector<complex<float>> GenerateCpu(const LfmParams& params, const SystemSampling& system);
vector<complex<float>> GenerateCpu(const NoiseParams& params, const SystemSampling& system);
```

### GPU генерация (N лучей)

```cpp
cl_mem GenerateGpu(const CwParams& params, const SystemSampling& system, size_t beam_count = 1);
cl_mem GenerateGpu(const LfmParams& params, const SystemSampling& system, size_t beam_count = 1);
cl_mem GenerateGpu(const NoiseParams& params, const SystemSampling& system, size_t beam_count = 1);
```

> **Внимание**: GPU методы возвращают `cl_mem`. Вызывающий код отвечает за `clReleaseMemObject()`!

---

## Примеры

### Прямое использование генераторов

```cpp
// CW
signal_gen::CwParams cw{.f0 = 440.0, .amplitude = 0.5};
auto gen = signal_gen::SignalGeneratorFactory::CreateCw(backend, cw);

signal_gen::SystemSampling sys{.fs = 44100.0, .length = 8192};
cl_mem gpu_data = gen->GenerateToGpu(sys, 16);  // 16 лучей
// ...
clReleaseMemObject(gpu_data);
```

### Через SignalService

```cpp
signal_gen::SignalService service(backend);

// LFM chirp 100→500 Hz
signal_gen::LfmParams lfm{.f_start = 100.0, .f_end = 500.0};
auto cpu_data = service.GenerateCpu(lfm, {44100.0, 4096});
```

### Через SignalRequest (variant)

```cpp
signal_gen::SignalRequest req;
req.kind = signal_gen::SignalKind::CW;
req.system = {1000.0, 4096};
req.params = signal_gen::CwParams{.f0 = 100.0};

auto gen = signal_gen::SignalGeneratorFactory::Create(backend, req);
```

---

*Обновлено: 2026-02-13*
