<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Да сделай развёрнутый план без силгентона с учётом ооп solid grasp gof

Сделаю план архитектуры генерации сигналов без синглтонов, в духе OOP + SOLID + GRASP + GoF, с явной интеграцией в твой существующий GPU‑пайплайн.

***

## 1. Общая архитектурная идея

Цели:

- Нет глобальных синглтонов и глобальных параметров.
- Чёткое разделение:
    - «что генерировать» (параметры и тип сигнала),
    - «как генерировать» (алгоритм),
    - «куда класть результат» (CPU/GPU, формат).
- Возможность легко добавлять новые типы сигналов (LFM‑PC, фазокод, мульти‑тон и т.д.) без правок существующего кода.

Ключевые паттерны:

- Strategy — разные алгоритмы генерации.
- Factory Method / Abstract Factory — создание генераторов по типу сигнала.
- Builder — построение сложных сценариев (пачки импульсов, композиции).
- Dependency Injection — конфигурируемые зависимости (GPU, аллокаторы, логи и т.д.).
- Interface Segregation + Single Responsibility — мелкие, чёткие интерфейсы.

***

## 2. Базовые сущности и интерфейсы

### 2.1. Параметры системы

```cpp
struct SystemSampling {
    double fs;          // частота дискретизации
    std::size_t length; // число отсчётов в одном кадре/сигнале
};
```

Эта структура передаётся в генераторы, а не хранится глобально.

### 2.2. Типы сигналов и их параметры

```cpp
enum class SignalKind {
    CW,
    LFM,
    PulseTrain,
    Custom
};

struct CwParams {
    double f0;
    double phase;
    double amplitude;
};

struct LfmParams {
    double f_start;
    double f_end;
    double duration;
    double amplitude;
    bool   complexIQ;
};

struct PulseTrainParams {
    std::size_t pulsesCount;
    double prf;
    // ссылка/указатель на «внутренний» сигнал (например, LFM внутри импульса)
};

struct SignalRequest {
    SignalKind kind;
    SystemSampling system;

    // вариант: std::variant<CwParams, LfmParams, PulseTrainParams, ...>
    std::variant<CwParams, LfmParams, PulseTrainParams> params;
};
```


***

## 3. Интерфейс генератора и стратегии

### 3.1. Общий интерфейс генератора

```cpp
class ISignalGenerator {
public:
    virtual ~ISignalGenerator() = default;

    // Генерация в CPU‑буфер
    virtual void generateToCpu(const SignalRequest& request,
                               std::complex<float>* out,
                               std::size_t outSize) = 0;

    // Генерация в GPU‑буфер (абстракция над DrvGPU)
    virtual void generateToGpu(const SignalRequest& request,
                               IGpuBuffer& buffer) = 0;
};
```

`IGpuBuffer` — твоя абстракция над `cl_mem` / HIP / ROCm:

```cpp
class IGpuBuffer {
public:
    virtual ~IGpuBuffer() = default;
    virtual std::size_t size() const = 0;    // в элементах или байтах
    virtual void* rawHandle() = 0;          // cl_mem / hipDeviceptr_t и т.п.
};
```


### 3.2. Конкретные стратегии

```cpp
class CwGenerator : public ISignalGenerator {
public:
    void generateToCpu(const SignalRequest& request,
                       std::complex<float>* out,
                       std::size_t outSize) override;

    void generateToGpu(const SignalRequest& request,
                       IGpuBuffer& buffer) override;
};

class LfmGenerator : public ISignalGenerator {
public:
    void generateToCpu(const SignalRequest& request,
                       std::complex<float>* out,
                       std::size_t outSize) override;

    void generateToGpu(const SignalRequest& request,
                       IGpuBuffer& buffer) override;
};
```

CPU‑реализации могут использовать чистый C++ (эталон), а `generateToGpu` — вызывать твой `DrvGPU`/kernels.

***

## 4. Фабрика генераторов (Factory Method / Abstract Factory)

### 4.1. Интерфейс фабрики

```cpp
class ISignalGeneratorFactory {
public:
    virtual ~ISignalGeneratorFactory() = default;

    virtual std::unique_ptr<ISignalGenerator>
    createGenerator(SignalKind kind) const = 0;
};
```


### 4.2. Простая реализация фабрики

```cpp
class DefaultSignalGeneratorFactory : public ISignalGeneratorFactory {
public:
    std::unique_ptr<ISignalGenerator>
    createGenerator(SignalKind kind) const override {
        switch (kind) {
        case SignalKind::CW:
            return std::make_unique<CwGenerator>();
        case SignalKind::LFM:
            return std::make_unique<LfmGenerator>();
        case SignalKind::PulseTrain:
            return std::make_unique<PulseTrainGenerator>();
        // ...
        default:
            throw std::runtime_error("Unsupported SignalKind");
        }
    }
};
```

В будущем можешь сделать отдельные фабрики: `GpuOptimizedFactory`, `CpuOnlyFactory`, `TestFactory` и т.д.

***

## 5. Ядро высокого уровня (фасад без синглтона)

Вместо синглтона — обычный фасад/сервис, который можно создать и передать туда зависимости.

```cpp
class SignalService {
public:
    SignalService(std::shared_ptr<ISignalGeneratorFactory> factory,
                  std::shared_ptr<IGpuContext> gpuContext)
        : factory_(std::move(factory))
        , gpuContext_(std::move(gpuContext)) {}

    // CPU
    std::vector<std::complex<float>>
    generateCpu(const SignalRequest& request) {
        auto gen = factory_->createGenerator(request.kind);
        std::vector<std::complex<float>> out(request.system.length);
        gen->generateToCpu(request, out.data(), out.size());
        return out;
    }

    // GPU
    void generateGpu(const SignalRequest& request, IGpuBuffer& buffer) {
        auto gen = factory_->createGenerator(request.kind);
        gen->generateToGpu(request, buffer);
    }

private:
    std::shared_ptr<ISignalGeneratorFactory> factory_;
    std::shared_ptr<IGpuContext> gpuContext_; // твой DrvGPU или обёртка
};
```

Все зависимости приходят снаружи (DI), никаких глобальных одиночек.

***

## 6. GRASP: роли и ответственность

- **Controller**: `SignalService` — координирует, что и как генерировать, но не знает деталей.
- **Creator**: `DefaultSignalGeneratorFactory` — создаёт нужную стратегию.
- **Low Coupling**: вызывающий код видит только `SignalService` и `SignalRequest`.
- **High Cohesion**:
    - каждый генератор отвечает только за один тип сигнала;
    - `SignalService` не занимается матанчиком, только оркестрацией.
- **Polymorphism**: `ISignalGenerator` задаёт единый контракт для всех типов сигналов.

***

## 7. Использование в твоём проекте

### 7.1. Инициализация на уровне приложения

```cpp
auto gpuContext = std::make_shared<DrvGpuContext>(/*devId, queues, ...*/);
auto factory    = std::make_shared<DefaultSignalGeneratorFactory>();

SignalService signalService(factory, gpuContext);
```


### 7.2. Создание запроса на ЛЧМ

```cpp
SystemSampling system { .fs = 100e6, .length = 1'000'000 };

LfmParams lfm {
    .f_start   = -10e6,
    .f_end     = +10e6,
    .duration  = 1e-3,
    .amplitude = 1.0,
    .complexIQ = true
};

SignalRequest req {
    .kind   = SignalKind::LFM,
    .system = system,
    .params = lfm
};

// CPU
auto refSignal = signalService.generateCpu(req);

// GPU
GpuBufferImpl gpuBuf(/*size = system.length * sizeof(complex<float>)*/);
signalService.generateGpu(req, gpuBuf);
```


***

## 8. Расширения (LFM‑PC, «сигналы типа ЛЧМ» и т.д.)

Чтобы добавить новый тип сигнала (например, LFM‑PC, гибриды, фазокод):

1. Добавляешь новый `enum` в `SignalKind`.
2. Создаёшь новую `struct XxxParams` и добавляешь в `std::variant` параметров.
3. Реализуешь `class XxxGenerator : public ISignalGenerator`.
4. Регистрируешь его в фабрике (`switch`/map).

Существующий код `SignalService` и остальная система не меняются (Open/Closed Principle).

***

## 9. GoF: куда ещё можно пойти

- **Builder** для сложных сценариев:

```cpp
class ScenarioBuilder {
public:
    ScenarioBuilder& withLfmPulse(const LfmParams& p);
    ScenarioBuilder& repeat(std::size_t count, double prf);
    ScenarioBuilder& addNoise(double snr);
    Scenario build();
};
```

- **Composite**: сложный сигнал как композиция базовых (CW + LFM + шум):

```cpp
class CompositeGenerator : public ISignalGenerator {
    std::vector<std::unique_ptr<ISignalGenerator>> parts_;
public:
    void addPart(std::unique_ptr<ISignalGenerator> g);
    // generateToCpu: суммировать результаты частей
};
```

- **Adapter**: обёртки для существующих GPU‑ядров, чтобы они тоже выглядели как `ISignalGenerator`.

***

Если нужно, дальше могу:

- расписать конкретный `LfmGenerator::generateToCpu` и версию `generateToGpu`, ожидая твой DrvGPU‑интерфейс;
- или предложить layout параметров для передачи на GPU (структура в constant buffer/SVM) и пример kernel’а под ROCm/OpenCL.

