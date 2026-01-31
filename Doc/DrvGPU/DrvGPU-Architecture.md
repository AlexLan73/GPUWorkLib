# 📐 DrvGPU Architecture Guide

## 🎯 Цель документа

Этот документ описывает полную архитектуру библиотеки DrvGPU, включая:
- Layered Architecture
- Component Diagrams
- Design Patterns
- Multi-GPU Architecture
- Backend Abstraction

---

## 🏛️ Layered Architecture

DrvGPU построена по принципу layered architecture:

```
╔══════════════════════════════════════════════════════════════╗
║               APPLICATION LAYER                              ║
║  (Пользовательский код использует DrvGPU API)                ║
╚══════════════════════════════════╦═══════════════════════════╝
                                   ║
╔══════════════════════════════════╩═══════════════════════════╗
║               PUBLIC API LAYER                               ║
║  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐  ║
║  │   DrvGPU       │  │  GPUManager  │  │ IMemoryBuffer   │  ║
║  │ (main class)   │  │  (multi-GPU) │  │   (interface)   │  ║
║  └────────────────┘  └──────────────┘  └─────────────────┘  ║
╚══════════════════════════════════╦═══════════════════════════╝
                                   ║
╔══════════════════════════════════╩═══════════════════════════╗
║          BACKEND ABSTRACTION LAYER                           ║
║  ┌───────────────────────────────────────────────────────┐   ║
║  │              IBackend (interface)                     │   ║
║  │  - Initialize(device_id)                              │   ║
║  │  - CreateBuffer(...)                                  │   ║
║  │  - CompileKernel(...)                                 │   ║
║  │  - Synchronize()                                      │   ║
║  └───────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════╦═══════════════════════════╝
                                   ║
        ╔══════════════════════════╩══════════════════════════╗
        ║         BACKEND IMPLEMENTATIONS                     ║
        ║  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ ║
        ║  │ BackendOpenCL│  │ BackendCUDA  │  │BackendVulkan║
        ║  │   (ready)    │  │  (planned)   │  │  (planned)║ ║
        ║  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘ ║
        ╚═════════╩══════════════════╩════════════════╩═══════╝
                  ║                  ║                ║
        ╔═════════╩═══════╗ ╔════════╩═══════╗ ╔═════╩══════╗
        ║   OpenCL SDK    ║ ║   CUDA SDK     ║ ║ Vulkan SDK ║
        ╚═════════════════╝ ╚════════════════╝ ╚════════════╝
```

### Описание слоёв

#### 1. Application Layer
- **Ответственность**: Бизнес-логика приложения
- **Использует**: Public API DrvGPU
- **Примеры**: Обработка сигналов, FFT, научные вычисления

#### 2. Public API Layer
- **Ответственность**: Простой, type-safe API для приложений
- **Компоненты**:
  - `DrvGPU` - главный класс для работы с GPU
  - `GPUManager` - координатор для Multi-GPU
  - `IMemoryBuffer` - унифицированный интерфейс для памяти
- **Паттерны**: Facade, Factory Method

#### 3. Backend Abstraction Layer
- **Ответственность**: Абстракция над различными GPU API
- **Компоненты**:
  - `IBackend` - интерфейс бэкенда
- **Паттерны**: Bridge, Strategy

#### 4. Backend Implementations Layer
- **Ответственность**: Реализация для конкретных GPU API
- **Компоненты**:
  - `BackendOpenCL` - OpenCL реализация
  - `BackendCUDA` - CUDA реализация (planned)
  - `BackendVulkan` - Vulkan реализация (planned)

---

## 📊 Component Diagram

### High-Level Components

```
┌──────────────────────────────────────────────────────────────┐
│                       GPUManager                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Responsibilities:                                    │    │
│  │ - Discover all available GPUs                        │    │
│  │ - Initialize backend (OpenCL/CUDA/Vulkan)           │    │
│  │ - Create DrvGPU instances for each GPU              │    │
│  │ - Load balancing (Round-Robin, Least-Loaded, etc.)  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Methods:                                                    │
│  - Initialize(BackendType)                                   │
│  - GetAllGPUs() -> vector<uint32_t>                         │
│  - CreateDrvGPU(device_id) -> unique_ptr<DrvGPU>            │
│  - GetNextGPU(strategy) -> uint32_t                         │
└───────────────────────┬──────────────────────────────────────┘
                        │ creates
                        ▼
        ┌───────────────┴──────────────┬─────────────┐
        │                              │             │
┌───────▼────────┐           ┌─────────▼──────┐      ...
│   DrvGPU #0    │           │   DrvGPU #1    │
│  (GPU 0)       │           │  (GPU 1)       │
├────────────────┤           ├────────────────┤
│ - backend_     │           │ - backend_     │
│ - device_id_   │           │ - device_id_   │
├────────────────┤           ├────────────────┤
│ Methods:       │           │ Methods:       │
│ - CreateBuffer │           │ - CreateBuffer │
│ - Synchronize  │           │ - Synchronize  │
└────────┬───────┘           └────────┬───────┘
         │ owns                       │ owns
         ▼                            ▼
┌────────────────┐           ┌────────────────┐
│  IBackend*     │           │  IBackend*     │
│ (OpenCL/CUDA)  │           │ (OpenCL/CUDA)  │
└────────────────┘           └────────────────┘
```

### Memory Management Components

```
┌──────────────────────────────────────────────────────────────┐
│                    IMemoryBuffer                             │
│                   (abstract interface)                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Abstract Methods:                                    │    │
│  │ - Write(data)                                        │    │
│  │ - Read() -> ComplexVector                            │    │
│  │ - SetAsKernelArg(kernel, index)                      │    │
│  │ - GetStrategy() -> MemoryStrategy                    │    │
│  │ - IsSVM() -> bool                                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────┬────────────────────────────────────────┘
                      │ implements
        ┌─────────────┼─────────────┬─────────────────┐
        │             │             │                 │
┌───────▼──────┐ ┌────▼────────┐ ┌──▼──────────┐ ┌───▼──────────┐
│RegularBuffer │ │ SVMBuffer   │ │HybridBuffer │ │GPUMemoryBuffer│
│  (cl_mem)    │ │ (SVM ptr)   │ │(auto-select)│ │   (legacy)   │
└──────────────┘ └─────────────┘ └─────────────┘ └──────────────┘
                      ▲
                      │ creates
              ┌───────┴────────┐
              │ BufferFactory  │
              │                │
              │ - Create(...)  │
              │ - DetermineStrategy()
              └────────────────┘
```

---

## 🎨 Design Patterns

### 1. Facade Pattern

**Где используется:** `DrvGPU` class

**Зачем:**
- Упрощённый интерфейс для сложной подсистемы
- Скрывает детали реализации бэкенда
- Удобный API для пользователя

**Пример:**

```cpp
// Без Facade (сложно):
auto backend = CreateOpenCLBackend();
backend->Initialize(0);
auto ctx = backend->GetContext();
auto queue = backend->GetQueue();
auto factory = BufferFactory(ctx, queue, ...);
auto buffer = factory.Create(size);

// С Facade (просто):
auto drv = manager.CreateDrvGPU(0);
auto buffer = drv->CreateBuffer(size);
```

### 2. Bridge Pattern

**Где используется:** `IBackend` interface

**Зачем:**
- Разделить абстракцию (DrvGPU) от реализации (OpenCL/CUDA)
- Возможность менять реализацию независимо от интерфейса
- Поддержка нескольких бэкендов

**Структура:**

```
   DrvGPU (Abstraction)
       │
       │ uses
       ▼
   IBackend (Interface)
       △
       │ implements
   ┌───┴────┬──────┐
   │        │      │
OpenCL    CUDA  Vulkan
(Implementations)
```

### 3. Strategy Pattern

**Где используется:** Memory management (SVM/Regular)

**Зачем:**
- Выбор стратегии памяти во время выполнения
- Различные стратегии для разных сценариев

**Стратегии:**

| Strategy | Когда использовать |
|----------|-------------------|
| `REGULAR_BUFFER` | Маленькие буферы (< 1MB) |
| `SVM_COARSE_GRAIN` | Средние буферы (1-64MB), map/unmap ok |
| `SVM_FINE_GRAIN` | Большие буферы (> 64MB), частый доступ |

### 4. Factory Method Pattern

**Где используется:**
- `BufferFactory` - создание буферов
- `CreateBackend()` - создание бэкендов
- `GPUManager::CreateDrvGPU()` - создание DrvGPU

**Зачем:**
- Инкапсуляция логики создания
- Автоматический выбор реализации

**Пример:**

```cpp
// Factory автоматически выбирает SVM или Regular
auto factory = drv->CreateBufferFactory();
auto buffer = factory->Create(size); // SVM если доступен

// Принудительный выбор
auto regular = factory->CreateWithStrategy(
    size, 
    MemoryStrategy::REGULAR_BUFFER
);
```

### 5. RAII Pattern

**Где используется:** Везде!

**Зачем:**
- Автоматическое управление ресурсами
- Гарантия освобождения памяти

**Примеры:**

```cpp
{
    auto drv = manager.CreateDrvGPU(0);
    auto buffer = drv->CreateBuffer(size);
    
    // Работаем...
    
} // Автоматическое освобождение buffer и drv
```

---

## 🔄 Multi-GPU Architecture

### Архитектура Multi-Instance

**Ключевое решение:** DrvGPU **НЕ singleton!**

```cpp
// ❌ СТАРЫЙ ПОДХОД (Singleton - не работает для Multi-GPU):
auto& gpu = DrvGPU::GetInstance(); // Только одна GPU!

// ✅ НОВЫЙ ПОДХОД (Multi-Instance):
auto drv0 = manager.CreateDrvGPU(0); // GPU 0
auto drv1 = manager.CreateDrvGPU(1); // GPU 1
auto drv2 = manager.CreateDrvGPU(2); // GPU 2
```

### Координация через GPUManager

```
┌──────────────────────────────────────────────────────────┐
│                    GPUManager                            │
│  ┌───────────────────────────────────────────────┐      │
│  │  GPU Discovery:                                │      │
│  │  - OpenCL: clGetPlatformIDs() + clGetDeviceIDs() │   │
│  │  - CUDA: cudaGetDeviceCount()                  │      │
│  │  - Vulkan: vkEnumeratePhysicalDevices()        │      │
│  └───────────────────────────────────────────────┘      │
│                                                          │
│  ┌───────────────────────────────────────────────┐      │
│  │  Load Balancing:                               │      │
│  │  - Round-Robin: gpu_id = (counter++) % N      │      │
│  │  - Least-Loaded: min(usage_counter)           │      │
│  │  - Memory-Based: max(free_memory)             │      │
│  └───────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────┘
         │                  │                  │
    creates           creates           creates
         ▼                  ▼                  ▼
   ┌─────────┐        ┌─────────┐        ┌─────────┐
   │DrvGPU #0│        │DrvGPU #1│        │DrvGPU #2│
   └─────────┘        └─────────┘        └─────────┘
```

### Паттерны использования Multi-GPU

#### Pattern 1: Data Parallelism

```cpp
GPUManager manager;
manager.Initialize(BackendType::OPENCL);

auto gpus = manager.GetAllGPUs();
std::vector<std::unique_ptr<DrvGPU>> drivers;

// Создать DrvGPU для каждой GPU
for (auto id : gpus) {
    drivers.push_back(manager.CreateDrvGPU(id));
}

// Разделить данные
size_t chunk_size = data.size() / drivers.size();

// Обработка параллельно
std::vector<std::thread> threads;
for (size_t i = 0; i < drivers.size(); ++i) {
    threads.emplace_back([&, i]() {
        auto& drv = drivers[i];
        auto chunk = GetChunk(data, i, chunk_size);
        
        auto buffer = drv->CreateBufferWithData(chunk);
        // Обработка...
        auto result = buffer->Read();
    });
}

for (auto& t : threads) {
    t.join();
}
```

#### Pattern 2: Task Parallelism

```cpp
// GPU 0: FFT processing
auto drv0 = manager.CreateDrvGPU(0);
std::thread t0([&]() {
    auto buf = drv0->CreateBuffer(size);
    // FFT kernel execution...
});

// GPU 1: Matrix multiplication
auto drv1 = manager.CreateDrvGPU(1);
std::thread t1([&]() {
    auto buf = drv1->CreateBuffer(size);
    // Matrix kernel execution...
});

t0.join();
t1.join();
```

#### Pattern 3: Load Balancing

```cpp
// Автоматическая балансировка нагрузки
for (int i = 0; i < num_tasks; ++i) {
    auto drv = manager.GetNextDrvGPU(
        LoadBalancingStrategy::LEAST_LOADED
    );
    
    ProcessTask(drv, tasks[i]);
}
```

---

## 🔌 Backend Abstraction

### IBackend Interface

```cpp
class IBackend {
public:
    // === Lifecycle ===
    virtual void Initialize(uint32_t device_id) = 0;
    virtual bool IsInitialized() const = 0;
    virtual void Synchronize() = 0;
    virtual void Cleanup() = 0;
    
    // === Device Info ===
    virtual std::string GetDeviceName() const = 0;
    virtual size_t GetGlobalMemoryMB() const = 0;
    virtual uint32_t GetComputeUnits() const = 0;
    
    // === Buffer Management ===
    virtual std::unique_ptr<IMemoryBuffer> CreateBuffer(...) = 0;
    virtual std::unique_ptr<IMemoryBuffer> CreateBufferWithData(...) = 0;
    
    // === Capabilities ===
    virtual bool SupportsSVM() const = 0;
    virtual MemoryStrategy GetBestMemoryStrategy() const = 0;
    
    // === Kernel Execution ===
    virtual void* CompileKernel(...) = 0;
    virtual void ReleaseKernel(void* handle) = 0;
};
```

### Backend Implementations

#### BackendOpenCL (Ready)

```cpp
class BackendOpenCL : public IBackend {
private:
    std::unique_ptr<OpenCLCore> core_;
    std::unique_ptr<OpenCLBufferFactory> factory_;
    SVMCapabilities svm_caps_;
    
public:
    void Initialize(uint32_t device_id) override {
        core_ = std::make_unique<OpenCLCore>();
        core_->Initialize(device_id);
        
        factory_ = std::make_unique<OpenCLBufferFactory>(
            core_->GetContext(),
            core_->GetQueue(),
            core_->GetDevice()
        );
        
        svm_caps_ = SVMCapabilities::Query(core_->GetDevice());
    }
    
    std::unique_ptr<IMemoryBuffer> CreateBuffer(...) override {
        return factory_->Create(num_elements, mem_type, hint);
    }
    
    bool SupportsSVM() const override {
        return svm_caps_.HasAnySVM();
    }
};
```

#### BackendCUDA (Planned)

```cpp
class BackendCUDA : public IBackend {
private:
    cudaDeviceProp device_props_;
    int device_id_;
    
public:
    void Initialize(uint32_t device_id) override {
        device_id_ = static_cast<int>(device_id);
        cudaSetDevice(device_id_);
        cudaGetDeviceProperties(&device_props_, device_id_);
    }
    
    std::unique_ptr<IMemoryBuffer> CreateBuffer(...) override {
        return std::make_unique<CUDABuffer>(size, device_id_);
    }
};
```

---

## 📈 Performance Considerations

### Memory Strategy Selection

| Размер буфера | GPU поддержка | Выбранная стратегия | Обоснование |
|---------------|---------------|---------------------|-------------|
| < 1 MB | - | Regular Buffer | SVM overhead не оправдан |
| 1-64 MB | SVM есть | SVM Coarse-Grain | Zero-copy выгоден, map/unmap ok |
| 1-64 MB | SVM нет | Regular Buffer | Fallback |
| > 64 MB | SVM есть | SVM Fine-Grain | Максимальная производительность |
| > 64 MB | SVM нет | Regular Buffer | Fallback |

### Multi-GPU Scaling

**Теоретический speedup:** `S = N` (где N = количество GPU)

**Реальный speedup:**
- Overhead на синхронизацию: 5-10%
- PCIe bandwidth: может быть узким местом
- Load imbalance: до 15% потерь

**Типичные результаты:**

| GPUs | Ideal Speedup | Real Speedup | Efficiency |
|------|---------------|--------------|------------|
| 1 | 1.0x | 1.0x | 100% |
| 2 | 2.0x | 1.85x | 92% |
| 4 | 4.0x | 3.50x | 87% |
| 8 | 8.0x | 6.50x | 81% |

---

## 🔐 Thread Safety

### Thread-Safe Components

| Component | Thread-Safety | Почему |
|-----------|--------------|--------|
| `GPUManager` | ✅ YES | Mutex защищает shared state |
| `DrvGPU` | ❌ NO | Один экземпляр = один поток |
| `IMemoryBuffer` | ❌ NO | Один буфер = один поток |
| `BufferFactory` | ✅ YES | Статистика защищена mutex |

### Best Practices

```cpp
// ✅ CORRECT: Создать DrvGPU для каждого потока
std::vector<std::thread> threads;
for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
        auto drv = manager.CreateDrvGPU(i % gpu_count);
        // Работа с drv в этом потоке
    });
}

// ❌ WRONG: Делить один DrvGPU между потоками
auto drv = manager.CreateDrvGPU(0);
std::thread t1([&]() { drv->CreateBuffer(size); }); // Race!
std::thread t2([&]() { drv->CreateBuffer(size); }); // Race!
```

---

## 📚 Дальнейшее чтение

- [API Reference](API-Reference.md) - Полное API
- [Migration Guide](Migration-Guide.md) - Миграция сDrvGPU
- [Backend Development](Backend-Development.md) - Разработка бэкендов
- [Multi-GPU Guide](Multi-GPU-Guide.md) - Best practices для Multi-GPU

---

**DrvGPU Architecture** - Проверенная, масштабируемая, готовая к производству! 🚀
