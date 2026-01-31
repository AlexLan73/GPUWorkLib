# Архитектура модульной GPU библиотеки для вычислений

## Оглавление
1. [Введение и цели проекта](#введение-и-цели-проекта)
2. [Анализ требований](#анализ-требований)
3. [Архитектурные принципы](#архитектурные-принципы)
4. [Архитектура системы](#архитектура-системы)
5. [Детальное проектирование компонентов](#детальное-проектирование-компонентов)
6. [Система сборки и тестирования](#система-сборки-и-тестирования)
7. [Стратегия расширения](#стратегия-расширения)
8. [Дорожная карта разработки](#дорожная-карта-разработки)

---

## Введение и цели проекта

### Цели
Создание долгосрочной, модульной, расширяемой библиотеки для GPU-вычислений с следующими характеристиками:

- **Абстракция бэкенда**: Единое API для работы с OpenCL и ROCm с возможностью переключения в runtime
- **Модульность**: Независимые библиотеки для разных функциональных областей
- **Тестируемость**: Полный набор unit-тестов с возможностью условной компиляции
- **Расширяемость**: Простое добавление новых модулей без изменения существующего кода
- **Производительность**: Эффективное управление памятью GPU и минимизация копирований

### Scope
- Библиотеки для цифровой обработки сигналов (DSP)
- Математические операции после FFT
- Гетеродинная обработка
- Дробная задержка сигналов
- Базовая статистика и обработка

---

## Анализ требований

### Функциональные требования

#### FR-1: Абстракция GPU бэкенда
- Поддержка OpenCL (текущая)
- Поддержка ROCm (будущая)
- Runtime переключение между бэкендами
- Единое управление памятью GPU

#### FR-2: Модульная структура библиотек
- **Core**: DrvGPU (синглетон), управление памятью, абстракция бэкенда
- **DSP модули**:
  - Fractional Delay (дробная задержка)
  - Heterodyne (гетеродин)
  - FFT Post-Processing (поиск максимумов, парабола, перегибы)
  - Signal Statistics (медиана, среднее, магнитуда)
- **Test модули**: К каждой библиотеке

#### FR-3: Управление памятью
- Shared memory pool на GPU
- Zero-copy операции между модулями
- Lazy allocation/deallocation
- RAII принципы

#### FR-4: Тестирование
- Unit тесты для каждого модуля
- Условная компиляция тестов (CMake опции)
- Интеграционные тесты
- Performance benchmarks

### Нефункциональные требования

#### NFR-1: Производительность
- Минимум CPU-GPU transfers
- Асинхронные операции где возможно
- Pipeline execution для последовательных операций

#### NFR-2: Maintainability
- SOLID принципы
- GRASP patterns
- Comprehensive documentation
- Clear module boundaries

#### NFR-3: Extensibility
- Plugin архитектура для модулей
- Версионирование API
- Backward compatibility

---

## Архитектурные принципы

### SOLID Principles

#### Single Responsibility Principle (SRP)
- **DrvGPU**: Только управление контекстом и памятью GPU
- **BackendInterface**: Только абстракция бэкенда
- **ComputeModule**: Только бизнес-логика вычислений
- **MemoryManager**: Только управление выделением/освобождением

#### Open/Closed Principle (OCP)
- Расширение через новые модули (наследование/композиция)
- Закрыто для модификации базовых интерфейсов
- Plugin система для загрузки модулей

#### Liskov Substitution Principle (LSP)
- Все Backend реализации взаимозаменяемы
- Все ComputeModule реализации используют единый контракт

#### Interface Segregation Principle (ISP)
- IBackend: Только методы работы с GPU
- IMemoryManager: Только методы работы с памятью
- IComputeModule: Только методы вычислений
- Не заставляем модули реализовывать ненужное

#### Dependency Inversion Principle (DIP)
- Зависимость на абстракции (интерфейсы)
- DrvGPU зависит от IBackend, а не от OpenCLBackend
- Модули зависят от IMemoryManager, а не от конкретной реализации

### GRASP Patterns

#### Information Expert
- **DrvGPU** - эксперт по состоянию GPU контекста
- **MemoryManager** - эксперт по распределению памяти
- **BackendInterface** - эксперт по специфике бэкенда

#### Creator
- **DrvGPU** создает MemoryManager (тесно связан)
- **ModuleFactory** создает ComputeModules
- **BackendFactory** создает Backend

#### Low Coupling
- Модули общаются через абстрактные интерфейсы
- Dependency Injection для связывания
- Event-based communication для асинхронности

#### High Cohesion
- Каждый модуль - одна функциональная область
- Связанные операции в одном классе
- Минимум "utility" классов

#### Controller
- **DrvGPU** - контроллер для всей системы
- **ModuleRegistry** - контроллер для управления модулями
- **ComputePipeline** - контроллер для цепочек операций

#### Polymorphism
- Backend через виртуальные функции
- ComputeModule через template method pattern
- MemoryManager через strategy pattern

#### Pure Fabrication
- **ModuleFactory** - создание модулей
- **ConfigLoader** - загрузка конфигурации
- **Logger** - логирование

#### Indirection
- Интерфейсы между слоями
- Registry для разрыва прямых зависимостей

#### Protected Variations
- Backend за интерфейсом (защита от изменений OpenCL/ROCm)
- MemoryAllocator за интерфейсом (защита от изменений стратегии)

### GoF Patterns

#### Singleton (с осторожностью!)
- **DrvGPU**: Thread-safe singleton для единственного GPU контекста
- **Альтернатива**: Dependency Injection через IoC container
- **Justification**: GPU контекст действительно уникален, но доступ через DI

#### Abstract Factory
- **BackendFactory**: Создание семейства объектов (Context, Queue, Memory)
- OpenCLBackendFactory vs ROCmBackendFactory

#### Factory Method
- **ModuleFactory**: Создание конкретных модулей
- Virtual createModule() в базе

#### Strategy
- **MemoryAllocationStrategy**: Разные стратегии выделения памяти
- **BackendStrategy**: Выбор бэкенда

#### Bridge
- **Backend абстракция**: Отделение абстракции от реализации
- Позволяет менять бэкенд независимо от модулей

#### Facade
- **DrvGPU**: Упрощенный интерфейс ко всей системе
- Скрывает сложность Backend, Memory, Module management

#### Template Method
- **ComputeModule**: Базовый алгоритм вычисления (initialize, compute, finalize)
- Конкретные модули переопределяют шаги

#### Observer (Optional)
- Уведомления об изменении состояния памяти
- Мониторинг выполнения операций

---

## Архитектура системы

### Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│            Application Layer (User Code)                │
│  - Pipeline Builder                                     │
│  - High-level API для конкретных задач                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│          Compute Modules Layer (Libraries)              │
│  - FFTPostProcessing                                    │
│  - FractionalDelay                                      │
│  - Heterodyne                                           │
│  - SignalStatistics                                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Core Layer (DrvGPU)                        │
│  - GPU Context Management                               │
│  - Memory Management                                    │
│  - Module Registry                                      │
│  - Pipeline Execution                                   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│           Backend Abstraction Layer                     │
│  - IBackend interface                                   │
│  - IMemoryBuffer interface                              │
│  - IKernel interface                                    │
└─────────────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │  OpenCL Backend  │    │   ROCm Backend   │
    │  Implementation  │    │  Implementation  │
    └──────────────────┘    └──────────────────┘
```

### Component Diagram

```
┌────────────────────────────────────────────────────────────┐
│                        DrvGPU (Singleton)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               GPU Context Manager                    │  │
│  │  - Backend selection                                │  │
│  │  - Context lifecycle                                │  │
│  │  - Error handling                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Memory Manager                          │  │
│  │  - Memory Pool (GPU-side)                           │  │
│  │  - Buffer allocation/deallocation                   │  │
│  │  - Zero-copy sharing                                │  │
│  │  - Defragmentation                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Module Registry                         │  │
│  │  - Module discovery                                 │  │
│  │  - Dependency resolution                            │  │
│  │  - Lifecycle management                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Pipeline Executor                          │  │
│  │  - Operation sequencing                             │  │
│  │  - Async execution                                  │  │
│  │  - Error propagation                                │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## Детальное проектирование компонентов

### 1. Core Layer: DrvGPU

```cpp
// DrvGPU.hpp
#pragma once

#include "IBackend.hpp"
#include "MemoryManager.hpp"
#include "ModuleRegistry.hpp"
#include <memory>
#include <mutex>

namespace gpu_lib {
namespace core {

/**
 * @brief Центральный класс-синглетон для управления GPU ресурсами
 * 
 * Responsibilities (SRP):
 * - Управление жизненным циклом GPU контекста
 * - Предоставление доступа к Memory Manager
 * - Координация работы модулей (Controller - GRASP)
 * 
 * Design Patterns:
 * - Singleton (thread-safe)
 * - Facade (упрощает доступ к подсистемам)
 */
class DrvGPU final {
public:
    // Singleton access (thread-safe C++11)
    static DrvGPU& getInstance();
    
    // Запрет копирования и перемещения
    DrvGPU(const DrvGPU&) = delete;
    DrvGPU& operator=(const DrvGPU&) = delete;
    DrvGPU(DrvGPU&&) = delete;
    DrvGPU& operator=(DrvGPU&&) = delete;
    
    /**
     * @brief Инициализация GPU с выбором бэкенда
     * @param backend_type Тип бэкенда (OpenCL, ROCm)
     * @param device_id ID устройства (по умолчанию 0)
     * @throws GPUException если инициализация не удалась
     */
    void initialize(BackendType backend_type, int device_id = 0);
    
    /**
     * @brief Переключение бэкенда в runtime
     * @param backend_type Новый тип бэкенда
     * @note Требует пересоздания всех GPU буферов
     */
    void switchBackend(BackendType backend_type);
    
    /**
     * @brief Получение Memory Manager
     * @return Reference на MemoryManager (Information Expert - GRASP)
     */
    MemoryManager& getMemoryManager();
    
    /**
     * @brief Получение Module Registry
     * @return Reference на ModuleRegistry
     */
    ModuleRegistry& getModuleRegistry();
    
    /**
     * @brief Получение текущего бэкенда
     * @return Reference на IBackend
     */
    IBackend& getBackend();
    
    /**
     * @brief Синхронизация всех операций на GPU
     */
    void synchronize();
    
    /**
     * @brief Получение информации об устройстве
     */
    DeviceInfo getDeviceInfo() const;
    
    /**
     * @brief Проверка инициализации
     */
    bool isInitialized() const noexcept { return initialized_; }
    
    /**
     * @brief Очистка всех ресурсов
     */
    void shutdown();
    
private:
    DrvGPU() = default;  // Приватный конструктор
    ~DrvGPU();
    
    // State
    bool initialized_ = false;
    BackendType current_backend_type_;
    
    // Owned components (Creator - GRASP)
    std::unique_ptr<IBackend> backend_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<ModuleRegistry> module_registry_;
    
    // Thread safety
    mutable std::mutex mutex_;
};

} // namespace core
} // namespace gpu_lib
```

### 2. Backend Abstraction Layer

```cpp
// IBackend.hpp
#pragma once

#include "Types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace gpu_lib {
namespace backend {

/**
 * @brief Абстрактная фабрика для GPU бэкенда
 * 
 * Design Patterns:
 * - Abstract Factory (создание семейства связанных объектов)
 * - Bridge (отделение абстракции от реализации)
 * 
 * Protected Variations (GRASP):
 * - Защищает систему от изменений в конкретных бэкендах
 */
class IBackend {
public:
    virtual ~IBackend() = default;
    
    // Управление контекстом
    virtual void initialize(int device_id) = 0;
    virtual void shutdown() = 0;
    virtual bool isInitialized() const = 0;
    
    // Информация об устройстве
    virtual DeviceInfo getDeviceInfo() const = 0;
    virtual std::string getBackendName() const = 0;
    virtual std::string getBackendVersion() const = 0;
    
    // Управление памятью (Factory Methods)
    virtual std::unique_ptr<IMemoryBuffer> createBuffer(
        size_t size, 
        MemoryFlags flags
    ) = 0;
    
    // Управление вычислениями
    virtual std::unique_ptr<IKernel> createKernel(
        const std::string& source,
        const std::string& kernel_name,
        const std::string& build_options = ""
    ) = 0;
    
    virtual std::unique_ptr<ICommandQueue> createCommandQueue() = 0;
    
    // Синхронизация
    virtual void synchronize() = 0;
    
    // Утилиты
    virtual size_t getMaxWorkGroupSize() const = 0;
    virtual size_t getMaxMemoryAlloc() const = 0;
};

/**
 * @brief Абстракция буфера памяти на GPU
 */
class IMemoryBuffer {
public:
    virtual ~IMemoryBuffer() = default;
    
    virtual void* map(MapFlags flags) = 0;
    virtual void unmap() = 0;
    
    virtual void copyFrom(const void* host_ptr, size_t size, size_t offset = 0) = 0;
    virtual void copyTo(void* host_ptr, size_t size, size_t offset = 0) = 0;
    
    virtual size_t getSize() const = 0;
    virtual void* getNativeHandle() const = 0;  // cl_mem или hipDevicePtr_t
    
    virtual bool isMapped() const = 0;
};

/**
 * @brief Абстракция kernel для вычислений
 */
class IKernel {
public:
    virtual ~IKernel() = default;
    
    // Установка аргументов (type-safe через templates)
    template<typename T>
    void setArg(int index, const T& value) {
        setArgImpl(index, sizeof(T), &value);
    }
    
    void setArgBuffer(int index, IMemoryBuffer& buffer) {
        void* handle = buffer.getNativeHandle();
        setArgImpl(index, sizeof(void*), &handle);
    }
    
    // Запуск kernel
    virtual void execute(
        const WorkSize& global_size,
        const WorkSize& local_size = WorkSize()
    ) = 0;
    
    virtual std::string getName() const = 0;
    
protected:
    virtual void setArgImpl(int index, size_t size, const void* value) = 0;
};

/**
 * @brief Command queue для асинхронного выполнения
 */
class ICommandQueue {
public:
    virtual ~ICommandQueue() = default;
    
    virtual void enqueueKernel(
        IKernel& kernel,
        const WorkSize& global_size,
        const WorkSize& local_size
    ) = 0;
    
    virtual void enqueueCopy(
        IMemoryBuffer& src,
        IMemoryBuffer& dst,
        size_t size
    ) = 0;
    
    virtual void finish() = 0;
    virtual void flush() = 0;
};

} // namespace backend
} // namespace gpu_lib
```

### 3. Backend Factory

```cpp
// BackendFactory.hpp
#pragma once

#include "IBackend.hpp"
#include <memory>

namespace gpu_lib {
namespace backend {

/**
 * @brief Фабрика для создания конкретных бэкендов
 * 
 * Design Patterns:
 * - Factory Method
 * - Strategy (выбор реализации)
 */
class BackendFactory {
public:
    /**
     * @brief Создание бэкенда по типу
     * @param type Тип бэкенда (OpenCL, ROCm)
     * @return Unique pointer на IBackend
     * @throws GPUException если тип не поддерживается
     */
    static std::unique_ptr<IBackend> create(BackendType type);
    
    /**
     * @brief Проверка доступности бэкенда
     */
    static bool isBackendAvailable(BackendType type);
    
    /**
     * @brief Получение списка доступных бэкендов
     */
    static std::vector<BackendType> getAvailableBackends();
};

/**
 * @brief Реализация для OpenCL
 */
class OpenCLBackend : public IBackend {
public:
    void initialize(int device_id) override;
    void shutdown() override;
    bool isInitialized() const override;
    
    DeviceInfo getDeviceInfo() const override;
    std::string getBackendName() const override { return "OpenCL"; }
    std::string getBackendVersion() const override;
    
    std::unique_ptr<IMemoryBuffer> createBuffer(
        size_t size, 
        MemoryFlags flags
    ) override;
    
    std::unique_ptr<IKernel> createKernel(
        const std::string& source,
        const std::string& kernel_name,
        const std::string& build_options
    ) override;
    
    std::unique_ptr<ICommandQueue> createCommandQueue() override;
    
    void synchronize() override;
    
    size_t getMaxWorkGroupSize() const override;
    size_t getMaxMemoryAlloc() const override;
    
private:
    // OpenCL specific members
    cl_context context_ = nullptr;
    cl_device_id device_id_ = nullptr;
    cl_command_queue queue_ = nullptr;
    bool initialized_ = false;
};

/**
 * @brief Реализация для ROCm (будущая)
 */
class ROCmBackend : public IBackend {
    // Аналогичная структура для ROCm/HIP
    // TODO: Implement when ROCm support is added
};

} // namespace backend
} // namespace gpu_lib
```

### 4. Memory Manager

```cpp
// MemoryManager.hpp
#pragma once

#include "IBackend.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace gpu_lib {
namespace core {

/**
 * @brief RAII wrapper для GPU памяти
 * 
 * Design Patterns:
 * - RAII (Resource Acquisition Is Initialization)
 * - Smart Pointer semantics
 */
class GPUMemoryHandle {
public:
    GPUMemoryHandle(std::shared_ptr<backend::IMemoryBuffer> buffer);
    ~GPUMemoryHandle() = default;
    
    // Movable but not copyable
    GPUMemoryHandle(GPUMemoryHandle&&) = default;
    GPUMemoryHandle& operator=(GPUMemoryHandle&&) = default;
    GPUMemoryHandle(const GPUMemoryHandle&) = delete;
    GPUMemoryHandle& operator=(const GPUMemoryHandle&) = delete;
    
    backend::IMemoryBuffer* get() { return buffer_.get(); }
    const backend::IMemoryBuffer* get() const { return buffer_.get(); }
    
    backend::IMemoryBuffer* operator->() { return buffer_.get(); }
    const backend::IMemoryBuffer* operator->() const { return buffer_.get(); }
    
    size_t size() const { return buffer_->getSize(); }
    
private:
    std::shared_ptr<backend::IMemoryBuffer> buffer_;
};

/**
 * @brief Стратегия выделения памяти
 * 
 * Design Pattern: Strategy
 */
class IAllocationStrategy {
public:
    virtual ~IAllocationStrategy() = default;
    
    virtual std::shared_ptr<backend::IMemoryBuffer> allocate(
        backend::IBackend& backend,
        size_t size,
        MemoryFlags flags
    ) = 0;
    
    virtual void deallocate(std::shared_ptr<backend::IMemoryBuffer> buffer) = 0;
    virtual void defragment() = 0;
};

/**
 * @brief Memory Manager для централизованного управления памятью GPU
 * 
 * Responsibilities (SRP):
 * - Выделение и освобождение GPU памяти
 * - Pool management для переиспользования
 * - Дефрагментация
 * - Статистика использования
 * 
 * Design Patterns:
 * - Object Pool (для переиспользования буферов)
 * - Strategy (разные стратегии выделения)
 */
class MemoryManager {
public:
    explicit MemoryManager(backend::IBackend& backend);
    ~MemoryManager();
    
    /**
     * @brief Установка стратегии выделения памяти
     */
    void setAllocationStrategy(std::unique_ptr<IAllocationStrategy> strategy);
    
    /**
     * @brief Выделение памяти на GPU
     * @param size Размер в байтах
     * @param flags Флаги памяти (read/write/etc)
     * @return RAII handle на память
     */
    GPUMemoryHandle allocate(size_t size, MemoryFlags flags = MemoryFlags::ReadWrite);
    
    /**
     * @brief Выделение из пула (для часто используемых размеров)
     */
    GPUMemoryHandle allocateFromPool(size_t size, MemoryFlags flags = MemoryFlags::ReadWrite);
    
    /**
     * @brief Создание shared буфера для нескольких модулей
     * @param name Уникальное имя буфера
     * @param size Размер
     * @return Handle на shared память
     */
    GPUMemoryHandle allocateShared(const std::string& name, size_t size);
    
    /**
     * @brief Получение shared буфера по имени
     */
    GPUMemoryHandle getShared(const std::string& name);
    
    /**
     * @brief Дефрагментация памяти (вызывается вручную или автоматически)
     */
    void defragment();
    
    /**
     * @brief Статистика использования памяти
     */
    struct MemoryStats {
        size_t total_allocated;
        size_t total_available;
        size_t peak_usage;
        size_t pool_size;
        size_t shared_buffers_count;
    };
    
    MemoryStats getStats() const;
    
    /**
     * @brief Очистка всех буферов (осторожно!)
     */
    void clear();
    
private:
    backend::IBackend& backend_;
    std::unique_ptr<IAllocationStrategy> strategy_;
    
    // Pool для переиспользования
    struct MemoryPool {
        std::vector<std::shared_ptr<backend::IMemoryBuffer>> available;
        std::vector<std::weak_ptr<backend::IMemoryBuffer>> in_use;
    };
    
    std::unordered_map<size_t, MemoryPool> pools_;  // size -> pool
    
    // Shared buffers между модулями
    std::unordered_map<std::string, std::shared_ptr<backend::IMemoryBuffer>> shared_buffers_;
    
    // Statistics
    mutable MemoryStats stats_;
    mutable std::mutex mutex_;
    
    // Helper methods
    void updateStats();
    void cleanupPool();
};

/**
 * @brief Простая стратегия - прямое выделение
 */
class DirectAllocationStrategy : public IAllocationStrategy {
public:
    std::shared_ptr<backend::IMemoryBuffer> allocate(
        backend::IBackend& backend,
        size_t size,
        MemoryFlags flags
    ) override;
    
    void deallocate(std::shared_ptr<backend::IMemoryBuffer> buffer) override;
    void defragment() override {}  // No-op для direct allocation
};

/**
 * @brief Стратегия с pool и дефрагментацией
 */
class PooledAllocationStrategy : public IAllocationStrategy {
    // TODO: Implement pooling logic
};

} // namespace core
} // namespace gpu_lib
```

### 5. Compute Module Base

```cpp
// IComputeModule.hpp
#pragma once

#include "MemoryManager.hpp"
#include "IBackend.hpp"
#include <string>
#include <memory>

namespace gpu_lib {
namespace modules {

/**
 * @brief Базовый интерфейс для всех вычислительных модулей
 * 
 * Design Patterns:
 * - Template Method (жизненный цикл модуля)
 * - Strategy (разные реализации compute())
 * 
 * GRASP:
 * - High Cohesion (одна функциональная область)
 * - Low Coupling (зависит только от абстракций)
 */
class IComputeModule {
public:
    virtual ~IComputeModule() = default;
    
    /**
     * @brief Получение имени модуля
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Получение версии модуля
     */
    virtual std::string getVersion() const = 0;
    
    /**
     * @brief Инициализация модуля (вызывается один раз)
     * @param backend Backend для создания kernels
     * @param memory_manager MemoryManager для выделения памяти
     */
    virtual void initialize(
        backend::IBackend& backend,
        core::MemoryManager& memory_manager
    ) = 0;
    
    /**
     * @brief Проверка инициализации
     */
    virtual bool isInitialized() const = 0;
    
    /**
     * @brief Освобождение ресурсов
     */
    virtual void shutdown() = 0;
    
    /**
     * @brief Получение списка зависимостей (имена других модулей)
     */
    virtual std::vector<std::string> getDependencies() const {
        return {};  // По умолчанию нет зависимостей
    }
};

/**
 * @brief Базовый класс для модулей с Template Method pattern
 */
class ComputeModuleBase : public IComputeModule {
public:
    explicit ComputeModuleBase(const std::string& name, const std::string& version)
        : name_(name), version_(version), initialized_(false) {}
    
    virtual ~ComputeModuleBase() = default;
    
    std::string getName() const override { return name_; }
    std::string getVersion() const override { return version_; }
    bool isInitialized() const override { return initialized_; }
    
    // Template Method
    void initialize(
        backend::IBackend& backend,
        core::MemoryManager& memory_manager
    ) override final {
        if (initialized_) {
            throw std::runtime_error("Module already initialized");
        }
        
        backend_ = &backend;
        memory_manager_ = &memory_manager;
        
        // Hook method для переопределения
        onInitialize();
        
        initialized_ = true;
    }
    
    void shutdown() override final {
        if (!initialized_) return;
        
        // Hook method
        onShutdown();
        
        initialized_ = false;
        backend_ = nullptr;
        memory_manager_ = nullptr;
    }
    
protected:
    // Hook methods для переопределения в наследниках
    virtual void onInitialize() = 0;
    virtual void onShutdown() = 0;
    
    // Protected доступ к ресурсам
    backend::IBackend& backend() { 
        if (!backend_) throw std::runtime_error("Backend not initialized");
        return *backend_; 
    }
    
    core::MemoryManager& memoryManager() { 
        if (!memory_manager_) throw std::runtime_error("MemoryManager not initialized");
        return *memory_manager_; 
    }
    
    bool isInitialized() { return initialized_; }
    
private:
    std::string name_;
    std::string version_;
    bool initialized_;
    
    backend::IBackend* backend_ = nullptr;
    core::MemoryManager* memory_manager_ = nullptr;
};

} // namespace modules
} // namespace gpu_lib
```

### 6. Пример конкретного модуля: FFT Post-Processing

```cpp
// FFTPostProcessing.hpp
#pragma once

#include "IComputeModule.hpp"
#include <vector>

namespace gpu_lib {
namespace modules {

/**
 * @brief Модуль для обработки результатов FFT
 * 
 * Функциональность:
 * - Поиск 3 максимумов
 * - Расчет параболы для уточнения пиков
 * - Поиск точек перегиба
 * - Поиск всех локальных максимумов
 * 
 * Information Expert (GRASP):
 * - Имеет всю информацию о FFT данных для обработки
 */
class FFTPostProcessing : public ComputeModuleBase {
public:
    FFTPostProcessing();
    
    struct PeakInfo {
        float frequency;      // Частота пика
        float magnitude;      // Амплитуда
        float phase;          // Фаза
        int bin_index;        // Индекс бина
        float interpolated_freq;  // Уточненная частота через параболу
    };
    
    struct Config {
        bool enable_parabolic_interpolation = true;
        float threshold = 0.0f;  // Порог для поиска пиков
        int max_peaks = 3;       // Максимум пиков для поиска
    };
    
    /**
     * @brief Поиск топ-N максимумов
     * @param fft_data Buffer с FFT данными (комплексные числа)
     * @param size Размер FFT
     * @param config Конфигурация
     * @return Вектор найденных пиков
     */
    std::vector<PeakInfo> findTopPeaks(
        const core::GPUMemoryHandle& fft_data,
        size_t size,
        const Config& config
    );
    
    /**
     * @brief Поиск всех локальных максимумов выше порога
     */
    std::vector<PeakInfo> findAllPeaks(
        const core::GPUMemoryHandle& fft_data,
        size_t size,
        float threshold
    );
    
    /**
     * @brief Уточнение частоты через параболическую интерполяцию
     */
    float refineFrequency(
        const core::GPUMemoryHandle& fft_data,
        int peak_index,
        float sample_rate
    );
    
    std::vector<std::string> getDependencies() const override {
        return {};  // Нет зависимостей
    }
    
protected:
    void onInitialize() override;
    void onShutdown() override;
    
private:
    // Kernels
    std::unique_ptr<backend::IKernel> magnitude_kernel_;
    std::unique_ptr<backend::IKernel> find_peaks_kernel_;
    std::unique_ptr<backend::IKernel> parabolic_interp_kernel_;
    
    // Temporary buffers
    core::GPUMemoryHandle magnitude_buffer_;
    core::GPUMemoryHandle peaks_buffer_;
    
    void compileKernels();
    void createBuffers(size_t max_size);
};

} // namespace modules
} // namespace gpu_lib
```

### 7. Module Registry

```cpp
// ModuleRegistry.hpp
#pragma once

#include "IComputeModule.hpp"
#include <memory>
#include <unordered_map>
#include <functional>

namespace gpu_lib {
namespace core {

/**
 * @brief Registry для управления модулями
 * 
 * Design Patterns:
 * - Registry (централизованное хранилище)
 * - Factory (создание модулей)
 * 
 * GRASP:
 * - Controller (координация модулей)
 * - Low Coupling (модули не знают друг о друге напрямую)
 */
class ModuleRegistry {
public:
    explicit ModuleRegistry(backend::IBackend& backend, MemoryManager& memory_manager);
    
    using ModuleFactory = std::function<std::unique_ptr<modules::IComputeModule>()>;
    
    /**
     * @brief Регистрация фабрики для типа модуля
     * @param name Имя модуля
     * @param factory Фабричная функция
     */
    void registerModule(const std::string& name, ModuleFactory factory);
    
    /**
     * @brief Создание и инициализация модуля
     * @param name Имя модуля
     * @return Shared pointer на модуль
     */
    std::shared_ptr<modules::IComputeModule> createModule(const std::string& name);
    
    /**
     * @brief Получение уже созданного модуля (singleton per registry)
     * @param name Имя модуля
     * @return Shared pointer на модуль или nullptr
     */
    std::shared_ptr<modules::IComputeModule> getModule(const std::string& name);
    
    /**
     * @brief Получение или создание модуля
     */
    std::shared_ptr<modules::IComputeModule> getOrCreateModule(const std::string& name);
    
    /**
     * @brief Список зарегистрированных модулей
     */
    std::vector<std::string> getRegisteredModules() const;
    
    /**
     * @brief Список созданных модулей
     */
    std::vector<std::string> getCreatedModules() const;
    
    /**
     * @brief Shutdown всех модулей
     */
    void shutdownAll();
    
private:
    backend::IBackend& backend_;
    MemoryManager& memory_manager_;
    
    std::unordered_map<std::string, ModuleFactory> factories_;
    std::unordered_map<std::string, std::shared_ptr<modules::IComputeModule>> modules_;
    
    mutable std::mutex mutex_;
    
    void resolveDependencies(const std::string& module_name);
};

/**
 * @brief Helper для автоматической регистрации модулей
 * 
 * Использование:
 * static ModuleRegistrar<FFTPostProcessing> fft_registrar("FFTPostProcessing");
 */
template<typename ModuleType>
class ModuleRegistrar {
public:
    explicit ModuleRegistrar(const std::string& name) {
        // Static registration при запуске программы
        // Альтернатива: manual registration в main()
        GlobalModuleFactory::instance().register(
            name,
            []() { return std::make_unique<ModuleType>(); }
        );
    }
};

} // namespace core
} // namespace gpu_lib
```

---

## Система сборки и тестирования

### CMake Structure

```
LibGPU/
├── CMakeLists.txt                    # Root CMake
├── cmake/
│   ├── FindOpenCL.cmake
│   ├── FindROCm.cmake
│   └── CompilerWarnings.cmake
│
├── core/                             # DrvGPU Core
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── gpu_lib/
│   │       ├── DrvGPU.hpp
│   │       ├── MemoryManager.hpp
│   │       ├── ModuleRegistry.hpp
│   │       └── IBackend.hpp
│   ├── src/
│   │   ├── DrvGPU.cpp
│   │   ├── MemoryManager.cpp
│   │   ├── ModuleRegistry.cpp
│   │   └── BackendFactory.cpp
│   └── tests/                        # Core tests
│       ├── CMakeLists.txt
│       └── test_memory_manager.cpp
│
├── backends/
│   ├── opencl/                       # OpenCL Backend
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── gpu_lib/backend/
│   │   │       └── OpenCLBackend.hpp
│   │   ├── src/
│   │   │   └── OpenCLBackend.cpp
│   │   └── tests/
│   │       └── test_opencl_backend.cpp
│   │
│   └── rocm/                         # ROCm Backend (future)
│       └── CMakeLists.txt
│
├── modules/
│   ├── fft_postproc/                 # FFT Post-Processing Module
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   └── gpu_lib/modules/
│   │   │       └── FFTPostProcessing.hpp
│   │   ├── src/
│   │   │   └── FFTPostProcessing.cpp
│   │   ├── kernels/
│   │   │   └── fft_postproc.cl
│   │   └── tests/
│   │       └── test_fft_postproc.cpp
│   │
│   ├── fractional_delay/             # Fractional Delay Module
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   ├── src/
│   │   ├── kernels/
│   │   └── tests/
│   │
│   ├── heterodyne/                   # Heterodyne Module
│   │   └── ...
│   │
│   └── signal_stats/                 # Signal Statistics Module
│       └── ...
│
├── examples/
│   ├── CMakeLists.txt
│   ├── basic_usage.cpp
│   └── pipeline_example.cpp
│
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── module_development.md
│
└── tests/
    ├── CMakeLists.txt                # Integration tests
    └── integration/
        └── test_full_pipeline.cpp
```

### Root CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(LibGPU VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(LIBGPU_BUILD_TESTS "Build unit tests" ON)
option(LIBGPU_BUILD_EXAMPLES "Build examples" ON)
option(LIBGPU_BUILD_DOCS "Build documentation" OFF)
option(LIBGPU_ENABLE_OPENCL "Enable OpenCL backend" ON)
option(LIBGPU_ENABLE_ROCM "Enable ROCm backend" OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Global settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find dependencies
if(LIBGPU_ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
    message(STATUS "OpenCL found: ${OpenCL_VERSION_STRING}")
endif()

if(LIBGPU_ENABLE_ROCM)
    find_package(ROCm REQUIRED)
    message(STATUS "ROCm found: ${ROCm_VERSION}")
endif()

# Testing
if(LIBGPU_BUILD_TESTS)
    enable_testing()
    # Use Google Test or Catch2
    find_package(GTest REQUIRED)
    include(GoogleTest)
endif()

# Compiler warnings
include(CompilerWarnings)

# Subdirectories
add_subdirectory(core)
add_subdirectory(backends)
add_subdirectory(modules)

if(LIBGPU_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(LIBGPU_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(LIBGPU_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Install
include(GNUInstallDirs)
install(DIRECTORY core/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    LibGPUConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/LibGPUConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibGPU
)
```

### Module CMakeLists.txt Example

```cmake
# modules/fft_postproc/CMakeLists.txt

# Module library
add_library(gpu_fft_postproc
    src/FFTPostProcessing.cpp
    src/FFTKernels.cpp
)

target_include_directories(gpu_fft_postproc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(gpu_fft_postproc
    PUBLIC
        gpu_core
    PRIVATE
        gpu_backend_opencl
)

# Compile OpenCL kernels to header (optional)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fft_kernels_cl.h
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fft_postproc.cl
        ${CMAKE_CURRENT_BINARY_DIR}/fft_kernels_cl.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fft_postproc.cl
)

target_sources(gpu_fft_postproc PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/fft_kernels_cl.h
)

# Tests (conditional)
if(LIBGPU_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Install
install(TARGETS gpu_fft_postproc
    EXPORT LibGPUTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

### Test CMakeLists.txt

```cmake
# modules/fft_postproc/tests/CMakeLists.txt

# Only build if testing is enabled
if(NOT LIBGPU_BUILD_TESTS)
    return()
endif()

# Test executable
add_executable(test_fft_postproc
    test_fft_postproc.cpp
    test_peak_finding.cpp
    test_parabolic_interp.cpp
)

target_link_libraries(test_fft_postproc
    PRIVATE
        gpu_fft_postproc
        GTest::gtest
        GTest::gtest_main
)

# Register with CTest
gtest_discover_tests(test_fft_postproc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES
        LABELS "unit;fft_postproc"
)

# Optional: Performance benchmarks (не включаются в обычные тесты)
if(LIBGPU_BUILD_BENCHMARKS)
    add_executable(bench_fft_postproc
        bench_fft_postproc.cpp
    )
    
    target_link_libraries(bench_fft_postproc
        PRIVATE
            gpu_fft_postproc
            benchmark::benchmark
    )
endif()
```

### Conditional Compilation Pattern

```cpp
// В коде модуля - опциональная компиляция тест-утилит

#ifdef LIBGPU_ENABLE_TEST_UTILS
namespace gpu_lib {
namespace modules {
namespace test_utils {

/**
 * @brief Утилиты для тестирования (только в test builds)
 */
class FFTTestUtils {
public:
    static std::vector<float> generateSineWave(
        float frequency, 
        float sample_rate, 
        size_t num_samples
    );
    
    static std::vector<std::complex<float>> computeFFT(
        const std::vector<float>& signal
    );
    
    static void validatePeaks(
        const std::vector<FFTPostProcessing::PeakInfo>& peaks,
        const std::vector<float>& expected_frequencies,
        float tolerance
    );
};

} // namespace test_utils
} // namespace modules
} // namespace gpu_lib
#endif // LIBGPU_ENABLE_TEST_UTILS
```

---

## Стратегия расширения

### Добавление нового модуля

**Шаг 1: Создание структуры**
```bash
mkdir -p modules/new_module/{include/gpu_lib/modules,src,kernels,tests}
```

**Шаг 2: Создание заголовка**
```cpp
// modules/new_module/include/gpu_lib/modules/NewModule.hpp
#pragma once
#include "IComputeModule.hpp"

namespace gpu_lib {
namespace modules {

class NewModule : public ComputeModuleBase {
public:
    NewModule() : ComputeModuleBase("NewModule", "1.0.0") {}
    
    // Ваш функционал
    void yourFunction(const core::GPUMemoryHandle& input);
    
protected:
    void onInitialize() override;
    void onShutdown() override;
    
private:
    std::unique_ptr<backend::IKernel> kernel_;
};

} // namespace modules
} // namespace gpu_lib
```

**Шаг 3: Создание CMakeLists.txt**
```cmake
add_library(gpu_new_module
    src/NewModule.cpp
)

target_link_libraries(gpu_new_module
    PUBLIC gpu_core
)

if(LIBGPU_BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

**Шаг 4: Регистрация в родительском CMakeLists.txt**
```cmake
# modules/CMakeLists.txt
add_subdirectory(new_module)
```

**Шаг 5: Регистрация модуля**
```cpp
// В main() или при инициализации
auto& registry = DrvGPU::getInstance().getModuleRegistry();
registry.registerModule("NewModule", []() {
    return std::make_unique<NewModule>();
});
```

### Добавление нового бэкенда (ROCm example)

**Шаг 1: Реализация интерфейса**
```cpp
// backends/rocm/include/gpu_lib/backend/ROCmBackend.hpp
class ROCmBackend : public IBackend {
    // Implement all virtual methods
};
```

**Шаг 2: Обновление BackendFactory**
```cpp
std::unique_ptr<IBackend> BackendFactory::create(BackendType type) {
    switch(type) {
        case BackendType::OpenCL:
            return std::make_unique<OpenCLBackend>();
        case BackendType::ROCm:
            return std::make_unique<ROCmBackend>();  // New
        default:
            throw GPUException("Unknown backend type");
    }
}
```

**Шаг 3: Переключение в runtime**
```cpp
auto& drv = DrvGPU::getInstance();

// Инициализация с OpenCL
drv.initialize(BackendType::OpenCL);

// ... работа ...

// Переключение на ROCm
drv.switchBackend(BackendType::ROCm);  // Все модули пересоздадут kernels
```

---

## Дорожная карта разработки

### Phase 1: Foundation (Месяцы 1-2)

**Week 1-2: Core Infrastructure**
- [ ] Базовая структура проекта и CMake
- [ ] IBackend интерфейс
- [ ] OpenCLBackend основная реализация
- [ ] Базовые unit тесты для Backend

**Week 3-4: Memory Management**
- [ ] MemoryManager базовая реализация
- [ ] GPUMemoryHandle (RAII wrapper)
- [ ] DirectAllocationStrategy
- [ ] Memory pool basics
- [ ] Тесты для Memory Manager

**Week 5-6: DrvGPU Core**
- [ ] DrvGPU singleton
- [ ] Integration Backend + Memory
- [ ] ModuleRegistry базовая версия
- [ ] IComputeModule интерфейс
- [ ] ComputeModuleBase

**Week 7-8: Build System & Testing**
- [ ] Полный CMake setup
- [ ] Google Test integration
- [ ] Conditional test compilation
- [ ] CI/CD basics (GitHub Actions)
- [ ] Documentation setup

### Phase 2: First Modules (Месяцы 3-4)

**Week 9-10: FFT Post-Processing Module**
- [ ] FFTPostProcessing класс
- [ ] Kernels для поиска пиков
- [ ] Parabolic interpolation
- [ ] Unit тесты с валидацией
- [ ] Performance benchmarks

**Week 11-12: Signal Statistics Module**
- [ ] Median calculation
- [ ] Mean calculation
- [ ] Magnitude computation
- [ ] Тесты и benchmarks

**Week 13-14: Integration & Examples**
- [ ] Pipeline builder
- [ ] Example applications
- [ ] Integration tests
- [ ] Performance profiling

**Week 15-16: Documentation & Polish**
- [ ] API documentation (Doxygen)
- [ ] Architecture diagrams
- [ ] Module development guide
- [ ] Code review and refactoring

### Phase 3: Advanced Features (Месяцы 5-6)

**Week 17-18: Fractional Delay Module**
- [ ] FractionalDelay implementation
- [ ] Farrow structure kernels
- [ ] Тесты с аудио данными

**Week 19-20: Heterodyne Module**
- [ ] Heterodyne processing
- [ ] NCO (Numerically Controlled Oscillator)
- [ ] Mixer kernels
- [ ] Тесты

**Week 21-22: Advanced Memory Management**
- [ ] PooledAllocationStrategy
- [ ] Defragmentation algorithms
- [ ] Shared buffers optimization
- [ ] Memory statistics и profiling

**Week 23-24: Optimization & Tuning**
- [ ] Performance profiling всех модулей
- [ ] Kernel optimization
- [ ] Pipeline optimization
- [ ] Асинхронное выполнение

### Phase 4: ROCm Support (Месяцы 7-8)

**Week 25-26: ROCm Backend**
- [ ] ROCmBackend implementation
- [ ] HIP kernel compilation
- [ ] Memory management для ROCm
- [ ] Тесты

**Week 27-28: Runtime Switching**
- [ ] Backend switching mechanism
- [ ] State migration OpenCL ↔ ROCm
- [ ] Тесты переключения
- [ ] Performance comparison

**Week 29-30: Compatibility & Testing**
- [ ] Cross-backend тесты
- [ ] Validation всех модулей на ROCm
- [ ] Performance benchmarks
- [ ] Bug fixes

**Week 31-32: Documentation & Release**
- [ ] Обновление документации
- [ ] Migration guide OpenCL → ROCm
- [ ] Release notes
- [ ] Version 2.0 release

### Phase 5: Production Readiness (Месяц 9+)

**Continuous:**
- [ ] Bug fixes и patches
- [ ] Performance optimization
- [ ] New modules по требованию
- [ ] Community support
- [ ] Security audits
- [ ] Новые backend (CUDA, SYCL, Metal?)

---

## Критерии качества

### Code Quality Metrics

1. **Test Coverage**: Минимум 80% для Core, 70% для Modules
2. **Documentation**: 100% public API documented
3. **Code Review**: All PRs reviewed by 2+ developers
4. **Static Analysis**: 0 критических предупреждений (cppcheck, clang-tidy)
5. **Performance**: Benchmarks для всех модулей

### Design Quality Checklist

- [ ] Все классы следуют Single Responsibility
- [ ] Зависимости на абстракции (DIP)
- [ ] Модули легко тестируются в изоляции
- [ ] Нет циклических зависимостей
- [ ] Публичный API минимален и стабилен
- [ ] Внутренние изменения не ломают пользовательский код

### Performance Requirements

- **Memory Allocation**: < 1ms для буферов < 1MB
- **Kernel Launch Overhead**: < 100μs
- **Backend Switching**: < 1s для полной миграции
- **Module Initialization**: < 100ms
- **Zero-copy Operations**: Где возможно между модулями

---

## Заключение

Эта архитектура обеспечивает:

✅ **Модульность**: Независимые библиотеки с четкими границами  
✅ **Расширяемость**: Простое добавление модулей и бэкендов  
✅ **Тестируемость**: Полное покрытие тестами с условной компиляцией  
✅ **Производительность**: Эффективное управление памятью GPU  
✅ **Долгосрочность**: SOLID/GRASP/GoF принципы для maintainability  
✅ **Гибкость**: Runtime переключение OpenCL ↔ ROCm  

Архитектура готова к долгосрочному развитию и легко адаптируется к новым требованиям!
