# DrvGPU — C4 Model (C1, C2, C3, C4)

> **Дата**: 2026-02-23  
> **Источник**: По образцу [Disane C4.md](Disane%20C4.md), адаптировано под GPUWorkLib/DrvGPU  
> **Референс**: [c4model.com](https://c4model.com)

---

## C1: Context (Системный контекст)

**Система GPUWorkLib**:

- **Engineer** (разработчик/инженер) использует C++ приложение или Python-скрипт.
- **GPUWorkLib System** принимает данные, выполняет ЦОС на GPU (FFT, фильтры, генерация сигналов) и возвращает результаты.
- Взаимодействует с **GPU Hardware** (AMD/NVIDIA/Intel) через драйверы.
- Опционально: **configGPU.json** (конфигурация), **Logs/** (логирование).

На уровне C1 — один прямоугольник «GPUWorkLib System» и стрелки к внешним системам.

---

## C2: Containers (Контейнеры)

Внутри GPUWorkLib System:

| Контейнер           | Технология | Назначение                                      |
|---------------------|------------|--------------------------------------------------|
| **DrvGPU Core**     | C++        | Драйвер GPU, бэкенды, память, очереди           |
| **Compute Modules**  | C++        | FFT, фильтры, генераторы, гетеродин, LCH Farrow |
| **Python Bindings**  | pybind11   | Python API (gpuworklib)                          |
| **Main Application** | C++ exe    | Точка входа, тесты, примеры                      |

Далее рассматривается **DrvGPU Core**.

---

## C3: Components (Компоненты внутри DrvGPU Core)

| Компонент        | Файлы                         | Назначение                                      |
|------------------|-------------------------------|-------------------------------------------------|
| **DrvGPU**       | drv_gpu.hpp/cpp               | Фасад, единая точка входа                       |
| **GPUManager**   | gpu_manager.hpp               | Multi-GPU, балансировка нагрузки                |
| **Backend Layer**| IBackend, OpenCLBackend, ROCmBackend | Абстракция GPU API                      |
| **MemoryManager**| memory_manager.hpp/cpp        | Буферы, GPUBuffer, SVMBuffer                    |
| **ModuleRegistry**| module_registry.hpp/cpp       | Регистр compute-модулей                         |
| **Services**     | GPUProfiler, BatchManager, KernelCache, FilterConfig | Профилирование, batch, кеш ядер |
| **Logger**       | logger.hpp, config_logger     | Логирование (plog, per-GPU)                     |

Далее — **C4** для компонента **DrvGPU** (фасад).

---

## C4: Code (DrvGPU — уровень кода)

### Диаграмма классов (упрощённая)

```
┌─────────────────────────────────────────────────────────────────┐
│                         DrvGPU                                   │
├─────────────────────────────────────────────────────────────────┤
│ + DrvGPU(BackendType, int device_index)                          │
│ + Initialize()                                                   │
│ + Cleanup()                                                      │
│ + GetMemoryManager() → MemoryManager&                            │
│ + GetModuleRegistry() → ModuleRegistry&                         │
│ + GetBackend() → IBackend&                                       │
│ + Synchronize() / Flush()                                        │
│ + GetDeviceInfo() → GPUDeviceInfo                                │
├─────────────────────────────────────────────────────────────────┤
│ - CreateBackend()                                                │
│ - InitializeSubsystems()                                         │
└──────────────────────────┬──────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   IBackend*     │ │ MemoryManager*  │ │ ModuleRegistry*  │
│   (backend_)    │ │ (memory_mgr_)   │ │ (module_reg_)    │
└────────┬────────┘ └─────────────────┘ └─────────────────┘
         │
         ├──► OpenCLBackend
         ├──► ROCmBackend
         └──► (CUDA planned)
```

### Псевдо-C++ (структура DrvGPU)

```cpp
class DrvGPU {
public:
    explicit DrvGPU(BackendType backend_type, int device_index = 0);
    ~DrvGPU();
    
    void Initialize();
    void Cleanup();
    bool IsInitialized() const;
    
    GPUDeviceInfo GetDeviceInfo() const;
    MemoryManager& GetMemoryManager();
    ModuleRegistry& GetModuleRegistry();
    IBackend& GetBackend();
    
    void Synchronize();
    void Flush();

private:
    void CreateBackend();        // Factory: создаёт OpenCL/ROCm/CUDA backend
    void InitializeSubsystems(); // Инициализирует MemoryManager, ModuleRegistry
    
    BackendType backend_type_;
    int device_index_;
    bool initialized_;
    std::unique_ptr<IBackend> backend_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<ModuleRegistry> module_registry_;
    mutable std::mutex mutex_;
};
```

### Sequence diagram: `DrvGPU::Initialize()`

```
Caller          DrvGPU              CreateBackend()      IBackend::Initialize()
  │                 │                      │                        │
  │──Initialize()──►│                      │                        │
  │                 │──CreateBackend()────►│                        │
  │                 │                      │──new OpenCLBackend()──►│
  │                 │                      │◄──────────────────────│
  │                 │◄─────────────────────│                        │
  │                 │──Initialize(device_index)───────────────────►│
  │                 │                      │         (clGetPlatform, clCreateContext...)
  │                 │◄─────────────────────────────────────────────│
  │                 │──InitializeSubsystems()                       │
  │                 │   (MemoryManager, ModuleRegistry)             │
  │                 │                                               │
  │◄────────────────│                                               │
```

### C4 для Backend Layer (IBackend → OpenCLBackend)

```cpp
// Интерфейс (Bridge Pattern)
class IBackend {
public:
    virtual void Initialize(int device_index) = 0;
    virtual void* Allocate(size_t size_bytes) = 0;
    virtual void Free(void* ptr) = 0;
    virtual void MemcpyHostToDevice(void* dst, const void* src, size_t n) = 0;
    virtual void MemcpyDeviceToHost(void* dst, const void* src, size_t n) = 0;
    virtual void Synchronize() = 0;
    virtual BackendType GetType() const = 0;
    virtual GPUDeviceInfo GetDeviceInfo() const = 0;
    // ... GetNativeContext, GetNativeDevice, GetNativeQueue
};

// Реализация OpenCL
class OpenCLBackend : public IBackend {
    // Внутри: OpenCLCore (cl_context, cl_device_id, cl_command_queue)
    //         CommandQueuePool, MemoryManager
};
```

---

## Связь уровней C1–C4

| Уровень | Объект |
|---------|--------|
| **C1** | GPUWorkLib System обрабатывает сигналы для Engineer |
| **C2** | DrvGPU Core — контейнер GPU-драйвера |
| **C3** | DrvGPU (фасад), Backend Layer, MemoryManager и др. |
| **C4** | Классы `DrvGPU`, `IBackend`, `OpenCLBackend`, методы `Initialize()`, `CreateBackend()` и т.п. |

---

## PlantUML для C4 (опционально)

```plantuml
@startuml DrvGPU_C4
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Context.puml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Component.puml

Person(engineer, "Engineer", "C++/Python developer")
System(gpuworklib, "GPUWorkLib System", "GPU signal processing")
System_Ext(gpu, "GPU Hardware", "AMD/NVIDIA/Intel")

Rel(engineer, gpuworklib, "Uses")
Rel(gpuworklib, gpu, "Runs on")

Container_Boundary(drvgpu_core, "DrvGPU Core") {
    Component(drvgpu, "DrvGPU", "Facade")
    Component(backend, "IBackend", "Bridge")
    Component(memory, "MemoryManager", "Buffers")
    Component(registry, "ModuleRegistry", "Modules")
}
Rel(drvgpu, backend, "uses")
Rel(drvgpu, memory, "uses")
Rel(drvgpu, registry, "uses")
@enduml
```

---

## Ссылки

- [GPUWorkLib_Design_C4_Full.md](GPUWorkLib_Design_C4_Full.md) — C4 для всего проекта
- [Doc/DrvGPU/Architecture.md](../Doc/DrvGPU/Architecture.md) — архитектура DrvGPU
- [Disane C4.md](Disane%20C4.md) — исходный пример C4
- [c4model.com](https://c4model.com) — C4 Model
