# DrvGPU - Multi-GPU Abstraction Library

**Версия:** 1.0.0  
**Дата:** 2026-01-31  
**Лицензия:** MIT  

---

## 📖 Описание

**DrvGPU** - это современная C++ библиотека для работы с GPU, предоставляющая единый интерфейс для различных backend'ов (OpenCL, CUDA, Vulkan) с полной поддержкой Multi-GPU сценариев.

### Ключевые особенности

✅ **Multi-GPU Support** - работа с несколькими GPU через единый интерфейс  
✅ **Backend Abstraction** - единый API для OpenCL, CUDA, Vulkan  
✅ **НЕ Singleton!** - каждая GPU имеет свой экземпляр DrvGPU  
✅ **GPUManager** - централизованное управление Multi-GPU  
✅ **Load Balancing** - Round-Robin, Least Loaded, Manual  
✅ **RAII** - автоматическое управление ресурсами  
✅ **Thread-Safe** - безопасный многопоточный доступ  
✅ **Memory Manager** - высокоуровневое управление памятью  
✅ **Module Registry** - регистр compute модулей (FFT, Matrix, etc.)  

---

## 🚀 Quick Start

### Установка

```bash
# Клонировать репозиторий
git clone https://github.com/your-org/DrvGPU.git
cd DrvGPU

# Сборка
mkdir build && cd build
cmake ..
make -j$(nproc)

# Установка
sudo make install
```

### Пример: Single GPU

```cpp
#include "drv_gpu.hpp"
#include "backend_type.hpp"

using namespace drv_gpu_lib;

int main() {
    // Создать и инициализировать DrvGPU для GPU #0
    DrvGPU gpu(BackendType::OPENCL, 0);
    gpu.Initialize();
    
    // Получить информацию об устройстве
    auto info = gpu.GetDeviceInfo();
    std::cout << "Device: " << info.name << "\n";
    std::cout << "Memory: " << info.GetGlobalMemoryGB() << " GB\n";
    
    // Создать буфер
    auto& mem_mgr = gpu.GetMemoryManager();
    auto buffer = mem_mgr.CreateBuffer<float>(1024);
    
    // Записать данные
    std::vector<float> data(1024, 1.0f);
    buffer->Write(data);
    
    // Прочитать данные
    auto result = buffer->Read();
    
    return 0;
}
```

### Пример: Multi-GPU

```cpp
#include "gpu_manager.hpp"
#include "backend_type.hpp"

using namespace drv_gpu_lib;

int main() {
    // Инициализировать все доступные GPU
    GPUManager manager;
    manager.InitializeAll(BackendType::OPENCL);
    
    std::cout << "Found " << manager.GetGPUCount() << " GPU(s)\n";
    
    // Round-Robin распределение задач
    for (int i = 0; i < 100; ++i) {
        auto& gpu = manager.GetNextGPU();
        auto buffer = gpu.GetMemoryManager().CreateBuffer<float>(1024);
        // ... работа с буфером ...
    }
    
    // Явный выбор GPU
    auto& gpu0 = manager.GetGPU(0);
    auto& gpu1 = manager.GetGPU(1);
    
    // Синхронизация всех GPU
    manager.SynchronizeAll();
    
    return 0;
}
```

---

## 📂 Структура проекта

```
DrvGPU/
├── include/                 # Публичные заголовочные файлы
│   ├── drv_gpu.hpp         # Главный класс DrvGPU
│   ├── gpu_manager.hpp     # Менеджер для Multi-GPU
│   ├── i_backend.hpp       # Интерфейс бэкенда
│   ├── opencl_backend.hpp  # OpenCL реализация
│   ├── memory_manager.hpp  # Управление памятью
│   ├── gpu_buffer.hpp      # RAII буфер
│   ├── module_registry.hpp # Регистр модулей
│   ├── i_compute_module.hpp # Интерфейс модулей
│   └── ...
├── src/                     # Исходные файлы (.cpp)
│   ├── core/               # Ядро библиотеки
│   ├── backend/            # Реализации бэкендов
│   ├── memory/             # Memory Manager
│   ├── modules/            # Compute модули
│   └── opencl/             # Ваш OpenCL код (интеграция)
├── examples/               # Примеры использования
│   ├── single_gpu.cpp     # Single GPU пример
│   ├── multi_gpu.cpp      # Multi-GPU пример
│   └── ...
├── tests/                  # Unit тесты
├── docs/                   # Документация
└── CMakeLists.txt         # CMake конфигурация
```

---

## 🏗️ Архитектура

### Layered Architecture

```
┌─────────────────────────────────────────────────────┐
│  Application Layer                                  │
│  (Ваше приложение)                                  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  API Layer (DrvGPU)                                 │
│  - DrvGPU class (НЕ Singleton!)                     │
│  - GPUManager (Multi-GPU Facade)                    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Abstraction Layer                                  │
│  - IBackend interface (Bridge Pattern)              │
│  - MemoryManager                                    │
│  - ModuleRegistry                                   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Backend Layer                                      │
│  - OpenCLBackend (ваш код)                          │
│  - CUDABackend (будущее)                            │
│  - VulkanBackend (будущее)                          │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Driver Layer                                       │
│  - OpenCL / CUDA / Vulkan API                       │
└─────────────────────────────────────────────────────┘
```

### Паттерны проектирования

- **Bridge Pattern** - отделение абстракции (DrvGPU) от реализации (backends)
- **Facade Pattern** - GPUManager упрощает работу с Multi-GPU
- **Factory Method** - создание backend'ов и буферов
- **Strategy Pattern** - load balancing стратегии
- **Registry Pattern** - ModuleRegistry для compute модулей
- **RAII** - автоматическое управление ресурсами

---

## 🔑 Ключевые классы

### DrvGPU

Главный класс библиотеки. **НЕ Singleton!** - можно создать экземпляр для каждой GPU.

```cpp
DrvGPU gpu(BackendType::OPENCL, device_index);
gpu.Initialize();

// Доступ к подсистемам
MemoryManager& mem_mgr = gpu.GetMemoryManager();
ModuleRegistry& registry = gpu.GetModuleRegistry();
IBackend& backend = gpu.GetBackend();
```

### GPUManager

Facade для управления множественными GPU. **Ключевой класс для Multi-GPU!**

```cpp
GPUManager manager;
manager.InitializeAll(BackendType::OPENCL);

// Round-Robin
auto& next_gpu = manager.GetNextGPU();

// Явный выбор
auto& gpu0 = manager.GetGPU(0);

// Least Loaded
auto& gpu = manager.GetLeastLoadedGPU();
```

### IBackend

Абстрактный интерфейс для всех backend'ов (Bridge Pattern).

```cpp
class IBackend {
    virtual void Initialize(int device_index) = 0;
    virtual void* Allocate(size_t size_bytes) = 0;
    virtual void Synchronize() = 0;
    // ...
};
```

Реализации:
- `OpenCLBackend` - интегрирует ваш OpenCL код
- `CUDABackend` - будущее
- `VulkanBackend` - будущее

### MemoryManager

Backend-агностичное управление памятью.

```cpp
MemoryManager& mem_mgr = gpu.GetMemoryManager();

// Создать буфер
auto buffer = mem_mgr.CreateBuffer<float>(1024);

// Записать/прочитать данные
buffer->Write(data);
auto result = buffer->Read();
```

### ModuleRegistry

Регистр compute модулей (FFT, Matrix, Convolution, etc.)

```cpp
ModuleRegistry& registry = gpu.GetModuleRegistry();

// Зарегистрировать модуль
auto fft_module = std::make_shared<FFTModule>(backend);
registry.RegisterModule("FFT", fft_module);

// Получить модуль
auto fft = registry.GetModule("FFT");
fft->Initialize();
```

---

## 🎯 Multi-GPU: Сравнение старого и нового подхода

### ❌ Старый подход (Singleton - ПРОБЛЕМА!)

```cpp
// Singleton - НЕВОЗМОЖНО работать с несколькими GPU!
auto& gpu = DrvGPU::GetInstance(); // Только ОДНА GPU

// Нельзя явно выбрать GPU #0 или GPU #1
```

### ✅ Новый подход (Multi-Instance + GPUManager)

```cpp
// Multi-Instance - каждая GPU имеет свой экземпляр
GPUManager manager;
manager.InitializeAll(BackendType::OPENCL);

// Явный выбор GPU
auto& gpu0 = manager.GetGPU(0);
auto& gpu1 = manager.GetGPU(1);

// Round-Robin
for (int i = 0; i < 100; ++i) {
    auto& gpu = manager.GetNextGPU();
    // ... работа с GPU ...
}
```

**См. документацию:** `Singleton-vs-MultiGPU-Comparison.md` ⭐⭐⭐

---

## 📊 Load Balancing стратегии

GPUManager поддерживает несколько стратегий распределения нагрузки:

### Round-Robin (по умолчанию)

Циклический выбор GPU (0 → 1 → 2 → 0 → ...).

```cpp
manager.SetLoadBalancingStrategy(LoadBalancingStrategy::ROUND_ROBIN);

for (int i = 0; i < 100; ++i) {
    auto& gpu = manager.GetNextGPU();
    // Автоматически распределяется по всем GPU
}
```

### Least Loaded

Выбор наименее загруженной GPU.

```cpp
manager.SetLoadBalancingStrategy(LoadBalancingStrategy::LEAST_LOADED);
auto& gpu = manager.GetLeastLoadedGPU();
```

### Manual

Ручной выбор GPU по индексу.

```cpp
auto& gpu0 = manager.GetGPU(0);
auto& gpu1 = manager.GetGPU(1);
```

---

## 🧩 Интеграция с вашим OpenCL кодом

DrvGPU полностью интегрирует ваш существующий OpenCL код:

```cpp
// OpenCLBackend использует ваши классы:
// -DrvGPU::OpenCLCore
// -DrvGPU::CommandQueuePool
// -DrvGPU::GPUMemoryManager
// -DrvGPU::SVMCapabilities

DrvGPU gpu(BackendType::OPENCL, 0);
auto& opencl_backend = dynamic_cast<OpenCLBackend&>(gpu.GetBackend());

// Доступ к вашему коду
auto& opencl_core = opencl_backend.GetCore();
auto& gpu_memory_mgr = opencl_backend.GetMemoryManager();
```

---

## 📚 Примеры

### Single GPU
```bash
cd build/examples
./example_single_gpu
```

### Multi-GPU
```bash
cd build/examples
./example_multi_gpu
```

Больше примеров в директории `examples/`.

---

## 🧪 Тестирование

```bash
cd build
ctest --output-on-failure
```

---

## 📖 Документация

Полная документация в директории `docs/`:

- **README.md** (этот файл) - обзор и quick start
- **Singleton-vs-MultiGPU-Comparison.md** ⭐⭐⭐ - КРИТИЧЕСКИ ВАЖНО!
- **GPU-Library-Multi-GPU-Updated.md** - полная архитектура
- **Quick-Start-Guide.md** - практическое руководство
- **Multi-GPU-Architecture.md** - детали архитектуры
- **PROJECT_INDEX.md** - навигация по документации

---

## 🔧 Требования

- **C++17** или выше
- **CMake 3.15+**
- **OpenCL** (для OpenCL backend)
- **CUDA** (опционально, для CUDA backend)
- **Vulkan** (опционально, для Vulkan backend)

---

## 🤝 Участие в разработке

Pull requests приветствуются! Пожалуйста, убедитесь что:
- Код соответствует стандарту C++17
- Все тесты проходят
- Документация обновлена

---

## 📜 Лицензия

MIT License - см. `LICENSE` файл.

---

## 👥 Авторы

DrvGPU Team

---

## 📞 Контакты

- GitHub Issues: https://github.com/your-org/DrvGPU/issues
- Email: drvgpu@your-org.com

---

## 🎯 Roadmap

### v1.0 (Текущая версия)
- ✅ OpenCL backend
- ✅ Multi-GPU support (GPUManager)
- ✅ Load balancing (Round-Robin, Least Loaded)
- ✅ Memory Manager
- ✅ Module Registry

### v1.1 (Планируется)
- 🔲 CUDA backend
- 🔲 Advanced load balancing (GPU affinity)
- 🔲 Асинхронные операции
- 🔲 Compute модули (FFT, Matrix)

### v2.0 (Будущее)
- 🔲 Vulkan Compute backend
- 🔲 Multi-node support (distributed GPU)
- 🔲 Python bindings
- 🔲 Performance profiling tools

---

**DrvGPU - современная библиотека для Multi-GPU вычислений!** 🚀
