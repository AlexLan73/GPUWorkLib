# 🗂️ Структура проекта DrvGPU - Руководство по настройке

## 📂 Рекомендуемая структура директорий

Для правильной работы CMake конфигурации, организуйте ваш проект следующим образом:

```
DrvGPU/                          # Корневая папка проекта
├── CMakeLists.txt              # Главный CMakeLists (используйте DrvGPU-CMakeLists-Main.txt)
│
├── cmake/                       # CMake вспомогательные файлы
│   └── DrvGPUConfig.cmake.in   # Config файл (используйте DrvGPU-CMake-Config.in)
│
├── include/                     # Публичные заголовочные файлы DrvGPU
│   ├── drv_gpu.hpp
│   ├── gpu_manager.hpp
│   ├── i_backend.hpp
│   ├── opencl_backend.hpp
│   ├── memory_manager.hpp
│   ├── gpu_buffer.hpp
│   ├── module_registry.hpp
│   ├── i_compute_module.hpp
│   ├── backend_type.hpp
│   ├── gpu_device_info.hpp
│   └── load_balancing_strategy.hpp
│
├── src/                         # Исходные файлы (.cpp)
│   ├── core/                    # Ядро библиотеки
│   │   ├── drv_gpu.cpp
│   │   ├── gpu_manager.cpp
│   │   └── gpu_device_info.cpp
│   │
│   ├── backends/                # Реализации бэкендов
│   │   └── opencl/
│   │       └── opencl_backend.cpp
│   │
│   ├── memory/                  # Memory Manager
│   │   └── memory_manager.cpp
│   │
│   └── modules/                 # Compute модули
│       └── module_registry.cpp
│
├── opencl/                      # Ваш существующий OpenCL код
│   ├── opencl_core.hpp
│   ├── opencl_core.cpp
│   ├── opencl_manager.h
│   ├── opencl_manager.cpp
│   ├── opencl_compute_engine.hpp
│   ├── opencl_compute_engine.cpp
│   ├── command_queue_pool.hpp
│   ├── command_queue_pool.cpp
│   ├── kernel_program.hpp
│   ├── kernel_program.cpp
│   ├── gpu_memory_manager.hpp
│   ├── gpu_memory_manager.cpp
│   ├── gpu_memory.hpp
│   ├── i_memory_buffer.hpp
│   ├── svm_buffer.hpp
│   ├── regular_buffer.hpp
│   ├── hybrid_buffer.hpp
│   ├── svm_capabilities.hpp
│   ├── memory_type.hpp
│   └── gpu_memory_buffer.hpp
│
├── examples/                    # Примеры использования
│   ├── CMakeLists.txt          # CMake для примеров (используйте DrvGPU-CMake-Examples.txt)
│   ├── single_gpu.cpp
│   └── multi_gpu.cpp
│
├── tests/                       # Unit тесты
│   └── CMakeLists.txt
│
├── docs/                        # Документация
│   └── README.md
│
└── README.md                    # Главная документация
```

---

## 🔧 Инструкции по настройке

### Шаг 1: Создайте структуру директорий

```bash
# Создать директории
mkdir -p DrvGPU/{cmake,include,src/{core,backends/opencl,memory,modules},opencl,examples,tests,docs}
```

### Шаг 2: Разместите заголовочные файлы

**Новые DrvGPU заголовки** → `include/`:
```bash
# Скопируйте все .hpp файлы из выгруженных:
cp DrvGPU-Core-drv_gpu.hpp include/drv_gpu.hpp
cp DrvGPU-Core-gpu_manager.hpp include/gpu_manager.hpp
cp DrvGPU-Backend-i_backend.hpp include/i_backend.hpp
cp DrvGPU-Backend-opencl_backend.hpp include/opencl_backend.hpp
cp DrvGPU-Memory-memory_manager.hpp include/memory_manager.hpp
cp DrvGPU-Memory-gpu_buffer.hpp include/gpu_buffer.hpp
cp DrvGPU-Modules-module_registry.hpp include/module_registry.hpp
cp DrvGPU-Modules-i_compute_module.hpp include/i_compute_module.hpp
cp DrvGPU-Common-backend_type.hpp include/backend_type.hpp
cp DrvGPU-Common-gpu_device_info.hpp include/gpu_device_info.hpp
cp DrvGPU-Common-load_balancing.hpp include/load_balancing_strategy.hpp
```

**Ваш OpenCL код** → `opencl/`:
```bash
# Переместите ваши существующие файлы:
mv *.hpp *.cpp opencl/
```

### Шаг 3: Создайте .cpp реализации

Вам нужно создать следующие .cpp файлы в `src/`:

#### `src/core/drv_gpu.cpp`
```cpp
#include "drv_gpu.hpp"
#include "opencl_backend.hpp"
#include <stdexcept>

namespace drv_gpu_lib {

DrvGPU::DrvGPU(BackendType backend_type, int device_index)
    : backend_type_(backend_type),
      device_index_(device_index),
      initialized_(false)
{
    CreateBackend();
}

DrvGPU::~DrvGPU() {
    Cleanup();
}

void DrvGPU::CreateBackend() {
    switch (backend_type_) {
        case BackendType::OPENCL:
            backend_ = std::make_unique<OpenCLBackend>();
            break;
        case BackendType::CUDA:
            throw std::runtime_error("CUDA backend not yet implemented");
        case BackendType::VULKAN:
            throw std::runtime_error("Vulkan backend not yet implemented");
        default:
            throw std::runtime_error("Unknown backend type");
    }
}

void DrvGPU::Initialize() {
    if (initialized_) {
        return;
    }
    
    backend_->Initialize(device_index_);
    InitializeSubsystems();
    
    initialized_ = true;
}

void DrvGPU::InitializeSubsystems() {
    // Создать Memory Manager
    memory_manager_ = std::make_unique<MemoryManager>(backend_.get());
    
    // Создать Module Registry
    module_registry_ = std::make_unique<ModuleRegistry>();
}

void DrvGPU::Cleanup() {
    if (!initialized_) {
        return;
    }
    
    module_registry_.reset();
    memory_manager_.reset();
    backend_->Cleanup();
    
    initialized_ = false;
}

// ... остальные методы ...

} // namespace drv_gpu_lib
```

#### `src/core/gpu_manager.cpp`
```cpp
#include "gpu_manager.hpp"
#include <algorithm>

namespace drv_gpu_lib {

GPUManager::GPUManager()
    : backend_type_(BackendType::AUTO),
      lb_strategy_(LoadBalancingStrategy::ROUND_ROBIN),
      round_robin_index_(0)
{
}

GPUManager::~GPUManager() {
    Cleanup();
}

void GPUManager::InitializeAll(BackendType backend_type) {
    backend_type_ = backend_type;
    
    int gpu_count = DiscoverGPUs(backend_type);
    
    if (gpu_count == 0) {
        throw std::runtime_error("No GPUs found");
    }
    
    // Создать DrvGPU для каждой GPU
    for (int i = 0; i < gpu_count; ++i) {
        auto gpu = std::make_unique<DrvGPU>(backend_type, i);
        gpu->Initialize();
        gpus_.push_back(std::move(gpu));
        gpu_task_count_.push_back(0);
    }
}

DrvGPU& GPUManager::GetNextGPU() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (gpus_.empty()) {
        throw std::runtime_error("No GPUs initialized");
    }
    
    size_t index = round_robin_index_.fetch_add(1) % gpus_.size();
    return *gpus_[index];
}

// ... остальные методы ...

} // namespace drv_gpu_lib
```

#### `src/backends/opencl/opencl_backend.cpp`
```cpp
#include "opencl_backend.hpp"
#include <stdexcept>

namespace drv_gpu_lib {

OpenCLBackend::OpenCLBackend()
    : device_index_(-1),
      initialized_(false),
      context_(nullptr),
      device_(nullptr),
      queue_(nullptr)
{
}

OpenCLBackend::~OpenCLBackend() {
    Cleanup();
}

void OpenCLBackend::Initialize(int device_index) {
    if (initialized_) {
        return;
    }
    
    device_index_ = device_index;
    
    InitializeOpenCLCore();
    InitializeMemoryManager();
    InitializeSVMCapabilities();
    
    initialized_ = true;
}

void OpenCLBackend::InitializeOpenCLCore() {
    // Использовать ваш OpenCLCore
    opencl_core_ = std::make_unique<ManagerOpenCL::OpenCLCore>();
    opencl_core_->Initialize(device_index_);
    
    // Кэшировать важные объекты
    context_ = opencl_core_->GetContext();
    device_ = opencl_core_->GetDevice();
    queue_ = opencl_core_->GetQueue();
}

// ... остальные методы ...

} // namespace drv_gpu_lib
```

#### `src/memory/memory_manager.cpp`
```cpp
#include "memory_manager.hpp"

namespace drv_gpu_lib {

MemoryManager::MemoryManager(IBackend* backend)
    : backend_(backend),
      total_allocations_(0),
      total_frees_(0),
      current_allocations_(0),
      total_bytes_allocated_(0),
      peak_bytes_allocated_(0)
{
}

MemoryManager::~MemoryManager() {
    Cleanup();
}

void* MemoryManager::Allocate(size_t size_bytes, unsigned int flags) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    void* ptr = backend_->Allocate(size_bytes, flags);
    TrackAllocation(size_bytes);
    
    return ptr;
}

void MemoryManager::Free(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    backend_->Free(ptr);
    // TrackFree вызывается после освобождения
}

// ... остальные методы ...

} // namespace drv_gpu_lib
```

#### `src/modules/module_registry.cpp`
```cpp
#include "module_registry.hpp"
#include <stdexcept>

namespace drv_gpu_lib {

ModuleRegistry::ModuleRegistry() {
}

ModuleRegistry::~ModuleRegistry() {
    Clear();
}

void ModuleRegistry::RegisterModule(const std::string& name, 
                                   std::shared_ptr<IComputeModule> module) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (modules_.find(name) != modules_.end()) {
        throw std::runtime_error("Module '" + name + "' already registered");
    }
    
    modules_[name] = module;
}

std::shared_ptr<IComputeModule> ModuleRegistry::GetModule(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = modules_.find(name);
    if (it == modules_.end()) {
        throw std::runtime_error("Module '" + name + "' not found");
    }
    
    return it->second;
}

// ... остальные методы ...

} // namespace drv_gpu_lib
```

### Шаг 4: Разместите CMake файлы

```bash
# Главный CMakeLists.txt
cp DrvGPU-CMakeLists-Main.txt CMakeLists.txt

# Config файл
mkdir -p cmake
cp DrvGPU-CMake-Config.in cmake/DrvGPUConfig.cmake.in

# CMake для примеров
cp DrvGPU-CMake-Examples.txt examples/CMakeLists.txt
```

### Шаг 5: Примеры

```bash
# Скопируйте примеры
cp DrvGPU-Examples-single_gpu.cpp examples/single_gpu.cpp
cp DrvGPU-Examples-multi_gpu.cpp examples/multi_gpu.cpp
```

---

## 🚀 Сборка проекта

### Вариант 1: Standalone сборка (DrvGPU как отдельный проект)

```bash
cd DrvGPU
mkdir build && cd build

# Конфигурация
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DDRVGPU_BUILD_SHARED=OFF \
    -DDRVGPU_BUILD_EXAMPLES=ON \
    -DDRVGPU_BUILD_TESTS=ON

# Сборка
cmake --build . -j$(nproc)

# Установка (опционально)
sudo cmake --install .
```

### Вариант 2: Интеграция в другой проект

#### Способ A: add_subdirectory

В вашем главном CMakeLists.txt:
```cmake
# Добавить DrvGPU как подпроект
add_subdirectory(DrvGPU)

# Ваша программа
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE DrvGPU::drvgpu)
```

#### Способ B: find_package (после установки)

```cmake
# Найти установленную библиотеку
find_package(DrvGPU 1.0 REQUIRED)

# Ваша программа
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE DrvGPU::drvgpu)
```

---

## 📝 Использование библиотеки в вашем коде

После сборки и установки:

```cpp
#include <drvgpu/drv_gpu.hpp>
#include <drvgpu/gpu_manager.hpp>
#include <drvgpu/backend_type.hpp>

using namespace drv_gpu_lib;

int main() {
    // Single GPU
    DrvGPU gpu(BackendType::OPENCL, 0);
    gpu.Initialize();
    
    // Multi-GPU
    GPUManager manager;
    manager.InitializeAll(BackendType::OPENCL);
    
    return 0;
}
```

Компиляция:
```bash
g++ -std=c++17 main.cpp -ldrvgpu -lOpenCL -lpthread
```

Или с CMake:
```cmake
find_package(DrvGPU REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE DrvGPU::drvgpu)
```

---

## 🔍 Проверка установки

После `cmake --install`:

```bash
# Проверить заголовки
ls /usr/local/include/drvgpu/

# Проверить библиотеку
ls /usr/local/lib/libdrvgpu.a

# Проверить CMake config
ls /usr/local/lib/cmake/DrvGPU/

# Проверить примеры
ls /usr/local/bin/examples/
```

---

## ⚙️ Опции CMake

| Опция | Описание | По умолчанию |
|-------|----------|--------------|
| `DRVGPU_BUILD_SHARED` | Собрать как shared library (.so/.dll) | OFF |
| `DRVGPU_BUILD_EXAMPLES` | Собрать примеры | ON |
| `DRVGPU_BUILD_TESTS` | Собрать тесты | ON |
| `DRVGPU_ENABLE_OPENCL` | Включить OpenCL backend | ON |
| `DRVGPU_ENABLE_CUDA` | Включить CUDA backend | OFF |
| `DRVGPU_ENABLE_VULKAN` | Включить Vulkan backend | OFF |

Пример использования:
```bash
cmake .. -DDRVGPU_BUILD_SHARED=ON -DDRVGPU_BUILD_EXAMPLES=OFF
```

---

## 🐛 Troubleshooting

### Проблема: "OpenCL not found"
```bash
# Ubuntu/Debian
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# CentOS/RHEL
sudo yum install opencl-headers ocl-icd-devel

# macOS (встроен в систему)
```

### Проблема: Не находит ваши OpenCL заголовки
Убедитесь что в CMakeLists.txt правильно указан путь:
```cmake
set(DRVGPU_OPENCL_DIR ${DRVGPU_ROOT_DIR}/opencl)
```

### Проблема: Ошибки линковки
Проверьте что все .cpp файлы добавлены в `DRVGPU_ALL_SOURCES`.

---

## 📚 Дополнительные ресурсы

- **README.md** - полная документация DrvGPU
- **examples/** - рабочие примеры
- **docs/** - детальная документация

---

**Готово!** Теперь у вас есть полная CMake конфигурация для DrvGPU как отдельной библиотеки.
