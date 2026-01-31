# LibGPU Quick Start Guide

## Введение

Этот гид поможет вам начать работу с LibGPU за **10 минут**. Мы покажем:
- Как собрать и установить библиотеку
- Как инициализировать GPU систему (Multi-GPU!)
- Как использовать compute модули
- Как создать свой модуль

**⚠️ ВАЖНО:** Сначала прочитайте **Singleton-vs-MultiGPU-Comparison.md** чтобы понять архитектурные решения!

---

## Установка (5 минут)

### Требования

- **C++17** compiler (GCC 8+, Clang 7+, MSVC 2019+)
- **CMake** >= 3.18
- **OpenCL** >= 2.0 (или ROCm для AMD GPU)
- **Google Test** (опционально, только для тестов)

### Сборка

```bash
# 1. Клонирование репозитория
git clone https://github.com/your-org/libgpu.git
cd libgpu

# 2. Создание build директории
mkdir build && cd build

# 3. Конфигурация CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DLIBGPU_BUILD_TESTS=OFF \
         -DLIBGPU_BUILD_EXAMPLES=ON

# 4. Сборка
cmake --build . -j8

# 5. Установка (опционально)
sudo cmake --install .
```

### Проверка установки

```bash
# Запуск примера
./examples/basic_usage

# Вы должны увидеть:
# Initialized 8 GPUs
# GPU 0: NVIDIA GeForce RTX 4090
# GPU 1: NVIDIA GeForce RTX 4090
# ...
```

---

## Hello World (2 минуты)

### Базовое использование одной GPU

```cpp
#include <gpu_lib/GPUManager.hpp>
#include <iostream>

int main() {
    using namespace gpu_lib;
    
    // 1. Создание менеджера GPU
    core::GPUManager manager;
    
    // 2. Инициализация всех GPU
    int num_gpus = manager.initializeAll(BackendType::OpenCL);
    std::cout << "Initialized " << num_gpus << " GPUs\n";
    
    // 3. Получение конкретной GPU
    auto& gpu = manager.getGPU(0);
    
    // 4. Выделение памяти
    auto& mem = gpu.getMemoryManager();
    auto buffer = mem.allocate(1024 * 1024);  // 1 MB
    
    std::cout << "Buffer allocated: " << buffer->getSize() << " bytes\n";
    
    // 5. Cleanup
    manager.shutdownAll();
    
    return 0;
}
```

### Сборка и запуск

```bash
# Создайте файл hello_world.cpp с кодом выше

# Сборка
g++ -std=c++17 hello_world.cpp \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lgpu_lib_core \
    -lOpenCL \
    -o hello_world

# Запуск
./hello_world
```

**Вывод:**
```
Initialized 8 GPUs
Buffer allocated: 1048576 bytes
```

---

## Multi-GPU примеры (3 минуты)

### Пример 1: Round-Robin распределение

```cpp
#include <gpu_lib/GPUManager.hpp>
#include <vector>

int main() {
    using namespace gpu_lib;
    
    core::GPUManager manager;
    manager.initializeAll(BackendType::OpenCL);
    
    // Обрабатываем 16 задач на 8 GPU (по кругу)
    for (int i = 0; i < 16; ++i) {
        // Выбираем следующую GPU циклически
        auto& gpu = manager.selectRoundRobin();
        
        std::cout << "Task " << i << " → GPU " << gpu.getDeviceId() << "\n";
        
        // Обработка на выбранной GPU
        auto& mem = gpu.getMemoryManager();
        auto buffer = mem.allocate(1024 * 1024);
        
        // ... ваша обработка ...
    }
    
    manager.shutdownAll();
    return 0;
}
```

**Вывод:**
```
Task 0 → GPU 0
Task 1 → GPU 1
Task 2 → GPU 2
...
Task 7 → GPU 7
Task 8 → GPU 0  (цикл повторяется)
Task 9 → GPU 1
...
```

### Пример 2: Load Balancing

```cpp
#include <gpu_lib/GPUManager.hpp>

int main() {
    using namespace gpu_lib;
    
    core::GPUManager manager;
    manager.initializeAll(BackendType::OpenCL);
    
    for (int i = 0; i < 10; ++i) {
        // Выбираем GPU с минимальной загрузкой памяти
        auto& gpu = manager.selectLeastLoaded();
        
        std::cout << "Task " << i << " → GPU " << gpu.getDeviceId() 
                  << " (least loaded)\n";
        
        // Обработка
        auto& mem = gpu.getMemoryManager();
        auto buffer = mem.allocate((i + 1) * 1024 * 1024);  // Разные размеры
    }
    
    // Статистика
    auto stats = manager.getTotalMemoryStats();
    std::cout << "\nTotal memory used: " 
              << stats.total_allocated_all_gpus / (1024*1024) << " MB\n";
    
    manager.shutdownAll();
    return 0;
}
```

### Пример 3: Параллельная обработка

```cpp
#include <gpu_lib/GPUManager.hpp>
#include <thread>
#include <vector>

void processOnGPU(gpu_lib::core::DrvGPU& gpu, int task_id) {
    std::cout << "Task " << task_id << " running on GPU " 
              << gpu.getDeviceId() << "\n";
    
    auto& mem = gpu.getMemoryManager();
    auto buffer = mem.allocate(1024 * 1024);
    
    // Симуляция обработки
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    gpu.synchronize();
    std::cout << "Task " << task_id << " completed\n";
}

int main() {
    using namespace gpu_lib;
    
    core::GPUManager manager;
    manager.initializeAll(BackendType::OpenCL);
    
    auto all_gpus = manager.getAllGPUs();
    std::vector<std::thread> threads;
    
    // Запускаем по одной задаче на каждую GPU параллельно
    for (size_t i = 0; i < all_gpus.size(); ++i) {
        threads.emplace_back(processOnGPU, std::ref(*all_gpus[i]), i);
    }
    
    // Ждем завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "\nAll " << all_gpus.size() << " GPUs completed!\n";
    
    manager.shutdownAll();
    return 0;
}
```

**Результат: Все 8 GPU работают одновременно!**

---

## Работа с модулями

### Использование существующего модуля

```cpp
#include <gpu_lib/GPUManager.hpp>
#include <gpu_lib/modules/FFTPostProcessing.hpp>

int main() {
    using namespace gpu_lib;
    
    // 1. Инициализация GPU
    core::GPUManager manager;
    manager.initializeAll(BackendType::OpenCL);
    auto& gpu = manager.getGPU(0);
    
    // 2. Получение модуля (создается автоматически если нужно)
    auto& registry = gpu.getModuleRegistry();
    auto fft_module = registry.getOrCreateModule<modules::FFTPostProcessing>();
    
    // 3. Подготовка данных
    std::vector<std::complex<float>> fft_data(1024);
    
    // Генерация тестового сигнала (синусоида)
    for (size_t i = 0; i < fft_data.size(); ++i) {
        float t = static_cast<float>(i) / fft_data.size();
        fft_data[i] = std::complex<float>(std::sin(2.0f * M_PI * 10.0f * t), 0.0f);
    }
    
    // 4. Обработка на GPU
    auto peaks = fft_module->findPeaks(
        fft_data,
        3,      // top N peaks
        0.1f    // threshold
    );
    
    // 5. Результаты
    std::cout << "Found " << peaks.size() << " peaks:\n";
    for (const auto& peak : peaks) {
        std::cout << "  Index: " << peak.index 
                  << ", Magnitude: " << peak.magnitude << "\n";
    }
    
    manager.shutdownAll();
    return 0;
}
```

---

## Создание своего модуля

### Шаг 1: Определение интерфейса

```cpp
// include/gpu_lib/modules/MyModule.hpp
#pragma once

#include <gpu_lib/core/IComputeModule.hpp>
#include <gpu_lib/core/ComputeModuleBase.hpp>
#include <vector>

namespace gpu_lib {
namespace modules {

/**
 * @brief Пример пользовательского модуля
 */
class MyModule : public core::ComputeModuleBase {
public:
    MyModule() = default;
    ~MyModule() override = default;
    
    /**
     * @brief Имя модуля
     */
    std::string getName() const override {
        return "MyModule";
    }
    
    /**
     * @brief Зависимости (если нужны другие модули)
     */
    std::vector<std::string> getDependencies() const override {
        return {};  // Нет зависимостей
    }
    
    /**
     * @brief Публичный API вашего модуля
     */
    std::vector<float> process(const std::vector<float>& input);
    
protected:
    /**
     * @brief Инициализация (вызывается автоматически)
     */
    void onInitialize() override;
    
    /**
     * @brief Cleanup (вызывается автоматически)
     */
    void onShutdown() override;
    
private:
    std::unique_ptr<core::IKernel> kernel_;
    std::shared_ptr<core::IMemoryBuffer> input_buffer_;
    std::shared_ptr<core::IMemoryBuffer> output_buffer_;
};

} // namespace modules
} // namespace gpu_lib
```

### Шаг 2: Реализация

```cpp
// src/modules/MyModule.cpp
#include <gpu_lib/modules/MyModule.hpp>

namespace gpu_lib {
namespace modules {

void MyModule::onInitialize() {
    // 1. Загрузка OpenCL kernel
    const char* kernel_source = R"(
        __kernel void my_kernel(
            __global const float* input,
            __global float* output,
            int size
        ) {
            int gid = get_global_id(0);
            if (gid < size) {
                // Ваша обработка (например, умножение на 2)
                output[gid] = input[gid] * 2.0f;
            }
        }
    )";
    
    kernel_ = backend_->createKernel(kernel_source, "my_kernel");
    
    std::cout << "MyModule initialized\n";
}

void MyModule::onShutdown() {
    // Очистка ресурсов
    kernel_.reset();
    input_buffer_.reset();
    output_buffer_.reset();
    
    std::cout << "MyModule shutdown\n";
}

std::vector<float> MyModule::process(const std::vector<float>& input) {
    // 1. Выделение памяти на GPU (если нужно)
    size_t size_bytes = input.size() * sizeof(float);
    
    if (!input_buffer_ || input_buffer_->getSize() < size_bytes) {
        input_buffer_ = memory_manager_->allocate(size_bytes);
        output_buffer_ = memory_manager_->allocate(size_bytes);
    }
    
    // 2. Копирование данных на GPU
    input_buffer_->copyFrom(input.data(), size_bytes);
    
    // 3. Запуск kernel
    kernel_->setArg(0, input_buffer_.get());
    kernel_->setArg(1, output_buffer_.get());
    kernel_->setArg(2, static_cast<int>(input.size()));
    
    size_t global_size = input.size();
    kernel_->execute(&global_size, nullptr);
    
    // 4. Копирование результата обратно на CPU
    std::vector<float> output(input.size());
    output_buffer_->copyTo(output.data(), size_bytes);
    
    return output;
}

} // namespace modules
} // namespace gpu_lib
```

### Шаг 3: Регистрация модуля

```cpp
// В вашем коде
#include <gpu_lib/modules/MyModule.hpp>

int main() {
    using namespace gpu_lib;
    
    core::GPUManager manager;
    manager.initializeAll(BackendType::OpenCL);
    auto& gpu = manager.getGPU(0);
    
    // Регистрация модуля (если нужно)
    auto& registry = gpu.getModuleRegistry();
    registry.registerModule<modules::MyModule>("MyModule");
    
    // Использование
    auto my_module = registry.getOrCreateModule<modules::MyModule>();
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto output = my_module->process(input);
    
    // output = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f}
    
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << "\n";
    
    manager.shutdownAll();
    return 0;
}
```

---

## CMake интеграция

### Добавление в ваш проект

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(MyGPUProject)

set(CMAKE_CXX_STANDARD 17)

# Поиск LibGPU
find_package(LibGPU REQUIRED)

# Ваш исполняемый файл
add_executable(my_app main.cpp)

# Линковка с LibGPU
target_link_libraries(my_app
    PRIVATE
        LibGPU::core
        LibGPU::modules
)

# OpenCL (если нужен)
find_package(OpenCL REQUIRED)
target_link_libraries(my_app PRIVATE OpenCL::OpenCL)
```

### Сборка вашего проекта

```bash
mkdir build && cd build
cmake ..
cmake --build .
./my_app
```

---

## Debugging и Troubleshooting

### Проверка доступности GPU

```cpp
#include <gpu_lib/GPUManager.hpp>

int main() {
    using namespace gpu_lib;
    
    core::GPUManager manager;
    
    try {
        int num_gpus = manager.initializeAll(BackendType::OpenCL);
        
        std::cout << "Found " << num_gpus << " GPUs:\n";
        
        for (int i = 0; i < num_gpus; ++i) {
            auto& gpu = manager.getGPU(i);
            auto info = gpu.getDeviceInfo();
            
            std::cout << "\nGPU " << i << ":\n";
            std::cout << "  Name: " << info.name << "\n";
            std::cout << "  Vendor: " << info.vendor << "\n";
            std::cout << "  Memory: " << info.global_memory_size / (1024*1024*1024) << " GB\n";
            std::cout << "  Compute Units: " << info.compute_units << "\n";
        }
        
    } catch (const core::GPUException& e) {
        std::cerr << "GPU Error: " << e.what() << "\n";
        return 1;
    }
    
    manager.shutdownAll();
    return 0;
}
```

### Проверка памяти

```cpp
auto& mem = gpu.getMemoryManager();
auto stats = mem.getStats();

std::cout << "Memory Statistics:\n";
std::cout << "  Allocated: " << stats.total_allocated / (1024*1024) << " MB\n";
std::cout << "  Available: " << stats.total_available / (1024*1024) << " MB\n";
std::cout << "  Peak usage: " << stats.peak_usage / (1024*1024) << " MB\n";
```

### Логирование ошибок

```cpp
try {
    auto& gpu = manager.getGPU(100);  // Несуществующая GPU
} catch (const std::out_of_range& e) {
    std::cerr << "Error: " << e.what() << "\n";
}

try {
    auto buffer = mem.allocate(SIZE_MAX);  // Слишком большой буфер
} catch (const core::MemoryException& e) {
    std::cerr << "Memory Error: " << e.what() << "\n";
}
```

---

## Best Practices

### ✅ DO

```cpp
// 1. Используйте GPUManager для управления GPU
GPUManager manager;
manager.initializeAll(BackendType::OpenCL);

// 2. Используйте RAII (автоматическая очистка)
{
    auto buffer = mem.allocate(size);
    // ... использование ...
}  // buffer автоматически освобождается

// 3. Проверяйте ошибки
try {
    auto& gpu = manager.getGPU(0);
} catch (const std::exception& e) {
    // обработка ошибки
}

// 4. Синхронизируйте GPU когда нужно
gpu.synchronize();

// 5. Cleanup в конце
manager.shutdownAll();
```

### ❌ DON'T

```cpp
// 1. НЕ используйте старый Singleton API
// ❌ auto& drv = DrvGPU::getInstance();  // УСТАРЕЛО!

// 2. НЕ забывайте синхронизацию
kernel->execute(...);
// Нужно: gpu.synchronize(); перед чтением результатов

// 3. НЕ создавайте огромные буферы без проверки
// ❌ auto buffer = mem.allocate(SIZE_MAX);

// 4. НЕ забывайте shutdown
// ❌ Без manager.shutdownAll() - утечка ресурсов
```

---

## Следующие шаги

### 1. Изучите архитектуру
→ Прочитайте **Singleton-vs-MultiGPU-Comparison.md** (обязательно!)
→ Изучите **GPU-Library-Multi-GPU-Updated.md**

### 2. Попробуйте примеры
→ Соберите и запустите `examples/` из репозитория
→ Модифицируйте их под свои задачи

### 3. Создайте свой модуль
→ Используйте шаблон из раздела "Создание своего модуля"
→ Добавьте тесты

### 4. Оптимизация
→ Изучите **Project-Summary-And-Next-Steps.md** (Performance Guidelines)
→ Профилируйте ваш код

---

## Полезные ссылки

- **Документация**: См. `PROJECT_INDEX.md` для навигации
- **Примеры**: `examples/` директория в репозитории
- **Тесты**: `tests/` - примеры использования всех компонентов
- **GitHub**: https://github.com/your-org/libgpu

---

## FAQ

### Q: Сколько GPU я могу использовать одновременно?
**A:** Все доступные в системе. `initializeAll()` найдет и инициализирует все GPU.

### Q: Что если у меня только 1 GPU?
**A:** Всё работает точно так же! GPUManager просто управляет 1 GPU.

### Q: Как выбрать конкретную GPU?
**A:** `manager.getGPU(device_id)` - прямой доступ по ID.

### Q: Можно ли переключить backend с OpenCL на ROCm?
**A:** Да: `gpu.switchBackend(BackendType::ROCm)` (требует пересоздания буферов).

### Q: Как узнать сколько памяти использует GPU?
**A:** `gpu.getMemoryManager().getStats()` - детальная статистика.

---

**Поздравляем! Вы готовы начать работу с LibGPU!** 🎉

**Важно:** Не забудьте прочитать **Singleton-vs-MultiGPU-Comparison.md** чтобы понять архитектурные решения! ⭐
