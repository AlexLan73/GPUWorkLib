# Singleton vs Multi-Instance: Сравнение подходов для Multi-GPU

## ⚠️ ВАЖНО ДЛЯ КОМАНДЫ

**Этот документ показывает РАЗНИЦУ между старым (Singleton) и новым (Multi-Instance) подходами.**

**Мы ОТКАЗЫВАЕМСЯ от Singleton в пользу Multi-Instance + GPUManager!**

---

## 📊 Краткое сравнение

| Аспект | ❌ Старый подход (Singleton) | ✅ Новый подход (Multi-Instance) |
|--------|------------------------------|----------------------------------|
| **Количество GPU** | Только 1 GPU | 8+ GPU одновременно |
| **Паттерн** | Singleton | Обычный класс + GPUManager |
| **Создание экземпляров** | `getInstance()` (только 1) | `DrvGPU(device_id)` (множество) |
| **Глобальное состояние** | Да (проблема для тестов) | Нет (изолированные экземпляры) |
| **Тестируемость** | Сложно mock'ировать | Легко создавать test instances |
| **Масштабируемость** | Нет | Линейная (1→8 GPU) |
| **Управление** | Нет центрального | GPUManager координирует всё |

---

## 🔴 Старый подход: Singleton (НЕ ИСПОЛЬЗУЕМ!)

### Проблема с Singleton

```cpp
// ❌ СТАРЫЙ КОД - НЕ ИСПОЛЬЗОВАТЬ!

class DrvGPU {
public:
    // Singleton pattern
    static DrvGPU& getInstance() {
        static DrvGPU instance;  // Только ОДИН экземпляр
        return instance;
    }
    
    void initialize(BackendType backend_type, int device_id) {
        // Может работать только с ОДНОЙ GPU
        device_id_ = device_id;  // Но это фиксировано!
        // ...
    }
    
private:
    // Приватный конструктор - нельзя создать больше экземпляров
    DrvGPU() = default;
    
    // Запрет копирования
    DrvGPU(const DrvGPU&) = delete;
    DrvGPU& operator=(const DrvGPU&) = delete;
    
    int device_id_;
};
```

### Использование старого Singleton

```cpp
// ❌ СТАРЫЙ СПОСОБ - ОГРАНИЧЕН ОДНОЙ GPU!

int main() {
    // Получаем единственный экземпляр
    auto& drv = DrvGPU::getInstance();
    
    // Инициализируем для GPU #0
    drv.initialize(BackendType::OpenCL, 0);
    
    // Работаем с GPU #0
    auto& mem = drv.getMemoryManager();
    auto buffer = mem.allocate(1024 * 1024);
    
    // ❌ ПРОБЛЕМА: Как использовать GPU #1, #2, ..., #7?
    // НЕВОЗМОЖНО! getInstance() всегда возвращает тот же объект
    
    return 0;
}
```

### Почему Singleton НЕ подходит для Multi-GPU?

```cpp
// ❌ ПОПЫТКА использовать несколько GPU с Singleton

// Получаем DrvGPU
auto& drv1 = DrvGPU::getInstance();
drv1.initialize(BackendType::OpenCL, 0);  // GPU #0

// Пытаемся работать с другой GPU
auto& drv2 = DrvGPU::getInstance();  // Это ТОТ ЖЕ объект что drv1!
drv2.initialize(BackendType::OpenCL, 5);  // Переинициализирует GPU #0 → GPU #5

// ❌ РЕЗУЛЬТАТ: 
// - drv1 и drv2 это ОДИН И ТОТ ЖЕ объект
// - Потеряли доступ к GPU #0
// - Невозможно использовать обе GPU одновременно
```

**Вывод: Singleton = фундаментальное ограничение на 1 GPU!**

---

## 🟢 Новый подход: Multi-Instance + GPUManager (ИСПОЛЬЗУЕМ!)

### 1. DrvGPU - обычный класс (НЕ Singleton)

```cpp
// ✅ НОВЫЙ КОД - ИСПОЛЬЗУЕМ!

class DrvGPU {
public:
    /**
     * @brief Конструктор для конкретной GPU
     * @param device_id ID GPU устройства (0, 1, 2, ..., 7)
     * 
     * ВАЖНО: Теперь это ОБЫЧНЫЙ конструктор, не приватный!
     * Можем создать МНОЖЕСТВО экземпляров для разных GPU!
     */
    explicit DrvGPU(int device_id = 0) 
        : device_id_(device_id)
    {
        // Каждый экземпляр управляет СВОЕЙ GPU
    }
    
    ~DrvGPU() {
        shutdown();
    }
    
    // Запрет копирования (GPU контекст нельзя копировать)
    DrvGPU(const DrvGPU&) = delete;
    DrvGPU& operator=(const DrvGPU&) = delete;
    
    // Разрешено перемещение
    DrvGPU(DrvGPU&&) = default;
    DrvGPU& operator=(DrvGPU&&) = default;
    
    /**
     * @brief Инициализация ЭТОЙ GPU
     * @param backend_type Тип бэкенда (OpenCL/ROCm)
     */
    void initialize(BackendType backend_type) {
        // Инициализация для device_id_, указанного в конструкторе
        backend_ = BackendFactory::create(backend_type, device_id_);
        memory_manager_ = std::make_unique<MemoryManager>(backend_.get());
        module_registry_ = std::make_unique<ModuleRegistry>(backend_.get());
        
        initialized_ = true;
    }
    
    // Остальные методы как раньше
    MemoryManager& getMemoryManager() { return *memory_manager_; }
    ModuleRegistry& getModuleRegistry() { return *module_registry_; }
    IBackend& getBackend() { return *backend_; }
    
    int getDeviceId() const noexcept { return device_id_; }
    
    void shutdown() {
        if (initialized_) {
            module_registry_.reset();
            memory_manager_.reset();
            backend_.reset();
            initialized_ = false;
        }
    }
    
private:
    int device_id_;  // ID GPU, которой управляет ЭТОТ экземпляр
    bool initialized_ = false;
    
    std::unique_ptr<IBackend> backend_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<ModuleRegistry> module_registry_;
};
```

### Ключевые отличия от Singleton:

| Singleton (старый) | Multi-Instance (новый) |
|-------------------|------------------------|
| `static DrvGPU& getInstance()` | `explicit DrvGPU(int device_id)` |
| Приватный конструктор | Публичный конструктор |
| Только 1 экземпляр | Можно создать N экземпляров |
| `getInstance()` → всегда тот же объект | `DrvGPU(0)`, `DrvGPU(1)` → разные объекты |

---

### 2. Прямое использование DrvGPU (без GPUManager)

```cpp
// ✅ СПОСОБ 1: Прямое создание экземпляров для каждой GPU

#include <vector>
#include <memory>

int main() {
    // Создаем отдельный DrvGPU для каждой GPU
    std::vector<std::unique_ptr<DrvGPU>> gpus;
    
    for (int i = 0; i < 8; ++i) {
        auto gpu = std::make_unique<DrvGPU>(i);  // device_id = i
        gpu->initialize(BackendType::OpenCL);
        gpus.push_back(std::move(gpu));
    }
    
    // Теперь можем работать с каждой GPU независимо!
    
    // GPU #0
    auto& mem0 = gpus[0]->getMemoryManager();
    auto buffer0 = mem0.allocate(1024 * 1024);
    
    // GPU #5
    auto& mem5 = gpus[5]->getMemoryManager();
    auto buffer5 = mem5.allocate(1024 * 1024);
    
    // GPU #7
    auto& mem7 = gpus[7]->getMemoryManager();
    auto buffer7 = mem7.allocate(1024 * 1024);
    
    // ✅ ВСЕ 8 GPU работают ОДНОВРЕМЕННО!
    
    // Cleanup (автоматически через unique_ptr)
    return 0;
}
```

---

### 3. GPUManager - центральный координатор (РЕКОМЕНДУЕТСЯ)

```cpp
// ✅ СПОСОБ 2: Использование GPUManager (проще и удобнее)

class GPUManager {
public:
    GPUManager() = default;
    ~GPUManager() { shutdownAll(); }
    
    // Запрет копирования и перемещения (управляет ресурсами)
    GPUManager(const GPUManager&) = delete;
    GPUManager& operator=(const GPUManager&) = delete;
    GPUManager(GPUManager&&) = delete;
    GPUManager& operator=(GPUManager&&) = delete;
    
    /**
     * @brief Инициализация ВСЕХ доступных GPU в системе
     * @param backend_type Тип бэкенда (OpenCL/ROCm)
     * @return Количество успешно инициализированных GPU
     */
    int initializeAll(BackendType backend_type) {
        // 1. Определяем сколько GPU в системе
        int num_devices = detectDevices(backend_type);
        
        // 2. Создаем DrvGPU для каждой GPU
        for (int i = 0; i < num_devices; ++i) {
            try {
                auto drv = std::make_unique<DrvGPU>(i);  // Создаем экземпляр
                drv->initialize(backend_type);           // Инициализируем
                
                gpu_drivers_.push_back(std::move(drv)); // Сохраняем
                
                std::cout << "✓ GPU " << i << " initialized\n";
            } catch (const std::exception& e) {
                std::cerr << "✗ GPU " << i << " failed: " << e.what() << "\n";
            }
        }
        
        return gpu_drivers_.size();
    }
    
    /**
     * @brief Получение конкретной GPU по ID
     * @param device_id ID устройства (0..N-1)
     * @return Reference на DrvGPU для этой GPU
     */
    DrvGPU& getGPU(int device_id) {
        for (auto& drv : gpu_drivers_) {
            if (drv->getDeviceId() == device_id) {
                return *drv;
            }
        }
        throw std::out_of_range("GPU not found: " + std::to_string(device_id));
    }
    
    /**
     * @brief Round-robin выбор GPU
     */
    DrvGPU& selectRoundRobin() {
        if (gpu_drivers_.empty()) {
            throw std::runtime_error("No GPUs initialized");
        }
        
        int selected = next_round_robin_ % gpu_drivers_.size();
        next_round_robin_++;
        
        return *gpu_drivers_[selected];
    }
    
    /**
     * @brief Выбор наименее загруженной GPU
     */
    DrvGPU& selectLeastLoaded() {
        if (gpu_drivers_.empty()) {
            throw std::runtime_error("No GPUs initialized");
        }
        
        size_t min_usage = SIZE_MAX;
        DrvGPU* best_gpu = nullptr;
        
        for (auto& drv : gpu_drivers_) {
            auto stats = drv->getMemoryManager().getStats();
            if (stats.total_allocated < min_usage) {
                min_usage = stats.total_allocated;
                best_gpu = drv.get();
            }
        }
        
        return *best_gpu;
    }
    
    /**
     * @brief Получение всех GPU для параллельной обработки
     */
    std::vector<DrvGPU*> getAllGPUs() {
        std::vector<DrvGPU*> result;
        for (auto& drv : gpu_drivers_) {
            result.push_back(drv.get());
        }
        return result;
    }
    
    /**
     * @brief Синхронизация всех GPU
     */
    void synchronizeAll() {
        for (auto& drv : gpu_drivers_) {
            drv->synchronize();
        }
    }
    
    /**
     * @brief Shutdown всех GPU
     */
    void shutdownAll() {
        for (auto& drv : gpu_drivers_) {
            drv->shutdown();
        }
        gpu_drivers_.clear();
    }
    
    int getDeviceCount() const { return gpu_drivers_.size(); }
    
private:
    // Пул DrvGPU экземпляров (по одному на каждую GPU)
    std::vector<std::unique_ptr<DrvGPU>> gpu_drivers_;
    
    int next_round_robin_ = 0;
    
    int detectDevices(BackendType backend_type) {
        // OpenCL device detection
        if (backend_type == BackendType::OpenCL) {
            cl_uint num_platforms;
            clGetPlatformIDs(0, nullptr, &num_platforms);
            
            if (num_platforms == 0) return 0;
            
            std::vector<cl_platform_id> platforms(num_platforms);
            clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
            
            cl_uint num_devices;
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 
                           0, nullptr, &num_devices);
            
            return num_devices;
        }
        
        return 0;
    }
};
```

---

## 📝 Примеры использования нового подхода

### Пример 1: Базовое использование с GPUManager

```cpp
#include <gpu_lib/GPUManager.hpp>
#include <iostream>

int main() {
    // ✅ Создаем GPUManager
    GPUManager manager;
    
    // Инициализируем все GPU в системе
    int num_gpus = manager.initializeAll(BackendType::OpenCL);
    std::cout << "Initialized " << num_gpus << " GPUs\n";
    
    // Получаем информацию о каждой GPU
    for (int i = 0; i < num_gpus; ++i) {
        auto& gpu = manager.getGPU(i);
        auto info = gpu.getDeviceInfo();
        
        std::cout << "GPU " << i << ": " << info.name << "\n";
        std::cout << "  Memory: " << info.global_memory_size / (1024*1024*1024) << " GB\n";
    }
    
    // Работа с конкретными GPU
    auto& gpu0 = manager.getGPU(0);  // Прямой доступ к GPU #0
    auto& gpu5 = manager.getGPU(5);  // Прямой доступ к GPU #5
    
    // Выделяем память на разных GPU
    auto& mem0 = gpu0.getMemoryManager();
    auto buffer0 = mem0.allocate(1024 * 1024);  // На GPU #0
    
    auto& mem5 = gpu5.getMemoryManager();
    auto buffer5 = mem5.allocate(1024 * 1024);  // На GPU #5
    
    // ✅ Обе GPU работают одновременно!
    
    manager.shutdownAll();
    return 0;
}
```

### Пример 2: Round-Robin распределение задач

```cpp
#include <gpu_lib/GPUManager.hpp>
#include <vector>

struct DataChunk {
    std::vector<float> data;
    size_t size() const { return data.size() * sizeof(float); }
};

void processDataBatch(const std::vector<DataChunk>& chunks) {
    GPUManager manager;
    manager.initializeAll(BackendType::OpenCL);
    
    // Обрабатываем каждый chunk на следующей GPU (циклически)
    for (const auto& chunk : chunks) {
        // ✅ Выбираем следующую GPU по кругу (0→1→2→...→7→0→1→...)
        auto& gpu = manager.selectRoundRobin();
        
        std::cout << "Processing chunk on GPU " << gpu.getDeviceId() << "\n";
        
        // Выделяем память на выбранной GPU
        auto& mem = gpu.getMemoryManager();
        auto buffer = mem.allocate(chunk.size());
        buffer->copyFrom(chunk.data.data(), chunk.size());
        
        // Запускаем kernel на этой GPU
        // ... kernel execution ...
    }
    
    // Ждем завершения всех GPU
    manager.synchronizeAll();
}
```

### Пример 3: Load Balancing (наименее загруженная GPU)

```cpp
#include <gpu_lib/GPUManager.hpp>

void smartProcessing(const std::vector<Task>& tasks) {
    GPUManager manager;
    manager.initializeAll(BackendType::OpenCL);
    
    for (const auto& task : tasks) {
        // ✅ Выбираем GPU с минимальной загрузкой памяти
        auto& gpu = manager.selectLeastLoaded();
        
        std::cout << "Assigning task to GPU " << gpu.getDeviceId() 
                  << " (least loaded)\n";
        
        // Обрабатываем на выбранной GPU
        auto& mem = gpu.getMemoryManager();
        auto buffer = mem.allocate(task.dataSize());
        
        // ... обработка ...
    }
    
    // Выводим статистику по всем GPU
    auto stats = manager.getTotalMemoryStats();
    std::cout << "\n=== Memory Statistics ===\n";
    std::cout << "Total used across all GPUs: " 
              << stats.total_allocated_all_gpus / (1024*1024) << " MB\n";
    
    for (size_t i = 0; i < stats.per_gpu_stats.size(); ++i) {
        std::cout << "GPU " << i << ": "
                  << stats.per_gpu_stats[i].total_allocated / (1024*1024) 
                  << " MB used\n";
    }
}
```

### Пример 4: Параллельная обработка на ВСЕХ GPU

```cpp
#include <gpu_lib/GPUManager.hpp>
#include <thread>
#include <vector>

struct DataSet {
    std::vector<float> data;
    size_t size() const { return data.size() * sizeof(float); }
};

void parallelProcessing(const std::vector<DataSet>& datasets) {
    GPUManager manager;
    int num_gpus = manager.initializeAll(BackendType::OpenCL);
    
    // Получаем все GPU
    auto all_gpus = manager.getAllGPUs();
    
    std::vector<std::thread> threads;
    
    // ✅ Создаем отдельный поток для каждой GPU
    for (int i = 0; i < num_gpus && i < datasets.size(); ++i) {
        threads.emplace_back([&, i]() {
            DrvGPU& gpu = *all_gpus[i];
            const DataSet& data = datasets[i];
            
            std::cout << "Thread " << i << " processing on GPU " 
                      << gpu.getDeviceId() << "\n";
            
            // Обработка на GPU #i
            auto& mem = gpu.getMemoryManager();
            auto buffer = mem.allocate(data.size());
            buffer->copyFrom(data.data.data(), data.size());
            
            // ... kernel execution ...
            
            gpu.synchronize();
            std::cout << "GPU " << gpu.getDeviceId() << " finished\n";
        });
    }
    
    // Ждем завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "\n✓ All " << num_gpus << " GPUs completed processing\n";
    manager.shutdownAll();
}
```

---

## 🔄 Миграция с Singleton на Multi-Instance

### Шаг 1: Изменения в DrvGPU.hpp

```cpp
// ❌ БЫЛО (Singleton):
class DrvGPU {
public:
    static DrvGPU& getInstance() {
        static DrvGPU instance;
        return instance;
    }
    
private:
    DrvGPU() = default;  // Приватный конструктор
};

// ✅ СТАЛО (Multi-Instance):
class DrvGPU {
public:
    explicit DrvGPU(int device_id = 0);  // Публичный конструктор
    
    ~DrvGPU();
    
    DrvGPU(const DrvGPU&) = delete;
    DrvGPU& operator=(const DrvGPU&) = delete;
    DrvGPU(DrvGPU&&) = default;
    DrvGPU& operator=(DrvGPU&&) = default;
    
    // НЕТ getInstance() метода!
};
```

### Шаг 2: Обновление пользовательского кода

```cpp
// ❌ СТАРЫЙ КОД:
auto& drv = DrvGPU::getInstance();
drv.initialize(BackendType::OpenCL, 0);

// ✅ НОВЫЙ КОД (вариант 1 - прямое использование):
DrvGPU drv(0);  // device_id = 0
drv.initialize(BackendType::OpenCL);

// ✅ НОВЫЙ КОД (вариант 2 - через GPUManager, РЕКОМЕНДУЕТСЯ):
GPUManager manager;
manager.initializeAll(BackendType::OpenCL);
auto& drv = manager.getGPU(0);
```

### Шаг 3: Обратная совместимость (опционально)

Если нужно временно поддержать старый код:

```cpp
// DrvGPU.hpp - добавить deprecated метод
class DrvGPU {
public:
    /**
     * @brief DEPRECATED: Для обратной совместимости
     * @deprecated Используйте GPUManager вместо этого
     */
    [[deprecated("Use GPUManager::getGPU(0) instead")]]
    static DrvGPU& getInstance() {
        static GPUManager internal_manager;
        static bool initialized = false;
        
        if (!initialized) {
            internal_manager.initializeAll(BackendType::OpenCL);
            initialized = true;
        }
        
        return internal_manager.getGPU(0);  // Всегда GPU #0
    }
    
    // ... остальное
};
```

**Результат:**
- Старый код работает (с warning)
- Новый код использует multi-GPU
- Постепенная миграция возможна

---

## 📊 Сравнение производительности

### Singleton (1 GPU)

```
Task 1 → GPU #0 (100ms)
Task 2 → GPU #0 (100ms)
Task 3 → GPU #0 (100ms)
Task 4 → GPU #0 (100ms)
Task 5 → GPU #0 (100ms)
Task 6 → GPU #0 (100ms)
Task 7 → GPU #0 (100ms)
Task 8 → GPU #0 (100ms)

Общее время: 800ms
```

### Multi-Instance + GPUManager (8 GPU)

```
Task 1 → GPU #0 (100ms) ┐
Task 2 → GPU #1 (100ms) │
Task 3 → GPU #2 (100ms) │
Task 4 → GPU #3 (100ms) ├─ Параллельно!
Task 5 → GPU #4 (100ms) │
Task 6 → GPU #5 (100ms) │
Task 7 → GPU #6 (100ms) │
Task 8 → GPU #7 (100ms) ┘

Общее время: ~100ms (8x ускорение!)
```

**Выигрыш: 8x производительность при 8 GPU!**

---

## ✅ Итоговые рекомендации для команды

### Что использовать:

1. **GPUManager** - для большинства случаев (рекомендуется)
   ```cpp
   GPUManager manager;
   manager.initializeAll(BackendType::OpenCL);
   auto& gpu = manager.selectLeastLoaded();
   ```

2. **Прямое создание DrvGPU** - когда нужен полный контроль
   ```cpp
   DrvGPU gpu0(0);
   DrvGPU gpu5(5);
   gpu0.initialize(BackendType::OpenCL);
   gpu5.initialize(BackendType::OpenCL);
   ```

### Что НЕ использовать:

❌ **Singleton pattern (`getInstance()`)** - ограничен 1 GPU, плохо для тестов

---

## 🎯 Заключение

**Ключевые изменения:**

1. **DrvGPU больше НЕ Singleton**
   - Можно создать экземпляр для каждой GPU
   - `DrvGPU(device_id)` вместо `getInstance()`

2. **GPUManager - новый центральный координатор**
   - Упрощает работу с несколькими GPU
   - Load balancing, round-robin, direct access

3. **Масштабируемость**
   - От 1 GPU до 8+ GPU без изменения кода модулей
   - Линейное ускорение производительности

4. **Изоляция**
   - Каждая GPU независима
   - Ошибка на одной не влияет на другие

**Следующие шаги:**
- Изучить примеры использования
- Обновить существующий код (если есть)
- Тестировать на multi-GPU системах

**Вопросы?** Обращайтесь к этому документу! 🚀
