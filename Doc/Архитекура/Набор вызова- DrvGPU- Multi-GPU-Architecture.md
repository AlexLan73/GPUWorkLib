# Multi-GPU Architecture Extension для LibGPU

## Проблема: Singleton и Multiple GPUs

### Текущее ограничение

```cpp
// Текущая архитектура: Singleton = 1 GPU
auto& drv = DrvGPU::getInstance();  // Только один экземпляр!
drv.initialize(BackendType::OpenCL, 0);  // device_id=0
```

**Проблема**: Singleton паттерн = только один экземпляр DrvGPU = только одна GPU в системе.

Если у вас 8 GPU:
- ❌ Нельзя использовать DrvGPU для GPU #1, #2, ..., #7
- ❌ Нельзя распараллелить задачи между GPUs
- ❌ Теряется масштабируемость

---

## Решение 1: Multi-Instance Pattern (рекомендуется)

### Архитектура

**Убрать Singleton, использовать обычные экземпляры:**

```cpp
// НЕ Singleton! Обычный класс
class DrvGPU {
public:
    // Обычный конструктор (не приватный)
    explicit DrvGPU(int device_id = 0);
    
    // Copyable = удалено, Movable = разрешено
    DrvGPU(const DrvGPU&) = delete;
    DrvGPU& operator=(const DrvGPU&) = delete;
    DrvGPU(DrvGPU&&) = default;
    DrvGPU& operator=(DrvGPU&&) = default;
    
    ~DrvGPU();
    
    void initialize(BackendType backend_type);
    void shutdown();
    
    // Все остальное как раньше
    MemoryManager& getMemoryManager();
    ModuleRegistry& getModuleRegistry();
    IBackend& getBackend();
    
    int getDeviceId() const { return device_id_; }
    
private:
    int device_id_;
    // ... остальные члены
};
```

### Использование: Создание экземпляров для каждой GPU

```cpp
#include <gpu_lib/DrvGPU.hpp>
#include <vector>
#include <thread>

int main() {
    // Определяем количество GPU в системе
    int num_gpus = 8;  // Или через clGetDeviceIDs()
    
    // Создаем отдельный DrvGPU для каждой GPU
    std::vector<std::unique_ptr<DrvGPU>> gpu_drivers;
    
    for (int i = 0; i < num_gpus; ++i) {
        auto drv = std::make_unique<DrvGPU>(i);  // device_id = i
        drv->initialize(BackendType::OpenCL);
        
        std::cout << "Initialized GPU " << i << ": " 
                  << drv->getDeviceInfo().name << "\n";
        
        gpu_drivers.push_back(std::move(drv));
    }
    
    // Теперь можно использовать каждую GPU независимо
    
    // Пример: Параллельная обработка на всех GPUs
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_gpus; ++i) {
        threads.emplace_back([&, i]() {
            auto& drv = *gpu_drivers[i];
            
            // Работа на GPU #i
            auto& mem = drv.getMemoryManager();
            auto buffer = mem.allocate(1024 * 1024);
            
            // ... обработка данных на этой GPU
            
            std::cout << "GPU " << i << " processing complete\n";
        });
    }
    
    // Ждем завершения всех GPU
    for (auto& t : threads) {
        t.join();
    }
    
    // Shutdown всех GPU
    for (auto& drv : gpu_drivers) {
        drv->shutdown();
    }
    
    return 0;
}
```

---

## Решение 2: GPUManager (Facade для Multi-GPU)

### Архитектура

**Обертка над несколькими DrvGPU экземплярами:**

```cpp
// gpu_lib/MultiGPU.hpp
#pragma once

#include <gpu_lib/DrvGPU.hpp>
#include <vector>
#include <memory>

namespace gpu_lib {
namespace core {

/**
 * @brief Менеджер для управления несколькими GPU
 * 
 * Design Pattern: Facade (упрощает работу с multi-GPU)
 */
class GPUManager {
public:
    GPUManager() = default;
    ~GPUManager() { shutdownAll(); }
    
    /**
     * @brief Инициализация всех доступных GPU
     * @param backend_type Тип бэкенда (OpenCL/ROCm)
     * @return Количество инициализированных GPU
     */
    int initializeAll(BackendType backend_type) {
        int num_devices = detectDevices(backend_type);
        
        for (int i = 0; i < num_devices; ++i) {
            auto drv = std::make_unique<DrvGPU>(i);
            
            try {
                drv->initialize(backend_type);
                gpu_drivers_.push_back(std::move(drv));
            } catch (const std::exception& e) {
                std::cerr << "Failed to initialize GPU " << i 
                          << ": " << e.what() << "\n";
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
        if (device_id < 0 || device_id >= gpu_drivers_.size()) {
            throw std::out_of_range("Invalid device_id");
        }
        return *gpu_drivers_[device_id];
    }
    
    /**
     * @brief Количество доступных GPU
     */
    int getDeviceCount() const {
        return gpu_drivers_.size();
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
     * @brief Выбор GPU с минимальной загрузкой памяти
     */
    DrvGPU& selectLeastLoaded() {
        if (gpu_drivers_.empty()) {
            throw std::runtime_error("No GPUs initialized");
        }
        
        size_t min_usage = SIZE_MAX;
        int best_gpu = 0;
        
        for (int i = 0; i < gpu_drivers_.size(); ++i) {
            auto stats = gpu_drivers_[i]->getMemoryManager().getStats();
            if (stats.total_allocated < min_usage) {
                min_usage = stats.total_allocated;
                best_gpu = i;
            }
        }
        
        return *gpu_drivers_[best_gpu];
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
     * @brief Shutdown всех GPU
     */
    void shutdownAll() {
        for (auto& drv : gpu_drivers_) {
            drv->shutdown();
        }
        gpu_drivers_.clear();
    }
    
private:
    std::vector<std::unique_ptr<DrvGPU>> gpu_drivers_;
    int next_round_robin_ = 0;
    
    int detectDevices(BackendType backend_type) {
        // OpenCL example
        if (backend_type == BackendType::OpenCL) {
            cl_uint num_platforms;
            clGetPlatformIDs(0, nullptr, &num_platforms);
            
            if (num_platforms == 0) return 0;
            
            std::vector<cl_platform_id> platforms(num_platforms);
            clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
            
            cl_uint num_devices;
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
            
            return num_devices;
        }
        
        return 0;
    }
};

} // namespace core
} // namespace gpu_lib
```

### Использование GPUManager

```cpp
#include <gpu_lib/MultiGPU.hpp>

int main() {
    gpu_lib::core::GPUManager gpu_manager;
    
    // Инициализация всех GPU
    int num_gpus = gpu_manager.initializeAll(gpu_lib::BackendType::OpenCL);
    std::cout << "Initialized " << num_gpus << " GPUs\n";
    
    // Способ 1: Прямой доступ к конкретной GPU
    auto& gpu0 = gpu_manager.getGPU(0);
    auto& gpu1 = gpu_manager.getGPU(1);
    
    // Работа на разных GPU
    auto& mem0 = gpu0.getMemoryManager();
    auto& mem1 = gpu1.getMemoryManager();
    
    auto buffer0 = mem0.allocate(1024 * 1024);  // На GPU #0
    auto buffer1 = mem1.allocate(1024 * 1024);  // На GPU #1
    
    // Способ 2: Round-robin распределение
    for (int i = 0; i < 16; ++i) {
        auto& gpu = gpu_manager.selectRoundRobin();  // Циклическая выборка
        
        // Обработка на выбранной GPU
        processData(gpu, data[i]);
    }
    
    // Способ 3: Load balancing (наименее загруженная GPU)
    auto& best_gpu = gpu_manager.selectLeastLoaded();
    processHeavyTask(best_gpu);
    
    // Способ 4: Параллельная обработка на всех GPU
    auto all_gpus = gpu_manager.getAllGPUs();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < all_gpus.size(); ++i) {
        threads.emplace_back([&, i]() {
            processOnGPU(*all_gpus[i], data_chunks[i]);
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Shutdown
    gpu_manager.shutdownAll();
    
    return 0;
}
```

---

## Решение 3: Thread-Local Storage (для thread-per-GPU)

### Концепция

Каждый поток работает со своей GPU через thread-local DrvGPU:

```cpp
#include <gpu_lib/DrvGPU.hpp>
#include <thread>
#include <vector>

// Thread-local переменная для текущей GPU
thread_local std::unique_ptr<DrvGPU> g_thread_gpu;

/**
 * @brief Инициализация GPU для текущего потока
 */
void initThreadGPU(int device_id) {
    g_thread_gpu = std::make_unique<DrvGPU>(device_id);
    g_thread_gpu->initialize(BackendType::OpenCL);
}

/**
 * @brief Получение GPU для текущего потока
 */
DrvGPU& getThreadGPU() {
    if (!g_thread_gpu) {
        throw std::runtime_error("GPU not initialized for this thread");
    }
    return *g_thread_gpu;
}

/**
 * @brief Shutdown GPU для текущего потока
 */
void shutdownThreadGPU() {
    if (g_thread_gpu) {
        g_thread_gpu->shutdown();
        g_thread_gpu.reset();
    }
}

// Использование
void workerThread(int gpu_id, const std::vector<float>& data) {
    // Инициализация GPU для этого потока
    initThreadGPU(gpu_id);
    
    auto& drv = getThreadGPU();
    auto& mem = drv.getMemoryManager();
    
    // Работа с GPU
    auto buffer = mem.allocate(data.size() * sizeof(float));
    buffer->copyFrom(data.data(), data.size() * sizeof(float));
    
    // ... обработка ...
    
    // Cleanup
    shutdownThreadGPU();
}

int main() {
    int num_gpus = 8;
    std::vector<std::thread> threads;
    
    // Создаем поток для каждой GPU
    for (int i = 0; i < num_gpus; ++i) {
        threads.emplace_back(workerThread, i, data_chunks[i]);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    return 0;
}
```

---

## Решение 4: Task Queue с GPU Pool

### Архитектура

Пул GPU с очередью задач:

```cpp
// gpu_lib/GPUPool.hpp
#pragma once

#include <gpu_lib/DrvGPU.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace gpu_lib {
namespace core {

/**
 * @brief Пул GPU с очередью задач
 * 
 * Каждая GPU имеет свой worker thread
 * Задачи автоматически распределяются между GPU
 */
class GPUPool {
public:
    using Task = std::function<void(DrvGPU&)>;
    
    explicit GPUPool(int num_gpus, BackendType backend_type) {
        // Инициализация GPU
        for (int i = 0; i < num_gpus; ++i) {
            auto drv = std::make_unique<DrvGPU>(i);
            drv->initialize(backend_type);
            gpu_drivers_.push_back(std::move(drv));
        }
        
        // Запуск worker threads
        for (int i = 0; i < num_gpus; ++i) {
            workers_.emplace_back([this, i]() {
                workerLoop(i);
            });
        }
    }
    
    ~GPUPool() {
        shutdown();
    }
    
    /**
     * @brief Добавление задачи в очередь
     */
    void enqueue(Task task) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.push(std::move(task));
        }
        queue_cv_.notify_one();
    }
    
    /**
     * @brief Ожидание завершения всех задач
     */
    void waitAll() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        done_cv_.wait(lock, [this]() {
            return task_queue_.empty() && active_tasks_ == 0;
        });
    }
    
    /**
     * @brief Shutdown пула
     */
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            shutdown_ = true;
        }
        queue_cv_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        for (auto& drv : gpu_drivers_) {
            drv->shutdown();
        }
    }
    
private:
    std::vector<std::unique_ptr<DrvGPU>> gpu_drivers_;
    std::vector<std::thread> workers_;
    
    std::queue<Task> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable done_cv_;
    
    int active_tasks_ = 0;
    bool shutdown_ = false;
    
    void workerLoop(int gpu_id) {
        auto& drv = *gpu_drivers_[gpu_id];
        
        while (true) {
            Task task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                
                queue_cv_.wait(lock, [this]() {
                    return shutdown_ || !task_queue_.empty();
                });
                
                if (shutdown_ && task_queue_.empty()) {
                    break;
                }
                
                task = std::move(task_queue_.front());
                task_queue_.pop();
                active_tasks_++;
            }
            
            // Выполнение задачи на этой GPU
            try {
                task(drv);
            } catch (const std::exception& e) {
                std::cerr << "GPU " << gpu_id << " task failed: " 
                          << e.what() << "\n";
            }
            
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                active_tasks_--;
                
                if (task_queue_.empty() && active_tasks_ == 0) {
                    done_cv_.notify_all();
                }
            }
        }
    }
};

} // namespace core
} // namespace gpu_lib
```

### Использование GPUPool

```cpp
#include <gpu_lib/GPUPool.hpp>

int main() {
    // Создание пула с 8 GPU
    gpu_lib::core::GPUPool gpu_pool(8, gpu_lib::BackendType::OpenCL);
    
    // Добавление задач (автоматически распределяются между GPU)
    for (int i = 0; i < 100; ++i) {
        gpu_pool.enqueue([i](DrvGPU& drv) {
            // Эта задача будет выполнена на одной из 8 GPU
            auto& mem = drv.getMemoryManager();
            auto buffer = mem.allocate(1024 * 1024);
            
            // ... обработка ...
            
            std::cout << "Task " << i << " completed on GPU " 
                      << drv.getDeviceId() << "\n";
        });
    }
    
    // Ожидание завершения всех задач
    gpu_pool.waitAll();
    
    std::cout << "All tasks completed\n";
    
    // Shutdown (автоматически в деструкторе)
    
    return 0;
}
```

---

## Сравнение решений

| Решение | Сложность | Гибкость | Use Case |
|---------|-----------|----------|----------|
| **Multi-Instance** | Низкая | Высокая | Полный контроль, разные алгоритмы на разных GPU |
| **GPUManager** | Средняя | Высокая | Удобный API, load balancing |
| **Thread-Local** | Средняя | Средняя | Thread-per-GPU модель |
| **GPUPool** | Высокая | Средняя | Task queue, автоматическое распределение |

---

## Рекомендации

### Для вашего проекта:

**Рекомендую: GPUManager (Решение 2)**

**Причины:**
1. ✅ Простой API для пользователя
2. ✅ Встроенный load balancing
3. ✅ Гибкий выбор GPU (round-robin, least loaded, direct)
4. ✅ Легко расширить функциональность
5. ✅ Совместимость с существующим кодом

**Реализация:**

```cpp
// 1. Убрать Singleton из DrvGPU (сделать обычный класс)
// 2. Добавить GPUManager в core/
// 3. Обновить примеры использования
```

**Backward compatibility** (если нужна):

```cpp
// Оставить старый API через wrapper
class DrvGPU {
public:
    // DEPRECATED: Singleton API (для совместимости)
    [[deprecated("Use GPUManager instead")]]
    static DrvGPU& getInstance() {
        static GPUManager manager;
        static bool initialized = false;
        
        if (!initialized) {
            manager.initializeAll(BackendType::OpenCL);
            initialized = true;
        }
        
        return manager.getGPU(0);  // Всегда возвращает GPU #0
    }
    
    // Новый API: обычный конструктор
    explicit DrvGPU(int device_id);
};
```

---

## Заключение

**Singleton был правильным решением для MVP**, но для production с 8 GPU нужно:

1. ✅ Убрать Singleton
2. ✅ Сделать DrvGPU обычным классом
3. ✅ Добавить GPUManager для удобного управления
4. ✅ Обновить документацию с multi-GPU примерами

**Следующий шаг**: Хотите, чтобы я обновил файлы с новой архитектурой?
