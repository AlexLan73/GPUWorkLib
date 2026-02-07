# DrvGPU Справочник классов

## Оглавление

1. [Core Layer](#core-layer)
2. [Backend Layer](#backend-layer)
3. [Memory Layer](#memory-layer)
4. [Common Services](#common-services)
5. [Interfaces](#interfaces)
6. [Сводные таблицы](#сводные-таблицы)

---

## Core Layer

Классы этого слоя предоставляют высокоуровневый API для работы с GPU.

### DrvGPU

**Файл**: [`drv_gpu.hpp`](../../include/DrvGPU/drv_gpu.hpp), [`drv_gpu.cpp`](../../include/DrvGPU/drv_gpu.cpp)

**Назначение**: Главный класс библиотеки, фасад для всех операций с GPU.

**Иерархия наследования**: Нет (Composition-based)

**Ключевые методы**:

| Метод | Описание | Сложность |
|-------|----------|-----------|
| `DrvGPU(BackendType, int)` | Конструктор | O(1) |
| `Initialize()` | Инициализировать GPU | O(n) |
| `Cleanup()` | Очистить ресурсы | O(n) |
| `GetDeviceInfo()` | Получить информацию об устройстве | O(1) |
| `GetMemoryManager()` | Получить менеджер памяти | O(1) |
| `GetModuleRegistry()` | Получить регистр модулей | O(1) |
| `GetBackend()` | Получить бэкенд | O(1) |
| `Synchronize()` | Синхронизировать | O(1) |
| `Flush()` | Flush команд | O(1) |

**Члены класса**:

| Член | Тип | Описание |
|------|-----|----------|
| `backend_type_` | `BackendType` | Тип бэкенда |
| `device_index_` | `int` | Индекс устройства |
| `initialized_` | `bool` | Флаг инициализации |
| `backend_` | `unique_ptr<IBackend>` | Бэкенд |
| `memory_manager_` | `unique_ptr<MemoryManager>` | Менеджер памяти |
| `module_registry_` | `unique_ptr<ModuleRegistry>` | Регистр модулей |
| `mutex_` | `mutable std::mutex` | Мьютекс |

**Паттерн**: Facade, Bridge, RAII, Move Semantics

**См. также**: [Architecture.md](Architecture.md)

---

### GPUManager

**Файл**: [`gpu_manager.hpp`](../../include/DrvGPU/gpu_manager.hpp)

**Назначение**: Фасад для управления несколькими GPU с поддержкой load balancing.

> ✅ **Multi-GPU (v2.0)**: Теперь `DiscoverGPUs()` использует `OpenCLCore::GetAvailableDeviceCount()` для реального обнаружения устройств!

**Иерархия наследования**: Нет (Header-only)

**Ключевые методы**:

| Метод | Описание |
|-------|----------|
| `GPUManager()` | Конструктор |
| `InitializeAll(BackendType)` | ✅ Инициализировать ВСЕ найденные GPU |
| `InitializeSpecific(BackendType, vector<int>)` | Инициализировать конкретные GPU |
| `GetGPU(size_t index)` | Получить GPU по индексу |
| `GetNextGPU()` | Получить следующую GPU (Round-Robin) |
| `GetLeastLoadedGPU()` | Получить наименее загруженную GPU |
| `SetLoadBalancingStrategy(LoadBalancingStrategy)` | Установить стратегию |
| `SynchronizeAll()` | Синхронизировать все |
| `FlushAll()` | Flush всех |
| `GetGPUCount()` | ✅ Реальное количество GPU |
| `GetAvailableGPUCount(BackendType)` | ✅ **Static**: Количество GPU в системе |

**Стратегии Load Balancing**:

| Стратегия | Описание |
|-----------|----------|
| `ROUND_ROBIN` | Циклический выбор |
| `LEAST_LOADED` | Наименее загруженная GPU |
| `MANUAL` | Ручной выбор |
| `FASTEST_FIRST` | Сначала самая быстрая |

**Паттерн**: Facade, Factory, Strategy

**См. также**: [Architecture.md](Architecture.md)

---

### MemoryManager

**Файл**: [`memory_manager.hpp`](../../include/DrvGPU/memory/memory_manager.hpp)

**Назначение**: Менеджер памяти GPU с отслеживанием статистики аллокаций.

**Иерархия наследования**: Нет

**Ключевые методы**:

| Метод | Описание |
|-------|----------|
| `MemoryManager(IBackend*)` | Конструктор |
| `Allocate(size_t, unsigned int)` | Выделить память |
| `Free(void*)` | Освободить память |
| `AllocateBuffer(size_t, MemoryType)` | Выделить буфер |
| `AllocateSVM(size_t)` | Выделить SVM буфер |
| `GetAllocationCount()` | Количество аллокаций |
| `GetTotalAllocatedBytes()` | Всего выделено байт |
| `GetPeakMemoryUsage()` | Пик использования памяти |
| `PrintStatistics()` | Вывести статистику |
| `GetStatistics()` | Получить статистику |
| `ResetStatistics()` | Сбросить статистику |

**Статистика**:

| Поле | Описание |
|------|----------|
| `total_allocations_` | Всего аллокаций |
| `total_frees_` | Всего освобождений |
| `current_allocations_` | Текущих аллокаций |
| `total_bytes_allocated_` | Всего байт |
| `peak_bytes_allocated_` | Пик байт |

**См. также**: [Memory.md](Memory.md)

---

### ModuleRegistry

**Файл**: [`module_registry.hpp`](../../include/DrvGPU/module_registry.hpp), [`module_registry.cpp`](../../include/DrvGPU/module_registry.cpp)

**Назначение**: Регистр compute модулей с возможностью типизированного доступа.

**Иерархия наследования**: Нет

**Ключевые методы**:

| Метод | Описание |
|-------|----------|
| `RegisterModule(name, module)` | Зарегистрировать модуль |
| `UnregisterModule(name)` | Удалить модуль |
| `HasModule(name)` | Проверить наличие |
| `GetModule(name)` | Получить модуль |
| `GetModule<T>(name)` | Получить типизированный модуль |
| `GetModuleCount()` | Количество модулей |
| `GetModuleNames()` | Список имён |
| `PrintModules()` | Вывести модули |
| `Clear()` | Очистить все |

**Паттерн**: Registry, Template Method

---

## Backend Layer

Классы этого слоя реализуют абстракцию над конкретными GPU API.

**См. также**: [OpenCL.md](OpenCL.md)

### IBackend (Интерфейс)

**Файл**: [`common/i_backend.hpp`](../../include/DrvGPU/common/i_backend.hpp)

**Назначение**: Абстрактный интерфейс для всех бэкендов.

**Методы интерфейса**:

| Категория | Методы |
|-----------|--------|
| Lifecycle | `Initialize`, `Cleanup`, `IsInitialized` |
| Device Info | `GetType`, `GetDeviceInfo`, `GetDeviceIndex`, `GetDeviceName` |
| Native Handles | `GetNativeContext`, `GetNativeDevice`, `GetNativeQueue` |
| Memory | `Allocate`, `Free`, `MemcpyHostToDevice`, `MemcpyDeviceToHost`, `MemcpyDeviceToDevice` |
| Sync | `Synchronize`, `Flush` |
| Capabilities | `SupportsSVM`, `SupportsDoublePrecision`, `GetMaxWorkGroupSize`, `GetGlobalMemorySize`, `GetLocalMemorySize` |

**Паттерн**: Bridge, Template Method

---

### OpenCLBackend

**Файл**: [`opencl/opencl_backend.hpp`](../../include/DrvGPU/backends/opencl/opencl_backend.hpp), [`opencl_backend.cpp`](../../include/DrvGPU/backends/opencl/opencl_backend.cpp)

**Назначение**: Реализация IBackend для OpenCL.

**Иерархия**: `IBackend` → `OpenCLBackend`

**Члены класса**:

| Член | Тип | Описание |
|------|-----|----------|
| `context_` | `cl_context` | OpenCL контекст |
| `device_` | `cl_device_id` | Устройство |
| `queue_` | `cl_command_queue` | Очередь команд |
| `initialized_` | `bool` | Флаг инициализации |

**Ключевые методы**: см. [OpenCL.md](OpenCL.md)

---

### OpenCLCore

**Файл**: [`opencl/opencl_core.hpp`](../../include/DrvGPU/backends/opencl/opencl_core.hpp), [`opencl_core.cpp`](../../include/DrvGPU/backends/opencl/opencl_core.cpp)

**Назначение**: Управление OpenCL устройством. **Per-device architecture для Multi-GPU (v2.0)!**

> ⚠️ **ВАЖНО**: Singleton паттерн УДАЛЁН! Каждый экземпляр работает со СВОИМ устройством.

**Ключевые методы**:

| Метод | Описание |
|-------|----------|
| `OpenCLCore(device_index, device_type)` | Конструктор для конкретного устройства |
| `Initialize()` | Инициализировать контекст для ЭТОГО устройства |
| `GetAvailableDeviceCount()` | **Static**: Количество доступных GPU |
| `GetAllDevices()` | **Static**: Все (platform, device) пары |
| `GetAllDevicesInfo()` | **Static**: Информация для вывода |
| `GetContext()` | OpenCL контекст ЭТОГО устройства |
| `GetDevice()` | OpenCL device ЭТОГО устройства |

**Методы**: см. [OpenCL.md](OpenCL.md)

---

### OpenCLBackendExternal

**Файл**: [`opencl/opencl_backend_external.hpp`](../../include/DrvGPU/backends/opencl/opencl_backend_external.hpp), [`opencl_backend_external.cpp`](../../include/DrvGPU/backends/opencl/opencl_backend_external.cpp)

**Назначение**: External Context поддержка для OpenCL.

**Иерархия**: `OpenCLBackend` → `OpenCLBackendExternal`

**См. также**: [OpenCL.md](OpenCL.md)

---

### CommandQueuePool

**Файл**: [`opencl/command_queue_pool.hpp`](../../include/DrvGPU/backends/opencl/command_queue_pool.hpp), [`command_queue_pool.cpp`](../../include/DrvGPU/backends/opencl/command_queue_pool.cpp)

**Назначение**: Пул очередей команд для оптимизации производительности.

**Методы**:

| Метод | Описание |
|-------|----------|
| `CommandQueuePool(context, device)` | Создать пул |
| `AcquireQueue()` | Получить очередь |
| `ReleaseQueue(queue)` | Вернуть очередь |
| `GetPoolSize()` | Размер пула |
| `SetPoolSize(size)` | Установить размер |
| `GetAvailableCount()` | Доступных очередей |
| `SynchronizeAll()` | Синхронизировать все |
| `Clear()` | Очистить пул |

**Паттерн**: Object Pool

**См. также**: [Command.md](Command.md)

---

## Memory Layer

Классы этого слоя управляют памятью GPU.

**См. также**: [Memory.md](Memory.md)

### IMemoryBuffer (Интерфейс)

**Файл**: [`memory/i_memory_buffer.hpp`](../../include/DrvGPU/memory/i_memory_buffer.hpp)

**Назначение**: Интерфейс буфера памяти.

**Методы интерфейса**:

| Метод | Описание |
|-------|----------|
| `GetSize()` | Размер буфера |
| `GetType()` | Тип памяти |
| `GetDevicePointer()` | Указатель устройства |
| `CopyFromHost(data, size)` | Копировать с хоста |
| `CopyToHost(data, size)` | Копировать на хост |
| `CopyToDevice(dst, size)` | Копировать на устройство |
| `Map()` | Отобразить в память |
| `Unmap()` | Убрать отображение |
| `GetMappedPointer()` | Получить отображённый указатель |
| `Wait()` | Дождаться операций |
| `IsReady()` | Проверить готовность |

---

### GPUBuffer

**Файл**: [`memory/gpu_buffer.hpp`](../../include/DrvGPU/memory/gpu_buffer.hpp)

**Назначение**: Стандартный буфер памяти GPU.

**Иерархия**: `IMemoryBuffer` → `GPUBuffer`

**Члены класса**:

| Член | Тип | Описание |
|------|-----|----------|
| `backend_` | `IBackend*` | Бэкенд |
| `size_` | `size_t` | Размер |
| `type_` | `MemoryType` | Тип памяти |
| `device_ptr_` | `void*` | Указатель GPU |
| `host_ptr_` | `void*` | Указатель хоста |
| `mapped_` | `bool` | Флаг отображения |

**См. также**: [Memory.md](Memory.md)

---

### SVMBuffer

**Файл**: [`memory/svm_buffer.hpp`](../../include/DrvGPU/memory/svm_buffer.hpp)

**Назначение**: Shared Virtual Memory буфер.

**Иерархия**: `IMemoryBuffer` → `SVMBuffer`

**Особенности**:
- Хост и GPU используют один указатель
- Требует поддержки SVM от устройства
- Автоматическая синхронизация

**См. также**: [Memory.md](Memory.md)

---

### MemoryType

**Файл**: [`memory/memory_type.hpp`](../../include/DrvGPU/memory/memory_type.hpp)

**Назначение**: Типы памяти.

```cpp
enum class MemoryType {
    Device,       // Только устройство
    Host,         // Только хост
    Shared,       // Shared memory
    SVM,          // Shared Virtual Memory
    Unified       // Unified memory
};
```

---

### SVMCapabilities

**Файл**: [`memory/svm_capabilities.hpp`](../../include/DrvGPU/memory/svm_capabilities.hpp)

**Назначение**: Возможности SVM устройства.

```cpp
struct SVMCapabilities {
    bool supports_svm = false;
    bool supports_coarse_grain = false;
    bool supports_fine_grain = false;
    bool supports_fine_grain_system = false;
    bool supports_atomic = false;
};
```

**См. также**: [Memory.md](Memory.md)

---

### ExternalCLBufferAdapter

**Файл**: [`memory/external_cl_buffer_adapter.hpp`](../../include/DrvGPU/memory/external_cl_buffer_adapter.hpp)

**Назначение**: Адаптер для внешних OpenCL буферов.

**Иерархия**: `IMemoryBuffer` → `ExternalCLBufferAdapter`

**См. также**: [Memory.md](Memory.md)

---

## Common Services

### Logger

**Файл**: [`common/logger.hpp`](../../include/DrvGPU/common/logger.hpp), [`logger.cpp`](../../include/DrvGPU/common/logger.cpp)

**Назначение**: Фасад системы логирования.

**Статические методы**:

| Метод | Описание |
|-------|----------|
| `GetInstance()` | Получить текущий логер |
| `SetInstance(logger)` | Установить свой логер |
| `ResetToDefault()` | Сбросить к DefaultLogger |
| `Debug(component, message)` | Логировать debug |
| `Info(component, message)` | Логировать info |
| `Warning(component, message)` | Логировать warning |
| `Error(component, message)` | Логировать error |
| `IsEnabled()` | Проверить включение |
| `Enable()` | Включить |
| `Disable()` | Выключить |

**Макросы**:
```cpp
DRVGPU_LOG_DEBUG(component, message)   // Только в Debug
DRVGPU_LOG_INFO(component, message)    // Всегда активен
DRVGPU_LOG_WARNING(component, message) // Всегда активен
DRVGPU_LOG_ERROR(component, message)   // Всегда активен
```

---

### ILogger (Интерфейс)

**Файл**: [`common/logger_interface.hpp`](../../include/DrvGPU/common/logger_interface.hpp)

**Назначение**: Интерфейс логирования для реализации собственных логеров.

**Методы интерфейса**:

| Метод | Описание |
|-------|----------|
| `Debug(component, message)` | Debug сообщение |
| `Info(component, message)` | Info сообщение |
| `Warning(component, message)` | Warning сообщение |
| `Error(component, message)` | Error сообщение |
| `IsDebugEnabled()` | Debug активен? |
| `IsInfoEnabled()` | Info активен? |
| `IsWarningEnabled()` | Warning активен? |
| `IsErrorEnabled()` | Error активен? |
| `Reset()` | Сбросить состояние |

---

### DefaultLogger

**Файл**: [`common/default_logger.hpp`](../../include/DrvGPU/common/default_logger.hpp), [`default_logger.cpp`](../../include/DrvGPU/common/default_logger.cpp)

**Назначение**: Реализация ILogger на основе spdlog.

**Иерархия**: `ILogger` → `DefaultLogger`

**Особенности**:
- Только файловый вывод
- Thread-safe
- Автоматическое создание структуры папок

---

### ConfigLogger

**Файл**: [`common/config_logger.hpp`](../../include/DrvGPU/common/config_logger.hpp), [`config_logger.cpp`](../../include/DrvGPU/common/config_logger.cpp)

**Назначение**: Конфигурация логирования (Singleton).

**Структура файлов логов**:
```
{log_path}/Logs/DRVGPU/{YYYY-MM-DD}/{HH-MM-SS}.log
```

**Методы**:

| Метод | Описание |
|-------|----------|
| `GetInstance()` | Получить Singleton |
| `SetLogPath(path)` | Установить путь |
| `GetLogPath()` | Получить путь |
| `GetLogFilePath()` | Получить полный путь к файлу |
| `SetEnabled(enabled)` | Включить/выключить |
| `IsEnabled()` | Проверить |
| `CreateLogDirectory()` | Создать директорию |
| `Reset()` | Сбросить |

---

### GPUDeviceInfo

**Файл**: [`common/gpu_device_info.hpp`](../../include/DrvGPU/common/gpu_device_info.hpp)

**Назначение**: Структура с информацией о GPU устройстве.

**Поля**:

| Поле | Тип | Описание |
|------|-----|----------|
| `name` | `string` | Название |
| `vendor` | `string` | Производитель |
| `driver_version` | `string` | Версия драйвера |
| `opencl_version` | `string` | Версия OpenCL |
| `device_index` | `int` | Индекс |
| `global_memory_size` | `size_t` | Глобальная память |
| `local_memory_size` | `size_t` | Локальная память |
| `max_mem_alloc_size` | `size_t` | Макс. аллокация |
| `max_compute_units` | `size_t` | Compute units |
| `max_work_group_size` | `size_t` | Max work group |
| `max_clock_frequency` | `size_t` | Частота MHz |
| `supports_svm` | `bool` | SVM поддержка |
| `supports_double` | `bool` | Double precision |
| `supports_half` | `bool` | Half precision |
| `supports_unified_memory` | `bool` | Unified memory |

**Методы**:
- `ToString()` - информация строкой
- `GetGlobalMemoryGB()` - память в GB

---

### BackendType

**Файл**: [`common/backend_type.hpp`](../../include/DrvGPU/common/backend_type.hpp)

**Назначение**: Перечисление типов бэкендов.

```cpp
enum class BackendType {
    OPENCL,        // OpenCL backend
    ROCm,          // ROCm backend (future)
    OPENCLandROCm, // OpenCL and ROCm
    AUTO           // Auto selection
};
```

**Функция**: `BackendTypeToString(type)` - конвертация в строку

---

### LoadBalancingStrategy

**Файл**: [`common/load_balancing.hpp`](../../include/DrvGPU/common/load_balancing.hpp)

**Назначение**: Стратегии распределения нагрузки.

```cpp
enum class LoadBalancingStrategy {
    ROUND_ROBIN,      // Циклический выбор
    LEAST_LOADED,     // Наименее загруженная
    MANUAL,           // Ручной выбор
    FASTEST_FIRST     // Сначала быстрая
};
```

**Функция**: `LoadBalancingStrategyToString(strategy)` - конвертация в строку

---

### IComputeModule (Интерфейс)

**Файл**: [`common/i_compute_module.hpp`](../../include/DrvGPU/common/i_compute_module.hpp)

**Назначение**: Интерфейс для compute модулей.

**Методы интерфейса**:

| Метод | Описание |
|-------|----------|
| `Initialize()` | Инициализировать модуль |
| `IsInitialized()` | Проверить инициализацию |
| `Cleanup()` | Очистить ресурсы |
| `GetName()` | Получить имя |
| `GetVersion()` | Получить версию |
| `GetDescription()` | Получить описание |
| `GetBackend()` | Получить бэкенд |

**Примеры модулей**:
- FFTModule - быстрое преобразование Фурье
- MatrixModule - операции с матрицами
- ConvolutionModule - свёртка
- SortModule - сортировка

---

## Interfaces

### Сводная таблица интерфейсов

| Интерфейс | Файл | Назначение |
|-----------|------|------------|
| `IBackend` | `common/i_backend.hpp` | Абстракция бэкенда |
| `IMemoryBuffer` | `memory/i_memory_buffer.hpp` | Буфер памяти |
| `ILogger` | `common/logger_interface.hpp` | Логирование |
| `IComputeModule` | `common/i_compute_module.hpp` | Compute модуль |

---

## Сводные таблицы

### Все классы по слоям

| Класс | Файл | Интерфейс | Слой |
|-------|------|-----------|------|
| `DrvGPU` | drv_gpu.hpp/cpp | - | Core |
| `GPUManager` | gpu_manager.hpp | - | Core |
| `MemoryManager` | memory/memory_manager.hpp | - | Memory |
| `ModuleRegistry` | module_registry.hpp/cpp | - | Core |
| `IBackend` | common/i_backend.hpp | Interface | Backend |
| `OpenCLBackend` | opencl/opencl_backend.hpp/cpp | IBackend | Backend |
| `OpenCLCore` | opencl/opencl_core.hpp/cpp | - | Backend |
| `OpenCLBackendExternal` | opencl/opencl_backend_external.hpp/cpp | OpenCLBackend | Backend |
| `CommandQueuePool` | opencl/command_queue_pool.hpp/cpp | - | Backend |
| `IMemoryBuffer` | memory/i_memory_buffer.hpp | Interface | Memory |
| `GPUBuffer` | memory/gpu_buffer.hpp | IMemoryBuffer | Memory |
| `SVMBuffer` | memory/svm_buffer.hpp | IMemoryBuffer | Memory |
| `ExternalCLBufferAdapter` | memory/external_cl_buffer_adapter.hpp | IMemoryBuffer | Memory |
| `Logger` | common/logger.hpp/cpp | - | Common |
| `ConfigLogger` | common/config_logger.hpp/cpp | - | Common |
| `DefaultLogger` | common/default_logger.hpp/cpp | ILogger | Common |
| `ILogger` | common/logger_interface.hpp | Interface | Common |
| `IComputeModule` | common/i_compute_module.hpp | Interface | Core |

---

### Классы по паттернам

| Паттерн | Классы |
|---------|--------|
| Bridge | `IBackend`, `OpenCLBackend`, `IMemoryBuffer`, `GPUBuffer` |
| Facade | `DrvGPU`, `GPUManager`, `Logger` |
| Singleton | `ConfigLogger`, `DefaultLogger`, `Logger` |
| Factory | `GPUManager` (через InitializeAll) |
| Strategy | `GPUManager` (LoadBalancingStrategy) |
| Registry | `ModuleRegistry` |
| Object Pool | `CommandQueuePool` |
| Template Method | `IComputeModule` |
| **Per-Device (v2.0)** | `OpenCLCore` - каждый экземпляр для своего GPU |

> ⚠️ **Примечание**: `OpenCLCore` больше НЕ Singleton! Теперь per-device для Multi-GPU.

---

## Зависимости интерфейсов

```
IBackend
    ├── GPUDeviceInfo
    ├── BackendType
    └── Memory operations

IMemoryBuffer
    └── MemoryType

IComputeModule
    └── IBackend (через GetBackend())

ILogger
    └── Нет зависимостей
```

---

## Ссылки на документацию

| Раздел | Файл | Описание |
|--------|------|----------|
| Общая архитектура | [Architecture.md](Architecture.md) | Архитектура проекта |
| OpenCL бэкенд | [OpenCL.md](OpenCL.md) | OpenCLBackend, OpenCLCore, External Context |
| Command Queue | [Command.md](Command.md) | CommandQueuePool |
| Система памяти | [Memory.md](Memory.md) | IMemoryBuffer, GPUBuffer, SVMBuffer |
