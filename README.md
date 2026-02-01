# DrvGPU - Модульная GPU библиотека для вычислений

## Описание

**DrvGPU** — модульная, расширяемая библиотека для GPU-вычислений с поддержкой нескольких бэкендов (OpenCL, ROCm) и модульной архитектурой для обработки сигналов.

## Архитектура

Проект построен на основе **модульной архитектуры**, обеспечивающей:

- **Абстракция бэкенда**: Единый API для OpenCL и ROCm
- **Модульность**: Независимые библиотеки для разных функциональных областей
- **Расширяемость**: Легкое добавление новых модулей и бэкендов
- **Тестируемость**: Полный набор unit-тестов

### Структура архитектуры

```
┌─────────────────────────────────────────────────────────────┐
│                        DrvGPU Library                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │               BACKEND ABSTRACTION LAYER             │   │
│   │  ┌──────────────┐  ┌──────────────┐                │   │
│   │  │   OpenCL     │  │    ROCm      │ (перспектива)  │   │
│   │  │   Backend    │  │   Backend    │                │   │
│   │  └──────────────┘  └──────────────┘                │   │
│   └─────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   CORE LAYER                        │   │
│   │  • DrvGPU        • GPUManager                       │   │
│   │  • Module Registry•                                 │   │
│   └─────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                  MEMORY LAYER                       │   │
│   │  • MemoryManager  • GPUBuffer                       │   │
│   │  • SVMBuffer      • IMemoryBuffer                   │   │
│   └─────────────────────────────────────────────────────┘   │
│                              │                               │
│                              ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                COMMON SERVICES                      │   │
│   │  • Logger         • GPUDeviceInfo                   │   │
│   │  • ConfigLogger   • LoadBalancing                   │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Компоненты

### Backend Layer (Слой бэкендов)

| Компонент | Файл | Статус | Описание |
|-----------|------|--------|----------|
| `OpenCLBackend` | [`backends/opencl/opencl_backend.hpp`](include/DrvGPU/backends/opencl/opencl_backend.hpp) | ✅ Готов | OpenCL бэкенд |
| `OpenCLCore` | [`backends/opencl/opencl_core.hpp`](include/DrvGPU/backends/opencl/opencl_core.hpp) | ✅ Готов | Низкоуровневые OpenCL операции |
| `OpenCLBackendExternal` | [`backends/opencl/opencl_backend_external.hpp`](include/DrvGPU/backends/opencl/opencl_backend_external.hpp) | ✅ Готов | External Context поддержка |
| `CommandQueuePool` | [`backends/opencl/command_queue_pool.hpp`](include/DrvGPU/backends/opencl/command_queue_pool.hpp) | ✅ Готов | Пул очередей команд |
| `ROCmBackend` | [`backends/rocm/rocm.txt`](include/DrvGPU/backends/rocm/rocm.txt) | 🔜 Планируется | ROCm/HIP реализация |

### Core Layer (Основной слой)

| Компонент | Файл | Описание |
|-----------|------|----------|
| `DrvGPU` | [`drv_gpu.hpp`](include/DrvGPU/drv_gpu.hpp) | Главный класс, фасад библиотеки |
| `GPUManager` | [`gpu_manager.hpp`](include/DrvGPU/gpu_manager.hpp) | Управление несколькими GPU |
| `ModuleRegistry` | [`module_registry.hpp`](include/DrvGPU/module_registry.hpp) | Регистр вычислительных модулей |

### Memory Layer (Слой памяти)

| Компонент | Файл | Описание |
|-----------|------|----------|
| `IMemoryBuffer` | [`memory/i_memory_buffer.hpp`](include/DrvGPU/memory/i_memory_buffer.hpp) | Интерфейс буфера памяти |
| `GPUBuffer` | [`memory/gpu_buffer.hpp`](include/DrvGPU/memory/gpu_buffer.hpp) | Стандартный буфер GPU |
| `SVMBuffer` | [`memory/svm_buffer.hpp`](include/DrvGPU/memory/svm_buffer.hpp) | Shared Virtual Memory буфер |
| `MemoryManager` | [`memory/memory_manager.hpp`](include/DrvGPU/memory/memory_manager.hpp) | Менеджер памяти |
| `ExternalCLBufferAdapter` | [`memory/external_cl_buffer_adapter.hpp`](include/DrvGPU/memory/external_cl_buffer_adapter.hpp) | Адаптер для внешних буферов |

### Common Services (Общие сервисы)

| Компонент | Файл | Описание |
|-----------|------|----------|
| `Logger` | [`common/logger.hpp`](include/DrvGPU/common/logger.hpp) | Фасад логирования |
| `ILogger` | [`common/logger_interface.hpp`](include/DrvGPU/common/logger_interface.hpp) | Интерфейс логера |
| `DefaultLogger` | [`common/default_logger.hpp`](include/DrvGPU/common/default_logger.hpp) | Реализация на spdlog |
| `ConfigLogger` | [`common/config_logger.hpp`](include/DrvGPU/common/config_logger.hpp) | Конфигурация логирования |
| `GPUDeviceInfo` | [`common/gpu_device_info.hpp`](include/DrvGPU/common/gpu_device_info.hpp) | Информация о GPU |
| `LoadBalancingStrategy` | [`common/load_balancing.hpp`](include/DrvGPU/common/load_balancing.hpp) | Стратегии балансировки |
| `IBackend` | [`common/i_backend.hpp`](include/DrvGPU/common/i_backend.hpp) | Интерфейс бэкенда |
| `IComputeModule` | [`common/i_compute_module.hpp`](include/DrvGPU/common/i_compute_module.hpp) | Интерфейс модуля |

## Установка

### Требования

- C++17 совместимый компилятор (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- OpenCL SDK (для OpenCL бэкенда)
- ROCm SDK (опционально, для ROCm бэкенда)
- spdlog (для логирования)

### Сборка

```bash
# Клонирование репозитория
git clone https://github.com/your-org/GPUWorkLib.git
cd GPUWorkLib

# Создание директории сборки
mkdir build && cd build

# Конфигурация CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DDRVGPU_BUILD_TESTS=ON \
    -DDRVGPU_BUILD_EXAMPLES=ON \
    -DDRVGPU_ENABLE_OPENCL=ON \
    -DDRVGPU_ENABLE_ROCM=OFF

# Сборка
cmake --build . --parallel

# Запуск тестов
ctest
```

### CMake Опции

| Опция | По умолчанию | Описание |
|-------|--------------|----------|
| `DRVGPU_BUILD_TESTS` | ON | Сборка unit-тестов |
| `DRVGPU_BUILD_EXAMPLES` | ON | Сборка примеров |
| `DRVGPU_ENABLE_OPENCL` | ON | Включить OpenCL бэкенд |
| `DRVGPU_ENABLE_ROCM` | OFF | Включить ROCm бэкенд |

## Быстрый старт

### Пример использования

```cpp
#include <DrvGPU/drv_gpu.hpp>
#include <DrvGPU/memory/gpu_buffer.hpp>

using namespace drvgpu;

int main() {
    // Создание GPU с OpenCL бэкендом
    DrvGPU gpu(BackendType::OPENCL, 0);
    gpu.Initialize();

    // Получение информации об устройстве
    auto device_info = gpu.GetDeviceInfo();
    printf("GPU: %s\n", device_info.name.c_str());

    // Создание буфера
    size_t buffer_size = 1024 * sizeof(float);
    auto buffer = gpu.GetMemoryManager().AllocateBuffer(buffer_size, MemoryType::Device);

    // Копирование данных на GPU
    float host_data[1024] = { /* ... */ };
    buffer->CopyFromHost(host_data, buffer_size);

    // ... обработка данных ...

    // Копирование результатов обратно
    float result[1024];
    buffer->CopyToHost(result, buffer_size);

    return 0;
}
```

### External Context использование

```cpp
#include <DrvGPU/backends/opencl/opencl_backend_external.hpp>

// Получение внешнего OpenCL контекста
cl_context external_context = /* ... */;
cl_device_id external_device = /* ... */;

auto backend = OpenCLBackendExternal::CreateFromExternalContext(
    external_context, external_device);

auto buffer = backend->Allocate(1024 * sizeof(float));
```

### Управление несколькими GPU

```cpp
#include <DrvGPU/gpu_manager.hpp>

int main() {
    GPUManager manager;
    
    // Инициализация всех доступных GPU
    manager.InitializeAll(BackendType::OPENCL);
    
    printf("Найдено GPU: %zu\n", manager.GetGPUCount());
    
    // Round-Robin выбор GPU
    auto gpu1 = manager.GetNextGPU();
    
    // Получение наименее загруженной GPU
    auto gpu2 = manager.GetLeastLoadedGPU();
    
    // Установка стратегии балансировки
    manager.SetLoadBalancingStrategy(LoadBalancingStrategy::LEAST_LOADED);
    
    return 0;
}
```

## Документация

| Раздел | Файл | Описание |
|--------|------|----------|
| Архитектура | [`Doc/DrvGPU/Architecture.md`](Doc/DrvGPU/Architecture.md) | Общая архитектура проекта |
| Классы | [`Doc/DrvGPU/Classes.md`](Doc/DrvGPU/Classes.md) | Справочник всех классов |
| Память | [`Doc/DrvGPU/Memory.md`](Doc/DrvGPU/Memory.md) | Система памяти |
| OpenCL | [`Doc/DrvGPU/OpenCL.md`](Doc/DrvGPU/OpenCL.md) | OpenCL бэкенд |
| Command | [`Doc/DrvGPU/Command.md`](Doc/DrvGPU/Command.md) | Command Queue |

## Структура проекта

```
GPUWorkLib/
├── CMakeLists.txt                    # Root CMake
├── README.md                          # Этот файл
├── run.bat                            # Windows launcher
├── run.sh                             # Linux launcher
├── .gitignore                         # Git ignore rules
├── CLAUDE.md                          # Claude AI instructions
│
├── clFFT/                            # clFFT библиотека
│   └── include/
│       ├── clFFT.h
│       ├── clFFT.version.h
│       ├── clAmdFft.h
│       └── clAmdFft.version.h
│
├── Doc/                              # Проектная документация
│   ├── Architecture/
│   │   └── LID-Architecture.md
│   ├── Config/
│   │   └── ...
│   └── DrvGPU/
│       ├── Architecture.md           # Архитектура DrvGPU
│       ├── Classes.md                # Справочник классов
│       ├── Memory.md                 # Система памяти
│       ├── OpenCL.md                 # OpenCL бэкенд
│       ├── Command.md                # Command Queue
│       ├── Backends.md               # Бэкенды
│       └── Arс/
│           └── Примеры кода
│
├── include/DrvGPU/                   # Заголовочные файлы DrvGPU
│   ├── CMakeLists.txt
│   ├── drv_gpu.hpp                   # Основной класс DrvGPU
│   ├── drv_gpu.cpp
│   ├── gpu_manager.hpp               # Менеджер GPU
│   ├── module_registry.hpp           # Реестр модулей
│   ├── module_registry.cpp
│   │
│   ├── common/                       # Общие интерфейсы и утилиты
│   │   ├── backend_type.hpp          # Типы бэкендов
│   │   ├── i_backend.hpp             # Интерфейс бэкенда
│   │   ├── i_compute_module.hpp      # Интерфейс вычислительного модуля
│   │   ├── gpu_device_info.hpp       # Информация об устройстве
│   │   ├── load_balancing.hpp        # Балансировка нагрузки
│   │   ├── logger.hpp                # Фасад логирования
│   │   ├── logger.cpp
│   │   ├── logger_interface.hpp      # Интерфейс логера
│   │   ├── default_logger.hpp        # Реализация на spdlog
│   │   ├── default_logger.cpp
│   │   ├── config_logger.hpp         # Конфигурация логирования
│   │   └── config_logger.cpp
│   │
│   ├── backends/                     # Реализации бэкендов
│   │   └── opencl/                   # OpenCL бэкенд
│   │       ├── opencl_backend.hpp
│   │       ├── opencl_backend.cpp
│   │       ├── opencl_backend_external.hpp
│   │       ├── opencl_backend_external.cpp
│   │       ├── opencl_core.hpp
│   │       ├── opencl_core.cpp
│   │       ├── command_queue_pool.hpp
│   │       └── command_queue_pool.cpp
│   │
│   └── memory/                       # Управление памятью
│       ├── memory_type.hpp           # Типы памяти
│       ├── i_memory_buffer.hpp       # Интерфейс буфера памяти
│       ├── gpu_buffer.hpp            # GPU буфер
│       ├── memory_manager.hpp        # Менеджер памяти
│       ├── svm_buffer.hpp            # SVM буфер
│       ├── svm_capabilities.hpp      # SVM возможности
│       └── external_cl_buffer_adapter.hpp
│
├── interface/                        # Внешние интерфейсы
│   ├── antenna_fft_params.h
│   ├── combined_delay_param.h
│   ├── DelayParameter.h
│   └── lfm_parameters.h
│
├── src/                              # Исходники
│   ├── CMakeLists.txt
│   └── main.cpp
│
├── modules/                          # Вычислительные модули
│   └── modul.txt
│
└── tests/                            # Тесты
    ├── single_gpu.hpp
    ├── multi_gpu.hpp
    └── example_external_context_usage.hpp
```

## Тестирование

### Запуск тестов

```bash
# Из директории build
ctest --output-on-failure

# Или запуск конкретного теста
./tests/unit/test_fft
```

## Паттерны проектирования

| Паттерн | Применение |
|---------|------------|
| **Bridge** | Разделение абстракции и реализации бэкендов (`IBackend` → `OpenCLBackend`) |
| **Facade** | `DrvGPU` как упрощённый интерфейс |
| **Singleton** | `Logger`, `ConfigLogger`, `DefaultLogger` |
| **Factory** | `GPUManager` через `InitializeAll` |
| **Strategy** | `LoadBalancingStrategy` |
| **Registry** | `ModuleRegistry` |
| **Object Pool** | `CommandQueuePool` |

## Разработка

### Добавление нового бэкенда

1. Создайте директорию в `include/DrvGPU/backends/new_backend/`
2. Реализуйте интерфейс `IBackend`
3. Обновите `BackendType` в [`backend_type.hpp`](include/DrvGPU/common/backend_type.hpp)
4. Добавьте CMakeLists.txt
5. Напишите тесты совместимости

### Добавление нового модуля памяти

1. Создайте класс, реализующий `IMemoryBuffer`
2. Добавьте в `MemoryManager` метод для создания буфера этого типа
3. Напишите тесты в `tests/`

## Лицензия

MIT License

## Контакты

- Repository: https://github.com/your-org/GPUWorkLib
- Issues: https://github.com/your-org/GPUWorkLib/issues

---

**Статус проекта**: 🚧 Активная разработка
**Версия**: 1.0.0-alpha
**Дата**: 2026-02-01
