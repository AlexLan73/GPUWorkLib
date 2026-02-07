# DrvGPU Архитектура проекта

## Оглавление

1. [Обзор архитектуры](#обзор-архитектуры)
2. [Структура директорий](#структура-директорий)
3. [Архитектурные слои](#архитектурные-слои)
4. [Паттерны проектирования](#паттерны-проектирования)
5. [Зависимости между компонентами](#зависимости-между-компонентами)
6. [Документация по разделам](#документация-по-разделам)

---

## Обзор архитектуры

DrvGPU — модульная библиотека для работы с GPU, предоставляющая единый интерфейс для различных бэкендов (OpenCL, ROCm, CUDA).

```
┌─────────────────────────────────────────────────────────────────┐
│                         DrvGPU Library                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Core Layer                             │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │  │ DrvGPU   │ │ GPUManager│ │ Module   │ │ Memory   │    │  │
│  │  │          │ │          │ │ Registry │ │ Manager  │    │  │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘    │  │
│  └───────┼────────────┼────────────┼────────────┼───────────┘  │
│          │            │            │            │               │
│          └────────────┴─────┬──────┴────────────┘               │
│                             │                                   │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Backend Abstraction Layer                    │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │                   IBackend Interface                │  │  │
│  │  └───────────────────────┬─────────────────────────────┘  │  │
│  │                          │                                 │
│  │    ┌─────────────────────┼─────────────────────┐          │
│  │    ▼                     ▼                     ▼          │
│  │  ┌──────────┐       ┌──────────┐       ┌──────────┐      │
│  │  │ OpenCL   │       │  ROCm    │       │  CUDA    │      │
│  │  │ Backend  │       │  Backend │       │  Backend │      │
│  │  │ (ГОТОВ)  │       │ (ПЛАН)   │       │ (ПЛАН)   │      │
│  │  └──────────┘       └──────────┘       └──────────┘      │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Common Services                         │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │  │ Logger   │ │ Config   │ │ GPUDevice│ │ Load     │    │  │
│  │  │          │ │ Logger   │ │ Info     │ │ Balancing│    │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Структура директорий

```
GPUWorkLib/
├── include/DrvGPU/
│   ├── CMakeLists.txt
│   ├── drv_gpu.cpp/hpp           # Главный класс
│   ├── gpu_manager.hpp           # Менеджер нескольких GPU
│   ├── module_registry.cpp/hpp   # Регистр модулей
│   │
│   ├── common/                   # Общие интерфейсы и утилиты
│   │   ├── backend_type.hpp      # Типы бэкендов
│   │   ├── i_backend.hpp         # Интерфейс бэкенда
│   │   ├── i_compute_module.hpp  # Интерфейс модуля
│   │   ├── gpu_device_info.hpp   # Инфо о GPU
│   │   ├── load_balancing.hpp    # Балансировка
│   │   ├── logger*.hpp/cpp       # Логирование
│   │   └── config_logger*.hpp/cpp# Конфиг логирования
│   │
│   ├── backends/                 # Бэкенды (см. OpenCL.md)
│   │   └── opencl/
│   │       ├── opencl_core.cpp/hpp
│   │       ├── opencl_backend.cpp/hpp
│   │       ├── opencl_backend_external.cpp/hpp
│   │       └── command_queue_pool.cpp/hpp
│   │
│   └── memory/                   # Управление памятью (см. Memory.md)
│       ├── memory_type.hpp
│       ├── i_memory_buffer.hpp
│       ├── gpu_buffer.hpp
│       ├── memory_manager.hpp
│       ├── svm_buffer.hpp
│       ├── svm_capabilities.hpp
│       └── external_cl_buffer_adapter.hpp
│
├── src/
│   ├── main.cpp
│   └── CMakeLists.txt
│
├── tests/
│   ├── single_gpu.hpp
│   ├── multi_gpu.hpp
│   └── example_external_context_usage.hpp
│
└── Doc/DrvGPU/
    ├── Architecture.md           # Этот файл
    ├── Memory.md                 # Система памяти
    ├── OpenCL.md                 # OpenCL бэкенд
    ├── Command.md                # Command Queue
    └── Classes.md                # Все классы по категориям
```

---

## Архитектурные слои

### 1. Core Layer (Основной слой)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `DrvGPU` | drv_gpu.hpp/cpp | Фасад библиотеки |
| `GPUManager` | gpu_manager.hpp | Управление несколькими GPU |
| `ModuleRegistry` | module_registry.hpp/cpp | Регистр compute модулей |

**См. также**: [Classes.md](Classes.md)

### 2. Backend Layer (Слой бэкендов)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `IBackend` | common/i_backend.hpp | Интерфейс бэкенда |
| `OpenCLBackend` | opencl/opencl_backend.hpp/cpp | OpenCL реализация |
| `OpenCLCore` | opencl/opencl_core.hpp/cpp | Низкоуровневые операции |
| `OpenCLBackendExternal` | opencl/opencl_backend_external.hpp/cpp | External Context |

**См. также**: [OpenCL.md](OpenCL.md)

### 3. Command Layer (Слой команд)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `CommandQueuePool` | opencl/command_queue_pool.hpp/cpp | Пул очередей команд |

**См. также**: [Command.md](Command.md)

### 4. Memory Layer (Слой памяти)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `IMemoryBuffer` | memory/i_memory_buffer.hpp | Интерфейс буфера |
| `GPUBuffer` | memory/gpu_buffer.hpp | Стандартный буфер |
| `SVMBuffer` | memory/svm_buffer.hpp | SVM буфер |
| `MemoryManager` | memory/memory_manager.hpp | Менеджер памяти |

**См. также**: [Memory.md](Memory.md)

### 5. Common Services (Общие сервисы)

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `Logger` | common/logger.hpp/cpp | Фасад логирования |
| `ILogger` | common/logger_interface.hpp | Интерфейс логера |
| `DefaultLogger` | common/default_logger.hpp/cpp | Реализация на spdlog |
| `ConfigLogger` | common/config_logger.hpp/cpp | Конфигурация |
| `GPUDeviceInfo` | common/gpu_device_info.hpp | Информация о GPU |

---

## Паттерны проектирования

| Паттерн | Применение |
|---------|------------|
| **Bridge** | Разделение абстракции и реализации бэкендов |
| **Facade** | DrvGPU как упрощённый интерфейс |
| **Singleton** | Logger, ConfigLogger, DefaultLogger |
| **Factory** | Создание бэкендов, GPUManager |
| **Strategy** | LoadBalancingStrategy |
| **Registry** | ModuleRegistry |
| **Object Pool** | CommandQueuePool |
| **Per-Device** | OpenCLCore (v2.0) - каждый экземпляр для своего GPU |

> ⚠️ **Примечание (v2.0)**: OpenCLCore больше НЕ использует Singleton! Теперь это per-device класс для поддержки Multi-GPU.

**См. также**: [Classes.md](Classes.md) - подробности о паттернах

---

## Зависимости между компонентами

```
                    ┌─────────────────┐
                    │     DrvGPU      │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  GPUManager   │   │ MemoryManager │   │ ModuleRegistry│
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                    │
        │                   ▼                    │
        │          ┌───────────────┐             │
        │          │  IBackend     │◄────────────┘
        │          └───────┬───────┘
        │                  │
        │    ┌─────────────┼─────────────┐
        │    │             │             │
        ▼    ▼             ▼             ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│  OpenCL   │   │   ROCm    │   │   CUDA    │
│  Backend  │   │  Backend  │   │  Backend  │
└───────────┘   └───────────┘   └───────────┘
```

---

## Документация по разделам

| Раздел | Файл | Описание |
|--------|------|----------|
| **Общая архитектура** | Architecture.md | Этот файл |
| **Система памяти** | Memory.md | IMemoryBuffer, GPUBuffer, SVMBuffer, MemoryManager |
| **OpenCL бэкенд** | OpenCL.md | OpenCLBackend, OpenCLCore, OpenCLBackendExternal |
| **Command Queue** | Command.md | CommandQueuePool, управление очередями |
| **Все классы** | Classes.md | Полный справочник классов |

### Как редактировать документацию

1. **Memory** — редактируйте `Memory.md` для изменений в системе памяти
2. **OpenCL** — редактируйте `OpenCL.md` для изменений в бэкенде
3. **Command** — редактируйте `Command.md` для изменений в очередях команд
4. **Классы** — `Classes.md` автоматически генерируется из исходников
5. **Архитектура** — `Architecture.md` для общих изменений структуры

---

## Связь с LID-Architecture

Документация DrvGPU соответствует концепции LID (Library Interface Definition):

- **LID-Core**: `DrvGPU`, `GPUManager`, `ModuleRegistry`
- **LID-Backend**: `IBackend`, `OpenCLBackend`, `OpenCLCore`
- **LID-Utils**: `Logger`, `GPUDeviceInfo`, `LoadBalancing`
- **LID-Memory**: `IMemoryBuffer`, `GPUBuffer`, `MemoryManager`

См. также: [`../../Архитекура/LID-Architecture.md`](../../Архитекура/LID-Architecture.md)
