# Graceful Shutdown — Анализ и рекомендации для GPUWorkLib

> **Источник**: Context7 (g3log, Go gs), Document Plan, sequential-thinking
> **Дата**: 2026-02-11
> **Связь**: [DiscussionPlan/Graceful Shutdown/](../../MemoryBank/DiscussionPlan/Graceful%20Shutdown/)

---

## 1. Текущее состояние проекта

### 1.1 Компоненты и их lifecycle

| Компонент | Тип | Cleanup | Потоки | Очередь |
|-----------|-----|---------|--------|---------|
| **DrvGPU** | Локальный (не singleton) | `Cleanup()` в деструкторе | Нет | Нет |
| **OpenCLBackend** | Внутри DrvGPU | queue, context release | Нет | Нет |
| **DefaultLogger** | Per-GPU (map) | `Shutdown()` в деструкторе | Нет | plog sync |
| **GPUProfiler** | Singleton | `Stop()` в `~AsyncServiceBase` | Да (worker) | Да |
| **SpectrumMaximaFinder** | Локальный | `ReleaseResources()` в деструкторе | Нет | Нет |
| **MemoryManager** | Внутри DrvGPU | `Cleanup()` | Нет | Нет |

### 1.2 Отличия от документа DiscussionPlan

Документ [Graceful Shutdown в многопоточном C++ приложении с.md](../../MemoryBank/DiscussionPlan/Graceful%20Shutdown/Graceful%20Shutdown%20в%20многопоточном%20C++%20приложении%20с.md) описывает:

- **Logger, ConsoleOutput, Profiler, DiskWriter** — каждый с очередью и worker-потоком
- **Observer** между ними
- **ShutdownManager** с фазами: requestStop → drain → joinThreads

В GPUWorkLib:

- **plog** — синхронный, без worker-потока
- **GPUProfiler** — один асинхронный сервис с очередью (AsyncServiceBase)
- **Нет Observer** между сервисами
- **Нет ShutdownManager** — всё завязано на RAII

### 1.3 Текущий main.cpp

```cpp
int main() {
    // ...
    test_spectrum_maxima::run();  // DrvGPU + finder — локальные, RAII
    return 0;
}
```

- Нет обработки SIGINT/SIGTERM
- При Ctrl+C — немедленный выход, деструкторы вызываются в неопределённом порядке
- `GPUProfiler` — static singleton, `~AsyncServiceBase` вызывает `Stop()` (drain + join)

---

## 2. Паттерны из Context7 и статей

### 2.1 Go gs (shengyanli1982/gs)

- `TerminateSignal` + `RegisterCancelHandles` — регистрация обработчиков
- `WaitForAsync` / `WaitForSync` — ожидание завершения
- При SIGINT: вызов handlers в заданном порядке

### 2.2 g3log (C++)

- `g3::setFatalPreLoggingHook([] { cleanup(); })` — cleanup перед фатальным сигналом
- `g3::overrideSetupSignals(...)` — кастомный набор сигналов
- Подходит для crash-сценариев (SIGSEGV и т.п.)

### 2.3 DiscussionPlan: ShutdownManager

- Фазы: requestStop → drain → joinThreads
- Приоритеты (Logger 10, Console 15, Profiler 20, DiskWriter 30)
- Принцип: «производители останавливаются раньше потребителей»

---

## 3. Адаптация под GPUWorkLib

### 3.1 Что уже есть

- **AsyncServiceBase::Stop()** — drain очереди + join worker (реализует drain + join)
- **DefaultLogger::Shutdown()** — корректное завершение plog
- **DrvGPU::Cleanup()** — освобождение OpenCL
- **RAII** — объекты в scope уничтожаются в обратном порядке создания

### 3.2 Чего не хватает

1. **Обработка SIGINT/SIGTERM** — при Ctrl+C процесс завершается без явного shutdown
2. **Упорядоченный shutdown** — порядок для static-синглтонов не гарантирован
3. **Прерывание длительной обработки** (batch 256×1300000) — возможность остановить без потери данных

---

## 4. Варианты реализации

### Вариант A: Минимальный (рекомендуется для старта)

**Цель**: Корректная реакция на SIGINT/SIGTERM.

```cpp
// В main() или в точке входа
static std::atomic<bool> g_shutdown_requested{false};

void SetupSignalHandlers() {
#ifndef _WIN32
    signal(SIGINT, [](int) {
        g_shutdown_requested.store(true);
        // Не exit() — пусть main loop проверит флаг
    });
    signal(SIGTERM, [](int) {
        g_shutdown_requested.store(true);
    });
#endif
}
```

**Использование**:

- В длительном цикле (batch): `if (g_shutdown_requested.load()) break;`
- При нормальном завершении: явный вызов `GPUProfiler::GetInstance().Stop()` перед `return`

**Плюсы**: мало кода, можно внедрить быстро  
**Минусы**: нет единого ShutdownManager, порядок — за счёт RAII

---

### Вариант B: ShutdownManager (по образцу DiscussionPlan)

**Цель**: Централизованный, упорядоченный shutdown.

```cpp
class IShutdownAware {
public:
    virtual void requestStop() = 0;
    virtual void drain() = 0;       // опционально, если есть очередь
    virtual void joinThreads() = 0; // опционально
    virtual int shutdownPriority() const = 0;
};

class ShutdownManager {
    static ShutdownManager& instance();
    void registerModule(IShutdownAware* m);
    void shutdown();
};
```

**Регистрация**:

- `GPUProfiler` — реализует IShutdownAware (Stop = requestStop + drain + join)
- `DefaultLogger` — обёртка, вызывающая Shutdown()
- `DrvGPU` — обычно не регистрируется (локальный, RAII)

**Порядок** (по приоритету):

| Компонент | Приоритет | Обоснование |
|-----------|-----------|-------------|
| GPUProfiler | 10 | Обрабатывает данные от GPU, должен закончить drain |
| DefaultLogger | 20 | Логирование после Profiler |

**Вызов**:

```cpp
signal(SIGINT, [](int) {
    ShutdownManager::instance().shutdown();
    std::exit(0);
});
```

**Плюсы**: единая точка входа, предсказуемый порядок  
**Минусы**: больше кода, нужно адаптировать синглтоны

---

### Вариант C: RAII + явный Stop в main

**Цель**: Без отдельных сигналов, порядок через scope.

```cpp
int main() {
    DrvGPU gpu(BackendType::OPENCL, 0);
    gpu.Initialize();

    auto& profiler = GPUProfiler::GetInstance();
    if (profiler.IsEnabled()) profiler.Start();

    int result = test_spectrum_maxima::run();

    // Явный Stop перед выходом (drain очереди)
    if (profiler.IsEnabled()) profiler.Stop();

    return result;
}
```

**Плюсы**: просто, без signal handlers  
**Минусы**: при Ctrl+C Stop() не вызывается, drain не гарантирован

---

## 5. Рекомендации по этапам

### Этап 1 (сейчас)

1. **Добавить явный `profiler.Stop()`** в `main` перед `return` (если используется Profiler).
2. **Опционально**: флаг `g_shutdown_requested` для будущих длительных циклов (batch).

### Этап 2 (при появлении batch 256×1300000)

1. **Проверка флага** в цикле batch: `if (g_shutdown_requested) break;`
2. **Обработка SIGINT** — установка флага, без немедленного `exit()`.
3. **После break** — завершить текущий batch, вызвать Stop/Cleanup, затем выход.

### Этап 3 (при росте числа сервисов)

1. Ввести **ShutdownManager** по образцу DiscussionPlan.
2. Перевести **GPUProfiler** на `IShutdownAware`.
3. Добавить **таймаут** в drain (например, 5 с), чтобы не зависнуть при сбоях.

---

## 6. Fatal-сигналы (g3log-подход)

Для SIGSEGV, SIGABRT и т.п.:

- **plog** не даёт pre-fatal hook (в отличие от g3log).
- Варианты:
  - Использовать `std::set_terminate()` для последнего шанса на flush.
  - При необходимости — добавить g3log или аналог с hook.

Для текущего тестового приложения это не критично.

---

## 7. Сводка

| Аспект | Рекомендация |
|--------|--------------|
| **Сейчас** | Явный `profiler.Stop()` в main перед return |
| **SIGINT** | Флаг `g_shutdown_requested` + проверка в длительных циклах |
| **ShutdownManager** | Вводить при появлении нескольких асинхронных сервисов |
| **Порядок** | Profiler.Stop() → Logger.Shutdown() → DrvGPU.Cleanup (RAII) |
| **Batch** | `if (g_shutdown_requested) break;` в цикле batch |

---

*Создано: 2026-02-11*
