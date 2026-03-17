<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Graceful Shutdown в многопоточном C++ приложении с синглтонами

Отличный вопрос — корректное завершение многопоточного приложения с несколькими синглтонами, очередями и паттерном Observer — это классическая и нетривиальная задача. Давай разберём все варианты и выберем оптимальный.

## Проблема

У тебя есть:

- **Синглтоны**: Logger, ConsoleOutput, Profiler, DiskWriter
- Каждый содержит **очередь сообщений** и **рабочий поток**
- Между ними связь через **Observer**
- При завершении нужно: остановить Observer → дождаться опустошения очередей → остановить потоки → корректно уничтожить объекты

Главные опасности:

1. **Порядок уничтожения синглтонов не определён** (static destruction order fiasco)
2. Потерянные сообщения в очередях
3. Deadlock при завершении
4. Обращение к уже уничтоженному синглтону

***

## Разбор твоих вариантов

### ❌ Вариант 1: Положиться на деструкторы синглтонов

```cpp
~Logger() {
    flush();
    stop_thread();
}
```

**Проблема**: порядок уничтожения `static`-объектов обратен порядку их создания, но ты его не контролируешь. Если `DiskWriter` уничтожится раньше `Logger`, а Logger в деструкторе попытается записать на диск — **undefined behavior**. Это самый хрупкий вариант.

### ❌ Вариант 2: Глобальный таймер с периодическим поллингом

```cpp
// Каждый модуль раз в секунду проверяет флаг
while (!g_shutdown_flag) {
    std::this_thread::sleep_for(1s);
}
```

**Проблемы**:

- Задержка до 1 секунды — неприемлемо для graceful shutdown
- Busy-waiting / polling — расход ресурсов
- Нет гарантии порядка остановки модулей
- Нет гарантии, что очередь будет опустошена


### ⚠️ Вариант 3: Глобальное событие (ближе к правильному, но недостаточно)

```cpp
g_shutdown_event.notify_all();
// все модули начинают flush
```

Идея верная, но нужна **оркестрация** — просто «всем стоп» недостаточно, потому что модули зависят друг от друга.

***

## ✅ Рекомендуемый подход: ShutdownManager (явная двухфазная остановка)

Лучшее решение — **централизованный менеджер завершения** с явным порядком и двумя фазами: сначала сигнал на остановку, потом ожидание завершения.

### Архитектура

```
main() завершается
    │
    ▼
ShutdownManager::shutdown()
    │
    ├─ Фаза 1: SIGNAL ──► Все модули получают stop_requested = true
    │                       Observer отписывает всех
    │                       Новые сообщения отклоняются
    │
    ├─ Фаза 2: DRAIN  ──► Каждый модуль дорабатывает свою очередь
    │                       (condition_variable разбужен)
    │
    ├─ Фаза 3: JOIN   ──► join() рабочих потоков в правильном порядке
    │                       (Logger → Console → Profiler → DiskWriter)
    │
    └─ Фаза 4: DESTROY ─► Явное уничтожение в обратном порядке зависимостей
```


### Реализация

**Базовый интерфейс для всех модулей:**

```cpp
class IShutdownAware {
public:
    virtual ~IShutdownAware() = default;
    virtual void requestStop() = 0;   // Фаза 1: перестать принимать новое
    virtual void drain() = 0;          // Фаза 2: дообработать очередь
    virtual void joinThreads() = 0;    // Фаза 3: остановить потоки
    virtual int shutdownPriority() const = 0; // порядок (меньше = раньше)
};
```

**Типичный модуль (например, Logger):**

```cpp
class Logger : public IShutdownAware {
    std::queue<LogEntry> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> accepting_{true};
    std::thread worker_;

public:
    void log(LogEntry entry) {
        if (!accepting_.load(std::memory_order_acquire)) return; // отклоняем
        {
            std::lock_guard lock(mutex_);
            queue_.push(std::move(entry));
        }
        cv_.notify_one();
    }

    void requestStop() override {
        accepting_.store(false, std::memory_order_release);
        stop_requested_.store(true, std::memory_order_release);
        cv_.notify_one(); // разбудить worker, если спит
    }

    void drain() override {
        // Worker сам дообработает — просто ждём
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return queue_.empty(); });
    }

    void joinThreads() override {
        if (worker_.joinable()) worker_.join();
    }

    int shutdownPriority() const override { return 10; } // Logger первым

private:
    void workerLoop() {
        while (true) {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] {
                return !queue_.empty() || stop_requested_.load();
            });

            // Обработать всё что есть в очереди
            while (!queue_.empty()) {
                auto entry = std::move(queue_.front());
                queue_.pop();
                lock.unlock();
                processEntry(entry);  // запись на диск / консоль
                lock.lock();
            }

            if (stop_requested_.load() && queue_.empty()) {
                cv_.notify_all(); // сигнал для drain()
                return;
            }
        }
    }
};
```

**ShutdownManager:**

```cpp
class ShutdownManager {
    std::vector<IShutdownAware*> modules_;
    std::mutex mutex_;

public:
    static ShutdownManager& instance() {
        static ShutdownManager inst;
        return inst;
    }

    void registerModule(IShutdownAware* module) {
        std::lock_guard lock(mutex_);
        modules_.push_back(module);
    }

    void shutdown() {
        std::lock_guard lock(mutex_);

        // Сортируем по приоритету
        std::sort(modules_.begin(), modules_.end(),
            [](auto* a, auto* b) {
                return a->shutdownPriority() < b->shutdownPriority();
            });

        // Фаза 1: Всем стоп (быстро, неблокирующе)
        for (auto* m : modules_) m->requestStop();

        // Фаза 2: Ждём опустошения очередей
        for (auto* m : modules_) m->drain();

        // Фаза 3: Join потоков
        for (auto* m : modules_) m->joinThreads();

        modules_.clear();
    }
};
```

**main():**

```cpp
int main() {
    // Инициализация — порядок гарантирован
    auto& disk   = DiskWriter::instance();   // priority 30
    auto& prof   = Profiler::instance();      // priority 20
    auto& logger = Logger::instance();        // priority 10
    auto& console = ConsoleOutput::instance();// priority 15

    // Обработка сигналов ОС
    std::signal(SIGINT, [](int) {
        ShutdownManager::instance().shutdown();
        std::exit(0);
    });

    // ... основная работа ...

    // Явное корректное завершение
    ShutdownManager::instance().shutdown();
    return 0;
}
```


***

## Порядок приоритетов — важно!

Порядок остановки должен учитывать зависимости:


| Модуль | Приоритет | Почему |
| :-- | :-- | :-- |
| Logger | 10 (первый) | Генерирует сообщения для других |
| ConsoleOutput | 15 | Может получать от Logger |
| Profiler | 20 | Может писать итоговый отчёт |
| DiskWriter | 30 (последний) | Все пишут через него — должен умереть последним |

Принцип: **производители останавливаются раньше потребителей**.

***

## Дополнительные рекомендации

**1. Таймаут на drain:**

```cpp
void drain() override {
    std::unique_lock lock(mutex_);
    if (!cv_.wait_for(lock, 5s, [this] { return queue_.empty(); })) {
        // Таймаут! Логируем потерю N сообщений
        std::cerr << "WARNING: " << queue_.size() << " messages lost\n";
    }
}
```

Без таймаута зависший worker заблокирует весь shutdown.

**2. Отписка Observer перед shutdown:**
В `requestStop()` каждый модуль должен отписаться от всех Observable, чтобы новые события не приходили после сигнала остановки.

**3. `std::stop_token` (C++20):**
Если используешь C++20, `std::jthread` + `std::stop_token` заменяют ручной `atomic<bool>`:

```cpp
std::jthread worker_([this](std::stop_token stoken) {
    while (!stoken.stop_requested()) {
        // ...
    }
});
// jthread::~jthread() сам делает request_stop() + join()
```

**4. Poison pill для очереди:**
Альтернатива `stop_requested` — отправить специальный «ядовитый» элемент в очередь:

```cpp
void requestStop() override {
    accepting_.store(false);
    {
        std::lock_guard lock(mutex_);
        queue_.push(LogEntry::poison()); // sentinel value
    }
    cv_.notify_one();
}
```

Worker видит poison → дорабатывает оставшееся перед ним → выходит. Это гарантирует, что всё что было **до** сигнала — обработано.

***

## Итого

| Подход | Корректность | Сложность | Вердикт |
| :-- | :-- | :-- | :-- |
| Деструкторы синглтонов | ❌ Порядок не определён | Низкая | Не использовать |
| Таймер + поллинг | ⚠️ Задержки, нет порядка | Средняя | Плохо |
| Глобальное событие | ⚠️ Нет фаз, нет порядка | Средняя | Недостаточно |
| **ShutdownManager** | ✅ Фазы + порядок + таймаут | Средняя | **Рекомендация** |
| `std::jthread` + stop_token | ✅ Нативно в C++20 | Низкая | Если доступен C++20 |

**Главная идея**: не полагайся на неявное (деструкторы static), а сделай **явный**, **упорядоченный**, **двухфазный** shutdown с таймаутами. `ShutdownManager` — это по сути тот же паттерн, что используется в серверных приложениях, игровых движках и системах реального времени.

