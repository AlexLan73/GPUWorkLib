@page drvgpu_console ConsoleOutput --- Потокобезопасный вывод

@tableofcontents

@section console_overview Назначение

@ref drv_gpu_lib::ConsoleOutput "ConsoleOutput" --- синглтон-сервис
для потокобезопасного вывода в консоль при одновременной работе нескольких GPU.

Файл: `DrvGPU/services/console_output.hpp`

@subsection console_problem Проблема

При одновременной записи в stdout с 8--10 GPU вывод перемешивается.
Сообщения от разных GPU чередуются непредсказуемо, теряется читаемость.

@subsection console_solution Решение

ConsoleOutput предоставляет:

- **Выделенный фоновый поток** для всего вывода в консоль
- **Очередь сообщений** --- потоки GPU только делают Enqueue (почти без задержки)
- **Форматирование**: `[ЧЧ:ММ:СС.ммм] [UVR] [GPU_XX] [Модуль] сообщение`
  - `UVR` = уровень (DBG, INF, WRN, ERR)
- **Per-GPU фильтрация** --- включение/отключение вывода для каждого GPU через `configGPU.json`

```
GPU Thread 0 --> Print(0, "FFT", "Done")    --> Enqueue() --+
GPU Thread 1 --> Print(1, "FFT", "Done")    --> Enqueue() --+--> [Очередь] --> Worker --> stdout
GPU Thread N --> PrintError(N, "FFT", "Err") --> Enqueue() --+
```

@note ConsoleOutput наследует `AsyncServiceBase<ConsoleMessage>`.
Вызовы `Print()` / `PrintError()` неблокирующие.

---

@section console_lifecycle Жизненный цикл

@code{.cpp}
// Запуск (обычно через ServiceManager)
drv_gpu_lib::ConsoleOutput::GetInstance().Start();

// ... работа модулей на GPU ...

// Остановка
drv_gpu_lib::ConsoleOutput::GetInstance().Stop();
@endcode

@warning Обычно запуск/остановка выполняется через
@ref drv_gpu_lib::ServiceManager "ServiceManager", а не напрямую.

---

@section console_api API вывода сообщений

Все методы **неблокирующие** --- сообщение помещается в очередь и возвращается
управление вызывающему потоку.

@subsection console_api_print Print --- информационное сообщение

@code{.cpp}
auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
con.Print(0, "FFT", "Processing 1024 beams...");
// Вывод: [14:30:00.123] [INF] [GPU_00] [FFT] Processing 1024 beams...
@endcode

@subsection console_api_warning PrintWarning --- предупреждение

@code{.cpp}
con.PrintWarning(0, "MemManager", "Low GPU memory: 512 MB remaining");
// Вывод: [14:30:00.124] [WRN] [GPU_00] [MemManager] Low GPU memory: 512 MB remaining
@endcode

@subsection console_api_error PrintError --- ошибка

@code{.cpp}
con.PrintError(0, "FFT", "Failed to allocate buffer!");
// Вывод в stderr: [14:30:00.125] [ERR] [GPU_00] [FFT] Failed to allocate buffer!
@endcode

@note Сообщения уровня `ERRLEVEL` выводятся в `stderr`, все остальные --- в `stdout`.

@subsection console_api_debug PrintDebug --- отладочное сообщение

@code{.cpp}
con.PrintDebug(0, "Kernel", "Work group size: 256");
// Вывод: [14:30:00.126] [DBG] [GPU_00] [Kernel] Work group size: 256
@endcode

@subsection console_api_system PrintSystem --- системное (без GPU)

@code{.cpp}
con.PrintSystem("ServiceManager", "All services started");
// Вывод: [14:30:00.127] [INF] [SYSTEM] [ServiceManager] All services started
@endcode

`gpu_id = -1` --- системное сообщение, без префикса GPU.

---

@section console_format Формат вывода

```
[ЧЧ:ММ:СС.ммм] [УВР] [GPU_XX] [Модуль] сообщение
```

| Компонент | Описание | Пример |
|-----------|----------|--------|
| Время | Часы:Минуты:Секунды.Миллисекунды | `[14:30:00.123]` |
| Уровень | DBG / INF / WRN / ERR | `[INF]` |
| GPU | GPU_XX (00--99) или SYSTEM | `[GPU_00]` |
| Модуль | Имя модуля-источника | `[FFT]` |
| Сообщение | Текст | `Processing 1024 beams...` |

---

@section console_message ConsoleMessage --- структура сообщения

@ref drv_gpu_lib::ConsoleMessage "ConsoleMessage" содержит:

| Поле | Тип | Описание |
|------|-----|----------|
| `gpu_id` | `int` | Индекс GPU (-1 = системное) |
| `module_name` | `string` | Имя модуля-источника |
| `level` | `ConsoleMessage::Level` | Уровень: DEBUG, INFO, WARNING, ERRLEVEL |
| `message` | `string` | Текст сообщения |
| `timestamp` | `time_point` | Время создания (автоматически) |

@subsection console_message_levels Уровни сообщений

| Уровень | Значение | Вывод | Описание |
|---------|----------|-------|----------|
| `DEBUG` | 0 | stdout | Отладочная информация |
| `INFO` | 1 | stdout | Нормальная работа |
| `WARNING` | 2 | stdout | Предупреждение |
| `ERRLEVEL` | 3 | stderr | Ошибка |

@note Уровень называется `ERRLEVEL`, а не `ERROR`, чтобы избежать конфликта
с макросом `ERROR` в Windows.

---

@section console_enable Включение/выключение

@subsection console_enable_global Глобальное

@code{.cpp}
auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

con.SetEnabled(true);   // Включить весь вывод
con.SetEnabled(false);  // Полностью отключить (сообщения отбрасываются)
bool on = con.IsEnabled();
@endcode

@subsection console_enable_pergpu Per-GPU

@code{.cpp}
con.SetGPUEnabled(0, true);   // GPU 0: включить
con.SetGPUEnabled(1, false);  // GPU 1: отключить вывод
bool on = con.IsGPUEnabled(0);
@endcode

@note При использовании `ServiceManager::InitializeFromConfig()` per-GPU
настройки берутся из поля `is_console` в `configGPU.json`.

---

@section console_config Конфигурация через configGPU.json

@code{.json}
{
  "gpus": [
    { "id": 0, "is_console": true  },
    { "id": 1, "is_console": true  },
    { "id": 2, "is_console": false }
  ]
}
@endcode

Поле `is_console` управляет включением ConsoleOutput для каждого GPU.
GPU с `is_console: false` не будут генерировать вывод (Print() для них игнорируется).

---

@section console_example Полный пример

@code{.cpp}
#include "DrvGPU/services/console_output.hpp"
#include "DrvGPU/services/service_manager.hpp"

int main() {
    // Вариант 1: через ServiceManager (рекомендуется)
    auto& sm = drv_gpu_lib::ServiceManager::GetInstance();
    sm.InitializeFromConfig("configGPU.json");
    sm.StartAll();

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

    // Информация
    con.Print(0, "Main", "GPU 0 initialized");
    con.Print(1, "Main", "GPU 1 initialized");

    // Предупреждение
    con.PrintWarning(0, "MemManager", "Using 80% of GPU memory");

    // Ошибка (идёт в stderr)
    con.PrintError(1, "FFT", "Kernel compilation failed!");

    // Системное сообщение (без GPU)
    con.PrintSystem("Main", "All modules loaded");

    // Отключить GPU 1
    con.SetGPUEnabled(1, false);
    con.Print(1, "FFT", "This will NOT be printed");

    // Снова включить
    con.SetGPUEnabled(1, true);
    con.Print(1, "FFT", "This WILL be printed");

    sm.StopAll();
    return 0;
}
@endcode

Ожидаемый вывод:

```
[14:30:00.001] [INF] [GPU_00] [Main] GPU 0 initialized
[14:30:00.002] [INF] [GPU_01] [Main] GPU 1 initialized
[14:30:00.003] [WRN] [GPU_00] [MemManager] Using 80% of GPU memory
[14:30:00.004] [ERR] [GPU_01] [FFT] Kernel compilation failed!
[14:30:00.005] [INF] [SYSTEM] [Main] All modules loaded
[14:30:00.010] [INF] [GPU_01] [FFT] This WILL be printed
[14:30:00.011] [INF] [SYSTEM] [ServiceManager] Stopping all services...
```

---

@section console_threading Потокобезопасность

ConsoleOutput гарантирует:

1. **Неблокирующий Enqueue** --- вызов `Print()` из потока GPU занимает наносекунды
2. **Упорядоченный вывод** --- сообщения выводятся в порядке поступления в очередь
3. **Один writer** --- только рабочий поток пишет в stdout/stderr
4. **Безопасное завершение** --- `Stop()` дожидается обработки всех сообщений в очереди

@warning Деструктор `~ConsoleOutput()` вызывает `Stop()` ДО сброса vtable.
Это критично: если `Stop()` вызывается только в деструкторе базового класса
`AsyncServiceBase`, `ProcessMessage()` (виртуальная функция) приведёт к UB.

---

@section console_inheritance Наследование

```
AsyncServiceBase<ConsoleMessage>
         |
         v
   ConsoleOutput (final)
```

Методы `AsyncServiceBase`:
- `Start()` --- запуск рабочего потока
- `Stop()` --- остановка (drain + join)
- `Enqueue(msg)` --- добавление в очередь
- `GetProcessedCount()` --- количество обработанных сообщений
- `GetQueueSize()` --- текущий размер очереди
- `IsRunning()` --- статус рабочего потока

---

@section console_rules Правила использования

@warning **Весь вывод в консоль** в проекте GPUWorkLib --- ТОЛЬКО через ConsoleOutput!
При наличии 10 GPU без единой точки вывода будет хаос.

@warning **Запрещено** использовать `std::cout` / `printf` напрямую
из модулей GPU. Только `ConsoleOutput::GetInstance().Print(...)`.

@note Исключение: `GPUProfiler::PrintReport()` и `GPUProfiler::PrintSummary()`
могут использовать `std::cout` напрямую, т.к. они вызываются из основного потока
после синхронизации.

---

@section console_related Связанные страницы

- @ref drvgpu_main --- Главная страница DrvGPU
- @ref drvgpu_architecture --- Архитектура DrvGPU
- @ref drvgpu_profiler --- GPUProfiler API
