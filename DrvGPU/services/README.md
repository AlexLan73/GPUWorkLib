# DrvGPU/services — Фоновые сервисы

Сервисы для профилирования, вывода в консоль, batch-обработки.

## Классы

| Файл | Класс | Описание |
|------|-------|----------|
| `gpu_profiler.hpp` | `GPUProfiler` | Singleton. Асинхронный сбор профилирования. `Start()`, `Record()`, `GetStats()`, `ExportJSON()`, `Stop()` |
| `batch_manager.hpp/cpp` | `BatchManager` | Разбиение данных на пакеты. `CreateBatches()`, расчёт batch_size |
| `console_output.hpp` | `ConsoleOutput` | Singleton. Потокобезопасный вывод в консоль. `Print()`, `PrintError()` |
| `service_manager.hpp` | `ServiceManager` | Singleton. Жизненный цикл сервисов (Logger, Profiler, Console). `Initialize()`, `Shutdown()` |
| `async_service_base.hpp` | `AsyncServiceBase<T>` | Базовый класс асинхронных сервисов |
| `profiling_types.hpp` | — | Типы сообщений профилирования |
| `profiling_stats.hpp` | `ProfilingStats` | Агрегированная статистика (min/max/avg) |

## Использование

```cpp
GPUProfiler::GetInstance().Start();
GPUProfiler::GetInstance().Record(0, "FFT", "Execute", data);
auto stats = GPUProfiler::GetInstance().GetStats(0);

ConsoleOutput::GetInstance().Print(0, "Message");
```

## Как тестировать

- **GPUProfiler**: `tests/test_gpu_profiler.hpp` — `test_gpu_profiler::run()` — Record, агрегация, PrintSummary
- **ConsoleOutput, ServiceManager**: `tests/test_services.hpp` — `test_services::run()` — многопоточные тесты
