<!-- ДАЛЬШЕ ПЕРЕНОСИТЬ: Doc_Addition/Info_Graceful_Shutdown.md -->

# Graceful Shutdown — Анализ и рекомендации для GPUWorkLib

> Источник: MemoryBank/specs/Precpectiva/graceful_shutdown_analysis.md  
> Дата: 2026-02-11

---

## 1. Текущее состояние проекта

### Компоненты и lifecycle

| Компонент | Тип | Cleanup | Потоки | Очередь |
|-----------|-----|---------|--------|---------|
| DrvGPU | Локальный | Cleanup() в деструкторе | Нет | Нет |
| OpenCLBackend | Внутри DrvGPU | queue, context release | Нет | Нет |
| DefaultLogger | Per-GPU | Shutdown() в деструкторе | Нет | plog sync |
| GPUProfiler | Singleton | Stop() в ~AsyncServiceBase | Да (worker) | Да |
| SpectrumMaximaFinder | Локальный | ReleaseResources() в деструкторе | Нет | Нет |

В GPUWorkLib: plog синхронный; GPUProfiler — один AsyncServiceBase; нет ShutdownManager, всё RAII.

---

## 2. Рекомендации по этапам

- **Сейчас**: явный `profiler.Stop()` в main перед return.
- **SIGINT**: флаг `g_shutdown_requested` + проверка в длительных циклах.
- **ShutdownManager**: вводить при появлении нескольких асинхронных сервисов.
- **Порядок**: Profiler.Stop() → Logger.Shutdown() → DrvGPU.Cleanup (RAII).
- **Batch**: `if (g_shutdown_requested) break;` в цикле batch.

(Полный текст с вариантами A/B/C и кодом — в исходном файле MemoryBank/specs/Precpectiva/.)
