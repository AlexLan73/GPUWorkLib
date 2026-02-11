# Дополнение к batch_large_data.md — Уточнённый алгоритм

> **Источник**: Рекомендации из анализа (2026-02-11)
> **Оригинал**: [batch_large_data.md](batch_large_data.md)

---

## I. ДОПОЛНЕНИЕ — Алгоритм принятия решения по batch

Алгоритм должен быть реализован следующим образом:

1. **Считаем требуемый размер данных** — байт на антенну × число антенн.
2. **Проверяем доступную память GPU** — `CL_DEVICE_GLOBAL_MEM_SIZE` (или через backend).
3. **Используем долю `memory_limit` от доступной памяти** — по умолчанию 70%; параметр задаётся программно извне.
4. **Если данные не помещаются** — работаем пачками по N антенн.
5. **Считаем batch** — N = floor(usable_memory / bytes_per_antenna); на базе этого создаём количество batch'ей.
6. **Слияние хвоста (merge tail)** — если в последнем batch остаётся [1..3] антенны, объединяем их с предыдущим batch (не создаём отдельный маленький batch).

---

## II. ССЫЛКА на BatchManager

**Реализация алгоритма**: `DrvGPU/services/batch_manager.hpp`

- `BatchManager::CalculateOptimalBatchSize(backend, total_items, item_memory_bytes, memory_limit)` — шаги 1–5
- `BatchManager::CreateBatches(total_items, items_per_batch, min_tail=3, merge_small_tail=true)` — шаг 6 (merge tail)

Использовать BatchManager при реализации batch-обработки в SpectrumMaximaFinder (или AntennaFFTCore).

---

## III. Параметр 70%

Параметр задаётся через `memory_limit` в:
- `BatchManager::CalculateOptimalBatchSize(..., memory_limit)`
- `BatchManager::CreateBatches(...)` — косвенно (batch_size уже рассчитан с учётом limit)

Значение по умолчанию: **0.7** (70%). Конфигурируемо программно.

---

*Создано: 2026-02-11*
