# IN PROGRESS — Текущие задачи

> **Обновлено**: 2026-02-21

---

## KernelCacheService (пакет тасок)

Детальные таски созданы. Выполняет другая AI, принимает Кодо (старшая).

| Таска | Статус |
|-------|--------|
| [TASK_KernelCacheService_001_StorageBackend](TASK_KernelCacheService_001_StorageBackend.md) | Ожидает |
| [TASK_KernelCacheService_002_KernelCacheService](TASK_KernelCacheService_002_KernelCacheService.md) | Ожидает |
| [TASK_KernelCacheService_003_FormScriptRefactor](TASK_KernelCacheService_003_FormScriptRefactor.md) | Ожидает |
| [TASK_KernelCacheService_004_Filters](TASK_KernelCacheService_004_Filters.md) | Ожидает |
| [TASK_KernelCacheService_005_FilterConfigService](TASK_KernelCacheService_005_FilterConfigService.md) | Ожидает |

План: [PLAN_KernelCacheService_DrvGPU.md](PLAN_KernelCacheService_DrvGPU.md)

---

## Следующие кандидаты (BACKLOG)

См. [BACKLOG.md](BACKLOG.md):
- Filters Stage 2: Text->kernel pipeline (FormScriptGenerator-like)
- ~~Filters Stage 3: Groq AI micro-agent~~ -> **DONE** (TASK-008)
- Performance benchmarks: замеры GPU vs CPU на разных конфигурациях
- Streaming: поддержка непрерывного потока (state persistence)
- Overlap-Save/Add: для длинных FIR через FFT
- ROCm Backend (полная реализация)
- Оконные функции
- Code Style рефакторинг

---

*Последнее обновление: 2026-02-18*
