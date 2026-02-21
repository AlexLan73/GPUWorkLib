# BACKLOG — Очередь задач

> **Обновлено**: 2026-02-18

---

## ✅ Выполнено (перенесено в COMPLETED)

- **Filters Stage 1** (TASK-007): FIR + IIR GPU, Python bindings
- **AI Filter Pipeline Stage 3** (TASK-008): NL → scipy → GPU → plot
- **FormSignalGenerator**: все 6 этапов (FORM-001..005)

---

## Перспективные задачи

### ROCm Backend (Средний приоритет)

**Цель**: Добавить поддержку AMD GPU через ROCm/HIP

| Задача | Описание |
|--------|----------|
| ROCM-001 | Создать `SpectrumProcessorROCm` (hipFFT) |
| ROCM-002 | Интегрировать в SpectrumMaximaFinder через Strategy Pattern |
| ROCM-003 | Тесты на AMD GPU |

**Зависимости**: Нужен доступ к AMD GPU  
**План**: [PLAN_AMD_Radeon_9070_ROCm.md](PLAN_AMD_Radeon_9070_ROCm.md)

---

### Filters Stage 2 (Средний приоритет)

**Цель**: Text→kernel pipeline (FormScriptGenerator-like)

- Кэш скомпилированных kernel на диск
- См. [Doc/Modules/filters/Full.md](../../Doc/Modules/filters/Full.md)

---

### Code Style (Низкий приоритет)

**Цель**: Google C++ Style + 2-пробельная табуляция

**Сфера**: `DrvGPU/`

---

## Отложенные (после основного функционала)

| Задача | Описание |
|--------|----------|
| Statistics модуль | mean, std, variance на GPU |
| Heterodyne модуль | NCO, MixDown/MixUp |
| Overlap-Save/Add | Длинные FIR через FFT |
| AI-003 | Multi-step AI pipeline (LangChain/AutoGen) |

---

*Последнее обновление: 2026-02-18*
