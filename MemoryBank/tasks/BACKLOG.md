# BACKLOG — Очередь задач

> **Обновлено**: 2026-02-11

---

## Перспективные задачи

### FormSignalGenerator (Высокий приоритет)

**Цель**: Мультиканальный генератор комплексных сигналов (формула getX) с задержкой, амплитудой, шумом.

| Задача | Описание |
|--------|----------|
| FORM-001 | FormParams + FormSignalGenerator (OpenCL kernel) |
| FORM-002 | FormScriptGenerator + DSL + on-disk кэш по имени |
| FORM-003 | SignalService + Factory |
| FORM-004 | Python bindings + example с графиками |
| FORM-005 | ROCm заглушки |

**Checklist**: [CHECKLIST_FormSignalGenerator.md](CHECKLIST_FormSignalGenerator.md)  
**Спецификация**: [specs/Form_signals.md](../specs/Form_signals.md)

---

### ROCm Backend (Средний приоритет)

**Цель**: Добавить поддержку AMD GPU через ROCm/HIP

| Задача | Описание |
|--------|----------|
| ROCM-001 | Создать `SpectrumProcessorROCm` (hipFFT) |
| ROCM-002 | Интегрировать в SpectrumMaximaFinder через Strategy Pattern |
| ROCM-003 | Тесты на AMD GPU |

**Зависимости**: Нужен доступ к AMD GPU




### Code Style (Низкий приоритет)

**Цель**: Google C++ Style + 2-пробельная табуляция

**Сфера**: `DrvGPU/`

---

## Отложенные (после основного функционала)

| Задача | Описание |
|--------|----------|
| Filters модуль | FIR, IIR фильтры на GPU |
| Statistics модуль | mean, std, variance на GPU |
| Heterodyne модуль | NCO, MixDown/MixUp |

---

*Последнее обновление: 2026-02-12*
