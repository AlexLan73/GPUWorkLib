# BACKLOG — Очередь задач

> **Обновлено**: 2026-02-12

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
