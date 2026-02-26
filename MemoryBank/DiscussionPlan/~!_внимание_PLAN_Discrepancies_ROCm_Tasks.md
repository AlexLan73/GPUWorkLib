# План разногласий: ROCm Tasks vs Plans

> **Дата**: 2026-02-24
> **Автор**: Кодо
> **Назначение**: Сводка расхождений между планами, задачами и фактическим состоянием для упрощения исполнения

---

## 📋 Содержание

1. [Сводная таблица разногласий](#1-сводная-таблица-разногласий)
2. [Разногласия по порядку модулей](#2-разногласия-по-порядку-модулей)
3. [Разногласия по ZeroCopy и HybridBackend](#3-разногласия-по-zerocopy-и-hybridbackend)
4. [Разногласия по документации и статусам](#4-разногласия-по-документации-и-статусам)
5. [Разногласия по BACKLOG и IN_PROGRESS](#5-разногласия-по-backlog-и-in_progress)
6. [План исполнения (пошаговый)](#6-план-исполнения-пошаговый)

---

## 1. Сводная таблица разногласий

| # | Разногласие | PLAN_AMD | PLAN_ROCm | Tasks / Факт | Рекомендация |
|---|-------------|----------|-----------|--------------|--------------|
| 1 | Порядок модулей | fft→fft_maxima→statistics→signal→filters→lch | fft→statistics→fft_maxima→filters→lch→signal→heterodyne | Факт: все Task_00–07 DONE | Унифицировать порядок в планах |
| 2 | Heterodyne в плане | ❌ Нет в списке | ✅ Модуль 7 | Task_07 DONE | Добавить в PLAN_AMD |
| 3 | ZeroCopy | ❌ Не упомянут | ✅ Часть 2, 6 | Код есть, тесты в all_test | Добавить в PLAN_AMD |
| 4 | HybridBackend | ❌ Не упомянут | ✅ Часть 3 | Код есть, тесты в all_test | Добавить в PLAN_AMD |
| 5 | BACKLOG | ROCM-001/002/003, Statistics «отложен» | — | Устарел: всё DONE | Обновить BACKLOG |
| 6 | MASTER_INDEX | ROCm «planned» | — | ROCm DONE, 30/30 Python | Обновить MASTER_INDEX |
| 7 | IN_PROGRESS | «ROCm Backend» | — | Task_00–07 DONE | Обновить на Task_08/09 |
| 8 | NEXT_SESSION | Task_08/09 «НЕ НАЧАТО» | — | Код ZeroCopy/Hybrid есть | Уточнить статус |
| 9 | rocm_profiling | ✅ В PLAN_AMD 2.1 | ❌ Не упомянут | Не проверено | Проверить наличие |
| 10 | Etap 1 (Setup) | check_rocm_env.sh, AMD_Radeon_9070_Setup.md | ROCm_Setup_Instructions.md | Файлы? | Проверить наличие |

---

## 2. Разногласия по порядку модулей

### 2.1 PLAN_AMD_Radeon_9070_ROCm.md (Этап 3)

```
3.1 fft_processor
3.2 fft_maxima
3.3 statistics
3.4 signal_generators
3.5 filters
3.6 lch_farrow
```

**Отсутствует**: Heterodyne (гетеродин)

### 2.2 PLAN_ROCm_DrvGPU_Full.md (Часть 5.1)

```
0  ROCmBackend + rocm_core
1  FFTProcessorROCm
2  StatisticsProcessorROCm
3  SpectrumProcessorROCm (fft_maxima)
4  FirFilterROCm, IirFilterROCm
5  LchFarrowROCm
6  FormSignalGeneratorROCm
7  HeterodyneProcessorROCm
```

### 2.3 Фактический порядок в TASKS_ROCm_INDEX

```
Task_00 DrvGPU
Task_01 FFTProcessor
Task_02 Statistics
Task_03 SpectrumProcessor
Task_04 Filters
Task_05 LchFarrow
Task_06 FormSignal
Task_07 Heterodyne
Task_08 ZeroCopy (опционально)
Task_09 HybridBackend (опционально)
```

### 2.4 Рекомендация

- **PLAN_AMD**: добавить пункт 3.7 HeterodyneProcessorROCm
- **Унификация**: PLAN_AMD и PLAN_ROCm должны отражать один и тот же порядок (как в TASKS_ROCm_INDEX)

---

## 3. Разногласия по ZeroCopy и HybridBackend

### 3.1 PLAN_AMD

- **ZeroCopy**: не упомянут
- **HybridBackend**: не упомянут
- **OPENCLandROCm**: не упомянут

### 3.2 PLAN_ROCm

- **Часть 2**: ZeroCopy — OpenCL ↔ ROCm через dma-buf
- **Часть 3**: OPENCLandROCm — гибридный режим (HybridBackend)
- **Часть 6**: ZeroCopy — детальный план интеграции

### 3.3 Фактическое состояние (проверено 2026-02-24)

| Компонент | Файл | Статус |
|-----------|------|--------|
| opencl_export.hpp | DrvGPU/backends/opencl/opencl_export.hpp | ✅ Есть (header-only) |
| opencl_export.cpp | — | ❌ Не нужен (inline в .hpp) |
| zero_copy_bridge | DrvGPU/backends/rocm/zero_copy_bridge.hpp/.cpp | ✅ Есть |
| hybrid_backend | DrvGPU/backends/hybrid/hybrid_backend.hpp/.cpp | ✅ Есть |
| test_zero_copy.hpp | DrvGPU/tests/test_zero_copy.hpp | ✅ Есть |
| test_hybrid_backend.hpp | DrvGPU/tests/test_hybrid_backend.hpp | ✅ Есть |
| all_test.hpp | Вызов test_zero_copy::run(), test_hybrid_backend::run() | ✅ Обновлён |
| CMakeLists.txt | zero_copy_bridge, hybrid_backend в DRVGPU_ROCM_SOURCES | ✅ Обновлён |
| drv_gpu.cpp | CreateBackend(OPENCLandROCm) | ✅ Есть |

### 3.4 Разногласие в COMPLETED vs NEXT_SESSION

- **COMPLETED.md**: «Task_09 — ZeroCopy + HybridBackend ✅ … ⚠️ all_test.hpp и CMakeLists.txt НЕ обновлены»
- **Факт**: all_test.hpp и CMakeLists.txt **обновлены**
- **NEXT_SESSION.md**: «Task_08 ZeroCopy ⏳ НЕ НАЧАТО», «Task_09 HybridBackend ⏳ НЕ НАЧАТО»

**Вывод**: Код ZeroCopy и HybridBackend написан и интегрирован. Статус в NEXT_SESSION устарел. Нужно уточнить: **тесты запускались на Linux с Radeon 9070?**

### 3.5 Рекомендация

1. Обновить PLAN_AMD: добавить Этап 4 — ZeroCopy и HybridBackend
2. Обновить NEXT_SESSION: уточнить статус Task_08/09 (код есть, тесты — проверить)
3. Запустить `test_zero_copy::run()` и `test_hybrid_backend::run()` на Debian с Radeon 9070

---

## 4. Разногласия по документации и статусам

### 4.1 MASTER_INDEX.md

| Поле | Текущее | Должно быть |
|------|---------|-------------|
| Statistics | Planned | **Active** (ROCm DONE) |
| ROCm backend | «В работе», «planned» | **DONE** (Task_00–07) |
| Python классы | Нет ROCm | Добавить: ROCmGPUContext, FirFilterROCm, IirFilterROCm, LchFarrowROCm, HeterodyneROCm, StatisticsProcessor |
| Дата обновления | 2026-02-23 | 2026-02-24 |

### 4.2 Doc/AMD_Radeon_9070_Setup.md

- **PLAN_AMD 1.5**: «Документ Doc/AMD_Radeon_9070_Setup.md с чек-листом установки»
- **Проверить**: существует ли файл? Если нет — создать по образцу ROCm_Setup_Instructions.md

### 4.3 scripts/check_rocm_env.sh

- **PLAN_AMD 1.5**: «Скрипт scripts/check_rocm_env.sh»
- **Проверить**: существует ли файл?

---

## 5. Разногласия по BACKLOG и IN_PROGRESS

### 5.1 BACKLOG.md

| Задача | Текущий статус | Факт |
|--------|----------------|------|
| ROCM-001 SpectrumProcessorROCm | В работе | ✅ DONE (Task_03) |
| ROCM-002 Strategy Pattern | В работе | ✅ DONE (Factory) |
| ROCM-003 Тесты на AMD GPU | В работе | ✅ DONE (30/30 Python) |
| Statistics модуль | Отложен | ✅ DONE (StatisticsProcessorROCm) |

**Рекомендация**: Обновить BACKLOG — перенести ROCm задачи в COMPLETED, убрать «отложен» у Statistics.

### 5.2 IN_PROGRESS.md

- **Текущее**: «ROCm Backend — см. PLAN_AMD»
- **Рекомендация**: «Task_08 ZeroCopy (тесты), Task_09 HybridBackend (тесты), MASTER_INDEX обновление»

---

## 6. План исполнения (пошаговый)

### Шаг 1: Проверка наличия файлов (5 мин)

**Проверено 2026-02-24**:
| Файл | Статус |
|------|--------|
| scripts/check_rocm_env.sh | ❌ НЕТ |
| Doc/AMD_Radeon_9070_Setup.md | ❌ НЕТ |
| DrvGPU/backends/rocm/rocm_profiling.hpp | ❌ НЕТ |

```bash
# Повторная проверка
ls -la scripts/check_rocm_env.sh 2>/dev/null || echo "НЕТ"
ls -la Doc/AMD_Radeon_9070_Setup.md 2>/dev/null || echo "НЕТ"
ls -la DrvGPU/backends/rocm/rocm_profiling.hpp 2>/dev/null || echo "НЕТ"
```

**Действия**:
- Если `check_rocm_env.sh` нет — создать (проверка rocminfo, hipinfo, CMake find_package(hip))
- Если `AMD_Radeon_9070_Setup.md` нет — создать на основе ROCm_Setup_Instructions.md

---

### Шаг 2: Обновить PLAN_AMD_Radeon_9070_ROCm.md (10 мин)

1. Добавить в таблицу «Порядок работ» (стр. 183):
   - 3.7 HeterodyneProcessorROCm
2. Добавить Этап 4:
   - 4.1 ZeroCopy (OpenCL↔ROCm через dma-buf)
   - 4.2 HybridBackend (OPENCLandROCm)
3. Добавить в таблицу тестов:
   - test_zero_copy.hpp
   - test_hybrid_backend.hpp

---

### Шаг 3: Обновить BACKLOG.md (3 мин)

- Удалить или пометить DONE: ROCM-001, ROCM-002, ROCM-003
- Удалить «Statistics модуль — отложен»
- Добавить: «Task_08 ZeroCopy — тесты на Linux», «Task_09 HybridBackend — тесты на Linux»

---

### Шаг 4: Обновить IN_PROGRESS.md (2 мин)

```
В работе:
- Task_08 ZeroCopy — запуск тестов на Debian/Radeon 9070
- Task_09 HybridBackend — запуск тестов
- MASTER_INDEX.md — обновление статусов ROCm
```

---

### Шаг 5: Обновить MASTER_INDEX.md (15 мин)

1. Statistics: Planned → **Active**
2. ROCm: «В работе» → «**DONE** (Task_00–07, 30/30 Python)»
3. Python классы: добавить ROCm-список (ROCmGPUContext, FirFilterROCm, …)
4. Перспективные задачи: ROCm backend → DONE
5. Дата: 2026-02-24

---

### Шаг 6: Обновить NEXT_SESSION.md (5 мин)

1. Уточнить статус Task_08/09:
   - Код: ✅ написан
   - Интеграция: ✅ all_test.hpp, CMakeLists.txt
   - Тесты: ⏳ запуск на Linux (sg render -c "./GPUWorkLib")
2. Убрать «НЕ НАЧАТО» для Task_08/09 — заменить на «Код готов, тесты — проверить»

---

### Шаг 7: Запуск тестов ZeroCopy и HybridBackend (на Linux)

```bash
# На Debian с Radeon 9070
sg render -c "./build/debian-radeon9070/GPUWorkLib"
# Проверить вывод: test_zero_copy::run(), test_hybrid_backend::run()
```

**Если тесты падают**:
- Открыть `DrvGPU/tests/test_zero_copy.hpp`, `test_hybrid_backend.hpp`
- Проверить логи в Logs/DRVGPU_XX/
- Использовать Context7 / sequential-thinking для отладки dma-buf / hipImportExternalMemory

---

### Шаг 8: Обновить COMPLETED.md (2 мин)

- Убрать «⚠️ all_test.hpp и CMakeLists.txt НЕ обновлены»
- Добавить: «all_test.hpp и CMakeLists.txt обновлены (2026-02-24)»

---

## 7. Инструменты для помощи (по CLAUDE.md)

| Задача | Инструмент | Когда использовать |
|--------|------------|-------------------|
| Контекст по hipImportExternalMemory, dma-buf | **Context7** | Отладка ZeroCopy |
| Статьи по cl_khr_external_memory_dma_buf | **mcp_web_fetch** | Поиск API |
| Выбор архитектуры HybridBackend (A vs B) | **sequential-thinking** | Если нужен рефакторинг |
| Примеры ZeroCopy на GitHub | **GitHub MCP** | Референсный код |

---

## 8. Чек-лист для исполнителя

- [ ] Шаг 1: Проверить check_rocm_env.sh, AMD_Radeon_9070_Setup.md
- [ ] Шаг 2: Обновить PLAN_AMD (Heterodyne, ZeroCopy, HybridBackend)
- [ ] Шаг 3: Обновить BACKLOG
- [ ] Шаг 4: Обновить IN_PROGRESS
- [ ] Шаг 5: Обновить MASTER_INDEX
- [ ] Шаг 6: Обновить NEXT_SESSION
- [ ] Шаг 7: Запустить тесты ZeroCopy/HybridBackend на Linux
- [ ] Шаг 8: Обновить COMPLETED (убрать предупреждение)

---

## Ссылки

- [PLAN_AMD_Radeon_9070_ROCm.md](../tasks/PLAN_AMD_Radeon_9070_ROCm.md)
- [PLAN_ROCm_DrvGPU_Full.md](../tasks/PLAN_ROCm_DrvGPU_Full.md)
- [NEXT_SESSION.md](../tasks/NEXT_SESSION.md)
- [Task_08_ZeroCopy.md](../tasks/Task_08_ZeroCopy.md)
- [Task_09_HybridBackend.md](../tasks/Task_09_HybridBackend.md)
- [AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md](../research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md)

---

*Создано: 2026-02-24, Кодо*
