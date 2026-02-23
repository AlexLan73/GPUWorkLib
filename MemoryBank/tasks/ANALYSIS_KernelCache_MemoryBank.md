# Анализ MemoryBank: всё, что касается Kernel Cache

> **Дата:** 2026-02-23
> **Цель:** Вывести для обсуждения — что оставить, что почистить

---

## 1. Прямо про KernelCacheService / on-disk cache

| Файл | Содержание | Рекомендация |
|------|------------|--------------|
| **tasks/PLAN_KernelCacheService_DrvGPU.md** | План (сокращён) — статус, архитектура, планы | ✅ Оставить (уже почищен) |
| **tasks/COMPLETED.md** | Запись KernelCacheService + FormSignal Refactor | ✅ Оставить |
| **DiscussionPlan/~6. KernelCache/Driver_Invalidation_Note.md** | Инвалидация кэша при смене драйвера | ✅ Оставить (backlog) |
| **tests/test_results_2026-02-21.md** | FormScript SaveKernel, LoadKernel, ListKernels — PASS | ✅ Оставить (история тестов) |

---

## 2. Упоминания kernel cache (контекстные)

| Файл | Контекст | Рекомендация |
|------|----------|--------------|
| **MASTER_INDEX.md** | FormScriptGenerator: "DSL + on-disk kernel cache" | ✅ Оставить |
| **sessions/2026-02-18.md** | "Text->kernel pipeline с FormScriptGenerator-like кэш-механизмом" (Filters Stage 2) | ✅ Оставить |
| **changelog/2026-02.md** | Упоминание kernels (FIR/IIR) | ✅ Оставить |

---

## 3. «kernel» в смысле OpenCL/HIP compute kernel (НЕ KernelCacheService)

Эти файлы про **GPU compute kernels** (ядра OpenCL/HIP), не про on-disk cache:

| Файл | Тема |
|------|------|
| DiscussionPlan/~1. FFT_FindAllMax/ | Custom kernel для детекции максимумов |
| DiscussionPlan/~2. Average/ | reduce_mean kernel |
| DiscussionPlan/~4. STD/ | Welford kernel |
| DiscussionPlan/~5. variance/ | Welford, Pairwise kernels |
| DiscussionPlan/~3. Median/ | Radix sort, kernel |
| tasks/PLAN_Heterodyne_LFM_Dechirp.md | dechirp_multiply.cl, dechirp_correct.cl |
| tasks/CHECK_Heterodyne_vs_Plan.md | Cache cl_kernel (кэш объекта в памяти) |
| tasks/PLAN_AMD_Radeon_9070_ROCm.md | Порт kernels в HIP |
| research/AMD_GPU_OpenCL_ROCm_ZeroCopy | kernel launch |
| sessions/2026-02-22.md | post_kernel_one_peak |

**Рекомендация:** Не трогать — это другая тема (compute kernels).

---

## 4. DiscussionPlan INDEX — ~6. KernelCache не в индексе

**DiscussionPlan/INDEX.md** — индекс Statistics Module (~1..~5). Папка **~6. KernelCache** не упомянута.

**Рекомендация:** Добавить в INDEX или оставить как есть (KernelCache — отдельная тема от Statistics).

---

## 5. Итог для обсуждения

### Оставить как есть
- PLAN_KernelCacheService_DrvGPU.md (сокращён)
- COMPLETED.md
- Driver_Invalidation_Note.md
- test_results, MASTER_INDEX, sessions, changelog

### Удалить
- Нет кандидатов (таски уже удалены ранее)

### Вопросы
1. **Driver_Invalidation_Note** — оставить в DiscussionPlan или перенести в Doc/DrvGPU/Services/?
2. **INDEX.md** — добавить ~6. KernelCache в DiscussionPlan INDEX?

---

*Создано: Кодо, 2026-02-23*
