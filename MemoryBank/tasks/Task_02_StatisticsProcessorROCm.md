# Task_02_StatisticsProcessorROCm — Статистика на ROCm

> **Памятка для ИИ**: Если работаешь под Windows — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu). ROCm не поддерживается на Windows.

---

## 1. Цель

Модуль **только ROCm** — не портируется на OpenCL в рамках этого плана. Входные данные: **все антенны сразу**, сигнал **complex float**.

---

## 2. Зависимости

- Task_00_DrvGPU (ROCmBackend)
- rocPRIM

---

## 3. Структура модуля

По аналогии с [modules/fft_processor/](../../modules/fft_processor/): `include/`, `src/`, `kernels/`, `tests/`.

```
modules/statistics/
├── include/
│   ├── statistics_processor.hpp
│   └── statistics_types.hpp
├── src/
│   └── statistics_processor.cpp
├── kernels/
│   └── *.hip
├── tests/
│   ├── test_statistics_rocm.hpp
│   └── all_test.hpp
└── CMakeLists.txt
```

---

## 4. Анализ: kernel vs rocPRIM

| Операция | Рекомендация | Обоснование |
|----------|--------------|-------------|
| Mean | Custom kernel (hierarchical reduction) или rocPRIM `reduce` | Простая редукция |
| Median | **rocPRIM** `radix_sort` + средний элемент или `rocprim::nth_element` | Свой kernel сложнее |
| Variance / STD | Custom kernel (Welford) | Один проход mean+variance+std, численно стабильно |

---

## 5. Задачи

### 5.1 StatisticsProcessor класс

- API: `ComputeMean()`, `ComputeMedian()`, `ComputeStatistics()` (mean+variance+std за один проход Welford)
- Вход: `void*` (hipDeviceptr_t), complex float, все антенны
- Выход: per-beam или scalar

### 5.2 Mean

- rocPRIM `reduce` или custom hierarchical reduction kernel

### 5.3 Median

- rocPRIM `radix_sort` + средний элемент или `rocprim::nth_element`
- Для complex float: сортировать по magnitude или real/imag — уточнить в [DiscussionPlan/INDEX.md](../DiscussionPlan/INDEX.md)

### 5.4 Variance / STD

- Custom HIP kernel: Welford's online algorithm
- Один проход: mean + variance + std

### 5.5 CMake

- Новый модуль `modules/statistics/CMakeLists.txt`
- Зависимости: drvgpu, rocPRIM, hip
- Условная сборка при `ENABLE_ROCM`

### 5.6 Тесты

- `modules/statistics/tests/test_statistics_rocm.hpp`
- Эталон: NumPy (`np.mean`, `np.std`, `np.median`)
- Под `#if ENABLE_ROCM`

---

## 6. Чек-лист

- [ ] modules/statistics/ структура
- [ ] StatisticsProcessor: Mean, Median, Variance, STD
- [ ] rocPRIM для median (и опционально mean)
- [ ] Welford kernel для variance/std
- [ ] CMake
- [ ] test_statistics_rocm.hpp + all_test.hpp
- [ ] Компиляция без ошибок

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — секция 5.3
- [MemoryBank/DiscussionPlan/INDEX.md](../DiscussionPlan/INDEX.md)
- [MemoryBank/specs/statistics_module.md](../specs/statistics_module.md)
