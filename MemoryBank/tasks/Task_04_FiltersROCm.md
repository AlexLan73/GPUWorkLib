# Task_04_FiltersROCm — FIR и IIR фильтры на ROCm

> **Памятка для ИИ**: Если работаешь под Windows — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu). ROCm не поддерживается на Windows.

---

## 1. Цель

Портировать FirFilter и IirFilter на ROCm: kernels из `.cl` в `.hip`.

---

## 2. Зависимости

- Task_00_DrvGPU (ROCmBackend)

---

## 3. Задачи

### 3.1 FirFilterROCm

- Порт FIR kernel из OpenCL в HIP
- Файлы: `modules/filters/kernels/` — создать `.hip` версии
- Референс: [modules/filters/include/filters/fir_filter.hpp](../../modules/filters/include/filters/fir_filter.hpp)

### 3.2 IirFilterROCm

- Порт IIR kernel из OpenCL в HIP
- Референс: [modules/filters/include/filters/iir_filter.hpp](../../modules/filters/include/filters/iir_filter.hpp)

### 3.3 Интеграция

- FirFilterROCm, IirFilterROCm классы или ветки по BackendType в существующих FirFilter/IirFilter
- Использование ROCmBackend для Allocate, Memcpy, Synchronize

### 3.4 CMake

- В `modules/filters/CMakeLists.txt`: условно добавлять ROCm sources при `ENABLE_ROCM`
- Линковка: hip

### 3.5 Тесты

- Создать `modules/filters/tests/test_filters_rocm.hpp`
- Добавить в `modules/filters/tests/all_test.hpp`
- Эталон: GPU vs SciPy (`lfilter`, `sosfilt`)
- Под `#if ENABLE_ROCM`

---

## 4. Чек-лист

- [x] FIR kernel в .hip ✅
- [x] IIR kernel в .hip ✅
- [x] FirFilterROCm, IirFilterROCm ✅
- [x] CMake ✅
- [x] test_filters_rocm.hpp + all_test.hpp ✅
- [x] Компиляция без ошибок ✅

> **Статус**: ✅ РЕАЛИЗОВАНО (2026-02-24, Кодо)
> C++ тесты: 6/6 PASSED. Python тесты: FIR 5/5, IIR 5/5 PASSED.
> Python биндинги: FirFilterROCm, IirFilterROCm в gpuworklib.

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — секция 5.5
- [modules/filters/](../../modules/filters/)
- [Doc/Modules/filters/Full.md](../../Doc/Modules/filters/Full.md)
