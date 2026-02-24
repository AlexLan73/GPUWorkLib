# Task_05_LchFarrowROCm — LCH Farrow на ROCm

> **Памятка для ИИ**: Если работаешь под Windows — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu). ROCm не поддерживается на Windows.

---

## 1. Цель

Портировать LchFarrow на ROCm: LCH_FARROW_KERNEL_SOURCE в `.hip`.

---

## 2. Зависимости

- Task_00_DrvGPU (ROCmBackend)

---

## 3. Задачи

### 3.1 Поиск kernel source

- Найти LCH_FARROW_KERNEL_SOURCE в модуле lch_farrow
- Порт в отдельный `.hip` файл

### 3.2 LchFarrowROCm класс

- Создать LchFarrowROCm или ветку BackendType::ROCm в LchFarrow
- Загрузка и запуск HIP kernel
- Использование ROCmBackend

### 3.3 CMake

- В `modules/lch_farrow/CMakeLists.txt`: условно добавлять ROCm sources при `ENABLE_ROCM`
- Линковка: hip

### 3.4 Тесты

- Создать `modules/lch_farrow/tests/test_lch_farrow_rocm.hpp`
- Добавить в `modules/lch_farrow/tests/all_test.hpp`
- Эталон: сравнение с OpenCL
- Под `#if ENABLE_ROCM`

---

## 4. Чек-лист

- [x] LCH_FARROW kernel в .hip ✅
- [x] LchFarrowROCm класс ✅
- [x] CMake ✅
- [x] test_lch_farrow_rocm.hpp + all_test.hpp ✅
- [x] Компиляция без ошибок ✅

> **Статус**: ✅ РЕАЛИЗОВАНО (2026-02-24, Кодо)
> C++ тесты: 4/4 PASSED. Python тесты: 5/5 PASSED.
> ⚠️ Не использовать целочисленные задержки! delay=3.0µs → row=47 матрицы Лагранжа (огромные коэффициенты). Использовать: 0.3, 1.7, 3.3, 4.9 µs.

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — секция 5.6
- [modules/lch_farrow/](../../modules/lch_farrow/)
