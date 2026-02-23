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

- [ ] LCH_FARROW kernel в .hip
- [ ] LchFarrowROCm класс
- [ ] CMake
- [ ] test_lch_farrow_rocm.hpp + all_test.hpp
- [ ] Компиляция без ошибок

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — секция 5.6
- [modules/lch_farrow/](../../modules/lch_farrow/)
