# Task_06_FormSignalGeneratorROCm — Генератор формы сигнала на ROCm

> **Памятка для ИИ**: Если работаешь под Windows — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu). ROCm не поддерживается на Windows.

---

## 1. Цель

Портировать FormSignalGenerator на ROCm: form_signal kernel в `.hip`.

---

## 2. Зависимости

- Task_00_DrvGPU (ROCmBackend)

---

## 3. Задачи

### 3.1 Поиск kernel source

- Найти form_signal kernel в модуле signal_generators
- Порт в `.hip` файл

### 3.2 FormSignalGeneratorROCm класс

- Создать FormSignalGeneratorROCm или ветку BackendType::ROCm в FormSignalGenerator
- Загрузка и запуск HIP kernel
- Референс: [modules/signal_generators/include/generators/form_script_generator.hpp](../../modules/signal_generators/include/generators/form_script_generator.hpp)

### 3.3 CMake

- В `modules/signal_generators/CMakeLists.txt`: условно добавлять ROCm sources при `ENABLE_ROCM`
- Линковка: hip

### 3.4 Тесты

- Создать `modules/signal_generators/tests/test_form_signal_rocm.hpp`
- Добавить в `modules/signal_generators/tests/all_test.hpp`
- Эталон: сравнение с OpenCL
- Под `#if ENABLE_ROCM`

---

## 4. Чек-лист

- [ ] form_signal kernel в .hip
- [ ] FormSignalGeneratorROCm класс
- [ ] CMake
- [ ] test_form_signal_rocm.hpp + all_test.hpp
- [ ] Компиляция без ошибок

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — секция 5.7
- [modules/signal_generators/](../../modules/signal_generators/)
- [Doc/Modules/signal_generators/Full.md](../../Doc/Modules/signal_generators/Full.md)
