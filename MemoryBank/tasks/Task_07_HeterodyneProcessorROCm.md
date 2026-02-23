# Task_07_HeterodyneProcessorROCm — Гетеродин на ROCm

> **Памятка для ИИ**: Если работаешь под Windows — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu). ROCm не поддерживается на Windows.

---

## 1. Цель

Реализовать HeterodyneProcessorROCm: Dechirp(), Correct() — порт kernels в HIP. Зависит от SpectrumProcessorROCm (SpectrumMaximaFinder).

---

## 2. Зависимости

- Task_00_DrvGPU (ROCmBackend)
- Task_03_SpectrumProcessorROCm (SpectrumMaximaFinder для f_beat)

---

## 3. Текущее состояние

Stub: `modules/heterodyne/src/heterodyne_processor_rocm.cpp` — все методы `throw std::runtime_error("not implemented")`.

---

## 4. Задачи

### 4.1 Dechirp

- Порт `dechirp_multiply.cl` в `.hip`
- Умножение на сопряжённый LFM
- Использовать LfmConjugateGenerator (CPU) или портировать в HIP

### 4.2 Correct

- Порт `dechirp_correct.cl` в `.hip`

### 4.3 Интеграция с SpectrumMaximaFinder

- FindAllMaxima или поиск f_beat для коррекции
- Использовать SpectrumProcessorROCm

### 4.4 CMake

- В `modules/heterodyne/CMakeLists.txt`: условно добавлять ROCm sources при `ENABLE_ROCM`
- Линковка: hip, spectrum_maxima

### 4.5 Тесты

- Создать `modules/heterodyne/tests/test_heterodyne_rocm.hpp`
- Добавить в `modules/heterodyne/tests/all_test.hpp`
- Эталон: f_beat в пределах F_BEAT_TOL (как в OpenCL тестах)
- Под `#if ENABLE_ROCM`

---

## 5. Чек-лист

- [ ] dechirp_multiply kernel в .hip
- [ ] dechirp_correct kernel в .hip
- [ ] HeterodyneProcessorROCm: Dechirp, Correct
- [ ] Интеграция с SpectrumProcessorROCm
- [ ] CMake
- [ ] test_heterodyne_rocm.hpp + all_test.hpp
- [ ] Компиляция без ошибок

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — секция 5.8
- [modules/heterodyne/](../../modules/heterodyne/)
- [Doc/Modules/heterodyne/Full.md](../../Doc/Modules/heterodyne/Full.md)
