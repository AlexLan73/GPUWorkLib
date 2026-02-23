# Task_01_FFTProcessorROCm — Порт FFT на ROCm

> **Памятка для ИИ**: Если работаешь под Windows — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu). ROCm не поддерживается на Windows.

---

## 1. Цель

Портировать FFTProcessor на ROCm: hipFFT вместо clFFT, pre-callback (padding), post-kernel (mag/phase) в HIP.

---

## 2. Зависимости

- Task_00_DrvGPU (ROCmBackend)
- hipFFT

---

## 3. Задачи

### 3.1 FFTProcessorROCm класс

- Создать `FFTProcessorROCm` или добавить ветку по `BackendType` в существующий `FFTProcessor`
- Использовать hipFFT вместо clFFT
- Референс: [modules/fft_processor/include/fft_processor.hpp](../../modules/fft_processor/include/fft_processor.hpp)

### 3.2 Pre-callback (padding)

- Порт логики padding из OpenCL в HIP kernel
- Файл: `modules/fft_processor/kernels/` — добавить `.hip` или использовать inline HIP

### 3.3 Post-kernel (mag/phase)

- Порт post-processing (magnitude, phase) из OpenCL kernel в HIP
- Режимы: COMPLEX, MAGNITUDE_PHASE, MAGNITUDE_PHASE_FREQ

### 3.4 CMake

- В `modules/fft_processor/CMakeLists.txt`: условно подключать hipFFT при `ENABLE_ROCM`
- Линковка: `hipfft` или `hip::hipfft`

### 3.5 Тесты

- Создать `modules/fft_processor/tests/test_fft_processor_rocm.hpp`
- Добавить вызов в `modules/fft_processor/tests/all_test.hpp`
- Эталон: сравнение с OpenCL FFT (одинаковые входные данные → одинаковый результат)
- Под `#if ENABLE_ROCM`

---

## 4. Чек-лист

- [ ] FFTProcessorROCm или ветка BackendType::ROCm в FFTProcessor
- [ ] hipFFT инициализация, план, выполнение
- [ ] Pre-callback (padding) в HIP
- [ ] Post-kernel (mag/phase) в HIP
- [ ] CMake: hipFFT при ENABLE_ROCM
- [ ] test_fft_processor_rocm.hpp + all_test.hpp
- [ ] Компиляция без ошибок

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — секция 5.2
- [modules/fft_processor/](../../modules/fft_processor/)
- [Doc/Modules/fft_processor/Full.md](../../Doc/Modules/fft_processor/Full.md)
