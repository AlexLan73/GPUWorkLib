# Task_03_SpectrumProcessorROCm — Поиск максимумов спектра на ROCm

> **Памятка для ИИ**: Если работаешь под Windows — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu). ROCm не поддерживается на Windows.

---

## 1. Цель

Реализовать SpectrumProcessorROCm: hipFFT вместо clFFT, HIP kernel для post-processing (поиск максимума, параболическая интерполяция). Заменить stub в `spectrum_processor_rocm.cpp`.

---

## 2. Зависимости

- Task_00_DrvGPU (ROCmBackend)
- Task_01_FFTProcessorROCm (опционально — для референса hipFFT)
- hipFFT

---

## 3. Текущее состояние

Stub: [modules/fft_maxima/src/spectrum_processor_rocm.cpp](../../modules/fft_maxima/src/spectrum_processor_rocm.cpp) — все методы `throw std::runtime_error("not implemented")`.

---

## 4. Задачи

### 4.1 SpectrumProcessorROCm::Initialize

- Инициализация hipFFT плана, буферов
- Параметры из `SpectrumParams`

### 4.2 ProcessFromCPU

- Загрузка CPU данных на GPU (hipMemcpy H2D)
- hipFFT выполнение
- HIP kernel: magnitude, поиск максимума, параболическая интерполяция
- Возврат `std::vector<SpectrumResult>`

### 4.3 ProcessFromGPU

- Вход: `void*` (hipDeviceptr_t), размеры
- hipFFT in-place или out-of-place
- HIP kernel post-processing
- Возврат `std::vector<SpectrumResult>`

### 4.4 ProcessBatch / ProcessBatchFromGPU

- Batch обработка нескольких лучей
- Аналогично ProcessFromCPU/ProcessFromGPU

### 4.5 FindAllMaximaFromCPU / FindAllMaximaFromGPUPipeline

- Поиск всех максимумов (не только одного)
- Референс: [MemoryBank/DiscussionPlan/~1. FFT_FindAllMax/](../DiscussionPlan/~1. FFT_FindAllMax/)
- Scan-based compaction или rocPRIM

### 4.6 CMake

- В `modules/fft_maxima/CMakeLists.txt`: условно добавлять `spectrum_processor_rocm.cpp` при `ENABLE_ROCM`
- Линковка: hip, hipfft

### 4.7 Тесты

- Создать `modules/fft_maxima/tests/test_spectrum_maxima_rocm.hpp`
- Добавить в `modules/fft_maxima/tests/all_test.hpp`
- Эталон: сравнение с OpenCL результатом
- Под `#if ENABLE_ROCM`

---

## 5. Чек-лист

- [ ] SpectrumProcessorROCm::Initialize
- [ ] ProcessFromCPU, ProcessFromGPU
- [ ] ProcessBatch, ProcessBatchFromGPU
- [ ] FindAllMaximaFromCPU, FindAllMaximaFromGPUPipeline
- [ ] HIP kernels: magnitude, peak detection, parabolic interpolation
- [ ] CMake
- [ ] test_spectrum_maxima_rocm.hpp + all_test.hpp
- [ ] Компиляция без ошибок

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — секция 5.4
- [modules/fft_maxima/](../../modules/fft_maxima/)
- [Doc/Modules/fft_maxima/Full.md](../../Doc/Modules/fft_maxima/Full.md)
