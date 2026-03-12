# Инструкция: проверка Task_13 Strategies Pipeline FftFunc

> **Для AI-проверяющего**: после выполнения Task_13 другой AI — проверить по этому чеклисту.
> **Автор инструкции**: Кодо | **Дата**: 2026-03-14

---

## 1. Порядок проверки

1. Собрать проект: `cmake -B build && cmake --build build`
2. Запустить тесты strategies: `./run_test strategies` (или через main)
3. Проверить, что все тесты проходят
4. Проверить по чеклисту ниже

---

## 2. Чеклист проверки

### 2.1. ProcessMagnitudeToBuffer (fft_func)

- [ ] В `modules/fft_func/include/complex_to_mag_phase_rocm.hpp` есть метод:
  ```cpp
  void ProcessMagnitudeToBuffer(void* gpu_complex_in, void* gpu_magnitude_out,
      const MagPhaseParams& params);
  ```
- [ ] Метод НЕ вызывает hipMalloc — пишет напрямую в `gpu_magnitude_out`
- [ ] Реализация в `complex_to_mag_phase_rocm.cpp` использует `ExecuteMagnitudeKernel(gpu_complex_in, gpu_magnitude_out, total, inv_n)`
- [ ] Unit-тест: ProcessMagnitudeToBuffer даёт те же результаты, что ProcessMagnitudeToGPU + hipMemcpy

### 2.2. strategies: do_window_fft

- [ ] В `antenna_processor_v1.cpp` шаг 4 (magnitudes) **НЕ** вызывает `magnitudes_kernel_`
- [ ] Вместо него вызывается `ComplexToMagPhaseROCm::ProcessMagnitudeToBuffer(d_spectrum_, d_magnitudes_, params)`
- [ ] strategies создаёт/держит экземпляр `ComplexToMagPhaseROCm` (или получает через dependency)
- [ ] После ProcessMagnitudeToBuffer `d_magnitudes_` заполнен — pipeline работает как раньше

### 2.3. CPU wrappers (vector<float>)

#### statistics

- [ ] `OneMaxFromFloat` или аналогичный — H2D mags, вызов one_max kernel, D2H
- [ ] `GlobalMinMaxFromFloat` — H2D mags, вызов minmax kernel, D2H
- [ ] `AllMaximaFromMagnitudes` — H2D mags, zeros для fft_data, Execute, D2H

#### fft_func

- [ ] `OneMaxParabolaFromFloat` — H2D mags, вызов one_max_no_phase (spectrum = zeros), D2H

### 2.4. DrvGPU backends

- [ ] В `DrvGPU/backends/rocm/rocm_backend.hpp` объявлен `AllocateManaged(size_t)` или флаг в `Allocate`
- [ ] В `rocm_backend.cpp` реализация использует `hipMallocManaged`
- [ ] `Free()` для managed — тот же `hipFree` (корректно для hipMallocManaged)

### 2.5. Streams (бенчмарк)

- [ ] В `antenna_processor_v1.hpp` добавлены `stream_debug3a_`, `stream_debug3b_`, `stream_debug3c_` (или конфиг)
- [ ] Бенчмарк сравнивает: 1 stream vs 3 streams — время всех post-FFT операций
- [ ] Результаты зафиксированы (или вывод в консоль/файл)

### 2.6. CMake и сборка

- [ ] strategies линкует fft_func (если ещё не линкует)
- [ ] Нет циклических зависимостей
- [ ] `ENABLE_ROCM` — все новые файлы под `#if ENABLE_ROCM`

### 2.7. Документация

- [ ] `Doc/Modules/strategies/Quick.md` или Full.md обновлён — описан новый flow (ProcessMagnitudeToBuffer)
- [ ] `Doc/Modules/fft_func/` — ProcessMagnitudeToBuffer документирован

---

## 3. Частые ошибки (проверить)

| Ошибка | Как проверить |
|--------|---------------|
| ProcessMagnitudeToBuffer делает hipMalloc | grep -r "hipMalloc" в ProcessMagnitudeToBuffer |
| strategies забыл включить fft_func | CMakeLists.txt strategies — target_link_libraries fft_func |
| OneMaxParabola kernel получает spectrum=nullptr | kernel crash или segfault — передать нулевой буфер |
| AllMaxima требует complex не-null | Execute принимает zeros — проверить null-check в AllMaximaPipeline |
| hipMallocManaged не освобождается | Тот же hipFree — проверить в docs HIP |

---

## 4. Результат проверки

Заполнить при проверке:

```
Дата проверки: _______________
Проверяющий: AI (Кодо)

ProcessMagnitudeToBuffer:     [ ] OK  [ ] FAIL
strategies do_window_fft:      [ ] OK  [ ] FAIL
CPU wrappers (statistics):    [ ] OK  [ ] FAIL
CPU wrappers (fft_func):      [ ] OK  [ ] FAIL
AllocateManaged (rocm):       [ ] OK  [ ] FAIL
Streams бенчмарк:             [ ] OK  [ ] FAIL
Сборка и тесты:               [ ] OK  [ ] FAIL

Замечания:
_________________________________
_________________________________
```

---

## 5. Ссылки

- План: `MemoryBank/DiscussionPlan/StrategiesPipeline/PLAN.md`
- Task: `MemoryBank/tasks/Task_13_StrategiesPipelineFftFunc.md`
- strategies: `modules/strategies/`
- fft_func: `modules/fft_func/`
- statistics: `modules/statistics/`
- rocm_backend: `DrvGPU/backends/rocm/`
