# TASK: FFT Processor — OpenCL Refactoring (Оптимизация)

**Статус**: 📋 BACKLOG
**Приоритет**: 🟠 Важно
**Модуль**: `modules/fft_processor`
**Создано**: 2026-03-07
**Оценка прироста**: +20-40% общей пропускной способности pipeline

---

## Контекст

В рамках оптимизации `fft_processor` (2026-03-07) были применены быстрые правки:
- ✅ `native_sqrt`/`native_atan2` в OpenCL ядрах
- ✅ `__atan2f`/`__fsqrt_rn` в ROCm ядрах
- ✅ `reqd_work_group_size(256,1,1)` в OpenCL
- ✅ `-cl-fast-relaxed-math -cl-std=CL2.0` флаги компиляции
- ✅ Динамический `WARP_SIZE` по архитектуре (gfx9* → 64, иначе 32)
- ✅ `-O3 -std=c++17` для hiprtc

Оставшиеся 3 задачи требуют значительных изменений pipeline OpenCL и вынесены сюда.

---

## OPT-3: Interleaved mag/phase буфер (OpenCL)

**Прирост**: ~40% снижение DtoH latency
**Сложность**: Средняя

### Проблема
OpenCL версия использует 2 отдельных буфера (`mag_output` + `phase_output`) и 2 отдельных DtoH-трансфера.
ROCm версия уже использует interleaved `float2_t {mag, phase}` — один буфер, один трансфер.

### Что сделать
1. **Ядро** `fft_processor_kernels.hpp` → `GetMagPhaseKernelSource_opencl()`:
   - Заменить 2 выходных буфера на 1 interleaved `float2 mag_phase`
   - Аналогично ROCm: `mag_phase[gid] = (float2)(mag, phase)`
2. **Host-код** `fft_processor.cpp`:
   - Один `cl::Buffer` вместо двух
   - Один `clEnqueueReadBuffer` вместо двух
   - Распаковка на CPU: `mag[i] = result[i].x; phase[i] = result[i].y`
3. **Python bindings**: Проверить, что формат возврата не изменился

### Файлы
- `modules/fft_processor/include/kernels/fft_processor_kernels.hpp`
- `modules/fft_processor/src/fft_processor.cpp`
- `python/gpu_worklib_bindings.cpp` (проверить)

---

## OPT-5: Замена pre-callback на pad_data kernel (OpenCL)

**Прирост**: +15-25%
**Сложность**: Высокая

### Проблема
OpenCL версия использует clFFT pre-callback для zero-padding (`prepareDataPre`).
Внутри callback — дорогие `div` и `mod` для каждого элемента.
ROCm версия уже использует отдельный `pad_data` kernel с 2D grid (без div/mod).

### Что сделать
1. **Добавить pad_data kernel** в OpenCL (аналог ROCm):
   ```opencl
   __kernel void pad_data(
       __global const float2* input,
       __global float2* output,
       uint n_point, uint nFFT)
   {
       uint beam_id = get_global_id(1);  // 2D grid
       uint pos = get_global_id(0);
       if (pos >= n_point) return;
       output[beam_id * nFFT + pos] = input[beam_id * n_point + pos];
   }
   ```
2. **Host-код**:
   - Выделить промежуточный буфер `padded_buf` (beam_count × nFFT × sizeof(float2))
   - `clEnqueueFillBuffer` для обнуления
   - Запуск `pad_data` с 2D NDRange `{nFFT, beam_count}`
   - Передать `padded_buf` в clFFT (без callback)
   - Удалить `clFFTSetPlanCallback` вызов
3. **Удалить** `GetPreCallbackSource_opencl()` после перехода

### Файлы
- `modules/fft_processor/include/kernels/fft_processor_kernels.hpp`
- `modules/fft_processor/src/fft_processor.cpp`

### Риски
- clFFT API без callback работает иначе — нужно проверить batch mode
- Дополнительный буфер увеличивает потребление VRAM

---

## OPT-6: Event chains вместо clFinish (OpenCL)

**Прирост**: +5-15%
**Сложность**: Средняя

### Проблема
`clFinish()` — полная синхронизация очереди (блокирует CPU, ждёт ВСЕ операции).
Правильный подход — event chains: каждая операция возвращает `cl_event`, следующая ждёт его.

### Что сделать
1. **Аудит** `fft_processor.cpp`: найти все `clFinish()` вызовы
2. **Заменить** на event-based зависимости:
   ```cpp
   cl_event ev_pad, ev_fft, ev_mag;
   clEnqueueNDRangeKernel(..., 0, nullptr, &ev_pad);        // pad_data
   clFFTEnqueueTransform(..., 1, &ev_pad, &ev_fft);          // FFT
   clEnqueueNDRangeKernel(..., 1, &ev_fft, &ev_mag);        // mag_phase
   clEnqueueReadBuffer(..., 1, &ev_mag, nullptr);            // DtoH (blocking)
   clReleaseEvent(ev_pad); clReleaseEvent(ev_fft); clReleaseEvent(ev_mag);
   ```
3. Оставить **один** `clFinish` в конце pipeline (или blocking read)

### Файлы
- `modules/fft_processor/src/fft_processor.cpp`

### Зависимости
- OPT-5 (pad_data kernel) должен быть реализован первым — event chain включает pad_data

---

## Порядок реализации

1. **OPT-5** — pad_data kernel (убирает pre-callback, готовит pipeline для event chains)
2. **OPT-6** — event chains (после OPT-5, т.к. chain включает pad_data event)
3. **OPT-3** — interleaved буфер (независим, можно параллельно с OPT-6)

## Критерии завершения
- [ ] Все 3 оптимизации применены
- [ ] Python тесты проходят (`Python_test/fft_processor/`)
- [ ] C++ тесты проходят (`modules/fft_processor/tests/`)
- [ ] Benchmark до/после (через GPUProfiler)
