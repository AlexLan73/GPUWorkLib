# 🚀 ИНСТРУКЦИЯ ДЛЯ СЛЕДУЮЩЕЙ СЕССИИ

> **Дата**: понедельник 2026-03-17
> **Ветка**: `main`
> **Задача**: Ref03 — проверить B+E на GPU, подключить C (Facade rewrite), signal_generators ROCm

---

## ЧТО ЧИТАТЬ ПЕРВЫМ

1. `Doc_Addition/PLAN/Ref03_Unified_Architecture.md` — ГЛАВНЫЙ ДОКУМЕНТ
2. `MemoryBank/tasks/TASK_Ref03_unified_architecture.md` — ВСЕ ТАСКИ
3. `MemoryBank/sessions/2026-03-14.md` — предыдущая сессия

---

## ЧТО УЖЕ НАПИСАНО (commit `bb0098a`)

### ✅ Ref03-A: Foundation (DrvGPU)
- `DrvGPU/services/buffer_set.hpp` — BufferSet<N> template (Layer 4)
- `DrvGPU/interface/i_gpu_operation.hpp` — IGpuOperation interface (Layer 2)
- `DrvGPU/services/gpu_kernel_op.hpp` + `src/gpu_kernel_op.cpp` — GpuKernelOp base (Layer 3)
- `DrvGPU/interface/gpu_context.hpp` + `src/gpu_context.cpp` — GpuContext per-module (Layer 1)

### ✅ Ref03-B: Statistics refactoring
- 6 Op-классов в `modules/statistics/include/operations/`:
  - `mean_reduction_op.hpp` — hierarchical complex mean (BufferSet<1>)
  - `welford_fused_op.hpp` — single-pass Welford, complex input (BufferSet<0>)
  - `welford_float_op.hpp` — Welford на float magnitudes (BufferSet<0>)
  - `median_radix_sort_op.hpp` — rocPRIM sort median (BufferSet<3>)
  - `median_histogram_op.hpp` — histogram median, float input (BufferSet<3>)
  - `median_histogram_complex_op.hpp` — histogram median, complex input (BufferSet<3>)
- `statistics_processor.hpp` + `.cpp` — ПЕРЕПИСАН как thin Facade (1290 → 320 строк)
- **API НЕ изменился** — Python bindings не трогали

### ✅ Ref03-E: FFT refactoring (commit `f0d0a67`)
- 2 Op-класса в `modules/fft_func/include/operations/`:
  - `pad_data_op.hpp` — zero-padding (memset + pad_data kernel)
  - `mag_phase_op.hpp` — complex → magnitude/phase conversion
- `fft_processor_rocm.hpp` + `.cpp` — ПЕРЕПИСАН как thin Facade (1027 → ~550 строк, -46%)
- BufferSet<4> заменяет 4 raw void* pipeline buffers
- hipFFT LRU-2 plan cache сохранён
- **НЕ ТЕСТИРОВАНО НА GPU**

### ✅ Ref03-C: Strategies Pipeline INFRASTRUCTURE (commit `7613def`)
- 4 инфраструктурных файла: PipelineContext, IPipelineStep, Pipeline, PipelineBuilder
- 6 Step классов: GemmStep, WindowFftStep, DebugStatsStep, OneMaxStep, AllMaximaStep, MinMaxStep
- **Facade rewrite НЕ сделан** — делаем в понедельник с живым тестированием
- Порядок: подключить Pipeline к AntennaProcessor_v1 → test_full_pipeline → test_benchmark_streams

### ✅ Signal Generators ROCm port (commit `0e6e395`)
- CwGeneratorROCm, LfmGeneratorROCm, NoiseGeneratorROCm
- HIP kernels (sincos→__sincosf, Philox PRNG, Box-Muller)
- GpuContext для lazy compilation + disk cache
- **НЕ ТЕСТИРОВАНО НА GPU**

### ❌ НЕ написано:
- Ref03-C Facade rewrite (antenna_processor_v1 → Pipeline delegation) — понедельник
- Ref03-D: Filters refactoring (низкий приоритет)

---

## ПЕРВОЕ ДЕЙСТВИЕ НА LINUX (понедельник)

### Шаг 1: Собрать
```bash
cd ~/GPUWorkLib
git pull
mkdir -p build && cd build
cmake .. -DENABLE_ROCM=ON
make -j$(nproc)
```

### Шаг 2: Прогнать тесты statistics + FFT
```bash
./gpu_work_lib  # запустить:
# test_statistics_rocm::run()  → 11/11 passed (T1-T11)
# test_fft_processor_rocm::run()  → все passed
```

### Шаг 3: Проверить baseline timing
- Сравнить с предыдущими benchmark результатами
- Не должно быть regression (Op-классы — zero overhead wrappers)

### Шаг 4: Ref03-C Facade rewrite (30 мин с GPU)
- Инфраструктура + Steps УЖЕ написаны (commit `7613def`)
- Нужно: переписать `antenna_processor_v1.hpp + .cpp` → Pipeline delegation
- Обновить `antenna_processor_test.hpp` → FindStep pattern
- Тесты: test_full_pipeline → test_external_weights → test_benchmark_streams

### Шаг 5: Signal Generators ROCm тесты
- Написать test_cw_rocm.hpp, test_lfm_rocm.hpp, test_noise_rocm.hpp
- Сравнить GPU vs CPU reference (GenerateToGpu vs GenerateToCpu)

---

## ВОЗМОЖНЫЕ ПРОБЛЕМЫ

### Если не компилируется:
1. **Include paths** — Op-классы включают `services/gpu_kernel_op.hpp` и `interface/gpu_context.hpp`. Проверить что CMakeLists.txt добавляет `DrvGPU/` в include dirs.
2. **gpu_context.cpp** и **gpu_kernel_op.cpp** — добавить в CMakeLists.txt (DrvGPU/src/).
3. **statistics_sort_gpu.hpp** include — путь может отличаться (проверить).
4. **FFT**: `fft_processor_types.hpp` include — проверить путь.
5. **FFT**: `hipfft/hipfft.h` — нужен `-lhipfft` при линковке.

### Если тесты падают:
1. Проверить что `kernels::GetStatisticsKernelSource()` возвращает ВСЕ 10 kernels.
2. Проверить что `GpuContext::CompileModule()` передаёт `-DBLOCK_SIZE=256`.
3. Debug: добавить `ConsoleOutput::Print()` в EnsureCompiled().

### Histogram median (commit dc11bc6):
- Ещё НЕ тестировался на GPU — тоже проверить (тесты T8-T11)!

---

## ТАКЖЕ В ОЧЕРЕДИ (не Ref03)
- Histogram median ждёт тестирования на ROCm
- В nvidia ветке stash с .gitignore (git stash pop при возврате)
