# Code Review: ZeroCopy + Capon Module — v6 DEEP REVIEW

## Дата: 2026-03-25 (ревизия v6 — полное ревью с исправлениями)
## Ревьюер: Кодо

---

## Scope ревью

Проверены **ВСЕ** файлы:
- `DrvGPU/backends/rocm/zero_copy_bridge.{hpp,cpp}` — ZeroCopy мост
- `DrvGPU/backends/opencl/opencl_export.hpp` — detection + enums
- `DrvGPU/backends/opencl/gpu_copy_kernel.hpp` — GPU Copy kernel
- `DrvGPU/backends/rocm/hsa_interop.hpp` — HSA Probe
- `modules/capon/` — **ВЕСЬ модуль** (21 файл, ~3800 строк)
- `DrvGPU/tests/test_zero_copy.hpp` — тесты ZeroCopy

---

## Найденные проблемы и исправления

### 🔴 Критические (исправлены)

#### 1. **BUG: cov_op_ не перемещён в move assignment** — ИСПРАВЛЕНО ✅
**Файл**: `modules/capon/src/capon_processor.cpp:84-106`

```cpp
// БЫЛО (строка 91): cov_op_ НЕ перемещался!
backend_     = other.backend_;
ctx_         = std::move(other.ctx_);
// cov_op_ — ПРОПУЩЕН!
inv_op_      = std::move(other.inv_op_);

// СТАЛО: добавлен move cov_op_
cov_op_      = std::move(other.cov_op_);
```

**Риск**: После move assignment `this->cov_op_` оставался в released/default состоянии.
Любой последующий `ComputeRelief()` падал бы при вызове `cov_op_.Execute()`.

#### 2. **MISSING TEST: SVM path с данными заказчика** — ДОБАВЛЕНО ✅
**Файл**: `modules/capon/tests/test_capon_opencl_to_rocm.hpp`

Alex запросил: "данные заказчика → запись на GPU через SVM → копирование на ROCm → расчёт → проверка"

**БЫЛО**: тесты 02-04 используют только `cl.Allocate()` (cl_mem), не SVM.

**ДОБАВЛЕН test_05_svm_customer_data()**:
1. `clSVMAlloc(CL_MEM_SVM_FINE_GRAIN_BUFFER)` — аллокация SVM
2. `std::memcpy()` → CPU→SVM (fine-grain SVM доступен из CPU)
3. `ZeroCopyBridge::ImportFromSVM()` → SVM→HIP (unified VA)
4. `CaponProcessor::ComputeRelief()` через HIP pointers
5. Сравнение с прямым путём (tolerance < 1e-4)

---

### 🟡 Важные замечания (не блокирующие)

#### 3. **hsa_interop.hpp: malloc_usable_size на cl_mem**
**Строка**: 191

`malloc_usable_size(cl_buffer)` — cl_mem может не быть malloc'd pointer.
На ROCm CLR работает (cl_mem = C++ объект на heap), но это **implementation-defined**.
Рекомендация: добавить комментарий о зависимости от ROCm CLR.

#### 4. **hsa_interop.hpp: сканирование внутренней структуры cl_mem**
**Строки**: 186-267

Чтение приватных полей cl_mem (cast к uint8_t* + probe) — **UB по стандарту C++**.
Работает на ROCm 7.2.0 (offset=+664), но может сломаться при обновлении ROCm.
**Задокументировано** в файле — осознанное решение.

#### 5. **gpu_copy_kernel.hpp: перекомпиляция при каждом вызове**
**Строки**: 86-95

`clCreateProgramWithSource` + `clBuildProgram` при каждом вызове (~1мс).
Для больших буферов (4ГБ @ ~8мс копия) — незначительно.
Для частых мелких вызовов — стоит кешировать `cl_program`.

#### 6. **capon_relief kernel: z=0 при acc≤0**
**Файл**: `capon_kernels_rocm.hpp:67`

```hip
z[m] = (acc > 0.0f) ? (1.0f / acc) : 0.0f;
```
Для корректно регуляризованной матрицы acc всегда > 0.
Но float precision может дать acc < 0 при очень малом mu.
Текущая защита (z=0) — приемлема.

---

## Архитектура модуля Capon — оценка

### Ref03 Unified Architecture ✅

| Слой | Класс | Реализация |
|------|-------|------------|
| 1 | GpuContext | `ctx_` — per-module stream, kernels, shared bufs ✅ |
| 2 | IGpuOperation | GpuKernelOp base ✅ |
| 3 | GpuKernelOp | cov_op_, weights_op_, relief_op_, beam_op_ ✅ |
| 4 | Shared Buffers | `shared_buf::kSignal..kOutput` (5 буферов) ✅ |
| 5 | Concrete Ops | 5 операций, каждая в отдельном файле ✅ |
| 6 | Facade | CaponProcessor — тонкий фасад ✅ |

### GPUWorkLib Standards ✅

| Стандарт | Статус |
|----------|--------|
| ConsoleOutput (multi-GPU safe) | ✅ |
| GPUProfiler (benchmark) | ✅ GpuBenchmarkBase |
| Google C++ Style | ✅ CamelCase + 2-space indent |
| ENABLE_ROCM guards | ✅ Все файлы |
| Windows stubs | ✅ В header |
| Column-major (rocBLAS/LAPACK) | ✅ |
| hiprtc lazy compile | ✅ через GpuContext |
| Move semantics | ✅ (после fix) |

---

## ZeroCopy — 4 стратегии

| # | Стратегия | Метод | Копий | 4ГБ | Доп. память |
|---|-----------|-------|-------|-----|-------------|
| A | HSA Probe | `HSA_PROBE` | **0** | **~μs** | **0** |
| B | DMA-BUF | `DMA_BUF` | **0** | ~μs | 0 |
| C | GPU Copy Kernel | `GPU_COPY` | 1 (VRAM→VRAM) | ~8-15мс | +size (VRAM) |
| D | SVM fallback | `SVM` | 1 (CPU) | секунды | +size (RAM) |

**Программное переключение**: `ZeroCopyStrategy` enum (AUTO, FORCE_*)

---

## Тесты модуля Capon — полный список

### test_capon_rocm.hpp (5 тестов)
| # | Тест | Что проверяет |
|---|------|--------------|
| 01 | relief_noise_only | Шум → z[m] > 0, конечный |
| 02 | relief_with_interference | CW помеха → MVDR подавление (z[m_int] < mean/2) |
| 03 | adaptive_beamform_dims | Выход [M×N], конечный |
| 04 | regularization | mu=0 vs mu>0, вырожденная R |
| 05 | gpu_to_gpu | hipMalloc → D2D → void* API |

### test_capon_reference_data.hpp (3 теста)
| # | Тест | Что проверяет |
|---|------|--------------|
| 01 | load_files | Загрузка x/y/signal_matlab |
| 02 | physical_relief | P=85, N=1000, M=1369, real data |
| 03 | cpu_vs_gpu | CPU Cholesky vs GPU, rel_error < 0.5% |

### test_capon_opencl_to_rocm.hpp (5 тестов) — **КЛЮЧЕВОЙ**
| # | Тест | Что проверяет |
|---|------|--------------|
| 01 | detect_interop | HSA/DMA-BUF/SVM capabilities |
| 02 | customer_data_pipeline | **cl_mem → ZeroCopy → Capon** (4 этапа) |
| 03 | zerocopy_matches_direct | Zero Copy == direct path (< 1e-4) |
| 04 | beamform_customer_data | AdaptiveBeamform через ZeroCopy |
| 05 | **svm_customer_data** | **НОВЫЙ: clSVMAlloc → memcpy → ImportFromSVM → Capon** |

---

## Итого изменений в этом ревью

| Файл | Изменение |
|------|-----------|
| `modules/capon/src/capon_processor.cpp` | 🔴 FIX: добавлен `cov_op_ = std::move(other.cov_op_)` |
| `modules/capon/tests/test_capon_opencl_to_rocm.hpp` | ✅ NEW: test_05_svm_customer_data |

## Оценка: 9.5/10

- 🔴 1 критический баг (move assignment) — исправлен
- ✅ SVM тест добавлен
- 🟡 3 замечания (hsa probe UB, kernel cache, malloc_usable_size) — не блокируют

Минус 0.5 за то что `hsa_interop.hpp` зависит от внутренней структуры ROCm CLR
(может сломаться при обновлении ROCm) — но альтернативы нет.

---

*Ревью v6: 2026-03-25*
*Кодо (AI Assistant)*
