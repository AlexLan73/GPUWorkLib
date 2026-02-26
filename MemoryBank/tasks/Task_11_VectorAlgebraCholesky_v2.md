# Task_11_VectorAlgebraCholesky_v2 — Модуль vector_algebra: Cholesky Inverter с нуля (Strategy: Roundtrip + GpuKernel)

> **Памятка для ИИ**: ROCm-only модуль. Код писать весь, **тестировать только под Linux** (Debian с Radeon). Весь ROCm-код под `#if ENABLE_ROCM`. **Task_10 удалён** — делаем с нуля по обновлённому плану.
>
> **Plan-источник**: [`MemoryBank/DiscussionPlan/PLAN_Vector_Algebra_Cholesky.md`](../DiscussionPlan/PLAN_Vector_Algebra_Cholesky.md)
>
> **Предыдущая версия**: Task_10 (✅ COMPLETED → удалён, переделываем архитектуру)
>
> **Статус**: ✅ COMPLETED (2026-02-26)

---

## ⚠️ ПРАВИЛА

- Вывод профилирования — **ТОЛЬКО** через `GPUProfiler`: `PrintReport()`, `ExportMarkdown()`, `ExportJSON()`. **ЗАПРЕЩЕНО** `GetStats()` + цикл + `con.Print`.
- Перед `profiler.Start()` — обязательно `profiler.SetGPUInfo(...)`. См. [`Examples/GPUProfiler_SetGPUInfo.md`](../../Examples/GPUProfiler_SetGPUInfo.md).
- Общий вывод — через `drv_gpu_lib::ConsoleOutput::GetInstance()` (мультиGPU-safe).
- Новые классы — в отдельных файлах (`.hpp` + `.cpp`).
- **10 GPU параллельно**: никаких `static`, всё через объект.
- **CholeskyResult** — единый тип (НЕ шаблонный), базовый `void*`, методы `AsVector()` + `AsHipPtr()`.
- **Два режима симметризации**: `SymmetrizeMode::Roundtrip` и `SymmetrizeMode::GpuKernel` — реализуем оба.
- **hiprtc kernels**: строка в `.hpp`, компиляция через `hiprtcCompileProgram`, запуск через `hipModuleLaunchKernel`.
- **rocblas_handle в заголовке**: `void* handle_` + cast в `.cpp`.

---

## 1. Цель

Инверсия эрмитовых положительно определённых матриц методом Холецкого. ВСЁ на GPU (ROCm: POTRF + POTRI + симметризация).

**Два подхода к симметризации** — реализуем оба, бенчмарк, выбираем лучший:
- **Roundtrip**: Download → CPU sym → Upload (простой, без kernel)
- **GpuKernel**: HIP kernel in-place на GPU (всё на GPU, без round-trip)

**Входные форматы**: `vector<>` (CPU), `void*` (HIP), `cl_mem` (ZeroCopy)
**Выходной формат**: `CholeskyResult` (базовый `void*`, методы `AsVector()` + `AsHipPtr()`)

---

## 2. Зависимости

- **Task_00_DrvGPU** (ROCmBackend, IBackend, GetNativeQueue)
- **Task_08_ZeroCopy** (ZeroCopyBridge, HybridBackend — для cl_mem пути)
- `DrvGPU/interface/input_data.hpp` + `input_data_traits.hpp`
- Linux: rocblas, rocsolver, hip, hiprtc (AMD ROCm stack)

---

## 3. Структура файлов (создать)

```
modules/vector_algebra/
├── CMakeLists.txt
├── include/
│   ├── cholesky_inverter_rocm.hpp         # Публичный API + enum SymmetrizeMode
│   ├── vector_algebra_types.hpp           # CholeskyResult (единый тип)
│   └── kernels/
│       └── symmetrize_kernel_sources_rocm.hpp  # HIP kernel как строка (hiprtc)
├── src/
│   ├── cholesky_inverter_rocm.cpp         # Core: POTRF+POTRI + Roundtrip + диспетчер
│   └── symmetrize_gpu_rocm.cpp            # CompileKernels + SymmetrizeUpperToFullGPU
├── tests/
│   ├── test_cholesky_inverter_rocm.hpp    # Функциональные тесты (оба режима)
│   ├── test_cross_backend_conversion.hpp  # Тест конвертации 85×85 (все пути данных)
│   ├── test_benchmark_symmetrize.hpp      # Бенчмарк: Roundtrip vs GpuKernel
│   ├── all_test.hpp
│   └── README.md
└── CMakeLists.txt

python/py_vector_algebra_rocm.hpp          # pybind11 биндинги
Doc/Python/vector_algebra_api.md           # Документация Python API
Python_test/vector_algebra/test_cholesky_inverter_rocm.py
```

---

## 4. Референсный код

| Элемент | Файл |
|---------|------|
| POTRF+POTRI | LCH-Farrow01/Matrix/matrix_invert.cpp:349-399 |
| Batched | LCH-Farrow01/Matrix/matrix_invert_advanced.cpp:470-586 |
| HIP kernel pattern | modules/fft_maxima/include/kernels/all_maxima_kernel_sources_rocm.hpp |
| hiprtc compile | modules/fft_maxima/src/all_maxima_pipeline_rocm.cpp |
| Другой ROCm контекст | DrvGPU/tests/test_hybrid_backend.hpp (test_rocm_allocate) |
| ZeroCopy cl_mem→HIP | DrvGPU/tests/test_zero_copy.hpp (test_data_integrity) |
| HybridBackend | DrvGPU/backends/hybrid/hybrid_backend.hpp |
| GPUReportInfo | modules/fft_maxima/tests/test_batch_all_maxima.hpp:282-292 |
| Record ROCm | DrvGPU/services/gpu_profiler.hpp:131-140 |

---

## 5. Задачи

---

### ГРУППА 1: Инфраструктура

> Фундамент: CMake, типы, traits. Делается первым, тестировать не нужно (только компиляция).

---

#### 5.1 CMake: зависимости + модуль

**Файл 1**: `cmake/dependencies.cmake`

Добавить под `if(ENABLE_ROCM AND IS_LINUX)`:
```cmake
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)
```

**Файл 2**: `modules/vector_algebra/CMakeLists.txt`

```cmake
if(NOT ROCM_ENABLED)
    return()
endif()

add_library(vector_algebra STATIC
    src/cholesky_inverter_rocm.cpp
    src/symmetrize_gpu_rocm.cpp
)

target_include_directories(vector_algebra PUBLIC include/)
target_link_libraries(vector_algebra PRIVATE
    roc::rocblas
    roc::rocsolver
    hip::host
    hiprtc::hiprtc
    DrvGPU
)
target_compile_definitions(vector_algebra PUBLIC ENABLE_ROCM=1)
```

**Файл 3**: `CMakeLists.txt` (корневой) — добавить `add_subdirectory(modules/vector_algebra)`

**Чек-лист 5.1**:
- [ ] `cmake/dependencies.cmake` — `find_package(rocblas)` + `find_package(rocsolver)` под `ENABLE_ROCM`
- [ ] `modules/vector_algebra/CMakeLists.txt` — создан, два .cpp, линковка rocblas+rocsolver+hip+hiprtc+DrvGPU
- [ ] `CMakeLists.txt` (корневой) — `add_subdirectory(modules/vector_algebra)` под `if(ROCM_ENABLED)`
- [ ] Компиляция проходит (пустые .cpp файлы-заглушки)

---

#### 5.2 Типы: CholeskyResult + SymmetrizeMode

**Файл**: `modules/vector_algebra/include/vector_algebra_types.hpp`

```cpp
#pragma once
#if ENABLE_ROCM

#include <complex>
#include <vector>
#include "DrvGPU/interface/i_backend.hpp"

namespace vector_algebra {

enum class SymmetrizeMode { Roundtrip, GpuKernel };

struct CholeskyResult {
    void* d_data = nullptr;              // HIP device ptr (базовый формат)
    drv_gpu_lib::IBackend* backend = nullptr;  // для Memcpy/Free
    int matrix_size = 0;                 // n (одна сторона матрицы)
    int batch_count = 0;                 // количество матриц

    // --- Доступ к данным ---

    // Download GPU → CPU vector (через SVM)
    std::vector<std::complex<float>> AsVector() const;

    // Вернуть HIP device ptr (caller не владеет — НЕ вызывать Free!)
    void* AsHipPtr() const { return d_data; }

    // --- Удобные методы ---

    // Одна матрица как 2D vector [n][n]
    std::vector<std::vector<std::complex<float>>> matrix() const;

    // Batch как 3D vector [batch][n][n]
    std::vector<std::vector<std::vector<std::complex<float>>>> matrices() const;

    // --- Владение памятью ---
    ~CholeskyResult();
    CholeskyResult() = default;
    CholeskyResult(CholeskyResult&& other) noexcept;
    CholeskyResult& operator=(CholeskyResult&& other) noexcept;
    CholeskyResult(const CholeskyResult&) = delete;
    CholeskyResult& operator=(const CholeskyResult&) = delete;
};

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
```

**Чек-лист 5.2**:
- [ ] `vector_algebra_types.hpp` создан
- [ ] `SymmetrizeMode` enum: `Roundtrip`, `GpuKernel`
- [ ] `CholeskyResult` — НЕ шаблонный, базовый `void*`
- [ ] `AsVector()` объявлен (реализация в .cpp)
- [ ] `AsHipPtr()` inline
- [ ] `matrix()` / `matrices()` объявлены
- [ ] Move semantics: move ctor + move assign, copy = delete
- [ ] Деструктор: `backend->Free(d_data)` если `d_data != nullptr`
- [ ] Всё под `#if ENABLE_ROCM`

---

#### 5.3 input_data_traits: is_cl_mem

**Файл**: `DrvGPU/interface/input_data_traits.hpp`

Проверить: если `is_cl_mem_v` уже есть (от Task_10) — пропустить. Если нет — добавить:

```cpp
#ifdef CL_VERSION_1_0
template<typename T>
struct is_cl_mem : std::false_type {};

template<>
struct is_cl_mem<cl_mem> : std::true_type {};

template<typename T>
inline constexpr bool is_cl_mem_v = is_cl_mem<T>::value;
#endif
```

**Чек-лист 5.3**:
- [ ] `is_cl_mem_v<T>` есть в `input_data_traits.hpp` (добавить если отсутствует)
- [ ] Под `#ifdef CL_VERSION_1_0` — не ломает сборку без OpenCL

---

### ГРУППА 2: Ядро (Core) — POTRF + POTRI + Roundtrip

> Основная логика: конструктор, POTRF+POTRI, Roundtrip-симметризация, все перегрузки Invert/InvertBatch.
> **После этой группы**: можно тестировать с `SymmetrizeMode::Roundtrip`.

---

#### 5.4 API класс — cholesky_inverter_rocm.hpp

**Файл**: `modules/vector_algebra/include/cholesky_inverter_rocm.hpp`

Полный API класс по разделу 4.1 плана. Ключевые моменты:
- `void* handle_` (не `rocblas_handle` в заголовке — opaque)
- Два группы private-методов: Core (POTRF/POTRI) + Symmetrize (Roundtrip + GpuKernel)
- `hipModule_t`, `hipFunction_t` для kernel — в private

**Чек-лист 5.4**:
- [ ] `cholesky_inverter_rocm.hpp` создан
- [ ] Конструктор: `(IBackend* backend, SymmetrizeMode mode = SymmetrizeMode::GpuKernel)`
- [ ] `SetSymmetrizeMode()` / `GetSymmetrizeMode()`
- [ ] 3× `Invert`: vector, void*, cl_mem
- [ ] 3× `InvertBatch`: vector, void*, cl_mem
- [ ] Private: `CorePotrf`, `CorePotri`, `CorePotrfBatched`, `CorePotriBatched`
- [ ] Private: `SymmetrizeRoundtrip`, `SymmetrizeRoundtripBatched`
- [ ] Private: `CompileKernels`, `SymmetrizeGpuKernel`, `SymmetrizeGpuKernelBatched`
- [ ] Private: `hipModule_t sym_module_`, `hipFunction_t sym_kernel_`, `bool kernels_compiled_`
- [ ] `void* handle_` (не `rocblas_handle` в заголовке!)
- [ ] Весь файл под `#if ENABLE_ROCM`

---

#### 5.5 Реализация Core — cholesky_inverter_rocm.cpp

**Файл**: `modules/vector_algebra/src/cholesky_inverter_rocm.cpp`

Разбита на подзадачи:

##### 5.5.1 Конструктор + Деструктор

```cpp
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

CholeskyInverterROCm::CholeskyInverterROCm(IBackend* backend, SymmetrizeMode mode)
    : backend_(backend), handle_(nullptr), mode_(mode) {
    rocblas_handle h;
    rocblas_create_handle(&h);
    handle_ = static_cast<void*>(h);
    hipStream_t stream = static_cast<hipStream_t>(backend_->GetNativeQueue());
    rocblas_set_stream(h, stream);
}

~CholeskyInverterROCm() {
    if (handle_) rocblas_destroy_handle(static_cast<rocblas_handle>(handle_));
    if (sym_module_) hipModuleUnload(sym_module_);
}
```

- [ ] `rocblas_create_handle` + `rocblas_set_stream`
- [ ] Деструктор: `rocblas_destroy_handle` + `hipModuleUnload`
- [ ] `SetSymmetrizeMode()` / `GetSymmetrizeMode()` — тривиальные

##### 5.5.2 ResolveMatrixSize

```cpp
int ResolveMatrixSize(uint32_t n_point, int n_hint) const;
```

- [ ] Если `n_hint > 0` → вернуть `n_hint`
- [ ] Иначе `sqrt(n_point)` с assert что `n*n == n_point`

##### 5.5.3 CorePotrf + CorePotri (single)

```cpp
void CorePotrf(void* d_matrix, int n, hipStream_t stream);
void CorePotri(void* d_matrix, int n, hipStream_t stream);
```

По референсу: `LCH-Farrow01/Matrix/matrix_invert.cpp:349-399`

- [ ] `rocsolver_cpotrf(handle, rocblas_fill_upper, n, ptr, n, dev_info)`
- [ ] `rocsolver_cpotri(handle, rocblas_fill_upper, n, ptr, n, dev_info)`
- [ ] Проверка `dev_info != 0` → throw/log
- [ ] `dev_info` — аллоцировать на GPU, читать через `hipMemcpy`

##### 5.5.4 CorePotrfBatched + CorePotriBatched

```cpp
void CorePotrfBatched(void** d_matrices, int n, int batch, hipStream_t stream);
void CorePotriBatched(void** d_matrices, int n, int batch, hipStream_t stream);
```

По референсу: `LCH-Farrow01/Matrix/matrix_invert_advanced.cpp:470-586`

- [ ] `rocsolver_cpotrf_batched` / `rocsolver_cpotri_batched`
- [ ] Массив указателей `d_matrices` — на GPU
- [ ] Массив `dev_info[batch]` — на GPU, проверить каждый

##### 5.5.5 SymmetrizeRoundtrip (single + batched)

```cpp
void SymmetrizeRoundtrip(void* d_matrix, int n, hipStream_t stream);
void SymmetrizeRoundtripBatched(void** d_matrices, int n, int batch, hipStream_t stream);
```

Паттерн: download → CPU `SymmetrizeUpperToFull()` → upload.

- [ ] `hipMemcpy D2H` → CPU буфер
- [ ] Цикл: `for(row) for(col > row) A[col][row] = conj(A[row][col])`
- [ ] `hipMemcpy H2D` → обратно на GPU
- [ ] Batched: цикл по матрицам

##### 5.5.6 Invert(vector) — CPU путь

```
vector → MemcpyH2D → Allocate copy → CorePotrf → CorePotri → Symmetrize → CholeskyResult
```

- [ ] `ResolveMatrixSize`
- [ ] `backend->Allocate(bytes)`
- [ ] `backend->MemcpyHostToDevice(gpu_buf, input.data.data(), bytes)`
- [ ] `MemcpyDeviceToDevice` (копия, не in-place!)
- [ ] `CorePotrf` + `CorePotri`
- [ ] Диспетчер: `if (mode_ == Roundtrip) SymmetrizeRoundtrip else SymmetrizeGpuKernel`
- [ ] Return `CholeskyResult{d_output, backend_, n, 1}`
- [ ] Free промежуточный upload буфер

##### 5.5.7 Invert(void*) — GPU HIP путь

```
void* → Allocate copy → MemcpyD2D → CorePotrf → CorePotri → Symmetrize → CholeskyResult
```

- [ ] Новый буфер (не in-place!)
- [ ] `MemcpyDeviceToDevice`
- [ ] `CorePotrf` + `CorePotri` + Symmetrize
- [ ] Return `CholeskyResult{d_output, backend_, n, 1}`

##### 5.5.8 Invert(cl_mem) — ZeroCopy путь

```
cl_mem → HybridBackend::SyncBeforeZeroCopy → CreateZeroCopyBridge → hip_ptr →
→ Allocate copy → MemcpyD2D(copy, hip_ptr) → CorePotrf → CorePotri → Symmetrize → CholeskyResult
```

По референсу: `DrvGPU/tests/test_zero_copy.hpp (test_data_integrity)`

- [ ] `HybridBackend::SyncBeforeZeroCopy()`
- [ ] `CreateZeroCopyBridge(cl_mem, size)` → `bridge->GetHipPtr()`
- [ ] Allocate d_output + `MemcpyD2D(d_output, hip_ptr, bytes)`
- [ ] `CorePotrf` + `CorePotri` + Symmetrize
- [ ] Return `CholeskyResult`
- [ ] SKIP если ZeroCopy не поддерживается

##### 5.5.9 InvertBatch(vector) — CPU batched

```
vector (batch × n²) → Upload → подготовить d_ptrs[batch] → CorePotrfBatched → CorePotriBatched →
→ SymmetrizeBatched → CholeskyResult
```

- [ ] Allocate contiguous GPU буфер `batch * n * n * sizeof(cf)`
- [ ] Upload
- [ ] Создать массив указателей `d_ptrs[k] = d_output + k * n * n * sizeof(cf)`
- [ ] `d_ptrs` на GPU (`hipMalloc` + `hipMemcpy`)
- [ ] `CorePotrfBatched` + `CorePotriBatched`
- [ ] Symmetrize (Roundtrip или GpuKernel) для каждой матрицы / batched kernel
- [ ] Return `CholeskyResult{d_output, backend_, n, batch}`

##### 5.5.10 InvertBatch(void*) — GPU batched

- [ ] Аналогично 5.5.9, но без upload (данные уже на GPU)
- [ ] `MemcpyDeviceToDevice` (копия)

##### 5.5.11 InvertBatch(cl_mem) — ZeroCopy batched

- [ ] Один ZeroCopyBridge на весь contiguous буфер (макс ~57 MB)
- [ ] Далее как 5.5.10

##### 5.5.12 CholeskyResult — реализация методов

```cpp
// В отдельном .cpp или в cholesky_inverter_rocm.cpp

CholeskyResult::~CholeskyResult() {
    if (d_data && backend) backend->Free(d_data);
}

std::vector<std::complex<float>> CholeskyResult::AsVector() const {
    size_t count = matrix_size * matrix_size * batch_count;
    std::vector<std::complex<float>> result(count);
    backend->MemcpyDeviceToHost(result.data(), d_data, count * sizeof(std::complex<float>));
    return result;
}

// matrix() — из AsVector() разрезать на [n][n]
// matrices() — из AsVector() разрезать на [batch][n][n]

// Move constructor/assignment — перенос d_data, обнуление источника
```

- [ ] Деструктор: `backend->Free(d_data)` если `d_data != nullptr`
- [ ] `AsVector()`: `MemcpyDeviceToHost` → vector
- [ ] `matrix()`: `AsVector()` → reshape [n][n]
- [ ] `matrices()`: `AsVector()` → reshape [batch][n][n]
- [ ] Move ctor: перенос всех полей, обнуление `other.d_data`
- [ ] Move assign: free старый `d_data`, перенос, обнуление

**Чек-лист Группы 2**:
- [ ] 5.4: API заголовок
- [ ] 5.5.1: Конструктор/деструктор
- [ ] 5.5.2: ResolveMatrixSize
- [ ] 5.5.3: CorePotrf + CorePotri
- [ ] 5.5.4: CorePotrfBatched + CorePotriBatched
- [ ] 5.5.5: SymmetrizeRoundtrip (single + batched)
- [ ] 5.5.6: Invert(vector)
- [ ] 5.5.7: Invert(void*)
- [ ] 5.5.8: Invert(cl_mem)
- [ ] 5.5.9: InvertBatch(vector)
- [ ] 5.5.10: InvertBatch(void*)
- [ ] 5.5.11: InvertBatch(cl_mem)
- [ ] 5.5.12: CholeskyResult methods

---

### ТЕСТ ГРУППЫ 2: Проверка Core + Roundtrip

> После Группы 2 — запустить тесты **ТОЛЬКО в режиме Roundtrip**.
> Убедиться что POTRF+POTRI работают, симметризация корректна.

#### 5.6 Тесты Core (Roundtrip mode)

**Файл**: `modules/vector_algebra/tests/test_cholesky_inverter_rocm.hpp`

Каждый тест принимает `SymmetrizeMode` — на этом этапе вызывать только с `Roundtrip`.

| # | Подзадача | Тест | n | batch | Допуск |
|---|-----------|------|---|-------|--------|
| 5.6.1 | `TestCpuIdentity` | vector вход, I(5×5) | 5 | 1 | Frobenius < 1e-5 |
| 5.6.2 | `TestCpu341` | vector вход, HPD(341) | 341 | 1 | Frobenius < 1e-2 |
| 5.6.3 | `TestGpuVoidPtr341` | void* вход | 341 | 1 | Frobenius < 1e-2 |
| 5.6.4 | `TestZeroCopyClMem` | cl_mem вход (SKIP если нет Hybrid) | 341 | 1 | Frobenius < 1e-2 |
| 5.6.5 | `TestBatchCpu_4x64` | vector batched | 64 | 4 | Frobenius < 1e-3 каждая |
| 5.6.6 | `TestBatchGpu_4x64` | void* batched | 64 | 4 | Frobenius < 1e-3 каждая |
| 5.6.7 | `TestBatchSizes` | void*, разные batch | 64 | 1,4,8,16 | Все проходят |
| 5.6.8 | `TestMatrixSizes` | void*, разные n | 32,64,128,256 | 4 | Frobenius < 1e-2 |
| 5.6.9 | `TestResultAccess` | `.matrix()` shape, `.matrices()` shape | 5+5 | 1+3 | Размерности верны |
| 5.6.10 | `TestResolveMatrixSize` | sqrt(n_point) логика | — | — | Без GPU |

**Вспомогательные функции** (в отдельном helper или в начале теста):

```cpp
// Создать HPD матрицу: A = B*B^H + n*I
std::vector<std::complex<float>> MakePositiveDefiniteHermitian(int n, int seed = 42);

// ||A * A⁻¹ - I||_F
float FrobeniusError(const std::vector<std::complex<float>>& A,
                     const std::vector<std::complex<float>>& A_inv, int n);
```

- [ ] 5.6.1: Identity 5×5 — `I⁻¹ = I`
- [ ] 5.6.2: HPD 341×341 — `||A * A⁻¹ - I||_F < 1e-2`
- [ ] 5.6.3: void* 341 — тот же тест через GPU указатель
- [ ] 5.6.4: cl_mem 341 — через ZeroCopy (SKIP если нет)
- [ ] 5.6.5: Batch CPU 4×64
- [ ] 5.6.6: Batch GPU 4×64
- [ ] 5.6.7: Batch sizes 1,4,8,16
- [ ] 5.6.8: Matrix sizes 32,64,128,256
- [ ] 5.6.9: Result access — `.matrix()` [n][n], `.matrices()` [batch][n][n]
- [ ] 5.6.10: ResolveMatrixSize — без GPU
- [ ] `MakePositiveDefiniteHermitian()` реализована
- [ ] `FrobeniusError()` реализована
- [ ] **ВСЕ тесты проходят в режиме Roundtrip**

---

### ГРУППА 3: GPU Kernel (Symmetrize)

> HIP kernel для симметризации in-place на GPU. Альтернатива Roundtrip.

---

#### 5.7 Kernel source — symmetrize_kernel_sources_rocm.hpp

**Файл**: `modules/vector_algebra/include/kernels/symmetrize_kernel_sources_rocm.hpp`

По паттерну `modules/fft_maxima/include/kernels/all_maxima_kernel_sources_rocm.hpp`:

```cpp
#pragma once
#if ENABLE_ROCM
namespace vector_algebra { namespace kernels {

inline const char* GetSymmetrizeKernelSource() {
    return R"HIP(
extern "C" __global__ void symmetrize_upper_to_full(
    float2* __restrict__ data, unsigned int n)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    if (col > row) {
        float2 v = data[row * n + col];
        data[col * n + row] = {v.x, -v.y};  // conjugate
    }
}
)HIP";
}

}}  // namespace vector_algebra::kernels
#endif
```

**Чек-лист 5.7**:
- [ ] Файл создан в `include/kernels/`
- [ ] Kernel как inline строка в `R"HIP(...)HIP"`
- [ ] `symmetrize_upper_to_full`: копирует верхний треугольник в нижний с conjugate
- [ ] `float2` = `complex<float>`, conjugate = `{v.x, -v.y}`
- [ ] Под `#if ENABLE_ROCM`

---

#### 5.8 Реализация GPU symmetrize — symmetrize_gpu_rocm.cpp

**Файл**: `modules/vector_algebra/src/symmetrize_gpu_rocm.cpp`

По паттерну `modules/fft_maxima/src/all_maxima_pipeline_rocm.cpp`:

##### 5.8.1 CompileKernels

```cpp
void CholeskyInverterROCm::CompileKernels() {
    if (kernels_compiled_) return;
    const char* src = kernels::GetSymmetrizeKernelSource();
    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, src, "symmetrize.hip", 0, nullptr, nullptr);
    hiprtcResult res = hiprtcCompileProgram(prog, 0, nullptr);
    // ... error handling, get code, load module, get function ...
    kernels_compiled_ = true;
}
```

- [ ] `hiprtcCreateProgram` + `hiprtcCompileProgram`
- [ ] Error check: `res != HIPRTC_SUCCESS` → log + throw
- [ ] `hiprtcGetCodeSize` + `hiprtcGetCode`
- [ ] `hipModuleLoadData(&sym_module_, code.data())`
- [ ] `hipModuleGetFunction(&sym_kernel_, sym_module_, "symmetrize_upper_to_full")`
- [ ] Guard: `if (kernels_compiled_) return`

##### 5.8.2 SymmetrizeGpuKernel (single)

```cpp
void CholeskyInverterROCm::SymmetrizeGpuKernel(void* ptr, int n, hipStream_t stream) {
    CompileKernels();
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    void* args[] = {&ptr, &n};
    hipModuleLaunchKernel(sym_kernel_,
        grid.x, grid.y, 1, block.x, block.y, 1,
        0, stream, args, nullptr);
}
```

- [ ] `CompileKernels()` перед launch
- [ ] `dim3 block(16, 16)`, `dim3 grid` — ceil division
- [ ] `hipModuleLaunchKernel` (НЕ `hipLaunchKernelGGL`!)
- [ ] `void* args[]` — указатели на аргументы

##### 5.8.3 SymmetrizeGpuKernelBatched

- [ ] Цикл по матрицам: `for(k=0; k<batch; k++) SymmetrizeGpuKernel(ptr[k], n, stream)`
- [ ] (Оптимизация 3D grid — в фазе оптимизации, пока цикл)

**Чек-лист Группы 3**:
- [ ] 5.7: Kernel source
- [ ] 5.8.1: CompileKernels
- [ ] 5.8.2: SymmetrizeGpuKernel (single)
- [ ] 5.8.3: SymmetrizeGpuKernelBatched

---

### ТЕСТ ГРУППЫ 3: Проверка GpuKernel mode

> Запустить ВСЕ тесты из Группы 2 (5.6.1–5.6.8) но с `SymmetrizeMode::GpuKernel`.
> Убедиться что результаты идентичны Roundtrip.

#### 5.9 Тесты GpuKernel mode

- [ ] Все тесты 5.6.1–5.6.8 проходят с `SymmetrizeMode::GpuKernel`
- [ ] Результаты GpuKernel == Roundtrip (Frobenius разница < 1e-6)
- [ ] Kernel компилируется без ошибок (hiprtc)

---

### ГРУППА 4: Cross-Backend тесты (85×85)

> Одна матрица, все пути ввода-вывода, проверка конвертации.

---

#### 5.10 test_cross_backend_conversion.hpp

**Файл**: `modules/vector_algebra/tests/test_cross_backend_conversion.hpp`

##### 5.10.1 TestConvert_VectorInput — ЭТАЛОН

- [ ] HPD матрица 85×85
- [ ] `Invert(InputData<vector>)` → `result.AsVector()` = **reference**
- [ ] Запуск для обоих SymmetrizeMode

##### 5.10.2 TestConvert_HipInput — другой ROCm контекст

По паттерну `DrvGPU/tests/test_hybrid_backend.hpp (test_rocm_allocate)`:

```cpp
// Создать ДРУГОЙ ROCm контекст на том же GPU
drv_gpu_lib::ROCmBackend other_rocm;
other_rocm.Initialize(0);

void* d_foreign = other_rocm.Allocate(bytes);
other_rocm.MemcpyHostToDevice(d_foreign, matrix.data(), bytes);

// Передать в НАШ инвертер
auto result = inverter.Invert(InputData<void*>{d_foreign, n*n, 1});
auto vec = result.AsVector();
// compare vec vs reference

other_rocm.Free(d_foreign);
other_rocm.Cleanup();
```

- [ ] Создать второй `ROCmBackend::Initialize(0)` — тот же GPU
- [ ] Записать матрицу через другой контекст
- [ ] Передать `void*` в наш инвертер
- [ ] `AsVector()` → сравнить с эталоном (Frobenius < 1e-3)
- [ ] Cleanup другого контекста

##### 5.10.3 TestConvert_ClMemInput — OpenCL → ZeroCopy

По паттерну `DrvGPU/tests/test_zero_copy.hpp (test_data_integrity)`:

- [ ] Создать `OpenCLBackend::Initialize(0)` или `HybridBackend`
- [ ] Записать матрицу через OpenCL
- [ ] `Synchronize()` перед ZeroCopy
- [ ] `Invert(InputData<cl_mem>)` → `AsVector()` → сравнить с эталоном
- [ ] SKIP если ZeroCopy не поддерживается

##### 5.10.4 TestConvert_OutputFormats — 2 конвертации

```cpp
auto result = inverter.Invert(InputData<vector<cf>>{matrix, n*n, 1});

// Путь 1: AsVector()
auto vec1 = result.AsVector();

// Путь 2: AsHipPtr() → ручной hipMemcpy D2H
void* hip_ptr = result.AsHipPtr();
std::vector<cf> vec2(n * n);
hipMemcpy(vec2.data(), hip_ptr, bytes, hipMemcpyDeviceToHost);

// vec1 == vec2 (Frobenius < 1e-7)
```

- [ ] `AsVector()` и `AsHipPtr()+hipMemcpy` дают одинаковый результат
- [ ] Запуск для обоих SymmetrizeMode

**Чек-лист Группы 4**:
- [ ] 5.10.1: Vector input — reference
- [ ] 5.10.2: HIP input от другого ROCm контекста
- [ ] 5.10.3: cl_mem input через ZeroCopy
- [ ] 5.10.4: Output formats — AsVector == AsHipPtr
- [ ] Все × 2 режима = 8 запусков

---

### ГРУППА 5: Python биндинги + тесты

---

#### 5.11 Python биндинги

**Файл**: `python/py_vector_algebra_rocm.hpp`

По паттерну существующих ROCm биндингов (каждый модуль — отдельный `.hpp`):

```cpp
#pragma once
#if ENABLE_ROCM
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "modules/vector_algebra/include/cholesky_inverter_rocm.hpp"

namespace py = pybind11;

inline void register_vector_algebra(py::module_& m) {
    // SymmetrizeMode enum
    py::enum_<vector_algebra::SymmetrizeMode>(m, "SymmetrizeMode")
        .value("Roundtrip", vector_algebra::SymmetrizeMode::Roundtrip)
        .value("GpuKernel", vector_algebra::SymmetrizeMode::GpuKernel);

    // CholeskyInverterROCm class
    py::class_<vector_algebra::CholeskyInverterROCm>(m, "CholeskyInverterROCm")
        .def(py::init<drv_gpu_lib::IBackend*, vector_algebra::SymmetrizeMode>(),
             py::arg("backend"),
             py::arg("mode") = vector_algebra::SymmetrizeMode::GpuKernel)
        .def("set_symmetrize_mode", &vector_algebra::CholeskyInverterROCm::SetSymmetrizeMode)
        .def("get_symmetrize_mode", &vector_algebra::CholeskyInverterROCm::GetSymmetrizeMode)

        // invert_cpu: numpy flat → result → numpy (n,n)
        .def("invert_cpu", [](auto& self, py::array_t<std::complex<float>> flat, int n) {
            // numpy → InputData<vector> → Invert → AsVector → numpy
            // ...
        })

        // invert_batch_cpu: numpy flat → result → numpy (batch,n,n)
        .def("invert_batch_cpu", [](auto& self, py::array_t<std::complex<float>> flat, int n, int batch) {
            // ...
        });
}
#endif
```

##### 5.11.1 Подзадачи:
- [ ] `py_vector_algebra_rocm.hpp` создан
- [ ] `SymmetrizeMode` enum зарегистрирован в Python
- [ ] `CholeskyInverterROCm` — конструктор с `backend` + `mode`
- [ ] `set_symmetrize_mode()` / `get_symmetrize_mode()`
- [ ] `invert_cpu(flat_array, n)` → numpy (n,n)
- [ ] `invert_batch_cpu(flat_array, n, batch)` → numpy (batch,n,n)
- [ ] Регистрация `register_vector_algebra(m)` в `gpu_worklib_bindings.cpp` под `#if ENABLE_ROCM`
- [ ] Линковка в `python/CMakeLists.txt`

---

#### 5.12 Python тесты

**Файл**: `Python_test/vector_algebra/test_cholesky_inverter_rocm.py`

| # | Подзадача | Тест | n | batch | Проверка |
|---|-----------|------|---|-------|----------|
| 5.12.1 | `test_invert_5x5` | маленькая | 5 | 1 | np.linalg.inv vs GPU, Frobenius < 1e-5 |
| 5.12.2 | `test_invert_341x341` | рабочий размер | 341 | 1 | Frobenius < 1e-2 |
| 5.12.3 | `test_batch_4x64` | batched | 64 | 4 | Frobenius < 1e-3 |
| 5.12.4 | `test_batch_sizes` | разные batch | 64 | 1,4,8 | Все проходят |
| 5.12.5 | `test_modes_roundtrip_vs_kernel` | оба режима | 64 | 4 | Одинаковый результат |
| 5.12.6 | `test_cross_backend_vector_vs_clmem` | кросс-бэкенд | 85 | 1 | vector vs cl_mem (HybridGPUContext) |

- [ ] 5.12.1: 5×5, ошибка < 1e-5
- [ ] 5.12.2: 341×341, ошибка < 1e-2
- [ ] 5.12.3: Batch 4×64×64, ошибка < 1e-3
- [ ] 5.12.4: Batch sizes 1,4,8
- [ ] 5.12.5: Roundtrip == GpuKernel
- [ ] 5.12.6: vector путь == cl_mem путь (через HybridGPUContext)
- [ ] `rocm_context` fixture — `pytest.skip()` если ROCm недоступен
- [ ] `make_positive_definite(n, seed)` helper

---

### ГРУППА 6: Benchmark — Roundtrip vs GpuKernel

---

#### 5.13 test_benchmark_symmetrize.hpp

**Файл**: `modules/vector_algebra/tests/test_benchmark_symmetrize.hpp`

| # | Подзадача | n | batch | Что меряем |
|---|-----------|---|-------|-----------|
| 5.13.1 | `BenchmarkSingle341` | 341 | 1 | Roundtrip vs GpuKernel (hipEvent) |
| 5.13.2 | `BenchmarkBatch_16x64` | 64 | 16 | Roundtrip vs GpuKernel |
| 5.13.3 | `BenchmarkBatch_4x256` | 256 | 4 | Roundtrip vs GpuKernel |

- [ ] 5.13.1: Single 341 — время обоих режимов
- [ ] 5.13.2: Batch 16×64 — время обоих режимов
- [ ] 5.13.3: Batch 4×256 — время обоих режимов
- [ ] hipEvent timing (`hipEventCreate`, `hipEventRecord`, `hipEventElapsedTime`)
- [ ] Вывод таблицы через GPUProfiler → `ExportMarkdown`
- [ ] `SetGPUInfo()` ПЕРЕД `Start()` — обязательно!

---

### ГРУППА 7: Прогнать ВСЕ тесты

---

#### 5.14 all_test.hpp + интеграция в main

**Файл**: `modules/vector_algebra/tests/all_test.hpp`

```cpp
#pragma once
#if ENABLE_ROCM

#include "test_cholesky_inverter_rocm.hpp"
#include "test_cross_backend_conversion.hpp"
#include "test_benchmark_symmetrize.hpp"

inline void RunVectorAlgebraTests(drv_gpu_lib::IBackend* backend) {
    using namespace vector_algebra::tests;

    // --- Утилиты (без GPU) ---
    TestResolveMatrixSize();

    // --- Core: Roundtrip mode ---
    TestCpuIdentity(backend, SymmetrizeMode::Roundtrip);
    TestCpu341(backend, SymmetrizeMode::Roundtrip);
    TestGpuVoidPtr341(backend, SymmetrizeMode::Roundtrip);
    TestBatchCpu_4x64(backend, SymmetrizeMode::Roundtrip);
    TestBatchGpu_4x64(backend, SymmetrizeMode::Roundtrip);
    // ... остальные в Roundtrip

    // --- GpuKernel mode ---
    TestCpuIdentity(backend, SymmetrizeMode::GpuKernel);
    TestCpu341(backend, SymmetrizeMode::GpuKernel);
    // ... остальные в GpuKernel

    // --- Cross-backend 85×85 ---
    TestConvert_VectorInput(backend);
    TestConvert_HipInput(backend);
    // TestConvert_ClMemInput — SKIP если нет HybridGPUContext
    TestConvert_OutputFormats(backend);

    // --- Result access ---
    TestResultAccess(backend);

    // --- Benchmark (опционально) ---
    // BenchmarkSingle341(backend);
    // BenchmarkBatch_16x64(backend);
    // BenchmarkBatch_4x256(backend);
}

#endif
```

**Файл**: `src/main.cpp` — добавить вызов `RunVectorAlgebraTests(backend)`

##### 5.14 Подзадачи:
- [ ] `all_test.hpp` — вызывает все тесты обоих режимов
- [ ] `src/main.cpp` — подключён `all_test.hpp`
- [ ] Компиляция проходит
- [ ] **ВСЕ функциональные тесты PASS**
- [ ] **ВСЕ cross-backend тесты PASS** (или SKIP если нет ZeroCopy)
- [ ] `tests/README.md` — описание всех тестов

---

### ГРУППА 8: Профилирование

---

#### 5.15 TestProfilerIntegration

**Файл**: добавить в `test_cholesky_inverter_rocm.hpp`

```cpp
void TestProfilerIntegration(drv_gpu_lib::IBackend* backend) {
    auto& profiler = drv_gpu_lib::GPUProfiler::GetInstance();

    // SetGPUInfo ПЕРЕД Start() — обязательно!
    auto dev_info = backend->GetDeviceInfo();
    drv_gpu_lib::GPUReportInfo report_info;
    report_info.gpu_name = dev_info.name;
    report_info.backend_type = drv_gpu_lib::BackendType::ROCm;
    // ... (см. раздел 10 плана)
    profiler.SetGPUInfo(backend->GetDeviceIndex(), report_info);
    profiler.Start();

    // hipEvent timing
    hipEvent_t ev_start, ev_stop;
    // ... Invert 341×341 ...
    // ... Record(ROCmProfilingData) ...

    profiler.Stop();
    profiler.PrintReport();
    profiler.ExportMarkdown("Results/Profiler/cholesky_invert.md");
    profiler.ExportJSON("Results/Profiler/cholesky_invert.json");
}
```

##### 5.15 Подзадачи:
- [ ] `SetGPUInfo()` с `GPUReportInfo` (gpu_name, backend_type, global_mem_mb, drivers)
- [ ] `SetGPUInfo()` ПЕРЕД `Start()` — **обязательно!**
- [ ] `hipEvent` timing для Invert 341×341
- [ ] `ROCmProfilingData` — start_ns, end_ns, device_id, kernel_name
- [ ] `profiler.Record()` — ROCm overload
- [ ] Вывод: `PrintReport()` + `ExportMarkdown()` + `ExportJSON()`
- [ ] **ЗАПРЕЩЕНО**: `GetStats()` + цикл + `con.Print` / `std::cout`
- [ ] Файлы результатов: `Results/Profiler/cholesky_invert.md` + `.json`

---

### ГРУППА 9: Документация + индексы

---

#### 5.16 Документация

##### 5.16.1 Doc/Python/vector_algebra_api.md
- [ ] Документирован конструктор с `SymmetrizeMode`
- [ ] `invert_cpu(flat, n)` — описание, пример, параметры, возврат
- [ ] `invert_batch_cpu(flat, n, batch)` — описание, пример
- [ ] `set_symmetrize_mode()` / `get_symmetrize_mode()`
- [ ] Таблица точности (5×5 < 1e-5, 341 < 1e-2)
- [ ] Зависимости (ROCm, rocBLAS, rocSOLVER)

##### 5.16.2 tests/README.md
- [ ] Перечислены все тесты (функциональные + cross-backend + benchmark + profiler)
- [ ] Описание режимов Roundtrip / GpuKernel

##### 5.16.3 Индексы MemoryBank
- [ ] `TASKS_ROCm_INDEX.md` — добавлен Task_11
- [ ] `MASTER_INDEX.md` — обновлён модуль vector_algebra

---

## 6. Зависимости между группами

```
ГРУППА 1 (Инфраструктура: CMake, types, traits)
    │
    ▼
ГРУППА 2 (Core: POTRF+POTRI + Roundtrip)
    │
    ├──── ТЕСТ ГРУППЫ 2 (Roundtrip mode: 10 тестов)
    │
    ▼
ГРУППА 3 (GPU Kernel: symmetrize)
    │
    ├──── ТЕСТ ГРУППЫ 3 (GpuKernel mode: те же 10 тестов)
    │
    ├──────────────────────┐
    ▼                      ▼
ГРУППА 4              ГРУППА 5
(Cross-backend 85×85)  (Python bindings + тесты)
    │                      │
    ├──────────────────────┘
    ▼
ГРУППА 6 (Benchmark: Roundtrip vs GpuKernel)
    │
    ▼
ГРУППА 7 (ВСЕ тесты: all_test.hpp + main)
    │
    ▼
ГРУППА 8 (Профилирование: GPUProfiler)
    │
    ▼
ГРУППА 9 (Документация + индексы)
```

**Параллельно можно**: Группы 4 и 5 (после Группы 3).

---

## 7. Чек-лист финальный

### Инфраструктура (Группа 1)
- [ ] 5.1: CMake (rocblas + rocsolver + hiprtc, `ENABLE_ROCM`)
- [ ] 5.2: `vector_algebra_types.hpp` — `CholeskyResult` + `SymmetrizeMode`
- [ ] 5.3: `is_cl_mem` trait (если не было)

### Core + Roundtrip (Группа 2)
- [ ] 5.4: `cholesky_inverter_rocm.hpp` — полный API
- [ ] 5.5.1–5.5.5: Core (конструктор, POTRF, POTRI, Roundtrip symmetrize)
- [ ] 5.5.6–5.5.11: Все 6 перегрузок Invert/InvertBatch
- [ ] 5.5.12: CholeskyResult methods (AsVector, matrix, matrices, move)

### Тесты Core (Тест Группы 2)
- [ ] 5.6.1–5.6.10: 10 функциональных тестов — ВСЕ PASS в Roundtrip

### GPU Kernel (Группа 3)
- [ ] 5.7: Kernel source (.hpp)
- [ ] 5.8.1–5.8.3: CompileKernels + Launch + Batched

### Тесты Kernel (Тест Группы 3)
- [ ] 5.9: Те же 10 тестов — ВСЕ PASS в GpuKernel

### Cross-backend (Группа 4)
- [ ] 5.10.1–5.10.4: 4 теста × 2 режима = 8 PASS

### Python (Группа 5)
- [ ] 5.11: Биндинги (py_vector_algebra_rocm.hpp + registration)
- [ ] 5.12.1–5.12.6: 6 Python тестов PASS

### Benchmark (Группа 6)
- [ ] 5.13.1–5.13.3: 3 бенчмарка — результаты записаны

### Все тесты (Группа 7)
- [ ] 5.14: all_test.hpp + main — **ВСЕ тесты PASS**

### Профилирование (Группа 8)
- [ ] 5.15: GPUProfiler integration — PrintReport + Export

### Документация (Группа 9)
- [ ] 5.16.1: Python API doc
- [ ] 5.16.2: tests/README.md
- [ ] 5.16.3: Индексы MemoryBank

### Компиляция (финальная проверка)
- [ ] Сборка с `ENABLE_ROCM=ON` (Linux): без ошибок
- [ ] Сборка с `ENABLE_ROCM=OFF` (Windows): без ошибок
- [ ] Все C++ тесты: PASS
- [ ] Все Python тесты: PASS
- [ ] Benchmark данные: записаны
- [ ] Profiler отчёт: сгенерирован

---

## 8. Ссылки

- **Plan**: [`MemoryBank/DiscussionPlan/PLAN_Vector_Algebra_Cholesky.md`](../DiscussionPlan/PLAN_Vector_Algebra_Cholesky.md)
- **Предыдущая версия**: Task_10_VectorAlgebraCholesky (✅ COMPLETED → удалён)
- **Зависимость Task_08**: [`MemoryBank/tasks/Task_08_ZeroCopy.md`](Task_08_ZeroCopy.md)
- **DrvGPU InputData**: [`DrvGPU/interface/input_data.hpp`](../../DrvGPU/interface/input_data.hpp)
- **DrvGPU Traits**: [`DrvGPU/interface/input_data_traits.hpp`](../../DrvGPU/interface/input_data_traits.hpp)
- **GPUProfiler пример**: [`Examples/GPUProfiler_SetGPUInfo.md`](../../Examples/GPUProfiler_SetGPUInfo.md)
- **Оптимизация**: [`Doc_Addition/Roc hip kernel оптимизация.md`](../../Doc_Addition/Roc%20hip%20kernel%20оптимизация.md)
- **Референс (LCH-Farrow01)**: `/home/alex/C++/LCH-Farrow01/Matrix/`
