# План: Модуль Vector Algebra — Cholesky Inverter (ROCm)

> **Дата**: 2026-02-25
> **Автор**: Кодо
> **Статус**: ✅ ФИНАЛЬНЫЙ ПЛАН — СОГЛАСОВАН

---

## 1. Цель

Инверсия эрмитовых положительно определённых матриц методом Холецкого.
**ВСЁ на GPU** (ROCm: POTRF + POTRI + симметризация).

**Два подхода к симметризации** — реализуем оба, бенчмарк, выбираем лучший:

| Подход | Симметризация | Описание |
|--------|--------------|----------|
| **Roundtrip** | Download → CPU sym → Upload | Без kernel, простой |
| **GpuKernel** | HIP kernel in-place на GPU | Всё на GPU, без round-trip |

**Источник**: [LCH-Farrow01/Matrix](/home/alex/C++/LCH-Farrow01/Matrix)

---

## 2. Архитектура (SOLID / GoF)

### Принцип: общий core + Strategy для симметризации

```
CholeskyInverterROCm (public API, фасад)
│
├── Core: POTRF + POTRI (rocSOLVER) ← общий для обоих подходов
│
├── SymmetrizeMode::Roundtrip  ← download → CPU sym → upload
└── SymmetrizeMode::GpuKernel  ← hiprtc kernel in-place
```

**SOLID:**
- **SRP**: core (POTRF+POTRI) отдельно, GPU kernel отдельно
- **OCP**: новый способ симметризации = новый mode, core не меняется
- **DIP**: публичный API не зависит от деталей kernel/roundtrip

**Переключение режима:**
```cpp
enum class SymmetrizeMode { Roundtrip, GpuKernel };

CholeskyInverterROCm inverter(backend, SymmetrizeMode::GpuKernel);
// или
inverter.SetSymmetrizeMode(SymmetrizeMode::Roundtrip);
```

---

## 3. Структура файлов

```
modules/vector_algebra/
├── include/
│   ├── cholesky_inverter_rocm.hpp         # Публичный API + enum SymmetrizeMode
│   ├── vector_algebra_types.hpp           # CholeskyResult
│   └── kernels/
│       └── symmetrize_kernel_sources_rocm.hpp  # HIP kernel как строка (hiprtc)
├── src/
│   ├── cholesky_inverter_rocm.cpp         # Core: POTRF+POTRI + Roundtrip + диспетчер
│   └── symmetrize_gpu_rocm.cpp            # CompileKernels + SymmetrizeUpperToFullGPU
├── tests/
│   ├── test_cholesky_inverter_rocm.hpp    # Функциональные тесты (оба режима)
│   ├── test_cross_backend_conversion.hpp  # Тест конвертации 85×85 (все пути данных)
│   ├── test_benchmark_symmetrize.hpp      # Бенчмарк: Roundtrip vs GpuKernel
│   └── all_test.hpp
└── CMakeLists.txt

python/py_vector_algebra_rocm.hpp          # pybind11
Doc/Python/vector_algebra_api.md
Python_test/vector_algebra/test_cholesky_inverter_rocm.py
```

**Почему 2 .cpp**: `symmetrize_gpu_rocm.cpp` — отдельный т.к. зависит от hiprtc и содержит CompileKernels/Launch. Core не знает про hiprtc. Файлы небольшие.

---

## 4. API

### 4.1 Класс

```cpp
namespace vector_algebra {

enum class SymmetrizeMode { Roundtrip, GpuKernel };

class CholeskyInverterROCm {
public:
    explicit CholeskyInverterROCm(
        drv_gpu_lib::IBackend* backend,
        SymmetrizeMode mode = SymmetrizeMode::GpuKernel);
    ~CholeskyInverterROCm();

    void SetSymmetrizeMode(SymmetrizeMode mode);
    SymmetrizeMode GetSymmetrizeMode() const;

    // --- Одна матрица (n из sqrt(n_point)) ---

    // CPU vector → upload → GPU compute → return CholeskyResult
    CholeskyResult Invert(
        const drv_gpu_lib::InputData<std::vector<std::complex<float>>>& input);

    // void* (HIP device ptr) → GPU compute → return CholeskyResult
    CholeskyResult Invert(
        const drv_gpu_lib::InputData<void*>& input);

    // cl_mem (OpenCL) → ZeroCopy → GPU compute → return CholeskyResult
    CholeskyResult Invert(
        const drv_gpu_lib::InputData<cl_mem>& input);

    // --- Batched (antenna_count = batch_count, n_point = n*n) ---

    CholeskyResult InvertBatch(
        const drv_gpu_lib::InputData<std::vector<std::complex<float>>>& input, int n);

    CholeskyResult InvertBatch(
        const drv_gpu_lib::InputData<void*>& input, int n);

    CholeskyResult InvertBatch(
        const drv_gpu_lib::InputData<cl_mem>& input, int n);

private:
    drv_gpu_lib::IBackend* backend_;
    void* handle_;                        // rocblas_handle (opaque)
    SymmetrizeMode mode_;

    // Core: POTRF + POTRI (общий для всех типов входа)
    void CorePotrf(void* d_matrix, int n, hipStream_t stream);
    void CorePotri(void* d_matrix, int n, hipStream_t stream);
    void CorePotrfBatched(void** d_matrices, int n, int batch, hipStream_t stream);
    void CorePotriBatched(void** d_matrices, int n, int batch, hipStream_t stream);

    // Симметризация: Roundtrip (download → CPU sym → upload)
    void SymmetrizeRoundtrip(void* d_matrix, int n, hipStream_t stream);
    void SymmetrizeRoundtripBatched(void** d_matrices, int n, int batch, hipStream_t stream);

    // Симметризация: GPU kernel (реализация в symmetrize_gpu_rocm.cpp)
    hipModule_t   sym_module_  = nullptr;
    hipFunction_t sym_kernel_  = nullptr;
    bool          kernels_compiled_ = false;
    void CompileKernels();
    void SymmetrizeGpuKernel(void* d_matrix, int n, hipStream_t stream);
    void SymmetrizeGpuKernelBatched(void** d_matrices, int n, int batch, hipStream_t stream);
};

}  // namespace vector_algebra
```

### 4.2 CholeskyResult — единый тип (базовый void\*)

Внутренний формат — **void\* (HIP device ptr)**. Два метода доступа: **AsVector()** и **AsHipPtr()**.

> **HIP→CL мост**: отложен. `AsClMem()` — НЕ реализуем сейчас.

```cpp
struct CholeskyResult {
    void* d_data;                   // HIP device ptr (базовый формат)
    drv_gpu_lib::IBackend* backend; // для Memcpy/Free
    int matrix_size;                // n (одна сторона матрицы)
    int batch_count;                // количество матриц

    // --- Доступ к данным ---

    // Download GPU → CPU vector (через SVM)
    std::vector<std::complex<float>> AsVector() const;

    // Вернуть HIP device ptr (caller не владеет — не вызывать Free!)
    void* AsHipPtr() const { return d_data; }

    // --- Удобные методы ---

    // Одна матрица как 2D vector [n][n]
    std::vector<std::vector<std::complex<float>>> matrix() const;

    // Batch как 3D vector [batch][n][n]
    std::vector<std::vector<std::vector<std::complex<float>>>> matrices() const;

    // --- Владение памятью ---
    // CholeskyResult ВЛАДЕЕТ d_data. Деструктор вызывает backend->Free(d_data).
    ~CholeskyResult();
    CholeskyResult(CholeskyResult&& other) noexcept;
    CholeskyResult& operator=(CholeskyResult&& other) noexcept;
    CholeskyResult(const CholeskyResult&) = delete;
    CholeskyResult& operator=(const CholeskyResult&) = delete;
};
```

### 4.3 Вход/Выход

**ВСЁ GPU** означает: ВЫЧИСЛЕНИЯ (POTRF+POTRI+Symmetrize) всегда на GPU.
Но **интерфейс** принимает три типа входных данных:

| Вход | Выход | Путь обработки |
|------|-------|----------------|
| `InputData<vector<complex<float>>>` | `CholeskyResult` | Upload → GPU compute → Symmetrize |
| `InputData<void*>` | `CholeskyResult` | GPU compute → Symmetrize → новый буфер |
| `InputData<cl_mem>` | `CholeskyResult` | ZeroCopy → GPU compute → Symmetrize |

**Выходные форматы** (из CholeskyResult):

| Метод | Возврат | Описание |
|-------|---------|----------|
| `AsVector()` | `vector<complex<float>>` | Download через SVM на CPU |
| `AsHipPtr()` | `void*` | HIP device pointer (GPU) |

- Вход **не изменяется** (создаётся копия, POTRF+POTRI in-place на копии)
- `AsVector()` использует SVM — работает и в C++ и в Python

> **Python обёртка**: numpy → upload → Invert(void*) → AsVector() → numpy

---

## 5. Зависимости

| Зависимость | Назначение |
|-------------|-----------|
| hip, hiprtc | Runtime + kernel compilation (только для GpuKernel mode) |
| rocblas | BLAS handle, stream |
| rocsolver | POTRF, POTRI, batched варианты |
| DrvGPU (IBackend) | Allocate, Free, Memcpy, GetNativeQueue, Synchronize |
| DrvGPU (HybridBackend) | Для cl_mem входа: ZeroCopyBridge (OpenCL → HIP) |
| DrvGPU (GPUProfiler) | SetGPUInfo, Record (ROCmProfilingData), PrintReport, Export* |

---

## 6. Пайплайн обработки

### 6.1 Одна матрица — Invert(void*)

```
Input: void* d_input (n*n complex<float>)
  │
  ├── 1. Allocate d_output (n*n)
  ├── 2. MemcpyDeviceToDevice(d_output, d_input)   // копия, не in-place
  ├── 3. CorePotrf(d_output, n, stream)             // разложение Холецкого
  ├── 4. CorePotri(d_output, n, stream)             // инверсия верхнетреугольной
  ├── 5а. [Roundtrip] SymmetrizeRoundtrip(d_output) // download → CPU → upload
  ├── 5б. [GpuKernel] SymmetrizeGpuKernel(d_output) // HIP kernel in-place
  └── return CholeskyResult{d_output, backend, n, 1}
```

### 6.2 Batched — InvertBatch(void*, n)

```
Input: void* d_input (batch * n*n complex<float>)
  │
  ├── 1. Allocate d_output (batch * n*n)
  ├── 2. MemcpyDeviceToDevice(d_output, d_input)
  ├── 3. Подготовить массив указателей d_ptrs[batch] (каждый → d_output + k*n*n)
  ├── 4. CorePotrfBatched(d_ptrs, n, batch, stream)
  ├── 5. CorePotriBatched(d_ptrs, n, batch, stream)
  ├── 6. Для каждой матрицы: Symmetrize (Roundtrip или GpuKernel)
  └── return CholeskyResult{d_output, backend, n, batch}
```

### 6.3 Invert(cl_mem) — через ZeroCopy

```
Input: InputData<cl_mem> (n*n complex<float>)
  │
  ├── 1. HybridBackend::SyncBeforeZeroCopy()
  ├── 2. CreateZeroCopyBridge(cl_mem, size) → hip_ptr
  ├── 3. Allocate d_output (n*n)
  ├── 4. MemcpyDeviceToDevice(d_output, hip_ptr)   // копия из bridge
  ├── 5. CorePotrf + CorePotri + Symmetrize (как в 6.1)
  └── return CholeskyResult{d_output, backend, n, 1}
```

---

## 7. C++ тесты — полная карта

### 7.1 Принцип: каждый тест запускается для ОБОИХ режимов

```cpp
void TestSingle341(IBackend* backend, SymmetrizeMode mode);
// Вызов:
TestSingle341(backend, SymmetrizeMode::Roundtrip);
TestSingle341(backend, SymmetrizeMode::GpuKernel);
```

### 7.2 Функциональные тесты (test_cholesky_inverter_rocm.hpp)

**По типу входных данных:**

| # | Тест | Тип входа | n | batch | Проверка |
|---|------|----------|---|-------|----------|
| 1 | `TestCpuIdentity` | vector\<\> | 5 | 1 | I⁻¹ = I, Frobenius < 1e-5 |
| 2 | `TestCpu341` | vector\<\> | 341 | 1 | Frobenius < 1e-2 |
| 3 | `TestGpuVoidPtr341` | void* | 341 | 1 | Frobenius < 1e-2 |
| 4 | `TestZeroCopyClMem` | cl_mem | 341 | 1 | Frobenius < 1e-2 (SKIP если нет HybridGPUContext) |

**По batch count:**

| # | Тест | Тип входа | n | batch | Проверка |
|---|------|----------|---|-------|----------|
| 5 | `TestBatchCpu_4x64` | vector\<\> | 64 | 4 | Frobenius < 1e-3 для каждой |
| 6 | `TestBatchGpu_4x64` | void* | 64 | 4 | Frobenius < 1e-3 для каждой |
| 7 | `TestBatchSizes` | void* | 64 | **1, 4, 8, 16** | Все проходят |
| 8 | `TestMatrixSizes` | void* | **32, 64, 128, 256** | 4 | Frobenius < 1e-2 |

**Утилиты и доступ:**

| # | Тест | Проверка |
|---|------|----------|
| 9 | `TestResultAccess` | `.matrix()` shape, `.matrices()` shape (1 + 3 batched) |
| 10 | `TestResolveMatrixSize` | sqrt(n_point) логика (без GPU) |

**Каждый тест (#1–#8) × 2 режима (Roundtrip, GpuKernel) = 16 запусков функциональных.**

### 7.3 Тест конвертации — CrossBackend 85×85 (test_cross_backend_conversion.hpp)

**Цель**: одна матрица 85×85, проверить ВСЕ комбинации входов/выходов.

#### Часть A: 3 входа → void\* → AsVector() → сравнить

Одна HPD матрица M(85×85). Три пути ввода, результаты должны совпасть:

| # | Тест | Вход | Откуда данные | Выход |
|---|------|------|---------------|-------|
| C1 | `TestConvert_VectorInput` | vector\<\> | CPU | AsVector() = **ЭТАЛОН** |
| C2 | `TestConvert_HipInput` | void* | **Другой ROCm контекст** (тот же GPU) | AsVector() → сравнить с C1 |
| C3 | `TestConvert_ClMemInput` | cl_mem | **OpenCL контекст** (через ZeroCopyBridge) | AsVector() → сравнить с C1 |

**Паттерн для C2** (другой ROCm контекст):
```cpp
// Второй ROCm контекст на том же GPU
drv_gpu_lib::ROCmBackend other_rocm;
other_rocm.Initialize(0);

// Записываем матрицу через ДРУГОЙ контекст
void* d_foreign = other_rocm.Allocate(n * n * sizeof(cf));
other_rocm.MemcpyHostToDevice(d_foreign, matrix.data(), ...);

// Передаём в НАШ инвертер (backend_ — основной ROCm)
auto result = inverter.Invert(InputData<void*>{d_foreign, n*n, 1});
auto vec = result.AsVector();
// compare vec vs reference

other_rocm.Free(d_foreign);
other_rocm.Cleanup();
```

**Паттерн для C3** (OpenCL → ZeroCopy):
```cpp
// По паттерну из DrvGPU/tests/test_zero_copy.hpp
drv_gpu_lib::OpenCLBackend ocl_backend;
ocl_backend.Initialize(0);

void* cl_buf = ocl_backend.Allocate(n * n * sizeof(cf));
ocl_backend.MemcpyHostToDevice(cl_buf, matrix.data(), ...);
ocl_backend.Synchronize();

auto result = inverter.Invert(InputData<cl_mem>{(cl_mem)cl_buf, n*n, 1});
auto vec = result.AsVector();
// compare vec vs reference

ocl_backend.Free(cl_buf);
ocl_backend.Cleanup();
```

#### Часть B: 1 вход → 2 конвертации выхода

| # | Тест | Описание |
|---|------|----------|
| C4 | `TestConvert_OutputFormats` | vector вход → result → AsVector() + AsHipPtr()→hipMemcpy→vector → сравнить оба |

```cpp
auto result = inverter.Invert(InputData<vector<cf>>{matrix, n*n, 1});

// Конвертация 1: AsVector()
auto vec1 = result.AsVector();

// Конвертация 2: AsHipPtr() → ручной download
void* hip_ptr = result.AsHipPtr();
std::vector<std::complex<float>> vec2(n * n);
hipMemcpy(vec2.data(), hip_ptr, n*n*sizeof(cf), hipMemcpyDeviceToHost);

// vec1 == vec2 (побитово или Frobenius < 1e-7)
```

#### Итого CrossBackend: 4 теста × 2 режима = 8 запусков

### 7.4 Бенчмарк (test_benchmark_symmetrize.hpp)

| # | Тест | n | batch | Что меряем |
|---|------|---|-------|-----------|
| B1 | `BenchmarkSingle341` | 341 | 1 | Roundtrip vs GpuKernel (hipEvent timing) |
| B2 | `BenchmarkBatch_16x64` | 64 | 16 | Roundtrip vs GpuKernel |
| B3 | `BenchmarkBatch_4x256` | 256 | 4 | Roundtrip vs GpuKernel |

Результат: таблица `[mode, n, batch, time_ms]` через GPUProfiler → `ExportMarkdown`.

### 7.5 Профилирование

| # | Тест | Что проверяем |
|---|------|--------------|
| P1 | `TestProfilerIntegration` | SetGPUInfo + Record(ROCmProfilingData) + Export |

### 7.6 Допуски Frobenius

| n | Допуск | Причина |
|---|--------|---------|
| 5 (identity) | < 1e-5 | Тривиальный случай |
| 32–128 | < 1e-3 | Маленькие матрицы, float32 хватает |
| 256–341 | < 1e-2 | Большие матрицы, float32 теряет точность |
| 85 (cross-backend) | < 1e-3 | Средняя матрица, проверка конвертации |

### 7.7 Batch counts в тестах

| Тест | batch_count |
|------|-------------|
| TestSingleIdentity | 1 |
| TestSingle341 | 1 |
| TestBatch_4x64 | **4** |
| TestBatchSizes | **1, 4, 8, 16** |
| TestMatrixSizes | **4** |
| TestResultAccess | 1 + **3** |
| TestCrossBackend* | 1 |
| BenchmarkBatch_16x64 | **16** |
| BenchmarkBatch_4x256 | **4** |

---

## 8. Python тесты

**Файл**: `Python_test/vector_algebra/test_cholesky_inverter_rocm.py`

| # | Тест | n | batch | Вход | Проверка |
|---|------|---|-------|------|----------|
| 1 | `test_invert_5x5` | 5 | 1 | numpy vector | np.linalg.inv vs GPU, Frobenius < 1e-5 |
| 2 | `test_invert_341x341` | 341 | 1 | numpy vector | Frobenius < 1e-2 |
| 3 | `test_batch_4x64` | 64 | 4 | numpy vector | Frobenius < 1e-3 |
| 4 | `test_batch_sizes` | 64 | 1, 4, 8 | numpy vector | Все проходят |
| 5 | `test_modes_roundtrip_vs_kernel` | 64 | 4 | numpy vector | Оба режима дают одинаковый результат |
| 6 | `test_cross_backend_vector_vs_clmem` | 85 | 1 | numpy + cl_mem | vector-путь vs cl_mem-путь (HybridGPUContext) → одинаковый результат |

**Тест #6 — кросс-бэкенд Python:**
```python
def test_cross_backend_vector_vs_clmem():
    ctx_rocm = gpuworklib.ROCmGPUContext(0)
    ctx_hybrid = gpuworklib.HybridGPUContext(0)

    M = generate_hpd_matrix(85)

    # Путь 1: numpy vector → ROCm → result
    inv1 = gpuworklib.cholesky_invert(ctx_rocm, M)

    # Путь 2: через cl_mem (HybridGPUContext) → result
    inv2 = gpuworklib.cholesky_invert(ctx_hybrid, M)

    # Оба результата — numpy vectors, сравниваем
    assert frobenius_norm(inv1 - inv2) < 1e-4
```

---

## 9. HIP kernel — паттерн (по fft_maxima)

### 9.1 Ядро — строка в `include/kernels/symmetrize_kernel_sources_rocm.hpp`

```cpp
#pragma once
#if ENABLE_ROCM
namespace vector_algebra { namespace kernels {

inline const char* GetSymmetrizeKernelSource() {
    return R"HIP(

extern "C" __global__ void symmetrize_upper_to_full(
    float2* __restrict__ data,
    unsigned int n)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    if (col > row) {
        float2 v = data[row * n + col];
        data[col * n + row] = {v.x, -v.y};
    }
}

)HIP";
}

}}  // namespace vector_algebra::kernels
#endif
```

### 9.2 CompileKernels — через hiprtc (в `symmetrize_gpu_rocm.cpp`)

```cpp
void CholeskyInverterROCm::CompileKernels() {
    if (kernels_compiled_) return;
    const char* src = vector_algebra::kernels::GetSymmetrizeKernelSource();

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, src, "symmetrize.hip", 0, nullptr, nullptr);
    hiprtcResult res = hiprtcCompileProgram(prog, 0, nullptr);
    if (res != HIPRTC_SUCCESS) { /* log + throw */ }

    size_t code_size;
    hiprtcGetCodeSize(prog, &code_size);
    std::string code(code_size, '\0');
    hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);

    hipModuleLoadData(&sym_module_, code.data());
    hipModuleGetFunction(&sym_kernel_, sym_module_, "symmetrize_upper_to_full");
    kernels_compiled_ = true;
}
```

### 9.3 Launch — через hipModuleLaunchKernel

```cpp
void CholeskyInverterROCm::SymmetrizeGpuKernel(
    void* ptr, int n, hipStream_t stream) {
    CompileKernels();
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    void* args[] = {&ptr, &n};
    hipModuleLaunchKernel(sym_kernel_,
        grid.x, grid.y, 1, block.x, block.y, 1,
        0, stream, args, nullptr);
}
```

### 9.4 Запрещённые паттерны

| ❌ Нельзя | ✅ Правильно |
|----------|-------------|
| `static` методы | Нестатические (10 GPU параллельно) |
| `__global__` inline в .cpp | Ядро как строка + hiprtc |
| `hipLaunchKernelGGL` | `hipModuleLaunchKernel` |
| Общий `hipModule_t` (static) | `hipModule_t` в каждом объекте |

---

## 10. Профилирование — GPUProfiler

### 10.1 SetGPUInfo (GPUDeviceInfo ≠ GPUReportInfo)

```cpp
auto dev_info = backend->GetDeviceInfo();
drv_gpu_lib::GPUReportInfo report_info;
report_info.gpu_name      = dev_info.name.empty() ? "Unknown" : dev_info.name;
report_info.backend_type  = drv_gpu_lib::BackendType::ROCm;
report_info.global_mem_mb = dev_info.global_memory_size / (1024 * 1024);

std::map<std::string, std::string> rocm_driver;
rocm_driver["driver_type"]    = "ROCm";
rocm_driver["driver_version"] = dev_info.driver_version;
rocm_driver["vendor"]         = dev_info.vendor;
report_info.drivers.push_back(rocm_driver);

profiler.SetGPUInfo(backend->GetDeviceIndex(), report_info);
```

### 10.2 Record — ROCm оверлоад (НЕ MakeOpenCLFromDurationMs!)

```cpp
// hipEvent timing
hipEvent_t ev_start, ev_stop;
hipEventCreate(&ev_start);
hipEventCreate(&ev_stop);
hipEventRecord(ev_start, stream);
/* ... Invert() ... */
hipEventRecord(ev_stop, stream);
hipEventSynchronize(ev_stop);
float elapsed_ms_f = 0.0f;
hipEventElapsedTime(&elapsed_ms_f, ev_start, ev_stop);
hipEventDestroy(ev_start);
hipEventDestroy(ev_stop);

// ROCmProfilingData — специальный оверлоад Record()
drv_gpu_lib::ROCmProfilingData rdata{};
rdata.start_ns    = 0;
rdata.end_ns      = static_cast<uint64_t>(elapsed_ms_f * 1e6f);
rdata.device_id   = backend->GetDeviceIndex();
rdata.kernel_name = "POTRF_POTRI_341x341";
profiler.Record(backend->GetDeviceIndex(), "Cholesky", "POTRF_POTRI_341x341", rdata);
```

### 10.3 Правила

- `SetGPUInfo()` **до** `Start()`
- Вывод **только** через `PrintReport()` / `ExportMarkdown()` / `ExportJSON()`
- **Запрещено**: `GetStats()` + цикл + `con.Print` / `std::cout`

---

## 11. Порядок работ

1. `cmake/dependencies.cmake` — добавить rocblas, rocsolver (если не добавлены)
2. Создать `modules/vector_algebra/` — структура файлов (раздел 3)
3. `vector_algebra_types.hpp` — `CholeskyResult`, `SymmetrizeMode`
4. `cholesky_inverter_rocm.hpp` — класс с двумя режимами
5. `cholesky_inverter_rocm.cpp` — Core (POTRF+POTRI) + Roundtrip symmetrize
6. `kernels/symmetrize_kernel_sources_rocm.hpp` — HIP kernel source
7. `symmetrize_gpu_rocm.cpp` — CompileKernels + SymmetrizeGpuKernel
8. `test_cholesky_inverter_rocm.hpp` — функциональные тесты (оба режима)
9. `test_cross_backend_conversion.hpp` — тест конвертации 85×85
10. `test_benchmark_symmetrize.hpp` — бенчмарк Roundtrip vs GpuKernel
11. `all_test.hpp` — точка входа
12. Python bindings + тесты (включая cross-backend #6)
13. Профилирование — GPUProfiler integration тест

---

## 12. Правила работы

- **Минимальные правки**: не переписывать рабочий код целиком
- **10 GPU параллельно**: никаких `static`, всё через объект
- **ВСЁ GPU**: вычисления на GPU, но интерфейс принимает vector\<\>, void*, cl_mem
- **Бенчмарк перед выбором**: не удалять Roundtrip пока не докажем что Kernel быстрее
- **Выход**: только vector (SVM) + void* (HIP). Мост HIP→CL — отложен.

---

## 13. Референсы

| Элемент | Файл |
|---------|------|
| POTRF+POTRI | LCH-Farrow01/Matrix/matrix_invert.cpp:349-399 |
| Batched | LCH-Farrow01/Matrix/matrix_invert_advanced.cpp:470-586 |
| HIP kernel pattern | modules/fft_maxima/include/kernels/all_maxima_kernel_sources_rocm.hpp |
| hiprtc compile | modules/fft_maxima/src/all_maxima_pipeline_rocm.cpp |
| GPUReportInfo | modules/fft_maxima/tests/test_batch_all_maxima.hpp:282-292 |
| Record ROCm | DrvGPU/services/gpu_profiler.hpp:131-140 |
| ROCmProfilingData | DrvGPU/services/profiling_types.hpp:72-83 |
| **Другой ROCm контекст** | DrvGPU/tests/test_hybrid_backend.hpp (test_rocm_allocate) |
| **ZeroCopy cl_mem→HIP** | DrvGPU/tests/test_zero_copy.hpp (test_data_integrity) |
| **External OpenCL** | DrvGPU/tests/example_external_context_usage.hpp |
| **HybridBackend** | DrvGPU/backends/hybrid/hybrid_backend.hpp |

---

## 14. Решения — отложено

| Решение | Статус | Примечание |
|---------|--------|-----------|
| HIP→CL мост (AsClMem) | ❌ Отложено | Alex помнит, сделаем позже |
| Roundtrip HIP→Host→CL | ❌ Не делаем | Нет смысла без моста |
| ~~InvertBatch(cl_mem)~~ | ✅ **Включено** | Макс 64×341² × 8B ≈ 57 MB — ничего для GPU. Один ZeroCopyBridge на весь batch буфер. |

---

## 15. Оптимизация — план ускорения ×10

> **Когда**: ПОСЛЕ того как тесты работают + получены данные профилирования (Roundtrip vs GpuKernel)
> **Цель**: ускорить полный пайплайн Cholesky Invert в 10 раз
> **Методология**: [Doc_Addition/Roc hip kernel оптимизация.md](/home/alex/C++/GPUWorkLib/Doc_Addition/Roc%20hip%20kernel%20оптимизация.md)

### 15.1 Фаза 0: Baseline профилирование (обязательно первый шаг!)

> «Прежде чем оптимизировать — измерь» — без данных оптимизация вслепую.

**Инструменты:**
- `GPUProfiler` (наш) — время каждого шага пайплайна
- `rocprofv3 --stats` — hardware counters, occupancy, bandwidth utilization
- `hipEvent` timing — точное измерение каждого этапа

**Что измеряем** (для single 341×341 и batched 4×256):

| Этап | Метрика | Зачем |
|------|---------|-------|
| Allocate(d_output) | время (мс) | Стоимость аллокации |
| MemcpyD2D (копия) | время + bandwidth | Overhead копирования |
| CorePotrf | время | Основной compute |
| CorePotri | время | Основной compute |
| Symmetrize | время | Наш kernel vs roundtrip |
| AsVector (download) | время + bandwidth | Стоимость вывода |
| **TOTAL** | сумма | Baseline для ×10 |

**Определяем тип bottleneck:**
- `Arithmetic Intensity = FLOPs / Bytes moved`
- Memory-bound → оптимизируем доступ к памяти
- Compute-bound → оптимизируем вычисления
- Latency-bound → оптимизируем overlap и launch overhead

**Результат фазы 0**: таблица `[этап, время_ms, % от total, bottleneck_type]` → ExportMarkdown

### 15.2 Фаза 1: Устранение overhead (Memory Management)

**Проблема**: `Allocate()` + `Free()` на каждый вызов Invert — дорого.

| Оптимизация | Описание | Ожидаемый эффект |
|-------------|----------|-----------------|
| **Memory Pool** | Кешировать буферы по размеру. `Allocate()` ищет в пуле, `Free()` возвращает в пул | -50% overhead аллокации |
| **Pre-allocated workspace** | Для повторных вызовов с одинаковым n — не аллоцировать заново | Почти нулевой malloc |
| **Pinned memory** | `hipHostMalloc` для host↔device передач (AsVector, vector input) | ×2-3 bandwidth H2D/D2H |
| **Avoid unnecessary D2D copy** | Если caller передаёт одноразовый буфер — POTRF/POTRI in-place без копии | -1 шаг пайплайна |

```cpp
// Memory Pool паттерн
class CholeskyInverterROCm {
    // ...
    struct MemoryPool {
        std::unordered_map<size_t, std::vector<void*>> free_buffers;
        void* Acquire(size_t size, IBackend* backend);
        void Release(void* ptr, size_t size);
        void Clear(IBackend* backend);  // в деструкторе
    } pool_;
};
```

### 15.3 Фаза 2: Оптимизация Symmetrize kernel

> Ядро **memory-bound**: минимум вычислений (read + conjugate + write), основная стоимость — memory transactions.

#### 2a. Coalesced Memory Access

Текущее ядро: `data[row * n + col]` — OK для row-major, потоки в warp идут по col → coalesced по строке.

Но `data[col * n + row]` (запись) — stride = n → **НЕ coalesced!**

**Решение**: тайлинг через LDS (shared memory):

```cpp
// Загрузить тайл в LDS → записать из LDS coalesced
__shared__ float2 tile[TILE_SIZE][TILE_SIZE + 1]; // +1 = padding vs bank conflicts

// Load: coalesced read по строке
tile[threadIdx.y][threadIdx.x] = data[row * n + col];
__syncthreads();

// Store: coalesced write (транспонированный доступ через LDS)
if (col > row) {
    float2 v = tile[threadIdx.y][threadIdx.x];
    data[col * n + row] = {v.x, -v.y};
}
```

#### 2b. Vectorized loads (float4)

```cpp
// Вместо одного float2 (8 байт) — загружаем float4 (16 байт)
// = 2 complex<float> за одну инструкцию
// Работает когда n кратно 2
float4 v = reinterpret_cast<float4*>(data)[idx];
```

#### 2c. `__launch_bounds__` + block size tuning

```cpp
// Подсказка компилятору для оптимального распределения регистров
extern "C" __global__ __launch_bounds__(256, 4)  // maxThreads=256, minBlocks=4
void symmetrize_upper_to_full(float2* data, unsigned int n)
```

**Auto-tuning block sizes** (AMD даёт до ×10 от правильного тюнинга!):

| Block size | Конфигурация | Тест |
|------------|-------------|------|
| 16×16 = 256 | Текущий baseline | ✅ |
| 32×8 = 256 | Больше по cols (лучше coalescing?) | Тест |
| 8×32 = 256 | Больше по rows | Тест |
| 32×32 = 1024 | Макс threads | Тест |
| 16×16 + LDS tiling | С shared memory | Тест |

#### 2d. Batched symmetrize — один launch

**Проблема**: цикл `for(k=0; k<batch; k++) SymmetrizeGpuKernel(ptr[k])` = batch запусков.

**Решение**: 3D grid — все матрицы за один launch:

```cpp
extern "C" __global__ void symmetrize_batched(
    float2** __restrict__ matrices,
    unsigned int n)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int batch_idx = blockIdx.z;  // ← матрица из batch
    if (row >= n || col >= n) return;
    if (col > row) {
        float2* data = matrices[batch_idx];
        float2 v = data[row * n + col];
        data[col * n + row] = {v.x, -v.y};
    }
}

// Launch: один вызов на весь batch
dim3 block(16, 16);
dim3 grid((n+15)/16, (n+15)/16, batch_count);  // z = batch
```

### 15.4 Фаза 3: Pipeline overlapping (Streams)

> Перекрытие вычислений с передачей данных.

#### 3a. Async upload + compute

Для `Invert(vector)` — перекрыть upload следующей матрицы с compute текущей:

```
Stream 0: [Upload M₀] [POTRF M₀] [POTRI M₀] [Sym M₀] [Download M₀]
Stream 1:             [Upload M₁] [POTRF M₁] [POTRI M₁] [Sym M₁] ...
```

#### 3b. Batched pipeline с несколькими streams

Для `InvertBatch` — разбить batch на chunks, каждый chunk в своём stream:

```
Stream 0: [POTRF chunk₀] [POTRI chunk₀] [Sym chunk₀]
Stream 1: [POTRF chunk₁] [POTRI chunk₁] [Sym chunk₁]
Stream 2: [POTRF chunk₂] [POTRI chunk₂] [Sym chunk₂]
```

> Нужен benchmark: при каком batch_count/n выгодно multi-stream.

### 15.5 Фаза 4: rocSOLVER tuning

POTRF + POTRI — основной compute. Мы не можем переписать rocSOLVER, но можем:

| Оптимизация | Описание |
|-------------|----------|
| **Проверить workspace** | rocSOLVER выделяет workspace автоматически. Можно pre-allocate оптимальный размер |
| **In-place вычисления** | Убедиться что POTRF/POTRI работают in-place (без внутренних копий) |
| **Правильный fill mode** | `rocblas_fill_upper` vs `rocblas_fill_lower` — проверить что совпадает с нашим layout |
| **Batched vs strided batched** | `rocsolver_cpotrf_batched` vs `rocsolver_cpotrf_strided_batched` — strided может быть быстрее (один contiguous буфер) |

### 15.6 Фаза 5: hipGraph (для повторных вызовов)

> Если пользователь вызывает Invert много раз с одинаковым n — фиксируем граф операций.

```cpp
// Capture
hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  MemcpyD2D(d_output, d_input, size);
  CorePotrf(d_output, n, stream);
  CorePotri(d_output, n, stream);
  SymmetrizeGpuKernel(d_output, n, stream);
hipStreamEndCapture(stream, &graph);
hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

// Replay (дёшево — нет launch overhead)
hipGraphLaunch(graph_exec, stream);
```

### 15.7 Фаза 6: Проверка register pressure

```bash
# Проверить VGPRs нашего symmetrize kernel
hipcc --resource-usage symmetrize_gpu_rocm.cpp
```

| Целевые VGPRs | Waves/SIMD | Occupancy |
|----------------|-----------|-----------|
| ≤ 64 | 8 (max) | 100% |
| ≤ 96 | 5 | 62.5% |
| ≤ 128 | 4 | 50% |

Symmetrize kernel простой → должен уложиться в ≤ 64 VGPRs → max occupancy.

Если нет:
- Определять переменные ближе к месту использования
- Избегать register spilling в scratch memory

### 15.8 Чеклист оптимизаций

| # | Оптимизация | Фаза | Ожидание | Сложность |
|---|-------------|------|----------|-----------|
| O1 | Baseline profiling | 0 | данные | низкая |
| O2 | Memory Pool | 1 | -30-50% overhead alloc | средняя |
| O3 | Pinned memory (H2D/D2H) | 1 | ×2-3 bandwidth | низкая |
| O4 | Avoid D2D copy (in-place mode) | 1 | -1 шаг pipeline | низкая |
| O5 | LDS tiling symmetrize | 2 | ×2-4 для symmetrize | средняя |
| O6 | Vectorized loads (float4) | 2 | +30-50% bandwidth | низкая |
| O7 | `__launch_bounds__` | 2 | +10-20% occupancy | низкая |
| O8 | Block size auto-tuning | 2 | до ×10 (AMD specific!) | средняя |
| O9 | Batched symmetrize (3D grid) | 2 | -batch×launch_overhead | низкая |
| O10 | Multi-stream pipeline | 3 | ×1.5-2 throughput | средняя |
| O11 | Strided batched POTRF/POTRI | 4 | зависит от профиля | низкая |
| O12 | hipGraph caching | 5 | -launch overhead × N calls | средняя |
| O13 | Register pressure check | 6 | max occupancy | низкая |

### 15.9 Порядок оптимизации

```
Шаг 0: BASELINE PROFILING (обязательно!)
  │
  ├── Bottleneck = Memory?
  │   → Фаза 1 (Pool, Pinned) + Фаза 2 (LDS, float4, tuning)
  │
  ├── Bottleneck = Launch overhead?
  │   → Фаза 2d (batched kernel) + Фаза 5 (hipGraph)
  │
  ├── Bottleneck = POTRF/POTRI?
  │   → Фаза 4 (rocSOLVER tuning, strided batched)
  │
  └── Bottleneck = Host↔Device?
      → Фаза 1 (Pinned memory) + Фаза 3 (stream overlap)

После каждой фазы: RE-PROFILE → определить новый bottleneck → следующая фаза
```

### 15.10 Правила оптимизации

- **Измерять ДО и ПОСЛЕ** каждой оптимизации через GPUProfiler
- **Один шаг за раз** — иначе не понятно что дало эффект
- **Не ломать тесты** — функциональные тесты (раздел 7) должны проходить после каждой оптимизации
- **FP32 only** — никаких `0.3` (double), только `0.3f` (float)
- **Результаты** → `Results/Profiler/vector_algebra_optimization.md`