# Task_12_FMCorrelatorROCm -- FFT-коррелятор фазовой модуляции на ROCm

> **Памятка для ИИ**: ROCm-only модуль. Тестировать **только под Linux** (Debian, Radeon 9070, gfx1201). Весь ROCm-код под `#if ENABLE_ROCM`. Вывод профилирования -- **ТОЛЬКО** через `GPUProfiler`: `PrintReport()`, `ExportMarkdown()`, `ExportJSON()`. **ЗАПРЕЩЕНО** `GetStats()` + цикл + `con.Print` или `std::cout`.
>
> **Документация модуля**: [`Doc/Modules/fm_correlator/Full.md`](../../Doc/Modules/fm_correlator/Full.md) -- **ЧИТАТЬ ПЕРВЫМ**, там вся математика, pipeline, ядра, API, буферы.
>
> **Руководство коллеги**: [`Doc/Modules/fm_correlator/FM_Correlator_ROCm_Guide.docx`](../../Doc/Modules/fm_correlator/FM_Correlator_ROCm_Guide.docx) -- детальное описание hipFFT планов, ядер, оптимизаций, rocFFT callbacks.
>
> **Оптимизация HIP/ROCm**: [`Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`](../../Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) -- coalesced access, LDS, block size, profiling.
>
> **Статус**: COMPLETED ✅ (2026-03-03)

---

## ПРАВИЛА

- Вывод профилирования -- **ТОЛЬКО** через `GPUProfiler`: `PrintReport()`, `ExportMarkdown()`, `ExportJSON()`. **ЗАПРЕЩЕНО** `GetStats()` + цикл + `con.Print`.
- Перед `profiler.Start()` -- обязательно `profiler.SetGPUInfo(...)`. См. [`Examples/GPUProfiler_SetGPUInfo.md`](../../Examples/GPUProfiler_SetGPUInfo.md).
- Общий вывод -- через `drv_gpu_lib::ConsoleOutput::GetInstance()` (мультиGPU-safe).
- Новые классы -- в отдельных файлах (`.hpp` + `.cpp`).
- **10 GPU параллельно**: никаких `static`, всё через объект.
- **hiprtc kernels**: строка в `.hpp`, компиляция через `hiprtcCompileProgram`, запуск через `hipModuleLaunchKernel`.
- Тесты -- в `.hpp` файлах в каталоге `tests/`.
- **hipFFT не нормирует IFFT** -- деление на N обязательно в `extract_magnitudes`.
- **R2C FFT выдаёт N/2+1 точек** -- все спектральные буферы и ядра используют `half_N = N/2+1`.
- Бенчмарк -- **ТОЛЬКО** через `hipEvent` (GPU hardware timer), **НЕ** через `std::chrono`.

---

## 1. Цель

Реализовать модуль `fm_correlator` -- корреляция в частотной области для сигналов с фазовой модуляцией M-последовательностями. Вся обработка на GPU (ROCm: hipFFT + HIP kernels).

**Алгоритм**: `corr = IFFT{ conj(FFT{ref_shifted}) * FFT{inp} }` для всех пар (signal, shift).

**Входные данные**:
- `ref[N]` -- float, M-последовательность {+1.0, -1.0}
- `inp[S x N]` -- float, вещественные входные сигналы

**Выход**: `peaks[S x K x n_kg]` -- float, магнитуды корреляционных пиков.

---

## 2. Зависимости

- **Task_00_DrvGPU** (ROCmBackend, IBackend, GetNativeQueue, GetProfiler)
- `DrvGPU/services/batch_manager.hpp` (BatchManager для больших S)
- `DrvGPU/services/console_output.hpp` (ConsoleOutput)
- Linux: hip, hipfft, hiprtc (AMD ROCm 7.2+ stack)

---

## 3. Структура файлов (создать)

```
modules/fm_correlator/
├── CMakeLists.txt                              # ROCm-only (skip if !ROCM_ENABLED)
├── include/
│   ├── fm_correlator.hpp                       # Фасад: FMCorrelator класс
│   ├── fm_correlator_types.hpp                 # FMCorrelatorParams, FMCorrelatorResult
│   ├── fm_correlator_processor_rocm.hpp        # ROCm backend (hipFFT + kernels)
│   └── kernels/
│       └── fm_kernels_rocm.hpp                 # HIP kernel source strings для hiprtc
├── src/
│   ├── fm_correlator.cpp                       # Фасад: делегирует в processor
│   └── fm_correlator_processor_rocm.cpp        # ROCm: буферы, планы, ядра, pipeline
├── tests/
│   ├── all_test.hpp                            # Точка входа тестов модуля
│   ├── test_fm_basic.hpp                       # Autocorrelation, shift pattern, basic/full pipeline
│   ├── test_fm_msequence.hpp                   # Тест LFSR генератора
│   ├── test_fm_benchmark_rocm.hpp              # Бенчмарк: warmup 3 + 20 runs + parametric
│   └── README.md                               # Описание тестов

python/
├── py_fm_correlator_rocm.hpp                   # pybind11 обёртка PyFMCorrelatorROCm

Python_test/fm_correlator/
├── test_fm_correlator_rocm.py                  # Python тесты (autocorrelation, shift pattern, cpu vs gpu)

Doc/Python/
├── fm_correlator_api.md                        # Документация Python API
```

---

## 4. Референсный код

| Элемент | Файл-образец | Описание |
|---------|-------------|----------|
| ROCm-only модуль (CMake) | `modules/statistics/CMakeLists.txt` | `if(NOT ROCM_ENABLED) return()` |
| hiprtc kernel strings | `modules/heterodyne/include/kernels/heterodyne_kernels_rocm.hpp` | Строки в `.hpp` |
| ROCm processor | `modules/heterodyne/include/processors/heterodyne_processor_rocm.hpp` | hipModule, hipFunction, буферы |
| Бенчмарк warmup+20 runs | `modules/vector_algebra/tests/test_benchmark_symmetrize.hpp` | hipEvent, avg/min/max |
| BatchManager | `DrvGPU/services/batch_manager.hpp` | CalculateOptimalBatchSize, CreateBatches |
| all_test.hpp | `modules/heterodyne/tests/all_test.hpp` | Реестр тестов |
| GPUProfiler | `Examples/GPUProfiler_SetGPUInfo.md` | SetGPUInfo + Start/Stop + PrintReport |
| ConsoleOutput | `DrvGPU/services/console_output.hpp` | con.Print(gpu_id, tag, msg) |
| Python bindings ROCm | `python/py_heterodyne_rocm.hpp` | PyClass, set_params, GIL release |
| Python bindings сборка | `python/gpu_worklib_bindings.cpp` | register_*, vector_to_numpy |
| Python тесты | `Python_test/vector_algebra/test_cholesky_inverter_rocm.py` | pytest, numpy, ROCmGPUContext |

---

## 5. Подробные задачи

### 5.1. FMCorrelatorParams и FMCorrelatorResult (`fm_correlator_types.hpp`)

```cpp
namespace drv_gpu_lib {

struct FMCorrelatorParams {
  size_t fft_size = 32768;       // N, степень 2
  int num_shifts = 32;            // K -- циклические сдвиги
  int num_signals = 5;            // S -- входные сигналы
  int num_output_points = 2000;   // n_kg -- первые точки IFFT
  uint32_t lfsr_polynomial = 0xB8000000;
  uint32_t lfsr_seed = 0x1;
};

struct FMCorrelatorResult {
  std::vector<float> peaks;       // [S * K * n_kg], row-major
  int num_signals;
  int num_shifts;
  int num_output_points;

  float at(int signal, int shift, int point) const {
    return peaks[(signal * num_shifts + shift) * num_output_points + point];
  }
};

}  // namespace drv_gpu_lib
```

### 5.2. HIP-ядра (`fm_kernels_rocm.hpp`)

Три ядра как строки для hiprtc. Обернуть в `#if ENABLE_ROCM`.

**Ядро 1: `apply_cyclic_shifts`** -- float[N] -> float2[K x N] с циклическими сдвигами:
```cpp
// ref[src] -> out[k*N+i], где src = (i+k) % N, imag = 0
// Grid: ((N+255)/256, K), Block: (256)
```

**Ядро 2: `multiply_conj_fused`** -- conj(ref_fft) * inp_fft за один проход:
```cpp
// a = ref_fft[k*half_N+i], a.y = -a.y (conj inline)
// b = inp_fft[s*half_N+i]
// out = (a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)
// half_N = N/2+1 (hermitian!)
// Grid: ((half_N+255)/256, K, S), Block: (256)
```

**Ядро 3: `extract_magnitudes_real`** -- |first n_kg| / N:
```cpp
// fabsf(corr_time[(s*K+k)*N+j]) / (float)N
// Grid: ((n_kg+255)/256, K, S), Block: (256)
```

**Ядро 4 (utility): `generate_test_inputs`** -- GPU-генерация тестовых входных:
```cpp
// ref уже на GPU. Для луча s: inp[s*N+i] = ref[(i + s*shift_step) % N]
// Нет H2D для S*N float -- всё на GPU, Python передаёт только shift_step
// Grid: ((N+255)/256, S), Block: (256)
```

Полный код ядер -- в `Doc/Modules/fm_correlator/Full.md`, раздел 5.

### 5.2.1. M-последовательность: CPU vs GPU

LFSR строго последователен (каждый бит зависит от предыдущего). Параллелизация нецелесообразна.

**Решение**: генерируем M-seq на CPU (~0.1 мс), загружаем на GPU **один раз** через `PrepareReference()`. Тестовые входные генерируются **на GPU** ядром `generate_test_inputs`.

### 5.3. FMCorrelatorProcessorROCm (`fm_correlator_processor_rocm.hpp` + `.cpp`)

Основной класс, инкапсулирует всю GPU-логику.

**Приватные поля:**
```cpp
IBackend* backend_;
FMCorrelatorParams params_;

// HIP streams
hipStream_t stream0_, stream1_;

// hipFFT plans (persistent -- создаются в SetParams, не в Process)
hipfftHandle plan_ref_;    // C2C Forward, batch=K
hipfftHandle plan_inp_;    // R2C Forward, batch=S (или batch=S_batch)
hipfftHandle plan_corr_;   // C2R Inverse, batch=S*K (или S_batch*K)

// GPU buffers
float*  d_ref_float_;      // [N]
float2* d_ref_complex_;    // [K * N] -- ref после shifts, in-place с d_ref_fft
float2* d_ref_fft_;        // = d_ref_complex_ (in-place FFT)
float*  d_inp_float_;      // [S * N]
float2* d_inp_fft_;        // [S * (N/2+1)]
float2* d_corr_fft_;       // [S * K * (N/2+1)]
float*  d_corr_time_;      // [S * K * N]
float*  d_peaks_;          // [S * K * n_kg]

// hiprtc compiled kernels
hipModule_t module_;
hipFunction_t fn_apply_shifts_;
hipFunction_t fn_multiply_conj_;
hipFunction_t fn_extract_mag_;
hipFunction_t fn_generate_test_inputs_;  // utility для тестового паттерна

bool ref_prepared_ = false;
```

**Методы:**
```cpp
void SetParams(const FMCorrelatorParams& params);
  // 1. Сохранить params_
  // 2. CompileKernels() -- hiprtc
  // 3. AllocateBuffers() -- hipMalloc для всех 8 буферов
  // 4. CreatePlans() -- hipfftPlanMany для plan_ref_, plan_inp_, plan_corr_
  // 5. hipfftSetStream для каждого плана

void PrepareReference(const std::vector<float>& ref);
  // 1. H2D: hipMemcpyAsync(d_ref_float_, ref.data(), stream0_)
  // 2. Kernel: apply_cyclic_shifts (d_ref_float_ -> d_ref_complex_)
  // 3. hipfftExecC2C(plan_ref_, d_ref_complex_, d_ref_fft_, HIPFFT_FORWARD)
  // 4. hipStreamSynchronize(stream0_)
  // 5. ref_prepared_ = true

FMCorrelatorResult Process(const std::vector<float>& inp);
  // ASSERT: ref_prepared_ == true
  // 1. H2D: hipMemcpyAsync(d_inp_float_, inp.data(), stream1_)
  // 2. hipfftExecR2C(plan_inp_, d_inp_float_, d_inp_fft_) на stream1_
  // 3. hipStreamSynchronize(stream0_); hipStreamSynchronize(stream1_);
  // 4. Kernel: multiply_conj_fused (d_ref_fft_, d_inp_fft_ -> d_corr_fft_) на stream0_
  // 5. hipfftExecC2R(plan_corr_, d_corr_fft_, d_corr_time_) на stream0_
  // 6. Kernel: extract_magnitudes_real (d_corr_time_ -> d_peaks_) на stream0_
  // 7. D2H: hipMemcpyAsync(result, d_peaks_, stream0_)
  // 8. hipStreamSynchronize(stream0_)
  // 9. Вернуть FMCorrelatorResult

FMCorrelatorResult RunTestPattern(int shift_step);
  // 1. ASSERT: ref_prepared_ == true (d_ref_float_ содержит M-seq)
  // 2. Kernel: generate_test_inputs(d_ref_float_ -> d_inp_float_, shift_step) на stream1_
  //    Grid: ((N+255)/256, S), Block: (256)
  //    Генерирует S сдвинутых копий ref ПРЯМО НА GPU -- нет H2D для S*N float!
  // 3. hipfftExecR2C(plan_inp_, d_inp_float_, d_inp_fft_) на stream1_
  // 4. hipStreamSynchronize(stream0_); hipStreamSynchronize(stream1_);
  // 5. Далее как Process(): multiply -> C2R -> extract -> D2H

FMCorrelatorResult ProcessWithBatching(const std::vector<float>& inp, int total_signals);
  // Для S > batch_size:
  // 1. BatchManager::CalculateOptimalBatchSize()
  // 2. BatchManager::CreateBatches()
  // 3. Для каждого batch: пересоздать plan_inp_ и plan_corr_ если batch.count != текущий S
  // 4. ProcessBatch() для каждого батча
  // 5. Объединить peaks

~FMCorrelatorProcessorROCm();
  // hipfftDestroy(plan_ref_), hipfftDestroy(plan_inp_), hipfftDestroy(plan_corr_)
  // hipFree для всех 8 буферов
  // hipModuleUnload(module_)
  // hipStreamDestroy(stream0_), hipStreamDestroy(stream1_)
```

### 5.4. FMCorrelator -- фасад (`fm_correlator.hpp` + `.cpp`)

```cpp
class FMCorrelator {
  std::unique_ptr<FMCorrelatorProcessorROCm> processor_;
  FMCorrelatorParams params_;
public:
  explicit FMCorrelator(IBackend* backend);
  void SetParams(const FMCorrelatorParams& params);

  std::vector<float> GenerateMSequence() const;
  std::vector<float> GenerateMSequence(uint32_t seed) const;

  void PrepareReference(const std::vector<float>& ref);   // внешний ref
  void PrepareReference();   // внутренний M-seq generator (seed из params)

  FMCorrelatorResult Process(const std::vector<float>& inp);

  // Тестовый паттерн: GPU-генерация входных через circshift(ref, s*shift_step)
  // Python передаёт ТОЛЬКО shift_step, данные не покидают GPU
  FMCorrelatorResult RunTestPattern(int shift_step = 2);

  // Step1/Step2/Step3 для отладки
};
```

M-sequence генератор (CPU):
```cpp
std::vector<float> FMCorrelator::GenerateMSequence(uint32_t seed) const {
  std::vector<float> seq(params_.fft_size);
  uint32_t lfsr = seed;
  for (size_t i = 0; i < params_.fft_size; ++i) {
    int bit = (lfsr >> 31) & 1;
    seq[i] = bit ? 1.0f : -1.0f;
    lfsr = bit ? ((lfsr << 1) ^ params_.lfsr_polynomial) : (lfsr << 1);
  }
  return seq;
}
```

### 5.5. CMakeLists.txt

По образцу `modules/statistics/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.21)
set(MODULE_NAME fm_correlator)

if(NOT ROCM_ENABLED)
    message(STATUS "fm_correlator: skipped (ROCm disabled)")
    return()
endif()

find_package(hipfft REQUIRED)

add_library(${MODULE_NAME} STATIC
    src/fm_correlator.cpp
    src/fm_correlator_processor_rocm.cpp
)

target_include_directories(${MODULE_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(${MODULE_NAME} PUBLIC
    drvgpu
    hip::host
    hip::hipfft
)

target_compile_definitions(${MODULE_NAME} PUBLIC ENABLE_ROCM=1)
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

add_library(GPUWorkLib::${MODULE_NAME} ALIAS ${MODULE_NAME})
```

### 5.6. Интеграция в проект

1. **`CMakeLists.txt` (корень)** -- добавить:
```cmake
add_subdirectory(modules/fm_correlator)
```

2. **`src/CMakeLists.txt`** -- добавить линковку:
```cmake
if(TARGET GPUWorkLib::fm_correlator)
    target_link_libraries(GPUWorkLib PRIVATE GPUWorkLib::fm_correlator)
endif()
```
И include:
```cmake
target_include_directories(GPUWorkLib PRIVATE
    ${CMAKE_SOURCE_DIR}/modules/fm_correlator/include
)
```

3. **`src/main.cpp`** -- добавить:
```cpp
#if ENABLE_ROCM
#include "modules/fm_correlator/tests/all_test.hpp"
#endif

// В run_module():
#if ENABLE_ROCM
if (n == "fm_correlator") { fm_correlator_all_test::run(); return true; }
#endif

// В kDefaultOrder:
"fm_correlator"
```

### 5.7. Тесты

#### `all_test.hpp`
```cpp
#pragma once
#if ENABLE_ROCM
#include "test_fm_msequence.hpp"
#include "test_fm_basic.hpp"
#include "test_fm_benchmark_rocm.hpp"

namespace fm_correlator_all_test {
inline void run() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  if (!con.IsRunning()) con.Start();
  int gpu_id = 0;
  con.Print(gpu_id, "FM_Corr", "===== FM Correlator Tests =====");

  fm_correlator::tests::run_test_msequence();
  fm_correlator::tests::run_test_autocorrelation();
  fm_correlator::tests::run_test_shift_pattern();      // сдвиговый паттерн CPU vs GPU
  fm_correlator::tests::run_test_basic_pipeline();
  fm_correlator::tests::run_test_full_pipeline();
  // fm_correlator::tests::run_benchmark();          // раскомментировать при необходимости
  // fm_correlator::tests::run_parametric_benchmark(); // раскомментировать при необходимости
}
}
#endif
```

#### `test_fm_msequence.hpp` -- Тест генератора

- Генерация M-seq длиной 32768
- Проверка: все значения {+1.0, -1.0}
- Проверка: примерно 50% единиц и 50% минус единиц
- Проверка: две разные seed дают разные последовательности
- Проверка: один и тот же seed даёт одинаковую последовательность

#### `test_fm_basic.hpp` -- Функциональные тесты

**Тест 1: Автокорреляция** (самый важный -- верификация корректности):
```
ref = GenerateMSequence(seed=1)
inp = ref  (тот же сигнал, S=1)
peaks = Process(inp)
Проверка: peaks[0][0][0] >> peaks[0][0][j>0]  (SNR > 10)
```

**Тест 2: Basic pipeline** (малые данные):
```
N=1024, K=4, S=2, n_kg=100
Проверка: pipeline отрабатывает без ошибок, размеры результата верны
```

**Тест 3: Full pipeline** (параметры по умолчанию):
```
N=32768, K=32, S=5, n_kg=2000
Проверка: pipeline отрабатывает, результат S*K*n_kg = 320000 float
```

**Тест 4: Сдвиговый паттерн (основной тест корректности)**:
```
N=4096, K=10, S=5, n_kg=200, shift_step=2

Алгоритм:
  ref = GenerateMSequence()
  PrepareReference()  // internal gen + upload
  Для каждого луча s: inp[s] = circshift(ref, s * shift_step)
  → пик для (signal=s, shift=k) в позиции (s*shift_step - k) mod N

Вариант A (CPU): генерируем inp циклом, Process(inp)
Вариант B (GPU): RunTestPattern(shift_step) -- generate_test_inputs kernel

Проверка:
  1. Для каждой пары (s, k) пик в ожидаемой позиции
  2. Результаты CPU и GPU совпадают (atol=1e-4)
```

**Тест 5: BatchManager** (большие данные):
```
N=32768, K=32, S=100, n_kg=2000  -- должно автоматически разбиться на батчи
Проверка: результат собран корректно, размер S*K*n_kg
```

#### `test_fm_benchmark_rocm.hpp` -- Бенчмарк

**Методика** (как `modules/vector_algebra/tests/test_benchmark_symmetrize.hpp`):

```
constexpr int kWarmupRuns = 3;
constexpr int kBenchmarkRuns = 20;
```

1. **Warmup**: 3 итерации Process() + `hipDeviceSynchronize()` после каждой
2. **Measurement**: 20 итераций, `hipEventRecord`/`hipEventElapsedTime` (GPU hardware timer)
3. **Метрики**: avg_ms, min_ms, max_ms
4. **Вывод**: через `ConsoleOutput`, формат таблицы
5. **Профилирование**: `GPUProfiler` (SetGPUInfo, Start, Stop, PrintReport, ExportMarkdown, ExportJSON)
6. **Результаты**: `Results/Profiler/fm_correlator_YYYY-MM-DD.md` и `.json`

**Параметрический прогон** (функция `RunParametricBenchmark()`):

Запускать **после** отладки базового бенчмарка.

| Параметр | Значения |
|----------|----------|
| fft_size | 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 |
| num_shifts | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 |
| num_signals | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 |

Для каждой конфигурации: warmup 3 + 20 замеров, вывод avg/min/max.
Результат: `Results/Profiler/fm_correlator_parametric_YYYY-MM-DD.md`

### 5.8. Python Bindings (`py_fm_correlator_rocm.hpp`)

По образцу `python/py_heterodyne_rocm.hpp`.

**Класс**: `PyFMCorrelatorROCm`
**Конструктор**: `explicit PyFMCorrelatorROCm(ROCmGPUContext& ctx)`

**Методы**:

| Метод | Вход | Выход | Описание |
|-------|------|-------|----------|
| `set_params(...)` | fft_size, num_shifts, num_signals, num_output_points, polynomial, seed | void | Установка параметров |
| `generate_msequence(seed)` | uint32_t | numpy [N] float | M-seq для анализа в Python |
| `prepare_reference()` | -- | void | Внутренняя генерация + upload |
| `prepare_reference_from_data(ref)` | numpy [N] float | void | Внешний ref |
| `process(input_signals)` | numpy [S, N] float | numpy [S, K, n_kg] float | Корреляция с данными |
| `run_test_pattern(shift_step)` | int | numpy [S, K, n_kg] float | Тестовый паттерн (GPU-only) |

**Ключевые паттерны**:
- GIL release: `py::gil_scoped_release release;` перед GPU-вычислениями
- Входы: `py::array_t<float, py::array::c_style | py::array::forcecast>`
- Выходы: `vector_to_numpy_3d()` для peaks [S, K, n_kg]

**Регистрация**: добавить `register_fm_correlator_rocm(m)` в `gpu_worklib_bindings.cpp`, обернуть в `#if ENABLE_ROCM`.

### 5.9. Python тесты (`Python_test/fm_correlator/test_fm_correlator_rocm.py`)

| Тест | Описание |
|------|----------|
| `test_autocorrelation` | ref vs ref, SNR > 10 |
| `test_shift_pattern` | run_test_pattern(shift_step=2), проверка позиций пиков |
| `test_cpu_vs_gpu_pattern` | numpy circshift vs run_test_pattern, atol=1e-4 |
| `test_params_only_mode` | Только set_params + prepare_reference + run_test_pattern (данные не передаются) |

### 5.10. Документация Python API (`Doc/Python/fm_correlator_api.md`)

```python
# Создание
ctx = gpuworklib.ROCmGPUContext(0)
corr = gpuworklib.FMCorrelatorROCm(ctx)
corr.set_params(fft_size=32768, num_shifts=32, num_signals=10)

# Режим 1: Только параметры (данные не покидают GPU)
corr.prepare_reference()
peaks = corr.run_test_pattern(shift_step=2)  # numpy [S, K, n_kg]

# Режим 2: Внешние данные
ref = corr.generate_msequence(seed=1)
corr.prepare_reference_from_data(ref)
peaks = corr.process(signals)  # signals: numpy [S, N] float
```

---

## 6. hipFFT планы -- детали

### 6.1. plan_ref -- C2C Forward, batch=K

```cpp
hipfftHandle plan_ref;
int n[1] = { (int)N };
hipfftPlanMany(&plan_ref, 1, n,
    nullptr, 1, N,      // in: float2[K*N]
    nullptr, 1, N,      // out: float2[K*N] (in-place допустимо)
    HIPFFT_C2C, K);
hipfftSetStream(plan_ref, stream0_);
```

### 6.2. plan_inp -- R2C Forward, batch=S

R2C **принимает float напрямую** -- ядро real_to_complex НЕ нужно.

```cpp
hipfftHandle plan_inp;
int n[1] = { (int)N };
hipfftPlanMany(&plan_inp, 1, n,
    nullptr, 1, N,          // in: float[S*N]
    nullptr, 1, N/2 + 1,    // out: float2[S*(N/2+1)]
    HIPFFT_R2C, S);
hipfftSetStream(plan_inp, stream1_);
```

**ВНИМАНИЕ**: выход R2C имеет **N/2+1** комплексных точек! Все буферы и ядра используют `half_N = N/2+1`.

### 6.3. plan_corr -- C2R Inverse, batch=S*K

```cpp
hipfftHandle plan_corr;
int n[1] = { (int)N };
hipfftPlanMany(&plan_corr, 1, n,
    nullptr, 1, N/2 + 1,    // in: float2[S*K*(N/2+1)]
    nullptr, 1, N,           // out: float[S*K*N]
    HIPFFT_C2R, S * K);
hipfftSetStream(plan_corr, stream0_);
```

**ВНИМАНИЕ**: hipFFT **НЕ нормирует IFFT**! Делить на N в `extract_magnitudes`.

---

## 7. GPU буферы -- расчёт памяти

| Буфер | Формула | Пример (N=32768, K=32, S=5) |
|-------|---------|---------------------------|
| `d_ref_float` | N * 4 | 128 КБ |
| `d_ref_complex` (=d_ref_fft) | K * N * 8 | 8 МБ |
| `d_inp_float` | S * N * 4 | 640 КБ |
| `d_inp_fft` | S * (N/2+1) * 8 | 640 КБ |
| `d_corr_fft` | S * K * (N/2+1) * 8 | 20 МБ |
| `d_corr_time` | S * K * N * 4 | 20 МБ |
| `d_peaks` | S * K * n_kg * 4 | 1.2 МБ |
| **ИТОГО** | | **~50 МБ** |

При большом S (>50) с N=131072 может не поместиться -- используем BatchManager.

---

## 8. BatchManager -- разбиение больших данных

Когда `S * K * N * 8` (corr_fft) + `S * K * N * 4` (corr_time) не помещается в GPU-память.

**Стратегия**: разбиваем по входным сигналам (S). Опорный `ref_fft[K*N]` остаётся на GPU.

```cpp
#include "DrvGPU/services/batch_manager.hpp"

size_t per_signal_memory =
    N * sizeof(float)              // d_inp_float (одна строка)
  + (N/2+1) * sizeof(float2)      // d_inp_fft
  + K * (N/2+1) * sizeof(float2)  // d_corr_fft (K корреляций)
  + K * N * sizeof(float)          // d_corr_time
  + K * n_kg * sizeof(float);      // d_peaks

size_t external_mem = K * N * sizeof(float2);  // ref_fft уже на GPU
auto& mgr = drv_gpu_lib::BatchManager::GetInstance();
size_t batch_sz = mgr.CalculateOptimalBatchSize(
    backend_, total_signals, per_signal_memory, 0.7, external_mem);
auto batches = mgr.CreateBatches(total_signals, batch_sz, 3, true);

for (auto& b : batches) {
    // Пересоздать plan_inp_ и plan_corr_ если b.count != текущий batch size
    // ProcessBatch(inp + b.start * N, b.count)
    // Собрать результаты
}
```

---

## 9. Частые ошибки (проверить при реализации)

| Ошибка | Как избежать |
|--------|-------------|
| R2C выдаёт N/2+1 точек, код ожидает N | `half_N = N/2+1` для ВСЕХ спектральных буферов и ядер |
| C2R IFFT не нормирует | `/= (float)N` в `extract_magnitudes` |
| hipfftPlanMany в каждом Process() | Создавать в `SetParams()`, переиспользовать |
| `hipfftSetStream` не вызван | Обязательно после каждого `hipfftPlanMany` |
| Забыт `hipStreamSynchronize` перед Step3 | sync обоих потоков перед `multiply_conj_fused` |
| `d_ref_fft` и `d_ref_complex` -- разные указатели | Один буфер, in-place FFT |
| `multiply_conj_fused` не делает conj | `a.y = -a.y` обязательно перед умножением |
| `extract_magnitudes` берёт N точек вместо n_kg | Проверка `j >= n_kg` в ядре |
| Нет граничной проверки в ядрах | `if (i >= N) return;` в каждом ядре |

---

## 10. Чек-лист

- [ ] **Структура каталогов** `modules/fm_correlator/{include,src,tests}` создана
- [ ] **fm_correlator_types.hpp** -- FMCorrelatorParams, FMCorrelatorResult
- [ ] **fm_kernels_rocm.hpp** -- 4 ядра как строки:
  - [ ] apply_cyclic_shifts (float -> float2 + shifts)
  - [ ] multiply_conj_fused (conj inline + multiply, half_N = N/2+1)
  - [ ] extract_magnitudes_real (|first n_kg| / N)
  - [ ] generate_test_inputs (utility: circshift(ref, s*shift_step) на GPU)
- [ ] **fm_correlator_processor_rocm.hpp/.cpp** -- ROCm backend:
  - [ ] CompileKernels() -- hiprtc (4 ядра)
  - [ ] AllocateBuffers() -- hipMalloc для 8 буферов
  - [ ] CreatePlans() -- hipfftPlanMany (C2C, R2C, C2R)
  - [ ] hipfftSetStream() для каждого плана
  - [ ] hipStreamCreate() для stream0_ и stream1_
  - [ ] PrepareReference(vector) -- H2D + shifts + FFT
  - [ ] PrepareReference() -- internal M-seq gen + upload
  - [ ] Process() -- H2D + R2C + sync + multiply + C2R + extract + D2H
  - [ ] RunTestPattern(shift_step) -- generate_test_inputs на GPU + R2C + correlation
  - [ ] ProcessWithBatching() -- BatchManager
  - [ ] Деструктор: hipfftDestroy + hipFree + hipModuleUnload + hipStreamDestroy
- [ ] **fm_correlator.hpp/.cpp** -- фасад:
  - [ ] GenerateMSequence() -- LFSR на CPU, возвращает float
  - [ ] SetParams(), PrepareReference(2 варианта), Process(), RunTestPattern()
- [ ] **CMakeLists.txt** -- ROCm-only, find_package(hipfft)
- [ ] **Интеграция в проект** -- корневой CMake, src/CMakeLists.txt, src/main.cpp
- [ ] **all_test.hpp** -- реестр тестов
- [ ] **test_fm_msequence.hpp** -- тест LFSR генератора
- [ ] **test_fm_basic.hpp**:
  - [ ] autocorrelation (SNR>10)
  - [ ] shift_pattern: CPU circshift vs GPU generate_test_inputs (пики в ожидаемых позициях)
  - [ ] basic pipeline (N=1024)
  - [ ] full pipeline (N=32768)
  - [ ] batch (S=100)
- [ ] **test_fm_benchmark_rocm.hpp**:
  - [ ] Бенчмарк: warmup 3 + 20 runs, hipEvent, avg/min/max
  - [ ] GPUProfiler: SetGPUInfo + Start/Stop + PrintReport + ExportMarkdown + ExportJSON
  - [ ] Параметрический прогон: fft 2^10..2^17, shifts 5..60, signals 5..50
- [ ] **Python bindings**:
  - [ ] py_fm_correlator_rocm.hpp -- PyFMCorrelatorROCm
  - [ ] Регистрация в gpu_worklib_bindings.cpp
  - [ ] set_params, prepare_reference, process, run_test_pattern, generate_msequence
- [ ] **Python тесты** (`Python_test/fm_correlator/test_fm_correlator_rocm.py`):
  - [ ] test_autocorrelation
  - [ ] test_shift_pattern (run_test_pattern, проверка позиций пиков)
  - [ ] test_cpu_vs_gpu_pattern (numpy circshift vs GPU, atol=1e-4)
- [ ] **Doc/Python/fm_correlator_api.md** -- документация Python API
- [ ] **tests/README.md** -- описание тестов
- [ ] **Компиляция без ошибок**
- [ ] **C++ тесты проходят**
- [ ] **Python тесты проходят**

---

## 11. Ссылки

| Документ | Путь |
|----------|------|
| Полная документация модуля | `Doc/Modules/fm_correlator/Full.md` |
| Руководство коллеги | `Doc/Modules/fm_correlator/FM_Correlator_ROCm_Guide.docx` |
| ROCm/HIP оптимизация | `Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md` |
| GPUProfiler пример | `Examples/GPUProfiler_SetGPUInfo.md` |
| BatchManager API | `DrvGPU/services/batch_manager.hpp` |
| Исходный проект | `/home/alex/C++/Correlator/` |
| Python референс | `/home/alex/C++/Correlator/Doc/Python_Examples/pyfftw_implementation.py` |
| Образец ROCm-only модуля | `modules/statistics/CMakeLists.txt` |
| Образец hiprtc kernels | `modules/heterodyne/include/kernels/heterodyne_kernels_rocm.hpp` |
| Образец бенчмарка | `modules/vector_algebra/tests/test_benchmark_symmetrize.hpp` |
| Python bindings ROCm | `python/py_heterodyne_rocm.hpp` |
| Python bindings сборка | `python/gpu_worklib_bindings.cpp` |
| Python тесты образец | `Python_test/vector_algebra/test_cholesky_inverter_rocm.py` |
| Правила проекта | `CLAUDE.md` |
