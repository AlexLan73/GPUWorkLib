# TASK: ProcessMagnitude + Statistics pipeline (GPU-only + тесты на SVM)

> **Приоритет**: Высокий
> **Модули**: fft_func, statistics
> **Платформа**: ROCm (main branch)
> **Создано**: 2026-03-11
> **Проверка**: другой AI как старший

---

## 1. Цель

1. Добавить `ProcessMagnitude` / `ProcessMagnitudeToGPU` в `ComplexToMagPhaseROCm` — magnitude-only (без phase), с нормировкой через умножение.
2. Добавить `vector<float>` обёртки в `StatisticsProcessor` для тестов (данные с CPU).
3. **Тесты**: малый объём данных, использовать SVM / `hipMallocManaged` и интерфейс `InputData<T>` — единый подход.

---

## 2. Поток данных

```
Production (большие данные):
  GPU complex (void*) → ProcessMagnitudeToGPU → GPU float (void*)
       → ComputeStatisticsFloat(void*) / ComputeMedianFloat(void*) → CPU results

Тесты (малые данные):
  hipMallocManaged → CPU заполняет → InputData<void*> → ProcessMagnitude / Stats
       → результаты на CPU
```

---

## 3. Часть 1: ComplexToMagPhaseROCm — ProcessMagnitude

### 3.1 mag_phase_types.hpp

```cpp
struct MagPhaseParams {
    uint32_t beam_count   = 1;
    uint32_t n_point     = 0;
    float    memory_limit = 0.80f;
    float    norm_coeff   = 1.0f;  // 0=skip, -1=1/n_point, >0=multiply
};

struct MagnitudeResult {
    uint32_t beam_id = 0;
    uint32_t n_point = 0;
    std::vector<float> magnitude;
};
```

### 3.2 complex_to_mag_phase_rocm.hpp — новые методы

```cpp
// GPU in → CPU out
std::vector<MagnitudeResult> ProcessMagnitude(
    void* gpu_data,
    const MagPhaseParams& params,
    size_t gpu_memory_bytes = 0);

// GPU in → GPU out (caller owns, Free после всех consumers)
void* ProcessMagnitudeToGPU(
    void* gpu_data,
    const MagPhaseParams& params,
    size_t gpu_memory_bytes = 0);
```

### 3.3 Kernel — умножение вместо деления

```c
// inv_n на host: norm_coeff<0 ? 1.0f/n_point : (norm_coeff>0 ? norm_coeff : 1.0f)
output[i] = hypotf(input[i].x, input[i].y) * inv_n;
```

### 3.4 InputData для единообразия

Для вызовов с GPU данными использовать `drv_gpu_lib::InputData<void*>`:
- `antenna_count` = beam_count
- `n_point` = n_point
- `data` = void* (GPU или managed)
- `gpu_memory_bytes` = beam_count × n_point × sizeof(complex<float>) для входа
- Для выхода magnitude: beam_count × n_point × sizeof(float)

Опционально: добавить перегрузки `ProcessMagnitude(InputData<void*> input, MagPhaseParams params)` если нужно единообразие с другими модулями. Иначе — оставить `(void*, params, bytes)` как основной низкоуровневый API.

---

## 4. Часть 2: StatisticsProcessor — vector<float> обёртки

### 4.1 Новые перегрузки

```cpp
std::vector<StatisticsResult> ComputeStatisticsFloat(
    const std::vector<float>& data,
    const StatisticsParams& params);

std::vector<MedianResult> ComputeMedianFloat(
    const std::vector<float>& data,
    const StatisticsParams& params);
```

Реализация: `hipMalloc` → `hipMemcpy` H2D → `ComputeStatisticsFloat(gpu_ptr, params)` → `hipFree` → return.

---

## 5. Тесты: SVM / hipMallocManaged + InputData

### 5.1 Правило для тестов

- Объём данных в тестах **малый** (например, 4 луча × 4096 точек).
- Использовать **hipMallocManaged** (unified memory): CPU и GPU используют один и тот же указатель, явный `hipMemcpy` не нужен.
- Везде единый интерфейс **InputData<T>**.

### 5.2 Хелпер для тестов

Создать в `DrvGPU` или в тестовом коде хелпер (например, `test_helpers_rocm.hpp`):

```cpp
/// Выделяет unified memory для тестов. CPU заполняет, GPU читает. Caller обязан hipFree.
inline void* AllocateManagedForTest(size_t bytes) {
    void* ptr = nullptr;
    hipError_t e = hipMallocManaged(&ptr, bytes);
    if (e != hipSuccess) throw std::runtime_error("hipMallocManaged failed");
    return ptr;
}

/// InputData<void*> для managed буфера (complex)
inline drv_gpu_lib::InputData<void*> MakeManagedInput(
    void* ptr, uint32_t beam_count, uint32_t n_point) {
    drv_gpu_lib::InputData<void*> out;
    out.data = ptr;
    out.antenna_count = beam_count;
    out.n_point = n_point;
    out.gpu_memory_bytes = beam_count * n_point * sizeof(std::complex<float>);
    return out;
}

/// InputData<void*> для managed буфера (float magnitudes)
inline drv_gpu_lib::InputData<void*> MakeManagedMagnitudeInput(
    void* ptr, uint32_t beam_count, uint32_t n_point) {
    drv_gpu_lib::InputData<void*> out;
    out.data = ptr;
    out.antenna_count = beam_count;
    out.n_point = n_point;
    out.gpu_memory_bytes = beam_count * n_point * sizeof(float);
    return out;
}
```

### 5.3 Сценарий теста (C++)

```cpp
// 1. Managed memory
size_t n = 4 * 4096;  // 4 луча × 4096 точек
void* managed = AllocateManagedForTest(n * sizeof(std::complex<float>));

// 2. Заполнение с CPU (без hipMemcpy)
auto* ptr = static_cast<std::complex<float>*>(managed);
FillTestData(ptr, n);  // например синусоида

// 3. InputData
auto input = MakeManagedInput(managed, 4, 4096);
MagPhaseParams params{.beam_count=4, .n_point=4096, .norm_coeff=-1.0f};

// 4. ProcessMagnitude (GPU читает managed напрямую)
auto results = mag_proc.ProcessMagnitude(input.data, params, input.gpu_memory_bytes);

// 5. Освобождение
hipFree(managed);
```

Для статистики по float — аналогично: managed буфер float, заполнение на CPU, вызов `ComputeStatisticsFloat(ptr, params)` или обёртки `ComputeStatisticsFloat(vector<float>)` если данные изначально в `vector`.

---

## 6. Таски для исполнителя (другой AI)

| # | Задача | Файлы | Критерий готовности |
|---|--------|-------|--------------------|
| 1 | Добавить `norm_coeff` и `MagnitudeResult` | `mag_phase_types.hpp` | Компиляция |
| 2 | Объявить `ProcessMagnitude`, `ProcessMagnitudeToGPU` | `complex_to_mag_phase_rocm.hpp` | Компиляция |
| 3 | HIP kernel `complex_to_magnitude` (умножение на inv_n) | `complex_to_mag_phase_rocm.cpp` | Kernel корректен |
| 4 | Реализовать `ProcessMagnitude` и `ProcessMagnitudeToGPU` | `complex_to_mag_phase_rocm.cpp` | Работает на тестовых данных |
| 5 | Объявить `vector<float>` обёртки | `statistics_processor.hpp` | Компиляция |
| 6 | Реализовать обёртки (H2D + void* вызов + Free) | `statistics_processor.cpp` | Работает на тестовых данных |
| 7 | Хелпер `AllocateManagedForTest`, `MakeManagedInput` | `DrvGPU` или `fft_func/tests/` | Используется в тестах |
| 8 | C++ тесты ProcessMagnitude (managed + InputData) | `fft_func/tests/test_process_magnitude_rocm.hpp` | Все тесты проходят |
| 9 | C++ тесты Statistics float (managed, pipeline) | `statistics/tests/` | Pipeline mag→stats работает |
| 10 | Python тесты ProcessMagnitude | `Python_test/fft_func/test_process_magnitude_rocm.py` | pytest проходит |
| 11 | Python тесты Statistics float | `Python_test/statistics/` | pytest проходит |
| 12 | Обновить `all_test.hpp` fft_func и statistics | Соответствующие `all_test.hpp` | Тесты вызываются из main |
| 13 | Обновить plan в `.cursor/plans/` | План | Соответствует этой задаче |

---

## 7. Проверка старшим AI

- [ ] `norm_coeff`: умножение на `inv_n`, без деления в kernel
- [ ] Тесты используют `hipMallocManaged`, не `hipMalloc`+`hipMemcpy` для малых данных
- [ ] Везде единый стиль через `InputData<T>`
- [ ] Нет лишнего копирования в production (данные на GPU)
- [ ] C++ и Python тесты добавлены и проходят
- [ ] Документация обновлена (при необходимости)

---

## 8. Ссылки

- План: `.cursor/plans/processmagnitude_gpu_+_stats_float_*.plan.md`
- Документация statistics: `Doc/Modules/statistics/Full.md`
- ComplexToMagPhaseROCm: `modules/fft_func/include/complex_to_mag_phase_rocm.hpp`
- InputData: `DrvGPU/interface/input_data.hpp`
