# Task_13 — Strategies Pipeline: fft_func + vector<float> API + Parallel

> **Памятка для ИИ**: ROCm-only. Код для Linux (Debian с Radeon). Всё под `#if ENABLE_ROCM`.
> **План-источник**: [MemoryBank/DiscussionPlan/StrategiesPipeline/PLAN.md](../DiscussionPlan/StrategiesPipeline/PLAN.md)
> **Проверяющий**: Другой AI будет верифицировать реализацию по этому Task.
> **Инструкция для проверки**: [MemoryBank/INSTRUCTION_StrategiesPipeline.md](../INSTRUCTION_StrategiesPipeline.md)

---

## ПРАВИЛА

- Вывод — через `ConsoleOutput::GetInstance()` (мультиGPU-safe).
- GPUProfiler: только `PrintReport()`, `ExportMarkdown()`, `ExportJSON()`. Запрещено `GetStats()` + цикл + con.Print.
- Перед `profiler.Start()` — `profiler.SetGPUInfo(...)`.
- Новые классы — в отдельных файлах.
- ROCmBackend — **класс** в `DrvGPU/backends/rocm/`, не отдельный модуль.

---

## 1. Цель

Скорректировать pipeline модуля **strategies**:

1. После Hamming + FFT вызывать **fft_func** для получения магнитуд (без фазы), без лишних аллокаций.
2. Параллельно запускать: Statistics, Median, OneMaxParabola, AllMaxima, GlobalMinMax.
3. Добавить CPU API с входом `vector<float>` для всех post-FFT сценариев (как в Statistics).
4. Поддержка `hipMallocManaged` в DrvGPU для отладки (CPU читает без D2H).
5. Бенчмарк: 1 stream vs 3 streams — выбрать лучший вариант по времени.

---

## 2. Зависимости

- DrvGPU (ROCmBackend, IBackend)
- fft_func (ComplexToMagPhaseROCm, AllMaximaPipelineROCm)
- statistics (StatisticsProcessor, ComputeStatisticsFloat, ComputeMedianFloat)
- strategies (AntennaProcessor_v1, one_max_kernel_, minmax_kernel_)

---

## 3. Подробные задачи

### 3.1. ProcessMagnitudeToBuffer в fft_func

**Файлы**: `modules/fft_func/include/complex_to_mag_phase_rocm.hpp`, `modules/fft_func/src/complex_to_mag_phase_rocm.cpp`

**Добавить метод** (писать в буфер вызывающей стороны, без hipMalloc):

```cpp
// .hpp (после ProcessMagnitudeToGPU):
/**
 * @brief Convert GPU complex data to magnitude only, write to caller's buffer
 *
 * Zero allocations. Reads from gpu_complex_in, writes to gpu_magnitude_out.
 * Used by strategies to fill d_magnitudes_ directly after FFT.
 *
 * @param gpu_complex_in  Device pointer to complex data [beam_count * n_point]
 * @param gpu_magnitude_out  Device pointer to float output [beam_count * n_point]
 * @param params  MagPhaseParams (beam_count, n_point, norm_coeff)
 */
void ProcessMagnitudeToBuffer(void* gpu_complex_in, void* gpu_magnitude_out,
    const MagPhaseParams& params);
```

**Реализация** (аналогично ProcessMagnitudeToGPU, но без hipMalloc):

```cpp
void ComplexToMagPhaseROCm::ProcessMagnitudeToBuffer(
    void* gpu_complex_in, void* gpu_magnitude_out,
    const MagPhaseParams& params)
{
    if (!gpu_complex_in || !gpu_magnitude_out) {
        throw std::invalid_argument("ProcessMagnitudeToBuffer: null pointer");
    }
    n_point_ = params.n_point;
    if (!magnitude_kernel_compiled_) { CompileMagnitudeKernel(); }

    float inv_n = 1.0f;
    if (params.norm_coeff < 0.0f)
        inv_n = (params.n_point > 0) ? 1.0f / static_cast<float>(params.n_point) : 1.0f;
    else if (params.norm_coeff > 0.0f)
        inv_n = params.norm_coeff;

    size_t total = static_cast<size_t>(params.beam_count) * params.n_point;
    ExecuteMagnitudeKernel(gpu_complex_in, gpu_magnitude_out, total, inv_n);
    hipStreamSynchronize(stream_);
}
```

**Тест**: добавить в `modules/fft_func/tests/test_process_magnitude_rocm.hpp` тест `TestProcessMagnitudeToBuffer` — сравнение с ProcessMagnitudeToGPU + hipMemcpy.

---

### 3.2. strategies: заменить magnitudes_kernel_ на ProcessMagnitudeToBuffer

**Файл**: `modules/strategies/src/antenna_processor_v1.cpp`

**Изменения в конструкторе**:

- Создать и хранить `ComplexToMagPhaseROCm` (через `backend_`).
- Убедиться, что `MagPhaseParams` настроены: `beam_count = n_ant`, `n_point = nFFT_`, `norm_coeff = 0` (без нормы) или `-1` (÷n_point) — согласовать с текущим magnitudes_kernel_.

**Изменения в do_window_fft()**:

Убрать блок 4 (magnitudes_kernel_). Заменить на:

```cpp
// 4. Compute magnitudes via fft_func (no extra alloc)
fft_processor::MagPhaseParams mp;
mp.beam_count = n_ant;
mp.n_point = nFFT_;
mp.norm_coeff = 0.0f;  // or -1.0f if divide by nFFT
complex_to_mag_->ProcessMagnitudeToBuffer(d_spectrum_, d_magnitudes_, mp);
```

После этого `d_spectrum_` можно обнулять (hipMemsetAsync) для переиспользования — если требуется по плану.

**Заголовок**: добавить `#include "complex_to_mag_phase_rocm.hpp"`, `#include "types/mag_phase_types.hpp"`.

**Зависимость**: strategies уже линкует fft_func — проверить CMakeLists.txt.

---

### 3.3. Statistics: OneMaxFromFloat, GlobalMinMaxFromFloat, AllMaximaFromMagnitudes

**Место**: `modules/statistics/` — добавить CPU wrappers с входом `vector<float>`.

**Референс**: `ComputeStatisticsFloat` в `statistics_processor.cpp` (строки 1046–1080) — паттерн H2D → вызов GPU API → D2H → hipFree.

#### 3.3.1. GlobalMinMaxFromFloat

**Сигнатура**:

```cpp
std::vector<MinMaxResult> GlobalMinMaxFromFloat(
    const std::vector<float>& mags,
    uint32_t n_ant, uint32_t nFFT, float sample_rate);
```

**Реализация**:

- Проверка `mags.size() == n_ant * nFFT`.
- hipMalloc magnitudes_buf, hipMemcpy H2D.
- Вызов `global_minmax` kernel (сейчас в strategies). Нужно либо:
  - перенести kernel в statistics и вызвать через StatisticsProcessor;
  - либо создать отдельный класс/функцию в statistics, которая получает backend и вызывает kernel.
- hipMemcpy D2H результатов, hipFree.

**Тип MinMaxResult**: сейчас в strategies (`include/result_types.hpp` или `kernels/strategies_kernels_rocm.hpp`). Нужно либо переиспользовать, либо определить в statistics_types.hpp.

#### 3.3.2. OneMaxFromFloat (простой, без параболы)

Если требуется отдельно от OneMaxParabola — уточнить. По плану: OneMaxParabola в fft_func, OneMax/GlobalMinMax в statistics. OneMax может быть подмножеством OneMaxParabola (только index + magnitude).

#### 3.3.3. AllMaximaFromMagnitudes

**Сигнатура**:

```cpp
AllMaximaResult AllMaximaFromMagnitudes(
    const std::vector<float>& mags,
    uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest,
    uint32_t search_start = 1, uint32_t search_end = 0,
    uint32_t max_maxima_per_beam = 1000);
```

**Реализация**:

- hipMalloc magnitudes_gpu, hipMemcpy H2D.
- hipMalloc zeros_gpu для fft_data_gpu (beam_count * nFFT * sizeof(float2)), hipMemset 0.
- Вызов `AllMaximaPipelineROCm::Execute(magnitudes_gpu, zeros_gpu, ...)`.
- Результат: MaxValue с index, magnitude, refined_frequency; real, imag, phase = 0.
- hipFree magnitudes_gpu, hipFree zeros_gpu.

**Зависимость**: statistics должен зависеть от fft_func (AllMaximaPipelineROCm) или вызывать через AllMaximaPipelineROCm из strategies. По плану — в statistics, значит statistics получает AllMaximaPipelineROCm или создаёт свой экземпляр.

---

### 3.4. fft_func: OneMaxParabolaFromFloat

**Место**: `modules/fft_func/` — новый файл или расширение существующего.

**Сигнатура**:

```cpp
std::vector<OneMaxParabolaLite> OneMaxParabolaFromFloat(
    const std::vector<float>& mags,
    uint32_t n_ant, uint32_t nFFT, float sample_rate);
```

**Реализация**:

- hipMalloc mags_gpu, hipMemcpy H2D.
- Буфер zeros для spectrum (one_max_no_phase kernel принимает spectrum, но не использует — передать zeros или малый dummy buffer).
- Вызов one_max_no_phase kernel. Ядро сейчас в strategies — нужно либо:
  - экспортировать kernel из strategies в fft_func;
  - либо вызывать через класс в strategies, который принимает vector<float>.
- hipMemcpy D2H результатов, hipFree.

**OneMaxParabolaLite**: тип из strategies (`include/result_types.hpp`). fft_func должен зависеть от strategies для типа, либо тип переносим в общий заголовок.

**Альтернатива**: Оставить one_max kernel в strategies, создать в strategies обёртку `OneMaxParabolaFromFloat`, которая вызывает свой kernel. По плану OneMaxParabola в fft_func — значит kernel или вызов должна быть в fft_func. Уточнение: kernel остаётся в strategies (он в strategies_kernels_rocm.hpp), fft_func предоставляет только фасад `OneMaxParabolaFromFloat`, который принимает IBackend* и вызывает kernel через strategies? Или kernel копируется в fft_func? План: "OneMaxParabola в fft_func" — логичнее иметь вызов в fft_func. strategies владеет kernel. Вариант: strategies добавляет `OneMaxParabolaFromFloat` как static или free function, принимает backend + mags. Тогда fft_func не нужен для этого — будет в strategies. Перечитываю план: "OneMaxParabola с vector — Место: fft_func". Значит API в fft_func. fft_func тогда должен вызывать kernel. Kernel в strategies. fft_func зависит от strategies? Или kernel переносится в fft_func? Проще: стратегии предоставляют `strategies::OneMaxParabolaFromFloat(backend, mags, ...)` которая внутри вызывает one_max kernel. Но план сказал fft_func. Оставляю в fft_func — fft_func будет зависеть от strategies для kernel, или kernel дублируется. Для простоты: создаём в strategies `OneMaxParabolaFromFloat` — strategies уже имеет kernel. Plan said fft_func. I'll note in the task: "Реализовать в fft_func. Kernel — либо переиспользовать из strategies (fft_func depends on strategies), либо скопировать kernel source в fft_func." Strategies уже зависит от fft_func. Если fft_func зависит от strategies — циклическая зависимость. Вывод: OneMaxParabolaFromFloat реализуем в **strategies** (у него есть kernel), но в плане указан fft_func. Исправлю план при создании: по архитектуре kernel в strategies, значит wrapper логичнее в strategies. Оставлю в Task: "Место: fft_func ИЛИ strategies (kernel в strategies, при циклической зависимости — в strategies)".

Упрощаю: OneMaxParabolaFromFloat в **strategies** (там kernel и result type). План говорит fft_func — возможно имелось в виду "логика параболы". Обновлю task: создаём в strategies, т.к. kernel там.

Нет, план явно: "OneMaxParabola — fft_func". Тогда fft_func вызывает kernel. Kernel нужно откуда-то взять. Варианты: 1) strategies экспортирует функцию LaunchOneMaxKernel(backend, d_mag, d_spectrum, d_out, ...); 2) kernel source копируется в fft_func. 3) fft_func зависит от strategies, strategies не зависит от fft_func — но strategies уже линкует fft_func. Значит strategies -> fft_func. fft_func -> strategies создаёт цикл. Итог: OneMaxParabolaFromFloat в strategies — единственный вариант без цикла.

Запишу в Task: "OneMaxParabolaFromFloat — в strategies (kernel там, тип OneMaxParabolaLite там). План указывал fft_func, но из-за циклической зависимости и наличия kernel в strategies — реализуем в strategies."

---

### 3.5. DrvGPU/backends/rocm: AllocateManaged

**Файлы**: `DrvGPU/backends/rocm/rocm_backend.hpp`, `rocm_backend.cpp`

**Добавить**:

```cpp
// .hpp (после Allocate):
/**
 * @brief Allocate unified memory (hipMallocManaged) for debug/checkpoint
 *
 * CPU can read without explicit hipMemcpy. Use for checkpoint when debug enabled.
 * Caller must Free() when done (hipFree works for managed memory).
 */
void* AllocateManaged(size_t size_bytes) override;
```

**Реализация**:

```cpp
void* ROCmBackend::AllocateManaged(size_t size_bytes) {
  if (!initialized_) return nullptr;
  void* ptr = nullptr;
  hipError_t err = hipMallocManaged(&ptr, size_bytes);
  if (err != hipSuccess) {
    DRVGPU_LOG_ERROR_GPU(device_index_, "ROCmBackend",
      "hipMallocManaged failed: " + std::string(hipGetErrorString(err)));
    return nullptr;
  }
  return ptr;
}
```

**IBackend**: добавить в `i_backend.hpp` виртуальный метод `AllocateManaged` с default реализациией `return nullptr` (OpenCL не поддерживает).

---

### 3.6. strategies: параллельный запуск + бенчмарк streams

**Файл**: `modules/strategies/src/antenna_processor_v1.cpp`, `antenna_processor_v1.hpp`

**Добавить** (опционально для бенчмарка): `stream_debug3a_`, `stream_debug3b_`, `stream_debug3c_`.

**В do_run_post_fft_scenarios**:

Вариант 1 (1 stream): оставить как есть — все на stream_debug3_.

Вариант 2 (3 streams): запустить OneMax, AllMaxima, GlobalMinMax на трёх разных stream'ах, затем hipStreamSynchronize для каждого.

**Бенчмарк**: конфигурация (через config или #define) — RUN_ONE_STREAM vs RUN_THREE_STREAMS. Замерить время process(), вывести через ConsoleOutput. Файл бенчмарка: `modules/strategies/tests/test_strategies_benchmark_streams.hpp`.

---

## 4. Чек-лист реализации

- [ ] **fft_func**: ProcessMagnitudeToBuffer в .hpp и .cpp
- [ ] **fft_func**: тест TestProcessMagnitudeToBuffer
- [ ] **strategies**: ComplexToMagPhaseROCm в конструкторе, do_window_fft вызывает ProcessMagnitudeToBuffer вместо magnitudes_kernel_
- [ ] **strategies**: убрать или закомментировать magnitudes_kernel_ launch
- [ ] **statistics**: GlobalMinMaxFromFloat (или в strategies если kernel там)
- [ ] **statistics**: AllMaximaFromMagnitudes
- [ ] **strategies**: OneMaxParabolaFromFloat (kernel в strategies)
- [ ] **DrvGPU**: AllocateManaged в ROCmBackend + IBackend
- [ ] **strategies**: добавлены stream_debug3a/b/c (опционально)
- [ ] **strategies**: бенчмарк 1 vs 3 streams
- [ ] Все тесты проходят
- [ ] Компиляция без ошибок

---

## 5. Частые ошибки

| Ошибка | Решение |
|--------|---------|
| ProcessMagnitudeToBuffer делает hipMalloc | Писать только в gpu_magnitude_out |
| strategies не видит ComplexToMagPhaseROCm | Добавить include, линковка fft_func |
| AllMaxima требует fft_data | Передать zeros_gpu (hipMalloc + hipMemset 0) |
| OneMax kernel ожидает spectrum | Передать zeros (kernel не читает) |
| Циклическая зависимость fft_func↔strategies | OneMaxParabolaFromFloat в strategies |
| AllocateManaged в OpenCLBackend | Default return nullptr в IBackend |

---

## 6. Ссылки

| Документ | Путь |
|----------|------|
| План | MemoryBank/DiscussionPlan/StrategiesPipeline/PLAN.md |
| Инструкция проверки | MemoryBank/INSTRUCTION_StrategiesPipeline.md |
| strategies Full | Doc/Modules/strategies/Full.md |
| fft_func complex_to_mag | modules/fft_func/include/complex_to_mag_phase_rocm.hpp |
| statistics ComputeStatisticsFloat | modules/statistics/src/statistics_processor.cpp:1046 |
| rocm_backend Allocate | DrvGPU/backends/rocm/rocm_backend.cpp:318 |
| antenna_processor_v1 do_window_fft | modules/strategies/src/antenna_processor_v1.cpp:443 |
