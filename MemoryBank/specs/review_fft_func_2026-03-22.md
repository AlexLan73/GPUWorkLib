# 🔍 Code Review: modules/fft_func

> **Дата**: 2026-03-22
> **Ревьюер**: Кодо (AI Assistant)
> **Объём**: ~45 файлов (.hpp/.cpp/.hip/.cl)
> **Методы анализа**: sequential-thinking, context7 (hipFFT/ROCm), web search (hipFFT batch planning)
> **Ветка**: main (Linux, AMD GPU, ROCm 7.2+)

## ✅ ИСПРАВЛЕНО В ЭТОЙ СЕССИИ (2026-03-22)

| # | Тип | Описание | Файлы |
|---|-----|----------|-------|
| 1 | 🔴→✅ | hipfftExecC2C — добавлена проверка result в обоих GPU-path | `fft_processor_rocm.cpp` |
| 2 | 🔴→✅ | sqrtf → `__fsqrt_rn`, atan2f → `__atan2f` в runtime kernel sources | `fft_processor_kernels_rocm.hpp`, `complex_to_mag_phase_kernels_rocm.hpp` |
| 3 | 🔴→✅ | warp_size: строковая эвристика → `ROCmCore::GetWarpSize()` | `complex_to_mag_phase_rocm.cpp`, `spectrum_processor_rocm.cpp`, `all_maxima_pipeline_rocm.cpp` |
| 4 | 🟡→✅ | C2MP мигрирован на Ref03: GpuContext + BufferSet<3> + MagPhaseOp + MagnitudeOp | `complex_to_mag_phase_rocm.hpp/cpp`, новые: `magnitude_op.hpp`, `complex_to_mag_phase_kernels_rocm.hpp` (GetCombinedC2MPKernelSource) |
| 6 | 🟡→✅ | MakeROCmDataFromEvents → shared `utils/rocm_profiling_helpers.hpp` | `fft_processor_rocm.cpp`, `spectrum_processor_rocm.cpp`, новый: `utils/rocm_profiling_helpers.hpp` |
| 9 | 🟡→✅ | In-place FFT: kFftInput + kFftOutput → единый kFftBuf (экономия batch×nFFT×8 bytes) | `fft_processor_rocm.hpp`, `fft_processor_rocm.cpp` |
| 5 | 🟡→✅ | SpectrumProcessorROCm → GpuContext + 3 Ops (SpectrumPadOp, ComputeMagnitudesOp, SpectrumPostOp). Execute* → Op.Execute(): ~120 строк → ~12 | `spectrum_processor_rocm.hpp/cpp`, новые: `spectrum_pad_op.hpp`, `compute_magnitudes_op.hpp`, `spectrum_post_op.hpp` |

---

## 📊 Общая оценка

| Аспект | Оценка | Комментарий |
|--------|--------|-------------|
| Архитектура | ⭐⭐⭐⭐ | FFTProcessorROCm = Ref03 ✅, но C2MP и Spectrum = legacy |
| hipFFT интеграция | ⭐⭐⭐⭐ | Корректная, LRU-2 кеш планов, stream binding |
| Kernels | ⭐⭐⭐⭐ | Хорошие: __launch_bounds__, __restrict__, 2D grid для pad |
| Batch processing | ⭐⭐⭐⭐⭐ | BatchManager правильно интегрирован везде |
| Memory management | ⭐⭐⭐⭐ | BufferSet в FFTProc, ручной hipMalloc в C2MP/Spectrum |
| Error handling | ⭐⭐⭐ | Пропущена проверка hipfftExecC2C в GPU-path |
| Profiling | ⭐⭐⭐⭐⭐ | ROCmProfEvents + GPUProfiler корректно |
| Документация | ⭐⭐⭐⭐⭐ | Отличные комментарии, описание архитектуры |
| Тесты | ⭐⭐⭐⭐ | 11 тестовых файлов + benchmarks |

---

## 📐 Архитектура модуля

Модуль содержит **3 класса-процессора** с **разным уровнем модернизации**:

| Класс | Архитектура | Namespace | Компиляция kernels |
|-------|-------------|-----------|-------------------|
| `FFTProcessorROCm` | ✅ **Ref03** (GpuContext + Ops) | `fft_processor` | Через `GpuContext::CompileModule()` |
| `ComplexToMagPhaseROCm` | ❌ Legacy (raw hipMalloc/hiprtc) | `fft_processor` | Ручной hiprtc + disk cache |
| `SpectrumProcessorROCm` | ❌ Legacy (raw hipMalloc/hiprtc) | `antenna_fft` | Ручной hiprtc + disk cache |

---

## 🔴 Критические проблемы (3)

### 1. hipfftExecC2C — return code не проверяется в GPU-input path

**Файлы**:
- `src/fft_processor_rocm.cpp:425-428` — ProcessComplex GPU overload
- `src/fft_processor_rocm.cpp:556-559` — ProcessMagPhase GPU overload

```cpp
// ❌ GPU path — ошибка МОЛЧА ИГНОРИРУЕТСЯ:
hipfftExecC2C(plan_, ..., HIPFFT_FORWARD);
// Нет проверки result!

// ✅ CPU path (строка 363-371) — ошибка ПРОВЕРЯЕТСЯ:
hipfftResult fft_result = hipfftExecC2C(plan_, ..., HIPFFT_FORWARD);
if (fft_result != HIPFFT_SUCCESS) {
    throw std::runtime_error("hipfftExecC2C failed");
}
```

**Влияние**: Если FFT план не совместим с данными (неправильный размер, misaligned buffers) — ошибка пройдёт незамеченной, результат будет мусором.

**Исправление**: Добавить проверку `hipfftResult` в обоих GPU-path overloads.

---

### 2. Kernel source inconsistency: sqrtf vs __fsqrt_rn

**Файлы**:
- `include/kernels/fft_processor_kernels_rocm.hpp:75-76` (runtime source)
- `kernels/fft_processor_kernels.hip:44-45` (reference .hip file)

| Source | sqrt | atan2 |
|--------|------|-------|
| `GetHIPKernelSource()` (реально компилируется) | `sqrtf()` | `atan2f()` |
| `fft_processor_kernels.hip` (reference, НЕ компилируется) | `__fsqrt_rn()` | `__atan2f()` |
| `GetComplexToMagnitudeKernelSource()` | `__fsqrt_rn()` | — |

**Проблема**: .hip reference файл использует быстрые GPU intrinsics (~4 ULP), а реальный runtime source — стандартные `sqrtf/atan2f`. Комментарии в .hip файле утверждают что используются intrinsics, но runtime код — нет.

**Влияние**: Потеря ~10-20% производительности на mag/phase вычислениях.

**Исправление**: В `GetHIPKernelSource()` заменить `sqrtf` → `__fsqrt_rn`, `atan2f` → `__atan2f`. Синхронизировать .hip reference файл.

---

### 3. ComplexToMagPhaseROCm — warp_size строковая эвристика

**Файл**: `src/complex_to_mag_phase_rocm.cpp:569-572`

```cpp
int warp_size = 32;
if (arch_name.find("gfx9") == 0) warp_size = 64;  // ❌ строковая эвристика
```

Та же проблема что в DrvGPU ревью (#9). Нужно `hipDeviceProp_t.warpSize` (или `ROCmCore::GetWarpSize()`).

Также встречается в `src/complex_to_mag_phase_rocm.cpp:773-774` (CompileMagnitudeKernel).

---

## 🟡 Важные замечания (6)

### 4. ComplexToMagPhaseROCm не использует Ref03

**Файл**: `src/complex_to_mag_phase_rocm.cpp` — ~200 строк ручной hiprtc компиляции

Вся логика `CompileKernels()` (строки 519-625) и `CompileMagnitudeKernel()` (строки 748-813) дублирует то, что `GpuContext::CompileModule()` делает в 1 вызов:
- hiprtcCreateProgram / hiprtcCompileProgram / hiprtcGetCode
- hipModuleLoadData / hipModuleGetFunction
- KernelCacheService load/save
- Определение arch_name и warp_size

Также ручное управление буферами (`hipMalloc/hipFree`) вместо `BufferSet<N>`.

**Рекомендация**: Мигрировать на GpuContext + BufferSet + GpuKernelOp по образцу FFTProcessorROCm.

---

### 5. SpectrumProcessorROCm не использует Ref03

**Файл**: `src/spectrum_processor_rocm.cpp` — ~600 строк, ручной hiprtc + hipMalloc

Аналогичная ситуация: ручные `CompileKernels()`, `CompilePostKernel()`, `AllocateBuffers()`.

**Рекомендация**: Мигрировать поэтапно — сначала GpuContext для компиляции, потом BufferSet для буферов.

---

### 6. MakeROCmDataFromEvents дублирован

**Файлы**:
- `src/fft_processor_rocm.cpp:40-67` (file-static)
- `src/spectrum_processor_rocm.cpp:50-77` (anonymous namespace)

Один и тот же helper скопирован в два файла.

**Рекомендация**: Вынести в `DrvGPU/services/profiling_utils.hpp` или в общий header модуля.

---

### 7. hipfftPlan1d для batch — работает, но hipfftPlanMany рекомендуется

**Файл**: `src/fft_processor_rocm.cpp:196`

```cpp
hipfftResult result = hipfftPlan1d(&plan_, nFFT_, HIPFFT_C2C, batch_beam_count);
```

`hipfftPlan1d` с batch > 1 внутренне вызывает `hipfftPlanMany`. Для контигуозных данных это эквивалентно. Но `hipfftPlanMany` даёт явный контроль over strides/distances — полезно если данные не контигуозны.

**Приоритет**: Низкий — текущий код корректен для контигуозных данных.

---

### 8. Два разных namespace для одного модуля

| Класс | Namespace |
|-------|-----------|
| `FFTProcessorROCm` | `fft_processor` |
| `ComplexToMagPhaseROCm` | `fft_processor` |
| `SpectrumProcessorROCm` | `antenna_fft` |
| `AllMaximaPipelineROCm` | `antenna_fft` |

Два namespace (`fft_processor` и `antenna_fft`) в одном модуле `fft_func`. Создаёт путаницу.

---

### 9. In-place FFT может сэкономить буфер

**Файл**: `src/fft_processor_rocm.cpp` — pipeline: memset → pad → FFT(in→out)

Текущий flow:
```
memset(kFftInput, 0) → pad(kInputBuf → kFftInput) → FFT(kFftInput → kFftOutput)
```

Можно:
```
memset(kFftInput, 0) → pad(kInputBuf → kFftInput) → in-place FFT(kFftInput → kFftInput)
```

Экономит `kFftOutput` буфер (`batch × nFFT × 8 bytes`). `hipfftExecC2C` поддерживает in-place (input == output).

---

## 🟢 Рекомендации (4)

### 10. Kernel design: block_size = 256 — оптимально ✅

256 threads/block = 8 waves (RDNA, wf=32) или 4 waves (CDNA, wf=64). Хороший универсальный выбор. `__launch_bounds__(256)` помогает компилятору оптимизировать register usage.

### 11. Interleaved mag/phase layout — оптимально ✅

`float2_t {mag, phase}` вместо двух отдельных массивов — single DtoH transfer, коалесцентный доступ. Правильное решение.

### 12. PadDataOp 2D grid — оптимально ✅

`blockIdx.y = beam_id` вместо `global_id / nFFT` — eliminates integer division per thread. Combined with pre-zeroed buffer — no divergent else-branch.

### 13. GetComplexToMagPhaseKernelSource — standalone компиляция ✅

Отдельный kernel source для C2MP (без pad_data) — правильное решение. Уменьшает объём компиляции и размер HSACO.

---

## ✅ Соответствие стандартам GPUWorkLib

| Критерий | Статус | Комментарий |
|----------|--------|-------------|
| **DrvGPU интеграция** | ✅ | Все классы принимают IBackend*, stream из backend |
| **Профилирование** | ✅ | ROCmProfEvents + GPUProfiler, GpuBenchmarkBase для benchmarks |
| **Консоль** | ✅ | ConsoleOutput::GetInstance() везде |
| **Стиль кода** | ✅ | Google C++ Style, CamelCase, snake_case |
| **Ref03** | ⚠️ | FFTProcessorROCm = ✅, C2MP и Spectrum = ❌ legacy |
| **Kernel файлы** | ✅ | Отдельные .hip файлы + inline sources для hiprtc |
| **Windows stubs** | ✅ | SpectrumProcessorROCm — корректный stub |
| **BatchManager** | ✅ | Используется во всех Process* методах |
| **Move semantics** | ✅ | Все классы поддерживают move |
| **Disk cache** | ✅ | KernelCacheService для HSACO |

---

## 📚 Источники

### Context7
- **hipFFT** (`/websites/rocm_amd_en`): hipfftPlan1d, hipfftExecC2C, hipfftSetStream, hipfftXtMakePlanMany (half-precision)
- **HIP Runtime** (`/websites/rocm_amd_projects_hip_en`): hipMemcpyHtoDAsync, hipStreamSynchronize, hipModuleLaunchKernel

### Web Search
- [hipFFT API Usage](https://rocm.docs.amd.com/projects/hipFFT/en/latest/api.html) — planning functions
- [hipFFT Overview](https://rocm.docs.amd.com/projects/hipFFT/en/latest/conceptual/overview.html) — architecture and backends
- [rocFFT issue #157](https://github.com/ROCm/rocFFT/issues/157) — hipfftPlanMany batch validation

---

## 📋 Сводка задач

| # | Приоритет | Описание | Файл | Сложность |
|---|-----------|----------|------|-----------|
| 1 | 🔴 | hipfftExecC2C — проверка result в GPU-path | fft_processor_rocm.cpp:425,556 | Низкая |
| 2 | 🔴 | sqrtf → __fsqrt_rn, atan2f → __atan2f в GetHIPKernelSource | fft_processor_kernels_rocm.hpp | Низкая |
| 3 | 🔴 | warp_size строковая эвристика → hipDeviceProp_t.warpSize | complex_to_mag_phase_rocm.cpp:569,773 | Низкая |
| 4 | 🟡 | Мигрировать C2MP на GpuContext + BufferSet | complex_to_mag_phase_rocm.cpp | Высокая |
| 5 | 🟡 | Мигрировать Spectrum на GpuContext + BufferSet | spectrum_processor_rocm.cpp | Высокая |
| 6 | 🟡 | MakeROCmDataFromEvents → общий utility | fft_processor_rocm.cpp, spectrum_processor_rocm.cpp | Низкая |
| 7 | 🟡 | hipfftPlan1d → hipfftPlanMany (опционально) | fft_processor_rocm.cpp:196 | Средняя |
| 8 | 🟡 | Два namespace (fft_processor + antenna_fft) в одном модуле | все файлы | Средняя |
| 9 | 🟡 | In-place FFT для экономии буфера kFftOutput | fft_processor_rocm.cpp | Средняя |

---

*Ревью подготовлено с: sequential-thinking (2 шага), context7 (hipFFT/ROCm), WebSearch (hipFFT batch API)*
