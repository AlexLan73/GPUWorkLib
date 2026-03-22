# 🔍 Code Review: lch_farrow + signal_generators + heterodyne

> **Дата**: 2026-03-22
> **Ревьюер**: Кодо (AI Assistant)
> **Объём**: ~80 файлов (3 модуля)
> **Методы анализа**: sequential-thinking, grep pattern analysis
> **Ветка**: main (Linux, AMD GPU, ROCm 7.2+)

## ✅ ИСПРАВЛЕНО В ЭТОЙ СЕССИИ (2026-03-22)

| # | Тип | Описание | Файлы |
|---|-----|----------|-------|
| 1 | 🟡→✅ | MakeROCmDataFromEvents → shared utility (удалены ВСЕ копии в ВСЕХ модулях, включая filters!) | `heterodyne_processor_rocm.cpp`, `lch_farrow_rocm.cpp`, `form_signal_generator_rocm.cpp`, `fir_filter_rocm.cpp`, `iir_filter_rocm.cpp` |
| 2 | 🟡→✅ | heterodyne warp_size → `ROCmCore::GetWarpSize()` | `heterodyne_processor_rocm.cpp` |
| 3 | 🟢→✅ | FormSignalGeneratorROCm → Ref03 (GpuContext). Удалено ~80 строк hiprtc | `form_signal_generator_rocm.hpp/cpp` |
| 4 | 🟢→✅ | LchFarrowROCm → Ref03 (GpuContext). Удалено ~110 строк hiprtc+KernelCacheService | `lch_farrow_rocm.hpp/cpp` |
| 5 | 🟢→✅ | HeterodyneProcessorROCm → Ref03 (GpuContext). Удалено ~120 строк hiprtc+KernelCacheService | `heterodyne_processor_rocm.hpp/cpp` |

---

## 📊 Сводная оценка (3 модуля)

| Модуль | Файлов | Ref03 | Kernels | Profiling | Windows Stub |
|--------|--------|-------|---------|-----------|--------------|
| **signal_generators** | ~55 | ⚠️ Частично | ⭐⭐⭐⭐⭐ | ✅ | ✅ |
| **lch_farrow** | ~15 | ❌ Legacy | ⭐⭐⭐⭐ | ✅ | ✅ |
| **heterodyne** | ~21 | ❌ Legacy | ⭐⭐⭐⭐⭐ | ✅ | ✅ |

**Общий вердикт**: Все 3 модуля рабочие и хорошо протестированные. Kernel'ы оптимизированы. Но архитектурно — legacy (ручной hiprtc + raw void* buffers). Есть 3 системных проблемы, повторяющихся во всех модулях.

---

## 📐 Архитектурная карта Ref03

| Класс | Модуль | GpuContext | BufferSet | Ops | Статус |
|-------|--------|------------|-----------|-----|--------|
| CwGeneratorROCm | signal_generators | ✅ | — | — | **Ref03** |
| LfmGeneratorROCm | signal_generators | ✅ | — | — | **Ref03** |
| NoiseGeneratorROCm | signal_generators | ✅ | — | — | **Ref03** |
| FormSignalGeneratorROCm | signal_generators | ❌ | ❌ | ❌ | **Legacy** |
| LchFarrowROCm | lch_farrow | ❌ | ❌ | ❌ | **Legacy** |
| HeterodyneProcessorROCm | heterodyne | ❌ | ❌ | ❌ | **Legacy** |
| HeterodyneDechirp | heterodyne | — | — | — | Facade ✅ |

---

## 🔴 Критические проблемы: 0

Все 3 модуля рабочие. Нет утечек памяти, нет unchecked return codes, нет data races.

---

## 🟡 Важные замечания (3 системных)

### 1. MakeROCmDataFromEvents скопирован в каждый модуль

Мы уже создали `include/rocm_profiling_helpers.hpp` при ревью fft_func. Но 3 модуля всё ещё имеют свои копии:

| Модуль | Файл | Строки |
|--------|------|--------|
| lch_farrow | `src/lch_farrow_rocm.cpp` | 315-350 (Events + Clock) |
| signal_generators | `src/form_signal_generator_rocm.cpp` | 29-50 (Events) |
| heterodyne | `src/heterodyne_processor_rocm.cpp` | 53-75 (Events) |

**Исправление**: Заменить на `#include "rocm_profiling_helpers.hpp"` + `using fft_func_utils::MakeROCmDataFromEvents`.

---

### 2. heterodyne: warp_size строковая эвристика

**Файл**: `modules/heterodyne/src/heterodyne_processor_rocm.cpp:682-685`

```cpp
int warp_size = 32;
if (arch_name.find("gfx9") == 0) {
    warp_size = 64;  // ❌ строковая эвристика
}
```

**Исправление**: `ROCmCore::GetWarpSize()` (как уже исправлено в fft_func и statistics).

---

### 3. FormSignal + LchFarrow + HeterodyneProcessor — legacy, не Ref03

Все 3 класса используют:
- Ручной hiprtc: `hiprtcCreateProgram → hiprtcCompileProgram → hipModuleLoadData` (~100 строк каждый)
- Ручные void* буферы: `hipMalloc/hipFree` для buf_rx_, buf_ref_, buf_dc_ и т.д.
- Ручной KernelCacheService (загрузка/сохранение HSACO кеша)

Это ~300 строк дублирующего кода, который `GpuContext` делает в 1 вызов `CompileModule()`.

**Рекомендация**: Миграция на Ref03 по образцу CwGeneratorROCm → GpuContext + (опционально) BufferSet.

**Приоритет**: Низкий — код работает, тестирован. Миграция — при рефакторинге.

---

## 🟢 Что отлично (по модулям)

### signal_generators ⭐⭐⭐⭐⭐ Kernels
- CW: `__sincosf` (single SFU pass), 2D grid (beam × sample)
- LFM: `__sincosf`, chirp phase `phi = 2πf0*t + π*μ*t²`
- Noise: Philox-2x32-10 PRNG + Box-Muller, reproducible по seed
- FormSignal: Multi-channel из скриптового DSL, `__sincosf`
- CW/LFM/Noise мигрированы на Ref03 (GpuContext) ✅

### lch_farrow ⭐⭐⭐⭐ Unique algorithm
- Lagrange 48x5 матрица: fractional delay через полиномиальную интерполяцию
- Philox PRNG встроен в kernel (optional noise)
- `__fmaf_rn` для FMA цепочек (если используется)
- Отдельные ref файлы (.cl + .hip) для dual-backend

### heterodyne ⭐⭐⭐⭐⭐ Best-in-class kernels
- `aligned(8) float2_t` → 64-bit load/store (единственный модуль с этим!)
- `__sincosf` → single SFU pass для cos+sin
- 2D grid (sample × antenna) — no div/mod
- `phase_step` precomputed on CPU (no division in kernel)
- OPT-3: DechirpWithGPURef — zero H2D transfer
- OPT-4: Cached conjugate LFM generator
- HeterodyneDechirp: чистый Facade pattern с strategy (OpenCL/ROCm)

---

## ✅ Соответствие стандартам GPUWorkLib

| Критерий | signal_generators | lch_farrow | heterodyne |
|----------|-------------------|------------|------------|
| DrvGPU интеграция | ✅ | ✅ | ✅ |
| ConsoleOutput | ✅ | ✅ | ✅ |
| GPUProfiler | ✅ benchmarks | ✅ benchmarks | ✅ benchmarks |
| GpuBenchmarkBase | ✅ | ✅ | ✅ |
| Ref03 | ⚠️ Частично | ❌ | ❌ |
| Windows stubs | ✅ | ✅ | ✅ |
| Move semantics | ✅ | ✅ | ✅ |
| Kernel cache (HSACO) | ✅ | ✅ | ✅ |
| Стиль Google C++ | ✅ | ✅ | ✅ |

---

## 📋 Сводка задач

| # | Приоритет | Модуль | Описание | Сложность |
|---|-----------|--------|----------|-----------|
| 1 | 🟡 | ALL 3 | MakeROCmDataFromEvents → shared `rocm_profiling_helpers.hpp` | Низкая |
| 2 | 🟡 | heterodyne | warp_size `gfx9` → `ROCmCore::GetWarpSize()` | Низкая |
| 3 | 🟢 | signal_generators | FormSignalGeneratorROCm → Ref03 (GpuContext) | Средняя |
| 4 | 🟢 | lch_farrow | LchFarrowROCm → Ref03 (GpuContext) | Средняя |
| 5 | 🟢 | heterodyne | HeterodyneProcessorROCm → Ref03 (GpuContext) | Средняя |
| 6 | 🟢 | heterodyne | `aligned(8) float2_t` — распространить на все модули | Низкая |

---

*Ревью подготовлено с: sequential-thinking (2 шага), grep pattern analysis по ~80 файлам*
