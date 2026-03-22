# 🔍 Code Review: modules/fm_correlator

> **Дата**: 2026-03-22
> **Ревьюер**: Кодо (AI Assistant)
> **Объём**: ~14 файлов
> **Методы анализа**: sequential-thinking, web search (GPU FM correlation)
> **Ветка**: main (Linux, AMD GPU, ROCm 7.2+)

## ✅ ИСПРАВЛЕНО В ЭТОЙ СЕССИИ (2026-03-22)

| # | Тип | Описание | Файлы |
|---|-----|----------|-------|
| 1 | 🟡→✅ | warp_size: строковая эвристика → `ROCmCore::GetWarpSize()` | `fm_correlator_processor_rocm.cpp` |
| 2 | 🟢→✅ | Ref03 миграция: GpuContext для компиляции, 2-stream launch сохранён. Удалено ~150 строк hiprtc+KernelCacheService | `fm_correlator_processor_rocm.hpp/cpp` |
| 3 | 🟢→✅ | Windows stub добавлен (else throw) | `fm_correlator_processor_rocm.hpp` |
| 4 | 🟢→✅ | Move semantics (move ctor + move op=) | `fm_correlator_processor_rocm.hpp/cpp` |

---

## 📊 Общая оценка

| Аспект | Оценка | Комментарий |
|--------|--------|-------------|
| Архитектура | ⭐⭐⭐⭐ | Чистый Facade + Processor, 2-stream pipeline |
| Kernels | ⭐⭐⭐⭐⭐ | Лучшие в проекте: aligned(8), bitwise fabsf, inv_N, AND trick |
| hipFFT | ⭐⭐⭐⭐⭐ | Все return codes проверены (единственный модуль!) |
| Error handling | ⭐⭐⭐⭐⭐ | Валидация params, throw на каждую ошибку |
| Ref03 | ❌ Legacy | Manual hiprtc + KernelCacheService |
| Тесты | ⭐⭐⭐⭐ | 6 тестов: basic, mseq, combined, benchmark, profiling, summary |
| Документация | ⭐⭐⭐⭐⭐ | Каждая функция и поле документированы с "ЗАЧЕМ" |

---

## 🔴 Критические проблемы: 0

Модуль exemplary по качеству кода. Все hipFFT return codes проверяются, параметры валидируются, ошибки логируются.

---

## 🟡 Важные замечания (2)

### 1. warp_size строковая эвристика (**ИСПРАВЛЕНО**)

**Файл**: `src/fm_correlator_processor_rocm.cpp:588-592`

Было: `if (arch_name.find("gfx9") == 0) warp_size = 64;`
Стало: `ROCmCore::GetWarpSize()`

### 2. Legacy hiprtc (~150 строк), не Ref03

CompileKernels(): manual hiprtc + KernelCacheService + loadModuleAndFunctions.

**Особенность**: fm_correlator использует **2 отдельных stream** (stream0/stream1) для параллельной обработки ref и inp. GpuContext хранит 1 stream.

**Решение для миграции**: использовать GpuContext **только для компиляции** (CompileModule + GetKernel), а kernel launch делать на своих stream'ах через `hipModuleLaunchKernel(ctx_.GetKernel("name"), ..., stream0_, ...)`.

**Приоритет**: Low — модуль работает, тестирован, 2-stream pipeline уникален.

---

## 🟢 Что отлично

### Kernel оптимизации ⭐⭐⭐⭐⭐

1. **`aligned(8) float2_t`** — 64-bit load/store
2. **Bitwise fabsf**: `u.u &= 0x7FFFFFFFu` — branchless abs (1 instruction V_AND)
3. **`inv_N = 1.0f/N`** passed as parameter — FMUL (~4 clocks) вместо FDIV (~20)
4. **`(i + k) & (N - 1)`** instead of `% N` — zero-cost cyclic shift (N = power of 2)
5. **3D grid** для multiply_conj_fused: `(half_N, K, S)` — no div/mod per thread
6. **R2C/C2R** вместо C2C — вдвое меньше FFT данных (hermitian symmetry)

### 2-stream pipeline

```
stream0: H2D(ref) → apply_shifts → C2C FFT(ref)
stream1: H2D(inp) → R2C FFT(inp)       ← параллельно!
         sync
stream0: multiply_conj → C2R IFFT → extract → D2H
```

Реальный GPU concurrency — stream1 работает пока stream0 обрабатывает ref.

### Другие плюсы
- Все 4 hipfftExec проверяют return code ✅
- Параметры валидируются в SetParams() (power-of-2, >0, etc.)
- LFSR M-sequence generator на CPU — математически корректный
- Auto-batching через BatchManager для больших S
- RunTestPattern() — GPU-генерация тестовых данных без H2D

---

## ✅ Соответствие стандартам GPUWorkLib

| Критерий | Статус | Комментарий |
|----------|--------|-------------|
| DrvGPU | ✅ | IBackend*, stream из backend |
| ConsoleOutput | ✅ | Используется |
| hipFFT checks | ✅ | ВСЕ 4 проверяются |
| Ref03 | ❌ Legacy | 2-stream design → GpuContext для компиляции, launch на своих streams |
| Windows stub | ❌ | Нет — модуль только `#if ENABLE_ROCM` без else stub |
| Kernel cache | ✅ | KernelCacheService |
| Move semantics | ❌ | Нет move ctor/op= |
| BatchManager | ✅ | ProcessWithBatching |

---

## 📚 Источники

### Web Search
- [GPU-Accelerated Signal Processing for Passive Bistatic Radar](https://www.mdpi.com/2072-4292/15/22/5421) — FM radar correlation patterns
- [Efficient GPU-accelerated parallel cross-correlation](https://www.sciencedirect.com/science/article/abs/pii/S0743731525000218) — GPU cross-correlation techniques

---

## 📋 Сводка задач

| # | Приоритет | Описание | Сложность |
|---|-----------|----------|-----------|
| 1 | 🟡 ✅ | warp_size → GetWarpSize() | Низкая (**DONE**) |
| 2 | 🟢 | Ref03 миграция (CompileModule for compilation, keep 2-stream launch) | Средняя |
| 3 | 🟢 | Добавить Windows stub (else throw) | Низкая |
| 4 | 🟢 | Добавить Move semantics | Низкая |

---

*Ревью подготовлено с: sequential-thinking (2 шага), WebSearch (GPU FM correlation)*
