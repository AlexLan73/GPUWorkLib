# Ref04: Миграция OpenCL кеширования → ROCm

## Статус: 🔶 IN_PROGRESS (guard'ы ✅, StreamPool создан, баги и интеграция — осталось)
## Дата: 2026-03-25
## Обновлено: 2026-03-25 (code review + аудит static/singleton для 10 GPU)

---

## Главный принцип (Alex, 2026-03-25)

> **OpenCL на main нужен ТОЛЬКО для согласования данных (ZeroCopy interop).**
> **ВСЕ вычисления — ТОЛЬКО ROCm. Никаких исключений.**
> **10 GPU работают параллельно — singleton и static shared state ЗАПРЕЩЕНЫ**
> **(кроме ConsoleOutput, GPUProfiler, ServiceManager — координация по дизайну).**

```
┌─────────────────────────────────────────────────┐
│  ВЫЧИСЛЕНИЯ (ROCm ONLY)                        │
│  GpuContext, hiprtc, rocBLAS, rocSOLVER,        │
│  BufferSet, hipMalloc, hipStream_t              │
│  ⚠️ Никаких singleton/static shared state!      │
│  Каждый GPU — свой экземпляр всего.             │
├─────────────────────────────────────────────────┤
│  СТЫКОВКА ПАМЯТИ (OpenCL → ROCm interop)       │
│  ZeroCopyBridge, HSA Probe, DMA-BUF,           │
│  OpenCLBackend (только для cl_mem доступа),     │
│  opencl_export.hpp (detection)                  │
│  ⚠️ Кеши — per-OpenCLCore, не global!           │
└─────────────────────────────────────────────────┘
```

---

## 🔴 КРИТИЧЕСКИЕ БАГИ (найдены при ревью)

### БАГ-1: `gpu_copy_kernel.hpp:117-156` — `GetOrCompileCopyKernels()` НЕ КОМПИЛИРУЕТСЯ

**Файл**: `DrvGPU/backends/opencl/gpu_copy_kernel.hpp`

Функция `ReleaseCopyKernelsForContext()` (стр. 91) обновлена под singleton `GpuCopyKernelCache`,
а `GetOrCompileCopyKernels()` — **НЕТ**. Рассинхрон:

| Строка | Код | Проблема |
|--------|-----|----------|
| 119 | `lock(cache.mutex)` | ✅ OK (singleton) |
| 121 | `lock(cache_mutex)` | ❌ `cache_mutex` НЕ СУЩЕСТВУЕТ, двойной lock_guard с именем `lock` |
| 123 | `cache.find(ctx)` | ❌ Нет метода `.find()` у `GpuCopyKernelCache`, нужно `cache.map.find(ctx)` |
| 151 | `cache[ctx]` | ❌ Нет `operator[]`, нужно `cache.map[ctx]` |

**Результат**: Файл не скомпилируется при сборке ZeroCopy.

---

### БАГ-2: `GpuCopyKernelCache` — SINGLETON, один mutex на 10 GPU

**Файл**: `DrvGPU/backends/opencl/gpu_copy_kernel.hpp:74-81`

```cpp
struct GpuCopyKernelCache {
  std::mutex mutex;                                    // ← ОДИН на 10 GPU!
  std::unordered_map<cl_context, GpuCopyKernels> map;  // ← Shared state!
  static GpuCopyKernelCache& Instance() {
    static GpuCopyKernelCache instance;                // ← SINGLETON!
    return instance;
  }
};
```

**Проблема**: При параллельном ZeroCopy на 10 GPU — все 10 ждут один mutex.
Это OpenCL interop (согласование данных), но даже interop не должен быть bottleneck.

**Решение**: Убрать singleton. `GpuCopyKernels` → member в `OpenCLCore` (per-GPU).
Каждый OpenCLCore хранит свой скомпилированный cl_program, никакого shared state.

---

### БАГ-3: `ReleaseCopyKernelsForContext()` написана, но НЕ вызывается

**Файл**: `DrvGPU/backends/opencl/opencl_core.cpp:181-191`

`ReleaseResources()` делает `clReleaseContext(context_)` но перед этим
**не вызывает** `ReleaseCopyKernelsForContext(context_)`. Dangling pointer в кеше.

**Примечание**: После БАГ-2 fix (перенос в member) эта функция станет не нужна —
cleanup будет в деструкторе OpenCLCore автоматически.

---

### ~~БАГ-4~~: `hsa_interop.hpp:152-153` — shared cached_offset на 10 GPU — ✅ НЕ БАГ

**Файл**: `DrvGPU/backends/rocm/hsa_interop.hpp`

```cpp
static std::atomic<int> cached_offset{-1};
static std::mutex probe_mutex;
```

**Проверено (code review 2026-03-25)**: Это **ПРАВИЛЬНО**.

`cached_offset` — byte offset внутри C++ объекта `amd::Memory` (ROCm CLR) где лежит GPU VA.
Это свойство **layout класса** в `libOpenCL.so`, а НЕ свойство конкретного GPU.
Все 10 GPU используют один `libOpenCL.so` → один класс `amd::Memory` → **offset одинаковый**.

`probe_mutex` захватывается только при первом скане (`cached_offset == -1`).
После кеширования — fast path через `atomic load`, без lock. Contention = 0.

---

## Карта static/singleton (аудит для 10 GPU)

### ✅ Легитимные singleton'ы (координация между GPU — так и задумано)

| Singleton | Файл | Почему OK |
|-----------|------|-----------|
| `ConsoleOutput::GetInstance()` | `services/console_output.hpp:102` | Координация вывода 10 GPU |
| `GPUProfiler::GetInstance()` | `services/gpu_profiler.hpp:84` | Координация профилирования |
| `ServiceManager::GetInstance()` | `services/service_manager.hpp:78` | Загрузка configGPU.json |
| `ConfigLogger` | `config_logger.cpp:23` | Конфигурация (read-only после init) |
| `GPUConfig` | `gpu_config.cpp:125` | Конфигурация (read-only после init) |
| `DefaultLogger::instances_` | `default_logger.hpp:146` | `map<int, Logger>` — per-GPU, правильно |

### ✅ Static const (readonly, thread-safe в C++17)

Все `static const std::vector<std::string> kKernelNames = {...}` в `.cpp` модулей —
compile-time readonly, инициализируются один раз. **Нет проблем.**

### 🔴 ПРОБЛЕМНЫЕ static (contention на 10 GPU)

| # | Что | Файл | Проблема | Решение |
|---|-----|------|----------|---------|
| БАГ-2 | `GpuCopyKernelCache` singleton | `gpu_copy_kernel.hpp:74` | 1 mutex на 10 GPU, lock на КАЖДЫЙ вызов | → member в `OpenCLCore` |

### ✅ Проверено — НЕ проблема

| # | Что | Файл | Почему OK |
|---|-----|------|-----------|
| ~~БАГ-4~~ | `cached_offset` + `probe_mutex` | `hsa_interop.hpp:152-153` | Offset = layout `amd::Memory` класса в `libOpenCL.so`, одинаковый для всех GPU. Mutex только на первый скан, дальше lock-free atomic. |

---

## Сводная таблица (актуальный статус)

| # | Компонент | Файл | Статус | Что осталось |
|---|-----------|------|--------|-------------|
| 1.1 | FormSignalGenerator | `form_signal_generator.hpp` | ✅ Guard `#if !ENABLE_ROCM` | — |
| 1.2 | DelayedFormSignalGenerator | `delayed_form_signal_generator.hpp` | ✅ Guard `#if !ENABLE_ROCM` | — |
| 1.3 | LchFarrow | `lch_farrow.hpp` | ✅ Guard `#if !ENABLE_ROCM` | — |
| 1.4 | SVMBuffer | `svm_buffer.hpp` | ✅ Guard `#if !ENABLE_ROCM` | — |
| 1.5 | SignalService | `signal_service.hpp` | ✅ Guard `#if !ENABLE_ROCM` | — |
| 1.6 | SignalGeneratorFactory | `signal_generator_factory.hpp` | ⚠️ Частично | `#include "form_signal_generator_rocm.hpp"` без `#if ENABLE_ROCM` guard (стр.16) |
| 2.1 | StreamPool | `rocm/stream_pool.{hpp,cpp}` | 🔶 Создан, в CMake | Не используется ни одним модулем |
| 2.2 | CommandQueuePool CMake | `DrvGPU/CMakeLists.txt:82` | ⚠️ Нет guard | Компилируется при `ENABLE_ROCM=ON` (не нужен) |
| 3.1 | gpu_copy_kernel | `gpu_copy_kernel.hpp` | 🔴 Не компилируется + singleton | БАГ-1, БАГ-2, БАГ-3 |
| 3.2 | hsa_interop | `hsa_interop.hpp` | ✅ Правильно (layout `libOpenCL.so`) | — |
| 3.3 | KernelCacheService | `kernel_cache_service.hpp` | ✅ Backend-agnostic | — |
| 3.4 | GpuContext | `gpu_context.hpp` | ✅ Эталон (per-GPU) | — |
| 3.5 | BufferSet\<N\> | `buffer_set.hpp` | ✅ Эталон (per-GPU) | — |

---

## Конкретный план действий

### ✅ ШАГ 1: Python биндинги → ROCm — ВЫПОЛНЕН

`GPUContext`(OpenCL) изолирован в `#if !ENABLE_ROCM` (строки 272-960).
ROCm path через `ROCmGPUContext` + `py_*_rocm.hpp` (строки 176-214).
Grep `clEnqueueReadBuffer` в `python/` вне `#if !ENABLE_ROCM` → **0 совпадений**.

---

### ✅ ШАГ 2: Guard'ы для OpenCL-only компонентов — ВЫПОЛНЕН

Все файлы обёрнуты в `#if !ENABLE_ROCM`:

| Файл | Строка guard | ROCm альтернатива |
|------|-------------|-------------------|
| `form_signal_generator.hpp` | стр.3 | `form_signal_generator_rocm.hpp` ✅ |
| `delayed_form_signal_generator.hpp` | стр.3 | `delayed_form_signal_generator_rocm.hpp` ✅ |
| `lch_farrow.hpp` | стр.3 | `lch_farrow_rocm.hpp` ✅ |
| `svm_buffer.hpp` | стр.3 | UVA (hipMalloc) ✅ |
| `signal_service.hpp` | стр.3 | ROCm модули используют генераторы напрямую ✅ |

---

### ✅ ШАГ 3: StreamPool — СОЗДАН (файлы + CMake)

`DrvGPU/backends/rocm/stream_pool.hpp` + `.cpp` — RAII, mutex, move semantics.
В `DrvGPU/CMakeLists.txt` строки 168-169 — добавлен.

**Осталось**: интеграция в модули (ШАГ 5).

---

### 🔴 ШАГ 4: Починить gpu_copy_kernel.hpp (БАГ-1 + БАГ-2 + БАГ-3)

**Цель**: Убрать singleton, сделать per-OpenCLCore.

**4a.** Перенести `GpuCopyKernels` как member в `OpenCLCore`:

```cpp
// opencl_core.hpp — добавить:
class OpenCLCore {
  // ...existing members...
  GpuCopyKernels copy_kernels_;  // Per-GPU, compile on first use
  bool copy_kernels_compiled_ = false;

public:
  GpuCopyKernels* GetOrCompileCopyKernels();  // Per-instance, no mutex needed
};
```

**4b.** В `OpenCLCore::ReleaseResources()` — освобождать copy_kernels_ перед clReleaseContext:

```cpp
void OpenCLCore::ReleaseResources() {
  // Сначала освобождаем kernels (зависят от context)
  if (copy_kernels_.program) clReleaseProgram(copy_kernels_.program);
  if (copy_kernels_.k_wide) clReleaseKernel(copy_kernels_.k_wide);
  if (copy_kernels_.k_tail) clReleaseKernel(copy_kernels_.k_tail);
  copy_kernels_ = {};
  copy_kernels_compiled_ = false;

  // Потом context
  if (context_) { clReleaseContext(context_); context_ = nullptr; }
  device_ = nullptr;
  platform_ = nullptr;
}
```

**4c.** Удалить из `gpu_copy_kernel.hpp`:
- `GpuCopyKernelCache` struct (singleton)
- `ReleaseCopyKernelsForContext()` (больше не нужна)
- `GetOrCompileCopyKernels(cl_context)` free function (заменена методом OpenCLCore)

**4d.** Обновить `GpuCopyClMemToSVM()` — принимать `OpenCLCore*` вместо `cl_context`:

```cpp
inline bool GpuCopyClMemToSVM(
    cl_command_queue queue,
    OpenCLCore* core,       // ← per-GPU, вместо cl_context
    cl_mem src,
    void* svm_dst,
    size_t size_bytes);
```

---

### ~~ШАГ 5~~: ~~Починить hsa_interop.hpp~~ — ОТМЕНЁН (не баг)

`cached_offset` = layout класса `amd::Memory` в `libOpenCL.so`. Одинаковый для всех GPU на одной машине. Проверено при review.

---

### 🟡 ШАГ 5: Мелкие правки

| Файл | Что | Приоритет |
|------|-----|-----------|
| `signal_generator_factory.hpp:16` | `#include "form_signal_generator_rocm.hpp"` без `#if ENABLE_ROCM` guard — на nvidia не соберётся | 🟡 |
| `DrvGPU/CMakeLists.txt:82` | `command_queue_pool.cpp` компилируется при `ENABLE_ROCM=ON` (лишнее) | 🟢 |
| `StreamPool::Cleanup()` | public метод без mutex — race с `GetStream()` если вызвать из другого потока. Сделать private, оставить только через Initialize/деструктор | 🟢 |

---

### 🟡 ШАГ 6: Интеграция StreamPool в ROCm-модули

`StreamPool` создан но **нигде не используется** (grep = 0 include).

Кандидаты на интеграцию:
- `ROCmBackend` — добавить `StreamPool` как member
- Модули с параллельными kernel launch'ами
- ZeroCopy bridge (если нужны параллельные копии)

---

### ШАГ 7: Тесты

- C++ тест `StreamPool`: Initialize/GetStream/SynchronizeAll на реальном GPU
- ZeroCopy: `test_zerocopy_rdna4` — все стратегии проходят после fix'ов
- Multi-GPU: проверить что 10 GPU не блокируют друг друга на interop

---

## НЕ ТРОГАТЬ (ZeroCopy interop API)

| Файл | Почему не трогать |
|------|------------------|
| `DrvGPU/include/drv_gpu.hpp:136,182` | `CreateFromExternalOpenCL`/`CreateFromExternalHybrid` — согласование данных |
| `DrvGPU/src/drv_gpu.cpp:63,107` | Реализация ZeroCopy API |
| `DrvGPU/backends/opencl/opencl_backend.*` | Доступ к cl_mem для interop |
| `DrvGPU/backends/opencl/opencl_core.*` | cl_context/device фундамент interop |
| `DrvGPU/backends/opencl/opencl_export.hpp` | Detection capabilities |
| `DrvGPU/backends/opencl/gpu_copy_kernel.hpp` | VRAM→VRAM fallback (**после fix БАГ-1/2/3**) |
| `DrvGPU/backends/rocm/zero_copy_bridge.*` | Мост cl_mem → HIP |
| `DrvGPU/backends/rocm/hsa_interop.hpp` | HSA probe (**после fix БАГ-4**) |
| `DrvGPU/backends/hybrid/hybrid_backend.*` | Гибридный backend |

---

## Ответы Alex (2026-03-25)

1. **Ветка main** → ВСЕ вычисления ТОЛЬКО через ROCm. OpenCL — только согласование данных.
2. **FormSignalGenerator (OpenCL)** → НЕ нужен на main. OpenCL только для ZeroCopy interop.
3. **ZeroCopy interop** → ОБЯЗАТЕЛЬНО оставить. Проверять что копирование = zero.
4. **FormSignalGeneratorROCm** → Создать **полноценный C++ класс** с `GpuContext` + `hiprtc`. Не просто guard.
5. **StreamPool** → Нужен **однозначно**. Создать и использовать **во всех ROCm-модулях**. Не откладывать.
6. **drv_gpu.cpp:63,107 / drv_gpu.hpp:136,182** — ZeroCopy interop API. **НЕ ТРОГАТЬ**.
7. **lch_farrow.hpp guard** → Делать сейчас (приоритет A), вместе с остальными guard'ами.
8. **Цель** → Подчистить всё для передачи в **ГЛАВНЫЙ репозиторий**.
9. **Singleton/static** → ЗАПРЕЩЕНЫ для данных per-GPU. 10 GPU — каждый свой экземпляр. Singleton только для координации (консоль, профилировщик).

---

## Исправленные ошибки в предыдущей версии таска

| Что было написано | Реальность |
|------------------|-----------|
| `svm_buffer.hpp:505-519 PrintStats() использует std::cout` | ❌ Неверно. Стр.524: `DRVGPU_LOG_INFO("SVMBuffer", ss.str())` — уже через логгер |
| `svm_buffer.hpp:25 #include <iostream>` | ❌ Неверно. Стр.28-29: `<sstream>` + `<iomanip>` (не iostream) |
| `gpu_copy_kernel.hpp:88 static unordered_map никогда не чистится` | ⚠️ Частично верно. `ReleaseCopyKernelsForContext()` уже написана (стр.91), но НЕ вызывается из `~OpenCLCore`. Реальная проблема — singleton (БАГ-2) |
| Шаг 2 "Guard'ы" как TODO | ❌ Уже выполнен |
| Шаг 3 "form_signal_generator.hpp → guard" как TODO | ❌ Уже выполнен |
| Шаг 4 "StreamPool — создать" как TODO | ⚠️ Частично. Файлы созданы + CMake, но не интегрирован |

---

*Автор: Кодо*
*Дата создания: 2026-03-25*
*Обновлено: 2026-03-25 (code review + аудит static/singleton для 10 GPU)*
