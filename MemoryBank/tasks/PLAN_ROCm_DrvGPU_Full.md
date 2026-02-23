# План доработки GPUWorkLib под ROCm

> **Дата**: 2026-02-24
> **Исполнитель**: другой ИИ / команда (на рабочей машине)
> **Контроль выполнения**: Кодо (проверка завтра)
> **Источник**: Кодо

---

## Какой план рабочий?

**PLAN_ROCm_DrvGPU_Full.md** (этот файл) — **рабочий план** для исполнения. Детальные шаги, чек-лист, порядок модулей.

**PLAN_AMD_Radeon_9070_ROCm.md** — общий план миграции (этапы 1–3, контекст). Использовать как справочник, не как пошаговую инструкцию.

---

## ВАЖНО: Инструкции для ИИ-исполнителя

**Создай тесты** (test_rocm_backend.hpp, test_spectrum_maxima_rocm.hpp и т.д.) — код тестов должен быть написан и добавлен в all_test.hpp.

**НЕ запускай тесты сегодня** — запуск и проверка будут завтра на Debian/Ubuntu с Radeon 9070 / MI100. Сегодня только: код, компиляция, создание тестовых файлов.

---

## Контекст

- **Оборудование**: Radeon 9070 (тесты), AMD Instinct MI100 (работа)
- **ОС**: Debian 13, Ubuntu 22.04/24.04 (см. [ROCm_Setup_Instructions.md](../../ROCm_Setup_Instructions.md))
- **Текущее состояние**: DrvGPU — только OpenCL; ROCm — заглушки в модулях
- **Цель**: Полноценный ROCm backend + ZeroCopy (OpenCL↔ROCm без копирования)

---

## Часть 1: DrvGPU — ROCm Backend

### 1.1 Структура файлов

Создать в `DrvGPU/backends/rocm/` по аналогии с `DrvGPU/backends/opencl/` — тот же набор классов и методов, только для HIP/ROCm. **Только для Linux** (ROCm не поддерживается на Windows).

| Файл | Назначение |
|------|------------|
| `rocm_backend.hpp` | Класс ROCmBackend : public IBackend |
| `rocm_backend.cpp` | Реализация Initialize, Allocate, Free, Memcpy, Synchronize |
| `rocm_core.hpp` | hipDevice_t, hipCtx_t, hipStream_t, инициализация HIP |
| `rocm_core.cpp` | hipDeviceGet, hipCtxCreate, hipStreamCreate |

### 1.2 ROCmBackend — ключевые методы

Реализовать все методы `DrvGPU/interface/i_backend.hpp`:

```cpp
// Инициализация
void Initialize(int device_index) override;
bool IsInitialized() const override;
void Cleanup() override;

// Нативные хэндлы (void* → hipCtx_t, hipDevice_t, hipStream_t)
void* GetNativeContext() const override;
void* GetNativeDevice() const override;
void* GetNativeQueue() const override;

// Память (hipMalloc/hipFree, hipMemcpy)
void* Allocate(size_t size_bytes, unsigned int flags = 0) override;
void Free(void* ptr) override;
void MemcpyHostToDevice(void* dst, const void* src, size_t size_bytes) override;
void MemcpyDeviceToHost(void* dst, const void* src, size_t size_bytes) override;
void MemcpyDeviceToDevice(void* dst, const void* src, size_t size_bytes) override;

// Синхронизация
void Synchronize() override;
void Flush() override;

// Возможности
bool SupportsSVM() const override { return false; }
BackendType GetType() const override { return BackendType::ROCm; }
```

**Важно**: `Allocate` возвращает `void*` (фактически `hipDeviceptr_t`). `Free` принимает этот указатель и вызывает `hipFree`.

### 1.3 Интеграция в DrvGPU

В `DrvGPU/src/drv_gpu.cpp`, метод `CreateBackend()`:

```cpp
case BackendType::ROCm:
    backend_ = std::make_unique<ROCmBackend>();
    break;
```

Добавить `#include "backends/rocm/rocm_backend.hpp"` и условную компиляцию `#if ENABLE_ROCM`.

### 1.4 CMake — подключение ROCm backend

В `DrvGPU/CMakeLists.txt`:

- Условно добавлять `backends/rocm/rocm_backend.cpp`, `rocm_core.cpp` при `ROCM_ENABLED`
- `target_link_libraries(drvgpu hip::hip)`
- `target_include_directories` для `backends/rocm`

---

## Часть 2: ZeroCopy — OpenCL ↔ ROCm через память

Источник: [MemoryBank/research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md](../research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md)

### 2.1 Выбор варианта

Рекомендуемый: **Вариант A (dma-buf)** — надёжный, Linux-only. Резерв: **Вариант B (CL_MEM_AMD_GPU_VA)** — нулевой overhead, AMD-only.

### 2.2 Структура ZeroCopy

Создать в `DrvGPU/memory/` или `DrvGPU/backends/rocm/`:

| Файл | Назначение |
|------|------------|
| `zero_copy_bridge.hpp` | Класс ZeroCopyBridge |
| `zero_copy_bridge.cpp` | ImportFromOpenCl, GetHipPtr, деструктор |
| `opencl_export.hpp` | export_cl_buffer_fd(cl_mem) — получение dma-buf fd |

### 2.3 OpenCL: экспорт cl_mem в dma-buf fd

**Требование**: буфер должен создаваться с флагами экспорта. Расширение `cl_khr_external_memory` / `cl_khr_external_memory_dma_buf`.

Добавить в `DrvGPU/backends/opencl/` или как утилиту:

```cpp
// opencl_export.hpp
int ExportClBufferToFd(cl_mem buffer);
// Возвращает dma-buf fd или -1 при ошибке
// Использует clGetMemObjectInfo(CL_MEM_LINUX_DMA_BUF_FD_KHR) или AMD-аналог
```

**Проверка**: перед вызовом проверять `cl_khr_external_memory_dma_buf` в `CL_DEVICE_EXTENSIONS`.

### 2.4 ZeroCopyBridge — импорт в HIP

```cpp
class ZeroCopyBridge {
public:
    hipError_t ImportFromOpenCl(int dma_buf_fd, size_t buffer_size);
    void* GetHipPtr() const { return hip_ptr_; }
    ~ZeroCopyBridge();  // hipDestroyExternalMemory
private:
    hipExternalMemory_t ext_mem_ = nullptr;
    void* hip_ptr_ = nullptr;
    size_t size_ = 0;
};
```

Логика: `hipImportExternalMemory` + `hipExternalMemoryGetMappedBuffer` (см. исследование, строки 176–207).

### 2.5 Синхронизация

Перед передачей указателя между OpenCL и ROCm:

```cpp
clFinish(cl_queue);           // OpenCL закончил
// барьер — данные в VRAM
hipLaunchKernelGGL(...);      // ROCm использует те же данные
hipStreamSynchronize(stream); // ROCm закончил
```

Документировать в ZeroCopyBridge или в отдельном `zero_copy_sync.hpp`.

### 2.6 Интеграция в IBackend (опционально)

Вариант: добавить в `IBackend` виртуальный метод:

```cpp
virtual void* ImportFromOpenCl(void* cl_mem_ptr, size_t size_bytes);
// OpenCLBackend: return nullptr (не поддерживается)
// ROCmBackend: ExportClBufferToFd + ZeroCopyBridge.ImportFromOpenCl
```

Или оставить ZeroCopyBridge как отдельный класс, вызываемый явно при `BackendType::OPENCLandROCm`.

---

## Часть 3: OPENCLandROCm — гибридный режим (по умолчанию)

### 3.1 Архитектура

Для `BackendType::OPENCLandROCm` на одном физическом GPU нужны оба runtime. **Гибридный режим — по умолчанию** при работе с OpenCL + ROCm.

Варианты:
- **A**: `HybridBackend` — обёртка, хранит `OpenCLBackend` и `ROCmBackend`, делегирует по контексту
- **B**: Два `DrvGPU` на одну GPU — один с OpenCL, один с ROCm; ZeroCopyBridge связывает их память

Рекомендация: **Вариант B** — проще, меньше изменений. `GPUManager::InitializeAll(OPENCLandROCm)` создаёт для каждой GPU два DrvGPU (или один DrvGPU с двумя backend-указателями — `GetOpenCLBackend()`, `GetROCmBackend()`).

### 3.2 Реализация HybridBackend (если выбран Вариант A)

```cpp
class HybridBackend : public IBackend {
    std::unique_ptr<OpenCLBackend> opencl_;
    std::unique_ptr<ROCmBackend> rocm_;
public:
    IBackend* GetOpenCL() { return opencl_.get(); }
    IBackend* GetROCm() { return rocm_.get(); }
    // Allocate по умолчанию через OpenCL (с экспортом)
    void* Allocate(size_t size_bytes, unsigned int flags) override;
};
```

`CreateBackend()` для `OPENCLandROCm`:
```cpp
backend_ = std::make_unique<HybridBackend>();
// HybridBackend::Initialize создаёт оба под-backend для одного device_index
```

---

## Часть 4: Память и InputData

### 4.1 HIPBuffer

Создать `DrvGPU/memory/hip_buffer.hpp` по аналогии с GPUBuffer:

- Конструктор: `HIPBuffer(void* hip_ptr, size_t num_elements, IBackend* backend)`
- `Write(host_data, size_bytes)` — hipMemcpy H2D
- `Read(host_data, size_bytes)` — hipMemcpy D2H
- `GetDevicePtr()` — возврат `void*`
- Деструктор: `hipFree` (если владеет) или нет (для ZeroCopy)

### 4.2 InputData для ROCm

`InputData` уже поддерживает `T = void*`. Для ROCm использовать `InputData<void*>` с `data` = указатель от `ROCmBackend::Allocate` или `ZeroCopyBridge::GetHipPtr()`.

Добавить в `input_data_traits.hpp` (если есть) или в документацию: `void*` для ROCm трактуется как `hipDeviceptr_t`.

---

## Часть 5: Модули — порядок портирования

**Порядок**: сначала DrvGPU (ROCmBackend), затем модули от простого к сложному.

### 5.1 Приоритет (от простого к сложному)

| # | Модуль | Путь | Зависимости | Тест |
|---|--------|------|-------------|------|
| 0 | ROCmBackend + rocm_core | DrvGPU/backends/rocm/ | hip | test_rocm_backend.hpp |
| 1 | FFTProcessorROCm | modules/fft_processor/ | ROCmBackend, hipFFT | test_fft_processor_rocm.hpp |
| 2 | **StatisticsProcessorROCm** | modules/statistics/ | ROCmBackend, rocPRIM | test_statistics_rocm.hpp |
| 3 | SpectrumProcessorROCm | modules/fft_maxima/ | ROCmBackend, hipFFT | test_spectrum_maxima_rocm.hpp |
| 4 | FirFilterROCm, IirFilterROCm | modules/filters/ | ROCmBackend | test_filters_rocm.hpp |
| 5 | LchFarrowROCm | modules/lch_farrow/ | ROCmBackend | test_lch_farrow_rocm.hpp |
| 6 | FormSignalGeneratorROCm | modules/signal_generators/ | ROCmBackend | test_form_signal_rocm.hpp |
| 7 | HeterodyneProcessorROCm | modules/heterodyne/ | ROCmBackend, fft_maxima | test_heterodyne_rocm.hpp |

### 5.2 FFTProcessorROCm (1 — простой)

- hipFFT вместо clFFT
- Порт pre-callback (padding), post-kernel (mag/phase) в HIP

### 5.3 StatisticsProcessorROCm (2 — ROCm only)

**Только ROCm** — модуль не портируется на OpenCL в рамках этого плана. Входные данные: **все антенны сразу**, сигнал **complex float**.

**Структура** — по аналогии с [modules/fft_processor/](../../modules/fft_processor/): `include/`, `src/`, `kernels/`, `tests/`.

**Анализ: kernel vs rocPRIM**

| Операция | Рекомендация | Обоснование |
|----------|--------------|-------------|
| Mean | Custom kernel (hierarchical reduction) или rocPRIM `reduce` | Простая редукция, rocPRIM reduce — готовое решение |
| Median | **rocPRIM** `radix_sort` + средний элемент или `rocprim::nth_element` | Свой kernel для медианы сложнее; rocPRIM — проверенный вариант |
| Variance / STD | Custom kernel (Welford) | Один проход mean+variance+std, численно стабильно |

**Источники**: [MemoryBank/DiscussionPlan/INDEX.md](../DiscussionPlan/INDEX.md), [MemoryBank/specs/statistics_module.md](../specs/statistics_module.md).

**Интеграция**: модуль простой (редукции), вставлен после FFTProcessor. Зависимость: ROCmBackend + rocPRIM.

### 5.4 SpectrumProcessorROCm (3)

Текущий stub: `modules/fft_maxima/src/spectrum_processor_rocm.cpp`.

- hipFFT вместо clFFT
- HIP kernel для post-processing (поиск максимума, параболическая интерполяция)
- `ProcessFromCPU`, `ProcessFromGPU` — загрузка данных, FFT, post-kernel

### 5.5 FirFilterROCm, IirFilterROCm (4)

- Порт FIR/IIR kernels в `.hip`

### 5.6 LchFarrowROCm (5)

- Порт LCH_FARROW_KERNEL_SOURCE в `.hip`

### 5.7 FormSignalGeneratorROCm (6)

- Порт form_signal в `.hip`

### 5.8 HeterodyneProcessorROCm (7 — сложный)

Текущий stub: `modules/heterodyne/src/heterodyne_processor_rocm.cpp`.

- `Dechirp()` — порт `dechirp_multiply.cl` в `.hip`
- `Correct()` — порт `dechirp_correct.cl` в `.hip`
- Использовать `LfmConjugateGenerator` (CPU) или портировать в HIP
- Зависит от SpectrumProcessorROCm (SpectrumMaximaFinder)

---

## Часть 6: ZeroCopy — детальный план интеграции

### 6.1 Когда использовать ZeroCopy

- `BackendType::OPENCLandROCm` — данные приходят в `cl_mem`, часть пайплайна на ROCm
- Пример: сеть → OpenCL (приём) → ZeroCopyBridge → ROCm (матричные операции)

### 6.2 Pipeline с ZeroCopy

```
OpenCL (cl_mem) → ExportClBufferToFd → dma_buf fd → ZeroCopyBridge → hip_ptr → HIP kernel
```

### 6.3 Проверка расширений

Перед использованием ZeroCopy проверять:
- `cl_khr_external_memory_dma_buf` в OpenCL device
- `hipImportExternalMemory` доступен в HIP

---

## Часть 7: Тесты

**ИИ-исполнитель**: создай файлы тестов, добавь вызовы в all_test.hpp. **Не запускай тесты** — запуск завтра (Кодо контролирует).

### 7.1 DrvGPU

| Тест | Файл | Что проверяет |
|------|------|---------------|
| ROCm init | test_rocm_backend.hpp | Initialize(0), IsInitialized |
| Allocate/Free | test_rocm_backend.hpp | Allocate, Free, повторное выделение |
| Memcpy | test_rocm_backend.hpp | MemcpyHostToDevice, MemcpyDeviceToHost, сравнение данных |
| Synchronize | test_rocm_backend.hpp | Synchronize после kernel |

### 7.2 ZeroCopy (если реализован)

| Тест | Что проверяет |
|------|---------------|
| Export fd | OpenCL buffer → fd != -1 |
| Import | fd → ZeroCopyBridge → hip_ptr != nullptr |
| Data integrity | Записать в cl_mem, прочитать через hip_ptr — совпадение |

### 7.3 Модули (порядок = порядок портирования)

| # | Тест | Модуль | Эталон |
|---|------|--------|--------|
| 1 | test_fft_processor_rocm | fft_processor | Сравнение с OpenCL FFT |
| 2 | test_statistics_rocm | statistics | ROCm only, mean/median/std vs NumPy |
| 3 | test_spectrum_maxima_rocm | fft_maxima | Сравнение с OpenCL результатом |
| 4 | test_filters_rocm | filters | GPU vs SciPy |
| 5 | test_lch_farrow_rocm | lch_farrow | Сравнение с OpenCL |
| 6 | test_form_signal_rocm | signal_generators | Сравнение с OpenCL |
| 7 | test_heterodyne_rocm | heterodyne | f_beat в пределах F_BEAT_TOL |

---

## Часть 8: Чек-лист для исполнителя

**Сегодня**: код + тесты (создать, не запускать). **Завтра**: Кодо запускает тесты на Debian/Ubuntu.

1. [ ] ROCmBackend + rocm_core — компиляция, линковка hip
2. [ ] DrvGPU::CreateBackend(ROCm) — без throw
3. [ ] test_rocm_backend.hpp — создать, добавить в all_test.hpp
4. [ ] FFTProcessorROCm + test_fft_processor_rocm.hpp
5. [ ] **StatisticsProcessorROCm** + test_statistics_rocm.hpp (ROCm only, complex float, все антенны)
6. [ ] SpectrumProcessorROCm + test_spectrum_maxima_rocm.hpp
7. [ ] FirFilterROCm, IirFilterROCm + test_filters_rocm.hpp
8. [ ] LchFarrowROCm + test_lch_farrow_rocm.hpp
9. [ ] FormSignalGeneratorROCm + test_form_signal_rocm.hpp
10. [ ] HeterodyneProcessorROCm + test_heterodyne_rocm.hpp
11. [ ] ExportClBufferToFd, ZeroCopyBridge (опционально)
12. [ ] OPENCLandROCm — HybridBackend (опционально)

---

## Ссылки

- [PLAN_AMD_Radeon_9070_ROCm.md](PLAN_AMD_Radeon_9070_ROCm.md)
- [AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md](../research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md)
- [ROCm_Setup_Instructions.md](../../ROCm_Setup_Instructions.md)

---

## Заметки Alex (включено в план)

1. Структура `DrvGPU/backends/rocm/` — по аналогии с `DrvGPU/backends/opencl/` (те же классы и методы).
2. `BackendType::OPENCLandROCm` — гибридный режим по умолчанию.
3. Порядок: сначала DrvGPU, затем модули от простого к сложному (fft_processor → **statistics** → fft_maxima → filters → lch_farrow → signal_generators → heterodyne).
