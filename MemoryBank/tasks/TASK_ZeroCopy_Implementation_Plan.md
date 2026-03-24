# TASK: ZeroCopy OpenCL → ROCm — План реализации

## Статус: PLAN_READY
## Дата: 2026-03-24
## Приоритет: 🔴 HIGH

---

## 1. Контекст и проблема

### Что нужно
256 векторов × 2M complex float (~4 ГБ) обрабатываются в OpenCL pipeline (FFT, фильтры).
Результат — `cl_mem` буферы в VRAM. Нужно передать их в ROCm (Capon) **без копирования**.

### Что было сломано
- `CL_MEM_AMD_GPU_VA = 0x4100` — **выдуманная** константа, нет в AMD заголовках
- `SupportsAmdGpuVA()` — проверяла расширения `cl_amd_svm`/`cl_khr_svm` (не связаны с GPU VA)
- `ExportClBufferToGpuVA()` — вызывала `clGetMemObjectInfo(buf, 0x4100)` → всегда CL_INVALID_VALUE
- Итог: ZeroCopy тесты **никогда не работали** на RDNA4

### Что мы выяснили (тест на реальном GPU)

**ПРОРЫВ**: GPU VA cl_mem буфера **можно извлечь** через сканирование HSA pointer info:

```
cl_mem handle (0x560befe70790) — C++ объект в heap
    offset +664:  0x7f85bc200000  — HSA type=1, size=4096 ← GPU VA!
    offset +736:  0x7f85bc200000  — тот же адрес (другое поле)
    offset +776:  0x7f85bc200000  — тот же адрес (третье поле)

hipMemcpy от 0x7f85bc200000 → данные [1, 2, 3, 4, ...] — СОВПАДАЮТ!
hsa_amd_portable_export_dmabuf → fd=3 — DMA-BUF РАБОТАЕТ!
```

**Вывод**: TRUE zero-copy из cl_mem возможен. Нужен probe при первом вызове.

---

## 2. Архитектура решения

### 2.1 Принцип: единый интерфейс, 4 стратегии

```
ZeroCopyBridge::ImportFromOpenCl(cl_mem, size, cl_device)
    │
    ├─ Стратегия A: HSA Probe (TRUE zero-copy)          ← НОВАЯ, приоритет 1
    │   cl_mem → scan → GPU VA → hip_ptr
    │   0 копий, 0 доп. памяти, ~микросекунды
    │
    ├─ Стратегия B: HSA DMA-BUF (TRUE zero-copy)        ← НОВАЯ, приоритет 2
    │   cl_mem → probe GPU VA → hsa_amd_portable_export_dmabuf → fd
    │   → hipImportExternalMemory → hip_ptr
    │   0 копий, 0 доп. памяти, ~микросекунды
    │
    ├─ Стратегия C: GPU Copy Kernel (1 VRAM→VRAM copy)  ← НОВАЯ, приоритет 3
    │   cl_mem → OpenCL copy kernel → coarse-grain SVM (VRAM) → hip_ptr
    │   1 GPU-internal копия, +4ГБ VRAM, ~8мс для 4ГБ
    │
    └─ Стратегия D: SVM Fine-grain (через CPU)          ← ЕСТЬ, fallback
        cl_mem → clEnqueueReadBuffer → fine-grain SVM (system RAM) → hip_ptr
        Данные через CPU, медленно для больших буферов
```

### 2.2 Для SVM-буферов (ImportFromSVM)

```
SVMBuffer (coarse-grain, VRAM) → ImportFromSVM(svm_ptr, size)
    → hip_ptr = svm_ptr  (TRUE zero-copy, тест подтвердил)
```

### 2.3 Интерфейс НЕ МЕНЯЕТСЯ

```cpp
// Для вызывающего кода — всё как раньше:
ZeroCopyBridge bridge;
bridge.ImportFromOpenCl(cl_buffer, size, cl_device);  // автовыбор стратегии
void* hip_ptr = bridge.GetHipPtr();                    // HIP указатель
ZeroCopyMethod method = bridge.GetMethod();            // какой метод выбран
```

---

## 3. Файлы и изменения

### Phase 1: HSA Probe — TRUE zero-copy из cl_mem (ГЛАВНОЕ)

#### 3.1 Новый файл: `DrvGPU/backends/rocm/hsa_interop.hpp`

```cpp
// HSA interop utilities для извлечения GPU VA из cl_mem
namespace drv_gpu_lib {

struct HsaProbeResult {
    void* gpu_va = nullptr;      // GPU virtual address
    size_t size = 0;             // Размер аллокации
    int offset_in_handle = -1;   // Offset внутри cl_mem для кеширования
    bool valid = false;
};

// Однократный probe: сканирует cl_mem, ищет GPU VA через hsa_amd_pointer_info
// Кеширует offset для последующих вызовов (O(1) после первого)
HsaProbeResult ProbeGpuVA(cl_mem cl_buffer, size_t expected_size);

// Проверка: доступна ли HSA runtime
bool IsHsaAvailable();

}  // namespace drv_gpu_lib
```

**Алгоритм ProbeGpuVA:**
1. Первый вызов: сканирует cl_mem handle по 8-байтовым шагам (0..2048)
2. Для каждого `void*` значения: `hsa_amd_pointer_info(val, &info)`
3. Если `info.type == HSA_EXT_POINTER_TYPE_HSA` и `info.sizeInBytes == expected_size` → нашли
4. Кешируем offset в `static int cached_offset`
5. Последующие вызовы: читаем по cached_offset (O(1), наносекунды)

**Валидация:**
- `hipMemcpy` от найденного GPU VA → проверяем что данные совпадают
- Если probe не нашёл → возвращаем `valid = false` → fallback на стратегию C/D

#### 3.2 Изменение: `DrvGPU/backends/opencl/opencl_export.hpp`

```diff
- #define CL_MEM_AMD_GPU_VA 0x4100                    // УДАЛИТЬ — фейк

- inline bool SupportsAmdGpuVA(cl_device_id device)   // УДАЛИТЬ — проверяла не то
- inline void* ExportClBufferToGpuVA(cl_mem buffer)    // УДАЛИТЬ — использовала фейк

+ // Заменить на:
+ inline bool SupportsHsaProbe()                       // Проверка HSA runtime
+ inline void* ExtractGpuVA(cl_mem buffer, size_t sz)  // HSA probe → GPU VA

  // DetectBestZeroCopyMethod() — обновить приоритеты:
  // 1. HSA_PROBE (если HSA доступна) — TRUE zero-copy
  // 2. DMA_BUF (если расширение есть) — оставить
  // 3. SVM (fine-grain) — fallback
```

**Новый enum:**
```cpp
enum class ZeroCopyMethod {
    NONE,
    HSA_PROBE,    // ← НОВЫЙ: GPU VA через HSA probe (true zero-copy)
    HSA_DMABUF,   // ← НОВЫЙ: HSA dma-buf export → HIP import
    DMA_BUF,      // OpenCL extension (не работает на RDNA4)
    SVM_VRAM,     // ← НОВЫЙ: coarse-grain SVM copy (VRAM→VRAM)
    SVM_SYSTEM,   // fine-grain SVM copy (через system RAM, медленно)
};
```

#### 3.3 Изменение: `DrvGPU/backends/rocm/zero_copy_bridge.hpp`

```diff
  // Добавить:
+ void ImportFromHsaProbe(cl_mem cl_buffer, size_t buffer_size);

  // ImportFromOpenCl() — обновить порядок:
  // 1. HSA Probe → ImportFromHsaProbe()
  // 2. HSA DMA-BUF → ImportFromDmaBuf()
  // 3. SVM VRAM → GPU copy kernel → ImportFromSVM()
  // 4. SVM System → clEnqueueReadBuffer → ImportFromSVM()
```

#### 3.4 Изменение: `DrvGPU/backends/rocm/zero_copy_bridge.cpp`

- `ImportFromHsaProbe()`: вызывает `ProbeGpuVA()`, сохраняет GPU VA как `hip_ptr_`
- `ImportFromOpenCl()`: новый порядок стратегий (A → B → C → D)
- `Release()`: HSA_PROBE не требует освобождения (owns_memory_=false)

### Phase 2: GPU Copy Kernel (стратегия C)

#### 3.5 Новый файл: `DrvGPU/backends/opencl/gpu_copy_kernel.hpp`

```cpp
// OpenCL kernel для VRAM→VRAM копии: cl_mem → SVM (coarse-grain)
// Данные не покидают GPU. ~8мс для 4ГБ.

bool GpuCopyClMemToSVM(cl_command_queue queue, cl_context ctx,
                        cl_mem src, void* svm_dst, size_t size_bytes);
```

**Kernel:**
```opencl
__kernel void copy_buffer(__global uchar* src, __global uchar* dst, uint N) {
    uint i = get_global_id(0);
    if (i < N) dst[i] = src[i];
}
```

- `src` — `clSetKernelArg` (cl_mem)
- `dst` — `clSetKernelArgSVMPointer` (coarse-grain SVM)
- Копия внутри GPU, без PCIe

### Phase 3: Тесты

#### 3.6 Обновление: `DrvGPU/tests/test_zero_copy.hpp`

Новые тесты:
- `test_hsa_probe` — извлечение GPU VA из cl_mem через HSA
- `test_hsa_probe_data_integrity` — запись в cl_mem, чтение через HIP от GPU VA
- `test_hsa_probe_large_buffer` — проверка на большом буфере (16MB)
- `test_gpu_copy_kernel` — VRAM→VRAM copy через OpenCL kernel
- `test_svm_coarse_grain_zerocopy` — SVM coarse-grain → HIP (подтверждено тестом)

Обновить существующие:
- `test_bridge_import` — проверить HSA_PROBE метод
- `test_data_integrity` — работает с HSA_PROBE

#### 3.7 Обновление: `modules/capon/tests/test_capon_opencl_to_rocm.hpp`

- `CheckZeroCopyAvailable()` — принимает HSA_PROBE
- `test_01_detect_interop` — показывает HSA probe capability

### Phase 4: Документация

#### 3.8 Обновление: `Doc_Addition/ZeroCopy.md`

- Добавить результаты тестирования HSA probe
- Описать offset probe алгоритм
- Таблица совместимости ROCm версий

#### 3.9 Обновление: `TASK_ZeroCopy_RDNA4_fix.md`

- Статус → COMPLETED
- Описание решения

### Phase 5: CMake

#### 3.10 Обновление: `DrvGPU/CMakeLists.txt`

```cmake
# Добавить линковку HSA runtime:
if(ROCM_ENABLED)
    target_link_libraries(drvgpu PUBLIC hsa-runtime64)
endif()
```

---

## 4. Приоритеты стратегий на разных GPU

| GPU | ROCm | HSA Probe | HSA DMA-BUF | GPU Copy | SVM System |
|-----|------|-----------|-------------|----------|------------|
| RDNA4 (gfx1201) | 7.2 | ✅ primary | ✅ backup | ✅ fallback | ✅ last resort |
| MI100 (gfx908) | 7.2 | вероятно ✅ | вероятно ✅ | ✅ | ✅ |
| MI250 (gfx90a) | 7.2 | вероятно ✅ | вероятно ✅ | ✅ | ✅ |

HSA probe — универсальный метод (HSA = фундамент ROCm, есть на всех AMD GPU).

---

## 5. Сравнение стратегий

| Стратегия | Копий | Доп. память | Время (4ГБ) | Надёжность |
|-----------|-------|-------------|-------------|------------|
| A: HSA Probe | **0** | **0** | **~μs** | Зависит от CLR layout |
| B: HSA DMA-BUF | **0** | **0** | **~μs** | Через dma-buf fd |
| C: GPU Copy | **1** VRAM→VRAM | +4ГБ VRAM | ~8мс | 100% надёжно |
| D: SVM System | **1** VRAM→CPU | +4ГБ system | ~секунды | 100% надёжно |

---

## 6. Порядок реализации

| # | Что | Оценка | Зависимости |
|---|-----|--------|-------------|
| 1 | `hsa_interop.hpp` (ProbeGpuVA) | 1ч | — |
| 2 | Обновить `opencl_export.hpp` (убрать фейк, добавить HSA) | 30мин | #1 |
| 3 | `ImportFromHsaProbe` в zero_copy_bridge | 30мин | #1, #2 |
| 4 | Обновить `ImportFromOpenCl` (порядок стратегий) | 30мин | #3 |
| 5 | `gpu_copy_kernel.hpp` (стратегия C) | 1ч | — |
| 6 | Интеграция стратегии C в bridge | 30мин | #4, #5 |
| 7 | Тесты | 1ч | #1-#6 |
| 8 | Сборка + проверка на GPU | 30мин | #7 |
| 9 | Документация | 30мин | #8 |

**Итого: ~6 часов**

---

## 7. Риски

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| Offset GPU VA меняется между версиями ROCm | Средняя | Auto-probe при каждом запуске (не хардкод) |
| HSA probe не найдёт GPU VA на другом GPU | Низкая | Fallback на стратегию C/D |
| Coarse-grain SVM не поддерживается | Очень низкая | Fallback на fine-grain (стратегия D) |
| Performance regression при probe | Очень низкая | Probe один раз, кеширование offset |

---

## 8. Диагностические файлы (уже созданы)

- `DrvGPU/tests/test_zerocopy_rdna4.cpp` — SVM + HSA dma-buf диагностика
- `DrvGPU/tests/test_clmem_gpu_va_probe.cpp` — HSA probe для извлечения GPU VA

Оба скомпилированы и протестированы на gfx1201. Результаты: 3/3 PASSED, probe нашёл GPU VA.

---

## 9. Связанные файлы проекта

| Файл | Что делает | Менять? |
|------|-----------|---------|
| `DrvGPU/backends/opencl/opencl_export.hpp` | Detection + export | ✅ Да |
| `DrvGPU/backends/rocm/zero_copy_bridge.hpp` | Bridge header | ✅ Да |
| `DrvGPU/backends/rocm/zero_copy_bridge.cpp` | Bridge impl | ✅ Да |
| `DrvGPU/backends/hybrid/hybrid_backend.cpp` | CreateZeroCopyBridge | ⚠️ Минимально |
| `DrvGPU/memory/svm_buffer.hpp` | SVM аллокация | ❌ Не трогаем |
| `DrvGPU/memory/svm_capabilities.hpp` | SVM detection | ❌ Не трогаем |
| `DrvGPU/memory/hip_buffer.hpp` | HIP wrapper | ❌ Не трогаем |
| `DrvGPU/memory/gpu_buffer.hpp` | GPU buffer | ❌ Не трогаем |
| `DrvGPU/interface/i_backend.hpp` | IBackend | ❌ Не трогаем |
| `DrvGPU/tests/test_zero_copy.hpp` | Тесты | ✅ Обновить |
| `modules/capon/tests/test_capon_opencl_to_rocm.hpp` | Capon тест | ✅ Обновить |
| `DrvGPU/CMakeLists.txt` | Сборка | ✅ Добавить hsa-runtime64 |

**Интерфейс `IBackend`, `IMemoryBuffer`, модули — НЕ МЕНЯЮТСЯ.**
