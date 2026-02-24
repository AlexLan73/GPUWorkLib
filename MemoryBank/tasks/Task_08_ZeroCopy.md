# Task_08_ZeroCopy — OpenCL ↔ ROCm без копирования (опционально)

> **Памятка для ИИ**: ZeroCopy через dma-buf — Linux-only. Тестировать под Debian с Radeon 9070.

---

## ⚠️ ПРАВИЛО: ТОЛЬКО НОВЫЕ ФАЙЛЫ

**НЕ трогать** существующие OpenCL файлы! Создавать только новые.

```
❌ НЕЛЬЗЯ изменять:
   DrvGPU/backends/opencl/*.hpp / *.cpp
   DrvGPU/src/drv_gpu.cpp        (без крайней нужды)

✅ СОЗДАВАТЬ новые файлы:
   DrvGPU/backends/opencl/opencl_export.hpp   (НОВЫЙ — экспорт dma-buf fd)
   DrvGPU/backends/rocm/zero_copy_bridge.hpp  (уже есть — только дополнять)
   DrvGPU/backends/rocm/zero_copy_bridge.cpp  (уже есть — только дополнять)
   DrvGPU/tests/test_zero_copy.hpp            (НОВЫЙ — изолированный тест)
   Python_test/zero_copy/test_zero_copy.py    (НОВЫЙ — Python тест)
```

---

## 1. Цель

Реализовать ZeroCopy: передача буфера из OpenCL в ROCm без копирования через dma-buf. Резерв: `CL_MEM_AMD_GPU_VA`.

---

## 2. Зависимости

- Task_00_DrvGPU (ROCmBackend, OpenCLBackend)
- Linux с AMD GPU (dma-buf — Linux-only)

---

## 3. Источник

[MemoryBank/research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md](../research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md)

---

## 4. Что уже есть

`DrvGPU/backends/rocm/zero_copy_bridge.cpp` — **уже реализован** (частично).
Нужно: написать тест + OpenCL-сторону экспорта.

---

## 5. Задачи

### 5.1 OpenCL экспорт — НОВЫЙ файл `opencl_export.hpp`

```cpp
// DrvGPU/backends/opencl/opencl_export.hpp  (НОВЫЙ файл)
#pragma once
#include <CL/cl.h>

namespace drvgpu {

// Экспортирует cl_mem в dma-buf fd.
// Возвращает fd >= 0 при успехе, -1 при ошибке.
int ExportClBufferToFd(cl_mem buffer);

// Проверяет поддержку расширения на устройстве
bool IsExternalMemorySupported(cl_device_id device);

}  // namespace drvgpu
```

Реализация в `opencl_export.cpp` (тоже НОВЫЙ файл).
Использовать: `cl_khr_external_memory_dma_buf` расширение.

### 5.2 ZeroCopyBridge — уже в `zero_copy_bridge.cpp`

Проверить наличие методов:
- `hipError_t ImportFromOpenCl(int dma_buf_fd, size_t buffer_size)`
- `void* GetHipPtr() const`
- Деструктор: `hipDestroyExternalMemory`

### 5.3 Синхронизация

Документировать в коде:
- Перед передачей: `clFinish(queue)` (OpenCL side)
- После ROCm: `hipStreamSynchronize(stream)`

### 5.4 C++ тест — НОВЫЙ файл `test_zero_copy.hpp`

```cpp
// DrvGPU/tests/test_zero_copy.hpp  (НОВЫЙ файл — НЕ модифицировать all_test.hpp!)
#pragma once
#if ENABLE_ROCM

#include "backends/opencl/opencl_export.hpp"
#include "backends/rocm/zero_copy_bridge.hpp"
// Не include-ить тесты из других модулей!

void TestZeroCopyExportFd();      // OpenCL buffer → fd != -1
void TestZeroCopyImport();        // fd → ZeroCopyBridge → hip_ptr != nullptr
void TestZeroCopyDataIntegrity(); // записать через OpenCL, прочитать через HIP → совпадение

#endif  // ENABLE_ROCM
```

> **Вызов**: добавить в `DrvGPU/tests/all_test.hpp` строку вызова — **только если Alex разрешит**!

### 5.5 Python тест — НОВЫЙ файл `Python_test/zero_copy/test_zero_copy.py`

```python
# Python_test/zero_copy/test_zero_copy.py  (НОВЫЙ файл)
import numpy as np
import gpuworklib

def test_zero_copy_available():
    """Проверяет доступность ZeroCopy на GPU"""
    ctx_ocl = gpuworklib.GPUContext(0)
    ctx_rocm = gpuworklib.ROCmGPUContext(0)
    bridge = gpuworklib.ZeroCopyBridge(ctx_ocl, ctx_rocm)
    assert bridge.is_supported()

def test_zero_copy_data_integrity():
    """Записывает данные через OpenCL, читает через ROCm"""
    ...
```

---

## 6. Чек-лист

- [ ] `DrvGPU/backends/opencl/opencl_export.hpp` — НОВЫЙ (ExportClBufferToFd)
- [ ] `DrvGPU/backends/opencl/opencl_export.cpp` — НОВЫЙ (реализация)
- [ ] `DrvGPU/backends/rocm/zero_copy_bridge.cpp` — проверить/дополнить
- [ ] `DrvGPU/tests/test_zero_copy.hpp` — НОВЫЙ (3 теста)
- [ ] `Python_test/zero_copy/test_zero_copy.py` — НОВЫЙ
- [ ] Компиляция (Linux, ENABLE_ROCM=ON)

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — Часть 2, 6
- [AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md](../research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md)
