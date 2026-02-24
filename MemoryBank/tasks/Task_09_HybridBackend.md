# Task_09_HybridBackend — OPENCLandROCm гибридный режим (опционально)

> **Памятка для ИИ**: Тестировать под Linux (Debian с Radeon 9070). Зависит от Task_08 (ZeroCopy).

---

## ⚠️ ПРАВИЛО: ТОЛЬКО НОВЫЕ ФАЙЛЫ

**НЕ трогать** существующие файлы DrvGPU и OpenCL модулей!

```
❌ НЕЛЬЗЯ изменять:
   DrvGPU/backends/opencl/*.hpp / *.cpp
   DrvGPU/backends/rocm/*.hpp / *.cpp   (уже готовый код!)
   modules/*/src/*.cpp                  (рабочие модули!)

✅ СОЗДАВАТЬ новые файлы:
   DrvGPU/backends/hybrid/hybrid_backend.hpp  (НОВЫЙ)
   DrvGPU/backends/hybrid/hybrid_backend.cpp  (НОВЫЙ)
   DrvGPU/tests/test_hybrid_backend.hpp       (НОВЫЙ — изолированный тест)
   Python_test/hybrid/test_hybrid_backend.py  (НОВЫЙ — Python тест)

⚠️  DrvGPU/src/drv_gpu.cpp — можно добавить case OPENCLandROCm, но ОСТОРОЖНО
    (не ломать существующие case OpenCL и ROCm!)
```

---

## 1. Цель

Гибридный режим `BackendType::OPENCLandROCm`: на одном GPU — OpenCL + ROCm одновременно. ZeroCopyBridge (Task_08) связывает их память без копирования.

---

## 2. Зависимости

- Task_00_DrvGPU (ROCmBackend, OpenCLBackend)
- Task_08_ZeroCopy (ZeroCopyBridge — обязательно!)

---

## 3. Рекомендуемый вариант: B — Два DrvGPU

```
GPU 0
  ├── DrvGPU [OpenCL backend]   ← OpenCL работа (сигналы, FFT, фильтры)
  └── DrvGPU [ROCm backend]     ← ROCm работа (статистика, специфика AMD)
       ↕ ZeroCopyBridge (dma-buf, без копирования)
```

**Почему B лучше A**: не нужно трогать существующие IBackend интерфейс и классы.

---

## 4. Вариант A (альтернатива): HybridBackend обёртка

```cpp
// DrvGPU/backends/hybrid/hybrid_backend.hpp  (НОВЫЙ файл)
#pragma once
#include "interface/i_backend.hpp"
#include "backends/opencl/opencl_backend.hpp"
#include "backends/rocm/rocm_backend.hpp"

namespace drvgpu {

class HybridBackend : public IBackend {
  std::unique_ptr<OpenCLBackend> opencl_;
  std::unique_ptr<ROCmBackend>   rocm_;

public:
  IBackend* GetOpenCL() { return opencl_.get(); }
  IBackend* GetROCm()   { return rocm_.get(); }

  // IBackend overrides — по умолчанию делегирует в OpenCL
  void  Initialize(int device_index) override;
  void* Allocate(size_t size_bytes, unsigned int flags = 0) override;
  void  Free(void* ptr) override;
  BackendType GetType() const override { return BackendType::OPENCLandROCm; }
  // ... остальные методы IBackend
};

}  // namespace drvgpu
```

Добавить в `drv_gpu.cpp`:
```cpp
case BackendType::OPENCLandROCm:
    backend_ = std::make_unique<HybridBackend>();
    break;
```

---

## 5. C++ тест — НОВЫЙ файл `test_hybrid_backend.hpp`

```cpp
// DrvGPU/tests/test_hybrid_backend.hpp  (НОВЫЙ файл)
#pragma once
#if ENABLE_ROCM

// Тест 1: Инициализация обоих backend
void TestHybridInit();

// Тест 2: Аллокация через OpenCL, чтение через ROCm (ZeroCopy)
void TestHybridZeroCopyTransfer();

// Тест 3: Параллельная работа — OpenCL FFT + ROCm статистика одновременно
void TestHybridParallelWork();

#endif
```

---

## 6. Python тест — НОВЫЙ файл `Python_test/hybrid/test_hybrid_backend.py`

```python
# Python_test/hybrid/test_hybrid_backend.py  (НОВЫЙ файл)
import numpy as np
import gpuworklib

def test_hybrid_context_init():
    """Создаём HybridContext — оба backend на одном GPU"""
    ctx = gpuworklib.HybridGPUContext(0)
    assert ctx.opencl_device_name != ""
    assert ctx.rocm_device_name != ""

def test_hybrid_pipeline():
    """OpenCL FFT → ZeroCopy → ROCm Statistics"""
    ctx = gpuworklib.HybridGPUContext(0)
    data = np.random.randn(1024).astype(np.complex64)

    # OpenCL делает FFT
    fft = gpuworklib.FFTProcessor(ctx.opencl_ctx)
    spectrum = fft.process(data)

    # Без копирования передаём в ROCm
    stats = gpuworklib.StatisticsProcessor(ctx.rocm_ctx)
    result = stats.compute_statistics(spectrum)  # ZeroCopy внутри

    assert result['mean'] is not None
```

---

## 7. Задачи (по порядку)

1. Убедиться что Task_08 (ZeroCopy) работает
2. Выбрать Вариант A или B (рекомендован B)
3. Создать `hybrid_backend.hpp` / `.cpp` — НОВЫЕ файлы
4. Добавить `case OPENCLandROCm` в `drv_gpu.cpp` (минимальная правка!)
5. Создать `test_hybrid_backend.hpp` — НОВЫЙ файл
6. Добавить Python биндинг `HybridGPUContext` в `gpu_worklib_bindings.cpp` (под `#if ENABLE_ROCM`)
7. Создать `Python_test/hybrid/test_hybrid_backend.py` — НОВЫЙ файл

---

## 8. Чек-лист

- [ ] Task_08 ZeroCopy работает (обязательная зависимость)
- [ ] `DrvGPU/backends/hybrid/hybrid_backend.hpp` — НОВЫЙ
- [ ] `DrvGPU/backends/hybrid/hybrid_backend.cpp` — НОВЫЙ
- [ ] `case OPENCLandROCm` в `drv_gpu.cpp` — минимальная добавка
- [ ] `DrvGPU/tests/test_hybrid_backend.hpp` — НОВЫЙ
- [ ] Python биндинг `HybridGPUContext`
- [ ] `Python_test/hybrid/test_hybrid_backend.py` — НОВЫЙ
- [ ] Компиляция (Linux, ENABLE_ROCM=ON)
- [ ] Тесты C++: 3/3 PASSED
- [ ] Тесты Python: запуск через `sg render -c "python3 ..."`

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — Часть 3
- [Task_08_ZeroCopy.md](Task_08_ZeroCopy.md) — обязательная зависимость
