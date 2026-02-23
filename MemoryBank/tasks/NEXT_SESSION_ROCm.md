# Задание на продолжение — ROCm Backend

> **Дата обновления**: 2026-02-23 (сессия 2)
> **Автор**: Кодо (AI Assistant)
> **Для**: Следующая сессия Кодо / другой ИИ

---

## Статус: 12/12 пунктов ВЫПОЛНЕНО ✅ (код написан)

Все модули + ZeroCopy + HybridBackend написаны. Осталась **интеграция в сборку** и **тестирование на Linux**.

### Выполнено (Task_00 — Task_09):

| # | Модуль | Task | Файлы |
|---|--------|------|-------|
| 0 | ROCmBackend + rocm_core | Task_00 | `DrvGPU/backends/rocm/` |
| 1 | FFTProcessorROCm | Task_002 | `modules/fft_processor/` |
| 2 | StatisticsProcessorROCm | Task_002 | `modules/statistics/` |
| 3 | SpectrumProcessorROCm | Task_03 | `modules/fft_maxima/` |
| 4 | FirFilter + IirFilter ROCm | Task_06 | `modules/filters/` |
| 5 | LchFarrowROCm | Task_05 | `modules/lch_farrow/` |
| 6 | FormSignalGeneratorROCm | Task_07 | `modules/signal_generators/` |
| 7 | HeterodyneProcessorROCm | Task_08 | `modules/heterodyne/` |
| 8 | **ZeroCopy Bridge** | Task_09 | `DrvGPU/backends/opencl/opencl_export.hpp`, `DrvGPU/backends/rocm/zero_copy_bridge.*` |
| 9 | **HybridBackend** | Task_09 | `DrvGPU/backends/hybrid/hybrid_backend.*` |

---

## ⚠️ Что нужно ДОДЕЛАТЬ (не успели в этой сессии)

### 🔴 Приоритет 0: Интеграция в сборку (5 минут)

#### 1. `DrvGPU/tests/all_test.hpp` — добавить include + вызовы

```cpp
// В секцию #if ENABLE_ROCM includes добавить:
#include "test_zero_copy.hpp"
#include "test_hybrid_backend.hpp"

// В секцию run() после test_rocm_backend::run() добавить:
    test_zero_copy::run();
    test_hybrid_backend::run();
```

#### 2. `DrvGPU/CMakeLists.txt` — добавить новые файлы в ROCm sources

```cmake
# В секцию if(ROCM_ENABLED) добавить:
if(ROCM_ENABLED)
    set(DRVGPU_ROCM_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/rocm/rocm_backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/rocm/rocm_core.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/rocm/rocm_backend.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/rocm/rocm_core.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/rocm/zero_copy_bridge.cpp"    # NEW
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/rocm/zero_copy_bridge.hpp"    # NEW
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/hybrid/hybrid_backend.cpp"    # NEW
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/hybrid/hybrid_backend.hpp"    # NEW
    )
    message(STATUS "ROCm Backend: ENABLED (+ ZeroCopy + Hybrid)")
endif()

# В target_include_directories добавить:
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/hybrid    # NEW
```

#### 3. Проверить что `drv_gpu.cpp` уже обновлён (✅ сделано)

`CreateBackend()` уже содержит:
```cpp
case BackendType::OPENCLandROCm:
#if ENABLE_ROCM
    backend_ = std::make_unique<HybridBackend>();
#else
    throw std::runtime_error("...");
#endif
    break;
```

---

### 🔴 Приоритет 1: Тестирование на Linux + AMD GPU

**Оборудование**: Debian 13 / Ubuntu 22.04 с Radeon 9070 или MI100.

1. **Собрать проект с `-DROCM_ENABLED=ON`** на Linux
2. **Раскомментировать ROCm тесты** в каждом `all_test.hpp` (все модули)
3. **Запустить тесты**, проверить результаты
4. **Исправить баги** — вероятные проблемы:
   - hiprtc компиляция (синтаксис kernel, missing типы)
   - hipMemcpy API differences
   - hipFFT plan creation
   - ZeroCopy: может не поддерживаться dma-buf на конкретном GPU (fallback → AMD GPU VA)
   - Точность: GPU sin/cos vs CPU (tolerance в тестах)

---

## Новые файлы (Task_09)

```
DrvGPU/
├── backends/
│   ├── opencl/
│   │   └── opencl_export.hpp           # NEW — экспорт cl_mem (dma-buf, GPU VA, SVM)
│   ├── rocm/
│   │   ├── zero_copy_bridge.hpp        # NEW — ZeroCopyBridge class
│   │   └── zero_copy_bridge.cpp        # NEW — реализация (3 метода импорта)
│   └── hybrid/                         # NEW directory
│       ├── hybrid_backend.hpp          # NEW — HybridBackend : IBackend
│       └── hybrid_backend.cpp          # NEW — OpenCL+ROCm wrapper
├── tests/
│   ├── test_zero_copy.hpp              # NEW — 6 тестов ZeroCopy
│   └── test_hybrid_backend.hpp         # NEW — 6 тестов HybridBackend
└── src/
    └── drv_gpu.cpp                     # MODIFIED — CreateBackend(OPENCLandROCm)
```

---

## Полезные ссылки

- **План**: `MemoryBank/tasks/PLAN_ROCm_DrvGPU_Full.md`
- **Исследование ZeroCopy**: `MemoryBank/research/AMD_GPU_OpenCL_ROCm_ZeroCopy_2026-02-06.md`
- **Completed**: `MemoryBank/tasks/COMPLETED.md` — Task_00..09
- **Setup ROCm**: `ROCm_Setup_Instructions.md` (корень проекта)

---

*Обновлено: 2026-02-23, Кодо (сессия 2)*
