# IN PROGRESS — Текущие задачи

> **Обновлено**: 2026-04-09

---

## 🟢 C++ Test Migration — оставшиеся модули

> Инфраструктура `modules/test_utils/` готова (TASK_CppTest_01-05).
> Эталон: statistics. Мигрированы: signal_generators, fft_func, heterodyne, filters.

### Очередь миграции (🟢 низкий приоритет):
- [ ] lch_farrow
- [ ] vector_algebra
- [ ] fm_correlator
- [ ] strategies (сложный — Pipeline + Strategy GoF)
- [ ] capon (зависит от rocBLAS)
- [ ] range_angle

---

## 🔴 capon_rocblas — TODO

- rocBLAS CGEMM в CovarianceMatrixOp / CaponReliefOp / AdaptBeamformOp
- Нужен rocblas_handle из backend

---

## TODO

- [ ] `py::keep_alive<>()` в Python bindings — защита от GC ctx lifetime
- [ ] SpectrumMaxima multi-beam test: 1 fail (beam frequency detection)
- [ ] Non-square матрица 2500×100 для strategies

---

## ✅ ZeroCopy OpenCL → ROCm — DONE (2026-03-24)
> Реализован в `DrvGPU/backends/rocm/zero_copy_bridge.hpp/.cpp`
> 4 метода: HSA Probe, DMA-BUF, GPU Copy, SVM fallback

---

## 🟠 LibGPU Integration — запланировано на 2026-03-30

> **Таск**: `TASK_LibGPU_Integration.md`
> **План**: `MemoryBank/research/cmake_libgpu_integration_plan.md`

Создать промежуточный репозиторий LibGPU для передачи GPUWorkLib в закрытый проект.

**Блоки работ:**
- **A** — Подготовка GPUWorkLib: DrvGPU install() + sync_to_libgpu.py
- **B** — Создание LibGPU: CMakeLists.txt + Presets + Config + README
- **C** — Первая синхронизация + smoke-тест сборки
- **D** — Интеграция в закрытый проект (git submodule на сервере)

**Ключевое правило:** LibGPU = конверт, проект остаётся GPUWorkLib. Код НЕ переписывается.

*Последнее обновление: 2026-03-29*
