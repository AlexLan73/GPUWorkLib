# COMPLETED — Завершённые задачи

> **Обновлено**: 2026-03-09

---

## 2026-03-09 — DrvGPU External Context Integration (ROCm 7.2+)

**Цель**: Позволить всем слоям DrvGPU принимать внешние GPU-хэндлы без захвата владения.
**Платформы**: AMD Radeon 9070 (gfx1201, RDNA4) + MI100 (gfx908, CDNA1).

### TASK A: ROCmBackend::InitializeFromExternalStream ✅

- `DrvGPU/backends/rocm/rocm_core.hpp/cpp` — добавлен `owns_stream_` флаг + `InitializeFromExternalStream()`
- `DrvGPU/backends/rocm/rocm_backend.hpp/cpp` — добавлен `InitializeFromExternalStream()`, `Cleanup()` делегирует решение `ROCmCore`
- `DrvGPU/tests/test_rocm_external_context.hpp` — 6 тестов (базовая init, ops, stream жив, native handles, device info, OwnsStream флаг)

### TASK B: HybridBackend::InitializeFromExternalContexts ✅

- `DrvGPU/backends/hybrid/hybrid_backend.hpp/cpp` — добавлен `InitializeFromExternalContexts(int, cl_context, cl_device_id, cl_command_queue, hipStream_t)`
- `DrvGPU/tests/test_hybrid_external_context.hpp` — 6 тестов (basic init, sub-backends, OpenCL ops, ROCm ops, native handles, resources survive cleanup)

### TASK C: DrvGPU Facade — Static Factory Methods ✅

- `DrvGPU/include/drv_gpu.hpp` — `ExternalInitTag` + приватный ctor + 3 static factory declarations
- `DrvGPU/src/drv_gpu.cpp` — реализация `ExternalInitTag` ctor + `CreateFromExternalOpenCL/ROCm/Hybrid`
- `DrvGPU/tests/test_drv_gpu_external.hpp` — 6 тестов (OpenCL: ops+survives; ROCm: ops+stream survives; Hybrid: ops+resources survive)
- `DrvGPU/tests/all_test.hpp` — добавлены закомментированные includes для всех 3 новых тест-файлов
- `DrvGPU/tests/README.md` — обновлена таблица тестов и покрытие

