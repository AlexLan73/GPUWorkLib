# Архитектура GPUWorkLib {#architecture_page}

> Единая 6-слойная модель (Ref03) + C4 Model архитектура.

---

## Ref03 — Единая архитектура GPU-операций

Все модули строятся по единой 6-слойной модели:

| Слой | Класс | Назначение |
|------|-------|-----------|
| 1 | `GpuContext` | Per-module: backend, stream, compiled module, shared buffers |
| 2 | `IGpuOperation` | Interface: Name, Initialize, IsReady, Release |
| 3 | `GpuKernelOp` | Base: доступ к compiled kernels через GpuContext |
| 4 | `BufferSet<N>` | Compile-time GPU buffer array (zero overhead, trivial move) |
| 5 | Concrete Ops | Маленькие классы: MeanReductionOp, MedianHistogramOp, FirFilterOp... |
| 6 | Facade + Strategy | Тонкий фасад (StatisticsProcessor), авто-выбор (MedianStrategy) |

### Ключевые правила

- **Один класс — один файл** (Op в `operations/`, Step в `steps/`)
- **BufferSet\<N\>** вместо raw `void*` полей (enum индексы, compile-time size)
- **GpuContext per-module** → thread-safe между модулями, parallel streams
- **Facade API НЕ меняется** → Python bindings не ломаются
- **Strategies**: `IPipelineStep` + `PipelineBuilder` для гибких pipeline'ов

---

## DrvGPU — ядро библиотеки

```
Приложение / Модули
       | принимают IBackend*
       v
   DrvGPU (Facade)
       |
       +-- OpenCLBackend  -> cl_mem / cl_command_queue
       +-- ROCmBackend    -> hipStream_t  (ENABLE_ROCM=1)
       +-- HybridBackend  -> OpenCL + ROCm + ZeroCopyBridge
```

@see @ref drvgpu_main "DrvGPU подробно"
@see @ref drvgpu_architecture "Архитектура DrvGPU"

---

## C4 Model

### C1 — System Context
Внешние акторы: разработчик, Python-скрипты, GPU hardware.

### C2 — Container Diagram
Модули: DrvGPU + 11 modules + Python bindings + Tests.

### C3 — Component Diagram
Классы внутри каждого модуля (Facade, Operations, Kernels, Tests).

### C4 — Code Diagram
Интерфейсы, сигнатуры, kernel parameters.

---

## GpuBenchmarkBase

Все бенчмарки наследуют `GpuBenchmarkBase`:

```cpp
class MyBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
protected:
    void RunIteration() override {
        // один прогон под профайлером
    }
};
```

Запуск: `Run(iterations=100)` → автоматический GPUProfiler → `ExportMarkdown()`.

@warning Перед `profiler.Start()` вызывать `profiler.SetGPUInfo(backend)` — иначе в отчёте «Unknown GPU».
