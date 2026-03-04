# Ссылка на GPU_Profiling_Mechanism.md

Полный документ: [Doc_Addition/GPU_Profiling_Mechanism.md](../../../Doc_Addition/GPU_Profiling_Mechanism.md)

## Структура модуля с профилированием

```
modules/my_module/
├── include/
│   ├── my_module.hpp            ← prof_events* в Process(...)
│   └── my_module_rocm.hpp       ← ROCmProfEvents* в Process(...)  [ENABLE_ROCM]
├── src/
│   ├── my_module.cpp            ← CollectOrRelease()
│   └── my_module_rocm.cpp       ← MakeROCmDataFromEvents/Clock()
└── tests/
    ├── my_module_benchmark.hpp          ← GpuBenchmarkBase
    ├── test_my_module_benchmark.hpp     ← CL_QUEUE_PROFILING_ENABLE
    ├── my_module_benchmark_rocm.hpp
    └── test_my_module_benchmark_rocm.hpp
```

## Паттерны для grep

- Production: `prof_events|ROCmProfEvents`
- OpenCL helper: `CollectOrRelease`
- ROCm helpers: `MakeROCmDataFromEvents`, `MakeROCmDataFromClock`
- Benchmark: `GpuBenchmarkBase`, `ExecuteKernel`, `ExecuteKernelTimed`
- OpenCL queue: `CL_QUEUE_PROFILING_ENABLE`
- Forbidden: `GetStats.*con\.Print|GetStats.*std::cout`
