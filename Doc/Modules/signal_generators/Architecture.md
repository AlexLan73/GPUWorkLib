# Signal Generators — Архитектура

## Паттерны проектирования

```
┌─────────────────────────────────────────────────────────────┐
│                     Signal Generators                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              SignalService (Facade)                   │   │
│  │  GenerateCpu(CwParams, SystemSampling)                │   │
│  │  GenerateGpu(LfmParams, SystemSampling, beam_count)   │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │         SignalGeneratorFactory (Factory)              │   │
│  │  CreateCw(backend, params) → unique_ptr<ISignalGen>  │   │
│  │  CreateLfm(backend, params)                          │   │
│  │  CreateNoise(backend, params)                        │   │
│  │  Create(backend, SignalRequest)                       │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │           ISignalGenerator (Strategy)                │   │
│  │  GenerateToCpu(system, out, size) → void             │   │
│  │  GenerateToGpu(system, beam_count) → cl_mem          │   │
│  │  Kind() → SignalKind                                 │   │
│  └──────────────────────┬───────────────────────────────┘   │
│           ┌──────────────┼──────────────┐                   │
│           ▼              ▼              ▼                   │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│   │ CwGenerator  │ │ LfmGenerator │ │NoiseGenerator│       │
│   └──────────────┘ └──────────────┘ └──────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        ScriptGenerator (Standalone)                  │   │
│  │  LoadScript(text) → compile OpenCL kernel            │   │
│  │  Generate() → cl_mem                                 │   │
│  │  GenerateToCpu() → vector<complex<float>>            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Все генераторы получают IBackend* через DI (Dependency Injection) │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Паттерны

| Паттерн | Класс | Назначение |
|---------|-------|------------|
| **Strategy** | `ISignalGenerator` | Единый интерфейс, разные алгоритмы |
| **Factory** | `SignalGeneratorFactory` | Создание генераторов по типу |
| **Facade** | `SignalService` | Упрощённый API для CPU/GPU генерации |
| **DI** | Все генераторы | `IBackend*` инжектируется через конструктор |

## Отличие ScriptGenerator

`ScriptGenerator` не реализует `ISignalGenerator`, потому что:
1. Использует `ANTENNAS`/`POINTS` вместо `SystemSampling`
2. Параметры задаются в текстовом DSL, а не в структурах
3. Не имеет CPU reference реализации (только GPU)

---

## Жизненный цикл GPU ресурсов

```
Constructor(backend)
    ├── Получить cl_context, cl_command_queue, cl_device_id из IBackend
    └── CompileKernel() — clCreateProgramWithSource + clBuildProgram
        │
        ▼
GenerateToGpu(system, beam_count)
    ├── clCreateBuffer(output, beam_count * length * sizeof(complex<float>))
    ├── clSetKernelArg(kernel, ...)
    ├── clEnqueueNDRangeKernel(global_size = total_samples)
    ├── clFinish(queue)
    └── return cl_mem  ← ВЫЗЫВАЮЩИЙ ОТВЕЧАЕТ за clReleaseMemObject()!
        │
        ▼
Destructor
    └── ReleaseGpuResources() — clReleaseKernel, clReleaseProgram
```

## OpenCL Kernel Compilation

Все генераторы (CW, LFM, Noise) компилируют kernel при создании:
- **CW**: `cw_signal` kernel с параметрами f0, phase, amplitude, freq_step
- **LFM**: `lfm_signal` kernel с параметрами f_start, chirp_rate, amplitude
- **Noise**: `noise_signal` kernel с Philox-2x32 PRNG + Box-Muller transform
- **Script**: `script_signal` kernel генерируется из DSL текста

Флаги компиляции: `-cl-fast-relaxed-math -cl-single-precision-constant`

---

*Обновлено: 2026-02-13*
