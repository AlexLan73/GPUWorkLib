# Signal Generators — Краткий справочник

> CW, LFM, Noise, Script, FormSignal на GPU

---

## Генераторы

| Класс | Формула / Назначение |
|-------|----------------------|
| **CwGenerator** | `A * exp(j*(2πf*t + φ))` |
| **LfmGenerator** | ЛЧМ: `exp(j*πk*t² + j*2πf_start*t)` |
| **NoiseGenerator** | Philox + Box-Muller |
| **ScriptGenerator** | Text DSL → OpenCL kernel |
| **FormSignalGenerator** | Мультиканальный (getX формула) |
| **FormScriptGenerator** | FormSignal + **on-disk kernel cache** (SaveKernel/LoadKernel) |

---

## Быстрый старт

### C++

```cpp
signal_gen::SignalService service(backend);
signal_gen::CwParams cw{.f0 = 100.0, .freq_step = 10.0};
signal_gen::SystemSampling sys{.fs = 1000.0, .length = 4096};
cl_mem gpu_data = service.GenerateGpu(cw, sys, 8);
```

### Python

```python
gen = gpuworklib.SignalGenerator(ctx)
data = gen.generate_cw(256, 4096, 1000.0, f0=100.0, freq_step=10.0)
```

---

## On-disk kernel cache (FormScriptGenerator)

| Метод | Действие |
|-------|----------|
| `save_kernel("name", "comment")` | Сохраняет `name.cl`, `bin/name_opencl.bin`, manifest.json |
| `load_kernel("name")` | Binary (fast) или source (compile) |
| `list_kernels()` | Список сохранённых кернелов |

Через DrvGPU [KernelCacheService](../../DrvGPU/Services/Quick.md). При коллизии: `name_00.cl`, `name_01.cl`, …

---

*Обновлено: 2026-02-23*
