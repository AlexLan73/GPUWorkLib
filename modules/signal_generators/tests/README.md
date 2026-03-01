# Signal Generators — Tests

Тесты и бенчмарки модуля `signal_generators`.

Точка входа: [`all_test.hpp`](all_test.hpp) — включается из `src/main.cpp`.

---

## Функциональные тесты

| Файл | Что тестирует |
|------|--------------|
| `test_signal_generators.hpp` | CW, LFM, Noise: GenerateToGpu / GenerateToCpu / сравнение с CPU |
| `test_form_signal.hpp` | FormSignalGenerator: мультиканальный getX, noise, задержки |
| `test_form_script.hpp` | FormScriptGenerator: DSL компиляция, кэш кернелов |
| `test_delayed_form_signal.hpp` | DelayedFormSignalGenerator: Farrow 48×5 задержка |
| `test_lfm_analytical_delay.hpp` | LfmGeneratorAnalyticalDelay: аналитическая задержка LFM |
| `test_form_signal_rocm.hpp` | FormSignalGeneratorROCm: HIP-порт getX (Linux + AMD GPU) |

---

## Бенчмарки OpenCL (GpuBenchmarkBase)

### signal_generators_benchmark.hpp

Benchmark-классы для базовых генераторов:

| Класс | Генератор | Событие | output_dir |
|-------|-----------|---------|-----------|
| `CwGeneratorBenchmark` | `CwGenerator::GenerateToGpu()` | `Kernel` | `Results/Profiler/GPU_00_CwGenerator/` |
| `LfmGeneratorBenchmark` | `LfmGenerator::GenerateToGpu()` | `Kernel` | `Results/Profiler/GPU_00_LfmGenerator/` |
| `LfmConjugateGeneratorBenchmark` | `LfmConjugateGenerator::GenerateToGpu()` | `Kernel` | `Results/Profiler/GPU_00_LfmConjugateGenerator/` |
| `NoiseGeneratorBenchmark` | `NoiseGenerator::GenerateToGpu()` | `Kernel` | `Results/Profiler/GPU_00_NoiseGenerator/` |

### form_signal_benchmark.hpp

Benchmark-классы для Form генераторов:

| Класс | Генератор | События | output_dir |
|-------|-----------|---------|-----------|
| `FormSignalGeneratorBenchmark` | `FormSignalGenerator::GenerateInputData()` | `Kernel` | `Results/Profiler/GPU_00_FormSignal/` |
| `DelayedFormSignalGeneratorBenchmark` | `DelayedFormSignalGenerator::GenerateInputData()` | `FormSignal` + `FarrowDelay` | `Results/Profiler/GPU_00_DelayedFormSignal/` |
| `LfmAnalyticalDelayBenchmark` | `LfmGeneratorAnalyticalDelay::GenerateToGpu()` | `Kernel` | `Results/Profiler/GPU_00_LfmAnalyticalDelay/` |
| `FormScriptGeneratorBenchmark` | `FormScriptGenerator::GenerateInputData()` | `Kernel` | `Results/Profiler/GPU_00_FormScriptGenerator/` |

---

## Test Runners OpenCL

### test_signal_generators_benchmark.hpp

Namespace: `test_signal_generators_benchmark`

Запускает 4 бенчмарка (CW / LFM / LfmConjugate / Noise).

Параметры:
- `system.fs = 12e6`, `system.length = 4096`
- `beam_count = 1`
- `n_warmup = 5`, `n_runs = 20`

### test_form_signal_benchmark.hpp

Namespace: `test_form_signal_benchmark`

Запускает 4 бенчмарка (FormSignal / DelayedFormSignal / LfmAnalytDelay / FormScript).

Параметры:
- `antennas = 8`, `points = 4096`, `fs = 12e6`, `f0 = 1e6`
- LfmAnalytDelay: 8 задержек `{0.0, 1.5, 3.0, ..., 10.5}` мкс
- FormScript: `Compile()` вызывается до бенчмарка
- `n_warmup = 5`, `n_runs = 20`

---

## Бенчмарки ROCm (ENABLE_ROCM=1)

### signal_generators_benchmark_rocm.hpp

| Класс | Генератор | Событие | output_dir |
|-------|-----------|---------|-----------|
| `FormSignalGeneratorROCmBenchmark` | `FormSignalGeneratorROCm::GenerateInputData()` | `Kernel` | `Results/Profiler/GPU_00_FormSignalROCm/` |

### test_signal_generators_benchmark_rocm.hpp

Namespace: `test_signal_generators_benchmark_rocm`

Запускает FormSignalGeneratorROCm benchmark.
Если нет AMD GPU → `[SKIP]`.

---

## Как запустить

Раскомментировать нужные вызовы в `all_test.hpp`:

```cpp
// OpenCL: CW / LFM / LfmConjugate / Noise
test_signal_generators_benchmark::run();

// OpenCL: FormSignal / Delayed / LfmAnalytDelay / FormScript
test_form_signal_benchmark::run();

// ROCm: FormSignalROCm (только Linux + AMD GPU)
test_signal_generators_benchmark_rocm::run();
```

⚠️ **Требование**: OpenCL queue должен создаваться с `CL_QUEUE_PROFILING_ENABLE`
(test runner делает это автоматически).

---

## Результаты

Результаты сохраняются в `Results/Profiler/GPU_00_*/`:
- `report.md` — Markdown-отчёт (min/max/avg по событиям)
- `report.json` — JSON для автоматической обработки

Вывод управляется через `GPUProfiler`:
- `bench.Report()` → `PrintReport()` + `ExportMarkdown()` + `ExportJSON()`
