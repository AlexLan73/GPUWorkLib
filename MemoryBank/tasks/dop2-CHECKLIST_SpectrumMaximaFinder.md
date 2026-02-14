# 3 метода API SpectrumMaximaFinder

> **Обновлено**: 2026-02-14
> **Статус**: ✅ Реализовано, все 7 тестов проходят

---

## Метод 1: `Process<T>(input, peak_mode, DriverType)` — 1/2 пика

**Статус**: ✅ Работает

Pipeline: `Input → Zero-Pad (pre-callback) → clFFT → PostKernel (reduction) → 1/2 пика`

```cpp
auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```

- Поддержка: `vector<complex<float>>`, `cl_mem`, `void*` (SVM planned)
- Находит 1 или 2 максимума с параболической интерполяцией
- BatchManager для больших данных
- GPUProfiler: Upload, FFT, PostKernel, Download

---

## Метод 2: `FindAllMaxima<T>(input, dest, DriverType)` — FFT → ВСЕ максимумы

**Статус**: ✅ Работает

Pipeline: `Input → Zero-Pad (pre-callback) → clFFT + PostCallback(magnitude) → Detect → Scan → Compact`

```cpp
auto result = finder.FindAllMaxima(input, OutputDestination::CPU, DriverType::OPENCL);
```

- **Делает FFT** из сырого сигнала (с pre-callback для zero-padding)
- Post-callback вычисляет `|FFT[i]|` прямо во время FFT
- Detect → Prefix Sum → Compaction (все лучи параллельно)
- GPUProfiler: Upload/Copy, FFT+PostCallback, ComputeMag, Detect, Scan, Compact

---

## Метод 3: `AllMaxima<T>(input, dest, DriverType)` — БЕЗ FFT, только максимумы

**Статус**: ✅ Работает

Pipeline: `FFT Data → ComputeMagnitudes → Detect → Scan → Compact`

```cpp
// input.data = готовые FFT данные (complex float2)
// input.n_point = nFFT (размер FFT!)
auto result = finder.AllMaxima(fft_input, OutputDestination::CPU, DriverType::OPENCL);
```

- **НЕ делает FFT** — принимает готовый FFT спектр
- `input.n_point` = nFFT (данные уже FFT, padding не нужен)
- compute_magnitudes kernel вычисляет `|FFT[i]|`
- Тот же detect → scan → compact pipeline что и в FindAllMaxima
- GPUProfiler: ComputeMag, Detect, Scan, Compact

---

## Тесты

| # | Тест | Метод | Статус |
|---|------|-------|--------|
| 1 | TestThreePeaks | FindAllMaxima(cl_mem) — старый API | ✅ |
| 2 | TestMultiBeam | FindAllMaxima(cl_mem) — 5 лучей | ✅ |
| 3 | TestGpuOutput | FindAllMaxima(cl_mem) → GPU buffers | ✅ |
| 4 | TestFullPipelineCPU | FindAllMaxima<vector> — FFT pipeline | ✅ |
| 5 | TestFullPipelineGPU | FindAllMaxima<cl_mem> — FFT pipeline | ✅ |
| 6 | TestAllMaximaCPU | AllMaxima<vector> — без FFT | ✅ |
| 7 | TestAllMaximaGPU | AllMaxima<cl_mem> — без FFT | ✅ |

**Результат**: 7/7 тестов PASS

---

## Профилирование (GPUProfiler)

### Что записывается:
- **FindAllMaximaFromCPU**: Upload → FFT+PostCallback → (Detect → Scan → Compact)
- **FindAllMaximaFromGPUPipeline**: GPU→GPU Copy → FFT+PostCallback → (Detect → Scan → Compact)
- **FindAllMaxima(cl_mem)** / **AllMaxima**: ComputeMagnitudes → Detect → Scan → Compact

### TODO:
- [ ] Запустить полный бенчмарк (256 лучей × большой nFFT)
- [ ] Сравнить времена Process vs FindAllMaxima vs AllMaxima
- [ ] Сохранить результаты в `Results/Profiler/`

---

## Файлы

| Файл | Описание |
|------|----------|
| `include/spectrum_maxima_finder.h` | 3 публичных метода + template implementations |
| `src/spectrum_maxima_finder_all_maxima.cpp` | AllMaxima pipeline (detect, scan, compact, FFT plan) |
| `src/spectrum_maxima_finder_process.cpp` | Process pipeline (FFT + post-kernel reduction) |
| `src/spectrum_maxima_finder.cpp` | Общий код (init, buffers, move, release) |
| `include/kernels/all_maxima_kernel_sources.hpp` | OpenCL kernels: detect, scan, compact, compute_magnitudes, post-callback |
| `tests/test_find_all_maxima.hpp` | 7 тестов для всех 3 методов |

---

*Последнее обновление: 2026-02-14*
