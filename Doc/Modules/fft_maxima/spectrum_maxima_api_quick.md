# SpectrumMaximaFinder API Quick Reference

## API v2.0 (2026-02-12)

### Process()
```cpp
template<typename T>
vector<SpectrumResult> Process(InputData<T>& input, PeakSearchMode mode, DriverType driver);
```

### InputData<T>
| Поле | Тип | Описание |
|------|-----|----------|
| antenna_count | uint32_t | Кол-во антенн |
| n_point | uint32_t | Точек на антенну |
| data | T | vector/cl_mem/void* |
| repeat_count | uint32_t | Повторений (def: 2) |
| sample_rate | float | Гц (def: 1000) |
| memory_limit | float | GPU лимит (def: 0.80) |

### T типы
- `vector<complex<float>>` — CPU
- `cl_mem` — GPU OpenCL
- `void*` — SVM

### PeakSearchMode
- `ONE_PEAK` — 1 пик, 4 MaxValue
- `TWO_PEAKS` — 2 пика, 8 MaxValue

### DriverType
- `OPENCL` — OpenCL (default)
- `ROCM` — ROCm (planned)

### SpectrumResult
```cpp
uint32_t antenna_id;
MaxValue center_point;
struct { float freq_hz; float amplitude; } interpolated;
```

### Пути
- Логи: `Logs/DRVGPU_XX/YYYY-MM-DD/HH-MM-SS.log`
- Профайлер: `Results/Profiler/GPU_XX_Profiler/*.md|json`

### Минимальный пример
```cpp
DrvGPU gpu(BackendType::OPENCL, 0);
gpu.Initialize();
SpectrumMaximaFinder finder(&gpu.GetBackend());

InputData<vector<complex<float>>> input{
    .antenna_count=5, .n_point=100000, .data=my_data,
    .repeat_count=2, .sample_rate=1000.0f
};

auto results = finder.Process(input, PeakSearchMode::ONE_PEAK, DriverType::OPENCL);
```
