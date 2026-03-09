# Heterodyne — API-справочник

> Краткий справочник по всем публичным классам, методам и типам модуля

**Namespace**: `drv_gpu_lib`
**Python module**: `gpuworklib`

---

## Содержание

1. [HeterodyneDechirp (Facade)](#1-heterodynedechirp)
2. [HeterodyneParams](#2-heterodyneparams)
3. [HeterodyneResult / AntennaDechirpResult](#3-heterodyneresult)
4. [IHeterodyneProcessor (Strategy interface)](#4-iheterodyneprocessor)
5. [HeterodyneProcessorOpenCL](#5-heterodyneprocessoropencl)
6. [HeterodyneProcessorROCm](#6-heterodyneprocessorrocm)
7. [Python — gpuworklib.HeterodyneDechirp](#7-python-heterodynedechirp)
8. [Python — gpuworklib.HeterodyneROCm](#8-python-heterodynerocm)
9. [Цепочки вызовов](#9-цепочки-вызовов)

---

## 1. HeterodyneDechirp

**Файл**: `include/heterodyne_dechirp.hpp`
**Описание**: Фасад — выбирает backend (OpenCL / ROCm) и запускает полный пайплайн.

```cpp
class HeterodyneDechirp {
public:
    // Constructors
    explicit HeterodyneDechirp(IBackend* backend,
                               BackendType type = BackendType::OPENCL);

    // Not copyable
    HeterodyneDechirp(const HeterodyneDechirp&) = delete;
    HeterodyneDechirp& operator=(const HeterodyneDechirp&) = delete;

    // Configuration
    void SetParams(const HeterodyneParams& params);
    HeterodyneParams GetParams() const;

    // Processing
    HeterodyneResult Process(
        const std::vector<std::complex<float>>& rx_data);
    // rx_data: flat [num_antennas × num_samples], row-major

    HeterodyneResult ProcessExternal(
        void* gpu_ptr,
        const HeterodyneParams& params);
    // OpenCL: gpu_ptr = (void*)&cl_mem_handle
    // ROCm:   gpu_ptr = (void*)&hip_device_ptr
    // ВАЖНО: буфер НЕ освобождается внутри метода

    // Query
    HeterodyneResult GetLastResult() const;
};
```

**BackendType enum**:
```cpp
enum class BackendType { OPENCL, ROCm };  // ROCm — строчная m!
```

---

## 2. HeterodyneParams

**Файл**: `include/heterodyne_params.hpp`

```cpp
struct HeterodyneParams {
    float f_start      = 0.0f;    // [Гц] начальная частота ЛЧМ
    float f_end        = 2e6f;    // [Гц] конечная частота ЛЧМ (B = f_end - f_start)
    float sample_rate  = 12e6f;   // [Гц] частота дискретизации
    int   num_samples  = 8000;    // N — точек на антенну
    int   num_antennas = 5;       // A — количество антенн

    // Производные параметры
    float GetBandwidth() const;   // f_end - f_start              [Гц]
    float GetDuration()  const;   // num_samples / sample_rate    [с]
    float GetChirpRate() const;   // GetBandwidth() / GetDuration() [Гц/с]
    float GetBinWidth()  const;   // sample_rate / num_samples    [Гц/бин]
};
```

---

## 3. HeterodyneResult

**Файл**: `include/heterodyne_params.hpp`

```cpp
struct AntennaDechirpResult {
    int   antenna_idx;
    float f_beat_hz;        // частота биений [Гц]
    float f_beat_bin;       // бин (дробный, параболическая интерполяция)
    float range_m;          // дальность [м]
    float peak_amplitude;   // амплитуда пика FFT
    float peak_snr_db;      // SNR = 20·log10(peak/noise_est) [дБ]
};

struct HeterodyneResult {
    bool success;
    std::vector<AntennaDechirpResult> antennas;  // [num_antennas]
    std::vector<float> max_positions;            // все найденные максимумы (контроль)
    std::string error_message;                   // непусто при success=false

    // Формула дальности: R = c·T·f_beat / (2·B)
    static float CalcRange(float f_beat_hz,
                           float sample_rate,
                           int   num_samples,
                           float bandwidth);
};
```

---

## 4. IHeterodyneProcessor

**Файл**: `include/i_heterodyne_processor.hpp`
**Описание**: Strategy interface — реализуется OpenCL и ROCm процессорами.

```cpp
class IHeterodyneProcessor {
public:
    virtual ~IHeterodyneProcessor() = default;

    // Базовый дечирп: dc = conj(rx × ref)
    virtual std::vector<std::complex<float>> Dechirp(
        const std::vector<std::complex<float>>& rx_data,
        const std::vector<std::complex<float>>& ref_data,
        const HeterodyneParams& params) = 0;

    // Коррекция фазы: out = dc × exp(-j·2π·f_beat/fs·n)
    virtual std::vector<std::complex<float>> Correct(
        const std::vector<std::complex<float>>& dc_data,
        const std::vector<float>& f_beat_hz,      // [num_antennas]
        const HeterodyneParams& params) = 0;

    // Дечирп из GPU-буфера (rx уже на GPU, OPT-3)
    virtual std::vector<std::complex<float>> DechirpFromGPU(
        void* rx_gpu_ptr,
        const std::vector<std::complex<float>>& ref_data,
        const HeterodyneParams& params) = 0;

    // Дечирп с обоими буферами на GPU (OpenCL path ProcessExternal)
    virtual std::vector<std::complex<float>> DechirpWithGPURef(
        void* rx_gpu_ptr,
        void* ref_gpu_ptr,
        const HeterodyneParams& params);
    // default: throws std::runtime_error (не все реализации поддерживают)
};
```

---

## 5. HeterodyneProcessorOpenCL

**Файл**: `include/processors/heterodyne_processor_opencl.hpp`

```cpp
// Тип для профилировочных событий
using HeterodyneOCLProfEvents =
    std::vector<std::pair<const char*, cl_event>>;

class HeterodyneProcessorOpenCL : public IHeterodyneProcessor {
public:
    explicit HeterodyneProcessorOpenCL(IBackend* backend);

    // --- Без профилирования ---
    std::vector<std::complex<float>> Dechirp(rx, ref, params) override;
    std::vector<std::complex<float>> Correct(dc, f_beat_hz, params) override;
    std::vector<std::complex<float>> DechirpFromGPU(rx_ptr, ref, params) override;
    std::vector<std::complex<float>> DechirpWithGPURef(rx_ptr, ref_ptr, params) override;

    // --- С профилированием (для GpuBenchmarkBase) ---
    // pe != nullptr → события записываются в pe для CollectOrRelease
    std::vector<std::complex<float>> Dechirp(rx, ref, params,
        HeterodyneOCLProfEvents* pe);
    std::vector<std::complex<float>> Correct(dc, f_beat_hz, params,
        HeterodyneOCLProfEvents* pe);
    std::vector<std::complex<float>> DechirpFromGPU(rx_ptr, ref, params,
        HeterodyneOCLProfEvents* pe);
    std::vector<std::complex<float>> DechirpWithGPURef(rx_ptr, ref_ptr, params,
        HeterodyneOCLProfEvents* pe);
};
```

---

## 6. HeterodyneProcessorROCm

**Файл**: `include/processors/heterodyne_processor_rocm.hpp`
**Требует**: `ENABLE_ROCM=1` (Linux + AMD GPU). При `!ENABLE_ROCM` — stub, все методы бросают `std::runtime_error`.

```cpp
using HeterodyneROCmProfEvents =
    std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

class HeterodyneProcessorROCm : public IHeterodyneProcessor {
public:
    explicit HeterodyneProcessorROCm(IBackend* backend);

    // Move-constructible, не копируемый
    HeterodyneProcessorROCm(HeterodyneProcessorROCm&&) noexcept;
    HeterodyneProcessorROCm& operator=(HeterodyneProcessorROCm&&) noexcept;

    // --- Без профилирования ---
    std::vector<std::complex<float>> Dechirp(rx, ref, params) override;
    std::vector<std::complex<float>> Correct(dc, f_beat_hz, params) override;
    std::vector<std::complex<float>> DechirpFromGPU(rx_ptr, ref, params) override;
    std::vector<std::complex<float>> DechirpWithGPURef(rx_ptr, ref_ptr, params) override;

    // --- С профилированием ---
    std::vector<std::complex<float>> Dechirp(rx, ref, params,
        HeterodyneROCmProfEvents* pe);
    std::vector<std::complex<float>> Correct(dc, f_beat_hz, params,
        HeterodyneROCmProfEvents* pe);
    std::vector<std::complex<float>> DechirpFromGPU(rx_ptr, ref, params,
        HeterodyneROCmProfEvents* pe);

    static constexpr unsigned int kBlockSize = 256;
};
```

---

## 7. Python — gpuworklib.HeterodyneDechirp

**Файл**: `python/py_heterodyne.hpp`

```python
class HeterodyneDechirp:
    # Constructors
    def __init__(self, ctx: GPUContext)         # OpenCL backend
    def __init__(self, ctx: ROCmGPUContext)     # ROCm backend (если доступен)

    def set_params(self,
                   f_start: float,     # [Гц] начальная частота ЛЧМ
                   f_end: float,       # [Гц] конечная частота ЛЧМ
                   sample_rate: float, # [Гц] частота дискретизации
                   num_samples: int,   # N точек на антенну
                   num_antennas: int   # количество антенн
                   ) -> None

    def process(self,
                rx_data: np.ndarray   # complex64, shape=(num_antennas*num_samples,)
                ) -> dict             # HeterodyneResult как dict

    def process_external(self,
                         gpu_ptr: int  # uintptr_t — GPU-адрес буфера rx
                         ) -> dict

    def get_params(self) -> dict
    # Ключи: 'f_start', 'f_end', 'sample_rate', 'num_samples', 'num_antennas',
    #         'bandwidth', 'duration', 'chirp_rate', 'bin_width'
```

**Структура возвращаемого dict из `process()`**:
```python
{
    'success': bool,
    'error_message': str,
    'antennas': [
        {
            'antenna_idx': int,
            'f_beat_hz': float,
            'f_beat_bin': float,   # дробный бин (параболическая интерполяция)
            'range_m': float,
            'peak_amplitude': float,
            'peak_snr_db': float
        },
        # ... num_antennas элементов
    ],
    'max_positions': [float, ...]  # все найденные максимумы
}
```

---

## 8. Python — gpuworklib.HeterodyneROCm

**Файл**: `python/py_heterodyne_rocm.hpp`
**Требует**: ROCm backend (AMD GPU, Linux)

```python
class HeterodyneROCm:
    def __init__(self, ctx: ROCmGPUContext)

    def set_params(self,
                   f_start: float = 0.0,
                   f_end: float = 2e6,
                   sample_rate: float = 12e6,
                   num_samples: int = 8000,
                   num_antennas: int = 5
                   ) -> None

    def dechirp(self,
                rx: np.ndarray,    # complex64, shape=(num_antennas*num_samples,)
                ref: np.ndarray    # complex64, shape=(num_samples,)
                ) -> np.ndarray    # complex64, shape=(num_antennas*num_samples,)
    # dc[ant*N+n] = conj(rx[ant*N+n] × ref[n])

    def correct(self,
                dc: np.ndarray,          # complex64, shape=(num_antennas*num_samples,)
                f_beat_hz: list[float]   # [num_antennas]
                ) -> np.ndarray          # complex64, shape=(num_antennas*num_samples,)
    # out[ant,n] = dc[ant,n] × exp(j × (−2π×f_beat[ant]/fs) × n)

    @property
    def params(self) -> dict
    # Ключи: 'f_start', 'f_end', 'sample_rate', 'num_samples', 'num_antennas',
    #         'bandwidth', 'chirp_rate'
```

**Сравнение HeterodyneROCm vs HeterodyneDechirp**:

| | `HeterodyneDechirp` | `HeterodyneROCm` |
|-|---------------------|-----------------|
| Backend | OpenCL или ROCm | ROCm only |
| Возвращает | dict (f_beat, range, SNR) | ndarray (dc-сигнал) |
| FFT включён | Да (SpectrumMaximaFinder) | Нет |
| Использование | Готовый радарный результат | Встраивание в свой пайплайн |

---

## 9. Цепочки вызовов

### Стандартный пайплайн (C++)

```cpp
// 1. Создать
HeterodyneDechirp het(backend);

// 2. Настроить
het.SetParams({.f_start=0, .f_end=2e6f, .sample_rate=12e6f,
               .num_samples=8000, .num_antennas=5});

// 3. Обработать
auto result = het.Process(rx_data);

// 4. Использовать результат
for (const auto& a : result.antennas)
    printf("Ant %d: f_beat=%.0f Hz, R=%.1f m\n",
           a.antenna_idx, a.f_beat_hz, a.range_m);
```

### Стандартный пайплайн (Python)

```python
ctx = gpuworklib.GPUContext(0)
het = gpuworklib.HeterodyneDechirp(ctx)
het.set_params(0.0, 2e6, 12e6, 8000, 5)
result = het.process(rx_flat)
for a in result['antennas']:
    print(f"Ant {a['antenna_idx']}: {a['f_beat_hz']:.0f} Hz, {a['range_m']:.1f} m")
```

### GPU-to-GPU пайплайн (OPT-3, C++)

```cpp
// rx уже лежит на GPU (например, после предыдущего этапа)
// OpenCL: cl_mem rx_gpu = ...
auto result = het.ProcessExternal(static_cast<void*>(&rx_gpu), params);
// ВАЖНО: rx_gpu НЕ освобождается фасадом — ваша ответственность
```

### Верификация дечирпа через коррекцию (C++)

```cpp
HeterodyneProcessorOpenCL proc(backend);
auto dc = proc.Dechirp(rx_data, ref_data, params);

// Пик должен быть на bin ≈ 0 после коррекции
std::vector<float> f_beats = {300e3f, 600e3f, 900e3f, 1200e3f, 1500e3f};
auto corrected = proc.Correct(dc, f_beats, params);
// Проверить FFT(corrected)[ant] → peak_bin ≤ 3
```

### ROCm dechirp + Python FFT

```python
ctx = gpuworklib.ROCmGPUContext(0)
het = gpuworklib.HeterodyneROCm(ctx)
het.set_params(f_start=0, f_end=2e6, sample_rate=12e6,
               num_samples=8000, num_antennas=5)

dc = het.dechirp(rx_flat, ref)            # GPU дечирп
dc_2d = dc.reshape(5, 8000)
spectrum = np.fft.fft(dc_2d, axis=1)
f_beat = np.argmax(np.abs(spectrum[:, :4096]), axis=1) * (12e6 / 8192)
```

### Benchmark (C++)

```cpp
// В tests/all_test.hpp:
test_heterodyne_benchmark::run();       // OpenCL
#if ENABLE_ROCM
test_heterodyne_benchmark_rocm::run();  // ROCm
#endif
// Результаты: Results/Profiler/GPU_00_Heterodyne/report.md
```

---

## См. также

- [Full.md](Full.md) — полная документация (математика, pipeline, тесты, бенчмарки)
- [Quick.md](Quick.md) — краткий справочник
- [Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md](../../../Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) — оптимизация HIP/ROCm ядер

---

*Обновлено: 2026-03-09*
