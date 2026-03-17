# RangeAngleProcessor — API Reference

> Справочник по всем публичным классам, методам и типам модуля `range_angle`

**Namespace**: `range_angle` | **Backend**: ROCm only (`ENABLE_ROCM=1`)
**Python module**: `gpu_worklib`

---

## Содержание

1. [PeakSearchMode](#1-peaksearchmode)
2. [RangeAngleParams](#2-rangeangleparams)
3. [TargetInfo](#3-targetinfo)
4. [RangeAngleResult](#4-rangeangleresult)
5. [RangeAngleProcessor](#5-rangeangleprocessor)
6. [Shared buffer constants](#6-shared-buffer-constants)
7. [Python API](#7-python-api)
8. [Цепочки вызовов](#8-цепочки-вызовов)

---

## 1. PeakSearchMode

**Файл**: `include/range_angle_types.hpp`

```cpp
enum class PeakSearchMode {
  TOP_1,  // Один глобальный максимум
  TOP_N,  // До n_peaks пиков (TODO)
};
```

**Python**: `gw.RangeAnglePeakMode.TOP_1`, `gw.RangeAnglePeakMode.TOP_N`

---

## 2. RangeAngleParams

**Файл**: `include/range_angle_params.hpp`

### Поля

| Поле | Тип | Умолчание | Описание |
|------|-----|-----------|---------|
| `n_ant_az` | `uint32_t` | 16 | Антенн по азимуту |
| `n_ant_el` | `uint32_t` | 16 | Антенн по элевации |
| `n_samples` | `uint32_t` | 1'300'000 | Отсчётов на антенну |
| `f_start` | `float` | -5e6f | Начало ЛЧМ (baseband), Гц |
| `f_end` | `float` | +5e6f | Конец ЛЧМ (baseband), Гц |
| `sample_rate` | `float` | 12e6f | Частота дискретизации, Гц |
| `nfft_range` | `uint32_t` | 0 | 0 = авто (следующая 2^n ≥ n_samples) |
| `antenna_spacing` | `float` | 0.345f | Расстояние между антеннами, м |
| `carrier_freq` | `float` | 435e6f | Несущая частота, Гц |
| `peak_mode` | `PeakSearchMode` | TOP_1 | Режим поиска пиков |
| `n_peaks` | `uint32_t` | 1 | Макс. число пиков (TOP_N) |

### Вычисляемые поля (заполняет `SetParams`)

| Поле | Тип | Формула |
|------|-----|---------|
| `n_range_bins` | `uint32_t` | `nfft_range / 2` |
| `range_res_m` | `float` | `c / (2·B)` |

### Методы

```cpp
uint32_t GetNAntennas()  const;  // n_ant_az * n_ant_el
float    GetBandwidth()  const;  // f_end - f_start
float    GetDuration()   const;  // n_samples / sample_rate
float    GetChirpRate()  const;  // GetBandwidth() / GetDuration()
```

---

## 3. TargetInfo

**Файл**: `include/range_angle_types.hpp`

```cpp
struct TargetInfo {
  float range_m;       // Дальность, м
  float angle_az_deg;  // Азимут, градусы
  float angle_el_deg;  // Элевация, градусы
  float range_bin;     // Дробный дальностный бин (после параболы)
  float az_bin;        // Дробный азимутальный бин
  float el_bin;        // Дробный элевационный бин
  float power_db;      // Мощность пика, дБ
  float snr_db;        // SNR, дБ (не вычислен = 0)
};
```

---

## 4. RangeAngleResult

**Файл**: `include/range_angle_types.hpp`

```cpp
struct RangeAngleResult {
  bool                    success = false;
  uint32_t                n_range_bins = 0;
  uint32_t                n_ant_az = 0;
  uint32_t                n_ant_el = 0;
  std::vector<float>      power_cube;        // [n_range_bins × n_az × n_el], float32
                                              // только при download_result=true
  void*                   gpu_power_cube = nullptr;  // GPU-указатель на kPowerCube
  std::vector<TargetInfo> targets;
  std::string             error_message;
};
```

---

## 5. RangeAngleProcessor

**Файл**: `include/range_angle_processor.hpp`

### Конструктор

```cpp
explicit RangeAngleProcessor(drv_gpu_lib::IBackend* backend);
```

Создаёт процессор, привязанный к `backend`. Компиляция kernels — ленивая (при первом `Process`).

### Деструктор

```cpp
~RangeAngleProcessor();
```

Освобождает все Op-ы и `GpuContext` (hipfft планы, GPU буферы).

### Move semantics

```cpp
RangeAngleProcessor(RangeAngleProcessor&&) noexcept;
RangeAngleProcessor& operator=(RangeAngleProcessor&&) noexcept;
// Copy — удалён
```

### Методы конфигурации

```cpp
void SetParams(const RangeAngleParams& params);
```
Устанавливает параметры. Вычисляет `n_range_bins`, `range_res_m`, `nfft_range`. Сбрасывает `compiled_` и `ref_built_` — следующий `Process` перекомпилирует kernels и перегенерирует опорный сигнал.

```cpp
const RangeAngleParams& GetParams() const;
```

### Основной метод

```cpp
RangeAngleResult Process(
    const std::vector<std::complex<float>>& data,
    bool download_result = true);
```

**Параметры**:
- `data` — IQ-данные `[n_ant × n_samples]`, antenna-major layout. Размер должен быть равен `GetParams().GetNAntennas() * GetParams().n_samples`.
- `download_result` — если `true`, копирует `power_cube` GPU→CPU в `result.power_cube`. Если `false` — только `gpu_power_cube` в результате.

**Возвращает**: `RangeAngleResult`. При ошибке: `success=false`, `error_message` заполнен.

**Внутренний порядок**:
1. `EnsureCompiled()` — lazy hiprtc JIT + hipfft планы
2. `BuildRefSignal()` — lazy генерация conj(ref_lfm) на GPU
3. `UploadData()` — hipMemcpyAsync H2D
4. `DechirpWindowOp::Execute` → `RangeFftOp::Execute` → `TransposeOp::Execute` → `BeamFftOp::Execute` → `PeakSearchOp::Execute`

---

## 6. Shared buffer constants

**Файл**: `include/range_angle_types.hpp`, namespace `range_angle::shared_buf`

```cpp
static constexpr size_t kInput      = 0;  // IQ-данные
static constexpr size_t kRef        = 1;  // conj(ref_lfm)
static constexpr size_t kDechirped  = 2;  // после dechirp+window; in-place Range FFT
static constexpr size_t kRangeFFT   = 3;  // alias kDechirped
static constexpr size_t kTransposed = 4;  // после transpose; in-place Beam FFT
static constexpr size_t kBeamFFT    = 5;  // alias kTransposed
static constexpr size_t kPowerCube  = 6;  // float power cube
static constexpr size_t kCount      = 7;
```

---

## 7. Python API

**Модуль**: `gpu_worklib` (после `cmake --build build -- python_bindings`)

### Классы

```python
gw.RangeAnglePeakMode     # enum: TOP_1, TOP_N
gw.RangeAngleParams       # параметры
gw.TargetInfo             # результат: range_m, angle_az_deg, angle_el_deg, ...
gw.RangeAngleResult       # полный результат
gw.RangeAngleProcessor    # основной класс
```

### RangeAngleParams (Python)

```python
p = gw.RangeAngleParams()
p.n_ant_az        # int, умолч. 16
p.n_ant_el        # int, умолч. 16
p.n_samples       # int, умолч. 1_300_000
p.f_start         # float, умолч. -5e6
p.f_end           # float, умолч. +5e6
p.sample_rate     # float, умолч. 12e6
p.nfft_range      # int, умолч. 0 (авто)
p.carrier_freq    # float, умолч. 435e6
p.antenna_spacing # float, умолч. 0.345
p.peak_mode       # RangeAnglePeakMode, умолч. TOP_1
p.n_peaks         # int, умолч. 1

# Read-only (заполняются после set_params):
p.n_range_bins    # int
p.range_res_m     # float

# Методы:
p.get_n_antennas()  # -> int
p.get_bandwidth()   # -> float
p.get_duration()    # -> float
p.get_chirp_rate()  # -> float
```

### RangeAngleProcessor (Python)

```python
proc = gw.RangeAngleProcessor(ctx)         # ctx = gw.ROCmGPUContext(0)
proc.set_params(p)                          # RangeAngleParams
params = proc.get_params()                  # -> RangeAngleParams (copy)
result = proc.process(data,                 # np.ndarray complex64, flatten
                      download_result=True) # -> RangeAngleResult
```

### RangeAngleResult (Python)

```python
result.success         # bool
result.n_range_bins    # int
result.n_ant_az        # int
result.n_ant_el        # int
result.targets         # List[TargetInfo]
result.error_message   # str
result.power_cube_numpy()  # -> np.ndarray float32, shape (n_range_bins, n_az, n_el)
                            # доступен только при download_result=True
```

### TargetInfo (Python)

```python
tgt = result.targets[0]
tgt.range_m         # float, дальность в метрах
tgt.angle_az_deg    # float, азимут в градусах
tgt.angle_el_deg    # float, элевация в градусах
tgt.range_bin       # float, дробный дальностный бин
tgt.az_bin          # float, дробный азимутальный бин
tgt.el_bin          # float, дробный элевационный бин
tgt.power_db        # float, мощность пика в дБ
tgt.snr_db          # float, SNR в дБ (= 0, не вычислен)
repr(tgt)           # "TargetInfo(R=75000m az=0.0deg el=0.0deg pwr=42.1dB)"
```

---

## 8. Цепочки вызовов

### Типичный C++ workflow

```cpp
// Инициализация (1 раз)
range_angle::RangeAngleProcessor proc(backend);
range_angle::RangeAngleParams p;
p.n_ant_az = 16; p.n_ant_el = 16; p.n_samples = 1'300'000;
proc.SetParams(p);
// ↳ ComputeDerivedParams: nfft=2^21, n_range_bins=1048576, range_res=15м

// Обработка (повторно)
auto result = proc.Process(iq_data);          // lazy compile + pipeline
// ↳ EnsureCompiled → BuildRefSignal → Upload → 5 ops → Sync

// Только GPU (без D2H)
auto r2 = proc.Process(iq_data, false);
// ↳ r2.gpu_power_cube — GPU-указатель, r2.power_cube — пустой

// Смена параметров (сбросит compiled_, ref_built_)
p.n_ant_az = 8; proc.SetParams(p);
// ↳ Следующий Process перекомпилирует kernels
```

### Типичный Python workflow

```python
ctx = gw.ROCmGPUContext(0)
proc = gw.RangeAngleProcessor(ctx)

p = gw.RangeAngleParams()
p.n_ant_az = 8; p.n_ant_el = 8; p.n_samples = 50_000
proc.set_params(p)

iq = make_iq_data(...)  # np.ndarray complex64, size = n_ant * n_samples
result = proc.process(iq, download_result=True)

cube = result.power_cube_numpy()                     # [n_rbins, 8, 8]
R = result.targets[0].range_m                        # дальность
az = result.targets[0].angle_az_deg                  # азимут
```

### Workflow с профилированием

```cpp
drv_gpu_lib::GPUProfiler profiler;
profiler.SetGPUInfo(backend->GetDeviceIndex(),       // ⚠️ перед Start()!
                    backend->GetDeviceName(),
                    backend->GetDriverVersion());
profiler.Start();

for (int i = 0; i < 5; i++) {
    profiler.Mark("iter_" + std::to_string(i));
    proc.Process(data, false);                       // без D2H для бенчмарка
}

profiler.Stop();
profiler.PrintReport();                              // вывод в консоль
profiler.ExportJSON("Results/Profiler/range_angle.json");
```
