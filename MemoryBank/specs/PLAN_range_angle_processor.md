# 📋 ПЛАН: RangeAngleProcessor — модуль 2D FFT для антенной решётки

> **Дата**: 2026-03-17 | **Автор**: Кодо
> **Архитектура**: Ref03 Unified (6-слойная модель)
> **Платформа**: ROCm/HIP (ветка `main`)
> **Зависимости**: heterodyne, fft_func, signal_generators (все уже есть!)

---

## ✅ Ответы на открытые вопросы

| Вопрос | Ответ | Влияние |
|--------|-------|---------|
| VRAM | **16 ГБ AMD Radeon 9070** | Батч-обработка НЕ нужна, всё влезает |
| Шаг антенн d | **для теста: 1.5 м** (VHF 100 МГц, λ/2) | Угловое разрешение 0.45° |
| Несущая f_carrier | **для теста: 100 МГц** (VHF P-диапазон) | λ = 3 м, d = 1.5 м |
| Девиация | **-5 до +5 МГц** → **B = 10 МГц** | ΔR = 15 м (не 50 м!), TBP = 1 083 000 |
| Число пиков | **2 режима: top-1 и top-10** | Два enum в PeakSearchMode |
| Тест большой | **16×16 × 1 300 000**, куб на GPU | Benchmark |
| Тест малый | **8×8 × 50 000**, скачать куб на CPU | Python визуализация |

### Почему VHF 100 МГц для теста?

```
"Квадратное поле" из 256 антенн = 16×16 URA (2D решётка):

         16 антенн по горизонтали × 16 антенн по вертикали

При d = λ/2 = 1.5 м (f₀=100 МГц):
  Апертура: 16 × 1.5 м = 24 м × 24 м  ← реалистично, физически компактно ✅

При d = λ/2 = 15 м (f₀=10 МГц, OTH):
  Апертура: 16 × 15 м = 240 м × 240 м  ← нереально для теста ❌
```

> Параметры легко менять в `RangeAngleParams` — это просто тестовые дефолты.

---

## 🎯 Что строим

Модуль принимает данные от **16×16 URA** с **ЛЧМ 430→440 МГц** (B=10 МГц, fs=12 МГц),
выполняет 3D FFT и возвращает **3D куб** [16 азимутов × 16 элеваций × 650 000 дальностей].

```
Вход:  [256 × 1 300 000] комплексных IQ  (256 = 16_az × 16_el)
         ↓ Dechirp × conj(ref) + Hamming + zero-pad → 2^21
         ↓ Range FFT batch=256 → crop до 650 000 бинов
         ↓ Transpose [256 × 650K] → [650K × 256]  (= [650K × 16 × 16])
         ↓ 2D Beam FFT hipfftPlanMany([16,16], batch=650K)
         ↓ 2D fftshift (оси az и el)
         ↓ |·|² → float32 куб
Выход: [650K × 16 × 16] 3D куб мощности (float32, 0.33 ГБ)
       + список целей (R, θ_az, θ_el) с параболической интерполяцией

Пересчёт (B=10 МГц, fs=12 МГц, N=1 300 000, f_c=435 МГц):
  ΔR   = c/(2B) = 15 м                   (разрешение по дальности)
  TBP  = B×T   = 1 083 000               (коэффициент сжатия)
  SNR gain     = 60.3 дБ                 (выигрыш от сжатия ЛЧМ)
  R_max        = 9 750 км
  λ    = c/f_c = 0.690 м,  d = 34.5 см   (λ/2 при 435 МГц)
  Δθ   = λ/(N_side×d) = 7.2°/ось         (по азимуту и элевации независимо)
  Апертура     = 5.5 м × 5.5 м           (реалистично для UHF радара)
```

---

## 🗂️ Структура файлов

```
modules/range_angle/
├── include/
│   ├── range_angle_processor.hpp       # Публичный фасад (Layer 6)
│   ├── range_angle_params.hpp          # RangeAngleParams, RangeAngleResult
│   ├── range_angle_types.hpp           # shared_buf::, TargetInfo
│   └── operations/
│       ├── dechirp_window_op.hpp       # Dechirp × conj(ref) + Hamming окно
│       ├── range_fft_op.hpp            # Batch FFT по строкам [256 × nfft_r]
│       ├── transpose_op.hpp            # [256 × N] → [N × 256] (HIP kernel)
│       ├── beam_fft_op.hpp             # Batch FFT по антеннам + fftshift
│       └── peak_search_op.hpp          # |·|² + поиск пиков + парабола
├── src/
│   ├── range_angle_processor.cpp       # Реализация фасада
│   └── transpose_kernel.hip            # HIP kernel: transpose
├── tests/
│   ├── all_test.hpp                    # Точка входа тестов
│   ├── test_range_angle_basic.hpp      # Базовые тесты
│   ├── test_range_angle_benchmark.hpp  # Бенчмарки
│   └── README.md
└── CMakeLists.txt
```

---

## 📦 Параметры и результаты (новые структуры)

### range_angle_params.hpp

```cpp
struct RangeAngleParams {
  // Антенная решётка (2D URA)
  uint32_t n_ant_az     = 16;        // антенн по азимуту  (горизонталь)
  uint32_t n_ant_el     = 16;        // антенн по элевации (вертикаль)
  uint32_t n_samples    = 1'300'000; // точек на антенну

  // Helper: общее число антенн
  uint32_t GetNAntennas() const { return n_ant_az * n_ant_el; }  // 256

  // ЛЧМ-сигнал (baseband)
  float f_start         = -5e6f;     // Гц, начало девиации
  float f_end           = +5e6f;     // Гц, конец девиации
  float sample_rate     = 12e6f;     // Гц, fs

  // FFT / padding
  uint32_t nfft_range   = 0;         // 0 = auto → следующая 2^n от n_samples

  // Антенная физика (перевод бина → угол)
  float antenna_spacing = 0.345f;    // d, метры (тест: λ/2 при 435 МГц = 34.5 см)
  float carrier_freq    = 435e6f;    // f_carrier, Гц (LFM 430–440 МГц → f_c=435)

  // Вычисляемые поля (заполняет SetParams())
  uint32_t n_range_bins = 0;         // полезных бинов ≈ 650 000
  float    range_res_m  = 0.0f;      // ΔR = c / (2B) = 15 м

  // Поиск пиков в 3D кубе
  PeakSearchMode peak_mode = PeakSearchMode::TOP_1;  // TOP_1 или TOP_N (до 10)
  uint32_t       n_peaks   = 1;   // используется при TOP_N

  // Helpers
  float GetBandwidth()  const { return f_end - f_start; }
  float GetDuration()   const { return float(n_samples) / sample_rate; }
  float GetChirpRate()  const { return GetBandwidth() / GetDuration(); }
  float GetDeltaF()     const { return sample_rate / float(n_samples); }
};
```

### range_angle_types.hpp

```cpp
enum class PeakSearchMode {
  TOP_1,   // найти один максимум (быстро, argmax)
  TOP_N,   // найти top-10 пиков (через threshold + сортировка)
};

struct TargetInfo {
  float range_m;          // дальность, метры
  float angle_az_deg;     // азимут, градусы (-53°..+53° при d=λ/2, N_az=16)
  float angle_el_deg;     // элевация, градусы (-53°..+53°)
  float range_bin;        // дробный дальностный бин (после параболы)
  float az_bin;           // дробный азимутальный бин (после параболы)
  float el_bin;           // дробный бин элевации (после параболы)
  float power_db;         // мощность пика, дБ
  float snr_db;           // SNR, дБ
};

struct RangeAngleResult {
  bool success = false;

  // Размеры 3D куба [n_range_bins × n_ant_az × n_ant_el]
  uint32_t n_range_bins = 0;   // 650 000 (B=10МГц, fs=12МГц, N=1.3M)
  uint32_t n_ant_az     = 0;   // 16 (азимут)
  uint32_t n_ant_el     = 0;   // 16 (элевация)

  // 3D куб мощности (CPU, если запрошен download)
  // Layout: [n_range_bins × n_ant_az × n_ant_el], row-major
  std::vector<float> power_cube;

  // GPU-указатель (если download_result=false)
  void* gpu_power_cube = nullptr;

  // Найденные цели
  std::vector<TargetInfo> targets;  // TOP_1: 1 элемент; TOP_N: до 10 элементов

  std::string error_message;
};

// Индексы разделяемых буферов (shared_buf::)
namespace shared_buf {
  static constexpr size_t kInput      = 0;  // [n_ant × n_samples] вход
  static constexpr size_t kRef        = 1;  // [n_samples] опорный ЛЧМ (conj)
  static constexpr size_t kDechirped  = 2;  // [n_ant × nfft_r] после dechirp+window
  static constexpr size_t kRangeFFT   = 3;  // [n_ant × n_range_bins]
  static constexpr size_t kTransposed = 4;  // [n_range_bins × n_ant] = [n_range_bins × n_az × n_el]
  static constexpr size_t kBeamFFT    = 5;  // [n_range_bins × n_az × n_el] 3D куб
  static constexpr size_t kPowerCube  = 6;  // float32 [n_range_bins × n_az × n_el]
  static constexpr size_t kCount      = 7;
}
```

---

## ♻️ Переиспользуемые компоненты

### 1. Генерация опорного ЛЧМ (ref signal) → `LfmConjugateGenerator`

📄 `modules/signal_generators/include/generators/lfm_conjugate_generator.hpp`

```cpp
// Генерируем conj(ref_lfm) один раз при инициализации:
LfmParams lfm_params;
lfm_params.f_start    = params.f_start;  // -5e6 (начало девиации)
lfm_params.f_end      = params.f_end;    // +5e6 (конец девиации) → B = 10 МГц
lfm_params.amplitude  = 1.0;
lfm_params.complex_iq = true;

SystemSampling samp;
samp.fs     = params.sample_rate;        // 12e6
samp.length = params.n_samples;          // 1 300 000

LfmConjugateGenerator gen(backend_, lfm_params);
gen.SetSampling(samp);
void* d_ref_conj = gen.GenerateToGpu();  // [n_samples] complex на GPU
// → сохранить в shared_buf::kRef
```

### 2. Dechirp → ядро из `HeterodyneDechirp`

📄 `modules/heterodyne/include/processors/heterodyne_processor_rocm.hpp`

**Не берём фасад целиком** — берём идею kernel-а: поэлементное умножение `rx[i] * ref_conj[i]`.
Пишем свой `DechirpWindowOp` с Hamming-окном внутри:

```cpp
// kernel: dechirp + window за один проход
__global__ void dechirp_window_kernel(
    const hipComplex* __restrict__ rx,       // [n_ant × n_samples]
    const hipComplex* __restrict__ ref_conj, // [n_samples]
    const float*     __restrict__ window,    // [n_samples] Hamming
    hipComplex*      __restrict__ out,       // [n_ant × nfft_r]
    uint32_t n_ant, uint32_t n_samples, uint32_t nfft_r)
{
    uint32_t ant = blockIdx.y;
    uint32_t i   = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant >= n_ant || i >= nfft_r) return;

    hipComplex val = {0.f, 0.f};
    if (i < n_samples) {
        hipComplex r = rx[ant * n_samples + i];
        hipComplex c = ref_conj[i];
        float w = window[i];
        val.x = (r.x*c.x - r.y*c.y) * w;  // dechirp × window
        val.y = (r.x*c.y + r.y*c.x) * w;
    }
    out[ant * nfft_r + i] = val;           // остальные = 0 (zero-pad)
}
```

### 3. Range FFT → `hipfftExecC2C` batch

📄 hipFFT API: `hipfft/hipfft.h`

```cpp
// RangeFftOp::Initialize()
hipfftPlan1d(&plan_range_, nfft_r, HIPFFT_C2C, n_antennas);  // batch = 256

// RangeFftOp::Execute()
hipfftExecC2C(plan_range_,
    (hipfftComplex*)ctx_->GetShared(shared_buf::kDechirped),
    (hipfftComplex*)ctx_->RequireShared(shared_buf::kRangeFFT, ...),
    HIPFFT_FORWARD);
// После FFT: взять только первые n_range_bins столбцов
```

### 4. Transpose → свой HIP kernel

```cpp
// transpose_kernel.hip — классический tiled transpose для coalesced access
__global__ void transpose_kernel(
    const hipComplex* __restrict__ in,   // [n_ant × n_range_bins]
    hipComplex*       __restrict__ out,  // [n_range_bins × n_ant]
    uint32_t n_ant, uint32_t n_range_bins)
{
    // Tile 32×32 с shared memory для coalesced reads+writes
    __shared__ hipComplex tile[32][33];  // +1 для избежания bank conflicts
    // ... стандартный tiled transpose
}
```

### 5. 2D Beam FFT → `hipfftPlanMany` (после transpose)

```cpp
// BeamFftOp::Initialize()
// 2D Beam FFT: [n_az × n_el] плоскость для каждого из n_range_bins
// Данные: [n_range_bins × n_az × n_el] — для каждого range_bin плоскость [n_az×n_el] contiguous!

int n[2] = { (int)n_ant_az, (int)n_ant_el };   // размер 2D FFT: [16 × 16]
int total_2d = n_ant_az * n_ant_el;             // 256

hipfftPlanMany(
    &plan_beam_,
    2,              // rank = 2D
    n,              // размеры: [16, 16]
    nullptr,        // inembed = null → contiguous
    1,              // istride
    total_2d,       // idist = 256 (расстояние между batch-элементами)
    nullptr,        // onembed
    1,              // ostride
    total_2d,       // odist
    HIPFFT_C2C,
    n_range_bins);  // batch = 650 000

// BeamFftOp::Execute()
hipfftExecC2C(plan_beam_,
    (hipfftComplex*)ctx_->GetShared(shared_buf::kTransposed),   // [650K × 16 × 16]
    (hipfftComplex*)ctx_->RequireShared(shared_buf::kBeamFFT, ...),
    HIPFFT_FORWARD);
// 2D fftshift по осям az и el — HIP kernel (свап квадрантов 16×16 для каждого range bin)
```

### 6. Поиск пиков + парабола → по образцу `MaxValue` из fft_func

📄 `modules/fft_func/include/types/fft_params.hpp` — структура `MaxValue`:

```cpp
// Переиспользуем идею MaxValue (index, magnitude, freq_offset)
// В нашем случае: 2D поиск пика в карте power_map[n_ant × n_range_bins]

// Шаг 1: |·|² → float map (HIP kernel)
// Шаг 2: глобальный argmax → (a_bin, r_bin)
// Шаг 3: параболическая интерполяция по строке и столбцу
// Шаг 4: перевод (r_bin_fine, a_bin_fine) → (R, θ)

float delta_r = 0.5f * (left_r - right_r) / (left_r - 2*center_r + right_r);
float delta_a = 0.5f * (left_a - right_a) / (left_a - 2*center_a + right_a);
float r_fine  = r_bin + delta_r;
float a_fine  = a_bin + delta_a;

// Бин → дальность
float f_beat = r_fine * params.sample_rate / params.nfft_range;
float R = f_beat * 3e8f / (2.f * params.GetChirpRate());

// Бин → угол (два угла: азимут и элевация)
// После fftshift: центр = n_ant/2
float k_az = az_fine - params.n_ant_az / 2.f;
float k_el = el_fine - params.n_ant_el / 2.f;

// sin(θ) = k × 2 / N_side  (т.к. d = λ/2, λ = c/f_c)
float sin_az = k_az * 2.f / float(params.n_ant_az);   // = k_az / 8
float sin_el = k_el * 2.f / float(params.n_ant_el);   // = k_el / 8

float theta_az_deg = degrees(asinf(clamp(sin_az, -1.f, 1.f)));  // ±53.1° max
float theta_el_deg = degrees(asinf(clamp(sin_el, -1.f, 1.f)));
```

---

## 🏗️ Публичный API фасада

```cpp
// range_angle_processor.hpp
class RangeAngleProcessor {
public:
  explicit RangeAngleProcessor(IBackend* backend);
  ~RangeAngleProcessor();

  // No copy, move OK
  RangeAngleProcessor(const RangeAngleProcessor&) = delete;
  RangeAngleProcessor& operator=(const RangeAngleProcessor&) = delete;
  RangeAngleProcessor(RangeAngleProcessor&&) noexcept;
  RangeAngleProcessor& operator=(RangeAngleProcessor&&) noexcept;

  // Установить параметры (пересчитывает nfft, n_range_bins, etc.)
  void SetParams(const RangeAngleParams& params);
  const RangeAngleParams& GetParams() const;

  // Обработка — CPU вход (upload → GPU → download)
  // data: [n_ant_az × n_ant_el × n_samples], row-major (= [256 × n_samples] flat)
  RangeAngleResult Process(
      const std::vector<std::complex<float>>& data,
      bool download_result = true);   // false = 3D куб остаётся на GPU

  // Обработка — GPU вход (данные уже на устройстве)
  RangeAngleResult ProcessFromGPU(
      void* gpu_data,                 // hipDeviceptr_t [n_ant × n_samples]
      bool download_result = true);

private:
  void EnsureCompiled();
  void UploadData(const std::complex<float>* data, size_t count);
  void BuildRefSignal();             // один раз при SetParams()

  IBackend* backend_;
  GpuContext ctx_;

  // Layer 5 Ops
  DechirpWindowOp   dechirp_op_;
  RangeFftOp        range_fft_op_;
  TransposeOp       transpose_op_;
  BeamFftOp         beam_fft_op_;
  PeakSearchOp      peak_op_;

  // hipFFT планы (создаются при SetParams)
  hipfftHandle plan_range_ = 0;
  hipfftHandle plan_beam_  = 0;

  RangeAngleParams params_;
  bool compiled_ = false;
  bool ref_built_ = false;
};
```

---

## 📋 Шаги разработки

### Фаза 1 — Скелет и параметры (1-2 ч)

- [ ] Создать `modules/range_angle/` — всю структуру папок
- [ ] Написать `range_angle_params.hpp` — структуры параметров
- [ ] Написать `range_angle_types.hpp` — shared_buf::, TargetInfo, Result
- [ ] Написать `CMakeLists.txt` — по образцу `modules/statistics/`
- [ ] Добавить `add_subdirectory(modules/range_angle)` в корневой CMakeLists
- [ ] Убедиться что проект собирается (пустой фасад)

### Фаза 2 — Генерация опорного сигнала (1 ч)

- [ ] В `RangeAngleProcessor::BuildRefSignal()`:
  - Использовать `LfmConjugateGenerator` из `signal_generators`
  - Сгенерировать `conj(ref_lfm)` на GPU → `shared_buf::kRef`
  - Сохранить в буфере — пересоздавать только при смене параметров

> 📄 API: `modules/signal_generators/include/generators/lfm_conjugate_generator.hpp`
> Конструктор: `LfmConjugateGenerator(backend, lfm_params)`
> Метод: `GenerateToGpu()` → `cl_mem` / `hipDeviceptr_t`

### Фаза 3 — DechirpWindowOp (2 ч)

- [ ] Написать HIP kernel `dechirp_window_kernel` в `.hip` файле:
  - Вход: `rx[n_ant × n_samples]`, `ref_conj[n_samples]`, `window[n_samples]`
  - Выход: `out[n_ant × nfft_r]` (с zero-padding до nfft_r)
  - Один поток — одна точка: dechirp + умножение на Hamming за один проход
  - Нули за пределами n_samples (zero-pad)
- [ ] Написать `DechirpWindowOp` — Layer 5 Op, вызывает kernel
- [ ] Окно Хэмминга предвычислить при инициализации (буфер на GPU)

> 📄 Референс: `modules/heterodyne/` — как устроен dechirp kernel
> 📄 Референс: `modules/statistics/include/operations/mean_reduction_op.hpp` — структура Op

### Фаза 4 — Range FFT Op (1 ч)

- [ ] Написать `RangeFftOp`:
  - `Initialize()`: `hipfftPlan1d(&plan_, nfft_r, HIPFFT_C2C, n_antennas)` batch=256
  - `Execute()`: `hipfftExecC2C(plan_, d_dechirped, d_range_fft, HIPFFT_FORWARD)`
  - После FFT: crop до `n_range_bins` столбцов (memcpy 2D или stride view)
- [ ] Деструктор: `hipfftDestroy(plan_)`

> 📄 API: `hipfft/hipfft.h` → `hipfftPlan1d`, `hipfftExecC2C`

### Фаза 5 — Transpose Op (1-2 ч)

- [ ] Написать tiled transpose HIP kernel:
  - Tile 32×32 со shared memory (избегаем bank conflicts через +1 padding)
  - `in[n_ant × n_range_bins]` → `out[n_range_bins × n_ant]`
  - Проверить на матрицах не кратных 32
- [ ] Написать `TransposeOp`

> 📄 Референс: `Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md` — паттерны transpose

### Фаза 6 — 2D Beam FFT + 2D fftshift Op (1.5 ч)

- [ ] Написать `BeamFftOp`:
  - `Initialize()`: `hipfftPlanMany(2, [16,16], ..., batch=n_range_bins)` — 2D FFT каждой плоскости
  - `Execute()`:
    1. `hipfftExecC2C` — 2D Beam FFT для всех 650K range-bins одновременно
    2. Запустить `fftshift_2d_kernel` — свапает 4 квадранта в каждой [16×16] плоскости
- [ ] `fftshift_2d_kernel`: для каждого range-bin k:
  - Свапает квадранты [0..7, 0..7] ↔ [8..15, 8..15]  и  [0..7, 8..15] ↔ [8..15, 0..7]
  - Grid: `(n_range_bins, 8, 8)` — каждый thread обрабатывает одну пару точек

### Фаза 7 — PeakSearchOp (2 ч)

- [ ] HIP kernel `magnitude_sq_kernel`: `out[i] = re²+im²` → float куб `[650K × 16 × 16]`
- [ ] HIP kernel или rocThrust `argmax_3d`: найти максимум в кубе → `(r_bin, az_bin, el_bin)`
- [ ] CPU код: параболическая интерполяция по трём осям → `(r_fine, az_fine, el_fine)`
- [ ] CPU код: бин → `R (м)`, `θ_az (°)`, `θ_el (°)` (формулы из раздела выше)
- [ ] TOP_N режим: порог = 0.1×max, найти все пики выше порога, отсортировать

> 📄 Референс: `modules/fft_func/include/types/fft_params.hpp` — структура `MaxValue`
> Поле `freq_offset` — дробная поправка от параболы, `refined_frequency` — готовая частота

### Фаза 8 — Сборка фасада (1 ч)

- [ ] Реализовать `RangeAngleProcessor::Process()`:
  ```
  UploadData() → EnsureCompiled() → BuildRefSignal()
  → dechirp_op_.Execute()
  → range_fft_op_.Execute()
  → transpose_op_.Execute()
  → beam_fft_op_.Execute()
  → peak_op_.Execute()
  → backend_->Synchronize()
  → читаем результат
  ```
- [ ] Реализовать `ProcessFromGPU()` — пропускаем UploadData()
- [ ] Добавить профилирование через GPUProfiler (SetGPUInfo перед Start!)

### Фаза 9 — Тесты (2-3 ч)

- [ ] `test_range_angle_basic.hpp`:
  - Тест 1: сгенерировать ЛЧМ через `LfmConjugateGenerator`, задержать на τ, запустить процессор, проверить что R совпадает с τ×c/2
  - Тест 2: два ЛЧМ с разными задержками — два пика на разных дальностях
  - Тест 3: сигнал под углом (синтетические данные) — пик на правильном угловом бине

- [ ] `test_range_angle_benchmark.hpp`:
  - **Тест БОЛЬШОЙ** (`n_ant_az=16, n_ant_el=16, n_samples=1_300_000`):
    - Синтетические данные (ЛЧМ + шум) генерировать на GPU
    - 3D куб `[650K × 16 × 16]` **НЕ скачивать** — только время и пики
    - Профилировать через GPUProfiler, сравнить с Capon (~22× быстрее)
  - **Тест МАЛЫЙ** (`n_ant_az=8, n_ant_el=8, n_samples=50_000`):
    - Загружать данные с CPU, **скачивать 3D куб** `[25K × 8 × 8]` на CPU
    - Сохранять в JSON для Python визуализации
    - Python тест: `Python_test/range_angle/test_range_angle.py` рисует срезы куба
    - График: `Results/Plots/range_angle/`

- [ ] `tests/README.md` — описание тестов

- [ ] Подключить в `src/main.cpp` через `all_test.hpp`

### Фаза 10 — Python биндинги (1 ч)

- [ ] `python/py_range_angle.hpp`:
  ```python
  proc = RangeAngleProcessor(context)
  proc.set_params(n_ant_az=8, n_ant_el=8, n_samples=50_000,
                  f_start=-5e6, f_end=5e6, sample_rate=12e6,
                  carrier_freq=435e6, antenna_spacing=0.345,
                  peak_mode="TOP_1")
  result = proc.process(data)   # numpy [64, 50000] complex64  (8×8 антенн)
  # result.power_cube: numpy [25000, 8, 8] float32  (3D куб малого теста)
  # result.targets: list of {range_m, angle_az_deg, angle_el_deg, power_db}
  ```
- [ ] Python тест: `Python_test/range_angle/test_range_angle.py`
  - Малый тест 8×8×50K: скачать 3D куб, нарисовать срезы
  - `cube[:, :, r_bin]` — угловая карта на фиксированной дальности (imshow 8×8)
  - `cube[az_bin, el_bin, :]` — дальностный профиль (plot)
  - Проверить что пик на правильной дальности и угле (синтетические данные)
- [ ] График 3D куба в `Results/Plots/range_angle/`
  - `range_angle_az_slice.png` — срез R-азимут (для фиксированной элевации)
  - `range_angle_el_slice.png` — срез R-элевация (для фиксированного азимута)
  - `angular_map_Rkm.png` — угловая карта 8×8 на конкретной дальности

---

## ⚠️ Важные детали

### Память (16×16 URA × 1.3М × complex64, B=10 МГц → 650K бинов)

```
Входные данные:    256 × 1 300 000 × 8  =  2.66 ГБ
После dechirp:     256 × 2 097 152 × 8  =  4.30 ГБ  (nfft_r = 2^21, zero-pad)
После Range FFT:   256 × 650 000  × 8  =  1.33 ГБ  (crop до 650K!)
После Transpose:   650K × 256    × 8  =  1.33 ГБ  (= 650K × 16 × 16 — просто view!)
После 2D Beam FFT: 650K × 16×16  × 8  =  1.33 ГБ  (ТОГО ЖЕ РАЗМЕРА что и Range FFT!)
Куб мощности:      650K × 16×16  × 4  =  0.33 ГБ  (float32 — в 4 раза меньше!)

Пиковое потребление: input + dechirped = 2.66 + 4.30 ≈ 7.0 ГБ
→ AMD Radeon 9070 (16 ГБ): батч-обработка НЕ нужна ✅

Бонус: куб мощности 0.33 ГБ — можно хранить несколько кубов (история целей)!
```

### Батч-обработка (если VRAM < 8 ГБ — на будущее)

```cpp
uint32_t batch = min(n_antennas, max_batch_);  // определять из VRAM
for (uint32_t start = 0; start < n_antennas; start += batch) {
    ProcessBatch(data, start, batch);
}
// Аккумулировать Range FFT результаты, потом Beam FFT
```

### nfft_r — почему не просто N=1.3М

hipFFT работает быстро на `2^n`. Но `2^20 = 1 048 576 < 1 300 000`, значит минимум `2^21 = 2 097 152`.
Нули добавляются в конце буфера — dechirp_window_kernel пишет 0 для `i >= n_samples`.

### 2D fftshift для Beam FFT

После 2D hipFFT каждая [16×16] плоскость расположена 0..15 × 0..15.
Нужно циклически сдвинуть оба измерения: `[8..15, 8..15, 0..7, 0..7]` → центр = (0°, 0°).
Это своп четырёх квадрантов 8×8:

```
До fftshift:   После fftshift:
Q3 | Q4        Q1 | Q2
───┼───    →   ───┼───
Q1 | Q2        Q3 | Q4

(Q1..Q4 — квадранты 8×8, Q1 = верхний-левый)
```

Реализация: HIP kernel с `blockIdx.z = range_bin`, `blockDim.x = blockDim.y = 8`.
Альтернатива — фазовый множитель `(-1)^(az+el)` перед FFT (менее читаемо, но без доп. kernel).

---

## 🔗 Ссылки на код в проекте

| Что | Файл | Зачем |
|-----|------|-------|
| LfmConjugateGenerator | [lfm_conjugate_generator.hpp](modules/signal_generators/include/generators/lfm_conjugate_generator.hpp) | Генерация conj(ref) на GPU |
| HeterodyneDechirp | [heterodyne_dechirp.hpp](modules/heterodyne/include/heterodyne_dechirp.hpp) | Референс: dechirp kernel |
| MaxValue + парабола | [fft_params.hpp](modules/fft_func/include/types/fft_params.hpp) | Структура результата пика |
| ISpectrumProcessor | [i_spectrum_processor.hpp](modules/fft_func/include/interface/i_spectrum_processor.hpp) | Референс: batch FFT API |
| StatisticsProcessor | [statistics_processor.hpp](modules/statistics/include/statistics_processor.hpp) | Референс: полный Ref03 фасад |
| MeanReductionOp | [mean_reduction_op.hpp](modules/statistics/include/operations/mean_reduction_op.hpp) | Референс: Layer 5 Op структура |
| CaponProcessor | [capon_processor.hpp](modules/capon/include/capon_processor.hpp) | Референс: сложный Ref03 фасад |
| GpuContext | [gpu_context.hpp](DrvGPU/interface/gpu_context.hpp) | Базовый per-module контекст |
| GPUProfiler SetGPUInfo | [GPUProfiler_SetGPUInfo.md](Examples/GPUProfiler_SetGPUInfo.md) | Профилирование (обязательно до Start!) |
| ROCm оптимизация | [Info_ROCm_HIP_Optimization_Guide.md](Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) | Паттерны transpose, coalesced |

---

## ✅ Все вопросы закрыты

> Все ответы — в таблице «Ответы на открытые вопросы» в начале документа.
> Параметры по умолчанию обновлены в `RangeAngleParams` и тестовых сценариях.