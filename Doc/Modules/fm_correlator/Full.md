# FM Correlator -- Полная документация

> Корреляция в частотной области для сигналов с фазовой модуляцией M-последовательностями (ФМ)

**Namespace**: `drv_gpu_lib`
**Каталог**: `modules/fm_correlator/`
**Backend**: ROCm-only (hipFFT / rocFFT + HIP kernels)
**Зависимости**: DrvGPU, hip::host, hip::hipfft
**Основа**: ROCm 7.2, amdclang++

📖 **Справочник API**: [API.md](API.md) | [Quick.md](Quick.md)

### статьи
GPU-Accelerated Signal Processing for Passive Bistatic Radar
Efficient GPU-accelerated parallel cross-correlation 

https://www.mdpi.com/2072-4292/15/22/5421
https://www.sciencedirect.com/science/article/abs/pii/S0743731525000218



---

## 1. Назначение

Модуль `fm_correlator` выполняет **корреляционную обработку** сигналов с фазовой модуляцией (ФМ) M-последовательностями на GPU.

**Задача**: по набору принятых сигналов и известной опорной M-последовательности определить корреляционные пики, которые указывают на наличие и задержку сигнала.

**Применение**:
- Обнаружение сигналов с ФМ (фазокодовая манипуляция)
- Оценка задержки распространения
- Многоканальная корреляционная обработка (S входных сигналов × K циклических сдвигов)

---

## 2. Входные данные

| Данные | Тип | Размерность | Описание |
|--------|-----|-------------|----------|
| `ref` (опорный) | **float** | `[N]` | M-последовательность {+1.0, -1.0} |
| `inp` (входные) | **float** | `[S × N]` | Принятые сигналы (вещественные) |

**Выход**: `peaks[S × K × n_kg]` float -- магнитуды корреляционных пиков.

> Оба входа -- float. M-sequence генератор возвращает float напрямую.

---

## 3. Математика алгоритма

### 3.1. M-последовательность (генератор)

LFSR (Linear Feedback Shift Register) с полиномом `0xB8000000` (степень 31):

```
LFSR[0] = seed (по умолчанию 0x1)
for i = 0..N-1:
    bit = (LFSR >> 31) & 1
    ref[i] = bit ? +1.0f : -1.0f      // сразу float
    LFSR = bit ? (LFSR << 1) ^ 0xB8000000 : (LFSR << 1)
```

### 3.2. Циклические сдвиги

Для каждого сдвига `k = 0..K-1`:
```
ref_shifted[k][i] = ref[(i + k) mod N]
```

### 3.3. Корреляция в частотной области

Теорема о корреляции:
```
corr(ref, inp) = IFFT{ conj(FFT{ref}) · FFT{inp} }
```

### 3.4. Формулы по шагам

**Step 1** -- Подготовка опорных спектров (один раз при смене ref):
```
ref_complex[k][i] = float2(ref[(i+k) % N], 0.0)   // float -> float2 + cyclic shifts
ref_fft[k] = FFT(ref_complex[k])                    // C2C Forward, batch=K
```

**Step 2** -- FFT входных сигналов:
```
inp_fft[s] = R2C_FFT(inp[s])    // R2C: float -> float2, N/2+1 точек (hermitian)
```

> R2C FFT принимает float напрямую -- ядро конвертации real_to_complex **не нужно**.

**Step 3** -- Корреляция:
```
// Для каждой пары (signal s, shift k):
corr_fft[s][k][i] = conj(ref_fft[k][i]) * inp_fft[s][i]     // conj fused в умножение
corr_time[s][k] = C2R_IFFT(corr_fft[s][k])                   // C2R -> float результат
peaks[s][k][j] = |corr_time[s][k][j]| / N,  j = 0..n_kg-1   // нормализация (hipFFT не нормирует!)
```

> **hipFFT не нормирует IFFT**. Деление на N обязательно при извлечении магнитуд.

### 3.5. Размерности данных

| Переменная | Размерность | Тип |
|------------|-------------|-----|
| `ref` | `[N]` | float |
| `inp` | `[S × N]` | float |
| `ref_complex` | `[K × N]` | float2 |
| `ref_fft` | `[K × N]` | float2 (in-place с ref_complex) |
| `inp_fft` | `[S × (N/2+1)]` | float2 (hermitian!) |
| `corr_fft` | `[S × K × (N/2+1)]` | float2 |
| `corr_time` | `[S × K × N]` | float (C2R выход) |
| `peaks` | `[S × K × n_kg]` | float |

> **N/2+1** -- R2C FFT использует hermitian symmetry, спектральные буферы вдвое меньше.

---

## 4. Pipeline

### 4.1. Схема с двумя HIP streams

```
Stream 0 (опорный — один раз при init/смене ref):
  H2D(ref_float[N])
  → kernel: apply_cyclic_shifts [grid(N/256, K)]
       float → float2 + cyclic shifts (imag=0)
       → ref_complex[K × N]
  → hipfftExecC2C(FORWARD, batch=K)
       → ref_fft[K × N]

Stream 1 (входные — каждый вызов Process()):
  H2D(inp_float[S × N])
  → hipfftExecR2C(FORWARD, batch=S)
       → inp_fft[S × (N/2+1)]

hipStreamSynchronize(stream0);
hipStreamSynchronize(stream1);

Stream 0:
  → kernel: multiply_conj_fused [grid((N/2+1)/256, K, S)]
       conj(ref_fft[k]) * inp_fft[s] — conj inline, один проход
       → corr_fft[S × K × (N/2+1)]
  → hipfftExecC2R(INVERSE, batch=S×K)
       → corr_time[S × K × N] float
  → kernel: extract_magnitudes [grid(n_kg/256, K, S)]
       |corr_time| / N → peaks[S × K × n_kg]
  → D2H(peaks)
```

### 4.2. Порядок GPU-операций

```
H2D(ref)  ──→  apply_cyclic_shifts  ──→  hipfftExecC2C(FWD) ──┐
                                                               ├──→ multiply_conj_fused
H2D(inp)  ──→  hipfftExecR2C(FWD)  ──────────────────────────┘          │
                                                                          ▼
                                                               hipfftExecC2R(INV)
                                                                          │
                                                                          ▼
                                                               extract_magnitudes
                                                                          │
                                                                          ▼
                                                                    D2H(peaks)
```

### 4.3. Сравнение с исходным планом

| Операция | Исходный план | Оптимизированный |
|----------|--------------|------------------|
| real_to_complex kernel | Отдельный kernel launch | **Убран** -- R2C FFT напрямую |
| complex_conjugate kernel | Отдельный kernel launch | **Убран** -- fused в multiply |
| complex_multiply | Отдельный kernel + conj буфер | Один kernel, conj inline |
| inp_complex буфер [S×N×8] | Нужен | **Не нужен** (R2C принимает float) |
| Спектральные буферы | N точек | **N/2+1** точек (hermitian) |
| IFFT выход | float2 (C2C) | **float** (C2R), вдвое меньше |
| Kernel launches на Process() | 5 | **3** |

---

## 5. HIP-ядра

Всего **3 production-ядра** + **1 utility-ядро** для тестирования.

### 5.1. `apply_cyclic_shifts` -- float -> float2 + циклические сдвиги

Раскладывает float-опорник в float2 массив с K циклическими сдвигами.

```cpp
__global__ void apply_cyclic_shifts(
    const float* __restrict__ ref,     // [N] float
    float2*      __restrict__ out,     // [K × N] float2
    int N, int num_shifts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (i >= N || k >= num_shifts) return;

    int src = (i + k) % N;
    out[k * N + i] = make_float2(ref[src], 0.0f);
}
// Grid: ((N+255)/256, K), Block: (256)
```

### 5.2. `multiply_conj_fused` -- conj(ref) × inp за один проход

Ключевая оптимизация: сопряжение ref_fft выполняется inline (a.y = -a.y), нет отдельного прохода по памяти и отдельного буфера ref_fft_conj.

R2C/C2R работают с **N/2+1** точками (hermitian symmetry).

```cpp
__global__ void multiply_conj_fused(
    const float2* __restrict__ ref_fft,    // [K × (N/2+1)] — НЕ сопряжённый
    const float2* __restrict__ inp_fft,    // [S × (N/2+1)]
    float2*       __restrict__ corr_fft,   // [S × K × (N/2+1)]
    int half_N, int K, int S)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int s = blockIdx.z;
    if (i >= half_N || k >= K || s >= S) return;

    float2 a = ref_fft[k * half_N + i];
    a.y = -a.y;                              // conj inline
    float2 b = inp_fft[s * half_N + i];

    corr_fft[(s * K + k) * half_N + i] = make_float2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}
// Grid: ((half_N+255)/256, K, S), Block: (256)
```

**LDS-оптимизация**: `ref_fft[k]` перечитывается S раз -- можно кешировать тайл в `__shared__`:

```cpp
__global__ void multiply_conj_fused_lds(
    const float2* __restrict__ ref_fft,
    const float2* __restrict__ inp_fft,
    float2*       __restrict__ corr_fft,
    int half_N, int K, int S)
{
    extern __shared__ float2 s_ref[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int s = blockIdx.z;

    if (i < half_N)
        s_ref[threadIdx.x] = ref_fft[k * half_N + i];
    __syncthreads();

    if (i >= half_N || k >= K || s >= S) return;

    float2 a = s_ref[threadIdx.x];
    a.y = -a.y;
    float2 b = inp_fft[s * half_N + i];

    corr_fft[(s * K + k) * half_N + i] = make_float2(
        a.x*b.x - a.y*b.y,
        a.x*b.y + a.y*b.x
    );
}
// Launch: shmem = 256 * sizeof(float2) = 2048 байт
```

### 5.3. `extract_magnitudes` -- |first n_kg| с нормализацией

C2R IFFT даёт вещественный результат (float). Нормализация на N обязательна (hipFFT не нормирует).

```cpp
__global__ void extract_magnitudes_real(
    const float* __restrict__ corr_time,   // [S × K × N] float (после C2R)
    float*       __restrict__ peaks,        // [S × K × n_kg]
    int N, int n_kg, int K, int S)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int s = blockIdx.z;
    if (j >= n_kg || k >= K || s >= S) return;

    int src = (s * K + k) * N + j;
    peaks[(s * K + k) * n_kg + j] = fabsf(corr_time[src]) / (float)N;
}
// Grid: ((n_kg+255)/256, K, S), Block: (256)
```

### 5.4. `generate_test_inputs` -- GPU-генерация тестовых входных сигналов (utility)

Вместо передачи S×N float с хоста генерируем сдвинутые копии ref прямо на GPU.
Для луча s входной сигнал = `circshift(ref, s * shift_step)`.

```cpp
__global__ void generate_test_inputs(
    const float* __restrict__ ref,     // [N] -- опорный (уже на GPU)
    float*       __restrict__ inp,     // [S × N] -- тестовые входные
    int N, int S, int shift_step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (i >= N || s >= S) return;

    int src = (i + s * shift_step) % N;
    inp[s * N + i] = ref[src];
}
// Grid: ((N+255)/256, S), Block: (256)
```

**Преимущества**:
- Нет H2D для S×N float (при N=32768, S=50 это 6.25 МБ экономии transfer)
- Python передаёт только параметры (shift_step), не данные
- Детерминированный тестовый паттерн: пик для (signal=s, shift=k) в позиции `(s*shift_step - k) mod N`

---

## 5.5. M-последовательность: CPU vs GPU

**LFSR строго последователен**: каждый бит зависит от предыдущего состояния регистра. Параллелизация невозможна без матричной экспоненциации (сложно, минимальный выигрыш).

**Решение**: генерируем M-seq на CPU (одна итерация, ~0.1 мс для N=131072), загружаем на GPU **один раз** через `PrepareReference()`. Все последующие операции (shifts, FFT, correlation) -- на GPU.

| Подход | Сложность | Время | Применение |
|--------|-----------|-------|------------|
| CPU generate + H2D once | Низкая | ~0.1 мс + H2D | **Рекомендуется** |
| GPU matrix exponentiation | Высокая | ~0.05 мс | Не оправдано |
| CPU generate + GPU constant mem | Низкая | ~0.1 мс | Для частого переиспользования |

Для тестирования: после загрузки ref на GPU, тестовые входные генерируются **на GPU** ядром `generate_test_inputs` (раздел 5.4).

---

## 6. hipFFT планы

### 6.1. Step 1 -- C2C Forward, batch=K

```cpp
hipfftHandle plan_ref;
int n[1] = { (int)N };
hipfftPlanMany(&plan_ref, 1, n,
    nullptr, 1, N,      // in: float2[K × N]
    nullptr, 1, N,      // out: float2[K × N] (in-place)
    HIPFFT_C2C, K);
hipfftSetStream(plan_ref, stream0);
hipfftExecC2C(plan_ref, d_ref_complex, d_ref_fft, HIPFFT_FORWARD);
```

### 6.2. Step 2 -- R2C Forward, batch=S

R2C принимает float напрямую, ядро `real_to_complex` не нужно.

```cpp
hipfftHandle plan_inp;
int n[1] = { (int)N };
hipfftPlanMany(&plan_inp, 1, n,
    nullptr, 1, N,          // in: float[S × N]
    nullptr, 1, N/2 + 1,    // out: float2[S × (N/2+1)]
    HIPFFT_R2C, S);
hipfftSetStream(plan_inp, stream1);
hipfftExecR2C(plan_inp, d_inp_float, d_inp_fft);
```

> Выход R2C: **N/2+1** комплексных точек. Все буферы и ядра должны использовать `half_N = N/2+1`.

### 6.3. Step 3 -- C2R Inverse, batch=S×K

C2R IFFT даёт float -- экономит память и упрощает extract_magnitudes.

```cpp
hipfftHandle plan_corr;
int n[1] = { (int)N };
hipfftPlanMany(&plan_corr, 1, n,
    nullptr, 1, N/2 + 1,    // in: float2[S×K × (N/2+1)]
    nullptr, 1, N,           // out: float[S×K × N]
    HIPFFT_C2R, S * K);
hipfftSetStream(plan_corr, stream0);
hipfftExecC2R(plan_corr, d_corr_fft, d_corr_time_real);
```

> **hipFFT не нормирует IFFT**. Деление на N в extract_magnitudes.

---

## 7. Буферы GPU

| Буфер | Размер | Тип | Описание |
|-------|--------|-----|----------|
| `d_ref_float` | N × 4 | float | Опорный с хоста |
| `d_ref_complex` | K × N × 8 | float2 | После convert_and_shift |
| `d_ref_fft` | K × N × 8 | float2 | In-place с d_ref_complex |
| `d_inp_float` | S × N × 4 | float | Входные с хоста |
| `d_inp_fft` | S × (N/2+1) × 8 | float2 | Спектр входных (R2C) |
| `d_corr_fft` | S × K × (N/2+1) × 8 | float2 | Результат умножения |
| `d_corr_time` | S × K × N × 4 | float | IFFT результат (C2R) |
| `d_peaks` | S × K × n_kg × 4 | float | Магнитуды |

### 7.1. Расчёт памяти (N=32768, K=32, S=5, n_kg=2000)

```
d_ref_complex/fft:  32 × 32768 × 8       =   8.0 МБ  (in-place)
d_inp_float:         5 × 32768 × 4       =   0.625 МБ
d_inp_fft:           5 × 16385 × 8       =   0.625 МБ  (N/2+1 hermitian!)
d_corr_fft:       5×32 × 16385 × 8       =  20.0 МБ   (N/2+1!)
d_corr_time:      5×32 × 32768 × 4       =  20.0 МБ   (float, не float2!)
d_peaks:          5×32 × 2000 × 4        =   1.22 МБ
─────────────────────────────────────────────────────
ИТОГО:                                      ~50.5 МБ
```

> Сравнение: без оптимизаций (C2C, N точек) было бы ~100 МБ.

---

## 8. Параметры конфигурации

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `fft_size` | size_t | 32768 (2^15) | Размер FFT, степень 2 |
| `num_shifts` | int | 32 | K -- циклические сдвиги опорного |
| `num_signals` | int | 5 | S -- кол-во входных сигналов |
| `num_output_points` | int | 2000 | n_kg -- первые точки из IFFT |
| `lfsr_polynomial` | uint32_t | 0xB8000000 | Полином LFSR |
| `lfsr_seed` | uint32_t | 0x1 | Начальное значение LFSR |

---

## 9. Отличия от оригинального Correlator

| Аспект | Correlator (оригинал) | fm_correlator (GPUWorkLib) |
|--------|----------------------|---------------------------|
| Backend | OpenCL + clFFT | ROCm + hipFFT/rocFFT |
| Опорный сигнал | int32 {+1,-1} | **float** {+1.0,-1.0} |
| Callbacks | clFFT pre/post callbacks | Отдельные HIP-ядра (fused) |
| FFT входных | C2C + отдельный kernel | **R2C** (принимает float напрямую) |
| IFFT | C2C (выход float2) | **C2R** (выход float) |
| Сопряжение | Отдельный kernel | **Fused** в multiply kernel |
| Спектр. буферы | N точек | **N/2+1** (hermitian symmetry) |
| Kernel launches | 5 | **3** |
| Контекст GPU | Свой OpenCL context | DrvGPU IBackend* |
| Профилирование | Свой Profiler | GPUProfiler из DrvGPU |
| Вывод | std::cout | ConsoleOutput из DrvGPU |

---

## 10. ROCm-оптимизации

### 10.1. Применённые

| Оптимизация | Выигрыш |
|-------------|---------|
| R2C вместо real_to_complex kernel | -1 kernel launch, -1 буфер (inp_complex) |
| Conj fused в multiply kernel | -1 kernel launch, -1 проход по 8 МБ |
| C2R вместо C2C для IFFT | Выход float вместо float2, -50% памяти corr_time |
| In-place FFT для ref (Step 1) | d_ref_fft == d_ref_complex, -8 МБ |
| Два HIP stream для Step1/Step2 | Параллельное выполнение |
| Persistent FFT plans | hipfftPlanMany дорог -- создаём при SetParams(), не в Process() |
| N/2+1 hermitian | Спектральные буферы и multiply работают с вдвое меньшими данными |

### 10.2. Запланированные

| Оптимизация | Описание |
|-------------|----------|
| LDS cache для ref_fft | В multiply kernel: ref_fft[k] читается S раз -- кешировать в `__shared__` |
| Pinned memory (H2D/D2H) | `hipHostMalloc` вместо std::vector -- ускоряет transfer |
| Block size tuning | Подобрать оптимальный block size через rocprofv3 |

### 10.3. Альтернатива: rocFFT native callbacks

rocFFT (не hipFFT) поддерживает экспериментальные load/store callbacks:
- `rocfft_execution_info_set_load_callback` / `_set_store_callback`
- Требуют `-fgpu-rdc` для всего проекта
- Shared memory недоступна (параметр = NULL)
- Interleaved формат обязателен

Для продакшн-кода рекомендуется подход с fused kernels (раздел 5).

---

## 11. Планируемый C++ API

```cpp
namespace drv_gpu_lib {

struct FMCorrelatorParams {
  size_t fft_size = 32768;
  int num_shifts = 32;
  int num_signals = 5;
  int num_output_points = 2000;
  uint32_t lfsr_polynomial = 0xB8000000;
  uint32_t lfsr_seed = 0x1;
};

struct FMCorrelatorResult {
  std::vector<float> peaks;       // [S × K × n_kg], row-major
  int num_signals;
  int num_shifts;
  int num_output_points;

  float at(int signal, int shift, int point) const {
    return peaks[(signal * num_shifts + shift) * num_output_points + point];
  }
};

class FMCorrelator {
public:
  explicit FMCorrelator(IBackend* backend);

  void SetParams(const FMCorrelatorParams& params);

  // M-sequence генерация (CPU, возвращает float {+1.0, -1.0})
  std::vector<float> GenerateMSequence() const;
  std::vector<float> GenerateMSequence(uint32_t seed) const;

  // Подготовка опорного спектра (один раз при смене ref)
  void PrepareReference(const std::vector<float>& reference_signal);

  // Подготовка опорного из внутреннего генератора M-seq (CPU gen + upload)
  void PrepareReference();  // использует lfsr_seed из params

  // Корреляция с внешними данными (вызывается многократно)
  // Если S > batch_size -- автоматически разбивает через BatchManager
  FMCorrelatorResult Process(const std::vector<float>& input_signals);

  // Тестовый паттерн: генерация входных на GPU через circshift(ref, s*shift_step)
  // ref уже на GPU после PrepareReference(), входные генерируются ядром generate_test_inputs
  // Python передаёт ТОЛЬКО shift_step, данные не покидают GPU
  FMCorrelatorResult RunTestPattern(int shift_step = 2);

  // Пошаговое выполнение (для отладки / профилирования)
  void Step1_ReferenceFFT(const std::vector<float>& reference_signal);
  void Step2_InputFFT(const std::vector<float>& input_signals);
  FMCorrelatorResult Step3_Correlate();
};

}  // namespace drv_gpu_lib
```

---

## 12. BatchManager -- разбиение на батчи

При большом количестве входных сигналов (S) или большом FFT size данные могут не помещаться в GPU-память. Используем `BatchManager` из `DrvGPU/services/batch_manager.hpp`.

### 12.1. Стратегия

Разбиваем по **входным сигналам** (S): берём часть лучей, считаем корреляцию, затем объединяем результаты.

Опорный спектр `ref_fft[K × N]` вычислен один раз и остаётся на GPU. Батчим только `inp` и `corr`.

```cpp
// Расчёт памяти на один луч
size_t per_signal_memory =
    N * sizeof(float)            // d_inp_float (одна строка)
  + (N/2+1) * sizeof(float2)    // d_inp_fft (одна строка)
  + K * (N/2+1) * sizeof(float2) // d_corr_fft (K корреляций)
  + K * N * sizeof(float)        // d_corr_time (K результатов IFFT)
  + K * n_kg * sizeof(float);    // d_peaks (K × n_kg)

// BatchManager считает оптимальный batch_size
auto& batch_mgr = BatchManager::GetInstance();
size_t external_mem = K * N * sizeof(float2);  // ref_fft уже на GPU
size_t batch_size = batch_mgr.CalculateOptimalBatchSize(
    backend, total_signals, per_signal_memory, 0.7, external_mem);

auto batches = batch_mgr.CreateBatches(total_signals, batch_size, 3, true);

std::vector<float> all_peaks;
for (auto& batch : batches) {
    auto batch_inp = GetBatchSlice(input_signals, batch.start, batch.count, N);
    auto result = ProcessBatch(batch_inp, batch.count);
    all_peaks.insert(all_peaks.end(), result.begin(), result.end());
}
```

### 12.2. Пересоздание FFT планов

При изменении batch_size пересоздаём `plan_inp` (R2C, batch=S_batch) и `plan_corr` (C2R, batch=S_batch × K).

---

## 13. Тесты и бенчмарк

### 13.1. Функциональные тесты

| Тест | Файл | Описание |
|------|-------|----------|
| M-sequence | `test_fm_msequence.hpp` | Проверка LFSR генератора |
| Auto-correlation | `test_fm_basic.hpp` | Корреляция ref с самим собой -> пик в j=0, SNR>10 |
| Basic Pipeline | `test_fm_basic.hpp` | Малые данные (N=1024, K=4, S=2) |
| Full Pipeline | `test_fm_basic.hpp` | Параметры по умолчанию (N=32768) |
| BatchManager | `test_fm_basic.hpp` | Большие данные -- автоматическое разбиение |

### 13.2. Верификация (Python)

```python
import numpy as np

def correlate(ref, inp_signals, num_shifts, n_kg):
    N, S = len(ref), len(inp_signals)
    peaks = np.zeros((S, num_shifts, n_kg), dtype=np.float32)
    for k in range(num_shifts):
        ref_shifted = np.roll(ref, -k)
        ref_fft_conj = np.conj(np.fft.fft(ref_shifted))
        for s in range(S):
            inp_fft = np.fft.fft(inp_signals[s])
            corr = np.fft.ifft(ref_fft_conj * inp_fft).real
            peaks[s, k, :] = np.abs(corr[:n_kg])
    return peaks
```

Допустимая погрешность: `1e-4` для float32.

### 13.3. Тест самокорреляции

```cpp
auto ref   = correlator.GenerateMSequence();
auto inp   = std::vector<float>(ref.begin(), ref.end());
auto peaks = correlator.Process(inp);   // S=1
float peak = peaks.at(0, 0, 0);
float noise = *std::max_element(peaks.peaks.begin() + 1, peaks.peaks.begin() + n_kg);
assert(peak / noise > 10.0f);   // SNR > 10
```

### 13.4. Тест сдвигового паттерна (основной тест корректности)

**Алгоритм**: для каждого луча s генерируем `inp[s] = circshift(ref, s * 2)`.
Тогда пик корреляции для пары (signal=s, shift=k) должен быть в позиции `(s*2 - k) mod N`.

Это **детерминированный** тест -- мы точно знаем где каждый пик.

```cpp
void test_fm_shift_pattern(IBackend* backend) {
  const int N = 4096, K = 10, S = 5, n_kg = 200;
  const int shift_step = 2;

  FMCorrelatorParams params;
  params.fft_size = N;
  params.num_shifts = K;
  params.num_signals = S;
  params.num_output_points = n_kg;

  FMCorrelator corr(backend);
  corr.SetParams(params);
  corr.PrepareReference();  // внутренняя генерация M-seq + upload

  // Вариант 1: CPU-генерация тестовых входных (для верификации)
  auto ref = corr.GenerateMSequence();
  std::vector<float> inp(S * N);
  for (int s = 0; s < S; ++s) {
    for (int i = 0; i < N; ++i) {
      inp[s * N + i] = ref[(i + s * shift_step) % N];
    }
  }
  auto result_cpu = corr.Process(inp);

  // Вариант 2: GPU-генерация (через RunTestPattern -- данные не покидают GPU)
  auto result_gpu = corr.RunTestPattern(shift_step);

  // Верификация: для каждой пары (s, k) проверяем позицию пика
  for (int s = 0; s < S; ++s) {
    for (int k = 0; k < K; ++k) {
      int expected_pos = ((s * shift_step - k) % N + N) % N;

      // Если expected_pos < n_kg -- пик должен быть в peaks[s][k][expected_pos]
      if (expected_pos < n_kg) {
        float peak_val = result_gpu.at(s, k, expected_pos);
        // Ищем максимум среди первых n_kg точек
        float max_val = 0;
        int max_pos = 0;
        for (int j = 0; j < n_kg; ++j) {
          float v = result_gpu.at(s, k, j);
          if (v > max_val) { max_val = v; max_pos = j; }
        }
        assert(max_pos == expected_pos);  // пик в ожидаемой позиции
      }
    }
  }

  // Сравнение CPU vs GPU результатов (должны совпадать)
  for (size_t i = 0; i < result_cpu.peaks.size(); ++i) {
    assert(std::abs(result_cpu.peaks[i] - result_gpu.peaks[i]) < 1e-4f);
  }
}
```

Тот же тест реализуется в Python (раздел 16.3).

### 13.4. Бенчмарк (как vector_algebra)

Файл: `test_fm_benchmark_rocm.hpp`

**Методика** (аналогично `modules/vector_algebra/tests/test_benchmark_symmetrize.hpp`):

```cpp
constexpr int kWarmupRuns = 3;
constexpr int kBenchmarkRuns = 20;

struct BenchStats {
  double avg_ms, min_ms, max_ms;
};

BenchStats MeasureProcessTime(FMCorrelator& corr,
                               const std::vector<float>& inp) {
  // 1. Warmup -- прогрев GPU, кеш, JIT
  for (int w = 0; w < kWarmupRuns; ++w) {
    corr.Process(inp);
    hipDeviceSynchronize();
  }

  // 2. Замеры через hipEvent (GPU hardware timer)
  hipEvent_t ev_start, ev_stop;
  hipEventCreate(&ev_start);
  hipEventCreate(&ev_stop);

  std::vector<double> times(kBenchmarkRuns);
  for (int r = 0; r < kBenchmarkRuns; ++r) {
    hipDeviceSynchronize();
    hipEventRecord(ev_start, stream);
    corr.Process(inp);
    hipEventRecord(ev_stop, stream);
    hipEventSynchronize(ev_stop);
    float ms = 0;
    hipEventElapsedTime(&ms, ev_start, ev_stop);
    times[r] = ms;
  }

  hipEventDestroy(ev_start);
  hipEventDestroy(ev_stop);

  BenchStats stats;
  stats.avg_ms = std::accumulate(times.begin(), times.end(), 0.0) / kBenchmarkRuns;
  stats.min_ms = *std::min_element(times.begin(), times.end());
  stats.max_ms = *std::max_element(times.begin(), times.end());
  return stats;
}
```

**Профилирование через GPUProfiler** (как в CLAUDE.md):

```cpp
auto& profiler = backend->GetProfiler();
profiler.SetGPUInfo(backend->GetDeviceName(), backend->GetDriverVersion());
profiler.Start("FM_Correlator_Process");
// ... Process() ...
profiler.Stop("FM_Correlator_Process");
profiler.PrintReport();
profiler.ExportMarkdown("Results/Profiler/fm_correlator_YYYY-MM-DD.md");
profiler.ExportJSON("Results/Profiler/fm_correlator_YYYY-MM-DD.json");
```

### 13.5. Параметрический прогон

Файл: `test_fm_benchmark_rocm.hpp`, функция `RunParametricBenchmark()`

После отладки базового бенчмарка прогоняем по сетке параметров:

**fft_size**: 2^n, n = 10..17 (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
**num_shifts**: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
**num_signals**: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50

```cpp
void RunParametricBenchmark(IBackend* backend) {
  const int fft_sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
  const int shifts[]    = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60};
  const int signals[]   = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

  // Таблица результатов: fft_size × (shifts, signals) -> avg_ms
  for (int N : fft_sizes) {
    for (int K : shifts) {
      for (int S : signals) {
        FMCorrelatorParams params;
        params.fft_size = N;
        params.num_shifts = K;
        params.num_signals = S;
        params.num_output_points = std::min(2000, (int)N);

        FMCorrelator corr(backend);
        corr.SetParams(params);

        auto ref = corr.GenerateMSequence();
        corr.PrepareReference(ref);

        // Генерация входных
        std::vector<float> inp(S * N);
        for (int s = 0; s < S; ++s) {
          auto sig = corr.GenerateMSequence(0x1 + s);
          std::copy(sig.begin(), sig.end(), inp.begin() + s * N);
        }

        // Если не помещаемся -- BatchManager разобьёт автоматически
        auto stats = MeasureProcessTime(corr, inp);

        con.Print(gpu_id, "FM_Bench",
            "N=%6d K=%2d S=%2d | avg=%.3f ms  min=%.3f  max=%.3f",
            N, K, S, stats.avg_ms, stats.min_ms, stats.max_ms);
      }
    }
  }
}
```

Результаты: `Results/Profiler/fm_correlator_parametric_YYYY-MM-DD.md`

---

## 14. Чеклист реализации

| Задача | Статус |
|--------|--------|
| CMake: find_package(hipfft) | |
| hipStreamCreate для stream0 и stream1 | |
| AllocateBuffers() -- 7 буферов hipMalloc | |
| CreatePlans(): plan_ref(C2C,K), plan_inp(R2C,S), plan_corr(C2R,S×K) | |
| hipfftSetStream для каждого плана | |
| apply_cyclic_shifts: float -> float2 + shifts | |
| multiply_conj_fused: half_N = N/2+1, conj inline | |
| extract_magnitudes: нормализация /N | |
| generate_test_inputs: circshift(ref, s*shift_step) на GPU | |
| PrepareReference() -- два варианта (internal + external) | |
| RunTestPattern(shift_step) -- тестовые входные на GPU | |
| BatchManager при S > batch_size | |
| Деструктор: hipfftDestroy + hipFree + hipStreamDestroy | |
| Бенчмарк: warmup 3 + 20 runs + hipEvent | |
| GPUProfiler: SetGPUInfo + Start/Stop + PrintReport | |
| Параметрический прогон: N×K×S сетка | |
| Python bindings: py_fm_correlator_rocm.hpp | |
| Python тест самокорреляции (atol=1e-4) | |
| Python тест сдвигового паттерна (CPU vs GPU) | |
| Python тест run_test_pattern (только параметры) | |
| C++ тест сдвигового паттерна (test_fm_basic.hpp) | |

### 14.1. Частые ошибки

| Ошибка | Решение |
|--------|---------|
| R2C выдаёт N/2+1 точек, а код ожидает N | Везде half_N = N/2+1 для спектральных буферов |
| C2R IFFT не нормализует результат | Делить на N в extract_magnitudes |
| Plan создаётся в каждом Process() | Создавать при SetParams(), использовать многократно |
| hipfftSetStream не вызван | План игнорирует stream, вызывать после PlanMany |
| Забыт hipStreamSynchronize перед Step3 | sync обоих потоков перед multiply |
| Большие S не помещаются в GPU | Использовать BatchManager для разбиения по лучам |

---

## 15. Python Bindings

### 15.1. Файл: `python/py_fm_correlator_rocm.hpp`

Паттерн: как `py_heterodyne_rocm.hpp` -- класс-обёртка `PyFMCorrelatorROCm`.

```cpp
class PyFMCorrelatorROCm {
  ROCmGPUContext& ctx_;
  FMCorrelator correlator_;
  FMCorrelatorParams params_;
public:
  explicit PyFMCorrelatorROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), correlator_(ctx.backend()) {}

  void set_params(int fft_size, int num_shifts, int num_signals,
                  int num_output_points, uint32_t polynomial, uint32_t seed) {
    params_.fft_size = fft_size;
    params_.num_shifts = num_shifts;
    params_.num_signals = num_signals;
    params_.num_output_points = num_output_points;
    params_.lfsr_polynomial = polynomial;
    params_.lfsr_seed = seed;
    correlator_.SetParams(params_);
  }

  // Генерация M-seq для анализа в Python (возвращает numpy)
  py::array_t<float> generate_msequence(uint32_t seed) {
    auto seq = correlator_.GenerateMSequence(seed);
    return vector_to_numpy(std::move(seq));
  }

  // Подготовка ref: генерация M-seq внутри + upload на GPU
  void prepare_reference() {
    py::gil_scoped_release release;
    correlator_.PrepareReference();
  }

  // Подготовка ref: из numpy-массива
  void prepare_reference_from_data(py::array_t<float, py::array::c_style> ref) {
    auto buf = ref.request();
    auto* ptr = static_cast<float*>(buf.ptr);
    std::vector<float> vec(ptr, ptr + buf.shape[0]);
    py::gil_scoped_release release;
    correlator_.PrepareReference(vec);
  }

  // Корреляция с numpy-входами [S, N] или [S*N]
  py::array_t<float> process(py::array_t<float, py::array::c_style> input) {
    auto buf = input.request();
    size_t total = 1;
    for (int d = 0; d < buf.ndim; ++d) total *= buf.shape[d];
    auto* ptr = static_cast<float*>(buf.ptr);
    std::vector<float> vec(ptr, ptr + total);
    FMCorrelatorResult result;
    {
      py::gil_scoped_release release;
      result = correlator_.Process(vec);
    }
    return vector_to_numpy_3d(std::move(result.peaks),
        result.num_signals, result.num_shifts, result.num_output_points);
  }

  // Тестовый паттерн: Python передаёт ТОЛЬКО shift_step
  // M-seq + тестовые входные генерируются на GPU, данные не покидают GPU
  py::array_t<float> run_test_pattern(int shift_step) {
    FMCorrelatorResult result;
    {
      py::gil_scoped_release release;
      result = correlator_.RunTestPattern(shift_step);
    }
    return vector_to_numpy_3d(std::move(result.peaks),
        result.num_signals, result.num_shifts, result.num_output_points);
  }
};
```

### 15.2. Регистрация в `gpu_worklib_bindings.cpp`

```cpp
inline void register_fm_correlator_rocm(py::module& m) {
  py::class_<PyFMCorrelatorROCm>(m, "FMCorrelatorROCm",
      "FM Correlator -- frequency-domain correlation for M-sequence phase modulation")
    .def(py::init<ROCmGPUContext&>(), py::arg("ctx"))
    .def("set_params", &PyFMCorrelatorROCm::set_params,
         py::arg("fft_size")=32768, py::arg("num_shifts")=32,
         py::arg("num_signals")=5, py::arg("num_output_points")=2000,
         py::arg("polynomial")=0xB8000000, py::arg("seed")=0x1)
    .def("generate_msequence", &PyFMCorrelatorROCm::generate_msequence,
         py::arg("seed")=0x1)
    .def("prepare_reference", &PyFMCorrelatorROCm::prepare_reference)
    .def("prepare_reference_from_data", &PyFMCorrelatorROCm::prepare_reference_from_data,
         py::arg("ref"))
    .def("process", &PyFMCorrelatorROCm::process, py::arg("input_signals"))
    .def("run_test_pattern", &PyFMCorrelatorROCm::run_test_pattern,
         py::arg("shift_step")=2);
}
```

### 15.3. Использование из Python

**Режим 1: Управление параметрами (данные не покидают GPU)**
```python
import gpuworklib

ctx = gpuworklib.ROCmGPUContext(0)
corr = gpuworklib.FMCorrelatorROCm(ctx)
corr.set_params(fft_size=32768, num_shifts=32, num_signals=10)
corr.prepare_reference()  # M-seq генерируется внутри

# Тестовый паттерн -- Python передаёт ТОЛЬКО shift_step
peaks = corr.run_test_pattern(shift_step=2)  # numpy [S, K, n_kg]

# Верификация
for s in range(10):
    for k in range(32):
        expected_pos = (s * 2 - k) % 32768
        if expected_pos < 2000:
            assert peaks[s, k, expected_pos] == peaks[s, k, :].max()
```

**Режим 2: Внешние данные**
```python
import numpy as np

ref = corr.generate_msequence(seed=1)
signals = np.stack([np.roll(ref, -s*2) for s in range(10)])
corr.prepare_reference_from_data(ref)
peaks = corr.process(signals.astype(np.float32))
```

**Режим 3: Только параметры через JSON (файл управления)**
```python
import json

config = {
    "fft_size": 32768,
    "num_shifts": 32,
    "num_signals": 10,
    "num_output_points": 2000,
    "polynomial": 0xB8000000,
    "seed": 1,
    "shift_step": 2
}
with open("fm_config.json", "w") as f:
    json.dump(config, f)

# C++ читает JSON и всё делает сам
# Или Python:
corr.set_params(**{k: v for k, v in config.items() if k != "shift_step"})
corr.prepare_reference()
peaks = corr.run_test_pattern(config["shift_step"])
```

### 15.4. Python тест: `Python_test/fm_correlator/test_fm_correlator_rocm.py`

```python

import numpy as np
import gpuworklib


def ctx():
    return gpuworklib.ROCmGPUContext(0)

def test_autocorrelation(ctx):
    corr = gpuworklib.FMCorrelatorROCm(ctx)
    corr.set_params(fft_size=4096, num_shifts=1, num_signals=1, num_output_points=200)
    ref = corr.generate_msequence(seed=1)
    corr.prepare_reference_from_data(ref)
    peaks = corr.process(ref)  # [1, 1, 200]
    assert peaks[0, 0, 0] > 10 * np.max(peaks[0, 0, 1:])  # SNR > 10

def test_shift_pattern(ctx):
    N, K, S = 4096, 10, 5
    shift_step = 2
    corr = gpuworklib.FMCorrelatorROCm(ctx)
    corr.set_params(fft_size=N, num_shifts=K, num_signals=S, num_output_points=200)
    corr.prepare_reference()

    peaks_gpu = corr.run_test_pattern(shift_step)  # [S, K, 200]

    # Verify peak positions
    for s in range(S):
        for k in range(K):
            expected = (s * shift_step - k) % N
            if expected < 200:
                assert np.argmax(peaks_gpu[s, k, :]) == expected

def test_cpu_vs_gpu_pattern(ctx):
    """CPU circshift vs GPU generate_test_inputs -- должны совпадать."""
    N, K, S = 4096, 8, 4
    shift_step = 2
    corr = gpuworklib.FMCorrelatorROCm(ctx)
    corr.set_params(fft_size=N, num_shifts=K, num_signals=S, num_output_points=200)

    ref = corr.generate_msequence(seed=1)
    corr.prepare_reference_from_data(ref)

    # CPU: numpy circshift
    signals = np.stack([np.roll(ref, -s * shift_step) for s in range(S)])
    peaks_cpu = corr.process(signals.astype(np.float32))

    # GPU: run_test_pattern
    corr.prepare_reference_from_data(ref)
    peaks_gpu = corr.run_test_pattern(shift_step)

    np.testing.assert_allclose(peaks_cpu, peaks_gpu, atol=1e-4)
```

---

## 16. Ссылки

- Исходный проект: `/home/alex/C++/Correlator/`
- Руководство коллеги: `Doc/Modules/fm_correlator/FM_Correlator_ROCm_Guide.docx`
- Python референс: `/home/alex/C++/Correlator/Doc/Python_Examples/pyfftw_implementation.py`
- ROCm оптимизация: `Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md`
- Python bindings образец: `python/py_heterodyne_rocm.hpp`
- Python bindings сборка: `python/gpu_worklib_bindings.cpp`
- hipFFT API: https://rocm.docs.amd.com/projects/hipFFT/en/latest/
- rocFFT callbacks: https://rocm.docs.amd.com/projects/rocFFT/en/latest/
