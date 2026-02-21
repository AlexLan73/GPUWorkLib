# 📡 PLAN: Гетеродин ЛЧМ — Dechirp Module

> **Версия**: 2.0 (расширено по ТЗ Alex)
> **Дата**: 2026-02-21
> **Автор плана**: Кодо (AI Senior)
> **Для**: AI-исполнитель (следует как инструкция)
> **Статус**: ✍️ Черновик — ждёт правки Alex

---

## 0. Контекст и цели

### Задача в двух словах
Реализовать дечирп-обработку ЛЧМ-радара (stretch processing):
```
s_dc = s_rx × conj(s_tx)  →  FFT  →  f_beat  →  R
```

### Параметры тестового сигнала (ЗАФИКСИРОВАНЫ)
```
fs       = 12e6 Hz         (12 МГц)
B        = 1e6  Hz         (девиация 1 МГц)
N        = 4000 points     (число отсчётов)
T        = N/fs = 333.33 μs
μ        = B/T  = 3e9 Hz/s (chirp rate)
antennas = 5
f_start  = 0 Hz            (начальная частота, или задать разумную)
```

### Задержки антенн для тестов
```
# 1) Линейный сдвиг (красиво — равные биты):
delays_linear_us = [100, 200, 300, 400, 500]  # мкс
f_beats_Hz       = [300k, 600k, 900k, 1200k, 1500k]  # Hz (μ × τ)
f_beats_bins     = [100, 200, 300, 400, 500]  # bin из 4000 — наглядно!

# 2) Случайная (красиво — случайный паттерн):
delays_random_us = np.random.uniform(50, 500, 5)  # seed=42 для воспроизводимости
```

**Проверка**: `f_beat = μ × τ = 3e9 × 100e-6 = 300 000 Hz = 300 кГц` → bin 100 из 4000 ✓

---

## 1. Новый класс: `LfmConjugateGenerator`

### 1.1 Назначение
Генерирует **сопряжённый опорный ЛЧМ** — `conj(s_tx(t))` при задержке τ=0:

```
s_ref*(t) = exp(-j[π·μ·t² + 2π·f_start·t])
```

Используется как опорный сигнал для дечирпа: `s_dc = s_rx × s_ref*`.

### 1.2 Место в проекте
```
modules/signal_generators/
├── include/generators/
│   └── lfm_conjugate_generator.hpp   ← НОВЫЙ
├── src/
│   └── lfm_conjugate_generator.cpp   ← НОВЫЙ
└── kernels/
    └── lfm_conjugate_generator.cl    ← НОВЫЙ
```

### 1.3 Заголовок `lfm_conjugate_generator.hpp`

```cpp
#pragma once
/**
 * @file lfm_conjugate_generator.hpp
 * @brief Генератор сопряжённого ЛЧМ — conj(s_tx) при τ=0
 *
 * Формула: s_ref*(t) = exp(-j[π·μ·t² + 2π·f_start·t])
 * где μ = (f_end - f_start) / T = B/T
 *
 * Используется как опорный сигнал для дечирп-обработки:
 *   s_dc = s_rx(t) × s_ref*(t)   // результат — тон на f_beat = μ·τ
 *
 * Аналог: LfmGeneratorAnalyticalDelay но (1) всегда delay=0, (2) берём conj
 */

#include "../i_signal_generator.hpp"
#include "../../DrvGPU/include/drv_gpu.hpp"
#include <vector>
#include <cstdint>

namespace drv_gpu_lib {

struct LfmConjugateParams {
    float f_start     = 0.0f;      // Гц, начальная частота
    float f_end       = 1e6f;      // Гц, конечная частота  (девиация B = f_end - f_start)
    float sample_rate = 12e6f;     // Гц, fs
    int   num_samples = 4000;      // N точек
    float amplitude   = 1.0f;     // амплитуда
};

class LfmConjugateGenerator {
public:
    explicit LfmConjugateGenerator(IBackend* backend);

    LfmConjugateGenerator(const LfmConjugateGenerator&) = delete;
    LfmConjugateGenerator& operator=(const LfmConjugateGenerator&) = delete;
    LfmConjugateGenerator(LfmConjugateGenerator&&) = default;

    /**
     * Установить параметры ЛЧМ.
     * Должен вызываться до Generate/GenerateToGpu.
     */
    void SetParams(const LfmConjugateParams& params);

    /**
     * Генерация → CPU (host vector, complex64).
     * Результат: conj(LFM(t)) = exp(-j[π·μ·t² + 2π·f_start·t])
     */
    std::vector<std::complex<float>> GenerateToCpu();

    /**
     * Генерация → GPU (возвращает GPUBuffer complex64, 1×N).
     * Данные остаются на GPU — дешевле для дечирп-пайплайна.
     */
    GPUBuffer GenerateToGpu();

private:
    IBackend*             backend_;
    LfmConjugateParams    params_;
    bool                  compiled_ = false;
    cl::Kernel            kernel_;

    void CompileKernel();
};

}  // namespace drv_gpu_lib
```

### 1.4 Ядро `lfm_conjugate_generator.cl`

```opencl
/**
 * lfm_conjugate_generator.cl
 * Генерация conj(LFM): exp(-j[π·μ·n²/fs² + 2π·f_start·n/fs])
 * где n = sample_id, μ = (f_end - f_start) / T = (f_end - f_start) * fs / N
 */
__kernel void lfm_conjugate_generate(
    __global float2* output,     // выход: complex float2 (.x=re, .y=im)
    const float      f_start,    // Гц
    const float      f_end,      // Гц
    const float      sample_rate,// Гц, fs
    const int        num_samples, // N
    const float      amplitude
) {
    int n = get_global_id(0);
    if (n >= num_samples) return;

    float T      = (float)num_samples / sample_rate;
    float mu     = (f_end - f_start) / T;           // chirp rate, Гц/с
    float t      = (float)n / sample_rate;           // время сэмпла

    // conj(LFM) = exp(-j[π·μ·t² + 2π·f_start·t])
    float phase  = -(M_PI_F * mu * t * t + 2.0f * M_PI_F * f_start * t);

    output[n].x = amplitude * cos(phase);
    output[n].y = amplitude * sin(phase);
}
```

### 1.5 Python binding (добавить в `gpu_worklib_bindings.cpp`)

```cpp
// В блоке PYBIND11_MODULE добавить:
py::class_<drv_gpu_lib::LfmConjugateParams>(m, "LfmConjugateParams")
    .def(py::init<>())
    .def_readwrite("f_start",     &drv_gpu_lib::LfmConjugateParams::f_start)
    .def_readwrite("f_end",       &drv_gpu_lib::LfmConjugateParams::f_end)
    .def_readwrite("sample_rate", &drv_gpu_lib::LfmConjugateParams::sample_rate)
    .def_readwrite("num_samples", &drv_gpu_lib::LfmConjugateParams::num_samples)
    .def_readwrite("amplitude",   &drv_gpu_lib::LfmConjugateParams::amplitude);

py::class_<drv_gpu_lib::LfmConjugateGenerator>(m, "LfmConjugateGenerator")
    .def(py::init<drv_gpu_lib::IBackend*>())
    .def("set_params",       &drv_gpu_lib::LfmConjugateGenerator::SetParams)
    .def("generate_to_cpu",  [](drv_gpu_lib::LfmConjugateGenerator& g) {
        auto vec = g.GenerateToCpu();
        return py::array_t<std::complex<float>>(vec.size(), vec.data());
    });
```

---

## 2. Новый модуль: `heterodyne`

### 2.1 Структура файлов

```
modules/heterodyne/
├── CMakeLists.txt
├── include/
│   ├── heterodyne_dechirp.hpp          ← Фасад (публичный API)
│   ├── heterodyne_params.hpp           ← Параметры + результаты
│   ├── i_heterodyne_processor.hpp      ← Интерфейс (OpenCL/ROCm)
│   └── processors/
│       ├── heterodyne_processor_opencl.hpp
│       └── heterodyne_processor_rocm.hpp    ← ЗАГЛУШКА
├── src/
│   ├── heterodyne_dechirp.cpp
│   ├── heterodyne_processor_opencl.cpp
│   └── heterodyne_processor_rocm.cpp        ← ЗАГЛУШКА (throw/return error)
├── kernels/
│   └── opencl/
│       ├── dechirp_multiply.cl         ← s_rx × conj(s_tx)
│       └── dechirp_correct.cl          ← s_dc × exp(-j·2π·f_beat·t)
├── tests/
│   ├── README.md
│   ├── all_test.hpp
│   ├── test_heterodyne_basic.hpp       ← Unit-тест дечирпа
│   ├── test_heterodyne_pipeline.hpp    ← Интеграционный тест
│   └── test_heterodyne_external_ctx.hpp ← Тест с внешним контекстом
└── bindings/
    └── heterodyne_bindings.cpp          ← Python биндинги
```

### 2.2 `heterodyne_params.hpp`

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace drv_gpu_lib {

/** Параметры ЛЧМ для дечирпа */
struct HeterodyneParams {
    float f_start      = 0.0f;      // Гц, начальная частота ЛЧМ
    float f_end        = 1e6f;      // Гц, конечная частота ЛЧМ (B = f_end - f_start)
    float sample_rate  = 12e6f;     // Гц, fs
    int   num_samples  = 4000;      // N, число отсчётов на антенну
    int   num_antennas = 5;         // число антенн (каналов)

    // Производные (вычисляет класс):
    // T   = num_samples / sample_rate
    // mu  = (f_end - f_start) / T        [Гц/с, chirp rate]
    // bin_width = sample_rate / num_samples [Гц/бин]
};

/** Результат по одной антенне */
struct AntennaDechirpResult {
    int   antenna_idx    = 0;
    float f_beat_hz      = 0.0f;    // частота биений [Гц]
    float f_beat_bin     = 0.0f;    // бин (с дробной частью, параболическая интерпол.)
    float range_m        = 0.0f;    // дальность R = c·T·f_beat/(2B) [м]
    float peak_amplitude = 0.0f;    // амплитуда пика
    float peak_snr_db    = 0.0f;    // SNR пика [дБ]
};

/** Итоговый результат дечирпа */
struct HeterodyneResult {
    bool  success = false;
    std::vector<AntennaDechirpResult> antennas;   // результат по каждой антенне
    std::vector<float> max_positions;             // позиции всех максимумов (для контроля)
    std::string error_message;

    // Вспомогательные формулы:
    // R = c * T * f_beat / (2 * B)
    // c = 3e8 м/с, T = num_samples/sample_rate, B = f_end - f_start
    static float CalcRange(float f_beat, float sample_rate,
                            int num_samples, float bandwidth) {
        float T = (float)num_samples / sample_rate;
        return (3e8f * T * f_beat) / (2.0f * bandwidth);
    }
};

}  // namespace drv_gpu_lib
```

### 2.3 `i_heterodyne_processor.hpp` (интерфейс)

```cpp
#pragma once
#include "heterodyne_params.hpp"
#include <vector>
#include <complex>

namespace drv_gpu_lib {

class IHeterodyneProcessor {
public:
    virtual ~IHeterodyneProcessor() = default;

    /**
     * Дечирп: s_dc = s_rx × conj(s_tx) на GPU.
     *
     * @param rx_data   Матрица [num_antennas × num_samples] complex float
     * @param ref_data  Вектор [num_samples] complex float = conj(s_tx)
     * @param params    Параметры ЛЧМ
     * @return          Матрица [num_antennas × num_samples] complex float (на GPU)
     */
    virtual std::vector<std::complex<float>> Dechirp(
        const std::vector<std::complex<float>>& rx_data,
        const std::vector<std::complex<float>>& ref_data,
        const HeterodyneParams& params) = 0;

    /**
     * Дечирп из внешнего GPU-буфера (не владеет указателем!).
     * Для интеграции с внешними OpenCL-программами.
     *
     * @param rx_cl_mem  cl_mem — внешний буфер [num_antennas × num_samples]
     * @param ref_data   Опорный сигнал (CPU → GPU внутри)
     * @param params     Параметры ЛЧМ
     */
    virtual HeterodyneResult ProcessExternal(
        void*  rx_cl_mem,
        const std::vector<std::complex<float>>& ref_data,
        const HeterodyneParams& params) = 0;
};

}  // namespace drv_gpu_lib
```

### 2.4 `heterodyne_dechirp.hpp` (публичный фасад)

```cpp
#pragma once
/**
 * @file heterodyne_dechirp.hpp
 * @brief Гетеродин дечирп ЛЧМ — публичный API
 *
 * ИСПОЛЬЗОВАНИЕ (нормальный режим):
 *   auto& backend = drv.GetBackend();
 *   HeterodyneDechirp het(backend);
 *   het.SetParams(params);
 *   auto result = het.Process(rx_matrix);  // num_antennas × num_samples
 *
 * ИСПОЛЬЗОВАНИЕ (внешний OpenCL контекст):
 *   // Внешняя программа передаёт cl_mem с данными на GPU
 *   // CPU передаёт метаданные (HeterodyneParams)
 *   het.SetParams(params);
 *   auto result = het.ProcessExternal(cl_mem_ptr, params);
 *
 * ПАЙПЛАЙН:
 *   1. LfmConjugateGenerator → s_ref* (GPU)
 *   2. dechirp_multiply.cl   → s_dc = s_rx × s_ref*  (GPU)
 *   3. SpectrumMaximaFinder  → f_beat (FFT + пик)     (GPU)
 *   4. [AllMaxima]           → все пики (для контроля) (GPU)
 *   5. dechirp_correct.cl    → компенсация f_beat      (GPU)
 *   6. Верификация: спектр → DC                         (GPU)
 */

#include "i_heterodyne_processor.hpp"
#include "heterodyne_params.hpp"
#include "../../DrvGPU/include/common/backend_type.hpp"
#include <memory>

namespace drv_gpu_lib {

class IBackend;

class HeterodyneDechirp {
public:
    /**
     * @param backend         Указатель на DrvGPU backend (не владеет)
     * @param compute_backend OpenCL (по умолчанию) или ROCm (заглушка)
     */
    explicit HeterodyneDechirp(
        IBackend* backend,
        BackendType compute_backend = BackendType::OPENCL);

    HeterodyneDechirp(const HeterodyneDechirp&) = delete;
    HeterodyneDechirp& operator=(const HeterodyneDechirp&) = delete;

    /** Установить параметры ЛЧМ (должен вызываться перед Process) */
    void SetParams(const HeterodyneParams& params);

    /**
     * Основной метод: полный пайплайн из CPU данных.
     * Входные данные: s_rx — матрица [num_antennas × num_samples], complex float
     */
    HeterodyneResult Process(
        const std::vector<std::complex<float>>& rx_data);

    /**
     * Вариант для внешнего OpenCL контекста.
     * rx_cl_mem — указатель на cl_mem (внешняя программа владеет буфером).
     * params передаются с CPU (метаданные: fs, B, N, antennas).
     */
    HeterodyneResult ProcessExternal(
        void* rx_cl_mem,
        const HeterodyneParams& params);

    /** Последний результат (кешированный) */
    const HeterodyneResult& GetLastResult() const;

private:
    std::unique_ptr<IHeterodyneProcessor> processor_;
    HeterodyneParams                      params_;
    HeterodyneResult                      last_result_;
};

}  // namespace drv_gpu_lib
```

---

## 3. OpenCL ядра

### 3.1 `dechirp_multiply.cl`

```opencl
/**
 * dechirp_multiply.cl
 * Дечирп: output[ant][n] = rx[ant][n] × conj(ref[n])
 *
 * Каждый work-item обрабатывает 1 сэмпл 1 антенны.
 * gid_ant = get_global_id(0)  [антенна]
 * gid_n   = get_global_id(1)  [сэмпл]
 *
 * Математика:
 *   (a+jb) × conj(c+jd) = (a+jb)(c-jd) = (ac+bd) + j(bc-ad)
 */
__kernel void dechirp_multiply(
    __global const float2* rx,          // [num_antennas × num_samples]
    __global const float2* ref,         // [num_samples] — broadcast по антеннам
    __global       float2* dc_out,      // [num_antennas × num_samples]
    const int num_samples
) {
    int ant = get_global_id(0);   // индекс антенны
    int n   = get_global_id(1);   // индекс сэмпла

    int gid     = ant * num_samples + n;
    float2 rx_v = rx[gid];
    float2 re_v = ref[n];          // broadcast: один вектор на все антенны

    // Комплексное умножение: rx × conj(ref)
    // conj(ref) = (re_v.x, -re_v.y)
    dc_out[gid].x = rx_v.x * re_v.x + rx_v.y * re_v.y;   // Re: a*c + b*d
    dc_out[gid].y = rx_v.y * re_v.x - rx_v.x * re_v.y;   // Im: b*c - a*d
}
```

### 3.2 `dechirp_correct.cl`

```opencl
/**
 * dechirp_correct.cl
 * Компенсация частоты биений: output = input × exp(-j·2π·f_beat·t)
 *
 * После дечирпа остаётся тон на f_beat.
 * Умножение на exp(-j·2π·f_beat·t) сдвигает спектр к DC (0 Гц).
 * Результат: спектр должен иметь пик строго при 0 Гц (верификация).
 */
__kernel void dechirp_correct(
    __global const float2* dc_in,        // дечирп [num_antennas × num_samples]
    __global       float2* corrected,    // выход [num_antennas × num_samples]
    __global const float*  f_beat,       // частота биений per-antenna [num_antennas]
    const float sample_rate,
    const int   num_samples
) {
    int ant = get_global_id(0);
    int n   = get_global_id(1);

    int gid  = ant * num_samples + n;
    float t  = (float)n / sample_rate;
    float fb = f_beat[ant];              // f_beat для данной антенны

    // Корректирующий множитель: exp(-j·2π·f_beat·t)
    float phase = -2.0f * M_PI_F * fb * t;
    float2 corr;
    corr.x = cos(phase);
    corr.y = sin(phase);

    // Комплексное умножение
    float2 in = dc_in[gid];
    corrected[gid].x = in.x * corr.x - in.y * corr.y;
    corrected[gid].y = in.y * corr.x + in.x * corr.y;
}
```

---

## 4. C++ Тесты

### 4.1 Параметры (общие для всех тестов)

```cpp
// test_heterodyne_params.hpp (inline, включается в тесты)
namespace test_het_params {
    constexpr float  FS         = 12e6f;    // 12 МГц
    constexpr float  F_START    = 0.0f;     // Гц
    constexpr float  F_END      = 1e6f;     // 1 МГц (B = F_END - F_START)
    constexpr int    N          = 4000;     // сэмплов
    constexpr int    ANTENNAS   = 5;
    constexpr float  C_LIGHT    = 3e8f;     // м/с

    // Линейные задержки [мкс] → f_beat = μ×τ
    // μ = B/T = 1e6/(4000/12e6) = 3e9 Гц/с
    // τ=100мкс → f_beat = 300 кГц → bin 100
    const float DELAYS_LINEAR_US[5] = {100.f, 200.f, 300.f, 400.f, 500.f};

    // Случайные задержки (seed=42, равномерно [50..500] мкс)
    const float DELAYS_RANDOM_US[5] = {273.f, 87.f, 421.f, 156.f, 312.f};
    // ^ предварительно сгенерированы np.random.seed(42); np.random.uniform(50,500,5)

    // Допуски
    constexpr float F_BEAT_TOL_HZ   = 5000.f;   // ±5 кГц (< 2 бина)
    constexpr float RANGE_TOL_M     = 0.5f;      // ±0.5 м
}
```

### 4.2 `test_heterodyne_basic.hpp` — Unit-тест ядра

```
ТЕСТ 1: Дечирп одной антенны
  - Генерировать s_rx с delay=100 мкс (LfmGeneratorAnalyticalDelay)
  - Генерировать s_ref* (LfmConjugateGenerator)
  - Вызвать dechirp_multiply на GPU
  - Взять FFT (FFTProcessor)
  - Найти пик (SpectrumMaximaFinder)
  - ASSERT: |f_peak - 300 000 Гц| < F_BEAT_TOL_HZ
  - Вывод: [PASS/FAIL] Test 1: Single antenna dechirp, f_beat=XXX Hz

ТЕСТ 2: Дечирп 5 антенн (линейные задержки)
  - Генерировать s_rx матрицу [5 × 4000] с delays_linear_us
  - Дечирп всех 5 антенн за один kernel launch
  - FindMaxima для каждой антенны
  - ASSERT: f_beat[k] ≈ μ × delays_linear_us[k] × 1e-6 для k=0..4
  - Вывод: таблица [антенна, задержка_мкс, f_beat_Hz, R_m, ошибка_м]

ТЕСТ 3: Дечирп 5 антенн (случайные задержки)
  - Аналогично тесту 2, delays = DELAYS_RANDOM_US
  - ASSERT: |R_measured - R_true| < RANGE_TOL_M

ТЕСТ 4: dechirp_correct — верификация
  - После теста 2: применить dechirp_correct для каждой антенны
  - FFT результата
  - ASSERT: пик при 0 Гц ± F_BEAT_TOL_HZ
  - Вывод: [PASS/FAIL] Correction verification: peak at DC
```

### 4.3 `test_heterodyne_pipeline.hpp` — Интеграция

```
ТЕСТ 5: Полный пайплайн (Process())
  Шаги:
  1. HeterodyneParams params{.f_start=0, .f_end=1e6, .fs=12e6, .N=4000, .antennas=5}
  2. HeterodyneDechirp het(backend)
  3. het.SetParams(params)
  4. Генерация rx = LfmGeneratorAnalyticalDelay(5 антенн, линейные задержки)
  5. result = het.Process(rx_data)
  6. ASSERT: result.success == true
  7. Для каждой антенны:
     a. ASSERT: |f_beat[k] - ожидаемая| < 5000 Гц
     b. ASSERT: |R[k] - R_true[k]| < 0.5 м
  8. result.max_positions — вывести все найденные максимумы
  9. Markdown отчёт → Results/heterodyne/test_pipeline_report.md

ТЕСТ 6: AllMaxima — контроль
  - Найти ВСЕ максимумы в FFT каждой антенны
  - Проверить, что ровно 1 значимый пик (SNR > 20 дБ)
  - Если шум — могут быть ложные, но основной пик должен быть самым высоким
```

### 4.4 `test_heterodyne_external_ctx.hpp` — Внешний OpenCL контекст

```
ТЕСТ 7: ProcessExternal()
  Имитация сценария "внешняя программа передаёт cl_mem":

  1. Создать DrvGPU gpu(OPENCL, 0)
  2. Выделить буфер: cl_mem external_buf = [5 × 4000 complex float]
  3. Записать в него s_rx (через OpenCL API напрямую, не через DrvGPU)
  4. HeterodyneDechirp het(gpu.GetBackend())
  5. result = het.ProcessExternal(external_buf, params)
  6. ASSERT: result.success == true
  7. ASSERT: f_beat корректные
  8. Проверить: буфер НЕ освобождён HeterodyneDechirp (владеет внешний код)

  Цель: демонстрация API для интеграции с внешними радар-системами.
```

### 4.5 `all_test.hpp`

```cpp
#pragma once
#include "test_heterodyne_basic.hpp"
#include "test_heterodyne_pipeline.hpp"
#include "test_heterodyne_external_ctx.hpp"

namespace heterodyne_all_test {
inline void run() {
    test_heterodyne_basic::run();        // Tests 1-4
    test_heterodyne_pipeline::run();     // Tests 5-6
    test_heterodyne_external_ctx::run(); // Test 7
}
}
```

---

## 5. Python Тесты

### 5.1 Параметры (общие)

```python
# heterodyne_test_params.py
import numpy as np

FS        = 12e6       # Гц
F_START   = 0.0        # Гц
F_END     = 1e6        # Гц
B         = F_END - F_START  # 1 МГц (девиация)
N         = 4000       # точек
ANTENNAS  = 5
T         = N / FS     # длительность, с
MU        = B / T      # chirp rate = 3e9 Гц/с
C_LIGHT   = 3e8        # м/с

# Линейные задержки
DELAYS_LINEAR_US = np.array([100., 200., 300., 400., 500.])
DELAYS_LINEAR_S  = DELAYS_LINEAR_US * 1e-6

# Случайные задержки (seed=42 для воспроизводимости)
rng = np.random.default_rng(seed=42)
DELAYS_RANDOM_US = rng.uniform(50, 500, ANTENNAS)
DELAYS_RANDOM_S  = DELAYS_RANDOM_US * 1e-6

# Ожидаемые f_beat
F_BEATS_LINEAR = MU * DELAYS_LINEAR_S    # [300k, 600k, 900k, 1200k, 1500k] Гц
RANGES_TRUE    = C_LIGHT * T * F_BEATS_LINEAR / (2 * B)
```

### 5.2 `test_heterodyne_step_by_step.py` — Пошаговый тест с выводом

**ГЛАВНЫЙ отладочный тест.** Выводит значения и графики после каждого шага.

```python
"""
test_heterodyne_step_by_step.py
=================================
Пошаговый тест дечирп-обработки ЛЧМ.
На каждом шаге выводит:
  - Числа (max, mean, f_peak, ошибку)
  - График (сохраняется в Results/Plots/heterodyne/step_XX_*.png)

Параллельно считает на GPU (gpuworklib) и CPU (numpy) и сравнивает.

Шаги:
  1. Генерация s_rx (5 антенн, линейные задержки) — GPU и NumPy
  2. Генерация s_ref* (сопряжённый ЛЧМ, delay=0) — GPU и NumPy
  3. Дечирп: s_dc = s_rx × conj(s_tx) — GPU и NumPy
  4. FFT дечирпованного сигнала — GPU и NumPy
  5. FindMaxima — f_beat, R — GPU и NumPy
  6. FindAllMaxima — контроль — GPU
  7. dechirp_correct — компенсация f_beat — GPU и NumPy
  8. Финальный FFT — верификация DC — GPU и NumPy
  9. Итоговый отчёт: таблица ошибок, сравнение GPU vs CPU
"""
```

**Структура теста step-by-step:**

```python
# === ШАГИ ТЕСТА (каждый шаг выводит на экран и сохраняет график) ===

def step01_generate_rx():
    """Шаг 1: Генерация принятого ЛЧМ сигнала."""
    print("\n" + "="*60)
    print("ШАГ 1: Генерация s_rx (5 антенн, линейные задержки)")
    print("="*60)

    # GPU — через LfmGeneratorAnalyticalDelay
    gen_gpu = gpuworklib.LfmGeneratorAnalyticalDelay(ctx, f_start=F_START, f_end=F_END)
    gen_gpu.set_sampling(fs=FS, length=N)
    gen_gpu.set_delays(DELAYS_LINEAR_US.tolist())  # [мкс]
    rx_gpu = gen_gpu.generate_gpu()               # shape (5, 4000)

    # CPU — NumPy эталон
    rx_cpu = generate_rx_numpy(DELAYS_LINEAR_S)   # shape (5, 4000)

    # Вывод значений:
    for k in range(ANTENNAS):
        max_gpu = np.max(np.abs(rx_gpu[k]))
        max_cpu = np.max(np.abs(rx_cpu[k]))
        err     = np.max(np.abs(rx_gpu[k] - rx_cpu[k]))
        print(f"  Антенна {k}: max_GPU={max_gpu:.4f}, max_CPU={max_cpu:.4f}, err={err:.2e}")

    # График: реальная часть для антенны 0
    plot_step("01_rx_signals", rx_gpu, rx_cpu,
              title="Шаг 1: s_rx — принятый ЛЧМ (5 антенн)",
              xlabel="Отсчёт", ylabel="Re(s_rx)")
    return rx_gpu, rx_cpu


def step02_generate_ref_conjugate():
    """Шаг 2: Генерация сопряжённого опорного ЛЧМ s_ref*."""
    print("\n" + "="*60)
    print("ШАГ 2: Генерация s_ref* = conj(s_tx), delay=0")
    print("="*60)

    # GPU — через LfmConjugateGenerator (новый класс)
    conj_gen = gpuworklib.LfmConjugateGenerator(ctx)
    params = gpuworklib.LfmConjugateParams()
    params.f_start = F_START
    params.f_end   = F_END
    params.sample_rate = FS
    params.num_samples = N
    conj_gen.set_params(params)
    ref_gpu = conj_gen.generate_to_cpu()   # shape (4000,)

    # CPU NumPy эталон:
    t       = np.arange(N) / FS
    mu      = (F_END - F_START) / T
    phase   = -(np.pi * mu * t**2 + 2*np.pi * F_START * t)
    ref_cpu = np.exp(1j * phase).astype(np.complex64)

    # Вывод:
    err = np.max(np.abs(ref_gpu - ref_cpu))
    print(f"  max|ref_GPU - ref_CPU| = {err:.2e}")
    print(f"  ref_gpu[0] = {ref_gpu[0]:.4f}  ref_cpu[0] = {ref_cpu[0]:.4f}")
    print(f"  ref_gpu[1] = {ref_gpu[1]:.4f}  ref_cpu[1] = {ref_cpu[1]:.4f}")

    # График: фаза s_ref* — должна быть параболической (отражённой)
    plot_step("02_ref_conjugate", ref_gpu.reshape(1,-1), ref_cpu.reshape(1,-1),
              title="Шаг 2: s_ref* = conj(LFM), фаза",
              xlabel="Отсчёт", ylabel="arg(s_ref*)", mode="phase")
    return ref_gpu, ref_cpu


def step03_dechirp():
    """Шаг 3: Дечирп s_dc = s_rx × s_ref*."""
    print("\n" + "="*60)
    print("ШАГ 3: Дечирп s_dc = s_rx × conj(s_tx)")
    print("="*60)
    # ... GPU через HeterodyneDechirp или прямой kernel
    # ... CPU: dc_cpu[k] = rx_cpu[k] * ref_cpu
    # Вывод: max_gpu, max_cpu, error для каждой антенны
    # График: Re(s_dc) — должна быть видна синусоида f_beat


def step04_fft():
    """Шаг 4: FFT дечирпованного сигнала."""
    # Вывод: бин пика, f_peak_Hz для каждой антенны
    # График: спектр |FFT(s_dc)| в дБ — пики на f_beat bins [100,200,300,400,500]


def step05_find_maxima():
    """Шаг 5: FindMaxima → f_beat → R."""
    # Вывод: таблица [ant, f_beat_Hz, R_m, R_true_m, err_m]
    # GPU: через SpectrumMaximaFinder
    # CPU: numpy argmax + параболическая интерполяция


def step06_find_all_maxima():
    """Шаг 6: FindAllMaxima — контроль (все пики)."""
    # Вывод: список всех значимых пиков для каждой антенны
    # Ожидается: ровно 1 доминирующий пик


def step07_dechirp_correct():
    """Шаг 7: Коррекция — компенсация f_beat."""
    # Умножение на exp(-j·2π·f_beat·t)
    # GPU: через dechirp_correct.cl kernel
    # CPU: numpy
    # Вывод: max_err, фаза DC-компоненты


def step08_verify_dc():
    """Шаг 8: Верификация — пик при 0 Гц."""
    # FFT скорректированного сигнала
    # ASSERT: пик в bin 0 ± tolerance
    # Вывод: bin_peak для каждой антенны (ожидаем ~0)
    # График: спектр после коррекции — пик при DC


def run_full_step_test():
    """Запуск всего пошагового теста."""
    rx_gpu, rx_cpu           = step01_generate_rx()
    ref_gpu, ref_cpu         = step02_generate_ref_conjugate()
    dc_gpu, dc_cpu           = step03_dechirp(rx_gpu, rx_cpu, ref_gpu, ref_cpu)
    spec_gpu, spec_cpu       = step04_fft(dc_gpu, dc_cpu)
    results_gpu, results_cpu = step05_find_maxima(spec_gpu, spec_cpu)
    step06_find_all_maxima(spec_gpu)
    corr_gpu, corr_cpu       = step07_dechirp_correct(dc_gpu, dc_cpu, results_gpu)
    step08_verify_dc(corr_gpu, corr_cpu)
    print_final_report(results_gpu, results_cpu)
```

### 5.3 `test_heterodyne_basic.py` — pytest тесты (быстрые)

```python
"""
test_heterodyne_basic.py
pytest-тесты гетеродина. Быстрые, без графиков.
"""

def test_lfm_conjugate_generator():
    """LfmConjugateGenerator: GPU vs NumPy."""
    # ...
    assert max_err < 1e-4

def test_dechirp_single_antenna():
    """Дечирп одной антенны, delay=100 мкс → f_beat≈300 кГц."""
    # GPU pipeline
    # assert |f_beat - 300000| < 5000

def test_dechirp_5_antennas_linear():
    """5 антенн, линейные задержки."""
    # assert все f_beat в допуске

def test_dechirp_5_antennas_random():
    """5 антенн, случайные задержки (seed=42)."""

def test_dechirp_correct_dc():
    """После коррекции пик при DC."""
    # assert bin_peak == 0 ± 2

def test_gpu_vs_cpu_comparison():
    """Сравнение GPU и CPU результатов."""
    # assert max_err < tolerance

def test_process_external():
    """Тест ProcessExternal (имитация внешнего cl_mem)."""
    # Создать GPU buffer напрямую через gpuworklib
    # Передать как external_cl_mem
    # assert result.success
```

### 5.4 `test_heterodyne_comparison.py` — Детальное сравнение GPU vs CPU

```python
"""
test_heterodyne_comparison.py
Полное сравнение CPU(NumPy) vs GPU(gpuworklib).
Генерирует отчёт в Results/heterodyne/comparison_report.md
"""

def run_comparison_test():
    results_table = []
    for delay_set, label in [(DELAYS_LINEAR_US, "linear"), (DELAYS_RANDOM_US, "random")]:
        # GPU пайплайн
        # CPU пайплайн (NumPy)
        # Сравнение f_beat, R, SNR
        # Запись в таблицу

    # Markdown отчёт:
    # | Антенна | Задержка мкс | f_beat GPU | f_beat CPU | Δf_beat Гц | R_GPU м | R_CPU м | ΔR м |
    generate_markdown_report(results_table)

    # Красивые 4-панельные графики (тёмная тема):
    # Панель 1: Спектры всех 5 антенн — GPU (линейные задержки)
    # Панель 2: Спектры всех 5 антенн — CPU (линейные задержки)
    # Панель 3: Спектры — случайные задержки
    # Панель 4: Ошибки |GPU - CPU| в дБ
    save_comparison_plots()
```

---

## 6. Графики — спецификация

### 6.1 Каталог
```
Results/Plots/heterodyne/
├── step_01_rx_signals.png         # Re(s_rx) 5 антенн
├── step_02_ref_conjugate.png      # фаза s_ref*
├── step_03_dechirp.png            # Re(s_dc) — видны тоны
├── step_04_fft.png                # спектры после дечирпа
├── step_05_maxima.png             # пики и f_beat
├── step_06_all_maxima.png         # все максимумы
├── step_07_corrected.png          # сигнал после коррекции
├── step_08_verify_dc.png          # спектр → DC
├── comparison_linear.png          # GPU vs CPU, линейные
├── comparison_random.png          # GPU vs CPU, случайные
└── final_report.png               # итоговая таблица ошибок (4 панели)
```

### 6.2 Стиль (как в других тестах проекта)
- Тёмная тема: `fig.patch.set_facecolor('#1a1a2e')`
- Цвет осей: `ax.set_facecolor('#16213e')`
- GPU данные: `#00ff88` (зелёный)
- CPU данные: `#00d2ff` (голубой)
- Ошибки: `#ff6b6b` (красный)
- DPI = 150

---

## 7. Структура CMakeLists.txt

### 7.1 `modules/heterodyne/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20)

add_library(heterodyne STATIC
    src/heterodyne_dechirp.cpp
    src/heterodyne_processor_opencl.cpp
    src/heterodyne_processor_rocm.cpp     # заглушка
)

target_include_directories(heterodyne PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(heterodyne
    drv_gpu
    signal_generators    # для LfmConjugateGenerator
    fft_maxima           # для SpectrumMaximaFinder
    OpenCL::OpenCL
)

# Kernels — копировать в KERNELS_DIR (аналогично другим модулям)
file(GLOB HETERODYNE_KERNELS kernels/opencl/*.cl)
file(COPY ${HETERODYNE_KERNELS} DESTINATION ${KERNELS_DIR}/heterodyne)
```

### 7.2 В корневом CMakeLists.txt добавить:

```cmake
add_subdirectory(modules/heterodyne)

# Python bindings
if(BUILD_PYTHON)
    target_sources(gpuworklib PRIVATE
        modules/heterodyne/bindings/heterodyne_bindings.cpp
    )
    target_link_libraries(gpuworklib PRIVATE heterodyne)
endif()
```

### 7.3 В `src/main.cpp` добавить:

```cpp
#include "modules/heterodyne/tests/all_test.hpp"
// ...
heterodyne_all_test::run();
```

---

## 8. Python биндинги `heterodyne_bindings.cpp`

```cpp
// Добавить в PYBIND11_MODULE:

py::class_<drv_gpu_lib::HeterodyneParams>(m, "HeterodyneParams")
    .def(py::init<>())
    .def_readwrite("f_start",      &drv_gpu_lib::HeterodyneParams::f_start)
    .def_readwrite("f_end",        &drv_gpu_lib::HeterodyneParams::f_end)
    .def_readwrite("sample_rate",  &drv_gpu_lib::HeterodyneParams::sample_rate)
    .def_readwrite("num_samples",  &drv_gpu_lib::HeterodyneParams::num_samples)
    .def_readwrite("num_antennas", &drv_gpu_lib::HeterodyneParams::num_antennas);

py::class_<drv_gpu_lib::AntennaDechirpResult>(m, "AntennaDechirpResult")
    .def_readonly("antenna_idx",    &drv_gpu_lib::AntennaDechirpResult::antenna_idx)
    .def_readonly("f_beat_hz",      &drv_gpu_lib::AntennaDechirpResult::f_beat_hz)
    .def_readonly("f_beat_bin",     &drv_gpu_lib::AntennaDechirpResult::f_beat_bin)
    .def_readonly("range_m",        &drv_gpu_lib::AntennaDechirpResult::range_m)
    .def_readonly("peak_amplitude", &drv_gpu_lib::AntennaDechirpResult::peak_amplitude)
    .def_readonly("peak_snr_db",    &drv_gpu_lib::AntennaDechirpResult::peak_snr_db);

py::class_<drv_gpu_lib::HeterodyneResult>(m, "HeterodyneResult")
    .def_readonly("success",       &drv_gpu_lib::HeterodyneResult::success)
    .def_readonly("antennas",      &drv_gpu_lib::HeterodyneResult::antennas)
    .def_readonly("error_message", &drv_gpu_lib::HeterodyneResult::error_message);

py::class_<drv_gpu_lib::HeterodyneDechirp>(m, "HeterodyneDechirp")
    .def(py::init<drv_gpu_lib::IBackend*>())
    .def("set_params", &drv_gpu_lib::HeterodyneDechirp::SetParams)
    .def("process", [](drv_gpu_lib::HeterodyneDechirp& h,
                       py::array_t<std::complex<float>> rx) {
        // rx.shape = (num_antennas, num_samples)
        auto buf = rx.request();
        std::vector<std::complex<float>> data(
            static_cast<std::complex<float>*>(buf.ptr),
            static_cast<std::complex<float>*>(buf.ptr) + buf.size);
        return h.Process(data);
    });
```

---

## 9. Последовательность реализации для AI-исполнителя

### Этап 1: `LfmConjugateGenerator` (1 день)
1. Создать `lfm_conjugate_generator.cl` — ядро conj(LFM)
2. Создать `lfm_conjugate_generator.hpp` + `.cpp`
3. Добавить в CMakeLists.txt signal_generators
4. Python binding
5. Unit-тест C++: GPU vs NumPy, max_err < 1e-4
6. Unit-тест Python: `test_lfm_conjugate_generator()`

### Этап 2: Ядра дечирпа (0.5 дня)
1. `dechirp_multiply.cl` — kernel
2. `dechirp_correct.cl` — kernel
3. Unit-тест ядра `dechirp_multiply`: vs NumPy для 1 антенны

### Этап 3: HeterodyneProcessorOpenCL (1 день)
1. `i_heterodyne_processor.hpp`
2. `heterodyne_processor_opencl.hpp` + `.cpp` — реализует `Dechirp()` и `ProcessExternal()`
3. `heterodyne_processor_rocm.hpp` + `.cpp` — ЗАГЛУШКА: `throw std::runtime_error("ROCm not implemented")`
4. `heterodyne_dechirp.hpp` + `.cpp` — фасад

### Этап 4: Интеграция SpectrumMaximaFinder (0.5 дня)
1. В `HeterodyneProcessorOpenCL::Process()` добавить вызов `SpectrumMaximaFinder`
2. В `HeterodyneProcessorOpenCL::Process()` добавить вызов `FindAllMaxima`
3. Заполнить `HeterodyneResult::antennas`

### Этап 5: C++ тесты (1 день)
1. `test_heterodyne_basic.hpp` — тесты 1-4
2. `test_heterodyne_pipeline.hpp` — тесты 5-6
3. `test_heterodyne_external_ctx.hpp` — тест 7
4. `all_test.hpp` — оркестратор
5. `README.md` — документация тестов
6. Подключить в `src/main.cpp`

### Этап 6: Python тесты (1 день)
1. `Python_test/heterodyne/test_heterodyne_basic.py` — pytest
2. `Python_test/heterodyne/test_heterodyne_step_by_step.py` — пошаговый
3. `Python_test/heterodyne/test_heterodyne_comparison.py` — GPU vs CPU сравнение
4. Создать `Results/Plots/heterodyne/` каталог

### Этап 7: Финальная проверка (0.5 дня)
1. Запустить все C++ тесты → все PASS
2. Запустить все Python тесты → 100%
3. Обновить `MemoryBank/tests/test_results_YYYY-MM-DD.md`
4. Обновить `CLAUDE.md` — статус Heterodyne

---

## 10. Критерии готовности

| Критерий | Цель |
|----------|------|
| C++ тесты | 7/7 PASS |
| Python тесты | 100% |
| f_beat ошибка | < 5000 Гц (< 2 бина) |
| Дальность ошибка | < 0.5 м |
| GPU vs CPU ошибка | < 1e-3 |
| После коррекции: пик @ DC | bin 0 ± 2 |
| Grafики | 10 штук в Results/Plots/heterodyne/ |
| ProcessExternal() | Работает, не освобождает чужой буфер |

---

## 11. Важные ограничения (НЕЛЬЗЯ нарушать)

- ✅ Вывод на консоль ТОЛЬКО через `ConsoleOutput` из DrvGPU (не std::cout в многопоточном коде)
- ✅ Профилирование ТОЛЬКО через `GPUProfiler` — `PrintReport()` / `ExportMarkdown()`
- ✅ Ядра OpenCL ТОЛЬКО в `.cl` файлах, НЕ inline в `.cpp`
- ✅ ROCm — только заглушка с `throw` или `return false`
- ✅ `HeterodyneDechirp` НЕ владеет внешним `cl_mem` (не освобождает!)
- ✅ Тесты: только `all_test.hpp` → не вызывать тесты напрямую из main
- ✅ Логирование: через plog, файлы в `Logs/DRVGPU_XX/`
- ✅ Python тесты в `Python_test/heterodyne/test_*.py`
- ✅ Графики в `Results/Plots/heterodyne/`

---

## 12. Открытые вопросы для Alex

> **Alex, пожалуйста, проверь и дополни эти моменты:**

1. **f_start** — какая начальная частота ЛЧМ использовать? `0 Hz` или например `100 MHz`? (влияет только на фазу, не на f_beat)

2. **Задержки в мкс** — для 5 антенн [100, 200, 300, 400, 500] мкс — это задержки распространения от РАЗНЫХ целей или от ОДНОЙ цели с доплеровским сдвигом по антеннам? (от ответа зависит трактовка результата)

3. **ProcessExternal** — внешняя программа передаёт `cl_mem` напрямую или через дескриптор? Нужен ли маршаллинг через `void*` или сразу `cl_mem`?

4. **dechirp_correct** — реализовать Метод A (повторный дечирп) или Метод B (частотный сдвиг)? В ТЗ обозначено как "умножить на exp(-j·2π·f_beat·t)" → это Метод B.

5. **AllMaxima** — пороговый SNR для отсечения шумовых пиков? По умолчанию возьмём 20 дБ.

6. **Результаты тестов** — Markdown экспорт в `Results/heterodyne/` или в `Results/Profiler/`? Предлагаю отдельную папку.

---

*Документ подготовлен Кодо (AI Senior) для передачи AI-исполнителю.*
*Последнее обновление: 2026-02-21*

#### Дополнение обсуждение
1. я не понял вопрос 
 стр  ├── dechirp_multiply.cl         ← s_rx × conj(s_tx)
я предлагаю сопряжонный сигнал создавать из модуля генератора это его зона ответственности 