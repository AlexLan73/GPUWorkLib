# strategies — API Reference

> Полный справочник публичных классов и методов модуля `modules/strategies/`

**Namespace**: `strategies`
**Платформа**: ROCm 7.2+ / AMD GPU (`ENABLE_ROCM=1`)

---

## Содержание

1. [AntennaProcessor](#1-antennaprocessor)
2. [AntennaProcessor_v1](#2-antennaprocessor_v1)
3. [AntennaProcessorTest](#3-antennaprocessortest)
4. [WeightGenerator](#4-weightgenerator)
5. [Конфигурация](#5-конфигурация)
6. [Результаты](#6-результаты)
7. [Интерфейсы](#7-интерфейсы)
8. [Цепочки вызовов](#8-цепочки-вызовов)
9. [Python API](#9-python-api)

---

## 1. AntennaProcessor

**Файл**: `modules/strategies/include/antenna_processor.hpp`

Абстрактный базовый класс. Точка входа pipeline.

```cpp
namespace strategies {

class AntennaProcessor {
public:
  virtual ~AntennaProcessor() = default;

  /**
   * @brief Запуск полного pipeline
   * @param d_S  [n_ant × n_samples] complex float — сигнал уже на GPU
   * @param d_W  [n_ant × n_ant]     complex float — матрица весов уже на GPU
   * @return AntennaResult — статистика + пики + тайминги
   *
   * ⚠️ d_S и d_W должны быть в VRAM ДО вызова!
   * ⚠️ Caller owns d_S и d_W — процессор их НЕ освобождает
   */
  virtual AntennaResult process(const void* d_S, const void* d_W) = 0;

  // ─── Runtime конфигурация (можно менять между вызовами) ───────────────

  /** Выбрать post-FFT сценарий */
  virtual void set_scenario_mode(PostFftScenarioMode mode) = 0;

  /** Статистика до GEMM (на d_S). StatPreset::NONE — отключить */
  virtual void set_pre_input_stats(StatisticsSet stats) = 0;

  /** Статистика после GEMM (на d_X). StatPreset::NONE — отключить */
  virtual void set_post_gemm_stats(StatisticsSet stats) = 0;

  /** Статистика после FFT (на |spectrum|). StatPreset::NONE — отключить */
  virtual void set_post_fft_stats(StatisticsSet stats) = 0;

  /** Включить D2H memcpy на debug точках (медленно!) */
  virtual void set_debug_mode(bool enabled) = 0;

  // ─── Информация ───────────────────────────────────────────────────────

  virtual const AntennaProcessorConfig& config() const = 0;
  virtual int gpu_id() const = 0;
};

}  // namespace strategies
```

---

## 2. AntennaProcessor_v1

**Файл**: `modules/strategies/include/antenna_processor_v1.hpp`
**Реализация**: `modules/strategies/src/antenna_processor_v1.cpp`

Конкретная ROCm-реализация. 4 HIP-потока, hipBLAS, hipFFT, hiprtc.

```cpp
namespace strategies {

class AntennaProcessor_v1 : public AntennaProcessor {
public:
  /**
   * @brief Конструктор — выделяет VRAM, компилирует kernels, строит FFT plan
   * @param backend  DrvGPU backend (должен быть ROCm)
   * @param cfg      Конфигурация pipeline
   * @throws std::runtime_error если ENABLE_ROCM=0
   */
  explicit AntennaProcessor_v1(drv_gpu_lib::IBackend* backend,
                               const AntennaProcessorConfig& cfg);

  ~AntennaProcessor_v1();  // освобождает все VRAM буферы, streams, handles

  // Реализация всех virtual методов AntennaProcessor
  AntennaResult process(const void* d_S, const void* d_W) override;
  void set_scenario_mode(PostFftScenarioMode mode) override;
  void set_pre_input_stats(StatisticsSet stats) override;
  void set_post_gemm_stats(StatisticsSet stats) override;
  void set_post_fft_stats(StatisticsSet stats) override;
  void set_debug_mode(bool enabled) override;
  const AntennaProcessorConfig& config() const override;
  int gpu_id() const override;

protected:
  // ─── Protected шаги (открыты для AntennaProcessorTest) ───────────────

  void do_debug_point_21();        // Statistics(d_S) на stream_debug1_
  void do_gemm();                   // hipBLAS Cgemm: d_X = d_W × d_S
  void do_debug_point_22();        // Statistics(d_X) на stream_debug2_
  void do_window_fft();            // hamming_pad_fused + hipFFT
  void do_debug_point_23();        // Statistics(|spectrum|) на stream_debug3_
  void do_run_post_fft_scenarios();// Step2.1/2.2/2.3 по scenario_mode

  // ─── Буферы (VRAM) ───────────────────────────────────────────────────

  void* d_X              = nullptr;  // [n_ant × n_samples] GEMM output
  void* d_spectrum       = nullptr;  // [n_ant × nFFT]      FFT output
  void* d_magnitudes     = nullptr;  // [n_ant × nFFT]      |spectrum|
  void* d_hamming_window = nullptr;  // [n_samples]         precomputed
  void* d_one_max_results= nullptr;  // Step2.1 raw
  void* d_minmax_results = nullptr;  // Step2.3 raw
};

}  // namespace strategies
```

---

## 3. AntennaProcessorTest

**Файл**: `modules/strategies/include/antenna_processor_test.hpp`

> ⚠️ Только для тестов! В production не использовать.

Наследник `AntennaProcessor_v1`, открывает step-by-step API для валидации каждого шага.

```cpp
namespace strategies {

class AntennaProcessorTest : public AntennaProcessor_v1 {
public:
  explicit AntennaProcessorTest(drv_gpu_lib::IBackend* backend,
                                const AntennaProcessorConfig& cfg);

  // ─── Пошаговый API (порядок важен!) ─────────────────────────────────

  /**
   * Шаг 0: сохранить указатели d_S и d_W для последующих шагов
   * Вызывать первым!
   */
  void step_0_prepare_input(const void* d_S, const void* d_W);

  /**
   * Шаг 1: Statistics(d_S) → D2H → вывод на консоль
   * Выводит: mean, std, min, max, median per antenna
   */
  void step_1_debug_input();

  /**
   * Шаг 2: GEMM (d_X = d_W × d_S) → D2H → вывод первых 5 значений
   * Ждёт завершения на CPU (hipStreamSynchronize)
   */
  void step_2_gemm();

  /**
   * Шаг 3: Statistics(d_X) → D2H → вывод
   */
  void step_3_debug_post_gemm();

  /**
   * Шаг 4: Hamming + FFT → D2H spectrum → вывод первых 5 бинов per beam
   */
  void step_4_window_fft();

  /**
   * Шаг 5: Statistics(|spectrum|) → D2H → вывод
   */
  void step_5_debug_post_fft();

  /**
   * Шаг 6.1: только Step2.1 (OneMax + parabola)
   */
  void step_6_1_one_max();

  /**
   * Шаг 6.2: только Step2.2 (AllMaxima, limit=cfg.maxima_limit)
   */
  void step_6_2_all_maxima();

  /**
   * Шаг 6.3: только Step2.3 (GlobalMinMax)
   */
  void step_6_3_global_minmax();

  /**
   * Запуск полного pipeline (эквивалентно базовому process())
   * Не нужно вызывать step_0 перед этим методом
   */
  AntennaResult process_full();
};

}  // namespace strategies
```

---

## 4. WeightGenerator

**Файл**: `modules/strategies/include/weight_generator.hpp`
**Реализация**: `modules/strategies/src/weight_generator.cpp`

Статический класс. Генерация и загрузка матрицы весов.

```cpp
namespace strategies {

struct WeightParams {
  uint32_t n_ant    = 5;       ///< Число антенн
  double   f0       = 2.0e6;   ///< Несущая частота, Гц
  double   tau_base = 0.0;     ///< Базовая задержка, секунды
  double   tau_step = 100e-6;  ///< Шаг задержки per antenna, секунды
};

class WeightGenerator {
public:
  /**
   * @brief CPU: вычислить матрицу весов delay-and-sum
   *
   * W[b][k] = exp(-j * 2π * f0 * τ_k) / sqrt(n_ant)
   * τ_k = tau_base + k * tau_step
   *
   * Возвращает вектор [n_ant * n_ant] (row-major, beam × antenna)
   * Совместимо с NumPy: np.exp(-1j * 2*π * f0 * tau) / sqrt(n)
   */
  static std::vector<std::complex<float>>
      generate_delay_and_sum(const WeightParams& params);

  /**
   * @brief GPU: выделить VRAM + скопировать матрицу
   *
   * @param backend  DrvGPU backend
   * @param weights  CPU матрица от generate_delay_and_sum()
   * @return void*   Указатель на d_W в VRAM
   *
   * ⚠️ Caller owns — нужно освободить через backend->Free(d_W) !
   */
  static void* upload_to_gpu(
      drv_gpu_lib::IBackend* backend,
      const std::vector<std::complex<float>>& weights);
};

}  // namespace strategies
```

---

## 5. Конфигурация

**Файл**: `modules/strategies/include/config/antenna_processor_config.hpp`

```cpp
namespace strategies {

/** Конфигурация checkpoint сохранений */
struct CheckpointSaveConfig {
  bool c1_signal   = false;  ///< d_S (2.5 ГБ — дорого!)
  bool c1_weights  = false;  ///< d_W
  bool c2_data     = false;  ///< d_X после GEMM (2.5 ГБ — дорого!)
  bool c2_stats    = false;  ///< PRE+POST stats
  bool c3_result   = true;   ///< MinMaxResult (дёшево) ← default
  bool c3_spectrum = false;  ///< Полный spectrum (4.9 ГБ — огромный!)
  bool c4_result   = true;   ///< MaxValue peaks (дёшево) ← default
  bool json_header = false;  ///< JSON+binary header
};

/** Основная конфигурация pipeline */
struct AntennaProcessorConfig {
  uint32_t n_ant         = 5;
  uint32_t n_samples     = 8000;
  float    sample_rate   = 12.0e6f;

  PostFftScenarioMode scenario_mode = PostFftScenarioMode::ALL_REQUIRED;
  uint32_t maxima_limit             = 1000;    ///< Step2.2 limit
  float    signal_frequency_hz      = 2.0e6f;  ///< для валидации

  StatisticsSet pre_input_stats = StatPreset::P61_ALL;
  StatisticsSet post_gemm_stats = StatPreset::P61_ALL;
  StatisticsSet post_fft_stats  = StatPreset::P61_ALL;

  const CheckpointSaveConfig* save_cfg = nullptr;  ///< nullptr = NullCheckpointSave
  bool debug_mode = false;
};

}  // namespace strategies
```

**Файл**: `modules/strategies/include/config/post_fft_scenario_mode.hpp`

```cpp
namespace strategies {

enum class PostFftScenarioMode : uint8_t {
  ALL_REQUIRED     = 0,  ///< Step2.1 + Step2.2 + Step2.3 (production)
  ONE_MAX_PARABOLA = 1,  ///< Step2.1 only
  ALL_MAXIMA       = 2,  ///< Step2.2 only
  GLOBAL_MINMAX    = 3   ///< Step2.3 only
};

}
```

**Файл**: `modules/strategies/include/config/statistics_set.hpp`

```cpp
namespace strategies {

using StatisticsSet = uint8_t;

// Биты
constexpr StatisticsSet STAT_MEAN   = 0x01;
constexpr StatisticsSet STAT_MEDIAN = 0x02;
constexpr StatisticsSet STAT_STD    = 0x04;
constexpr StatisticsSet STAT_VAR    = 0x08;
constexpr StatisticsSet STAT_MIN    = 0x10;
constexpr StatisticsSet STAT_MAX    = 0x20;

namespace StatPreset {
  constexpr StatisticsSet NONE         = 0x00;
  constexpr StatisticsSet P61_ALL      = 0x3F;  ///< MEAN|MED|STD|VAR|MIN|MAX
  constexpr StatisticsSet P62_MEAN_MED = STAT_MEAN | STAT_MEDIAN;
  constexpr StatisticsSet P63_MED_MM   = STAT_MEAN | STAT_MEDIAN | STAT_MIN | STAT_MAX;
  constexpr StatisticsSet P64_STD_VAR  = STAT_STD | STAT_VAR;
}

}
```

---

## 6. Результаты

**Файл**: `modules/strategies/include/result_types.hpp`

```cpp
namespace strategies {

/** Step2.1: главный пик + параболическое уточнение (без фазы) */
struct OneMaxParabolaLite {
  uint32_t beam_id         = 0;
  uint32_t bin_index       = 0;    ///< FFT bin пика
  float    magnitude       = 0.f;  ///< |FFT[bin]|
  float    freq_offset     = 0.f;  ///< delta в [-0.5, +0.5]
  float    refined_freq_hz = 0.f;  ///< (bin + delta) * fs / nFFT
};

/** Step2.3: глобальный MIN+MAX per beam */
struct MinMaxResult {
  uint32_t beam_id          = 0;
  float    min_magnitude    = 0.f;
  uint32_t min_bin          = 0;
  float    min_frequency_hz = 0.f;
  float    max_magnitude    = 0.f;
  uint32_t max_bin          = 0;
  float    max_frequency_hz = 0.f;
  float    dynamic_range_dB = 0.f;  ///< 20·log10(max/max(min, 1e-30))
  uint32_t pad              = 0;    ///< 32-byte alignment
};

/** Тайминги каждого шага в мс */
struct PerfMetrics {
  float debug_21_ms = 0.f;  // Statistics(d_S)
  float gemm_ms     = 0.f;  // GEMM
  float debug_22_ms = 0.f;  // Statistics(d_X)
  float window_ms   = 0.f;  // Hamming fused
  float fft_ms      = 0.f;  // hipFFT batch
  float debug_23_ms = 0.f;  // Statistics(|spectrum|)
  float step21_ms   = 0.f;  // OneMax
  float step22_ms   = 0.f;  // AllMaxima
  float step23_ms   = 0.f;  // GlobalMinMax
  float total_ms    = 0.f;
};

/** Агрегированный результат process() */
struct AntennaResult {
  // Статистика на 3 debug точках (size = n_ant)
  std::vector<statistics::StatisticsResult> pre_input_stats;
  std::vector<statistics::StatisticsResult> post_gemm_stats;
  std::vector<statistics::StatisticsResult> post_fft_stats;

  std::vector<statistics::MedianResult> pre_input_medians;
  std::vector<statistics::MedianResult> post_gemm_medians;
  std::vector<statistics::MedianResult> post_fft_medians;

  // Post-FFT результаты (size = n_ant)
  std::vector<OneMaxParabolaLite>               one_max;     // Step2.1
  std::vector<antenna_fft::AllMaximaBeamResult> all_maxima;  // Step2.2
  std::vector<MinMaxResult>                     minmax;      // Step2.3

  PostFftScenarioMode scenario_mode = PostFftScenarioMode::ALL_REQUIRED;
  PerfMetrics         perf;
};

}  // namespace strategies
```

---

## 7. Интерфейсы

### ICheckpointSave

**Файл**: `modules/strategies/include/interfaces/i_checkpoint_save.hpp`

```cpp
namespace strategies {

class ICheckpointSave {
public:
  virtual ~ICheckpointSave() = default;

  virtual void save_c1_signal(const void* d_data, uint32_t n_ant,
      uint32_t n_samples, float sample_rate, int gpu_id) = 0;
  virtual void save_c1_weights(const void* d_weights,
      uint32_t n_ant, int gpu_id) = 0;
  virtual void save_c2_data(const void* d_X, uint32_t n_ant,
      uint32_t n_samples, float sample_rate, int gpu_id) = 0;
  virtual void save_c2_stats(const std::vector<statistics::StatisticsResult>& pre,
      const std::vector<statistics::StatisticsResult>& post,
      uint32_t n_ant, int gpu_id) = 0;
  virtual void save_c3_spectrum(const void* d_spectrum, uint32_t n_ant,
      uint32_t nFFT, int gpu_id) = 0;
  virtual void save_c3_minmax(const std::vector<MinMaxResult>& results,
      uint32_t n_ant, int gpu_id) = 0;
  virtual void save_c4_one_max(const std::vector<OneMaxParabolaLite>& results,
      uint32_t n_ant, int gpu_id) = 0;
};

}
```

### NullCheckpointSave

**Файл**: `modules/strategies/include/checkpoint/null_checkpoint_save.hpp`

```cpp
namespace strategies {

// Null Object — все методы пусты (inline), нулевой overhead
class NullCheckpointSave : public ICheckpointSave {
public:
  void save_c1_signal(...) override {}
  void save_c1_weights(...) override {}
  void save_c2_data(...) override {}
  void save_c2_stats(...) override {}
  void save_c3_spectrum(...) override {}
  void save_c3_minmax(...) override {}
  void save_c4_one_max(...) override {}
};

}
```

### IPostFftScenario

**Файл**: `modules/strategies/include/interfaces/i_post_fft_scenario.hpp`

```cpp
namespace strategies {

class IPostFftScenario {
public:
  virtual ~IPostFftScenario() = default;

  virtual void execute(
      const void* d_spectrum,    // FFT output
      const void* d_magnitudes,  // |FFT| per beam
      uint32_t n_ant,
      uint32_t nFFT,
      float sample_rate,
      uint32_t maxima_limit,
      hipStream_t stream) = 0;

  virtual const char* name() const = 0;
};

}
```

---

## 8. Цепочки вызовов

### Production — полный pipeline

```cpp
#include "antenna_processor_v1.hpp"
#include "weight_generator.hpp"

// ── 1. Конфиг ──────────────────────────────────────────────────────────
strategies::AntennaProcessorConfig cfg;
cfg.n_ant            = 256;
cfg.n_samples        = 1'200'000;
cfg.sample_rate      = 12.0e6f;
cfg.scenario_mode    = strategies::PostFftScenarioMode::ALL_REQUIRED;
cfg.pre_input_stats  = strategies::StatPreset::P61_ALL;
cfg.post_gemm_stats  = strategies::StatPreset::P61_ALL;
cfg.post_fft_stats   = strategies::StatPreset::P61_ALL;
// cfg.save_cfg = nullptr → NullCheckpointSave (zero overhead)

// ── 2. Матрица весов ────────────────────────────────────────────────────
strategies::WeightParams wp;
wp.n_ant    = cfg.n_ant;
wp.f0       = 2.0e6;
wp.tau_base = 0.0;
wp.tau_step = 100e-6;

auto W_cpu = strategies::WeightGenerator::generate_delay_and_sum(wp);
void* d_W  = strategies::WeightGenerator::upload_to_gpu(backend, W_cpu);
// ⚠️ Помнить: backend->Free(d_W) после использования!

// ── 3. Создать процессор ────────────────────────────────────────────────
strategies::AntennaProcessor_v1 proc(backend, cfg);

// ── 4. Запуск (d_S уже в VRAM) ─────────────────────────────────────────
strategies::AntennaResult r = proc.process(d_S, d_W);

// ── 5. Результаты ───────────────────────────────────────────────────────
for (uint32_t b = 0; b < cfg.n_ant; ++b) {
  float f     = r.one_max[b].refined_freq_hz;
  float dr_db = r.minmax[b].dynamic_range_dB;
  float median= r.pre_input_stats[b].median;
}
float total_ms = r.perf.total_ms;

// ── 6. Освобождение ─────────────────────────────────────────────────────
backend->Free(d_W);  // Caller owns d_W!
// proc, r — RAII, освободятся автоматически
```

### Debug — пошаговый тест

```cpp
#include "antenna_processor_test.hpp"

strategies::AntennaProcessorConfig cfg;
cfg.n_ant       = 5;
cfg.n_samples   = 8000;
cfg.debug_mode  = true;  // включить D2H на debug точках

strategies::AntennaProcessorTest proc(backend, cfg);

proc.step_0_prepare_input(d_S, d_W);
proc.step_1_debug_input();      // Выводит статистику d_S
proc.step_2_gemm();             // X = W×S, выводит первые значения
proc.step_3_debug_post_gemm();  // Статистика d_X
proc.step_4_window_fft();       // Hamming + FFT, выводит спектр
proc.step_5_debug_post_fft();   // Статистика |spectrum|
proc.step_6_1_one_max();        // Step2.1
proc.step_6_3_global_minmax();  // Step2.3
// или proc.process_full();     // Весь pipeline без step_0
```

### Профилирование

```cpp
auto& profiler = backend->GetProfiler();
profiler.SetGPUInfo(backend->GetDeviceInfo());  // ← ОБЯЗАТЕЛЬНО!
profiler.Start("strategies_benchmark");

auto r = proc.process(d_S, d_W);

profiler.Stop();
profiler.PrintReport();
profiler.ExportMarkdown("Results/Profiler/strategies_2026-03-09.md");
profiler.ExportJSON("Results/Profiler/strategies_2026-03-09.json");
```

---

## 9. Python API

### pipeline_runner.py

```python
from Python_test.strategies.pipeline_runner import (
    PipelineRunner,
    PipelineConfig,
    PipelineResult,  # .peaks, .stats, .spectrum
    PeakInfo,        # .beam_id, .bin_index, .freq_hz, .magnitude, .phase_rad
    ChannelStats,    # .mean, .std, .power, .max, .min
)

runner = PipelineRunner(output_dir="Results/strategies/run_01")
cfg    = PipelineConfig(save_input=True, save_spectrum=True,
                        save_stats=True, save_results=True)

# Pipeline A: фазовая коррекция
result_a = runner.run_pipeline_a(scenario, steer_theta=30,
                                 steer_freq=2e6, config=cfg)

# Pipeline B: Farrow delay alignment
result_b = runner.run_pipeline_b(scenario, steer_theta=30, config=cfg)

# Сравнение двух pipeline
comp = runner.compare(result_a, result_b)

# Доступ
peaks    = result_a.peaks      # List[PeakInfo]
spectrum = result_a.spectrum   # np.ndarray [n_ant, nFFT]
stats    = result_a.stats      # List[ChannelStats]
```

### scenario_builder.py

```python
from Python_test.strategies.scenario_builder import (
    ScenarioBuilder,
    ULAGeometry,
    EmitterSignal,
    make_single_target,
    make_target_and_jammer,
    make_multi_target,
)

# Fluent API
scenario = (
    ScenarioBuilder(n_ant=8, d_ant_m=0.075, c=3e8)
    .add_target(theta_deg=30, f0_hz=2e6, fdev_hz=0, amplitude=1.0)
    .add_jammer(theta_deg=60, f0_hz=3e6, amplitude=0.3)
    .set_noise(snr_db=20)
    .build()
)

# Готовые фабрики
scenario = make_single_target(n_ant=8, theta_deg=30, f0_hz=2e6, snr_db=20)
scenario = make_multi_target(n_ant=8, thetas=[15, 30, 45], f0_hz=2e6)
scenario = make_target_and_jammer(n_ant=8, target_theta=30, jammer_theta=60)

# Доступ
delays       = scenario.geometry.compute_delays(30)  # seconds per antenna
signal_matrix= scenario.signal_matrix                 # np.ndarray [n_ant, n_samples]
```

### farrow_delay.py

```python
from Python_test.strategies.farrow_delay import FarrowDelay

farrow = FarrowDelay()  # загружает Lagrange 48×5 из JSON

# Одна антенна, задержка в отсчётах
delayed = farrow.apply_single(signal_1d, delay_samples=2.7)

# Матрица антенн, задержки в отсчётах
delayed = farrow.apply(signal_matrix, delays_samples=[0, 1.5, 3.0, 4.5])

# Задержки в секундах (автоматически конвертирует через fs)
delayed = farrow.apply_seconds(signal_matrix,
                               delays_sec=[0, 1e-6, 2e-6],
                               fs=12e6)

# Компенсировать задержку (обратная операция)
restored = farrow.compensate(delayed, delays_samples=[0, 1.5, 3.0, 4.5])
restored = farrow.compensate_seconds(delayed,
                                     delays_sec=[0, 1e-6, 2e-6],
                                     fs=12e6)
```

### conftest.py fixtures

```python

def farrow():
    return FarrowDelay()


def scenario_8ant():
    """ULA 8 антенн, 1 цель @ 30°, f0=2МГц"""
    return make_single_target(n_ant=8, theta_deg=30, f0_hz=2e6)


def scenario_multi():
    """ULA 8 антенн, 3 цели @ 15°/30°/45°"""
    return make_multi_target(n_ant=8, thetas=[15, 30, 45], f0_hz=2e6)


def pipeline_runner(tmp_path):
    return PipelineRunner(output_dir=None)


def strategy_plot_dir():
    return Path("Results/Plots/strategies")
```

---

## См. также

- [Full.md](Full.md) — математика, pipeline, тесты, VRAM layout
- [Quick.md](Quick.md) — концепция и быстрый старт
- [AP_C4_Code.md](AP_C4_Code.md) — детальные C4 Code диаграммы

---

*Обновлено: 2026-03-09 | Автор: Кодо*
