# 🔧 C++ Test Utils — Предложение по извлечению общего кода

> **Дата**: 2026-03-21
> **Автор**: Кодо
> **Статус**: PROPOSAL (на согласование с Alex)

---

## 📊 Текущее состояние C++ тестов

| Метрика | Значение |
|---------|----------|
| Тестовых файлов | **83** (.hpp) |
| Объём кода | **~20,000 LOC** |
| Модулей с тестами | **13** |
| Benchmark'ов (GpuBenchmarkBase) | **15+** классов |
| Файлов с дублированием | **40-80** (в зависимости от паттерна) |

---

## 🔴 Найденное дублирование (5 областей)

### 1. CPU Reference Functions (~200 LOC дублировано в 8+ файлах)

```
Функция                    Где дублируется
─────────────────────      ─────────────────────────────────────────
MaxError(a, b, n)          test_signal_generators, test_form_signal,
                           test_fir_basic, test_lch_farrow и др.
CpuMean(data, n)           test_statistics_rocm + копии
CpuMedian(data, n)         test_statistics_rocm + копии
CpuVariance(data, n)       test_statistics_rocm + копии
FindPeakBin(fft, fs)       test_signal_generators, test_heterodyne
GetXReference(...)         test_form_signal (FormSignal CPU-эталон)
```

**Аналог в Python**: `common/references/` (SignalReferences, StatisticsReferences, FftReferences)

---

### 2. Signal Generation Helpers (~150 LOC дублировано в 6+ файлах)

```
Функция                      Где дублируется
──────────────────────        ──────────────────────────────────
GenerateCw(fs, f0, n, amp)   test_signal_generators, test_lch_farrow,
                              test_fir_basic
GenerateLfm(fs, f0, f1, n)   test_heterodyne_basic, test_signal_generators
delayedLFM(fs, ...)           test_heterodyne_basic
GenerateSinusoid(...)         test_statistics_rocm
GenerateConstant(val, n)      test_statistics_rocm
GenerateTestSignal(...)       test_fir_basic (CW 100Hz + 5kHz)
```

**Аналог в Python**: `common/references/signal_refs.py` — cw(), lfm(), noise()

---

### 3. GPU Memory Read Pattern (~100 LOC × 40 файлов!)

Один и тот же код в ~40 местах:
```cpp
// Типичный паттерн (5-8 строк каждый раз):
cl_mem gpu_buf = processor.ProcessToGpu(input);
std::vector<std::complex<float>> result(n);
auto q = static_cast<cl_command_queue>(backend->GetNativeQueue());
clEnqueueReadBuffer(q, gpu_buf, CL_TRUE, 0,
                    n * sizeof(std::complex<float>),
                    result.data(), 0, nullptr, nullptr);
clReleaseMemObject(gpu_buf);
```

ROCm-аналог:
```cpp
void* gpu_ptr = processor.ProcessToGpu(input);
std::vector<float> result(n);
hipMemcpy(result.data(), gpu_ptr, n * sizeof(float), hipMemcpyDeviceToHost);
hipFree(gpu_ptr);
```

**Аналог в Python**: нет (pybind11 возвращает ndarray напрямую)

---

### 4. Validation / PASS-FAIL (~50 LOC × 80 файлов)

Каждый тест сам форматирует результат:
```cpp
float err = MaxError(gpu_data.data(), cpu_ref.data(), n);
bool pass = (err < 1e-3f);
con.Print(0, "Signal", pass ? "[PASS]" : "[FAIL]");
con.Print(0, "Signal", "  MaxError = " + std::to_string(err));
```

Нет единого формата. Tolerance хардкодится (1e-2, 1e-3, 1e-4).

**Аналог в Python**: `common/validators/` (DataValidator, RelativeValidator, CompositeValidator)

---

### 5. Backend Init (тривиально, но ~80 раз)

```cpp
auto backend = std::make_unique<ROCmBackend>();
backend->Initialize(gpu_id);
// или
auto& drv = DrvGPU::GetInstance();
drv.Initialize(gpu_id);
```

---

## 🎯 Предложение: `include/test_utils/`

### Структура

```
include/test_utils/
├── test_utils.hpp          ← Master include (все 4 файла)
├── references.hpp          ← CPU-эталоны сигналов и статистики
├── validators.hpp          ← Сравнение GPU vs CPU
├── gpu_transfer.hpp        ← GPU→CPU чтение буферов
└── test_format.hpp         ← Единый формат PASS/FAIL вывода
```

### Namespace: `gpu_test_utils`

---

### 📄 references.hpp — CPU-эталоны

```cpp
namespace gpu_test_utils {

// ── Сигналы ──────────────────────────────────────────────────────
inline std::vector<std::complex<float>>
GenerateCw(float fs, float f0, size_t n_samples, float amplitude = 1.0f) {
    std::vector<std::complex<float>> sig(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        float t = static_cast<float>(i) / fs;
        float phase = 2.0f * M_PI * f0 * t;
        sig[i] = amplitude * std::complex<float>(cosf(phase), sinf(phase));
    }
    return sig;
}

inline std::vector<std::complex<float>>
GenerateLfm(float fs, float f_start, float f_end, size_t n_samples,
            float amplitude = 1.0f) { ... }

inline std::vector<std::complex<float>>
GenerateFormSignal(float fs, size_t points, float f0, float amplitude,
                   float phase, float fdev, float norm_val,
                   float tau = 0.0f) { ... }  // GetXReference

inline std::vector<std::complex<float>>
GenerateNoise(size_t n_samples, float amplitude = 1.0f,
              uint32_t seed = 42) { ... }

// ── Статистика ───────────────────────────────────────────────────
template<typename T>
inline T CpuMean(const T* data, size_t n) {
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) sum += data[i];
    return sum / static_cast<T>(n);
}

template<typename T> inline T CpuMedian(const T* data, size_t n) { ... }
template<typename T> inline T CpuVariance(const T* data, size_t n) { ... }
template<typename T> inline T CpuStd(const T* data, size_t n) { ... }

// ── FFT ──────────────────────────────────────────────────────────
inline size_t FindPeakBin(const float* magnitude, size_t n_bins) {
    return std::distance(magnitude,
                         std::max_element(magnitude, magnitude + n_bins));
}

inline float PeakFreqHz(const float* magnitude, size_t n_bins, float fs) {
    size_t peak = FindPeakBin(magnitude, n_bins);
    return static_cast<float>(peak) * fs / static_cast<float>(n_bins);
}

} // namespace gpu_test_utils
```

---

### 📄 validators.hpp — Сравнение GPU vs CPU

```cpp
namespace gpu_test_utils {

struct ValidationResult {
    bool passed;
    std::string metric_name;
    float actual_value;
    float threshold;
    std::string message;
};

// max|a-b| / max|b| < tolerance
template<typename T>
inline ValidationResult
MaxRelError(const T* actual, const T* reference, size_t n,
            float tolerance, const std::string& name = "max_rel") {
    float max_diff = 0.0f;
    float max_ref = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(actual[i] - reference[i]));
        max_ref  = std::max(max_ref,  std::abs(reference[i]));
    }
    float err = (max_ref > 1e-15f) ? max_diff / max_ref : max_diff;
    return {err < tolerance, name, err, tolerance, ""};
}

// max|a-b| < tolerance
template<typename T>
inline ValidationResult
AbsError(const T* actual, const T* reference, size_t n,
         float tolerance, const std::string& name = "abs") { ... }

// Проверка пика FFT
inline ValidationResult
CheckPeakFreq(const float* magnitude, size_t n_bins, float fs,
              float expected_hz, float tolerance_hz) {
    float actual_hz = PeakFreqHz(magnitude, n_bins, fs);
    float err = std::abs(actual_hz - expected_hz);
    return {err < tolerance_hz, "peak_freq_hz", actual_hz, tolerance_hz,
            "expected=" + std::to_string(expected_hz) + "Hz"};
}

} // namespace gpu_test_utils
```

---

### 📄 gpu_transfer.hpp — GPU→CPU чтение

```cpp
namespace gpu_test_utils {

// OpenCL: прочитать cl_mem → vector<T>, автоматический release
template<typename T>
inline std::vector<T>
ReadClBuffer(cl_command_queue queue, cl_mem buffer, size_t count,
             bool release_buffer = true) {
    std::vector<T> result(count);
    clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0,
                        count * sizeof(T), result.data(), 0, nullptr, nullptr);
    if (release_buffer) clReleaseMemObject(buffer);
    return result;
}

// ROCm: прочитать device ptr → vector<T>, автоматический hipFree
template<typename T>
inline std::vector<T>
ReadHipBuffer(void* device_ptr, size_t count, bool free_buffer = true) {
    std::vector<T> result(count);
    hipMemcpy(result.data(), device_ptr, count * sizeof(T),
              hipMemcpyDeviceToHost);
    if (free_buffer) hipFree(device_ptr);
    return result;
}

// Backend-agnostic: определяет тип по IBackend
template<typename T>
inline std::vector<T>
ReadGpuBuffer(IBackend* backend, void* buffer, size_t count,
              bool release = true) {
    if (backend->GetBackendType() == BackendType::ROCm)
        return ReadHipBuffer<T>(buffer, count, release);
    else
        return ReadClBuffer<T>(
            static_cast<cl_command_queue>(backend->GetNativeQueue()),
            static_cast<cl_mem>(buffer), count, release);
}

} // namespace gpu_test_utils
```

---

### 📄 test_format.hpp — Единый вывод

```cpp
namespace gpu_test_utils {

inline void PrintTestHeader(ConsoleOutput& con, int gpu_id,
                            const std::string& module,
                            const std::string& test_name) {
    con.Print(gpu_id, module, "════════════════════════════════════════");
    con.Print(gpu_id, module, "  TEST: " + test_name);
    con.Print(gpu_id, module, "────────────────────────────────────────");
}

inline void PrintValidation(ConsoleOutput& con, int gpu_id,
                            const std::string& module,
                            const ValidationResult& vr) {
    std::string status = vr.passed ? "[PASS]" : "[FAIL]";
    std::string line = "  " + status + " " + vr.metric_name +
        ": " + std::to_string(vr.actual_value) +
        " (tol=" + std::to_string(vr.threshold) + ")";
    if (!vr.message.empty()) line += " " + vr.message;

    if (vr.passed)
        con.Print(gpu_id, module, line);
    else
        con.PrintError(gpu_id, module, line);
}

inline void PrintTestSummary(ConsoleOutput& con, int gpu_id,
                             const std::string& module,
                             int n_pass, int n_fail, int n_skip = 0) {
    con.Print(gpu_id, module, "════════════════════════════════════════");
    con.Print(gpu_id, module, "  SUMMARY: " +
              std::to_string(n_pass) + " passed, " +
              std::to_string(n_fail) + " failed" +
              (n_skip > 0 ? ", " + std::to_string(n_skip) + " skipped" : ""));
    con.Print(gpu_id, module, "════════════════════════════════════════");
}

} // namespace gpu_test_utils
```

---

## 📊 Маппинг: C++ test_utils ↔ Python common/

```
C++ include/test_utils/       Python_test/common/
═══════════════════════       ═══════════════════════
references.hpp                references/
  GenerateCw()                  SignalReferences.cw()
  GenerateLfm()                 SignalReferences.lfm()
  GenerateFormSignal()          SignalReferences.form_signal()
  GenerateNoise()               SignalReferences.noise()
  CpuMean/Median/Variance()    StatisticsReferences.mean/median/std()
  FindPeakBin()                 FftReferences.peak_freq()

validators.hpp                validators/
  ValidationResult              result.py → ValidationResult
  MaxRelError()                 RelativeValidator
  AbsError()                    AbsoluteValidator
  CheckPeakFreq()               FrequencyValidator

gpu_transfer.hpp              (нет аналога — pybind11 автоматически)
  ReadClBuffer<T>()             —
  ReadHipBuffer<T>()            —
  ReadGpuBuffer<T>()            —

test_format.hpp               runner.py + reporters.py
  PrintTestHeader()             TestRunner.run() + ConsoleReporter
  PrintValidation()             ConsoleReporter.on_passed/on_failed
  PrintTestSummary()            TestRunner.print_summary()
```

---

## ⚡ Оценка эффекта

| Утилита | Файлов затронет | LOC экономия | Риск | Приоритет |
|---------|----------------|-------------|------|-----------|
| **validators.hpp** | 80+ | ~400 | 🟢 Низкий | 🔴 Первый |
| **references.hpp** | 8+ | ~200 | 🟡 Средний | 🟠 Второй |
| **gpu_transfer.hpp** | 40+ | ~300 | 🟠 Средний | 🟡 Третий |
| **test_format.hpp** | 80+ | ~100 | 🟢 Низкий | 🔵 Четвёртый |
| **Итого** | — | **~1000** | — | — |

---

## 🚀 Стратегия внедрения

```
Этап 0: Создать include/test_utils/ (4 файла) — НЕ трогать существующие тесты
Этап 1: Новые тесты пишутся с test_utils — проверяем на практике
Этап 2: При правке старых тестов — мигрировать на test_utils (по одному)
Этап 3: (опционально) Массовая миграция через sed/script
```

**Принцип**: НЕ переписываем 83 файла разом. Создаём инструменты → новый код их использует → старый мигрируется при удобном случае.

---

*Предложение: Кодо | 2026-03-21*
