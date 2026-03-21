# TASK_CppTest_03 — references/ (CPU-эталоны)

> **Фаза**: 0 (инфраструктура)
> **Зависимости**: TASK_CppTest_01 (test_configs.hpp → SignalParams)
> **Статус**: ⬜ TODO
> **Оценка**: ~1.5 часа
> **Паттерны**: Information Expert (GRASP), DRY, SRP (SOLID)

---

## 🎯 Цель

Заменить 8+ копий `GenerateCw()`, `CpuMean()`, `FindPeakBin()` единым набором CPU-эталонов.

Зеркало Python: `common/references/` (signal_refs.py, statistics_refs.py, fft_refs.py)

---

## 📁 Создаваемые файлы (3 штуки)

```
modules/test_utils/references/
├── signal_refs.hpp      ← 1. GenerateCw, GenerateLfm, FormSignal, Noise, MultiBeam
├── statistics_refs.hpp  ← 2. CpuMean, CpuMedian, CpuVariance, CpuStd (template)
└── fft_refs.hpp         ← 3. FindPeakBin, PeakFreqHz, CpuMagnitude, FreqAxis
```

---

## 📝 Детальное ТЗ

### 1. `modules/test_utils/references/signal_refs.hpp`

```cpp
#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace gpu_test_utils {
namespace refs {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * CW сигнал (непрерывная синусоида).
 * Зеркало Python: SignalReferences.cw()
 *
 * @return vector<complex<float>>, size = n_samples
 */
inline std::vector<std::complex<float>>
GenerateCw(float fs, size_t n_samples, float f0,
           float amplitude = 1.0f, float phase = 0.0f)
{
    std::vector<std::complex<float>> sig(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(fs);
        double ph = 2.0 * M_PI * static_cast<double>(f0) * t + static_cast<double>(phase);
        sig[i] = std::complex<float>(
            amplitude * static_cast<float>(std::cos(ph)),
            amplitude * static_cast<float>(std::sin(ph))
        );
    }
    return sig;
}

/**
 * ЛЧМ сигнал (линейная частотная модуляция).
 * Зеркало Python: SignalReferences.lfm()
 */
inline std::vector<std::complex<float>>
GenerateLfm(float fs, size_t n_samples, float f_start, float f_end,
            float amplitude = 1.0f, float phase = 0.0f)
{
    std::vector<std::complex<float>> sig(n_samples);
    double duration = static_cast<double>(n_samples) / static_cast<double>(fs);
    double rate = (static_cast<double>(f_end) - f_start) / duration;
    for (size_t i = 0; i < n_samples; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(fs);
        double ph = 2.0 * M_PI * (f_start * t + 0.5 * rate * t * t) + phase;
        sig[i] = std::complex<float>(
            amplitude * static_cast<float>(std::cos(ph)),
            amplitude * static_cast<float>(std::sin(ph))
        );
    }
    return sig;
}

/**
 * ЛЧМ с задержкой (для тестов гетеродина/дечирпа).
 * Зеркало Python: SignalReferences.lfm_with_delay()
 *
 * Сигнал = 0 при t < delay_s, потом ЛЧМ от t = delay_s.
 */
inline std::vector<std::complex<float>>
GenerateDelayedLfm(float fs, size_t n_samples, float f_start, float f_end,
                   float delay_s, float amplitude = 1.0f)
{
    std::vector<std::complex<float>> sig(n_samples, {0.0f, 0.0f});
    double duration = static_cast<double>(n_samples) / static_cast<double>(fs);
    double rate = (static_cast<double>(f_end) - f_start) / duration;
    for (size_t i = 0; i < n_samples; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(fs);
        if (t < delay_s) continue;
        double t_local = t - delay_s;
        double ph = 2.0 * M_PI * (f_start * t_local + 0.5 * rate * t_local * t_local);
        sig[i] = std::complex<float>(
            amplitude * static_cast<float>(std::cos(ph)),
            amplitude * static_cast<float>(std::sin(ph))
        );
    }
    return sig;
}

/**
 * Несколько ЛЧМ с разными задержками (массив антенн).
 * Зеркало Python: SignalReferences.lfm_multi_antenna()
 *
 * @param delays_s  задержки по каждой антенне
 * @return vector<complex<float>>, size = n_antennas * n_samples (row-major)
 */
inline std::vector<std::complex<float>>
GenerateMultiAntennaLfm(float fs, size_t n_samples, float f_start, float f_end,
                         const std::vector<float>& delays_s, float amplitude = 1.0f)
{
    size_t n_ant = delays_s.size();
    std::vector<std::complex<float>> result(n_ant * n_samples, {0.0f, 0.0f});
    for (size_t a = 0; a < n_ant; ++a) {
        auto row = GenerateDelayedLfm(fs, n_samples, f_start, f_end, delays_s[a], amplitude);
        std::copy(row.begin(), row.end(), result.begin() + a * n_samples);
    }
    return result;
}

/**
 * FormSignal CPU reference (getX без шума).
 * Зеркало Python: SignalReferences.form_signal()
 * Воспроизводит GPU FormSignalGenerator: окно + центрированная фаза.
 */
inline std::vector<std::complex<float>>
GenerateFormSignal(float fs, size_t points, float f0, float amplitude,
                   float phase, float fdev, float norm_val, float tau = 0.0f)
{
    double dt = 1.0 / static_cast<double>(fs);
    double ti = static_cast<double>(points) * dt;
    std::vector<std::complex<float>> result(points, {0.0f, 0.0f});
    for (size_t i = 0; i < points; ++i) {
        double t = static_cast<double>(i) * dt + static_cast<double>(tau);
        if (t < 0.0 || t > ti - dt) continue;
        double t_centered = t - ti / 2.0;
        double ph = 2.0 * M_PI * f0 * t + M_PI * fdev / ti * (t_centered * t_centered) + phase;
        float a = amplitude * norm_val;
        result[i] = std::complex<float>(a * static_cast<float>(std::cos(ph)),
                                         a * static_cast<float>(std::sin(ph)));
    }
    return result;
}

/**
 * Синусоида (для тестов статистики). Одноканальная.
 * Заменяет GenerateSinusoid() из test_statistics_rocm.hpp
 */
inline std::vector<std::complex<float>>
GenerateSinusoid(float freq, float sample_rate, size_t n_point,
                 float amplitude = 1.0f)
{
    return GenerateCw(sample_rate, n_point, freq, amplitude);
}

/**
 * Многоканальный сигнал (для multi-beam тестов статистики).
 * Каждый канал = синусоида с amplitude = amp_base + i * amp_step.
 *
 * @return vector size = n_beams * n_point (row-major)
 */
inline std::vector<std::complex<float>>
GenerateMultiBeam(size_t n_beams, size_t n_point, float fs, float freq,
                  float amp_base = 1.0f, float amp_step = 0.5f)
{
    std::vector<std::complex<float>> result(n_beams * n_point);
    for (size_t b = 0; b < n_beams; ++b) {
        float amp = amp_base + static_cast<float>(b) * amp_step;
        auto beam = GenerateCw(fs, n_point, freq, amp);
        std::copy(beam.begin(), beam.end(), result.begin() + b * n_point);
    }
    return result;
}

/**
 * Постоянное значение (edge case тесты).
 */
inline std::vector<std::complex<float>>
GenerateConstant(std::complex<float> value, size_t n_point) {
    return std::vector<std::complex<float>>(n_point, value);
}

/**
 * Гауссов шум (CPU reference, воспроизводимый через seed).
 *
 * ⚠️ GPU (Philox) и C++ std::mt19937 дают РАЗНЫЕ числа при одном seed!
 * Для валидации GPU statistics: GPU генерирует → копия на CPU → сравниваем stats.
 */
inline std::vector<std::complex<float>>
GenerateNoise(size_t n_samples, float amplitude = 1.0f, uint32_t seed = 42)
{
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::complex<float>> sig(n_samples);
    float scale = amplitude / std::sqrt(2.0f);
    for (size_t i = 0; i < n_samples; ++i)
        sig[i] = std::complex<float>(dist(gen) * scale, dist(gen) * scale);
    return sig;
}

/**
 * Дечирп: s_dc = s_rx * conj(s_ref). CPU-эталон.
 */
inline std::vector<std::complex<float>>
CpuDechirp(const std::complex<float>* s_rx,
            const std::complex<float>* s_ref, size_t n)
{
    std::vector<std::complex<float>> result(n);
    for (size_t i = 0; i < n; ++i)
        result[i] = s_rx[i] * std::conj(s_ref[i]);
    return result;
}

/**
 * Тестовый сигнал для фильтров (CW f_low + CW f_high).
 * Заменяет GenerateTestSignal() из test_fir_basic.hpp.
 */
inline std::vector<std::complex<float>>
GenerateComposite(float fs, size_t n_samples,
                  float f_low, float f_high,
                  float amp_low = 1.0f, float amp_high = 0.5f)
{
    auto s1 = GenerateCw(fs, n_samples, f_low, amp_low);
    auto s2 = GenerateCw(fs, n_samples, f_high, amp_high);
    for (size_t i = 0; i < n_samples; ++i)
        s1[i] += s2[i];
    return s1;
}

} // namespace refs
} // namespace gpu_test_utils
```

---

### 2. `modules/test_utils/references/statistics_refs.hpp`

```cpp
#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstddef>

namespace gpu_test_utils {
namespace refs {

/**
 * CPU-эталоны для статистики. Шаблонные.
 *
 * Сценарий валидации GPU statistics:
 *   1. GPU генерирует данные → копия на CPU
 *   2. GPU StatisticsProcessor считает stats
 *   3. Эти функции считают stats из ТЕХ ЖЕ данных
 *   4. Сравниваем через MaxRelError / ScalarRelError
 */

/// Среднее (complex → mean of values)
template<typename T>
inline T CpuMean(const T* data, size_t n) {
    double sum_r = 0.0, sum_i = 0.0;
    // fallback for non-complex
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += static_cast<double>(data[i]);
    return static_cast<T>(sum / static_cast<double>(n));
}

/// Среднее для complex<float>
template<>
inline std::complex<float> CpuMean(const std::complex<float>* data, size_t n) {
    double sum_r = 0.0, sum_i = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_r += data[i].real();
        sum_i += data[i].imag();
    }
    auto dn = static_cast<double>(n);
    return {static_cast<float>(sum_r / dn), static_cast<float>(sum_i / dn)};
}

/// Среднее по амплитуде |x|
inline float CpuMeanMagnitude(const std::complex<float>* data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += static_cast<double>(std::abs(data[i]));
    return static_cast<float>(sum / static_cast<double>(n));
}

/// Медиана по амплитуде |x|
inline float CpuMedianMagnitude(const std::complex<float>* data, size_t n) {
    std::vector<float> mags(n);
    for (size_t i = 0; i < n; ++i)
        mags[i] = std::abs(data[i]);
    std::sort(mags.begin(), mags.end());
    if (n % 2 == 0)
        return (mags[n/2 - 1] + mags[n/2]) / 2.0f;
    return mags[n/2];
}

/// Дисперсия по амплитуде (population variance)
inline float CpuVarianceMagnitude(const std::complex<float>* data, size_t n) {
    float mean = CpuMeanMagnitude(data, n);
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(std::abs(data[i])) - mean;
        sum_sq += diff * diff;
    }
    return static_cast<float>(sum_sq / static_cast<double>(n));
}

/// Стандартное отклонение по амплитуде
inline float CpuStdMagnitude(const std::complex<float>* data, size_t n) {
    return std::sqrt(CpuVarianceMagnitude(data, n));
}

/// Среднее для float массива
inline float CpuMeanFloat(const float* data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += data[i];
    return static_cast<float>(sum / static_cast<double>(n));
}

/// Медиана для float массива
inline float CpuMedianFloat(const float* data, size_t n) {
    std::vector<float> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());
    if (n % 2 == 0)
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0f;
    return sorted[n/2];
}

/// Дисперсия для float массива (population)
inline float CpuVarianceFloat(const float* data, size_t n) {
    float mean = CpuMeanFloat(data, n);
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = static_cast<double>(data[i]) - mean;
        sum_sq += d * d;
    }
    return static_cast<float>(sum_sq / static_cast<double>(n));
}

/// Стандартное отклонение для float массива
inline float CpuStdFloat(const float* data, size_t n) {
    return std::sqrt(CpuVarianceFloat(data, n));
}

} // namespace refs
} // namespace gpu_test_utils
```

---

### 3. `modules/test_utils/references/fft_refs.hpp`

```cpp
#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace gpu_test_utils {
namespace refs {

/**
 * Найти индекс максимального бина в спектре (первая половина).
 * Заменяет FindPeakBin() из test_signal_generators.hpp.
 */
inline size_t FindPeakBin(const float* magnitude, size_t n_bins,
                           size_t search_range = 0)
{
    size_t range = (search_range > 0) ? search_range : n_bins / 2;
    range = std::min(range, n_bins);
    size_t peak = 0;
    float peak_val = 0.0f;
    for (size_t i = 0; i < range; ++i) {
        if (magnitude[i] > peak_val) {
            peak_val = magnitude[i];
            peak = i;
        }
    }
    return peak;
}

/**
 * FindPeakBin из complex спектра (автоматический |x|).
 */
inline size_t FindPeakBinComplex(const std::complex<float>* spectrum,
                                  size_t n_bins, size_t search_range = 0)
{
    std::vector<float> mag(n_bins);
    for (size_t i = 0; i < n_bins; ++i)
        mag[i] = std::abs(spectrum[i]);
    return FindPeakBin(mag.data(), n_bins, search_range);
}

/**
 * Частота пика спектра (Гц).
 * Зеркало Python: FftReferences.peak_freq()
 */
inline float PeakFreqHz(const float* magnitude, size_t n_bins, float fs,
                         size_t search_range = 0)
{
    size_t peak = FindPeakBin(magnitude, n_bins, search_range);
    return static_cast<float>(peak) * fs / static_cast<float>(n_bins);
}

/**
 * |x| — амплитудный спектр из complex.
 */
inline std::vector<float>
CpuMagnitude(const std::complex<float>* spectrum, size_t n_bins)
{
    std::vector<float> mag(n_bins);
    for (size_t i = 0; i < n_bins; ++i)
        mag[i] = std::abs(spectrum[i]);
    return mag;
}

/**
 * Ось частот (Гц) — аналог np.fft.fftfreq.
 */
inline std::vector<float> FreqAxis(size_t n_fft, float fs) {
    std::vector<float> freqs(n_fft);
    float df = fs / static_cast<float>(n_fft);
    for (size_t i = 0; i < n_fft; ++i) {
        if (i <= n_fft / 2)
            freqs[i] = static_cast<float>(i) * df;
        else
            freqs[i] = static_cast<float>(static_cast<int>(i) - static_cast<int>(n_fft)) * df;
    }
    return freqs;
}

} // namespace refs
} // namespace gpu_test_utils
```

---

## ✅ Критерии завершения

- [ ] Все 3 файла созданы в `modules/test_utils/references/`
- [ ] `refs::GenerateCw(12e6f, 4096, 2e6f)` → vector<complex<float>> size=4096
- [ ] `refs::GenerateLfm(12e6f, 4096, 0.f, 2e6f)` → корректный chirp
- [ ] `refs::GenerateDelayedLfm(...)` → нули до delay
- [ ] `refs::GenerateFormSignal(...)` → совпадает с getX_numpy из Python
- [ ] `refs::GenerateMultiBeam(4, 500000, 12e6f, 100.f)` → size = 4*500000
- [ ] `refs::CpuMeanMagnitude(data, n)` → корректное среднее
- [ ] `refs::CpuMedianMagnitude(data, n)` → корректная медиана
- [ ] `refs::FindPeakBin(mag, 4096)` → правильный индекс
- [ ] `refs::PeakFreqHz(mag, 4096, 12e6f)` ≈ f0 (±freq_resolution)
- [ ] Все вычисления в double где есть накопление (CpuMean, Variance)
- [ ] Компилируется: `g++ -std=c++17 -fsyntax-only references/signal_refs.hpp`

---

*Создан: 2026-03-21 | Кодо | Фаза 0*
