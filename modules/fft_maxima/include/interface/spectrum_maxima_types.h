#pragma once

/**
 * @file spectrum_maxima_types.h
 * @brief Общие структуры данных для SpectrumMaximaFinder и CPU reference
 *
 * Перенесено из spectrum_maxima_finder.h (Ref01)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-08
 */

#include <cstdint>

namespace antenna_fft {

/**
 * @enum PeakSearchMode
 * @brief Режим поиска пиков в спектре
 */
enum class PeakSearchMode {
    ONE_PEAK,   ///< Поиск ОДНОГО пика (сравнение левого и правого, выбор большего) → 4 MaxValue
    TWO_PEAKS   ///< Поиск ДВУХ пиков (независимо левый и правый) → 8 MaxValue
};

/**
 * @struct SpectrumParams
 * @brief Параметры для поиска максимума спектра
 */
struct SpectrumParams {
    uint32_t antenna_count = 5;         ///< Количество антен (1-256)
    uint32_t n_point = 1000;            ///< Точек на антену (исходный размер сигнала)
    uint32_t repeat_count = 2;          ///< Множитель размера FFT (2^n: 1,2,4,8...)
    float sample_rate = 1000.0f;        ///< Частота дискретизации (Гц)
    uint32_t search_range = 0;          ///< Диапазон поиска максимума (0 = авто = nFFT/4)
    PeakSearchMode peak_mode = PeakSearchMode::ONE_PEAK; ///< Режим поиска (ВЕЗДЕ ищем ОДИН пик!)

    // ═══════════════════════════════════════════════════════════════════════
    // Batch Processing (для больших данных, использует BatchManager)
    // ═══════════════════════════════════════════════════════════════════════
    float memory_limit = 0.80f;         ///< Доля СВОБОДНОЙ памяти GPU (0.0-1.0, по умолчанию 80%)
                                        ///< Теперь берётся от GetFreeMemorySize()!

    // Вычисляемые параметры (заполняются в Initialize)
    uint32_t nFFT = 0;                  ///< Размер FFT = nextPow2(n_point) * repeat_count
    uint32_t base_fft = 0;              ///< Базовый размер = nextPow2(n_point)
};

/**
 * @struct MaxValue
 * @brief Результат поиска максимума (должен совпадать с GPU структурой!)
 */
struct MaxValue {
    uint32_t index;             ///< Индекс в FFT спектре
    float real;                 ///< Re компонента
    float imag;                 ///< Im компонента
    float magnitude;            ///< |magnitude| = sqrt(re^2 + im^2)
    float phase;                ///< Фаза в градусах
    float freq_offset;          ///< Параболическая поправка [-0.5, 0.5]
    float refined_frequency;    ///< Уточнённая частота (Гц)
    uint32_t pad;               ///< Padding для выравнивания (32 bytes total)
};

/**
 * @struct SpectrumResult
 * @brief Результат обработки для одной антены (один пик: left или right)
 */
struct SpectrumResult {
    uint32_t antenna_id;        ///< Номер антены
    MaxValue interpolated;      ///< Результат параболической интерполяции
    MaxValue left_point;        ///< Левая точка (index-1)
    MaxValue center_point;      ///< Центральная точка (максимум)
    MaxValue right_point;       ///< Правая точка (index+1)
};

/**
 * @struct CPUSpectrumResult
 * @brief Результат для одного луча: два пика (левый и правый диапазон спектра)
 * Используется и CPU reference, и GPU SpectrumMaximaFinder (Ref02)
 */
struct CPUSpectrumResult {
    SpectrumResult SpectrMax_left;   ///< Максимум левого диапазона [0, half_range]
    SpectrumResult SpectrMax_right;  ///< Максимум правого диапазона [nFFT-half_range, nFFT]
};

/**
 * @struct ProfilingData
 * @brief Данные профилирования GPU
 */
struct ProfilingData {
    double upload_time_ms = 0.0;        ///< Время загрузки данных Host→GPU
    double fft_time_ms = 0.0;           ///< Время выполнения FFT (с pre-callback)
    double post_kernel_time_ms = 0.0;   ///< Время выполнения post-kernel
    double download_time_ms = 0.0;      ///< Время выгрузки результатов GPU→Host
    double total_time_ms = 0.0;         ///< Общее время
};

} // namespace antenna_fft
