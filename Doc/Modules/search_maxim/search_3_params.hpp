#pragma once

/**
 * @file antenna_params.hpp
 * @brief Параметры для Antenna FFT модуля
 */

#include <cstddef>
#include <string>

namespace drv_gpu_lib {
namespace antenna {

/**
 * @struct AntennaParams
 * @brief Параметры обработки FFT для антенной системы
 */
struct AntennaParams {
    size_t beam_count;            ///< Количество лучей (beams)
    size_t count_points;          ///< Количество точек на луч (входные данные)
    size_t out_count_points_fft;  ///< Количество выходных точек FFT для анализа
    size_t max_peaks_count;       ///< Максимальное количество пиков для поиска
    
    std::string task_id;          ///< ID задачи (опционально)
    std::string module_name;      ///< Имя модуля (опционально)
    
    /**
     * @brief Конструктор по умолчанию
     */
    AntennaParams()
        : beam_count(0)
        , count_points(0)
        , out_count_points_fft(0)
        , max_peaks_count(1)
        , task_id("antenna")
        , module_name("antenna_module")
    {}
    
    /**
     * @brief Конструктор с параметрами
     */
    AntennaParams(size_t beams, size_t points, size_t out_points, size_t max_peaks,
                  const std::string& task = "antenna", const std::string& module = "antenna_module")
        : beam_count(beams)
        , count_points(points)
        , out_count_points_fft(out_points)
        , max_peaks_count(max_peaks)
        , task_id(task)
        , module_name(module)
    {}
    
    /**
     * @brief Проверить валидность параметров
     */
    bool IsValid() const {
        return beam_count > 0 &&
               count_points > 0 &&
               out_count_points_fft > 0 &&
               max_peaks_count > 0;
    }
};

/**
 * @struct BatchConfig
 * @brief Конфигурация batch processing
 */
struct BatchConfig {
    double memory_usage_limit = 0.65;   ///< Порог использования памяти (65%)
    double batch_size_ratio = 0.22;     ///< Размер батча относительно total beams (22%)
    size_t min_beams_for_batch = 10;    ///< Минимум лучей для включения batch режима
    
    BatchConfig() = default;
};

} // namespace antenna
} // namespace drv_gpu_lib
