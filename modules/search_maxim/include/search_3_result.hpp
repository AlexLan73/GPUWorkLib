#pragma once

/**
 * @file antenna_result.hpp
 * @brief Структуры результатов Antenna FFT обработки
 */

#include <vector>
#include <cstddef>
#include <complex>

namespace drv_gpu_lib {
namespace search_3_ {

/**
 * @struct FFTMaxValue
 * @brief Найденное максимальное значение в спектре
 */
struct FFTMaxValue {
    size_t index_point;        ///< Индекс точки в спектре
    float amplitude;           ///< Амплитуда (magnitude)
    float phase;               ///< Фаза в радианах
    float real;                ///< Вещественная часть
    float imag;                ///< Мнимая часть
    
    FFTMaxValue()
        : index_point(0), amplitude(0.0f), phase(0.0f), real(0.0f), imag(0.0f)
    {}
};

/**
 * @struct BeamFFTResult
 * @brief Результат FFT обработки для одного луча
 */
struct BeamFFTResult {
    std::vector<FFTMaxValue> max_values;  ///< Найденные максимумы (топ N)
    float freq_offset;                    ///< Смещение частоты (параболическая интерполяция)
    float refined_frequency;              ///< Уточнённая частота в Гц
    
    BeamFFTResult()
        : freq_offset(0.0f), refined_frequency(0.0f)
    {}
};

/**
 * @struct Search3FFTResult
 * @brief Полный результат FFT обработки для всех лучей
 */
struct Search3FFTResult {
    std::vector<BeamFFTResult> results;  ///< Результаты для каждого луча
    size_t nFFT;                         ///< Использованный размер FFT
    std::string task_id;                 ///< ID задачи
    std::string module_name;             ///< Имя модуля
    
    Search3FFTResult()
        : nFFT(0), task_id(""), module_name("")
    {}
    
    Search3FFTResult(size_t beam_count, size_t nfft, 
                     const std::string& task, const std::string& module)
        : nFFT(nfft), task_id(task), module_name(module)
    {
        results.reserve(beam_count);
    }
};

} // namespace search_3_
} // namespace drv_gpu_lib
