#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <complex>

/**
 * @file antenna_fft_params.h
 * @brief Структуры данных для AntennaFFTProcMax
 * 
 * Все структуры содержат признаки задачи (task_id, module_name) для масштабируемости
 * при использовании множества модулей обработки.
 */

namespace antenna_fft {

/**
 * @struct FFTMaxResult
 * @brief Результат одного максимального значения после FFT
 */
struct FFTMaxResult {
    size_t index_point;    // Индекс точки в спектре (после fftshift)
    float real;            // Вещественная часть комплексного значения
    float imag;            // Мнимая часть комплексного значения
    float amplitude;       // Амплитуда (magnitude)
    float phase;           // Фаза в градусах
    
    FFTMaxResult() : index_point(0), real(0.0f), imag(0.0f), amplitude(0.0f), phase(0.0f) {}
    FFTMaxResult(size_t idx, float re, float im, float amp, float ph) 
        : index_point(idx), real(re), imag(im), amplitude(amp), phase(ph) {}
};

/**
 * @struct FFTResult
 * @brief Результат FFT для одного луча/антенны
 */
struct FFTResult {
    size_t v_fft;                              // Количество точек в FFT (out_count_points_fft)
    std::vector<FFTMaxResult> max_values;     // Вектор из 3-5 максимальных значений
    
    // Результаты параболической интерполяции (для главного максимума)
    float freq_offset;                         // Смещение частоты в долях бина (-0.5..+0.5)
    float refined_frequency;                   // Уточнённая частота в Гц (рассчитанная)
    
    // Признаки задачи для масштабируемости
    std::string task_id;                       // Идентификатор задачи
    std::string module_name;                   // Имя модуля
    
    FFTResult() : v_fft(0), freq_offset(0.0f), refined_frequency(0.0f) {}
    FFTResult(size_t fft_size, const std::string& task = "", const std::string& module = "")
        : v_fft(fft_size), freq_offset(0.0f), refined_frequency(0.0f), 
          task_id(task), module_name(module) {}
};

/**
 * @struct AntennaFFTParams
 * @brief Входные параметры для AntennaFFTProcMax
 */
struct AntennaFFTParams {
    size_t beam_count;         // Количество лучей/антенн
    size_t count_points;       // Количество точек в луче (входные данные)
    size_t out_count_points_fft; // Количество точек в FFT для вывода
    size_t max_peaks_count;    // Количество максимальных значений (3-5, по умолчанию 3)
    
    // Признаки задачи для масштабируемости
    std::string task_id;       // Идентификатор задачи
    std::string module_name;   // Имя модуля
    
    AntennaFFTParams() 
        : beam_count(0), count_points(0), out_count_points_fft(0), max_peaks_count(3) {}
    
    AntennaFFTParams(size_t beams, size_t points, size_t out_fft, size_t max_peaks = 3,
                     const std::string& task = "", const std::string& module = "")
        : beam_count(beams), count_points(points), out_count_points_fft(out_fft),
          max_peaks_count(max_peaks), task_id(task), module_name(module) {}
    
    bool IsValid() const noexcept {
        return beam_count > 0 && count_points > 0 && out_count_points_fft > 0 &&
               max_peaks_count >= 3 && max_peaks_count <= 5;
    }
};

/**
 * @struct AntennaFFTResult
 * @brief Полный результат FFT обработки для всех лучей
 */
struct AntennaFFTResult {
    std::vector<FFTResult> results;  // Вектор результатов для каждого луча
    
    // Признаки задачи для масштабируемости
    std::string task_id;              // Идентификатор задачи
    std::string module_name;           // Имя модуля
    
    // Метаданные
    size_t total_beams;               // Общее количество обработанных лучей
    size_t nFFT;                      // Размер FFT (вычисленный)
    
    AntennaFFTResult() : total_beams(0), nFFT(0) {}
    
    AntennaFFTResult(size_t beams, size_t fft_size, 
                     const std::string& task = "", const std::string& module = "")
        : results(), task_id(task), module_name(module), total_beams(beams), nFFT(fft_size) {
        results.reserve(beams);
    }
};

/**
 * @struct FFTProfilingResults
 * @brief Результаты профилирования FFT операций
 */
struct FFTProfilingResults {
    double total_time_ms;
    double upload_time_ms;
    double pre_callback_time_ms;
    double fft_time_ms;
    double post_callback_time_ms;
    double reduction_time_ms;
    double download_time_ms;
    
    FFTProfilingResults() 
        : total_time_ms(0.0), upload_time_ms(0.0), pre_callback_time_ms(0.0),
          fft_time_ms(0.0), post_callback_time_ms(0.0), reduction_time_ms(0.0),
          download_time_ms(0.0) {}
};

} // namespace antenna_fft

