#pragma once

#include <cstddef>
#include <cmath>
#include <vector>
#include <map>

// Структура для параметров синусоиды
struct SinusoidParameter {
    float amplitude;    // Амплитуда
    float period;       // Период в точках (количество точек на один период)
    float phase_deg;    // Фаза в градусах
    
    SinusoidParameter(float amp = 1.0f, float period_points = 100.0f, float phase = 0.0f)
        : amplitude(amp), period(period_points), phase_deg(phase) {}
};

// Тип для маппинга: номер луча → вектор параметров синусоиды
typedef std::map<int, std::vector<SinusoidParameter>> RaySinusoidMap;

// Структура для параметров генерации синусоид
struct SinusoidGenParams {
    size_t num_rays;      // Количество лучей/антенн
    size_t count_points;  // Количество точек на антенну
    
    SinusoidGenParams(size_t rays = 0, size_t points = 0)
        : num_rays(rays), count_points(points) {}
};

struct LFMParameters {
  float f_start = 100.0f;              // Начальная частота (Гц)
  float f_stop = 500.0f;               // Конечная частота (Гц)
  float sample_rate = 12.0e6f;         // Частота дискретизации (12 МГц)
  mutable float duration = 0.0f;       // Длительность сигнала (сек)
  size_t num_beams = 256;              // Количество лучей
  float steering_angle = 30.0f;        // Базовый угол (градусы)

  // 🆕 НОВЫЕ ПОЛЯ для задержки с шагом угла:
  float angle_step_deg = 0.5f;         // Шаг по углу (градусы) - СТАНДАРТ 0.5°
  float angle_start_deg = 0.0f;      // Начальный угол (градусы)  - 64.0f
  float angle_stop_deg = 0.0f;        // Конечный угол (градусы)   65.0f
  mutable size_t count_points = 1024*8;  // Количество точек (отсчётов) на луч

    // ДЛЯ ГЕТЕРОДИНА:
  bool apply_heterodyne = false;       // Применять ли сопряжение

    // ВАЛИДАЦИЯ (обновлена)
  bool IsValid() const noexcept {
    if(count_points > 0) {
      duration = static_cast<float>(count_points) / static_cast<float>(sample_rate);
      // Если задано count_points, то duration игнорируется
      return f_start > 0.0f && f_stop > f_start &&
              sample_rate > 2.0f * f_stop &&
              count_points > 0 && num_beams > 0 &&
              angle_step_deg > 0.0f;
      }
      
      if(duration > 0.0f) {
        count_points = static_cast<size_t>(duration * sample_rate);
        // Если задано duration, то count_points игнорируется
        return f_start > 0.0f && f_stop > f_start &&
              sample_rate > 2.0f * f_stop &&
              duration > 0.0f && num_beams > 0 &&
              angle_step_deg > 0.0f;
      }

      return count_points > 0 && duration > 0.0f &&
              f_start > 0.0f && f_stop > f_start &&
              sample_rate > 2.0f * f_stop &&
              duration > 0.0f && num_beams > 0 &&
              angle_step_deg > 0.0f;
    }

    float GetChirpRate() const noexcept { 
        return (f_stop - f_start) / duration;
    }

    size_t GetNumSamples() const noexcept {
        return static_cast<size_t>(duration * sample_rate);
    }

    float GetWavelength() const noexcept {
        float f_center = (f_start + f_stop) / 2.0f;
        return SPEED_OF_LIGHT / f_center;
    }

    void SetAngle(float angle_start = 0.0f, float angle_stop = 0.0f){
      if(((angle_start == 0.0f & angle_stop == 0.0f))
        | (angle_start_deg == 0.0f & angle_stop_deg == 0.0f)){
          float x =  std::round((static_cast<float>(num_beams) / 4.0f) * 2.0f) / 2.0f;
          angle_start_deg = -x;      // Начальный угол (градусы)  - 64.0f
          angle_stop_deg = x;
          return;
      } 
      angle_start_deg = angle_start; 
      angle_stop_deg = angle_stop;
    }
    private:
    float SPEED_OF_LIGHT = 3.0e8f;    
};
