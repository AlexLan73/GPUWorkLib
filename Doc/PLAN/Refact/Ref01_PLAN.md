# Ref01: Рефакторинг fft_maxima — План реализации

## Цель

Перенос структур SpectrumParams, MaxValue, SpectrumResult, ProfilingData в `modules/fft_maxima/include/interface/`. Обновление cpu_fft_reference.hpp на использование общих типов. Замена цепочки в тесте: сравнение CPU vs GPU.

---

## Шаг 1: Создать spectrum_maxima_types.h

**Файл:** `modules/fft_maxima/include/interface/spectrum_maxima_types.h`

Перенести из spectrum_maxima_finder.h (строки 38–86) в `namespace antenna_fft`:
- `struct SpectrumParams`
- `struct MaxValue`
- `struct SpectrumResult`
- `struct ProfilingData`

Зависимости: `<cstdint>`, `<vector>`, `<complex>` (если нужны в будущем).

---

## Шаг 2: Обновить spectrum_maxima_finder.h

- Удалить определения SpectrumParams, MaxValue, SpectrumResult, ProfilingData (строки 30–86)
- Добавить `#include "interface/spectrum_maxima_types.h"` после существующих includes
- Оставить только класс SpectrumMaximaFinder

---

## Шаг 3: Обновить cpu_fft_reference.hpp

### 3.1 Include
```cpp
#include "interface/spectrum_maxima_types.h"
```
Вместо antenna_fft_params.h

### 3.2 CPUSpectrumResult — новая структура
```cpp
struct CPUSpectrumResult {
    SpectrumResult SpectrMax_left;   // максимум левого диапазона
    SpectrumResult SpectrMax_right;  // максимум правого диапазона
};
```
- antenna_id в SpectrMax_left и SpectrMax_right указывать как номер луча

### 3.3 FindMaximumWithInterpolation
- Возвращать CPUSpectrumResult (с двумя SpectrumResult)
- Заполнять SpectrMax_left: antenna_id, interpolated (left), left_point, center_point, right_point
- Заполнять SpectrMax_right: antenna_id, interpolated (right), left_point, center_point, right_point
- Использовать MaxValue с pad=0 при инициализации

### 3.4 Вспомогательная функция ProcessSingleBeamCore
Выделить цепочку: padding → ComputeFFT_CPU → FindMaximumWithInterpolation
Сигнатура: принимает beam_data, n_point, nFFT, search_range, sample_rate, antenna_id. Возвращает CPUSpectrumResult.

### 3.5 NewProcessAllBeams_CPU
```cpp
std::map<int, CPUSpectrumResult> NewProcessAllBeams_CPU(
    const std::vector<std::complex<float>>& data,
    const SpectrumParams& params);
```
Алгоритм для каждого луча i:
1. vi = data[n_point*i .. n_point*(i+1)-1]
2. base_fft = nextPow2(n_point), nFFT = base_fft * repeat_count
3. Padding: vi → base_fft нулями, затем repeat_count раз (итого nFFT точек)
4. Вызвать ProcessSingleBeamCore(vi, n_point, nFFT, search_range, sample_rate, i)
5. Результат в map[i]

search_range: если 0, использовать nFFT/4 (как на GPU).

### 3.6 Удалить/заменить
- Удалить CPUMaxValue
- Удалить старые ProcessAllBeams_CPU и ProcessSingleBeam_CPU (или оставить как deprecated, вызывающие NewProcessAllBeams_CPU)

---

## Шаг 4: Обновить test_spectrum_maxima.hpp

### 4.1 Вызов CPU
```cpp
auto cpu_results = cpu_reference::NewProcessAllBeams_CPU(input_data, params);
```
params получаем после finder.Initialize() (с вычисленным nFFT, search_range).

### 4.2 ValidateResults — добавить сравнение CPU vs GPU
- GPU: results[i] — 1 SpectrumResult (главный пик из left/right)
- CPU: cpu_results[i] — CPUSpectrumResult { SpectrMax_left, SpectrMax_right }
- Сравнение: главный пик GPU vs max(CPU_left, CPU_right) по magnitude
- Или: сравнить refined_frequency, index с допустимой погрешностью

### 4.3 Параметры
Убедиться что params.search_range заполнен (finder.GetParams() или nFFT/4 если 0).

---

## Шаг 5: CMakeLists.txt

Добавить в HEADERS:
```
include/interface/spectrum_maxima_types.h
```

---

## Шаг 6: Порядок выполнения

1. Создать spectrum_maxima_types.h
2. Обновить spectrum_maxima_finder.h
3. Обновить cpu_fft_reference.hpp (CPUSpectrumResult, FindMaximumWithInterpolation, NewProcessAllBeams_CPU)
4. Обновить test_spectrum_maxima.hpp
5. Обновить CMakeLists.txt
6. Сборка и тесты

---

## Уточнения (из диалога)

1. antenna_id — указывать номер луча в SpectrMax_left и SpectrMax_right
2. nFFT — вычисляется внутри: nextPow2(n_point) * repeat_count
3. Заменить всю цепочку в тесте — использовать NewProcessAllBeams_CPU, сравнивать CPU и GPU
