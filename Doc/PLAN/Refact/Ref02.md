# Ref02: GPU post_kernel — вывод 8 MaxValue на луч (левый + правый)

## Цель

Привести реализацию OpenCL post_kernel к плану из комментариев: вывод **8** структур MaxValue на луч (4 для левого диапазона + 4 для правого). Сравнение CPU vs GPU — 8 на 8.

---

## Текущее состояние

- **Комментарии** в [fft_kernel_sources.hpp](modules/fft_maxima/include/kernels/fft_kernel_sources.hpp): описывают 8 MaxValue на луч
- **Строка 110**: `maxima_output - выходной массив (beam_count * 4 структуры MaxValue)` — **ошибка, должно быть 8**
- **Код kernel**: выводит 4 MaxValue (один пик — главный из left/right)
- **CPU** (Ref01): выводит CPUSpectrumResult = 8 MaxValue (SpectrMax_left + SpectrMax_right)
- **Тест**: сравнивает GPU (1 пик) с max(CPU_left, CPU_right)

---

## Задачи Ref02

### 1. Исправить комментарий (строка 110)

```
// Было:
//   maxima_output - выходной массив (beam_count * 4 структуры MaxValue)

// Должно быть:
//   maxima_output - выходной массив (beam_count * 8 структуры MaxValue)
```

Аналогично: комментарий в сигнатуре kernel (строка 209) — должно быть `beam_count * 8`.

---

### 2. Реализовать kernel: вывод 8 MaxValue на луч

**Алгоритм:**

1. Искать максимум **отдельно** в левом диапазоне [0, half_range]
2. Искать максимум **отдельно** в правом диапазоне [nFFT - half_range, nFFT - 1]
3. Для каждого из двух пиков:
   - Взять 3 точки: [max_idx-1, max_idx, max_idx+1]
   - Параболическая интерполяция
   - Записать 4 MaxValue: interpolated, left_point, center_point, right_point
4. Выход: `out_base = beam_idx * 8`
   - [0..3] — левый диапазон
   - [4..7] — правый диапазон

**Ссылка:** [fft_kernel_sources.hpp](modules/fft_maxima/include/kernels/fft_kernel_sources.hpp) GetPostKernelSource()

---

### 3. Обновить spectrum_maxima_finder

- **Буфер**: `antenna_count * 8 * sizeof(MaxValue)` (было 4)
- **ReadResults**: читать 8 MaxValue на луч, формировать `SpectrumResult` для left и right
- **Возвращаемый тип**: `std::vector<CPUSpectrumResult>` или `std::vector<std::pair<SpectrumResult, SpectrumResult>>` — либо расширить `SpectrumResult` до пары left+right

Вариант: `Process()` возвращает `std::vector<CPUSpectrumResult>` (как CPU), тогда GPU и CPU выдают один и тот же формат.
!!!!
посмотри модет проще будет сразу представить в виде 
std::map<int, CPUSpectrumResult>  
для валидации

---

### 4. Обновить тест: сравнение 8 на 8

- GPU: 8 MaxValue на луч (SpectrMax_left + SpectrMax_right)
- CPU: 8 MaxValue на луч (CPUSpectrumResult)
- Сравнивать: left с left, right с right (или оба набора целиком)

---

## Порядок выполнения

1. Исправить комментарии (4 → 8) в fft_kernel_sources.hpp
2. Переписать post_kernel: два независимых поиска (left, right), вывод 8 MaxValue
3. Обновить spectrum_maxima_finder.cpp: буфер x8, парсинг 8→CPUSpectrumResult
4. Обновить spectrum_maxima_finder.h: изменить возвращаемый тип Process() при необходимости
5. Обновить test_spectrum_maxima.hpp: сравнение 8 на 8
6. Сборка и тесты

---

## Связь с Ref01

Ref01: CPU выдаёт CPUSpectrumResult (8 MaxValue).  
Ref02: GPU kernel выдаёт тот же формат (8 MaxValue) для прямого сравнения.
