# 🔧 Kernel Refactoring — OnePeak & TwoPeaks

> **Тема**: ТЕМА 3 (была, теперь ТЕМА 2 по приоритету)
> **Приоритет**: 🔥 Высокий (делаем СРАЗУ после анализа DrvGPU)
> **Статус**: 📋 Planned
> **Дата создания**: 2026-02-10
> **Автор**: Кодо (AI Assistant)

---

## 🎯 ЦЕЛЬ

Разделить поиск пиков на **два отдельных кернела**:
1. **OnePeak** — ищет **одну** вершину во всём диапазоне → выводит **4 MaxValue**
2. **TwoPeaks** — ищет **две** вершины (левый/правый диапазон) → выводит **8 MaxValue**

**ВАЖНО**: Alex напишет алгоритм для OnePeak в комментариях над кернелом!

---

## 🔥 ПРОБЛЕМА

**Текущее состояние**:
- Есть только `GetPostKernelSource()` — ищет ДВЕ вершины (левый/правый диапазон)
- Выводит **8 MaxValue** структур на каждый луч:
  ```
  [0-3] левый пик  (интерполяция + 3 точки)
  [4-7] правый пик (интерполяция + 3 точки)
  ```
- Нет возможности искать только ОДИН пик
- Название не отражает функциональность (не понятно, что ищет 2 пика)

---

## ✅ РЕШЕНИЕ

### 1️⃣ Переименовать существующий кернел

**До**:
```cpp
inline const char* GetPostKernelSource() { ... }
```

**После**:
```cpp
inline const char* GetPostKernelSource_TwoPeaks() {
    return R"CL(
// ════════════════════════════════════════════════════════════════════════════
// Post Kernel: TWO PEAKS (левый и правый диапазон)
// ════════════════════════════════════════════════════════════════════════════
// НАЗНАЧЕНИЕ:
//   Поиск ДВУХ независимых максимумов в краевых диапазонах спектра
//   (левый [0, half_range] и правый [nFFT-half_range, nFFT-1])
//
// АЛГОРИТМ:
//   1. Делим search_range пополам → half_range
//   2. Ищем максимум ОТДЕЛЬНО в левом и правом диапазонах
//   3. Для каждого пика: 3 точки [max_idx-1, max_idx, max_idx+1] + параболическая интерполяция
//
// ВЫХОД: 8 MaxValue структур на каждый луч
//   [0] - левый пик: интерполяция (freq_offset, refined_frequency)
//   [1] - левый пик: левая точка (index-1)
//   [2] - левый пик: центральная точка (главный максимум)
//   [3] - левый пик: правая точка (index+1)
//   [4] - правый пик: интерполяция
//   [5] - правый пик: левая точка
//   [6] - правый пик: центральная точка
//   [7] - правый пик: правая точка
// ════════════════════════════════════════════════════════════════════════════
... kernel code ...
)CL";
}
```

### 2️⃣ Создать новый кернел OnePeak

**Файл**: `modules/fft_maxima/include/kernels/fft_kernel_sources.hpp`

```cpp
inline const char* GetPostKernelSource_OnePeak() {
    return R"CL(
// ════════════════════════════════════════════════════════════════════════════
// Post Kernel: ONE PEAK (весь диапазон)
// ════════════════════════════════════════════════════════════════════════════
// НАЗНАЧЕНИЕ:
//   Поиск ОДНОГО максимума во ВСЁМ диапазоне [0, search_range]
//
// АЛГОРИТМ:
//   ┌──────────────────────────────────────────────────────────────────┐
//   │ ALEX НАПИШЕТ АЛГОРИТМ ЗДЕСЬ! 📝                                  │
//   │ Кодо проверит и учтёт пожелания!                                 │
//   └──────────────────────────────────────────────────────────────────┘
//
//   Предположительно:
//   1. Ищем максимум во всём диапазоне [0, search_range]
//   2. Берём 3 точки вокруг максимума [max_idx-1, max_idx, max_idx+1]
//   3. Параболическая интерполяция для уточнения частоты
//
// ВЫХОД: 4 MaxValue структуры на каждый луч
//   [0] - интерполяция (freq_offset, refined_frequency)
//   [1] - левая точка (index-1)
//   [2] - центральная точка (главный максимум)
//   [3] - правая точка (index+1)
//
// ПАРАМЕТРЫ: те же что и в TwoPeaks (совместимость)
// ════════════════════════════════════════════════════════════════════════════

__kernel void post_kernel_one_peak(
    __global const float2* fft_output,
    __global MaxValue* output,
    uint nFFT,
    uint beam_count,
    uint search_range,
    float sample_rate
) {
    uint beam_idx = get_global_id(0);
    if (beam_idx >= beam_count) return;

    __global const float2* beam_spectrum = fft_output + beam_idx * nFFT;
    __global MaxValue* beam_output = output + beam_idx * 4;  // 4 MaxValue!

    // ═══════════════════════════════════════════════════════════
    // ШАГ 1: Найти максимум во всём диапазоне [0, search_range]
    // ═══════════════════════════════════════════════════════════
    uint max_idx = 0;
    float max_magnitude = 0.0f;

    for (uint i = 0; i < search_range && i < nFFT; ++i) {
        float2 val = beam_spectrum[i];
        float magnitude = sqrt(val.x * val.x + val.y * val.y);

        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
            max_idx = i;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // ШАГ 2: Собрать 3 точки [max_idx-1, max_idx, max_idx+1]
    // ═══════════════════════════════════════════════════════════
    uint idx_left  = (max_idx > 0) ? max_idx - 1 : 0;
    uint idx_center = max_idx;
    uint idx_right = (max_idx + 1 < nFFT) ? max_idx + 1 : nFFT - 1;

    float2 val_left   = beam_spectrum[idx_left];
    float2 val_center = beam_spectrum[idx_center];
    float2 val_right  = beam_spectrum[idx_right];

    float mag_left   = sqrt(val_left.x * val_left.x + val_left.y * val_left.y);
    float mag_center = sqrt(val_center.x * val_center.x + val_center.y * val_center.y);
    float mag_right  = sqrt(val_right.x * val_right.x + val_right.y * val_right.y);

    // ═══════════════════════════════════════════════════════════
    // ШАГ 3: Параболическая интерполяция
    // ═══════════════════════════════════════════════════════════
    float denom = 2.0f * (2.0f * mag_center - mag_left - mag_right);
    float freq_offset = 0.0f;

    if (fabs(denom) > 1e-6f) {
        freq_offset = (mag_left - mag_right) / denom;
        // Ограничиваем offset: [-0.5, 0.5]
        freq_offset = clamp(freq_offset, -0.5f, 0.5f);
    }

    float refined_index = (float)idx_center + freq_offset;
    float bin_width = sample_rate / (float)nFFT;
    float refined_frequency = refined_index * bin_width;

    // ═══════════════════════════════════════════════════════════
    // ШАГ 4: Заполнить выходной массив (4 MaxValue)
    // ═══════════════════════════════════════════════════════════

    // [0] - Интерполированный результат
    beam_output[0].index = idx_center;
    beam_output[0].magnitude = mag_center;
    beam_output[0].freq_offset = freq_offset;
    beam_output[0].refined_frequency = refined_frequency;

    // [1] - Левая точка
    beam_output[1].index = idx_left;
    beam_output[1].magnitude = mag_left;
    beam_output[1].freq_offset = 0.0f;
    beam_output[1].refined_frequency = (float)idx_left * bin_width;

    // [2] - Центральная точка
    beam_output[2].index = idx_center;
    beam_output[2].magnitude = mag_center;
    beam_output[2].freq_offset = 0.0f;
    beam_output[2].refined_frequency = (float)idx_center * bin_width;

    // [3] - Правая точка
    beam_output[3].index = idx_right;
    beam_output[3].magnitude = mag_right;
    beam_output[3].freq_offset = 0.0f;
    beam_output[3].refined_frequency = (float)idx_right * bin_width;
}
)CL";
}
```

### 3️⃣ Обновить вызовы кернела

**Файлы для изменения**:
- `modules/fft_maxima/src/spectrum_maxima_finder.cpp`
- `modules/fft_maxima/src/antenna_fft_core.cpp`

**Пример**:
```cpp
// Выбор кернела по mode
const char* kernel_source;
size_t output_size_per_beam;

if (mode == PeakSearchMode::ONE_PEAK) {
    kernel_source = kernels::GetPostKernelSource_OnePeak();
    output_size_per_beam = 4;
} else {
    kernel_source = kernels::GetPostKernelSource_TwoPeaks();
    output_size_per_beam = 8;
}
```

---

## 🔧 ИСПРАВЛЕНИЕ: Амплитуда правой стороны

**Проблема** (из задачи Alex 3.5):
> "Исправить на уровне кернел значение пика в правой стороне привести его к значению как с лева"

**Что исправить**: В `GetPostKernelSource_TwoPeaks()` правая сторона спектра имеет неправильную амплитуду.

**Решение**: Будет реализовано в таске при работе над ТЕМОЙ 3.

---

## 📋 ЗАДАЧИ (создаются при начале работы)

При начале работы над ТЕМОЙ 3 будут созданы таски:

- `T-XXX`: Переименовать `GetPostKernelSource()` → `GetPostKernelSource_TwoPeaks()`
- `T-XXX`: Обновить все вызовы на `GetPostKernelSource_TwoPeaks()`
- `T-XXX`: Создать `GetPostKernelSource_OnePeak()` (базовая версия)
- `T-XXX`: Alex пишет алгоритм OnePeak в комментариях
- `T-XXX`: Кодо реализует алгоритм и проверяет
- `T-XXX`: Исправить амплитуду правой стороны в TwoPeaks
- `T-XXX`: Реализовать выбор кернела по `PeakSearchMode`
- `T-XXX`: Создать `test_one_peak.hpp` (валидация + профилирование)
- `T-XXX`: Создать `test_two_peaks.hpp` (валидация + профилирование)
- `T-XXX`: Сравнить производительность OnePeak vs TwoPeaks

---

## 🎨 ОБСУЖДЕНИЕ: Один кернел или два?

**ВОПРОС**: Может объединить в один кернел с параметром `mode`?

```cpp
__kernel void post_kernel_unified(..., uint mode) {
    if (mode == 0) {
        // ONE_PEAK
    } else {
        // TWO_PEAKS
    }
}
```

**РЕШЕНИЕ**: ❌ **НЕТ, ЛУЧШЕ ДВА ОТДЕЛЬНЫХ!**

**Почему?**
1. ✅ Нет ветвлений (`if`) → лучше производительность
2. ✅ Single Responsibility Principle
3. ✅ Разные размеры выходного буфера (4 vs 8 MaxValue)
4. ✅ Проще отладка и тестирование
5. ✅ Проще оптимизация каждого кернела отдельно

**Alex согласен**: Два отдельных кернела! ✅

---

## 🔗 СВЯЗИ С ДРУГИМИ ТЕМАМИ

- **ТЕМА 1** (API Refactoring): Использует `PeakSearchMode` для выбора кернела
- **ТЕМА 2** (Batch processing): Использует оба кернела для тестирования

---

## ✅ КРИТЕРИИ ГОТОВНОСТИ

- ✅ `GetPostKernelSource_TwoPeaks()` — переименован и работает
- ✅ `GetPostKernelSource_OnePeak()` — создан и работает
- ✅ Амплитуда правой стороны в TwoPeaks — исправлена
- ✅ Выбор кернела по `PeakSearchMode` — реализован
- ✅ Тесты для обоих кернелов — проходят
- ✅ Валидация CPU vs GPU — совпадают
- ✅ Профилирование — работает

---

*Последнее обновление: 2026-02-10*
*Автор: Кодо (AI Assistant)*
