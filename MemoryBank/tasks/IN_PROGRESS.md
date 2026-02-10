# 🔄 IN PROGRESS — Текущие задачи

> **Обновлено**: 2026-02-10
> **Тема**: ТЕМА 3 — Kernel Refactoring (OnePeak & TwoPeaks)
> **Статус**: ✅ ЗАВЕРШЕНО!

---

## 🎯 ТЕМА 3: Kernel Refactoring — ВЫПОЛНЕНО!

### 📋 Таски (12 шт.) — ВСЕ ЗАВЕРШЕНЫ!

| ID | Фаза | Задача | Статус | Зависит от |
|----|------|--------|--------|------------|
| **KERN-01** | Исследование | Изучить структуру fft_kernel_sources.hpp | ✅ completed | - |
| **KERN-02** | Исследование | Понять структуру MaxValue и выходных данных | ✅ completed | KERN-01 |
| **KERN-03** | Рефакторинг | **Добавить суффикс `_opencl` ко ВСЕМ кернелам** | ✅ completed | KERN-02 |
| **KERN-04** | TwoPeaks | Переименовать → `GetPostKernelSource_TwoPeaks_opencl` | ✅ completed | KERN-03 |
| **KERN-05** | Рефакторинг | Обновить все вызовы кернелов (новые имена с _opencl) | ✅ completed | KERN-04 |
| **KERN-06** | TwoPeaks | Добавить документацию + компиляция | ✅ completed | KERN-05 |
| **KERN-07** | OnePeak | Создать `GetPostKernelSource_OnePeak_opencl()` | ✅ completed | KERN-06 |
| **KERN-08** | OnePeak | Реализовать алгоритм OnePeak (алгоритм от Alex) | ✅ completed | KERN-07 |
| **KERN-09** | Исправление | Исправить амплитуду правой стороны в TwoPeaks | ✅ completed | KERN-08 |
| **KERN-10** | Интеграция | Интегрировать PeakSearchMode (выбор кернела) | ✅ completed | KERN-09 |
| **KERN-11** | Тесты | **Добавить переключатель 1/2 пика в test_spectrum_maxima.hpp** | ✅ completed | KERN-10 |
| **KERN-12** | Финал | Документация, обновление статуса | ✅ completed | KERN-11 |

**Легенда**: ⚪ pending | 🔵 in_progress | ✅ completed

---

## 📊 Результат рефакторинга

### 1️⃣ Переименованные кернелы с суффиксом `_opencl`

```cpp
// Было → Стало:
GetPaddingKernelSource()     → GetPaddingKernelSource_opencl()
GetPreCallbackSource32()     → GetPreCallbackSource32_opencl()
GetPostKernelSource()        → GetPostKernelSource_TwoPeaks_opencl()
// Новый:
                              GetPostKernelSource_OnePeak_opencl()
```

### 2️⃣ PeakSearchMode enum

```cpp
// В spectrum_maxima_types.h:
enum class PeakSearchMode {
    ONE_PEAK,   // 4 MaxValue на луч
    TWO_PEAKS   // 8 MaxValue на луч (по умолчанию)
};

// В SpectrumParams:
PeakSearchMode peak_mode = PeakSearchMode::TWO_PEAKS;
```

### 3️⃣ Переключатель теста

```cpp
// В test_spectrum_maxima.hpp:
enum class TestMode { ONE_PEAK, TWO_PEAKS };

inline int run(TestMode mode = TestMode::TWO_PEAKS);
inline int run_one_peak();   // Удобный вызов для ONE_PEAK
inline int run_two_peaks();  // Удобный вызов для TWO_PEAKS
```

---

## 📁 Изменённые файлы

| Файл | Что сделано |
|------|-------------|
| `modules/fft_maxima/include/kernels/fft_kernel_sources.hpp` | `_opencl` суффиксы, OnePeak кернел |
| `modules/fft_maxima/include/interface/spectrum_maxima_types.h` | PeakSearchMode enum |
| `modules/fft_maxima/src/spectrum_maxima_finder.cpp` | Интеграция PeakSearchMode |
| `modules/fft_maxima/src/antenna_fft_release.cpp` | Обновлены вызовы кернелов |
| `modules/fft_maxima/tests/test_spectrum_maxima.hpp` | TestMode переключатель |

---

## ✅ Критерии готовности — ВЫПОЛНЕНЫ

- ✅ Все кернелы имеют суффикс `_opencl`
- ✅ `GetPostKernelSource_TwoPeaks_opencl()` — документирован и работает
- ✅ `GetPostKernelSource_OnePeak_opencl()` — создан по алгоритму Alex
- ✅ `PeakSearchMode` enum — интегрирован
- ✅ Переключатель теста 1/2 пика — добавлен
- ✅ Компиляция — успешна

---

*Завершено: 2026-02-10*
*Исполнитель: Кодо (AI Assistant)*
