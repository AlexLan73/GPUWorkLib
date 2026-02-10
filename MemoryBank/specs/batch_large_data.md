# 📊 Batch Processing — Large Data Testing

> **Тема**: ТЕМА 2 (изначально, теперь ТЕМА 3 по приоритету)
> **Приоритет**: 🔥 Высокий
> **Статус**: 📋 Planned
> **Дата создания**: 2026-02-10
> **Автор**: Кодо (AI Assistant)

---

## 🎯 ЦЕЛЬ

Протестировать `fft_maxima` на **больших данных**:
- **256 лучей** × **1 300 000 точек** на луч
- Использовать **batch processing** (данные не поместятся в GPU целиком)
- Вывод **валидации** (CPU vs GPU)
- Вывод **профилирования** (GPUProfiler)

---

## 🔥 ПРОБЛЕМА

**Оценка памяти**:
```
Дано:
  antenna_count = 256
  n_point = 1 300 000
  repeat_count = 4

Расчёт:
  nFFT = NextPowerOf2(1300000 * 4) ≈ 8 388 608 (≈8M)

Память на луч:
  - Input:  1 300 000 × 8 bytes (complex<float>) = 10.4 MB
  - FFT:    8 388 608 × 8 bytes = 67.1 MB
  - Output: 8 × sizeof(MaxValue) ≈ 256 bytes

Всего на 256 лучей:
  - Input:  256 × 10.4 MB = 2.66 GB
  - FFT:    256 × 67.1 MB = 17.18 GB
  - ИТОГО: ~20 GB ❌ (не поместится на большинстве GPU!)
```

**ПРОБЛЕМА**: GPU не имеет 20GB памяти!

---

## ✅ РЕШЕНИЕ

### 1️⃣ Использовать BatchManager

**Где**: `DrvGPU/services/batch_manager.hpp` — УЖЕ СУЩЕСТВУЕТ! ✅

**Функциональность**:
- ✅ `CalculateOptimalBatchSize()` — расчёт оптимального размера пакета по памяти GPU
- ✅ `GetBatchRanges()` — разбиение на диапазоны [start, end)
- ✅ Универсальный (не зависит от FFT)

### 2️⃣ Использовать antenna_fft_core

**Файл**: `modules/fft_maxima/src/antenna_fft_core.cpp`

**Метод**: `AntennaFFTCore::ProcessNew(cl_mem)` — УЖЕ ПОДДЕРЖИВАЕТ `cl_mem`! ✅

**Batch режим**: Поддерживается через `padding_kernel` с `beam_offset`

### 3️⃣ Создать тест test_large_batch.hpp

**Структура** (по образцу `test_spectrum_maxima.hpp`):
```cpp
#pragma once
#include "spectrum_maxima_finder.h"
#include "drv_gpu.hpp"
#include "DrvGPU/services/gpu_profiler.hpp"
#include "modules/fft_maxima/tests/cpu_fft_reference.hpp"

namespace test_large_batch {

using namespace antenna_fft;
using namespace drv_gpu_lib;

// ════════════════════════════════════════════════════════════════
// Генерация тестовых данных (256 лучей × 1300000 точек)
// ════════════════════════════════════════════════════════════════
inline std::vector<std::complex<float>> GenerateLargeTestData() {
    const int antenna_count = 256;
    const int n_point = 1300000;

    std::vector<std::complex<float>> data(antenna_count * n_point);

    for (int antenna = 0; antenna < antenna_count; ++antenna) {
        float freq = 2.5f * (1.0f + (antenna + 1) / 10.0f);

        for (int t = 0; t < n_point; ++t) {
            float phase = 2.0f * M_PI * freq * t / 1000.0f;
            float value = std::sin(phase);
            data[antenna * n_point + t] = std::complex<float>(value, 0.0f);
        }
    }

    return data;
}

// ════════════════════════════════════════════════════════════════
// Главная функция теста
// ════════════════════════════════════════════════════════════════
inline int run() {
    try {
        std::cout << "\n╔══════════════════════════════════════════════════╗\n";
        std::cout << "║  TEST: Large Batch (256 × 1300000)              ║\n";
        std::cout << "╚══════════════════════════════════════════════════╝\n\n";

        // 1. Инициализация DrvGPU
        DrvGPU gpu(BackendType::OPENCL, 0);
        gpu.Initialize();

        // 2. Запуск GPUProfiler
        auto& profiler = GPUProfiler::GetInstance();
        if (profiler.IsEnabled()) {
            profiler.Start();
        }

        // 3. Создать SpectrumMaximaFinder (новый API!)
        SpectrumMaximaFinder finder(&gpu.GetBackend());

        // 4. Сгенерировать данные на CPU
        std::cout << "📊 Генерация тестовых данных (256 × 1300000)...\n";
        auto cpu_data = GenerateLargeTestData();
        std::cout << "  ✅ Сгенерировано: " << cpu_data.size() << " точек\n\n";

        // 5. Создать InputData (новый API!)
        InputData<std::vector<std::complex<float>>> input;
        input.antenna_count = 256;
        input.n_point = 1300000;
        input.data = std::move(cpu_data);

        // 6. Параметры обработки
        ProcessingParams params;
        params.repeat_count = 4;
        params.sample_rate = 1000.0f;

        // 7. Обработка (автоматический batch!)
        std::cout << "🚀 Запуск обработки (автоматический batch)...\n";
        auto results = finder.Process(input, params, PeakSearchMode::ONE_PEAK);
        std::cout << "  ✅ Обработка завершена!\n\n";

        // 8. Профилирование
        if (profiler.IsEnabled()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            profiler.PrintReport();
            profiler.Stop();
        }

        // 9. Валидация (упрощённая — только первые 10 лучей)
        std::cout << "🔍 Валидация (первые 10 лучей)...\n";
        // ... валидация CPU vs GPU для первых 10 лучей

        std::cout << "\n✅ ТЕСТ ЗАВЕРШЁН!\n\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ ОШИБКА: " << e.what() << "\n\n";
        return 1;
    }
}

} // namespace test_large_batch
```

---

## 🔍 АНАЛИЗ: BatchManager в antenna_fft_core

**ВОПРОС от Alex** (задача 2.2):
> "Вернуться к вопросу о переносе этой функции на DrvGPU/services/batch_manager.cpp — анализ проблемы решения"

**ТЕКУЩЕЕ СОСТОЯНИЕ**:
- ✅ `BatchManager` УЖЕ в `DrvGPU/services/batch_manager.hpp`
- ✅ Универсальный (расчёт размеров, диапазоны)
- ❓ Используется ли в `antenna_fft_core.cpp`?

**ЗАДАЧА ПРИ РАБОТЕ НАД ТЕМОЙ**:
1. Проанализировать `antenna_fft_core.cpp`
2. Проверить, использует ли он `BatchManager` или дублирует логику
3. Если дублирует → рефакторинг (использовать `BatchManager`)
4. Если использует → документировать

---

## 📋 ЗАДАЧИ (создаются при начале работы)

При начале работы над ТЕМОЙ 2 будут созданы таски:

- `T-XXX`: Создать `test_large_batch.hpp` (структура по образцу `test_spectrum_maxima.hpp`)
- `T-XXX`: Реализовать генерацию данных (256 × 1300000)
- `T-XXX`: Интеграция с новым API (`InputData<vector>`)
- `T-XXX`: Реализовать автоматический batch processing
- `T-XXX`: Вывод профилирования (GPUProfiler)
- `T-XXX`: Вывод валидации (CPU vs GPU, упрощённая)
- `T-XXX`: Анализ использования `BatchManager` в `antenna_fft_core.cpp`
- `T-XXX`: Рефакторинг (если найдено дублирование)
- `T-XXX`: Документирование результатов анализа в `research/`

---

## 🔗 СВЯЗИ С ДРУГИМИ ТЕМАМИ

- **ТЕМА 1** (API Refactoring): Использует новый API (`InputData`, `ProcessingParams`)
- **ТЕМА 3** (Кернелы): Тестирует оба режима (`ONE_PEAK`, `TWO_PEAKS`)

---

## ✅ КРИТЕРИИ ГОТОВНОСТИ

- ✅ `test_large_batch.hpp` — создан и работает
- ✅ Batch processing — работает автоматически
- ✅ 256 лучей × 1300000 точек — обрабатываются без ошибок
- ✅ Профилирование — выводится корректно
- ✅ Валидация — проходит (CPU vs GPU)
- ✅ Анализ `BatchManager` — выполнен
- ✅ Дублирование кода — устранено (если найдено)
- ✅ Документация — обновлена

---

*Последнее обновление: 2026-02-10*
*Автор: Кодо (AI Assistant)*
