# 📝 fft_maxima OpenCL Regression Fix — Спецификация

> **Модуль**: `fft_maxima`
> **Статус**: 🟢 Active
> **Платформы**: OpenCL
> **Автор**: Alex
> **Создано**: 2026-03-07
> **Обновлено**: 2026-03-07

---

## 🎯 Назначение

Зафиксировать причину OpenCL regression в `fft_maxima`, источник старой рабочей логики, фактические исправления и результат проверки.

Документ нужен, чтобы не создавать путаницу между разными проектами, сборками и ветками, а также чтобы при следующей оптимизации было понятно, какие места уже ломались.

---

## 📌 Короткий вывод

Проблема была не в ROCm commit `47edd7b` напрямую.

OpenCL regression сидела в двух местах:

1. В legacy OpenCL пути `AllMaxima` kernel `detect_all_maxima` уже был переписан на `2D NDRange`, но хост-код в `SpectrumMaximaFinder::FindAllMaxima()` продолжал запускать его как `1D`.
2. В OpenCL `clFFT` pre-callback была внесена bitwise-оптимизация `div/mod -> >> &`, которая оказалась небезопасной для реального поведения callback и ломала обычный FFT pipeline (`ONE_PEAK`, `TWO_PEAKS`).

После исправления оба проблемных участка OpenCL C++ тесты снова проходят.

---

## 🔍 Откуда взята старая рабочая логика

Старая рабочая логика была взята не "из головы" и не из внешнего источника.

Источники были такие:

1. Git-ревизия `9663328` с сообщением `работает`.
2. Старый стабильный kernel source для OpenCL pre-callback из:
   `git show 9663328:modules/fft_maxima/include/kernels/fft_kernel_sources.hpp`
3. Старый inline/legacy OpenCL pipeline в коде модуля:
   `modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp`

Именно сравнение текущего состояния с `9663328` показало, что:

- раньше pre-callback использовал обычные:
  - `beam_idx = inoffset / nFFT`
  - `pos_in_fft = inoffset % nFFT`
- позже это было заменено на:
  - `beam_idx = inoffset >> nFFT_log2`
  - `pos_in_fft = inoffset & (nFFT - 1)`

Также сравнение legacy pipeline показало, что после оптимизации kernel `detect_all_maxima` стал `2D`, но один из путей вызова остался `1D`.

---

## 🧨 Симптомы regression

Наблюдаемые симптомы были такие:

- `FindAllMaxima` multi-beam работал некорректно или обрабатывал только `beam 0`
- full pipeline на OpenCL давал неправильные результаты в `ONE_PEAK/TWO_PEAKS`
- часть C++ OpenCL тестов `fft_maxima` падала после недавних оптимизаций
- Python-путь не удалось использовать как основной источник валидации, потому что текущая сборка проекта не содержит `gpuworklib` (`BUILD_PYTHON=OFF`)

---

## 🧠 Корневая причина №1

### 1. `detect_all_maxima` переведён на 2D, а legacy launch остался 1D

Файл с ошибкой:

- `modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp`

Kernel уже ожидал:

- `get_global_id(0)` = позиция в FFT
- `get_global_id(1)` = индекс луча

То есть модель запуска стала такой:

```cpp
size_t detect_global[2] = { nFFT, beam_count };
size_t detect_local[2]  = { 256, 1 };
```

Но один из хост-путей продолжал запускать kernel как `1D`:

```cpp
size_t global_size = ((total_elements + 255) / 256) * 256;
size_t local_size = 256;
clEnqueueNDRangeKernel(..., 1, ..., &global_size, &local_size, ...)
```

Что происходило фактически:

- `get_global_id(1)` в таком вызове не давал корректного beam dimension
- kernel реально обрабатывал только первый луч или давал повреждённую beam-индексацию
- multi-beam AllMaxima начинал вести себя нестабильно

### Исправление

Legacy launch был приведён в соответствие kernel'у:

```cpp
size_t detect_global[2] = { static_cast<size_t>(nFFT), static_cast<size_t>(beam_count) };
size_t detect_local[2]  = { 256, 1 };
clEnqueueNDRangeKernel(..., 2, nullptr, detect_global, detect_local, ...)
```

---

## 🧠 Корневая причина №2

### 2. Bitwise-оптимизация pre-callback сломала FFT path

Файл с ошибкой:

- `modules/fft_maxima/include/kernels/fft_kernel_sources.hpp`

Оптимизация была такой:

```cpp
beam_idx   = inoffset >> nFFT_log2;
pos_in_fft = inoffset & (nFFT - 1);
```

До этого рабочий вариант был обычный:

```cpp
beam_idx   = inoffset / nFFT;
pos_in_fft = inoffset % nFFT;
```

Почему это опасно:

- `clFFT` callback использует свой контракт на `inoffset`
- теоретически bitwise-вариант кажется эквивалентным только при очень жёстком предположении о layout
- в реальном OpenCL пути эта оптимизация дала regression в обычной FFT обработке
- по факту после возврата к `div/mod` тесты снова проходят

Важно:

- поле `nFFT_log2` оставлено в header как legacy/reserved, чтобы не ломать layout структуры
- но в callback оно больше не используется

### Исправление

В pre-callback возвращена старая стабильная логика:

```cpp
uint beam_idx   = inoffset / nFFT;
uint pos_in_fft = inoffset % nFFT;
```

---

## 🔧 Изменённые файлы

Исправления были внесены в:

1. `modules/fft_maxima/include/kernels/fft_kernel_sources.hpp`
2. `modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp`

Временные изменения в `modules/fft_maxima/tests/all_test.hpp` использовались только для прогона OpenCL тестов и затем были возвращены обратно.

---

## ✅ Фактическая проверка

### Проверено

- Сборка `GPUWorkLib` на Windows проходит
- OpenCL C++ тесты `fft_maxima` были временно включены и запущены вручную
- После исправления прошли:
  - `test_spectrum_maxima`
  - `test_gpu_generator_integration`
  - `test_find_all_maxima`

Особенно важно, что снова прошли кейсы:

- `CPU vs GPU` для `TWO_PEAKS`
- `GPU generator integration`
- `FindAllMaxima: Multi-beam`
- `FindAllMaxima: Full Pipeline (CPU data)`
- `FindAllMaxima: Full Pipeline (GPU data)`
- `AllMaxima<cl_mem>: GPU FFT data`

Итог:

- `FindAllMaxima: 7/7 tests passed`

### Не проверено

Python tests `Python_test/fft_maxima` не были валидным источником финальной проверки, потому что:

- `BUILD_PYTHON=OFF` в текущем `build/CMakeCache.txt`
- модуль `gpuworklib` не собран
- оба Python interpreter (`Python312`, `Python314`) падают на `ModuleNotFoundError: No module named 'gpuworklib'`

Это не ошибка `fft_maxima` OpenCL логики, а отсутствие Python bindings в данной конфигурации сборки.

---

## 🚫 Что важно не повторить

При следующих оптимизациях OpenCL пути нельзя:

- менять размерность launch (`1D/2D`) без синхронного обновления всех хост-путей вызова
- заменять `div/mod` на bitwise в callback'ах clFFT без отдельной полной валидации на реальном runtime
- считать, что ROCm diff автоматически объясняет OpenCL regression

Любая micro-optimization в callback/NDRange должна проверяться минимум на:

- single-beam
- multi-beam
- CPU data -> FFT -> peaks
- GPU data -> FFT -> peaks
- AllMaxima raw FFT path

---

## 📚 Ссылки

- Рабочая ревизия: `9663328` (`работает`)
- Подозреваемый поздний ROCm commit: `47edd7b` (`оптимизация kernels`)
- Файл с legacy OpenCL pipeline:
  `modules/fft_maxima/src/spectrum_maxima_finder_all_maxima.cpp`
- Файл с OpenCL callback/kernel sources:
  `modules/fft_maxima/include/kernels/fft_kernel_sources.hpp`

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-03-07 | Кодо | Зафиксирована причина OpenCL regression в fft_maxima и описан фактический fix |
