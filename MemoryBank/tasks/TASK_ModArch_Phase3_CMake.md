# ⚙️ Фаза 3: Правим CMakeLists.txt + namespace dsp::

> **Индекс**: [`TASK_Modular_Architecture_INDEX.md`](TASK_Modular_Architecture_INDEX.md)
> **Спецификация**: [`specs/modular_architecture_plan.md`](../specs/modular_architecture_plan.md) — раздел 7 Фаза 3, разделы 4 и 8.3
> **Статус**: ⬜ BACKLOG
> **Зависимость**: Фаза 2 должна быть ✅ DONE
> **Платформа**: Windows (редактирование файлов, GPU не нужен)
> **Результат**: каждый репо собирается standalone `cmake --build`, namespace `dsp::` в C++

---

## 🔴 КРИТИЧЕСКИ ВАЖНО перед началом

### 1. find_package — только lowercase на Linux!

```cmake
# ❌ ВСЕГДА НЕПРАВИЛЬНО на Debian:
find_package(HIP REQUIRED)
find_package(HipFFT REQUIRED)

# ✅ ПРАВИЛЬНО:
find_package(hip REQUIRED)
find_package(hipfft REQUIRED)
find_package(rocprim REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)
find_package(hiprtc REQUIRED)
```

**Проверить в каждом файле:** `grep -r "find_package(H" .` → не должно быть результатов!

### 2. target_include_directories — kernels/ всегда PRIVATE

```cmake
target_include_directories(DspXxx
  PUBLIC  include      # заголовки для потребителей
  PRIVATE kernels/     # ROCm .hip файлы — НЕ экспортируем!
  PRIVATE src          # приватные impl заголовки
)
```

### 3. Diamond dependency — централизованный FetchContent в DSP

В `DSP/CMakeLists.txt` все `FetchContent_Declare` должны быть ПЕРЕД любым `add_subdirectory`.
Это гарантирует что все модули получают одну версию зависимостей.

---

## Что делаем

1. В каждом модуле заменяем `target_link_libraries` — старые внутренние пути → новые target'ы
2. Убираем абсолютные include-пути (`${CMAKE_SOURCE_DIR}/modules/...`)
3. Добавляем `$<BUILD_INTERFACE:...>` + `$<INSTALL_INTERFACE:...>`
4. Добавляем `install()` + `export()` + `{Repo}Config.cmake.in`
5. Добавляем `namespace dsp::` — **сразу**, не откладываем
6. Python биндинги — в Фазе 3b (отдельная задача)

---

## Чеклист

### ⚙️ core

- [ ] **M1.1** Проверить `target_link_libraries` — нет зависимостей на другие dsp репо ✅
- [ ] **M1.2** Убрать relative include пути типа `../../` из заголовков DrvGPU
- [ ] **M1.3** Добавить `namespace dsp { class DrvGPU {...}; }` во все публичные заголовки
- [ ] **M1.4** Проверить `target_include_directories`:
  ```cmake
  target_include_directories(DspCore
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE kernels/ src)
  ```
- [ ] **M1.5** Проверить что `tests/CMakeLists.txt` линкует `DspCore::TestUtils`:
  ```cmake
  target_link_libraries(test_core PRIVATE DspCore::DspCore DspCore::TestUtils)
  ```
- [ ] **M1.6** Проверить `find_package(hip REQUIRED)` — lowercase ✅
- [ ] **M1.7** `cmake -S . -B build --preset local-dev && cmake --build build` — ✅

---

### ⚙️ spectrum

- [ ] **M2.1** Заменить `target_link_libraries`:
  ```cmake
  # БЫЛО (GPUWorkLib):
  target_link_libraries(fft_func PRIVATE drvgpu)
  # СТАЛО:
  target_link_libraries(DspSpectrum PUBLIC DspCore::DspCore hip::hipfft)
  ```
- [ ] **M2.2** Убрать `${CMAKE_SOURCE_DIR}/DrvGPU/include` из include directories
- [ ] **M2.3** Добавить `target_include_directories` с PRIVATE kernels/:
  ```cmake
  target_include_directories(DspSpectrum
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE kernels/ src)
  ```
- [ ] **M2.4** Добавить `namespace dsp::spectrum {}` во все публичные заголовки:
  - `FFTProcessor` → `dsp::spectrum::FFTProcessor`
  - `FirFilter` → `dsp::spectrum::FirFilter`
  - `IirFilter` → `dsp::spectrum::IirFilter`
  - `LchFarrow` → `dsp::spectrum::LchFarrow`
- [ ] **M2.5** Проверить `find_package(hip REQUIRED)` + `find_package(hipfft REQUIRED)` — lowercase ✅
- [ ] **M2.6** `cmake --build` — ✅

---

### ⚙️ stats

- [ ] **M3.1** Заменить `target_link_libraries`:
  ```cmake
  # БЫЛО: target_link_libraries(statistics PRIVATE drvgpu rocprim)
  # СТАЛО:
  target_link_libraries(DspStats PUBLIC DspCore::DspCore roc::rocprim)
  ```
- [ ] **M3.2** Убрать include пути на DrvGPU из папки GPUWorkLib
- [ ] **M3.3** `target_include_directories` с PRIVATE kernels/
- [ ] **M3.4** Добавить `namespace dsp::stats {}`:
  - `StatisticsProcessor` → `dsp::stats::StatisticsProcessor`
- [ ] **M3.5** `find_package(rocprim REQUIRED)` — lowercase ✅
- [ ] **M3.6** `cmake --build` — ✅

---

### ⚙️ signal_generators

- [ ] **M4.1** Заменить `target_link_libraries`:
  ```cmake
  # БЫЛО: target_link_libraries(signal_generators PRIVATE drvgpu lch_farrow hiprtc)
  # СТАЛО:
  target_link_libraries(DspSignalGenerators
    PUBLIC  DspCore::DspCore DspSpectrum::DspSpectrum
    PRIVATE hiprtc)
  ```
- [ ] **M4.2** Убрать прямые пути к DrvGPU/ и lch_farrow/
- [ ] **M4.3** `target_include_directories` с PRIVATE kernels/
- [ ] **M4.4** Добавить `namespace dsp::signal_generators {}`:
  - `CwGenerator` → `dsp::signal_generators::CwGenerator`
  - `LfmGenerator` → `dsp::signal_generators::LfmGenerator`
  - `NoiseGenerator` → `dsp::signal_generators::NoiseGenerator`
  - `ScriptGenerator` → `dsp::signal_generators::ScriptGenerator`
  - `FormSignalGenerator` → `dsp::signal_generators::FormSignalGenerator`
  - `DelayedFormSignalGenerator` → `dsp::signal_generators::DelayedFormSignalGenerator`
- [ ] **M4.5** `cmake --build` — ✅

---

### ⚙️ heterodyne

- [ ] **M5.1** Заменить `target_link_libraries`:
  ```cmake
  # БЫЛО: target_link_libraries(heterodyne PRIVATE drvgpu signal_generators fft_func)
  # СТАЛО:
  target_link_libraries(DspHeterodyne
    PUBLIC DspCore::DspCore DspSpectrum::DspSpectrum DspSignalGenerators::DspSignalGenerators)
  ```
- [ ] **M5.2** Убрать прямые пути ко всем модулям GPUWorkLib
- [ ] **M5.3** `target_include_directories` с PRIVATE kernels/
- [ ] **M5.4** Добавить `namespace dsp::heterodyne {}`:
  - `HeterodyneDechirp` → `dsp::heterodyne::HeterodyneDechirp`
  - `NcoMixDown` → `dsp::heterodyne::NcoMixDown` (если есть)
- [ ] **M5.5** `cmake --build` — ✅

---

### ⚙️ linalg

- [ ] **M6.1** Заменить `target_link_libraries`:
  ```cmake
  # БЫЛО: target_link_libraries(capon PRIVATE drvgpu vector_algebra rocblas rocsolver)
  # СТАЛО:
  target_link_libraries(DspLinalg
    PUBLIC  DspCore::DspCore roc::rocblas roc::rocsolver
    PRIVATE hiprtc)
  ```
  > ⚠️ Проверить актуальные target-имена rocBLAS: `roc::rocblas` или `hip::hipblas`?
  > ROCm 6+: предпочтителен `roc::rocblas`. Выполнить: `find /opt/rocm -name "rocblasConfig.cmake"`.
- [ ] **M6.2** `target_include_directories` с PRIVATE kernels/
- [ ] **M6.3** `namespace dsp::linalg {}`:
  - `VectorAlgebra` → `dsp::linalg::VectorAlgebra`
  - `Capon` → `dsp::linalg::Capon`
  - `MatrixOps` → `dsp::linalg::MatrixOps`
- [ ] **M6.4** `find_package(rocblas REQUIRED)` + `find_package(rocsolver REQUIRED)` — lowercase ✅
- [ ] **M6.5** `cmake --build` — ✅

---

### ⚙️ radar

- [ ] **M7.1** Заменить `target_link_libraries`:
  ```cmake
  # БЫЛО: target_link_libraries(range_angle PRIVATE drvgpu fft_func statistics)
  # СТАЛО:
  target_link_libraries(DspRadar
    PUBLIC DspCore::DspCore DspSpectrum::DspSpectrum DspStats::DspStats)
  ```
- [ ] **M7.2** `target_include_directories` с PRIVATE kernels/
- [ ] **M7.3** `namespace dsp::radar {}`:
  - `RangeAngleProcessor` → `dsp::radar::RangeAngleProcessor`
  - `FmCorrelator` → `dsp::radar::FmCorrelator`
- [ ] **M7.4** `cmake --build` — ✅

---

### ⚙️ strategies

- [ ] **M8.1** Заменить `target_link_libraries`:
  ```cmake
  target_link_libraries(DspStrategies
    PUBLIC DspCore::DspCore DspSpectrum::DspSpectrum DspStats::DspStats
           DspSignalGenerators::DspSignalGenerators DspHeterodyne::DspHeterodyne
           DspLinalg::DspLinalg)
  ```
- [ ] **M8.2** `target_include_directories` с PRIVATE kernels/
- [ ] **M8.3** `namespace dsp::strategies {}`:
  - `AntennaProcessor` → `dsp::strategies::AntennaProcessor`
  - `PipelineBase` → `dsp::strategies::PipelineBase`
- [ ] **M8.4** `cmake --build` — ✅

---

### ⚙️ DSP мета-репо

- [ ] **M9.0** Проверить `option(DSP_BUILD_*)` — есть для каждого модуля
- [ ] **M9.1** Проверить что `DSP/CMakeLists.txt` объявляет все `FetchContent_Declare`
  централизованно ПЕРЕД зависимыми модулями (diamond dependency fix)
- [ ] **M9.2** `cmake --preset local-dev` — все 8 репо подтягиваются
- [ ] **M9.3** `cmake --build build` — всё собирается
- [ ] **M9.4** `cmake --install build --prefix DSP/Python/lib` — `.so` файлы на месте
- [ ] **M9.5** Проверить `cmake --preset spectrum-only` — собирает только core + spectrum

---

## Definition of Done

- [ ] Каждый репо: `cmake --build` проходит без ошибок
- [ ] Нет `find_package(HIP)` с заглавной буквы: `grep -r "find_package(H" .` → пусто
- [ ] Нет include-путей вида `../../../DrvGPU/` или `${CMAKE_SOURCE_DIR}/modules/`
- [ ] В каждом репо: `target_include_directories` содержит `PRIVATE kernels/`
- [ ] Все публичные классы в `namespace dsp::{repo_name}::`
- [ ] DSP мета-репо: полная сборка проходит
- [ ] DSP мета-репо: `spectrum-only` пресет работает
- [ ] Можно переходить к Фазе 3b (Python migration)

---

*Создан: 2026-04-12 | Обновлён: 2026-04-12 (добавлены find_package lowercase, kernels/ PRIVATE, diamond dep fix, rocblas target warning, option guards) | Автор: Кодо*
