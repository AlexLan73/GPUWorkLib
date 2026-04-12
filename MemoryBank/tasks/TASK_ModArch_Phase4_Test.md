# 🧪 Фаза 4: Тестирование на GPU (Debian + Radeon9070)

> **Индекс**: [`TASK_Modular_Architecture_INDEX.md`](TASK_Modular_Architecture_INDEX.md)
> **Спецификация**: [`specs/modular_architecture_plan.md`](../specs/modular_architecture_plan.md) — раздел 7, Фаза 4
> **Статус**: ⬜ BACKLOG
> **Зависимость**: Фазы 3 и 3b должны быть ✅ DONE
> **Платформа**: 🐧 Debian + ROCm 7.2+ + Radeon9070 (gfx1201)
> **Результат**: все C++ тесты зелёные, Python тесты работают

---

## Порядок тестирования: строго от base к dependent

```
core → spectrum → stats → linalg → signal_generators → heterodyne → radar → strategies
```

Не пропускать шаги! Если `core` не работает — всё остальное тоже не будет.

---

## Чеклист

### 🖥️ Подготовка рабочей станции Debian

- [ ] **T0.1** Проверить ROCm версию: `rocm-smi --version` → должно быть 7.2+
- [ ] **T0.2** Проверить GPU: `rocm-smi --showproductname` → Radeon RX 9070 / gfx1201
- [ ] **T0.3** Установить зависимости если нужно:
  ```bash
  sudo apt install cmake ninja-build git
  pip install pybind11 numpy
  ```
- [ ] **T0.4** Создать рабочую папку и склонировать репо:
  > ⚠️ Фазы 1-3b должны быть DONE — репо не пустые!
  ```bash
  mkdir ~/dsp-gpu && cd ~/dsp-gpu
  git clone https://github.com/dsp-gpu/DSP
  git clone https://github.com/dsp-gpu/core
  git clone https://github.com/dsp-gpu/spectrum
  git clone https://github.com/dsp-gpu/stats
  git clone https://github.com/dsp-gpu/signal_generators
  git clone https://github.com/dsp-gpu/heterodyne
  git clone https://github.com/dsp-gpu/linalg
  git clone https://github.com/dsp-gpu/radar
  git clone https://github.com/dsp-gpu/strategies
  ```
- [ ] **T0.5** Создать `CMakeUserPresets.json` в `DSP/` для Debian (локальные пути):
  ```json
  {
    "version": 3,
    "configurePresets": [
      {
        "name": "debian-local-dev",
        "inherits": "local-dev",
        "cacheVariables": {
          "FETCHCONTENT_SOURCE_DIR_DSPCORE":             "/home/user/dsp-gpu/core",
          "FETCHCONTENT_SOURCE_DIR_DSPSPECTRUM":         "/home/user/dsp-gpu/spectrum",
          "FETCHCONTENT_SOURCE_DIR_DSPSTATS":            "/home/user/dsp-gpu/stats",
          "FETCHCONTENT_SOURCE_DIR_DSPSIGNALGENERATORS": "/home/user/dsp-gpu/signal_generators",
          "FETCHCONTENT_SOURCE_DIR_DSPHETERODYNE":       "/home/user/dsp-gpu/heterodyne",
          "FETCHCONTENT_SOURCE_DIR_DSPLINALG":           "/home/user/dsp-gpu/linalg",
          "FETCHCONTENT_SOURCE_DIR_DSPRADAR":            "/home/user/dsp-gpu/radar",
          "FETCHCONTENT_SOURCE_DIR_DSPSTRATEGIES":       "/home/user/dsp-gpu/strategies"
        }
      }
    ]
  }
  ```
- [ ] **T0.6** Настроить PYTHONPATH:
  ```bash
  export PYTHONPATH=~/dsp-gpu/DSP/Python/lib:$PYTHONPATH
  echo 'export PYTHONPATH=~/dsp-gpu/DSP/Python/lib:$PYTHONPATH' >> ~/.bashrc
  ```
- [ ] **T0.7** Сконфигурировать DSP:
  ```bash
  cd ~/dsp-gpu/DSP
  cmake -S . -B build --preset debian-local-dev
  cmake --build build -j$(nproc)
  ```

---

### 🧪 Тест 1: core

- [ ] **T1.1** Запустить C++ тесты DrvGPU:
  ```bash
  cd ~/dsp-gpu/core
  cmake -S . -B build --preset debian-local-dev 2>/dev/null || \
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build
  ./build/test_drv_gpu
  ```
- [ ] **T1.2** Ожидаемый вывод: GPU найден, context инициализирован, profiler работает
- [ ] **T1.3** Если ошибка — проверить:
  - `configGPU.json` — `"PLATFORM": "AMD"`, `"GPU_COUNT": 1`
  - ROCm HIP headers: `/opt/rocm/include/hip/`
  - `DrvGPU::InitializeFromExternalStream` работает?
- [ ] **T1.4** Тест Python (базовый):
  ```bash
  cd ~/dsp-gpu/DSP
  cmake --install build --prefix Python/lib
  python Python/common/test_gpu_context.py    # если есть
  ```

---

### 🧪 Тест 2: spectrum (fft_func + filters + lch_farrow)

- [ ] **T2.1** C++ тесты FFTProcessor:
  ```bash
  cd ~/dsp-gpu/spectrum
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build
  ./build/test_fft_processor
  ./build/test_fir_filter
  ./build/test_lch_farrow
  ```
- [ ] **T2.2** Ожидаемый вывод: FFT результаты совпадают с эталоном, ошибка < 1e-4
- [ ] **T2.3** Python:
  ```bash
  python ~/dsp-gpu/DSP/Python/spectrum/test_fft.py
  python ~/dsp-gpu/DSP/Python/spectrum/test_filters.py
  ```

---

### 🧪 Тест 3: stats

- [ ] **T3.1** C++ тесты StatisticsProcessor:
  ```bash
  cd ~/dsp-gpu/stats
  cmake --build build
  ./build/test_statistics
  ```
- [ ] **T3.2** Проверить: welford mean/std, medians, radix sort — совпадают с CPU
- [ ] **T3.3** Python:
  ```bash
  python ~/dsp-gpu/DSP/Python/stats/test_statistics.py
  ```

---

### 🧪 Тест 4: linalg (vector_algebra + capon)

- [ ] **T4.1** C++ тесты VectorAlgebra:
  ```bash
  cd ~/dsp-gpu/linalg
  cmake --build build
  ./build/test_vector_algebra
  ```
- [ ] **T4.2** C++ тесты Capon (rocBLAS Cholesky):
  ```bash
  ./build/test_capon
  ```
  > ⚠️ Capon использует rocBLAS CGEMM — если не работает, проверить target-имена rocBLAS
- [ ] **T4.3** Python:
  ```bash
  python ~/dsp-gpu/DSP/Python/linalg/test_vector_algebra.py
  python ~/dsp-gpu/DSP/Python/linalg/test_capon.py
  ```

---

### 🧪 Тест 5: signal_generators

- [ ] **T5.1** C++ тесты генераторов:
  ```bash
  cd ~/dsp-gpu/signal_generators
  cmake --build build
  ./build/test_cw_generator
  ./build/test_lfm_generator
  ./build/test_noise_generator
  ./build/test_form_signal
  ```
- [ ] **T5.2** Ожидаемый вывод: сигналы совпадают с эталоном (по SNR и амплитуде)
- [ ] **T5.3** Python:
  ```bash
  python ~/dsp-gpu/DSP/Python/signal_generators/test_lfm.py
  python ~/dsp-gpu/DSP/Python/signal_generators/test_form_signal.py
  ```

---

### 🧪 Тест 6: heterodyne

- [ ] **T6.1** C++ тесты Dechirp:
  ```bash
  cd ~/dsp-gpu/heterodyne
  cmake --build build
  ./build/test_heterodyne_dechirp
  ```
- [ ] **T6.2** Ожидаемый вывод: 7 тестов Dechirp (как было в GPUWorkLib)
- [ ] **T6.3** Python:
  ```bash
  python ~/dsp-gpu/DSP/Python/heterodyne/test_dechirp.py
  ```

---

### 🧪 Тест 7: radar

- [ ] **T7.1** C++ тесты RangeAngle + FmCorrelator:
  ```bash
  cd ~/dsp-gpu/radar
  cmake --build build
  ./build/test_range_angle
  ./build/test_fm_correlator
  ```
- [ ] **T7.2** Python:
  ```bash
  python ~/dsp-gpu/DSP/Python/radar/test_range_angle.py
  ```

---

### 🧪 Тест 8: strategies

- [ ] **T8.1** C++ тесты AntennaProcessor:
  ```bash
  cd ~/dsp-gpu/strategies
  cmake --build build
  ./build/test_strategies
  ```
- [ ] **T8.2** Python:
  ```bash
  python ~/dsp-gpu/DSP/Python/strategies/test_antenna_processor.py
  ```

---

### 🧪 Тест 9: Интеграционный тест (DSP целиком)

- [ ] **T9.1** Полная сборка DSP:
  ```bash
  cd ~/dsp-gpu/DSP
  cmake --build build -j$(nproc)
  cmake --install build --prefix Python/lib
  ```
- [ ] **T9.2** Интеграционный Python тест:
  ```bash
  python ~/dsp-gpu/DSP/Python/integration/test_full_pipeline.py
  ```
  > Тест: core → spectrum → signal_generators → heterodyne → strategies

---

### 📊 Профилирование после успешных тестов

- [ ] **T10.1** Запустить benchmark одного модуля (например fft_processor):
  ```bash
  ./build/bench_fft_processor  # генерирует Results/Profiler/
  ```
- [ ] **T10.2** Установить baseline производительности на gfx1201:
  > ⚠️ Прямое сравнение с GPUWorkLib невозможно (GPUWorkLib не запускался на gfx1201).
  > Записать результаты как **baseline** в `DSP/Doc/Performance/baseline_gfx1201.md`.
  > Дальнейшие оптимизации сравниваем с этим baseline (не с GPUWorkLib).

---

## Типичные ошибки и их решение

| Ошибка | Вероятная причина | Решение |
|--------|------------------|---------|
| `hip::hipfft not found` | hipFFT пакет не установлен | `sudo apt install hipfft-dev` |
| `Target DspCore::DspCore not found` | FetchContent не нашёл репо | Проверить `FETCHCONTENT_SOURCE_DIR_DSPCORE` |
| `namespace dsp not found` | Заголовки без namespace | Добавить `using namespace dsp;` или исправить include |
| `configGPU.json not found` | Не скопирован в build/ | Добавить `configure_file` в CMakeLists.txt |
| `rocprim` тест падает | gfx1201 не поддерживается | Проверить версию ROCm (нужен 7.2+) |
| Python `ImportError: dsp_core` | `.so` не в Python path | `export PYTHONPATH=~/dsp-gpu/DSP/Python/lib:$PYTHONPATH` |
| `find_package(hip) FAILED` | Неправильный путь ROCm | `export CMAKE_PREFIX_PATH=/opt/rocm` |
| `cannot find -lroc_rocblas` | Неправильное имя target | Проверить: `find /opt/rocm -name "rocblas*.cmake"` |

---

## Definition of Done

- [ ] Все C++ тесты 8 репо зелёные (0 fail)
- [ ] Все Python тесты запускаются без ошибок
- [ ] Интеграционный тест `test_full_pipeline.py` проходит
- [ ] Baseline производительности записан в `DSP/Doc/Performance/baseline_gfx1201.md`
- [ ] Написать сессию `MemoryBank/sessions/YYYY-MM-DD_dsp_gpu_launch.md`

---

*Создан: 2026-04-12 | Обновлён: 2026-04-12 (добавлены: debian-local-dev пресет, T0.6 PYTHONPATH, исправлен T10.2 baseline, добавлены ошибки find_package/rocblas) | Автор: Кодо*
