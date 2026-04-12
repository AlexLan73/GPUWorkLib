# 🐍 Фаза 3b: Python API Migration

> **Индекс**: [`TASK_Modular_Architecture_INDEX.md`](TASK_Modular_Architecture_INDEX.md)
> **Статус**: ⬜ BACKLOG
> **Зависимость**: Фаза 3 должна быть ✅ DONE
> **Платформа**: Windows (редактирование Python файлов, GPU не нужен)
> **Результат**: 8 отдельных `.pyd` модулей вместо одного `gpuworklib.pyd`, все тесты работают

---

## Суть проблемы

**Текущее состояние (GPUWorkLib):**
```python
import gpuworklib as gl          # один .pyd файл

ctx = gl.GPUContext(config_path)
fft = gl.FFTProcessor(ctx, ...)
stats = gl.StatisticsProcessor(ctx, ...)
```

**Целевое состояние (DSP-GPU):**
```python
import dsp_core as core          # 8 отдельных .pyd файлов
import dsp_spectrum as spectrum
import dsp_stats as stats
import dsp_linalg as linalg
import dsp_signal_generators as sig_gen
import dsp_heterodyne as heterodyne
import dsp_radar as radar
import dsp_strategies as strategies

ctx  = core.DrvGPU(config_path)
fft  = spectrum.FFTProcessor(ctx, ...)
stat = stats.StatisticsProcessor(ctx, ...)
```

**Файлы которые сломаются без этой фазы:**
- `DSP/Python/common/gpu_loader.py` — импортирует `gpuworklib`
- Все тесты в `DSP/Python/*/` — используют старые имена

---

## Стратегия миграции

**Вариант A (рекомендуется): Shim-модуль обратной совместимости**

Создать `DSP/Python/gpuworklib.py` — тонкая обёртка, реэкспортирует из 8 модулей:
```python
# gpuworklib.py — shim для обратной совместимости
from dsp_core import DrvGPU as GPUContext
from dsp_spectrum import FFTProcessor, FirFilter, LchFarrow
from dsp_stats import StatisticsProcessor
# ... и так далее
```

Преимущество: старый Python-код работает без изменений. Потом постепенно обновляем.

**Вариант B: Big-bang замена**

Сразу заменить все импорты в тестах. Быстро, но рискованно.

---

## Чеклист

### 🔧 Шаг 1: Разбить gpu_worklib_bindings.cpp на 8 файлов

- [ ] **P1.1** `core/python/dsp_core_module.cpp` — только DrvGPU, GPUContext:
  ```cpp
  PYBIND11_MODULE(dsp_core, m) {
    py::class_<dsp::DrvGPU>(m, "DrvGPU")
      .def(py::init<std::string>())
      .def("get_device_count", &dsp::DrvGPU::GetDeviceCount)
      ...;
  }
  ```
- [ ] **P1.2** `spectrum/python/dsp_spectrum_module.cpp` — FFTProcessor, FirFilter, LchFarrow:
  ```cpp
  PYBIND11_MODULE(dsp_spectrum, m) {
    py::class_<dsp::spectrum::FFTProcessor>(m, "FFTProcessor") ...;
    py::class_<dsp::spectrum::FirFilter>(m, "FirFilter") ...;
    py::class_<dsp::spectrum::LchFarrow>(m, "LchFarrow") ...;
  }
  ```
- [ ] **P1.3** `stats/python/dsp_stats_module.cpp` — StatisticsProcessor:
  ```cpp
  PYBIND11_MODULE(dsp_stats, m) {
    py::class_<dsp::stats::StatisticsProcessor>(m, "StatisticsProcessor") ...;
  }
  ```
- [ ] **P1.4** `linalg/python/dsp_linalg_module.cpp` — VectorAlgebra, Capon:
  ```cpp
  PYBIND11_MODULE(dsp_linalg, m) {
    py::class_<dsp::linalg::VectorAlgebra>(m, "VectorAlgebra") ...;
    py::class_<dsp::linalg::Capon>(m, "Capon") ...;
  }
  ```
- [ ] **P1.5** `signal_generators/python/dsp_signal_generators_module.cpp`:
  ```cpp
  PYBIND11_MODULE(dsp_signal_generators, m) {
    py::class_<dsp::signal_generators::CwGenerator>(m, "CwGenerator") ...;
    py::class_<dsp::signal_generators::LfmGenerator>(m, "LfmGenerator") ...;
    py::class_<dsp::signal_generators::NoiseGenerator>(m, "NoiseGenerator") ...;
    py::class_<dsp::signal_generators::FormSignalGenerator>(m, "FormSignalGenerator") ...;
  }
  ```
- [ ] **P1.6** `heterodyne/python/dsp_heterodyne_module.cpp`:
  ```cpp
  PYBIND11_MODULE(dsp_heterodyne, m) {
    py::class_<dsp::heterodyne::HeterodyneDechirp>(m, "HeterodyneDechirp") ...;
  }
  ```
- [ ] **P1.7** `radar/python/dsp_radar_module.cpp`:
  ```cpp
  PYBIND11_MODULE(dsp_radar, m) {
    py::class_<dsp::radar::RangeAngleProcessor>(m, "RangeAngleProcessor") ...;
    py::class_<dsp::radar::FmCorrelator>(m, "FmCorrelator") ...;
  }
  ```
- [ ] **P1.8** `strategies/python/dsp_strategies_module.cpp`:
  ```cpp
  PYBIND11_MODULE(dsp_strategies, m) {
    py::class_<dsp::strategies::AntennaProcessor>(m, "AntennaProcessor") ...;
  }
  ```

---

### 🔧 Шаг 2: CMake для Python биндингов (каждый репо)

- [ ] **P2.1** В каждом `{repo}/python/CMakeLists.txt`:
  ```cmake
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(dsp_spectrum dsp_spectrum_module.cpp)
  target_link_libraries(dsp_spectrum PRIVATE DspSpectrum::DspSpectrum)
  install(TARGETS dsp_spectrum DESTINATION ${CMAKE_INSTALL_PREFIX})
  ```
- [ ] **P2.2** Собрать и установить: `cmake --install build --prefix DSP/Python/lib`
- [ ] **P2.3** Проверить что в `DSP/Python/lib/` появились все 8 `.pyd` (Windows) или `.so` (Linux)

---

### 🔧 Шаг 3: Создать shim gpuworklib.py

- [ ] **P3.1** Создать `DSP/Python/gpuworklib.py`:
  ```python
  """
  Shim модуль для обратной совместимости с GPUWorkLib.
  Реэкспортирует классы из 8 модульных .pyd файлов.
  Постепенно заменяйте `import gpuworklib` на конкретные импорты.
  """
  try:
      from dsp_core import DrvGPU
      DrvGPU as GPUContext  # старое имя
      from dsp_spectrum import FFTProcessor, FirFilter, IirFilter, LchFarrow
      from dsp_stats import StatisticsProcessor
      from dsp_linalg import VectorAlgebra, Capon
      from dsp_signal_generators import (
          CwGenerator, LfmGenerator, NoiseGenerator,
          ScriptGenerator, FormSignalGenerator
      )
      from dsp_heterodyne import HeterodyneDechirp
      from dsp_radar import RangeAngleProcessor, FmCorrelator
      from dsp_strategies import AntennaProcessor
  except ImportError as e:
      raise ImportError(
          f"DSP modules not found: {e}\n"
          "Run: cmake --install build --prefix DSP/Python/lib\n"
          "Then: export PYTHONPATH=DSP/Python/lib:$PYTHONPATH"
      ) from e
  ```

---

### 🔧 Шаг 4: Рефакторинг gpu_loader.py

- [ ] **P4.1** Обновить `DSP/Python/common/gpu_loader.py`:
  ```python
  import dsp_core as _core

  class GPULoader:
      """Singleton: загружает GPU контекст из dsp_core."""
      _instance = None

      @classmethod
      def get_instance(cls, config_path="configGPU.json"):
          if cls._instance is None:
              cls._instance = _core.DrvGPU(config_path)
          return cls._instance

      @classmethod
      def reset(cls):
          cls._instance = None
  ```
- [ ] **P4.2** Проверить все тесты которые используют `GPULoader` — должны работать без изменений

---

### 🔧 Шаг 5: Обновить Python тесты (постепенно)

- [ ] **P5.1** `DSP/Python/spectrum/test_fft.py`:
  ```python
  # БЫЛО: import gpuworklib as gl; fft = gl.FFTProcessor(...)
  # СТАЛО:
  import dsp_core as core
  import dsp_spectrum as spectrum
  ctx = core.DrvGPU(config_path)
  fft = spectrum.FFTProcessor(ctx, ...)
  ```
- [ ] **P5.2** `DSP/Python/stats/test_statistics.py` — аналогично
- [ ] **P5.3** `DSP/Python/linalg/test_vector_algebra.py` — аналогично
- [ ] **P5.4** `DSP/Python/signal_generators/` — аналогично
- [ ] **P5.5** `DSP/Python/heterodyne/test_dechirp.py` — аналогично
- [ ] **P5.6** `DSP/Python/radar/` — аналогично
- [ ] **P5.7** `DSP/Python/strategies/` — аналогично

---

### 🔧 Шаг 6: Интеграционный тест

- [ ] **P6.1** Создать `DSP/Python/integration/test_full_pipeline.py`:
  ```python
  """Интеграционный тест: core → spectrum → signal_generators → heterodyne → strategies"""
  import dsp_core as core
  import dsp_spectrum as spectrum
  import dsp_signal_generators as sig_gen
  import dsp_heterodyne as heterodyne
  import dsp_strategies as strategies

  def test_full_pipeline():
      ctx = core.DrvGPU("configGPU.json")
      # ... pipeline тест
  ```
- [ ] **P6.2** Запустить: `python DSP/Python/integration/test_full_pipeline.py` — OK ✅

---

## Definition of Done

- [ ] 8 отдельных `.pyd` / `.so` файлов в `DSP/Python/lib/`
- [ ] `gpuworklib.py` shim существует и работает (обратная совместимость)
- [ ] `gpu_loader.py` использует `dsp_core` напрямую
- [ ] Все Python тесты запускаются без ошибок
- [ ] Интеграционный тест `test_full_pipeline.py` проходит
- [ ] Можно переходить к Фазе 4

---

*Создан: 2026-04-12 | Автор: Кодо*
