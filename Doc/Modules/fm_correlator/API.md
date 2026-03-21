# FM Correlator — API Reference

> ROCm-only модуль для частотно-доменной корреляции ФМ-сигналов с M-последовательностями.

---

## C++ API

### Заголовки

```cpp
#include "fm_correlator.hpp"          // FMCorrelator (фасад)
#include "fm_correlator_types.hpp"    // FMCorrelatorParams, FMCorrelatorResult
```

### FMCorrelatorParams

```cpp
struct FMCorrelatorParams {
  size_t   fft_size          = 32768;       // Размер FFT (степень 2)
  int      num_shifts        = 32;          // K — циклических сдвигов ref
  int      num_signals       = 5;           // S — входных сигналов
  int      num_output_points = 2000;        // n_kg — первых точек из IFFT
  uint32_t lfsr_polynomial   = 0x00400007; // Полином LFSR
  uint32_t lfsr_seed         = 0x12345678; // Начальное состояние LFSR (!= 0)
};
```

### FMCorrelatorResult

```cpp
struct FMCorrelatorResult {
  std::vector<float> peaks;    // flat [S × K × n_kg], row-major
  int num_signals       = 0;
  int num_shifts        = 0;
  int num_output_points = 0;

  // Безопасный 3D-доступ
  float at(int signal, int shift, int point) const;
};
```

### FMCorrelator

```cpp
namespace drv_gpu_lib {

class FMCorrelator {
public:
  // Конструктор — backend не владеем, должен жить дольше FMCorrelator
  explicit FMCorrelator(IBackend* backend);

  // Применить параметры: пересоздаёт GPU-буферы и FFT-планы.
  // После SetParams() обязателен повторный PrepareReference().
  void SetParams(const FMCorrelatorParams& params);

  // CPU: генерировать M-последовательность из params_.lfsr_seed
  std::vector<float> GenerateMSequence() const;

  // CPU: генерировать M-последовательность с явным seed
  std::vector<float> GenerateMSequence(uint32_t seed) const;

  // Загрузить ref[N] float: H2D → cyclic_shifts → C2C FFT
  // ref.size() должен == params_.fft_size
  void PrepareReference(const std::vector<float>& ref);

  // Сгенерировать M-seq из params_.lfsr_seed и подготовить
  void PrepareReference();

  // Корреляция внешних сигналов: inp[S×N] → пики [S×K×n_kg]
  // inp — flat row-major [S × N] float
  FMCorrelatorResult Process(const std::vector<float>& inp);

  // Тестовый паттерн (без H2D для inp): GPU генерирует circshift(ref, s*shift_step)
  // Пик сигнала s, сдвига k находится в позиции (s*shift_step - k + N) % N
  FMCorrelatorResult RunTestPattern(int shift_step = 2);

  // Авто-батчинг для total_signals > params_.num_signals
  FMCorrelatorResult ProcessWithBatching(const std::vector<float>& inp,
                                          int total_signals);

  const FMCorrelatorParams& GetParams() const;
};

} // namespace drv_gpu_lib
```

### Пример C++

```cpp
#include "fm_correlator.hpp"

// backend — уже инициализированный ROCmBackend
drv_gpu_lib::FMCorrelator corr(backend);

drv_gpu_lib::FMCorrelatorParams p;
p.fft_size          = 32768;
p.num_shifts        = 32;
p.num_signals       = 5;
p.num_output_points = 2000;
corr.SetParams(p);

corr.PrepareReference();  // M-seq из lfsr_seed → GPU (один раз)

auto result = corr.RunTestPattern(/*shift_step=*/2);
// result.at(signal=0, shift=0, point=0) — пик сигнала 0 на сдвиге 0

// Или с внешними данными:
std::vector<float> inp(p.num_signals * p.fft_size);
// ... заполнить inp ...
auto result2 = corr.Process(inp);
```

---

## Python API

### Класс `gpuworklib.FMCorrelatorROCm`

```python
import gpuworklib
import numpy as np

ctx = gpuworklib.ROCmGPUContext(0)
corr = gpuworklib.FMCorrelatorROCm(ctx)
```

### Методы

#### `set_params`

```python
corr.set_params(
    fft_size: int = 32768,
    num_shifts: int = 32,
    num_signals: int = 5,
    num_output_points: int = 2000,
    polynomial: int = 0x00400007,
    seed: int = 0x12345678,
)
```

Устанавливает параметры коррелятора. Пересоздаёт GPU-буферы и FFT-планы.

#### `generate_msequence`

```python
seq = corr.generate_msequence(seed: int = 0x12345678) -> np.ndarray
# Returns: float32 [N], значения {+1.0, -1.0}
```

Генерирует M-последовательность на CPU (LFSR).

#### `prepare_reference`

```python
corr.prepare_reference()
```

Генерирует M-seq из seed (заданного в `set_params`) и загружает на GPU.

#### `prepare_reference_from_data`

```python
corr.prepare_reference_from_data(ref: np.ndarray)
# ref: float32 [N]
```

Загружает внешний массив как эталонный сигнал.

#### `process`

```python
peaks = corr.process(input_signals: np.ndarray) -> np.ndarray
# input_signals: float32 [S, N] или [S*N]
# Returns: float32 [S, K, n_kg] — корреляционные пики
```

Корреляция внешних входных сигналов.

#### `run_test_pattern`

```python
peaks = corr.run_test_pattern(shift_step: int = 2) -> np.ndarray
# Returns: float32 [S, K, n_kg]
```

GPU-генерация тестовых сигналов (`circshift(ref, s*shift_step)`) без H2D передачи.
Пик сигнала `s`, сдвига `k` ожидается в позиции `(s*shift_step - k) % N`.

### Полный пример Python

```python
import gpuworklib
import numpy as np

ctx = gpuworklib.ROCmGPUContext(0)
corr = gpuworklib.FMCorrelatorROCm(ctx)

# Параметры
corr.set_params(fft_size=32768, num_shifts=32, num_signals=5,
                num_output_points=2000)

# Вариант 1: тестовый паттерн (быстрее, данные не покидают GPU)
corr.prepare_reference()
peaks = corr.run_test_pattern(shift_step=2)  # [5, 32, 2000]
print(f"Пик сигнала 0, сдвига 0: {peaks[0, 0, 0]:.4f}")

# Вариант 2: внешние данные
ref = corr.generate_msequence(seed=0x12345678)
corr.prepare_reference_from_data(ref)

# Создаём S сигналов — циклические сдвиги ref
S = 5
signals = np.stack([np.roll(ref, s * 2) for s in range(S)]).astype(np.float32)
peaks2 = corr.process(signals)  # [5, 32, 2000]

# Вариант 3: авто-батчинг (через C++ API)
# используйте ProcessWithBatching для total_signals > num_signals
```

---

## Тесты

| Файл | Тест | Что проверяет |
|------|------|---------------|
| `tests/test_fm_msequence.hpp` | `run_test_msequence` | ±1, ~50/50, разные seed, воспроизводимость |
| `tests/test_fm_basic.hpp` | `run_test_autocorrelation` | SNR автокорреляции > 10 |
| `tests/test_fm_basic.hpp` | `run_test_basic_pipeline` | форма результата [S×K×n_kg] |
| `tests/test_fm_basic.hpp` | `run_test_shift_pattern` | пик в ожидаемой позиции |
| `Python_test/fm_correlator/test_fm_correlator.py` | `TestMSequence` | 4 NumPy-теста (без GPU) |
| `Python_test/fm_correlator/test_fm_correlator.py` | `TestCorrelationNumpy` | 4 NumPy-теста корреляции |
| `Python_test/fm_correlator/test_fm_correlator.py` | `TestFMCorrelatorROCm` | 4 GPU-теста (skip без ROCm) |

Запуск тестов:

```bash
# C++
./GPUWorkLib fm_correlator

# Python (NumPy-only, без GPU)
python3 -m python Python_test/fm_correlator/test_fm_correlator.py

# Python (с GPU после пересборки)
cmake .. -DENABLE_ROCM=ON -DBUILD_PYTHON=ON
cmake --build . -j$(nproc)
PYTHONPATH=build/python python3 -m python run_tests.py -m fm_correlator
```

---

## Связанные файлы

| Файл | Описание |
|------|----------|
| `include/fm_correlator.hpp` | Фасад — публичный API |
| `include/fm_correlator_types.hpp` | `FMCorrelatorParams`, `FMCorrelatorResult` |
| `include/fm_correlator_processor_rocm.hpp` | ROCm-процессор (детали реализации) |
| `python/py_fm_correlator_rocm.hpp` | Python binding |
| `Doc/Modules/fm_correlator/Full.md` | Полная документация (математика, pipeline, kernels) |
| `Doc/Modules/fm_correlator/Quick.md` | Краткий справочник |

---

*Создано: 2026-03-10*
