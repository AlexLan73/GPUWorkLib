# 📝 ScenarioBuilder — Спецификация

> **Модуль**: `Python_test/strategies/scenario_builder.py`
> **Статус**: 🟡 WIP
> **Платформы**: Python (numpy), CPU-side генерация для GPU pipeline
> **Автор**: Alex + Кодо
> **Создано**: 2026-03-08
> **Обновлено**: 2026-03-08

---

## 🎯 Назначение

Python-генератор тестовых сценариев для модуля `strategies` (AntennaProcessor).
Создаёт физически корректные сигнальные матрицы `S [n_ant × n_samples]` и весовые
матрицы `W [n_ant × n_ant]` для тестирования beamforming pipeline на GPU.

**Ключевая идея**: задаём физику решётки (шаг антенн, углы, скорость распространения)
вместо прямого задания задержек в секундах.

---

## 📋 Требования

### Функциональные
- [x] REQ-001: ULA геометрия — шаг d, n_ant, скорость c → задержки τ
- [x] REQ-002: ЛЧМ сигнал — f0, fdev, amplitude, phase → complex samples
- [x] REQ-003: Несколько целей — суммирование ЛЧМ с разных углов
- [x] REQ-004: ЛЧМ помеха (chirp jammer) — отдельный источник с другого угла
- [x] REQ-005: AWGN шум — комплексный гауссов шум с заданной sigma
- [x] REQ-006: Матрица весов W — Delay-and-sum beamforming
- [x] REQ-007: Совместимость с FormSignalGeneratorROCm (та же формула ЛЧМ)
- [x] REQ-008: Готовые сценарии-фабрики для типовых случаев

### Нефункциональные
- [x] NFR-001: Чистый Python/numpy — без GPU зависимостей
- [x] NFR-002: Выход complex64 — совместимо с hipBLAS CGEMM
- [x] NFR-003: Воспроизводимость — фиксированный seed для шума

---

## 🔧 API

### Классы

```python
@dataclass
class ULAGeometry:
    """Uniform Linear Array — геометрия антенной решётки"""
    n_ant: int              # количество антенн
    d_ant_m: float          # шаг между антеннами (м)
    c: float = 3e8          # скорость распространения (м/с), РЛС по умолчанию

    def compute_delays(self, theta_deg: float) -> np.ndarray:
        """Задержки [n_ant] в секундах для угла прихода theta"""

    @staticmethod
    def from_lambda_half(n_ant: int, carrier_freq_hz: float, c: float = 3e8):
        """Создать решётку с d = λ/2 для заданной несущей"""


@dataclass
class EmitterSignal:
    """Один источник излучения (цель или помеха)"""
    theta_deg: float        # угол прихода (градусы от нормали)
    f0_hz: float            # несущая частота (Гц)
    fdev_hz: float = 0.0    # девиация ЛЧМ (0 = CW)
    amplitude: float = 1.0  # амплитуда
    phase_rad: float = 0.0  # начальная фаза (рад)
    label: str = ""         # метка для логов


class ScenarioBuilder:
    """Строитель тестовых сценариев для AntennaProcessor"""

    def __init__(self, array: ULAGeometry, fs: float, n_samples: int)
    def add_target(self, theta_deg, f0_hz, fdev_hz=0, ...) -> 'ScenarioBuilder'
    def add_jammer(self, theta_deg, f0_hz, fdev_hz=0, ...) -> 'ScenarioBuilder'
    def set_noise(self, sigma: float, seed: int = 42) -> 'ScenarioBuilder'
    def build(self) -> dict   # → {'S': ndarray, 'targets': [...], ...}
    def generate_weight_matrix(self, steer_theta_deg, steer_freq_hz=None) -> np.ndarray
```

### Готовые сценарии

```python
def make_single_target(n_ant=8, theta_deg=30, f0_hz=2e6, fdev_hz=1e6,
                       noise_sigma=0.1) -> dict

def make_target_and_jammer(n_ant=8, target_theta=30, jammer_theta=-20,
                           noise_sigma=0.1) -> dict

def make_multi_target(n_ant=8, thetas=[20, 45], f0s=[2e6, 3e6],
                      noise_sigma=0.1) -> dict
```

---

## 🏗️ Архитектура

### Физика задержек (ULA для РЛС)

```
Антенна 0    Антенна 1    Антенна 2    ...    Антенна N-1
    |            |            |                    |
    |<--- d ---->|<--- d ---->|                    |
    |            |            |                    |

Волновой фронт приходит под углом θ к нормали:

    τ_i = i · d · sin(θ) / c

    где c = 3·10⁸ м/с (РЛС)
```

### Формула ЛЧМ (совместимая с C++ FormSignalGeneratorROCm)

```
Ti = n_samples / fs                         (длительность сигнала)
t_d = t - τ_ant                             (задержанное время)

phase = 2π·f0·t_d + π·(fdev/Ti)·(t_d - Ti/2)² + φ

x[ant, sample] = A · (1/√2) · exp(j · phase)   при 0 ≤ t_d < Ti
               = 0                                иначе
```

### Суммирование источников

```
S[ant, sample] = Σ target_signals + Σ jammer_signals + noise

noise = σ/√2 · (randn + j·randn)
```

### Матрица весов W (Delay-and-sum)

```
W[beam, ant] = (1/√N) · exp(-j·2π·f_steer·τ_ant(θ_steer))
```

### Pipeline интеграция

```
Python:                          GPU (C++):
ScenarioBuilder.build()
  → S [n_ant × n_samples]  ──→  proc.step_0_prepare_input(S, W)
  → W [n_ant × n_ant]      ──→    ↓
                                 GEMM(W, S) → FFT → peaks → result
```

---

## 📊 Тестовые сценарии

| Сценарий | Цели | Помехи | Шум | Назначение |
|----------|------|--------|-----|------------|
| single_target | 1 ЛЧМ 30° | — | σ=0.1 | Базовый beamforming |
| target_jammer | 1 ЛЧМ 30° | 1 ЛЧМ -20° | σ=0.1 | Подавление помехи |
| multi_target | 2 ЛЧМ 20°,45° | — | σ=0.1 | Разделение целей |

---

## 🔗 Зависимости

- `numpy` — генерация сигналов, FFT, линейная алгебра
- Совместимость с `AntennaProcessorTest` (pybind11 bindings)
- Формула ЛЧМ из `form_params.hpp` (FormSignalGeneratorROCm)

---

## 📁 Файлы

| Файл | Описание |
|------|----------|
| `Python_test/strategies/scenario_builder.py` | Основной класс |
| `Python_test/strategies/test_scenario_builder.py` | Тесты (numpy-only) |

---

---

## ✅ Полный список реализованного API

> Актуально на 2026-03-08. Всё ниже **уже работает** в `scenario_builder.py`.

### ULAGeometry
| Метод | Статус | Описание |
|-------|--------|----------|
| `compute_delays(theta_deg)` | ✅ | `τᵢ = i·d·sin(θ)/c` → [n_ant] |
| `max_unambiguous_angle(freq_hz)` | ✅ | Максимальный угол без grating lobes |
| `from_lambda_half(n_ant, freq_hz)` | ✅ | static: d = λ/2 |

### ScenarioBuilder
| Метод | Статус | Описание |
|-------|--------|----------|
| `add_target(θ, f0, fdev, A, φ)` | ✅ | Fluent API, накапливает цели |
| `add_jammer(θ, f0, fdev, A, φ)` | ✅ | Fluent API, накапливает помехи |
| `set_noise(sigma, seed)` | ✅ | AWGN с фиксированным seed |
| `build()` | ✅ | → dict {S, targets, jammers, ...} |
| `generate_weight_matrix(θ, freq)` | ✅ | W [n_ant × n_ant], одно направление |
| `generate_scan_weight_matrix(θs, freq)` | ✅ | W [n_beams × n_ant], сканирование |
| `summary()` | ✅ | Текстовое описание сценария |

### Фабрики
| Функция | Статус |
|---------|--------|
| `make_single_target(...)` | ✅ |
| `make_target_and_jammer(...)` | ✅ |
| `make_multi_target(...)` | ✅ |

---

## 🚀 Дорожная карта (Roadmap)

```
Фаза 1 (Python Reference)  ████████████████████░  95%  ← СЕЙЧАС
Фаза 2 (Визуализация)      ░░░░░░░░░░░░░░░░░░░░░   0%
Фаза 3 (C++ + GPU)         ░░░░░░░░░░░░░░░░░░░░░   0%
Фаза 4 (Адаптивные)        ░░░░░░░░░░░░░░░░░░░░░   0%
```

### Фаза 1: Python Reference ✅ (почти завершено)

**Цель**: Полностью рабочая CPU-версия для валидации GPU

| Задача | Статус |
|--------|--------|
| ULAGeometry + compute_delays | ✅ |
| EmitterSignal (CW + LFM) | ✅ |
| ScenarioBuilder (все методы) | ✅ |
| generate_scan_weight_matrix | ✅ |
| Фабрики (3 шт) | ✅ |
| test_scenario_builder.py (15+ тестов) | ✅ |
| JSON export/import сценария | ⬜ TASK-SB-02 |

### Фаза 2: Визуализация

**Цель**: Графический анализ сценариев и результатов beamforming

| Задача | ID | Приоритет |
|--------|----|-----------|
| `beam_pattern.py` — диаграмма направленности vs θ | TASK-SB-03 | P1 |
| `plot_scenario.py` — S[ant, :] временны́е диаграммы | TASK-SB-04 | P1 |
| Интеграция с `Results/Plots/strategies/` | TASK-SB-05 | P1 |
| Dear PyGui визуализация (real-time) | TASK-SB-06 | P2 |

```
beam_pattern:
  Вход: ULAGeometry + W матрица
  Выход: plot |W · a(θ)| vs θ, от -90° до +90°
  Файл: Results/Plots/strategies/beam_pattern_*.png
```

### Фаза 3: C++ AntennaProcessor + GPU

**Цель**: GPU реализация pipeline с Python биндингами

**Новый модуль**: `modules/strategies/`

```
modules/strategies/
├── AntennaProcessorROCm.h        ← основной класс
├── AntennaProcessorROCm.cpp      ← hipBLAS + hipFFT + lch_farrow
├── AntennaProcessorOpenCL.h      ← OpenCL backend (будущее)
├── tests/
│   ├── test_antenna_processor.hpp
│   ├── test_antenna_benchmark.hpp
│   └── all_test.hpp
└── python/
    └── antenna_processor_bindings.cpp  ← pybind11
```

**Планируемый C++ API**:
```cpp
class AntennaProcessorROCm {
public:
    AntennaProcessorROCm(DrvGPU& ctx, int n_ant, int n_samples);

    // Загрузка входных данных
    void load_input(const std::complex<float>* S,    // [n_ant × n_samples]
                    const std::complex<float>* W);   // [n_ant × n_ant]

    // Pipeline B: Farrow выравнивание (через LchFarrowROCm)
    void apply_farrow(const float* delays_samples);  // [n_ant]

    // GEMM: X = W @ S (hipBLAS CGEMM)
    void gemm();

    // FFT + windowing (hipFFT)
    void fft_with_window();

    // Детектирование пиков
    void find_peaks(int n_top = 5);

    // Результаты → Python
    py::array_t<float>   get_magnitudes();   // [n_beams, nFFT]
    py::array_t<float>   get_peaks_freq();   // [n_peaks]
    py::array_t<float>   get_peaks_mag();    // [n_peaks]
};
```

**Python binding** (`AntennaProcessorTest`):
```python
proc = AntennaProcessorTest(gpu_ctx, n_ant=8, n_samples=8000)
proc.load_input(S, W)
proc.apply_farrow(delays)
proc.gemm()
proc.fft_with_window()
proc.find_peaks()
peaks = proc.get_peaks_freq()
```

**Зависимости GPU модуля**:
```
AntennaProcessorROCm
  ├── DrvGPU           ← контекст, очереди, буферы
  ├── LchFarrowROCm    ← fractional delay
  ├── hipBLAS          ← CGEMM W @ S
  ├── hipFFT           ← FFT после GEMM
  └── GPUProfiler      ← профилирование шагов
```

### Фаза 4: Адаптивные алгоритмы

**Цель**: MVDR beamformer — оптимальное подавление помех

```
MVDR (Minimum Variance Distortionless Response):

  R = (1/K) · S · S†                    [n_ant × n_ant] матрица ковариации
  a(θ) = steering vector для угла θ      [n_ant]

  w_mvdr = R⁻¹ · a / (a† · R⁻¹ · a)   оптимальные веса

  Преимущество vs delay-and-sum: -15..30 dB подавление помех
```

| Задача | ID |
|--------|----|
| `MVDRBeamformer` класс (Python reference) | TASK-SB-10 |
| Тесты: SNR gain vs delay-and-sum | TASK-SB-11 |
| GPU реализация (hipBLAS potrs) | TASK-SB-12 |

---

## 📋 Задачи (Backlog)

### P0 — Критические (блокируют фазу 2)

| ID | Задача | Описание |
|----|--------|----------|
| TASK-SB-01 | Запустить тесты | `python Python_test/strategies/test_scenario_builder.py` — убедиться, что все pass |
| TASK-SB-02 | JSON export/import | `builder.to_json(path)` / `ScenarioBuilder.from_json(path)` |

### P1 — Важные (фаза 2)

| ID | Задача | Файл |
|----|--------|------|
| TASK-SB-03 | beam_pattern.py | `Python_test/strategies/beam_pattern.py` |
| TASK-SB-04 | plot_scenario.py | `Python_test/strategies/plot_scenario.py` |
| TASK-SB-05 | Docs: generate_scan_weight_matrix | Добавить в раздел API этой спеки |

### P2 — Расширения (фаза 3+)

| ID | Задача | Описание |
|----|--------|----------|
| TASK-SB-06 | URA геометрия | 2D антенная решётка (Uniform Rectangular Array) |
| TASK-SB-07 | Доплеровский сдвиг | EmitterSignal + velocity → freq shift |
| TASK-SB-08 | Многолучевое распространение | Эхо, задержанные копии сигнала |
| TASK-SB-09 | C++ AntennaProcessorROCm | Модуль `modules/strategies/` |
| TASK-SB-10 | MVDR reference | Python класс MVDRBeamformer |

---

## ✅ Критерии успеха

| Фаза | Критерий |
|------|----------|
| Фаза 1 | Все тесты `test_scenario_builder.py` pass ✅ |
| Фаза 2 | Графики `beam_pattern*.png` сохраняются в `Results/Plots/strategies/` |
| Фаза 3 | GPU пики совпадают с Python ±1% по магнитуде и ±freq_res по частоте |
| Фаза 4 | MVDR подавляет помеху не менее чем на 15 dB vs delay-and-sum |

---

## 🔮 Будущие расширения (не в ближайших планах)

- **URA** — Uniform Rectangular Array: 2D решётка, угломестный и азимутальный beam
- **Конформные решётки** — цилиндр, сфера
- **Когерентный MIMO** — несколько источников, spatial multiplexing
- **Пространственно-временная обработка** (STAP) — комбинация Doppler + beamforming
- **Deep learning beamforming** — NN весовые матрицы

---

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-03-08 | Alex + Кодо | Создание спецификации, первая реализация |
| 2026-03-08 | Кодо | Добавлены: полный список API, дорожная карта (4 фазы), backlog задач, C++ план, критерии успеха |
