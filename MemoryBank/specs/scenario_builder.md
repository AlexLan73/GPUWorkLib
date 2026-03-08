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

## 📝 История изменений

| Дата | Автор | Изменение |
|------|-------|-----------|
| 2026-03-08 | Alex + Кодо | Создание спецификации, первая реализация |
