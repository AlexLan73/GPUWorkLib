# TASK Python-04: strategies/signal_factory.py — ISignalSource + Factory

**Статус**: 🔲 TODO
**Приоритет**: 🟡 ВЫСОКИЙ — нужен для TASK_Python_06
**Файл**: `Python_test/strategies/signal_factory.py` (создать новый)
**Стиль**: ООП, SOLID, GRASP, GoF — обязательно!

---

## 🎯 Цель

Создать фабрику сигналов: 5 вариантов (V1–V5), выбираемых до старта теста.

Паттерны:
- **Strategy (GoF)**: `ISignalSource` — интерфейс, каждая реализация = один вариант
- **Factory Method (GoF)**: `SignalSourceFactory.create(variant)` — создаёт нужную реализацию
- **OCP (SOLID)**: добавить V6 = новый класс, ничего не трогать в существующем коде

---

## 📖 Что прочитать перед реализацией

- `Python_test/strategies/scenario_builder.py` — `ULAGeometry`, `WeightGenerator` аналог
- `Python_test/strategies/farrow_delay.py` — `FarrowDelay` для V3/V4
- `Python_test/common/gpu_context.py` — `GPUContextManager`
- `MemoryBank/specs/python_test_refactoring.md` — раздел "5 вариантов сигналов"
- `modules/strategies/include/weight_generator.hpp` — формулы delay_and_sum

---

## ⚠️ Параметры тестового сценария

```python
# Фиксированные для всех вариантов (дефолты):
N_ANT     = 5
N_SAMPLES = 8000
FS        = 12e6    # Гц
F0        = 2e6     # Гц (CW тон)
TAU_STEP  = 100e-6  # сек (задержка между антеннами, для V3/V4)
N_FFT     = 8192    # следующая степень 2 >= N_SAMPLES
SNR_DB    = 20.0    # дБ (для V2/V4)
```

---

## 🏗️ Реализация

### `SignalVariant` enum

```python
from enum import Enum

class SignalVariant(Enum):
    """Выбор сценария перед стартом теста.

    Information Expert (GRASP): знает параметры каждого варианта.
    """
    V1_CW_CLEAN        = 1  # CW без шума,  W = Identity
    V2_CW_NOISE        = 2  # CW + AWGN,    W = Identity
    V3_CW_PHASE_DELAY  = 3  # CW + задержки, W = delay_and_sum, без шума
    V4_CW_PHASE_NOISE  = 4  # CW + задержки, W = delay_and_sum, + AWGN
    V5_FROM_FILE       = 5  # Загрузка из файла → GPU (заглушка)
```

### Dataclass конфигурации

```python
@dataclass
class SignalConfig:
    """Конфигурация тестового сигнала."""
    n_ant    : int   = 5
    n_samples: int   = 8000
    fs       : float = 12e6
    f0       : float = 2e6
    tau_step : float = 100e-6    # для V3/V4
    snr_db   : float = 20.0      # для V2/V4
    n_fft    : int   = 8192
    file_path: str   = ""        # для V5
```

### `SignalData` — результат генерации

```python
@dataclass
class SignalData:
    """Результат генерации: GPU-указатели + CPU-эталоны.

    Creator (GRASP): создаётся ISignalSource.generate().
    """
    d_S  : object          # GPU pointer (opaque, из gpuworklib)
    d_W  : object          # GPU pointer (opaque)
    S_ref: np.ndarray      # [n_ant, n_samples] complex64  CPU-эталон
    W_ref: np.ndarray      # [n_ant, n_ant] complex64      CPU-эталон
    cfg  : SignalConfig     # параметры которые использовались
    variant: SignalVariant  # какой вариант сгенерирован
```

### `ISignalSource` — интерфейс (Strategy)

```python
class ISignalSource(ABC):
    """Strategy: источник тестового сигнала.

    Каждая реализация — отдельный вариант V1..V5.
    Интерфейс стабилен — добавление V6 не ломает существующий код (OCP).
    """

    @abstractmethod
    def generate(self, ctx, cfg: SignalConfig) -> SignalData:
        """Создать сигнал на GPU + CPU-эталон.

        Args:
            ctx: ROCmGPUContext (из GPUContextManager.get_rocm())
            cfg: конфигурация сигнала

        Returns:
            SignalData с d_S, d_W на GPU и S_ref, W_ref на CPU
        """
        ...
```

### V1 — CW без шума, W = Identity

```python
class GpuCwCleanSignalSource(ISignalSource):
    """V1: CW без шума, весовая матрица = Identity.

    GEMM тривиален: X = I @ S = S.
    Используется для проверки базового pipeline без алгоритмической нагрузки.
    """

    def generate(self, ctx, cfg: SignalConfig) -> SignalData:
        gw = GPULoader.get()

        # CPU эталон (complex64)
        t = np.arange(cfg.n_samples, dtype=np.float32) / cfg.fs
        cw = np.exp(1j * 2 * np.pi * cfg.f0 * t).astype(np.complex64)
        S_ref = np.tile(cw, (cfg.n_ant, 1))  # [n_ant, n_samples]
        W_ref = np.eye(cfg.n_ant, dtype=np.complex64)

        # GPU upload через gpuworklib
        d_S = ctx.upload_complex64(S_ref.ravel())  # проверить реальный API!
        d_W = ctx.upload_complex64(W_ref.ravel())

        return SignalData(d_S=d_S, d_W=d_W,
                          S_ref=S_ref, W_ref=W_ref,
                          cfg=cfg, variant=SignalVariant.V1_CW_CLEAN)
```

### V2 — CW + AWGN, W = Identity

```python
class GpuCwNoiseSignalSource(ISignalSource):
    """V2: CW + белый гауссов шум (AWGN), W = Identity.

    S[k,n] = exp(j*2π*f0*n/fs) + noise[k,n]
    Шум добавляется на CPU перед загрузкой на GPU.
    SNR вычисляется через cfg.snr_db.
    """

    def generate(self, ctx, cfg: SignalConfig) -> SignalData:
        rng = np.random.default_rng(seed=42)  # воспроизводимость!
        t = np.arange(cfg.n_samples, dtype=np.float32) / cfg.fs
        cw = np.exp(1j * 2 * np.pi * cfg.f0 * t).astype(np.complex64)
        S_ref = np.tile(cw, (cfg.n_ant, 1))

        # Добавить шум с нужным SNR
        signal_power = 1.0  # |exp(jx)|^2 = 1
        noise_power = signal_power / (10 ** (cfg.snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)  # / 2 потому что I+Q
        noise = (rng.normal(0, noise_std, S_ref.shape) +
                 1j * rng.normal(0, noise_std, S_ref.shape)).astype(np.complex64)
        S_ref = S_ref + noise

        W_ref = np.eye(cfg.n_ant, dtype=np.complex64)

        d_S = ctx.upload_complex64(S_ref.ravel())
        d_W = ctx.upload_complex64(W_ref.ravel())

        return SignalData(d_S=d_S, d_W=d_W,
                          S_ref=S_ref, W_ref=W_ref,
                          cfg=cfg, variant=SignalVariant.V2_CW_NOISE)
```

### V3 — CW + фазовая задержка, W = delay_and_sum

```python
class GpuCwDelayedSignalSource(ISignalSource):
    """V3: CW с межантенной задержкой, W = delay_and_sum (без шума).

    Задержка k-й антенны: tau[k] = tau_step * k
    Сигнал: S[k,n] = exp(j*2π*f0*(n/fs - tau[k]))
    Весовая матрица W компенсирует эти задержки (beamforming).
    """

    def generate(self, ctx, cfg: SignalConfig) -> SignalData:
        # Задержки антенн
        tau = np.arange(cfg.n_ant, dtype=np.float32) * cfg.tau_step  # [n_ant]
        t = np.arange(cfg.n_samples, dtype=np.float32) / cfg.fs       # [n_samples]

        # S[k,n] = exp(j*2π*f0*(t[n] - tau[k]))
        phase = 2 * np.pi * cfg.f0 * (t[np.newaxis, :] - tau[:, np.newaxis])
        S_ref = np.exp(1j * phase).astype(np.complex64)  # [n_ant, n_samples]

        # Весовая матрица delay_and_sum (формула из WeightGenerator)
        # W[b,k] = exp(-j*2π*f0*tau[k]) / sqrt(n_ant)
        W_ref = np.exp(-1j * 2 * np.pi * cfg.f0 * tau) / np.sqrt(cfg.n_ant)
        # Расширить до квадратной матрицы [n_ant x n_ant]:
        # Каждый луч b имеет одинаковые веса (одна целевая частота)
        W_ref = np.tile(W_ref[np.newaxis, :], (cfg.n_ant, 1)).astype(np.complex64)

        d_S = ctx.upload_complex64(S_ref.ravel())
        d_W = ctx.upload_complex64(W_ref.ravel())

        return SignalData(d_S=d_S, d_W=d_W,
                          S_ref=S_ref, W_ref=W_ref,
                          cfg=cfg, variant=SignalVariant.V3_CW_PHASE_DELAY)
```

### V4 — CW + задержка + шум

```python
class GpuCwPhaseNoiseSignalSource(ISignalSource):
    """V4: V3 + AWGN шум. Полный реальный сценарий."""

    def generate(self, ctx, cfg: SignalConfig) -> SignalData:
        # Взять V3 за основу
        v3_source = GpuCwDelayedSignalSource()
        data = v3_source.generate(ctx, cfg)

        # Добавить шум к S_ref
        rng = np.random.default_rng(seed=42)
        signal_power = 1.0
        noise_power = signal_power / (10 ** (cfg.snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)
        noise = (rng.normal(0, noise_std, data.S_ref.shape) +
                 1j * rng.normal(0, noise_std, data.S_ref.shape)).astype(np.complex64)
        S_ref_noisy = (data.S_ref + noise).astype(np.complex64)

        # Переписать d_S на GPU
        d_S = ctx.upload_complex64(S_ref_noisy.ravel())

        return SignalData(d_S=d_S, d_W=data.d_W,
                          S_ref=S_ref_noisy, W_ref=data.W_ref,
                          cfg=cfg, variant=SignalVariant.V4_CW_PHASE_NOISE)
```

### V5 — Загрузка из файла (заглушка)

```python
class FileSignalSource(ISignalSource):
    """V5: Загрузка данных из файла → GPU. ЗАГЛУШКА для будущего.

    Когда появятся реальные тестовые данные — реализовать полностью.
    Сейчас: бросает SkipTest если файл не найден.

    Формат файла (планируемый):
        .npz с ключами "S" (complex64) и "W" (complex64)
    """

    def generate(self, ctx, cfg: SignalConfig) -> SignalData:
        from common.runner import SkipTest

        if not cfg.file_path:
            raise SkipTest("V5: file_path не задан (реальные данные ещё не готовы)")

        import os
        if not os.path.exists(cfg.file_path):
            raise SkipTest(f"V5: файл не найден: {cfg.file_path}")

        # Загрузить из .npz
        data = np.load(cfg.file_path)
        S_ref = data["S"].astype(np.complex64)
        W_ref = data["W"].astype(np.complex64) if "W" in data else \
                np.eye(S_ref.shape[0], dtype=np.complex64)

        d_S = ctx.upload_complex64(S_ref.ravel())
        d_W = ctx.upload_complex64(W_ref.ravel())

        return SignalData(d_S=d_S, d_W=d_W,
                          S_ref=S_ref, W_ref=W_ref,
                          cfg=cfg, variant=SignalVariant.V5_FROM_FILE)
```

### `SignalSourceFactory` — фабрика

```python
class SignalSourceFactory:
    """Factory Method (GoF): создаёт ISignalSource по SignalVariant.

    Creator (GRASP): отвечает за создание правильного источника сигнала.

    Usage:
        source = SignalSourceFactory.create(SignalVariant.V3_CW_PHASE_DELAY)
        data = source.generate(ctx, cfg)
    """

    _registry: dict = {
        SignalVariant.V1_CW_CLEAN       : GpuCwCleanSignalSource,
        SignalVariant.V2_CW_NOISE       : GpuCwNoiseSignalSource,
        SignalVariant.V3_CW_PHASE_DELAY : GpuCwDelayedSignalSource,
        SignalVariant.V4_CW_PHASE_NOISE : GpuCwPhaseNoiseSignalSource,
        SignalVariant.V5_FROM_FILE      : FileSignalSource,
    }

    @classmethod
    def create(cls, variant: SignalVariant) -> ISignalSource:
        """Создать источник сигнала по варианту."""
        if variant not in cls._registry:
            raise ValueError(f"Неизвестный вариант: {variant}")
        return cls._registry[variant]()
```

---

## ⚠️ Важно: проверить реальный GPU API

Метод `ctx.upload_complex64(array_1d)` — **нужно проверить реальное имя**!

Прочитать `Python_test/common/gpu_context.py` и существующие тесты:
```bash
grep -r "upload\|hipMemcpy\|AllocateBuffer\|CreateBuffer" Python_test/strategies/ --include="*.py"
```

Возможные варианты API:
- `ctx.upload_complex64(arr)` — наш желаемый
- `ctx.allocate_and_upload(arr)` — другое имя
- `gw.hipMalloc(size); gw.hipMemcpyH2D(dst, src)` — низкоуровневый

**Используй тот API что реально есть в gpuworklib!**

---

## ✅ Критерии готовности

1. `SignalVariant` enum с 5 значениями
2. `SignalConfig` dataclass со всеми полями
3. `SignalData` dataclass (d_S, d_W, S_ref, W_ref, cfg, variant)
4. `ISignalSource` ABC с методом `generate(ctx, cfg) -> SignalData`
5. 4 конкретных реализации (V1/V2/V3/V4) + заглушка V5
6. V5 бросает `SkipTest` если файл не задан/не найден
7. `SignalSourceFactory.create(variant)` → правильный экземпляр
8. seed=42 для всех RNG → воспроизводимые результаты

---

## ❌ Что НЕ делать

- НЕ хардкодить GPU API — проверить реальный интерфейс
- НЕ делать V5 рабочим если данных нет — только `SkipTest`
- НЕ смешивать V3 и V4 логику — два отдельных класса (V4 использует V3 внутри)
- НЕ изменять `ISignalSource` интерфейс при добавлении новых вариантов
