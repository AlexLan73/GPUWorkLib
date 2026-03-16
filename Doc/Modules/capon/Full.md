# Capon (MVDR) — Полная документация

**Namespace**: `capon` | **Каталог**: `modules/capon/`
**Зависимости**: DrvGPU, **vector_algebra** (CholeskyInverterROCm), rocBLAS, rocSOLVER, hiprtc
**Backend**: ROCm only (`ENABLE_ROCM=1`, Linux + AMD GPU)

---

## Содержание

1. [Обзор](#1-обзор)
2. [Зачем нужен / Связь с другими модулями](#2-зачем-нужен)
3. [Математика](#3-математика)
4. [Pipeline](#4-pipeline)
5. [Kernels](#5-kernels)
6. [C4 Диаграммы](#6-c4-диаграммы)
7. [API](#7-api)
8. [Тесты](#8-тесты)
9. [Файловое дерево](#9-файловое-дерево)
10. [Важные нюансы](#10-важные-нюансы)

---

## 1. Обзор

`CaponProcessor` — GPU-реализация алгоритма Кейпона (MVDR beamformer) для антенных решёток.

**Два режима:**
- **`ComputeRelief`** — пространственный MVDR-спектр: `z[m] = 1/Re(u_m^H R⁻¹ u_m)` → `float[M]`
- **`AdaptiveBeamform`** — адаптивное ДО: `Y_out = (R⁻¹U)^H Y` → `complex<float>[M×N]`

**Архитектура**: Ref03 Unified Architecture (Layer 6 Facade).

| Op-класс | Назначение | Библиотека |
|----------|-----------|------------|
| `CovarianceMatrixOp` | R = Y·Y^H/N + μI | rocBLAS CGEMM + HIP `add_regularization` |
| `CaponInvertOp` | R⁻¹ | **vector_algebra::CholeskyInverterROCm** |
| `CaponReliefOp` | z[m] = 1/Re(u^H R⁻¹ u) | rocBLAS CGEMM + HIP `compute_capon_relief` |
| `AdaptBeamformOp` | Y_out = (R⁻¹U)^H Y | rocBLAS CGEMM × 2 |

**Прототип**: `Doc_Addition/Capon/capon_test/` — ArrayFire реализация с 6 методами инверсии. Тестировалась на P=85 каналах, 25 помехах, f0=3.9 ГГц. GPUWorkLib использует typeCalc=4 (chol+inv) через `CholeskyInverterROCm`.

---

## 2. Зачем нужен

### Проблема: классическое ДО не подавляет помехи

Классическое диаграммообразование: `Y_out = U^H * Y`. Если помеха приходит из стороннего направления — она попадает в боковые лепестки, результат загрязнён.

### Решение: MVDR

MVDR минимизирует мощность выхода при ограничении — сигнал из целевого направления не искажается. Оптимальный вес: `w_m = R⁻¹ u_m / (u_m^H R⁻¹ u_m)`.

### Связь с vector_algebra

Ключевая зависимость — `vector_algebra::CholeskyInverterROCm` (POTRF+POTRI+symmetrize). **Capon не реализует инверсию сам** — делегирует полностью. Это исключает дублирование кода и использует уже оптимизированный и протестированный модуль.

### Связь с statistics

`CovarianceMatrixOp` — накопление по N отсчётам, как `MeanReductionOp` и `WelfordFusedOp` в statistics. Отличие: result — матрица [P×P] (GEMM Y·Y^H), а не скаляр. Диагональная загрузка `add_regularization` аналогична epsilon-стабилизации в WelfordFused.

### Методы инверсии (прототип ArrayFire)

| typeCalc | Метод | В GPUWorkLib |
|----------|-------|-------------|
| 0 | linsolve | — |
| 1 | linsolve + Cholesky | — |
| 2 | `af::inverse(R)` | — |
| 3 | итерации Шульца | — |
| **4** | **chol + inv (POTRF+POTRI)** | ✅ через CholeskyInverterROCm |
| 5 | Шульц batch | — |
| 6 | SVD | — |

---

## 3. Математика

### Ковариационная матрица

$$R = \frac{1}{N} Y Y^H + \mu I \in \mathbb{C}^{P \times P}$$

$Y \in \mathbb{C}^{P \times N}$ — матрица сигнала, $\mu \geq 0$ — диагональная загрузка, гарантирует HPD.

### Инверсия Холецкого

$$R = L L^H \;\;(\text{POTRF}) \;\;\Rightarrow\;\; R^{-1} \;\;(\text{POTRI})$$

После POTRI верхний треугольник содержит R⁻¹. HIP kernel `symmetrize_upper_to_full` (из vector_algebra) заполняет нижний.

### Рельеф Кейпона

$$z[m] = \frac{1}{\text{Re}(u_m^H R^{-1} u_m)}, \quad m = 0,\ldots,M-1$$

Через промежуточную матрицу $W = R^{-1} U \in \mathbb{C}^{P \times M}$:

$$z[m] = \frac{1}{\displaystyle\sum_{p=0}^{P-1} \text{Re}\!\left(\bar{u}_{pm} \cdot w_{pm}\right)}$$

### Адаптивное ДО

$$W = R^{-1} U \in \mathbb{C}^{P \times M}, \qquad Y_{\text{out}} = W^H Y \in \mathbb{C}^{M \times N}$$

### Управляющие векторы (ULA, d/λ=0.5)

$$u_p(\theta) = \exp\!\left(j \cdot 2\pi \cdot p \cdot 0.5 \cdot \sin\theta\right)$$

Для 2D ФАР (из прototипа, несущая $f_0$):

$$U_{pm} = \frac{1}{\sqrt{P}} \exp\!\left(j \cdot \frac{2\pi f_0}{c}(x_p u_m + y_p v_m)\right)$$

---

## 4. Pipeline

### ComputeRelief

```
Y [P×N]  U [P×M]
    │         │
    ▼         ▼
┌────────────────────────────────────────────────┐
│ 1. Upload / CopyGpu                             │
│    H2D: Y → kSignal,  U → kSteering           │
│    D2D: CopySignalGpu / CopySteeringGpu        │
└────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────┐
│ 2. CovarianceMatrixOp::Execute(P, N, mu)        │
│    rocBLAS CGEMM: R = Y*Y^H/N  [TODO]         │
│    HIP add_regularization: R[i,i] += mu         │
│    → kCovMatrix [P×P]                          │
└────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────┐
│ 3. CaponInvertOp::Execute(R_gpu, P)             │
│    → vector_algebra::CholeskyInverterROCm       │
│       POTRF: R = L·L^H                          │
│       POTRI: L → R⁻¹ (верхний треугольник)      │
│       HIP symmetrize_upper_to_full              │
│    → last_inv_ (CholeskyResult, GPU ptr)        │
└────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────┐
│ 4. CaponReliefOp::Execute(P, M, R⁻¹_ptr)       │
│    rocBLAS CGEMM: W = R⁻¹·U [P×M]  [TODO]    │
│    HIP compute_capon_relief:                    │
│      z[m] = 1/Re(Σ_p conj(U[p,m]) * W[p,m])   │
│    → kOutput float[M]                           │
└────────────────────────────────────────────────┘
    │
    ▼
ctx_.Synchronize() → ReadReliefResult() → CaponReliefResult
```

### AdaptiveBeamform (шаги 1–3 идентичны)

```
(Upload + CovarianceMatrixOp + CaponInvertOp — идентично)
    │
    ▼
┌────────────────────────────────────────────────┐
│ 4. AdaptBeamformOp::Execute(P, N, M, R⁻¹_ptr)  │
│    CGEMM 1: W = R⁻¹·U [P×M]        [TODO]    │
│    CGEMM 2: Y_out = W^H·Y [M×N]    [TODO]    │
│    → kOutput complex<float>[M×N]               │
└────────────────────────────────────────────────┘
    │
    ▼
ctx_.Synchronize() → ReadBeamResult() → CaponBeamResult
```

### Mermaid

```mermaid
flowchart LR
  A1[Y signal P×N] --> B[Upload/D2D\nkSignal kSteering]
  A2[U steering P×M] --> B
  B --> C[CovarianceMatrixOp\nCGEMM R=YYH÷N\nadd_regularization\nkCovMatrix]
  C --> D[CaponInvertOp\nvector_algebra\nCholeskyInverterROCm\nPOTRF+POTRI+sym\nlast_inv_]
  D --> E1[CaponReliefOp\nCGEMM W=R-1·U\ncompute_capon_relief\nkOutput float M]
  D --> E2[AdaptBeamformOp\nCGEMM x2\nY_out=WH·Y\nkOutput cx MxN]
  E1 --> F1[CaponReliefResult]
  E2 --> F2[CaponBeamResult]
```

---

## 5. Kernels

Два небольших HIP kernel (hiprtc). Источник: `include/kernels/capon_kernels_rocm.hpp` → `GetCaponKernelSource()`.

Все тяжёлые GEMM-операции и инверсия — через rocBLAS/rocSOLVER (не hiprtc).

### add_regularization

**Назначение**: R[i,i] += mu — делает R HPD для POTRF.

```c
// grid=(P+255)/256, block=256, каждый thread i → диагональный элемент
extern "C" __global__ void add_regularization(float2* R, float mu, unsigned int P) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P) return;
  R[i * P + i].x += mu;  // column-major: R[i,i] → индекс i*P + i
}
```

### compute_capon_relief

**Назначение**: z[m] = 1/Re(u_m^H · w_m) после W = R⁻¹·U через rocBLAS.

```c
// grid=(M+255)/256, block=256, каждый thread m → одно направление
extern "C" __global__ void compute_capon_relief(
    const float2* U, const float2* W, float* z, unsigned int P, unsigned int M) {
  unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) return;
  float acc = 0.0f;
  for (unsigned int p = 0; p < P; ++p) {
    float2 u = U[m*P + p], w = W[m*P + p];
    acc += u.x*w.x + u.y*w.y;  // Re(conj(u)*w)
  }
  z[m] = (acc > 0.0f) ? (1.0f / acc) : 0.0f;
}
```

### rocBLAS операции (TODO)

| Операция | rocBLAS |
|----------|---------|
| R = (1/N)·Y·Y^H | `rocblas_cgemm(NoTrans, ConjTrans, P,P,N, 1/N, Y,P, Y,P, 0, R,P)` |
| W = R⁻¹·U | `rocblas_cgemm(NoTrans, NoTrans, P,M,P, 1, Rinv,P, U,P, 0, W,P)` |
| Y_out = W^H·Y | `rocblas_cgemm(ConjTrans, NoTrans, M,N,P, 1, W,P, Y,P, 0, Yout,M)` |

---

## 6. C4 Диаграммы

### C1 — Контекст

```
Антенная система / ФАР
  Y [P×N], U [P×M]  →  CaponProcessor (ROCm GPU)
                        → CaponReliefResult (z[M])
                        → CaponBeamResult (Y_out[M×N])
```

### C2 — Контейнеры

```
modules/capon/
  CaponProcessor (Facade, L6)
  ├── GpuContext ctx_             stream, compiled kernels, shared bufs
  ├── CovarianceMatrixOp          GpuKernelOp: rocBLAS CGEMM [TODO] + add_regularization
  ├── CaponInvertOp               НЕ GpuKernelOp — обёртка CholeskyInverterROCm
  │     └── CholeskyInverterROCm  ← modules/vector_algebra/ (POTRF+POTRI+symmetrize)
  ├── CaponReliefOp               GpuKernelOp: rocBLAS CGEMM [TODO] + compute_capon_relief
  ├── AdaptBeamformOp             GpuKernelOp: 2x rocBLAS CGEMM [TODO]
  └── CholeskyResult last_inv_    R⁻¹ на GPU (RAII, обновляется каждый вызов)
```

### C3 — Компоненты CaponProcessor

```
EnsureCompiled()  [ленивый, один раз]
  ctx_.CompileModule(GetCaponKernelSource(), {"add_regularization","compute_capon_relief"})
  cov_op_.Initialize(ctx_) / relief_op_.Initialize(ctx_) / beam_op_.Initialize(ctx_)
  inv_op_.CompileKernels()   // warmup symmetrize kernel

RunCovAndInvert(params)   [общий шаг для обоих режимов]
  cov_op_.Execute(P, N, mu)
  last_inv_ = inv_op_.Execute(kCovMatrix_ptr, P)

ComputeRelief()    = Upload + RunCovAndInvert + relief_op_.Execute + Sync + Read
AdaptiveBeamform() = Upload + RunCovAndInvert + beam_op_.Execute  + Sync + Read
```

### C4 — HIP kernel compute_capon_relief

```
Thread grid: ceil(M/256) блоков × 256 threads
  m = blockIdx.x * blockDim.x + threadIdx.x
  if m >= M: return
  acc = 0
  for p = 0..P-1:
    u = U[m*P+p],  w = W[m*P+p]     // column-major
    acc += u.x*w.x + u.y*w.y         // Re(conj(u)*w)
  z[m] = (acc>0) ? 1/acc : 0         // защита от нуля
```

---

## 7. API

Полный API-справочник: [API.md](API.md)

### C++ — полный пример

```cpp
#include "capon_processor.hpp"

// Параметры (P=8 каналов, N=128 отсчётов, M=32 направления)
capon::CaponParams params{8, 128, 32, 0.01f};
capon::CaponProcessor proc(backend);

// Y [P×N], U [P×M] — complex<float>, column-major
// ULA: U[m*P+p] = exp(j * 2π * p * 0.5 * sin(θ[m]))

// Рельеф — пространственный спектр
auto relief = proc.ComputeRelief(Y, U, params);
auto peak = std::max_element(relief.relief.begin(), relief.relief.end());
size_t main_dir = peak - relief.relief.begin();

// Адаптивные лучи
auto beam = proc.AdaptiveBeamform(Y, U, params);
// beam.output[main_dir * 128 + n] — сигнал в главном направлении
```

### Python — NumPy эталон

```python
import numpy as np

def capon_relief_numpy(Y, U, mu=0.01):
    P, N = Y.shape
    R = (Y @ Y.conj().T) / N + mu * np.eye(P, dtype=complex)
    R_inv = np.linalg.inv(R)
    W = R_inv @ U
    return (1.0 / np.real(np.sum(U.conj() * W, axis=0))).astype(np.float32)

def capon_beamform_numpy(Y, U, mu=0.01):
    P, N = Y.shape
    R = (Y @ Y.conj().T) / N + mu * np.eye(P, dtype=complex)
    W = np.linalg.inv(R) @ U
    return (W.conj().T @ Y).astype(np.complex64)

def make_ula_steering(P, thetas_rad):
    p = np.arange(P)[:, np.newaxis]
    return np.exp(1j * 2*np.pi * 0.5 * np.sin(thetas_rad) * p).astype(np.complex64)
```

---

## 8. Тесты

### Тестовые данные

```
MakeNoise:            re[i] = σ·cos(i·1.23),  im[i] = σ·sin(i·2.34)  [детерминированный]
MakeSteeringMatrix:   U[m*P+p] = exp(j·2π·p·0.5·sin(θ_m)),  θ_m ∈ [θ_min, θ_max]
```

### C++ тесты — ROCm (`tests/test_capon_rocm.hpp`)

| # | Тест | P | N | M | mu | Что проверяет | Порог | Что ловит |
|---|------|---|---|---|----|---------------|-------|-----------|
| 01 | `test_01_relief_noise_only` | 8 | 64 | 16 | 0.01 | relief.size()==M; все > 0 | `> 0` | Баг в полном pipeline (GEMM/POTRF/kernel/Read). При R≈σ²I рельеф аналитически постоянный и > 0 |
| 02 | `test_02_relief_with_interference` | 8 | 128 | 32 | 0.001 | relief.size()==M | размерность | Корректность при больших параметрах. TODO: добавить помеху, проверить argmin(relief) |
| 03 | `test_03_adaptive_beamform_dims` | 4 | 32 | 6 | 0.01 | output.size()==M×N | точное равенство | Ошибку в ReadBeamResult или CGEMM Y_out=W^H*Y |
| 04 | `test_04_regularization` | 4 | **16 (N<P!)** | 8 | 0.1 | isfinite && ≥ 0 | `isfinite && ≥ 0` | Вырожденность: без mu POTRF→info!=0→исключение. С mu=0.1 восстановлена HPD. NaN из kernel при acc=0 |
| 05 | `test_05_gpu_to_gpu` | 8 | 64 | 16 | 0.01 | GPU-to-GPU | SKIP | TODO: GPU alloc в тесте |

**Почему шум в тесте 01?** При $Y \sim CN(0,\sigma^2)$: $R \approx \sigma^2 I$, $R^{-1} \approx \sigma^{-2} I$, $z[m] \approx 1/P$ — аналитически постоянно. Любое отклонение (NaN/Inf/z<0) однозначно указывает на баг.

**Почему N<P в тесте 04?** $YY^H/N$ вырождена при N<P. Без μ: POTRF вернёт `info!=0`. С μ=0.1: HPD восстановлена. `isfinite && ≥ 0` ловит: недостаточную регуляризацию, NaN из kernel при acc=0.

### Python тесты (запланированы: `Python_test/capon/test_capon.py`)

| # | Тест | Порог |
|---|------|-------|
| 1 | GPU рельеф ≈ NumPy эталон (P=4, N=32, M=8) | ATOL=1e-4 |
| 2 | output.shape == (M, N) | точное равенство |
| 3 | argmin(relief) ≈ направление помехи | ±0.5 бина |
| 4 | N<P, mu>0 → all isfinite | isfinite |

---

## 9. Файловое дерево

```
modules/capon/
├── CMakeLists.txt                        ROCm+rocBLAS+rocSOLVER+vector_algebra
├── include/
│   ├── capon_types.hpp                   CaponParams, CaponReliefResult, CaponBeamResult, shared_buf
│   ├── capon_processor.hpp               Facade (Ref03 L6)
│   ├── kernels/
│   │   └── capon_kernels_rocm.hpp        GetCaponKernelSource() — hiprtc kernel sources
│   └── operations/
│       ├── covariance_matrix_op.hpp      L5: R = Y*Y^H/N + μI
│       ├── capon_invert_op.hpp           L5: обёртка CholeskyInverterROCm (не GpuKernelOp)
│       ├── capon_relief_op.hpp           L5: z[m] = 1/Re(u^H R⁻¹ u)
│       └── adapt_beam_op.hpp             L5: Y_out = (R⁻¹U)^H Y
├── src/
│   └── capon_processor.cpp               Facade + Upload/Copy/Read + Move semantics
└── tests/
    ├── all_test.hpp                       capon_all_test::run()
    ├── test_capon_rocm.hpp                5 тестов (01-04 активны, 05 SKIP)
    └── README.md

Doc_Addition/Capon/capon_test/            ArrayFire прототип (CPU)
├── src/capon_relief.cpp                  keypon_relief(), adapt(), adapt_beams() — 6 методов инверсии
└── include/inv_schulz.h                  Schulz iterations (typeCalc=3/5)

modules/vector_algebra/                   Зависимость — инверсия R⁻¹
└── include/cholesky_inverter_rocm.hpp    CholeskyInverterROCm: POTRF+POTRI+symmetrize
```

---

## 10. Важные нюансы

1. **ROCm-only.** `CMakeLists.txt` пропускает модуль (`return()`) при отсутствии `ROCM_ENABLED`, `rocblas_FOUND` или `rocsolver_FOUND`.

2. **Column-major обязателен.** rocBLAS/rocSOLVER — column-major. `[p, m]` → `m*P + p`. NumPy: `np.asfortranarray(Y)` или `order='F'`.

3. **N < P без mu → POTRF fail.** При N < P матрица $YY^H/N$ вырождена → не HPD → POTRF вернёт `info != 0` → исключение. Всегда `mu > 0`.

4. **CaponInvertOp — не GpuKernelOp.** Не вызывать `Release()`. В `~CaponProcessor()` явно освобождаются только `cov_op_`, `relief_op_`, `beam_op_`. `inv_op_` — стандартный деструктор.

5. **CholeskyInverterROCm не перемещаемый.** Move assignment `CaponProcessor` не переприсваивает `inv_op_` (помечено TODO). Для fix — обернуть в `unique_ptr<CaponInvertOp>`.

6. **Статус: CGEMM — TODO.** rocBLAS CGEMM в `CovarianceMatrixOp`, `CaponReliefOp`, `AdaptBeamformOp` помечены TODO. HIP kernels реализованы. Нужно получить `rocblas_handle` из backend.

7. **last_inv_ пересоздаётся каждый вызов.** `RunCovAndInvert()` делает `last_inv_ = ...` — старый `CholeskyResult` (с `hipFree`) уничтожается. Нельзя хранить `AsHipPtr()` дольше одного пайплайна.

8. **Прototип: P=85, 25 помех, f0=3.9 ГГц.** typeCalc=4 (chol+inv) — лучший баланс скорости и стабильности. SVD (typeCalc=6) стабильнее, но медленнее. Шульц (typeCalc=3/5) быстрее при P>>100, но требует начального приближения.

---

*Обновлено: 2026-03-16*
*[Quick.md](Quick.md) | [API.md](API.md)*
