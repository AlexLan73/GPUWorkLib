# Capon (MVDR) — API-справочник

**Namespace**: `capon` | **Backend**: ROCm only (`ENABLE_ROCM=1`)

---

## Типы (`capon_types.hpp`)

```cpp
struct CaponParams {
  uint32_t n_channels;    // P — число каналов (строки Y и U)
  uint32_t n_samples;     // N — число отсчётов (столбцы Y)
  uint32_t n_directions;  // M — число направлений (столбцы U)
  float    mu = 0.0f;     // регуляризация, всегда > 0 рекомендуется
};

struct CaponReliefResult {
  std::vector<float> relief;  // [M] — z[m] = 1/Re(u^H·R⁻¹·u)
};

struct CaponBeamResult {
  uint32_t n_directions = 0;
  uint32_t n_samples    = 0;
  std::vector<std::complex<float>> output;  // [M×N] row-major по лучам
};

namespace shared_buf {
  constexpr size_t kSignal    = 0;  // Y [P×N]
  constexpr size_t kSteering  = 1;  // U [P×M]
  constexpr size_t kCovMatrix = 2;  // R [P×P]
  constexpr size_t kInvMatrix = 3;  // зарезервирован (R⁻¹ в CholeskyResult)
  constexpr size_t kOutput    = 4;  // float[M] или complex[M×N]
  constexpr size_t kCount     = 5;
}
```

---

## CaponProcessor — Facade (`capon_processor.hpp`)

```cpp
// Конструктор — компиляция kernels ленивая (при первом вызове)
explicit CaponProcessor(drv_gpu_lib::IBackend* backend);

// CPU входные данные (H2D upload внутри)
CaponReliefResult ComputeRelief(
    const std::vector<std::complex<float>>& signal,    // Y [P×N], column-major
    const std::vector<std::complex<float>>& steering,  // U [P×M], column-major
    const CaponParams& params);

CaponBeamResult AdaptiveBeamform(
    const std::vector<std::complex<float>>& signal,
    const std::vector<std::complex<float>>& steering,
    const CaponParams& params);

// GPU входные данные (D2D copy, caller владеет буферами)
CaponReliefResult ComputeRelief(void* gpu_signal, void* gpu_steering, const CaponParams&);
CaponBeamResult   AdaptiveBeamform(void* gpu_signal, void* gpu_steering, const CaponParams&);

// Не копируемый, перемещаемый
// NOTE: move assignment не переприсваивает inv_op_ (CholeskyInverterROCm не перемещаемый)
```

---

## Op-классы (Layer 5)

### CovarianceMatrixOp (`operations/covariance_matrix_op.hpp`)

```cpp
// Base: drv_gpu_lib::GpuKernelOp
void Execute(uint32_t n_channels, uint32_t n_samples, float mu);
// R = (1/N)*Y*Y^H + mu*I
// kSignal → kCovMatrix
// Шаг 1: rocBLAS CGEMM [TODO]
// Шаг 2: HIP kernel add_regularization
```

### CaponInvertOp (`operations/capon_invert_op.hpp`)

```cpp
// НЕ наследует GpuKernelOp — обёртка над vector_algebra::CholeskyInverterROCm
explicit CaponInvertOp(drv_gpu_lib::IBackend* backend);

vector_algebra::CholeskyResult Execute(void* gpu_R, uint32_t n_channels);
// kCovMatrix → CholeskyResult (владеет GPU ptr R⁻¹)
// POTRF + POTRI + symmetrize (vector_algebra)

void SetCheckInfo(bool enabled);  // false = benchmark-режим
void CompileKernels();            // warmup hiprtc symmetrize kernel
```

### CaponReliefOp (`operations/capon_relief_op.hpp`)

```cpp
// Base: drv_gpu_lib::GpuKernelOp
void Execute(uint32_t n_channels, uint32_t n_directions, void* R_inv_ptr);
// R_inv_ptr = last_inv_.AsHipPtr() из CaponProcessor
// kSteering (U) + R_inv_ptr → kOutput float[M]
// Шаг 1: rocBLAS CGEMM W = R⁻¹·U [P×M]  [TODO]
// Шаг 2: HIP kernel compute_capon_relief
// Приватный буфер: W [P×M]
```

### AdaptBeamformOp (`operations/adapt_beam_op.hpp`)

```cpp
// Base: drv_gpu_lib::GpuKernelOp
void Execute(uint32_t n_channels, uint32_t n_samples,
             uint32_t n_directions, void* R_inv_ptr);
// kSignal (Y) + kSteering (U) + R_inv_ptr → kOutput complex[M×N]
// Шаг 1: rocBLAS CGEMM W = R⁻¹·U [P×M]  [TODO]
// Шаг 2: rocBLAS CGEMM Y_out = W^H·Y [M×N]  [TODO]
// Приватный буфер: W [P×M]
```

---

## HIP Kernels (`include/kernels/capon_kernels_rocm.hpp`)

Компилируются через hiprtc. Источник: `GetCaponKernelSource()`.

```c
// R[i,i].x += mu  (column-major: диагональ = i*P + i)
// grid=(P+255)/256, block=256
extern "C" __global__ void add_regularization(float2* R, float mu, unsigned int P);

// z[m] = 1/Re(Σ_p conj(U[p,m]) * W[p,m])
// grid=(M+255)/256, block=256
extern "C" __global__ void compute_capon_relief(
    const float2* U, const float2* W, float* z, unsigned int P, unsigned int M);
```

---

## Цепочка вызовов: ComputeRelief

```
CaponProcessor::ComputeRelief(Y_cpu, U_cpu, params)
  EnsureCompiled()
    ctx_.CompileModule(GetCaponKernelSource(), {"add_regularization","compute_capon_relief"})
    cov_op_.Initialize(ctx_) / relief_op_.Initialize(ctx_) / beam_op_.Initialize(ctx_)
    inv_op_.CompileKernels()
  UploadSignal()   → ctx_.UploadToDevice(kSignal, ...)
  UploadSteering() → ctx_.UploadToDevice(kSteering, ...)
  RunCovAndInvert(params)
    cov_op_.Execute(P, N, mu)
      rocBLAS CGEMM: R = Y*Y^H/N  [TODO]
      HIP add_regularization: R[i,i] += mu
    inv_op_.Execute(kCovMatrix_ptr, P)
      CholeskyInverterROCm::Invert() → POTRF + POTRI + symmetrize
      → last_inv_ (CholeskyResult, GPU ptr R⁻¹)
  relief_op_.Execute(P, M, last_inv_.AsHipPtr())
    rocBLAS CGEMM: W = R⁻¹·U  [TODO]
    HIP compute_capon_relief: z[m] = 1/Re(Σ conj(U)*W)
  ctx_.Synchronize()
  ReadReliefResult(M) → D2H download
  → CaponReliefResult
```

---

*Обновлено: 2026-03-16 | [Full.md](Full.md) | [Quick.md](Quick.md)*
