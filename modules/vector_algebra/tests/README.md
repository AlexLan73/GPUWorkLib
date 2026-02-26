# vector_algebra — C++ тесты (Task_11 v2)

## Описание

Тесты модуля `vector_algebra` — инверсия эрмитовой положительно определённой матрицы методом Холецкого (ROCm).

**Два режима симметризации**: каждый тест запускается для Roundtrip и GpuKernel.

## Функциональные тесты (test_cholesky_inverter_rocm.hpp)

| # | Функция | Описание | n | batch | Допуск |
|---|---------|----------|---|-------|--------|
| 5.6.1 | `TestCpuIdentity` | I(5×5): I⁻¹ = I | 5 | 1 | Frobenius < 1e-5 |
| 5.6.2 | `TestCpu341` | HPD 341×341 | 341 | 1 | Frobenius < 1e-2 |
| 5.6.3 | `TestGpuVoidPtr341` | void* вход | 341 | 1 | Frobenius < 1e-2 |
| 5.6.4 | `TestZeroCopyClMem` | cl_mem (SKIP) | 341 | 1 | — |
| 5.6.5 | `TestBatchCpu_4x64` | CPU batched | 64 | 4 | Frobenius < 1e-3 |
| 5.6.6 | `TestBatchGpu_4x64` | GPU void* batched | 64 | 4 | Frobenius < 1e-3 |
| 5.6.7 | `TestBatchSizes` | Batch 1,4,8,16 | 64 | var | PASS |
| 5.6.8 | `TestMatrixSizes` | n=32,64,128,256 | var | 4 | Frobenius < 1e-2 |
| 5.6.9 | `TestResultAccess` | .matrix()/.matrices()/AsHipPtr() | 5 | 1+3 | Shape OK |
| 5.6.10 | `TestResolveMatrixSize` | sqrt(n_point) логика | — | — | Без GPU |

## Cross-backend тесты (test_cross_backend_conversion.hpp)

| # | Функция | Описание |
|---|---------|----------|
| 5.10.1 | `TestConvert_VectorInput` | Эталон: vector → Invert → AsVector |
| 5.10.2 | `TestConvert_HipInput` | Другой ROCm контекст → наш инвертер |
| 5.10.3 | `TestConvert_ClMemInput` | cl_mem → ZeroCopy (SKIP) |
| 5.10.4 | `TestConvert_OutputFormats` | AsVector() == AsHipPtr()+hipMemcpy |

## Benchmark (test_benchmark_symmetrize.hpp)

| # | Функция | n | batch | Описание |
|---|---------|---|-------|----------|
| 5.13.1 | `BenchmarkSingle341` | 341 | 1 | Roundtrip vs GpuKernel |
| 5.13.2 | `BenchmarkBatch_16x64` | 64 | 16 | Roundtrip vs GpuKernel |
| 5.13.3 | `BenchmarkBatch_4x256` | 256 | 4 | Roundtrip vs GpuKernel |

## Profiler (test_benchmark_symmetrize.hpp)

| # | Функция | Описание |
|---|---------|----------|
| 5.15 | `TestProfilerIntegration` | SetGPUInfo → Start → Record → Stop → PrintReport/Export |

## Запуск

```bash
# Из main.cpp (через RunVectorAlgebraTests)
./GPUWorkLib

# Все тесты под #if ENABLE_ROCM
```

## Режимы симметризации

| Режим | Описание | Когда |
|-------|----------|-------|
| **Roundtrip** | Download GPU → CPU sym → Upload | Простой, без kernel |
| **GpuKernel** | HIP kernel in-place (hiprtc) | Быстрый, всё на GPU |

## Вспомогательные функции

| Функция | Описание |
|---------|----------|
| `MakePositiveDefiniteHermitian(n, seed)` | A = B*B^H + n*I |
| `FrobeniusError(A, B, n)` | `‖A*B − I‖_F` |
| `ModeName(mode)` | "Roundtrip" / "GpuKernel" |

## Зависимости

- ROCm (AMD GPU)
- rocBLAS + rocSOLVER + hiprtc
- `ENABLE_ROCM=ON` в CMake
