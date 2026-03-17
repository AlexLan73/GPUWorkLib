# Capon Module — Tests

## Описание

Тесты модуля `capon` — алгоритм Кейпона (MVDR beamformer) на GPU (ROCm).

## Запуск

```bash
./GPUWorkLib capon          # Запустить тесты модуля capon
./GPUWorkLib all            # Запустить все модули
```

## Тесты

### `test_capon_rocm.hpp` — ROCm тесты `CaponProcessor`

| # | Тест | Что проверяет |
|---|------|---------------|
| 01 | `test_01_relief_noise_only` | ComputeRelief — только шум → все z[m] > 0, размер верный |
| 02 | `test_02_relief_with_interference` | MVDR подавление помехи: z[m_int] < mean(z)/2 |
| 03 | `test_03_adaptive_beamform_dims` | AdaptiveBeamform — размерность выхода [M × N] |
| 04 | `test_04_regularization` | Устойчивость при N < P (вырожденная матрица, mu > 0) |
| 05 | `test_05_gpu_to_gpu` | SKIP (TODO: GPU alloc/upload API) |

### `capon_benchmark.hpp` + `test_capon_benchmark_rocm.hpp` — бенчмарки

| Класс | Что измеряет |
|-------|--------------|
| `CaponReliefBenchmarkROCm` | ComputeRelief: 5 warmup + 20 runs (hipEvent timing) |
| `CaponBeamformBenchmarkROCm` | AdaptiveBeamform: 5 warmup + 20 runs |

Запускается только при `is_prof=true` в `configGPU.json`.

## Алгоритм Кейпона (MVDR)

```
Y — матрица сигнала         [P × N]   P каналов, N отсчётов
U — управляющие векторы     [P × M]   M направлений

R = (1/N) * Y * Y^H + μI   — ковариационная матрица [P × P]
R^{-1}                      — обращение (rocSOLVER POTRF+POTRI)

Рельеф:   z[m] = 1 / Re(u_m^H * R^{-1} * u_m)
Адапт ДО: Y_out = (R^{-1} * U)^H * Y   [M × N]
```

## Архитектура (Ref03)

| Op-класс | Файл | Реализация |
|----------|------|------------|
| `CovarianceMatrixOp` | `operations/covariance_matrix_op.hpp` | ✅ rocBLAS CGEMM |
| `CaponInvertOp` | `operations/capon_invert_op.hpp` | ✅ vector_algebra::CholeskyInverterROCm |
| `ComputeWeightsOp` | `operations/compute_weights_op.hpp` | ✅ rocBLAS CGEMM (W = R⁻¹·U) |
| `CaponReliefOp` | `operations/capon_relief_op.hpp` | ✅ HIP kernel compute_capon_relief |
| `AdaptBeamformOp` | `operations/adapt_beam_op.hpp` | ✅ rocBLAS CGEMM (Y_out = W^H·Y) |
| `CaponProcessor` (Facade) | `src/capon_processor.cpp` | ✅ thin facade, Ref03 Layer 6 |

## Эталон

Сравнение с CPU прототипом: `Doc_Addition/Capon/capon_test/` (ArrayFire).

TODO: Python тесты в `Python_test/capon/test_capon.py` — сравнение с NumPy/SciPy.

## Статус

- [x] Реализация CovarianceMatrixOp (rocBLAS CGEMM)
- [x] Реализация CaponInvertOp (vector_algebra::CholeskyInverterROCm)
- [x] Реализация ComputeWeightsOp (rocBLAS CGEMM)
- [x] Реализация CaponReliefOp (HIP hiprtc kernel)
- [x] Реализация AdaptBeamformOp (rocBLAS CGEMM)
- [x] Тесты 01-04 (написаны, НЕ тестировано на GPU)
- [ ] Тест 05 (GPU-to-GPU: нужен GPU alloc/upload API)
- [ ] Python тесты (сравнение с NumPy)
