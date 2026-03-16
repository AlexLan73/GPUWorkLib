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
| 01 | `test_01_relief_noise_only` | ComputeRelief — только шум → плоский спектр |
| 02 | `test_02_relief_with_interference` | ComputeRelief — подавление помехи (TODO: добавить помеху) |
| 03 | `test_03_adaptive_beamform_dims` | AdaptiveBeamform — размерность выхода [M × N] |
| 04 | `test_04_regularization` | Регуляризация — устойчивость при N < P |
| 05 | `test_05_gpu_to_gpu` | GPU-to-GPU пайплайн (TODO: GPU alloc в тесте) |

## Алгоритм Кейпона (MVDR)

```
Y — матрица сигнала         [P × N]   P каналов, N отсчётов
U — управляющие векторы     [P × M]   M направлений

R = (1/N) * Y * Y^H + μI   — ковариационная матрица [P × P]
R^{-1}                      — обращение (rocSOLVER POTRF+POTRI)

Рельеф:   z[m] = 1 / Re(u_m^H * R^{-1} * u_m)
Адапт ДО: Y_out = (R^{-1} * U)^H * Y   [M × N]
```

## Эталон

Сравнение с CPU прототипом: `Doc_Addition/Capon/capon_test/` (ArrayFire).

TODO: Python тесты в `Python_test/capon/test_capon.py` — сравнение с NumPy/SciPy.

## Статус

- [ ] Реализация CovarianceMatrixOp (rocBLAS CGEMM)
- [ ] Реализация CaponInvertOp (rocSOLVER POTRF+POTRI)
- [ ] Реализация CaponReliefOp (rocBLAS CGEMM + HIP kernel)
- [ ] Реализация AdaptBeamformOp (rocBLAS CGEMM)
- [ ] Тесты 01-04 (запускаются после реализации)
- [ ] Тест 05 (GPU-to-GPU)
- [ ] Python тесты (сравнение с NumPy)
