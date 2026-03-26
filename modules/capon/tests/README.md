
# Capon Module — Tests

## Описание

Тесты модуля `capon` — алгоритм Кейпона (MVDR beamformer) на GPU (ROCm).
Все тесты компилируются только при `ENABLE_ROCM=1` (Linux + AMD GPU).

## Запуск

```bash
./GPUWorkLib capon          # Запустить тесты модуля capon
./GPUWorkLib all            # Запустить все модули
```

## Файлы тестов

### `test_capon_rocm.hpp` — Базовые ROCm тесты CaponProcessor

| # | Тест | Что проверяет |
|---|------|---------------|
| 01 | `test_01_relief_noise_only` | ComputeRelief — только шум → все z[m] > 0, размер верный |
| 02 | `test_02_relief_with_interference` | MVDR подавление помехи: z[m_int] < mean(z)/2 |
| 03 | `test_03_adaptive_beamform_dims` | AdaptiveBeamform — размерность выхода [M × N] |
| 04 | `test_04_regularization` | Устойчивость при N < P (вырожденная матрица, mu > 0) |
| 05 | `test_05_gpu_to_gpu` | GPU-to-GPU pipeline (hipMalloc + hipMemcpy D2D + void* API) |

### `test_capon_reference_data.hpp` — Тесты на реальных данных MATLAB

Загружает эталонные данные из `modules/capon/tests/data/` (копия из прототипа `Doc_Addition/Capon/capon_test/build/`):
- `x_data.txt`, `y_data.txt` — координаты антенных секций (340 значений)
- `signal_matlab.txt` — MATLAB сигнал (341 строка x 1000 комплексных чисел)

| # | Тест | Что проверяет |
|---|------|---------------|
| 01 | `test_01_load_files` | Загрузка и валидация файлов (x, y, signal) |
| 02 | `test_02_physical_relief_properties` | GPU рельеф на реальных данных (P=85, N=1000, M=1369): > 0 и конечный |
| 03 | `test_03_cpu_vs_gpu_small_p` | CPU эталон vs GPU (P=8, N=64, M=16): относительная погрешность < 0.5% |

Физическая модель: f0 = 3921.15 МГц, ULA с физическими координатами.
CPU эталон: Cholesky + ForwardSolve (формула GPU с делением на N).

### `test_capon_hip_opencl_to_rocm.hpp` — HIP alloc + OpenCL write (hipMalloc → clEnqueueSVMMemcpy → Capon)

**Демонстрация pipeline с HIP-owned памятью для заказчика. Аналог HIPmemTest для 4 ГБ данных.**

#### Ключевая идея

Существует два способа передать данные в Capon на ROCm:

| Способ | Кто выделяет память | Кто пишет данные | Нужен ZeroCopyBridge? |
|--------|--------------------|-----------------|-----------------------|
| **Старый** (`opencl_to_rocm`) | OpenCL (`cl_mem`) | OpenCL (`MemcpyH2D`) | **ДА** — мост OCL→HIP |
| **Новый** (`hip_opencl_to_rocm`) | HIP (`hipMalloc`) | OpenCL (`clEnqueueSVMMemcpy`) | **НЕТ** — уже HIP ptr |

#### Почему hipMalloc-указатель работает в OpenCL?

На AMD GPU оба рантайма (HIP и OpenCL) построены поверх одного слоя — **HSA Runtime (ROCR)**. `hipMalloc` выделяет память через `hsa_amd_memory_pool_allocate` и возвращает указатель в **едином виртуальном адресном пространстве HSA**. OpenCL видит тот же указатель как валидный **SVM (Shared Virtual Memory) pointer** — передавать его в `clSetKernelArgSVMPointer` или `clEnqueueSVMMemcpy` безопасно, без конвертации.

```
                  ┌─────────────────────────────────────────┐
                  │         AMD GPU   (один физический)      │
   hipMalloc() ──►│  hsa_amd_memory_pool_allocate → VRAM    │
                  │     ▲  адрес 0x7f1234560000              │
                  │     │  единое VA пространство HSA        │
clEnqueueSVMMemcpy│─────┘                                    │
                  └─────────────────────────────────────────┘
   HIP и OpenCL видят ОДИН буфер. Нет копирования между API.
```

#### Сравнение с HIPmemTest (`/home/alex/C++/HIPmemTest/`)

```
HIPmemTest (доказательство концепции, 10 элементов):
  hipMalloc(d_vec_a, d_vec_b) → hipMemset(0)
  clSetKernelArgSVMPointer(kernel, 0, d_vec_a)   ← hipMalloc ptr → OpenCL
  clEnqueueNDRangeKernel(fill_vectors)           ← OpenCL пишет в VRAM
  hip_kernel<<<>>>(d_vec_a, d_vec_b, d_vec_c)   ← HIP читает — ноль копий!

Этот тест (реальные данные заказчика, до 4 ГБ):
  hipMalloc(d_hip_Y, d_hip_U) → hipMemset(0)
  clEnqueueSVMMemcpy(d_hip_Y, h_signal.data(), bytes_Y)   ← один DMA CPU→VRAM
  clEnqueueSVMMemcpy(d_hip_U, h_steering.data(), bytes_U) ← один DMA CPU→VRAM
  clFinish()
  CaponProcessor.ComputeRelief(d_hip_Y, d_hip_U, params)  ← HIP ptr напрямую!
```

**Отличие**: HIPmemTest использует OpenCL kernel чтобы ВЫЧИСЛИТЬ данные. Здесь данные приходят с CPU (реальный сигнал из файлов) через `clEnqueueSVMMemcpy` — один DMA-трансфер без промежуточных буферов.

#### Что НЕ используется (и почему это важно для 4 ГБ)

| Не используется | Почему важно для больших данных |
|-----------------|--------------------------------|
| `clSVMAlloc` (staging-буфер) | Экономия памяти: нет дублирования 4 ГБ в SVM |
| copy-kernel (GPU→GPU копирование) | Нет лишнего GPU-bandwidth на staging→VRAM |
| `ZeroCopyBridge` | Нет накладных расходов на конвертацию адресов |

#### Pipeline теста 02

```
[ЭТАП 1] Загрузка: signal_matlab.txt + x_data.txt + y_data.txt → CPU RAM
     ↓
[ЭТАП 2] hipMalloc:  d_hip_Y[P×N], d_hip_U[P×M] в VRAM
         hipMemset(0) — обнуляем, доказываем что OpenCL запишет данные
     ↓
[ЭТАП 3] OpenCL пишет в HIP-память (NO staging, NO copy-kernel):
         clEnqueueSVMMemcpy(d_hip_Y, h_signal.data(), bytes_Y)
         clEnqueueSVMMemcpy(d_hip_U, h_steering.data(), bytes_U)
         clFinish()
     ↓
[ЭТАП 4] Capon ROCm:
         CaponProcessor.ComputeRelief(d_hip_Y, d_hip_U, params) → z[m]
         (d_hip_Y — тот же указатель от hipMalloc, ZeroCopyBridge не нужен!)
     ↓
         hipFree(d_hip_Y); hipFree(d_hip_U)
```

| # | Тест | Что проверяет |
|---|------|---------------|
| 01 | `test_01_detect_hip_svm` | SVM capabilities (coarse/fine/system), проверка применимости pipeline |
| 02 | `test_02_hip_opencl_capon_pipeline` | **ПОЛНЫЙ PIPELINE** с данными заказчика (P=85, N=1000, M≈37) |
| 03 | `test_03_hip_opencl_matches_direct` | Корректность: новый путь == прямой путь (допуск < 1e-4) |

#### Требования

- AMD GPU с ROCm 7.2+ (проверено: gfx1201 / Radeon 9070)
- OpenCL SVM coarse-grain (минимум, как HIPmemTest)
- Данные заказчика: `modules/capon/tests/data/` (оригинал: `Doc_Addition/Capon/capon_test/build/`)
- Proof of concept: `/home/alex/C++/HIPmemTest/`

---

### `test_capon_opencl_to_rocm.hpp` — Zero Copy Interop (OpenCL → ROCm) с данными заказчика

**Демонстрация полного production pipeline с реальными данными заказчика.**

Тест 02 (customer_data_pipeline) чётко разделён на **4 этапа для заказчика**:

| Этап | Описание | Что происходит |
|------|----------|----------------|
| **1. ЗАГРУЗКА ДАННЫХ** | Файлы заказчика (MATLAB) | x_data, y_data, signal_matlab → P=85, N=1000, M=37 |
| **2. ЗАПИСЬ НА GPU** | OpenCL `cl_mem` | `cl.Allocate()` + `cl.MemcpyHostToDevice()` + `cl.Synchronize()` |
| **3. ZERO COPY** | OpenCL → ROCm | `ZeroCopyBridge::ImportFromOpenCl()` — 0 копий, тот же VRAM |
| **4. РАСЧЁТ КЕЙПОНА** | ROCm GPU pipeline | `CaponProcessor::ComputeRelief(hip_ptr, hip_ptr, params)` |

| # | Тест | Что проверяет |
|---|------|---------------|
| 01 | `test_01_detect_interop` | Определение метода Zero Copy (DMA-BUF / AMD GPU VA / NONE) |
| 02 | `test_02_customer_data_pipeline` | **ПОЛНЫЙ PIPELINE** с данными заказчика (4 этапа, P=85, N=1000) |
| 03 | `test_03_zerocopy_matches_direct` | Прозрачность: Zero Copy путь == прямой путь (< 1e-4) |
| 04 | `test_04_beamform_customer_data` | AdaptiveBeamform с данными заказчика через Zero Copy |
| 05 | `test_05_svm_customer_data` | **SVM PATH**: clSVMAlloc → memcpy → ImportFromSVM → Capon → verify |

Pipeline теста 02 (данные заказчика):
```
[ЭТАП 1] Загрузка: signal_matlab.txt, x_data.txt, y_data.txt
     ↓
[ЭТАП 2] OpenCL:  cl.Allocate() → cl.MemcpyH2D(signal, steering) → cl.Synchronize()
     ↓
[ЭТАП 3] ZeroCopy: ZeroCopyBridge.ImportFromOpenCl() → hip_Y, hip_U (0 копий!)
     ↓
[ЭТАП 4] Capon:   CaponProcessor.ComputeRelief(hip_Y, hip_U, params) → z[m]
```

Подробное руководство: [GUIDE_opencl_to_rocm.md](GUIDE_opencl_to_rocm.md)

### `capon_benchmark.hpp` + `test_capon_benchmark_rocm.hpp` — Бенчмарки

| Класс | Что измеряет |
|-------|--------------|
| `CaponReliefBenchmarkROCm` | ComputeRelief: 5 warmup + 20 runs (hipEvent timing) |
| `CaponBeamformBenchmarkROCm` | AdaptiveBeamform: 5 warmup + 20 runs |

Параметры: P=16 каналов, N=256 отсчётов, M=64 направления, mu=0.01.
Результаты → `Results/Profiler/GPU_00_Capon_ROCm/`.
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

Python тесты: `Python_test/capon/test_capon.py` — см. [Python_test/capon/README.md](../../../Python_test/capon/README.md).

## Связанные файлы

| Файл | Описание |
|------|----------|
| `capon_test_helpers.hpp` | Общие утилиты: backend, загрузка данных, steering, noise |
| [GUIDE_opencl_to_rocm.md](GUIDE_opencl_to_rocm.md) | Руководство разработчика: паттерн Zero Copy OpenCL → ROCm |
| `DrvGPU/backends/rocm/zero_copy_bridge.hpp` | ZeroCopyBridge API |
| `DrvGPU/tests/test_zero_copy.hpp` | Базовые тесты ZeroCopyBridge |
| `Python_test/capon/test_capon.py` | Python тесты с NumPy + MATLAB данные |
| `modules/capon/tests/data/` | Данные заказчика (MATLAB: signal, координаты, эталон). [Памятка](data/README_DATA.md) |
| `Doc_Addition/Capon/capon_test/` | Оригинальный прототип заказчика (ArrayFire CPU) |

## Статус

- [x] Реализация CovarianceMatrixOp (rocBLAS CGEMM)
- [x] Реализация CaponInvertOp (vector_algebra::CholeskyInverterROCm)
- [x] Реализация ComputeWeightsOp (rocBLAS CGEMM)
- [x] Реализация CaponReliefOp (HIP hiprtc kernel)
- [x] Реализация AdaptBeamformOp (rocBLAS CGEMM)
- [x] Тесты 01-04 ROCm базовые (написаны, НЕ тестировано на GPU)
- [x] Тесты reference_data 01-03 (MATLAB данные, CPU vs GPU)
- [x] Тесты opencl_to_rocm 01-05 (Zero Copy Interop + SVM path)
- [x] Тесты hip_opencl_to_rocm 01-03 (hipMalloc + clEnqueueSVMMemcpy → Capon)
- [x] Бенчмарки (ComputeRelief + AdaptiveBeamform, GpuBenchmarkBase)
- [x] Тест 05 GPU-to-GPU: hipMalloc + D2D в test_capon_rocm.hpp
- [x] Python тесты (`Python_test/capon/test_capon.py`) — NumPy + реальные данные MATLAB
- [ ] Миграция на test_utils (GpuTestBase/TestRunner) — сейчас assert()
