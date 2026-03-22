# 🔍 Code Review: vector_algebra + capon

> **Дата**: 2026-03-22
> **Ревьюер**: Кодо (AI Assistant)
> **Объём**: ~31 файлов (2 модуля)
> **Методы анализа**: sequential-thinking, grep analysis
> **Ветка**: main (Linux, AMD GPU, ROCm 7.2+)

## ✅ ИСПРАВЛЕНО В ЭТОЙ СЕССИИ (2026-03-22)

| # | Тип | Описание | Файлы |
|---|-----|----------|-------|
| 1 | 🟡→✅ | symmetrize hiprtc: добавлен `--offload-arch` для ISA-оптимизаций | `symmetrize_gpu_rocm.cpp` |
| 2 | 🟡→✅ | DiagonalLoadRegularizer hiprtc: добавлен `--offload-arch` | `diagonal_load_regularizer.cpp` |
| 3 | 🟢→✅ | Windows stubs для capon (CaponProcessor, types) | `capon_processor.hpp`, `capon_types.hpp` |
| 4 | 🟢→✅ | Windows stubs для vector_algebra (CholeskyInverterROCm, MatrixOpsROCm, types) | `cholesky_inverter_rocm.hpp`, `matrix_ops_rocm.hpp`, `vector_algebra_types.hpp` |

---

## 📊 Сводная оценка

| Аспект | vector_algebra | capon |
|--------|---------------|-------|
| **Ref03** | ⚠️ Частично | ✅ Полный |
| **API** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Kernel качество** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **BLAS интеграция** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Документация** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Patterns** | Strategy (IMatrixRegularizer) | Facade + Ops + Strategy |

**Вердикт**: capon — **эталонный** Ref03 модуль (наравне с statistics). vector_algebra — фундамент (rocBLAS/rocSOLVER), частично legacy.

---

## 📐 Архитектурная карта

```
capon::CaponProcessor (Facade, Ref03)
  ├── GpuContext ctx_
  ├── CovarianceMatrixOp → MatrixOpsROCm → rocBLAS CGEMM
  ├── IMatrixRegularizer → DiagonalLoadRegularizer (Strategy)
  ├── CaponInvertOp → CholeskyInverterROCm → rocSOLVER POTRF+POTRI
  ├── ComputeWeightsOp → MatrixOpsROCm → rocBLAS CGEMM
  ├── CaponReliefOp → HIP kernel (compute_capon_relief)
  └── AdaptBeamformOp → MatrixOpsROCm → rocBLAS CGEMM

vector_algebra:
  ├── MatrixOpsROCm (GpuContext-aware) ✅
  ├── CholeskyInverterROCm (legacy: own handle, manual hiprtc) ❌
  ├── DiagonalLoadRegularizer (legacy: manual hiprtc) ❌
  └── IMatrixRegularizer (Strategy interface) ✅
```

---

## 🔴 Критические проблемы: 0

Оба модуля рабочие и корректные.

---

## 🟡 Важные замечания (3)

### 1. symmetrize hiprtc: отсутствует --offload-arch

**Файл**: `vector_algebra/src/symmetrize_gpu_rocm.cpp:94-95`

```cpp
const char* options[] = {"-O3"};
rtc_err = hiprtcCompileProgram(prog, 1, options);
// ❌ Нет --offload-arch=gfxXXXX → компилируется для default arch
```

Все другие модули передают `--offload-arch` для ISA-оптимизаций. Без него hiprtc генерирует generic код.

### 2. CholeskyInverterROCm — legacy hiprtc для symmetrize kernel

**Файл**: `vector_algebra/src/symmetrize_gpu_rocm.cpp` + `cholesky_inverter_rocm.hpp`

Ручной hiprtc (~80 строк) + manual KernelCacheService. Мог бы использовать GpuContext для компиляции symmetrize kernel.

**Но**: CholeskyInverterROCm управляет своим rocBLAS handle и rocSOLVER (POTRF/POTRI) — эту часть нельзя перенести в GpuContext.

**Решение**: Частичная миграция — только CompileKernels (symmetrize) через GpuContext, остальное (POTRF/POTRI) оставить.

### 3. DiagonalLoadRegularizer — отдельная manual hiprtc компиляция

**Файл**: `vector_algebra/include/diagonal_load_regularizer.hpp`

Комментарий в файле: _"Запуск через hipModuleLaunchKernel — без GpuContext, напрямую через IBackend"_

Отдельная от symmetrize kernel компиляция. Мог бы быть включён в общий GpuContext kernel source.

---

## 🟢 Что отлично

### capon — эталонный Ref03 ⭐⭐⭐⭐⭐
- 5 Op-классов, GpuContext, shared_buf slots — полный Ref03
- CovarianceMatrixOp делегирует в MatrixOpsROCm (DRY)
- Strategy pattern: IMatrixRegularizer → DiagonalLoadRegularizer
- Чёткое разделение: capon → vector_algebra → rocBLAS

### MatrixOpsROCm — чистая обёртка rocBLAS ✅
- CovarianceMatrix, Multiply, MultiplyConjTransA, CGEMM — полный набор
- Column-major (BLAS/LAPACK convention) документирован
- Handle из GpuContext (lazy init, привязан к stream)

### CholeskyInverterROCm — продвинутый rocSOLVER ✅
- POTRF + POTRI + 2 режима симметризации (Roundtrip/GpuKernel)
- Поддержка batched инверсии
- Cross-backend: CPU, void* (ROCm), cl_mem (ZeroCopy)
- Предаллоцированный d_info_ (Task_12: убрана аллокация на каждый вызов)
- SetCheckInfo(false) для benchmark

### compute_capon_relief kernel ✅
- Column-major indexing: `U + m * P` — корректно для BLAS layout
- `Re(conj(u)*w) = u.x*w.x + u.y*w.y` — оптимально (1 FMA)
- Защита от нуля: `(acc > 0.0f) ? (1.0f / acc) : 0.0f`

---

## ✅ Соответствие стандартам GPUWorkLib

| Критерий | vector_algebra | capon |
|----------|---------------|-------|
| Ref03 | ⚠️ MatrixOps ✅, Cholesky/DiagLoad ❌ | ✅ Полный |
| DrvGPU | ✅ IBackend | ✅ IBackend + GpuContext |
| ConsoleOutput | ✅ | ✅ |
| Kernel cache | ✅ (manual) | ✅ (GpuContext) |
| Move semantics | ⚠️ MatrixOps ✅, Cholesky ❌ (non-movable) | ✅ |
| Windows stub | ❌ | ❌ |
| rocBLAS | ✅ Exemplary | ✅ Via MatrixOpsROCm |
| rocSOLVER | ✅ POTRF+POTRI | ✅ Via CaponInvertOp |

---

## 📋 Сводка задач

| # | Приоритет | Модуль | Описание | Сложность |
|---|-----------|--------|----------|-----------|
| 1 | 🟡 | vector_algebra | symmetrize: добавить `--offload-arch` в hiprtc | Низкая |
| 2 | 🟢 | vector_algebra | CholeskyInverterROCm: migrate symmetrize → GpuContext | Средняя |
| 3 | 🟢 | vector_algebra | DiagonalLoadRegularizer: migrate → GpuContext | Средняя |
| 4 | 🟢 | both | Windows stubs | Низкая |

---

*Ревью подготовлено с: sequential-thinking (2 шага), grep analysis по ~31 файлам*
