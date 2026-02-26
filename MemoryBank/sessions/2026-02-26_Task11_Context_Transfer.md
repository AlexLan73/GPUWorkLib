# Context Transfer — Task_11 VectorAlgebra v2

> **Дата**: 2026-02-26
> **Статус**: ✅ РЕАЛИЗАЦИЯ ЗАВЕРШЕНА, нужна сборка + тесты

---

## Что было сделано (Task_11_VectorAlgebraCholesky_v2.md)

Полная переработка модуля `vector_algebra` — 9 групп работ.

### Архитектурные изменения (Task_10 → Task_11)

| Аспект | Task_10 (старое) | Task_11 v2 (новое) |
|--------|-------------------|---------------------|
| CholeskyResult | template `<T>` + `vector<T> data` | non-template, `void* d_data` (RAII, GPU memory ownership) |
| Symmetrize | Только CPU roundtrip | `SymmetrizeMode`: Roundtrip (CPU) / GpuKernel (hiprtc) |
| Kernel | нет | `symmetrize_gpu_rocm.cpp` + `symmetrize_kernel_sources_rocm.hpp` (hiprtc) |
| Тесты C++ | ~8 тестов | 10 functional × 2 modes + 4 cross-backend + 3 benchmark + 1 profiler = ~30 |
| Python | 5 тестов | 6 тестов с SymmetrizeMode |

### Изменённые/созданные файлы

**Core (переписано):**
- `modules/vector_algebra/include/vector_algebra_types.hpp` — SymmetrizeMode enum, CholeskyResult (RAII)
- `modules/vector_algebra/include/cholesky_inverter_rocm.hpp` — новый интерфейс
- `modules/vector_algebra/src/cholesky_inverter_rocm.cpp` — Core POTRF/POTRI + dispatchers

**Новые файлы:**
- `modules/vector_algebra/include/kernels/symmetrize_kernel_sources_rocm.hpp` — HIP kernel source
- `modules/vector_algebra/src/symmetrize_gpu_rocm.cpp` — hiprtc compile + launch
- `modules/vector_algebra/tests/test_cross_backend_conversion.hpp` — 4 cross-backend теста
- `modules/vector_algebra/tests/test_benchmark_symmetrize.hpp` — 3 benchmark + profiler

**Обновлено:**
- `modules/vector_algebra/CMakeLists.txt` — добавлен symmetrize_gpu_rocm.cpp + hiprtc link
- `modules/vector_algebra/tests/test_cholesky_inverter_rocm.hpp` — 10 тестов с SymmetrizeMode param
- `modules/vector_algebra/tests/all_test.hpp` — запуск в обоих режимах
- `python/py_vector_algebra_rocm.hpp` — SymmetrizeMode enum + set/get
- `Python_test/vector_algebra/test_cholesky_inverter_rocm.py` — 6 тестов
- `modules/vector_algebra/tests/README.md` — полное описание тестов

**Документация:**
- `Doc/Python/vector_algebra_api.md` — обновлён (v2 с SymmetrizeMode)
- `MemoryBank/MASTER_INDEX.md` — обновлён
- `MemoryBank/tasks/TASKS_ROCm_INDEX.md` — обновлён

**НЕ менялись (уже были готовы):**
- `cmake/dependencies.cmake` — rocblas + rocsolver + hiprtc уже есть
- `CMakeLists.txt` (root) — add_subdirectory уже есть
- `DrvGPU/interface/input_data_traits.hpp` — is_cl_mem_v уже есть
- `python/gpu_worklib_bindings.cpp` — register_cholesky_inverter_rocm(m) уже есть
- `python/CMakeLists.txt` — vector_algebra link уже есть

---

## Что нужно сделать дальше

### 1. Сборка (обязательно)
```bash
cd /home/alex/C++/GPUWorkLib
mkdir -p build && cd build
cmake .. -DENABLE_ROCM=ON -DBUILD_PYTHON=ON -DPython3_EXECUTABLE=$(which python3)
make -j$(nproc)
```

### 2. C++ тесты
```bash
./GPUWorkLib  # запускает RunVectorAlgebraTests(backend)
```
Ожидаемый вывод: ~30 тестов PASSED в двух режимах (Roundtrip + GpuKernel).

### 3. Python тесты
```bash
cd /home/alex/C++/GPUWorkLib
pytest Python_test/vector_algebra/test_cholesky_inverter_rocm.py -v
```
Ожидается: 6 тестов PASSED.

### 4. Возможные проблемы при сборке
- **hiprtc не найден**: Проверить `cmake/dependencies.cmake` — ищем `hiprtc::hiprtc` target
- **Ошибки линковки**: `symmetrize_gpu_rocm.cpp` требует hiprtc headers
- **Runtime ошибки kernel**: Kernel source string в `symmetrize_kernel_sources_rocm.hpp` — проверить синтаксис HIP

### 5. После успешных тестов
- Отметить Task_11 как COMPLETED в `MemoryBank/tasks/Task_11_VectorAlgebraCholesky_v2.md`
- Раскомментировать benchmarks в `all_test.hpp` для замера производительности
- Опционально: запуск benchmarks для сравнения Roundtrip vs GpuKernel

---

## Ключевые паттерны для понимания кода

### CholeskyResult (RAII)
```cpp
struct CholeskyResult {
  void* d_data = nullptr;       // GPU memory (owned)
  size_t byte_size = 0;
  int n = 0;
  int batch_count = 0;
  IBackend* backend_ = nullptr;

  ~CholeskyResult();                              // Free GPU memory
  CholeskyResult(CholeskyResult&&);               // Move only
  std::vector<std::complex<float>> AsVector();    // GPU → CPU
  void* AsHipPtr();                               // Raw GPU pointer
  std::vector<std::vector<std::complex<float>>> matrix();    // n×n
  std::vector<std::vector<std::vector<std::complex<float>>>> matrices(); // batch×n×n
};
```

### SymmetrizeMode dispatch
```cpp
void Symmetrize(void* d_data, int n) {
  if (mode_ == SymmetrizeMode::Roundtrip)
    SymmetrizeRoundtrip(d_data, n);
  else
    SymmetrizeGpuKernel(d_data, n);
}
```

### hiprtc lazy compilation
```cpp
void CompileKernels() {
  if (kernels_compiled_) return;
  // hiprtcCreateProgram → hiprtcCompileProgram → hipModuleLoadData → hipModuleGetFunction
  kernels_compiled_ = true;
}
```

---

*Создан: 2026-02-26 | Кодо*
