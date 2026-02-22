# TASK-004: Интеграция KernelCacheService в FirFilter и IirFilter

> **План:** [PLAN_KernelCacheService_DrvGPU.md](PLAN_KernelCacheService_DrvGPU.md)  
> **Зависимость:** TASK-002 (KernelCacheService)  
> **Проверка:** Кодо (старшая)

---

## 1. ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ

| № | Файл | Зачем |
|---|------|-------|
| 1 | `modules/filters/src/fir_filter.cpp` | CompileKernel, конструктор |
| 2 | `modules/filters/src/iir_filter.cpp` | Аналогично |
| 3 | `modules/filters/include/kernels/fir_kernels.hpp` | GetFirDirectSource_opencl |
| 4 | `modules/filters/include/kernels/iir_kernels.hpp` | GetIirSource |
| 5 | `PLAN_KernelCacheService_DrvGPU.md` | Раздел 4 — интеграция filters |

---

## 2. ЦЕЛЬ

При создании FirFilter/IirFilter: пытаться загрузить скомпилированный kernel из cache (binary). При отсутствии — компилировать из source и сохранять в cache. Fallback: при ошибке cache — компилировать как сейчас.

**base_dir для filters:** `modules/filters/kernels` (раздельная папка).

**Cache key:** `fir_filter_cf32` (FirFilter), `iir_filter_cf32` (IirFilter) — kernel source не зависит от коэффициентов.

---

## 3. ИЗМЕНЕНИЯ В FirFilter

### 3.1. Добавить член

```cpp
drv_gpu_lib::KernelCacheService kernel_cache_;
```

Или `std::unique_ptr<KernelCacheService>` — создавать при первом обращении.

### 3.2. Путь base_dir

Через CMake define: `FILTERS_KERNELS_DIR` (аналогично SIGNAL_GENERATORS_KERNELS_DIR). Fallback: `"modules/filters/kernels"`.

### 3.3. CompileKernel() — новая логика

```cpp
void FirFilter::CompileKernel() {
  const std::string kernel_name = "fir_filter_cf32";
  auto entry = kernel_cache_.Load(kernel_name);

  if (entry.has_binary()) {
    LoadFromBinary(entry.binary);
    return;
  }

  if (entry.has_source()) {
    LoadFromSource(entry.source);
  } else {
    const char* source = kernels::GetFirDirectSource_opencl();
    LoadFromSource(source);
  }

  try {
    auto binary = GetProgramBinary();
    kernel_cache_.Save(kernel_name, GetCurrentSource(), binary, "", "FIR direct-form");
  } catch (...) {
    // Non-critical: cache save failed
  }
}
```

Нужны: `LoadFromBinary`, `LoadFromSource`, `GetProgramBinary` — аналогично FormScriptGenerator. FirFilter уже имеет `program_`, `context_`, `device_` — добавить приватные методы.

### 3.4. LoadFromBinary, GetProgramBinary

Скопировать логику из form_script_generator.cpp (строки 720-766): `clGetProgramInfo(CL_PROGRAM_BINARIES)`, `clCreateProgramWithBinary`, `clBuildProgram`.

---

## 4. ИЗМЕНЕНИЯ В IirFilter

Аналогично FirFilter:
- kernel_name = `"iir_filter_cf32"`
- base_dir = `modules/filters/kernels`
- Source: `kernels::GetIirSource_opencl()` (или как называется в iir_kernels.hpp)

---

## 5. СТРУКТУРА ПОСЛЕ ИНТЕГРАЦИИ

```
modules/filters/
├── kernels/
│   ├── bin/                    # создаётся при первом запуске
│   │   ├── fir_filter_cf32_opencl.bin
│   │   └── iir_filter_cf32_opencl.bin
│   └── manifest.json
```

---

## 6. CMake

- Добавить в filters CMakeLists.txt: `add_compile_definitions(FILTERS_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")` или передать путь.
- filters линкуется с drvgpu — KernelCacheService доступен.

---

## 7. ТЕСТЫ

- Существующие тесты FirFilter, IirFilter должны пройти.
- Первый запуск: создаётся bin/, manifest.json.
- Второй запуск: Load из cache — быстрее (проверить вручную или добавить тест на наличие bin после Process).

---

## 8. КРИТЕРИИ ПРИЁМКИ

- [ ] FirFilter: CompileKernel использует KernelCacheService
- [ ] IirFilter: аналогично
- [ ] При первом запуске: компиляция + Save в cache
- [ ] При втором запуске: Load из cache (binary)
- [ ] Fallback: при отсутствии/ошибке cache — компиляция из source
- [ ] `pytest Python_test/filters/` (если есть) — проходят
- [ ] C++ тесты filters — проходят

---

## 9. ОТЧЁТ

```
✅ TASK-004 выполнено:
- FirFilter, IirFilter интегрированы с KernelCacheService
- [результаты тестов]

Проверь (Кодо): компиляция, тесты filters, наличие bin/ после запуска.
```
