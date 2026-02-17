# Checklist: FormSignalGenerator

> **Дата создания**: 2026-02-11
> **Статус**: ✅ ЗАВЕРШЕНО (все 6 этапов выполнены)
> **Цель**: Мультиканальный генератор комплексных сигналов (формула getX) с задержкой, амплитудой, шумом. C++/Python API, DSL, on-disk кэш кернелов.
> **Спецификация**: [MemoryBank/specs/Form_signals.md](../specs/Form_signals.md)

---

## Образец для реализации

| Что | Референс |
|-----|----------|
| DSL → kernel | [ScriptGenerator](modules/signal_generators/include/generators/script_generator.hpp) |
| Документация | [ScriptGenerator.md](Doc/Modules/signal_generators/ScriptGenerator.md) |
| Python + графики | [test_gpuworklib.py](Python_test/test_gpuworklib.py) — Test 8, Test 9 |
| Параметрический генератор | [CwGenerator](modules/signal_generators/include/generators/cw_generator.hpp) |
| C++ тесты | [test_signal_generators.hpp](modules/signal_generators/tests/test_signal_generators.hpp) |
| Python bindings | [gpu_worklib_bindings.cpp](python/gpu_worklib_bindings.cpp) |

---

## Этап 1: FormParams и FormSignalGenerator (2–3 дня) ✅ DONE (2026-02-17)

### FormParams
- [x] Создать `form_params.hpp` с парсером из строки
- [x] Параметры: fs=12e6, f0=0.0, freq_min/max, amplitude, noise_amplitude, delay, phase, fdev, norm, noise_seed
- [x] Парсинг: `"f0=1e6,a=1.0,an=0.1,tau=0.001"`

### FormSignalGenerator (OpenCL)
- [x] Создать `form_signal_generator.hpp` / `.cpp`
- [x] Kernel inline в `.cpp` (Philox+Box-Muller+getX в одном source)
- [x] Формула getX: X = a*norm*exp(j*(2πf0*t + π*fdev/ti*((t-ti/2)²) + phi)) + an*norm*(randn + j*randn)
- [x] Окно: X=0 при t<0 или t>ti-dt
- [x] Philox+Box-Muller встроен в kernel (Вариант A — один проход)
- [x] Multi-channel: параллельно на все антенны (gid = antenna_id * points + sample_id)
- [x] Память: `backend->Allocate()`, очередь: `backend->GetNativeQueue()`
- [x] Output: GPU (cl_mem), CPU (vector<vector<complex<float>>>)
- [x] Unit-тест C++: 6 тестов, сравнение с CPU reference ✅
- [x] Python bindings (pybind11): PyFormSignalGenerator ✅
- [x] Python тесты: 7 тестов + NumPy reference ✅
- [x] Python графики: 6 публикационных графиков ✅

### Задержка per-channel
- [x] TAU_STEP: `tau = TAU_BASE + ID * TAU_STEP`
- [x] TAU_RANDOM: Philox uniform в kernel, `tau = TAU_MIN + u*(TAU_MAX - TAU_MIN)`

### Файлы Этапа 1
- `modules/signal_generators/include/params/form_params.hpp` (NEW)
- `modules/signal_generators/include/generators/form_signal_generator.hpp` (NEW)
- `modules/signal_generators/src/form_signal_generator.cpp` (NEW)
- `modules/signal_generators/tests/test_form_signal.hpp` (NEW)
- `python/gpu_worklib_bindings.cpp` (MODIFIED — PyFormSignalGenerator)
- `Python_test/test_form_signal.py` (NEW — 7 тестов + 6 графиков)
- `Results/Plots/FormSignal/` (6 PNG)

---

## Этап 2: FormScriptGenerator и DSL (2 дня) ✅ DONE (2026-02-17)

### FormScriptGenerator
- [x] Создать `form_script_generator.hpp` / `.cpp`
- [x] Самостоятельный генератор с DSL (FormParams → скрипт + OpenCL kernel)
- [x] Preset-скрипт для getX из FormParams (GenerateScript())
- [x] DSL: TAU_BASE/TAU_STEP или TAU_MIN/TAU_MAX/TAU_SEED
- [x] Функция `philox_uniform(ID, seed)` встроена в kernel
- [x] Параметры как #define → оптимизация OpenCL компилятором
- [x] Kernel с 1 аргументом (output) вместо 18

### On-disk кэш кернелов
- [x] Сохранение по имени: `name.cl` + `bin/name_opencl.bin`
- [x] При коллизии: старые → `name_00.cl`, `name_opencl_00.bin`
- [x] Manifest: `manifest.json` (имена, комментарии, дата, params)
- [x] Загрузка по имени (binary fast path → source fallback)
- [x] Префикс `_opencl` в имени бинарника (`_rocm` — при появлении ROCm)
- [x] README.md в `kernels/` для навигации
- [x] Python API: save_kernel(), load_kernel(), list_kernels()

### CMake
- [x] `SIGNAL_GENERATORS_KERNELS_DIR` через target_compile_definitions
- [x] Только `CMAKE_CURRENT_SOURCE_DIR`, без глобальных путей
- [x] stdc++fs для std::filesystem

### C++ тесты (7/7 PASS)
1. DSL генерация — [Params]/[Defs]/[Signal]
2. Compile + Generate vs FormSignalGenerator (err < 1e-3)
3. SaveKernel — файлы на диске (.cl + .bin + manifest)
4. LoadKernel — загрузка + генерация (err = 0)
5. Versioning — _00, _01 при коллизии
6. ListKernels — из manifest.json
7. Chirp + noise — сложная конфигурация

### Python bindings
- [x] PyFormScriptGenerator с полным API
- [x] Python test: compile → generate → save → load → generate → compare

### Файлы Этапа 2
- `modules/signal_generators/include/generators/form_script_generator.hpp` (NEW)
- `modules/signal_generators/src/form_script_generator.cpp` (NEW)
- `modules/signal_generators/tests/test_form_script.hpp` (NEW)
- `modules/signal_generators/kernels/README.md` (NEW)
- `modules/signal_generators/kernels/bin/` (NEW — auto-created)
- `modules/signal_generators/CMakeLists.txt` (MODIFIED)
- `modules/signal_generators/tests/all_test.hpp` (MODIFIED)
- `python/gpu_worklib_bindings.cpp` (MODIFIED)

---

## Этап 3: SignalService и Factory (0.5 дня) ✅ DONE (2026-02-17)

- [x] SignalKind::FORM_SIGNAL + FormParams в variant SignalRequest
- [x] SignalGeneratorFactory::CreateForm(backend, FormParams) → FormSignalGenerator
- [x] SignalGeneratorFactory::CreateFormScript(backend, FormParams) → FormScriptGenerator
- [x] SignalService::GenerateFormGpu(FormParams) → InputData<cl_mem>
- [x] SignalService::GenerateFormCpu(FormParams) → vector<vector<complex>>
- [x] Create(FORM_SIGNAL) → бросает исключение с пояснением (standalone API)

### Файлы Этапа 3
- `include/params/signal_request.hpp` (MODIFIED — SignalKind, variant)
- `include/signal_generator_factory.hpp` (MODIFIED — CreateForm, CreateFormScript)
- `src/signal_generator_factory.cpp` (MODIFIED)
- `include/signal_service.hpp` (MODIFIED — GenerateFormGpu, GenerateFormCpu)
- `src/signal_service.cpp` (MODIFIED)

---

## Этап 4: Python bindings (1–2 дня) ✅ DONE (из Этапов 1-2-3)

- [x] PyFormSignalGenerator: generate(), set_params(), set_params_from_string() ✅ Этап 1
- [x] PyFormScriptGenerator: compile(), generate(), save/load_kernel(), list_kernels() ✅ Этап 2
- [x] np.ndarray shape (n_channels, n_samples) complex64 ✅
- [x] Output: "cpu" | "gpu" — реализовано (GPUBuffer + .read())

---

## Этап 5: Пример и документация (1 день) ✅ DONE (2026-02-17)

- [x] Создать `Python_test/example_form_signal.py` — 5 демо + 5 графиков
- [x] Графики как на презентацию (time, magnitude, спектр FFT, waterfall, DSL)
- [x] Сохранение PNG в `Results/Plots/FormSignal/`
- [x] Создать `Doc/Python/signal_generators_api.md` — полная документация API
- [x] Сравнение с NumPy getX (reference) — Demo 5

### Файлы Этапа 5
- `Python_test/example_form_signal.py` (NEW — 5 демо, 5 графиков)
- `Doc/Python/signal_generators_api.md` (NEW — полная документация Python API)

---

## Этап 6: ROCm ветка (заглушки) ✅ DONE (2026-02-17)

- [x] FormSignalGeneratorROCm — header-only stub (throw "not implemented")
- [x] form_signal.hip — HIP kernel заглушка в `kernels/rocm/`
- [x] SignalGeneratorFactory::CreateFormROCm() — отдельный метод для ROCm
- [x] Factory include `form_signal_generator_rocm.hpp` + `backend_type.hpp`

### Файлы Этапа 6
- `modules/signal_generators/include/generators/form_signal_generator_rocm.hpp` (NEW)
- `modules/signal_generators/kernels/rocm/form_signal.hip` (NEW)
- `modules/signal_generators/include/signal_generator_factory.hpp` (MODIFIED — CreateFormROCm)
- `modules/signal_generators/src/signal_generator_factory.cpp` (MODIFIED)

---

## Output и метаданные

- [x] GPU: InputData<cl_mem> (data, antenna_count, n_point, gpu_memory_bytes) — реализовано
- [ ] CPU: vector<vector<complex<float>>> ✅ реализовано

---

## Ссылки

- Спецификация: [Form_signals.md](../specs/Form_signals.md)
- Python API docs: [Doc/Python/signal_generators_api.md](../../Doc/Python/signal_generators_api.md)
- Driver invalidation (отложено): [DiscussionPlan/~6. KernelCache/Driver_Invalidation_Note.md](../DiscussionPlan/~6.%20KernelCache/Driver_Invalidation_Note.md)

---

*✅ Все 6 этапов завершены. 2026-02-17.*
