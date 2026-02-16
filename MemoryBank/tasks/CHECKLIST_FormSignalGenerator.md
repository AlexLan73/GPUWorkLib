# Checklist: FormSignalGenerator

> **Дата создания**: 2026-02-11
> **Статус**: ⬜ Не начато
> **Цель**: Мультиканальный генератор комплексных сигналов (формула getX) с задержкой, амплитудой, шумом. C++/Python API, DSL, on-disk кэш кернелов.
> **Спецификация**: [MemoryBank/specs/Form_signals.md](../specs/Form_signals.md)

---

## Этап 1: FormParams и FormSignalGenerator (2–3 дня)

### FormParams
- [ ] Создать `form_params.hpp` с парсером из строки
- [ ] Параметры: fs=12e6, f0=0.0, freq_min/max, amplitude, noise_amplitude, delay, phase, fdev, norm, noise_seed
- [ ] Парсинг: `"f0=1e6,a=1.0,an=0.1,tau=0.001"`

### FormSignalGenerator (OpenCL)
- [ ] Создать `form_signal_generator.hpp` / `.cpp`
- [ ] Kernel в `modules/signal_generators/kernels/form_signal.cl`
- [ ] Формула getX: X = a*norm*exp(j*(2πf0*t + π*fdev/ti*((t-ti/2)²) + phi)) + an*norm*(randn + j*randn)
- [ ] Окно: X=0 при t<0 или t>ti-dt
- [ ] Philox+Box-Muller в kernel (вынести в `kernels/prng.cl` как include)
- [ ] Multi-channel: параллельно на все антенны (gid = antenna_id * points + sample_id)
- [ ] Память: `backend->Allocate()`, очередь: `backend->GetNativeQueue()`
- [ ] Output: GPU (cl_mem), CPU (vector<vector<complex<float>>>)
- [ ] Unit-тест: сравнение с NumPy getX

### Задержка per-channel
- [ ] TAU_STEP: `tau = TAU_BASE + ID * TAU_STEP`
- [ ] TAU_RANDOM: Philox uniform в kernel, `tau = TAU_MIN + u*(TAU_MAX - TAU_MIN)`

---

## Этап 2: FormScriptGenerator и DSL (2 дня)

### FormScriptGenerator
- [ ] Создать `form_script_generator.hpp` / `.cpp`
- [ ] Обёртка над ScriptGenerator, маппинг FormParams → [Params]
- [ ] Preset-скрипт для getX из FormParams
- [ ] DSL: TAU_BASE/TAU_STEP или TAU_MIN/TAU_MAX/TAU_SEED
- [ ] Функция `philox_uniform(ID, seed)` в DSL

### On-disk кэш кернелов
- [ ] Сохранение по имени: `work_sig0.cl` + `work_sig0_opencl.bin` в `modules/signal_generators/kernels/bin/`
- [ ] При коллизии: старые → `work_sig0_00.cl`, `work_sig0_opencl_00.bin`
- [ ] Manifest: `manifest.json` (имена, комментарии, дата)
- [ ] Загрузка по имени
- [ ] Префикс `_opencl` / `_rocm` в имени бинарника
- [ ] README.md в каждом kernels/ для навигации

### CMake
- [ ] `SIGNAL_GENERATORS_KERNELS_DIR`, `SIGNAL_GENERATORS_KERNELS_BIN`
- [ ] `configure_file` или `KERNELS_SOURCE_DIR` для runtime
- [ ] Только `CMAKE_CURRENT_SOURCE_DIR`, без глобальных путей

---

## Этап 3: SignalService и Factory (0.5 дня)

- [ ] SignalService::GenerateForm(FormParams, SystemSampling, beam_count)
- [ ] SignalGeneratorFactory::CreateForm()
- [ ] SignalRequest: добавить FormParams в variant

---

## Этап 4: Python bindings (1–2 дня)

- [ ] PyFormSignalGenerator: generate(), generate_from_string()
- [ ] PyFormScriptGenerator (или расширение PyScriptGenerator)
- [ ] np.ndarray shape (n_channels, n_samples) complex64
- [ ] Output: "cpu" | "gpu"

---

## Этап 5: Пример и документация (1 день)

- [ ] Создать `Python_test/example_form_signal.py` (или `examples/form_signal_demo.py`)
- [ ] Графики как на презентацию (time, magnitude, спектр FFT)
- [ ] Сохранение PNG/PDF
- [ ] Обновить `Doc/Python/signal_generators_api.md`
- [ ] Сравнение с NumPy getX (reference)

---

## Этап 6: ROCm ветка (заглушки)

- [ ] FormSignalGeneratorROCm — stub (throw "not implemented")
- [ ] form_signal.hip — заглушка в `kernels/rocm/`
- [ ] SignalGeneratorFactory::CreateForm() — ветвление по BackendType
- [ ] `#ifdef USE_ROCM` в factory

---

## Output и метаданные

- [ ] GPU: cl_mem + существующие BufferInfo (адрес, size, num_antennas, points)
- [ ] Не добавлять kernel_name, backend_type
- [ ] CPU: vector<vector<complex<float>>>

---

## Ссылки

- Спецификация: [Form_signals.md](../specs/Form_signals.md)
- План: `.cursor/plans/formsignalgenerator_plan_*.plan.md`
- Driver invalidation (отложено): [DiscussionPlan/~6. KernelCache/Driver_Invalidation_Note.md](../DiscussionPlan/~6.%20KernelCache/Driver_Invalidation_Note.md)

---

*Отмечай галочки по мере выполнения.*
