# Задача для AI-исполнителя: lch_farrow + LfmGeneratorAnalyticalDelay

> **Роль:** Ты — AI-исполнитель. Выполняешь реализацию по плану.  
> **Проверка:** Главный AI (Кодо) проверяет результат — компиляцию, тесты, соответствие спецификации.

---

## 1. ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ ПЕРЕД НАЧАЛОМ

Прочитай **в указанном порядке**:

| № | Файл | Зачем |
|---|------|-------|
| 1 | `CLAUDE.md` | Правила проекта, Python bindings, GPUProfiler, console_output |
| 2 | `MemoryBank/DiscussionPlan/Plan_DelayedFormSignal_Semantics_Farrow.md` | **Главный план** — раздел 7 целиком |
| 3 | `MemoryBank/DiscussionPlan/DelayedFormSignal_Kernel_CORRECT.md` | Корректные формулы kernel (read_pos, frac, center, row) |
| 4 | `MemoryBank/DiscussionPlan/Генератор дробной задержки аналит.md` | Семантика аналитического генератора |
| 5 | `Doc_Addition/Info_FarrowFractionalDelay.md` | Алгоритм Farrow, Lagrange 48×5 |
| 6 | `Examples/GPUProfiler_SetGPUInfo.md` | **Обязательно** SetGPUInfo до Start — иначе «Unknown» в отчёте |
| 7 | `modules/signal_generators/include/generators/lfm_generator.hpp` | Референс: LfmGenerator, LfmParams |
| 8 | `modules/signal_generators/include/generators/delayed_form_signal_generator.hpp` | Референс: API DelayedFormSignalGenerator |
| 9 | `modules/signal_generators/include/i_signal_generator.hpp` | Интерфейс ISignalGenerator |
| 10 | `modules/signal_generators/include/params/signal_request.hpp` | LfmParams, GetChirpRate |
| 11 | `modules/fft_processor/CMakeLists.txt` | Образец структуры модуля |
| 12 | `Python_test/test_delayed_form_signal.py` | Образец Python-теста |
| 13 | `Doc/Python/signal_generators_api.md` | Формат документации Python API |
| 14 | `MemoryBank/DiscussionPlan/lagrange_matrix_48x5.json` | Матрица 48×5 (путь для копирования/загрузки) |

**Опционально (контекст):**
- `MemoryBank/tasks/DelayedFormSignalGenerator_Разногласия.md` — что уже реализовано, отличия от плана
- `MemoryBank/tasks/TASK_DelayedFormSignalGenerator_Farrow.md` — предыдущая задача

---

## 2. ПОРЯДОК ВЫПОЛНЕНИЯ

### Шаг 1: Создать модуль lch_farrow

1. Создать `modules/lch_farrow/` по образцу `modules/fft_processor`:
   - `include/` — заголовки
   - `src/` — исходники
   - `tests/` — тесты
   - `CMakeLists.txt`
2. Скопировать из `modules/signal_generators`:
   - kernel дробной задержки (Lagrange 48×5)
   - матрицу 48×5: встроенную или `MemoryBank/DiscussionPlan/lagrange_matrix_48x5.json`
3. Добавить `add_subdirectory(modules/lch_farrow)` в корневой `CMakeLists.txt`
4. **Результат:** `cmake -B build && cmake --build build` — сборка OK

### Шаг 2: LfmGeneratorAnalyticalDelay

1. Создать класс `LfmGeneratorAnalyticalDelay` в `modules/signal_generators`.  
2. **API:** как у DelayedFormSignalGenerator — `SetParams(LfmParams)`, `SetDelays(delay_us[])`, `GenerateToCpu()` → `vector<vector<complex<float>>>`, `GenerateInputData()` или аналог для GPU. Не `ISignalGenerator` (multi-antenna output).  
3. **Вход:** SystemSampling (fs, length), LfmParams (f_start, f_end, amplitude), `delay_us[]` (float, мкс), antennas  
4. **Выход CPU:** `vector<vector<complex<float>>>` [antenna][sample]  
5. **Выход GPU:** `InputData<cl_mem>` или `cl_mem`  
6. Формула: `chirp_rate = params_.GetChirpRate(duration)`, `phase = π·chirp_rate·t_local² + 2π·f_start·t_local`, `t_local = t − τ`  
7. Граница: `t < τ` → output = 0  
8. Добавить в `Doc/Python/signal_generators_api.md` раздел `LfmGeneratorAnalyticalDelay`  
9. Создать `Python_test/test_lfm_analytical_delay.py` — pytest, сравнение с NumPy (фаза, граница, задержка 3.24 сэмпла → первый ненулевой в индексе 4)

### Шаг 3: lch_farrow — kernel, профилирование, Python

1. Реализовать/доработать kernel по `DelayedFormSignal_Kernel_CORRECT.md`  
   - Формулы: `read_pos = (float)sample_id - delay_samples`, `center = (int)floor(read_pos)`, `frac = read_pos - center`, `row = (uint)(frac * 48) % 48`  
   - Эталон NumPy: `apply_delay_numpy_correct` из раздела 6 DelayedFormSignal_Kernel_CORRECT.md (не старый apply_delay_numpy!)
2. **Профилирование:**  
   - `GPUProfiler::GetInstance()`  
   - `SetGPUInfo(gpu_id, gpu_info)` **до** `profiler.Start()`  
   - `profiler.Record(gpu_id, "LchFarrow", "KernelName", pdata)`  
   - Вывод через `console_output` (мультиGPU-безопасно)
3. Создать `Doc/Python/lch_farrow_api.md`  
4. Создать `Python_test/test_lch_farrow.py` — pytest, сравнение с NumPy и/или LfmGeneratorAnalyticalDelay

---

## 3. КЛЮЧЕВЫЕ ПРАВИЛА

- **Не плодить сущности:** использовать `ISignalGenerator`, `LfmParams`, `SystemSampling`  
- **GPUProfiler:** всегда `SetGPUInfo()` до `Start()` — см. `Examples/GPUProfiler_SetGPUInfo.md`  
- **Консоль:** вывод только через `console_output` из DrvGPU (мультиGPU)  
- **Python:** `Doc/Python/*.md` + `Python_test/test_*.py`
- **Структура:** каждый класс — отдельный `.h` + `.cpp`

---

## 4. ЧЕКЛИСТ ПЕРЕД «ГОТОВО»

- [ ] `cmake -B build && cmake --build build` — успешно
- [ ] C++ тесты (если есть) — проходят
- [ ] `python -m pytest Python_test/test_lfm_analytical_delay.py -v` — проходят
- [ ] `python -m pytest Python_test/test_lch_farrow.py -v` — проходят
- [ ] Doc/Python обновлён
- [ ] GPUProfiler: SetGPUInfo до Start, Record для kernel/Upload/Download

---

## 5. ПЕРЕДАТЬ ГЛАВНОМУ AI (Кодо)

После выполнения напиши:

```
✅ Выполнено:
- [список сделанного]
- [пути к созданным/изменённым файлам]
- [результат сборки]

Проверь, пожалуйста:
1. Компиляция
2. Тесты
3. Соответствие плану
```

---

## 6. ССЫЛКИ НА ВАЖНЫЕ ФАЙЛЫ

- План (согласованный): `MemoryBank/DiscussionPlan/Plan_DelayedFormSignal_Semantics_Farrow.md`  
- Формулы kernel: `MemoryBank/DiscussionPlan/DelayedFormSignal_Kernel_CORRECT.md`  
- GPUProfiler: `Examples/GPUProfiler_SetGPUInfo.md`  
- Правила: `CLAUDE.md`  
- Согласования/разногласия DelayedFormSignal: `MemoryBank/tasks/DelayedFormSignalGenerator_Разногласия.md`
