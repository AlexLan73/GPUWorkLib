# План: Семантика задержки, описание kernel и переход на Farrow

> Сохранено для чтения и дополнения. Дополни — сохрани обратно или отредактируй в репо.

---

## 1. Семантика задержки (зафиксировать в документе)

- **Целая задержка 5:** выход[0..4] = 0; с выход[5] идёт задержанный сигнал; форма повторяет входной сигнал.
- **Дробная задержка 5.23:** «начало» задержки в позиции 5.23; до этого — нули; **первое значение после пересчёта** — в точке **[6]** (т.е. выход[0..5] = 0, с выход[6] — интерполяция).

Текущая реализация в `modules/signal_generators/src/delayed_form_signal_generator.cpp` уже даёт это поведение: `if ((float)sample_id < delay_samples) { output[gid] = 0; return; }`. Имеет смысл явно описать эту семантику в спецификации (например в `MemoryBank/specs/Form_signals.md` или в отдельном подразделе DelayedFormSignal) и в комментарии в kernel.

---

## 2. Подробное описание kernel (документ)

Создать или расширить документ (например `MemoryBank/specs/DelayedFormSignal_Kernel.md` или раздел в `Plan_FractionalDelay_Farrow.md`) с разделами:

- **Назначение:** применение дробной задержки (per-antenna) к комплексному сигналу + опциональный шум.
- **Входы:** `input` (float2, чистый сигнал), `lagrange_matrix` (48×5 или базисная матрица Farrow), `delay_us`, `antennas`, `points`, `sample_rate`, `noise_amplitude`, `norm_val`, `noise_seed`.
- **Выход:** `output` (float2), задержанный сигнал + шум.
- **Формула задержки:**
  - `delay_samples = delay_us * 1e-6 * sample_rate`
  - `D = floor(delay_samples)`, `μ = delay_samples - D` ∈ [0, 1).
  - Для **текущей** реализации: `row = (uint)(μ * 48) % 48`, выходной отсчёт = Σ L[row][k] * input[center - 1 + k] при center = sample_id - D, с нулём при sample_id < delay_samples и при чтении за границами.
- **Границы:** при чтении индексов за [0, points) подставлять 0; при `sample_id < delay_samples` выход = 0.
- **Отличие от настоящего Farrow:** сейчас используется выборка строки матрицы 48×5 по дискретному μ; по ТЗ нужна **структура Farrow** — вычисление коэффициентов как полинома по μ (схема Горнера) в ядре.

Указать в описании ссылки на источники: CCRMA Farrow Structure, `Farrow_GPU_OpenCL_Algorithm.md` (вариант со свёрткой FFT — альтернатива), план `Plan_FractionalDelay_Farrow.md`.

---

## 3. Исследование для перехода на Farrow

- **Context7:** по запросу "farrow fractional delay" подходящей библиотеки не найдено; оставить в плане использование статей и референсов по URL.
- **Статьи по URL (уже подтянуты):**
  - CCRMA: [Farrow Structure](https://ccrma.stanford.edu/~jos/pasp/Farrow_Structure.html) — H_Δ(z) = Σ C_m(z) Δ^m, схема Горнера.
  - CCRMA: [Farrow Structure for Variable Delay](https://ccrma.stanford.edu/~jos/pasp05/Farrow_Structure_Variable_Delay.html) — Lagrange через Farrow, finite difference filters.
- **Локальные материалы:** `Farrow_GPU_OpenCL_Algorithm.md` описывает вариант через **свёртку FFT** (h[48] на CPU из Farrow_coeff .* pw, затем FFT-based свёртка). Текущий код использует **time-domain 5-point** интерполяцию; целевой вариант — time-domain Farrow в ядре: 5 коэффициентов L[k] как полином от μ, схема Горнера.
- **GitHub:** при рабочей авторизации — поиск кода (например "Farrow fractional delay", "Lagrange Horner μ") для референса коэффициентов и формул.
- **Sequential-thinking:** при необходимости разобрать варианты (time-domain Farrow в ядре vs FFT-свёртка; вывод базисной матрицы из 48×5) через MCP sequential-thinking.

---

## 4. Коэффициенты: использование старых

- Текущая матрица `lagrange_matrix_48x5.json` и встроенная `kBuiltinLagrangeMatrix` — 48 строк по 5 коэффициентов Лагранжа для μ = 0/48 … 47/48.
- Для **настоящего Farrow** в ядре нужна матрица **базисных** коэффициентов: каждый из 5 коэффициентов интерполяции — полином 4-й степени по μ:
  - `L[k](μ) = c0[k] + c1[k]*μ + c2[k]*μ² + c3[k]*μ³ + c4[k]*μ⁴`.
  - Тогда матрица размером 5×5 (по одному столбцу на степень μ) задаёт базис; в ядре: для каждого k вычислить L[k] по схеме Горнера по μ, затем выход = Σ L[k]*s[k].
- **Связь со старыми коэффициентами:** строки 48×5 — это выборки L[k](μ) при μ = row/48. По ним можно восстановить коэффициенты полинома (например МНК по 48 точкам или решение по 5 точкам для каждого k). Если полученная базисная матрица даёт при подстановке μ = row/48 те же значения, что в lagrange_matrix_48x5 — старые коэффициенты «подходят» и их можно оставить в виде производной базисной матрицы 5×5 (или 5×5 хранить, 48×5 использовать только для эталона/тестов).
- В плане явно указать: **эталон и тесты** продолжать использовать текущую матрицу 48×5 (или её строки при μ = row/48) для сравнения; после перехода на Farrow в ядре — либо эталон перевести на ту же формулу Горнера с базисной матрицей, либо оставить эталон 48×5 и сравнивать с допустимой погрешностью.

---

## 5. Шаги реализации (без изменений кода в этой сессии)

1. **Документ семантики и kernel:** обновить/создать спецификацию с п. 1 и п. 2 (семантика задержки + подробное описание kernel с формулами и границами).
2. **Исследование Farrow:** оформить краткую выжимку из CCRMA и из Farrow_GPU_OpenCL_Algorithm.md (различие time-domain 5-point vs FFT, формула Farrow через C_m(z) и Горнер) в MemoryBank (например `MemoryBank/research/Farrow_kernel_sources.md` или раздел в DiscussionPlan).
3. **Базисная матрица из 48×5:** при переходе на Farrow — скрипт или небольшая утилита (Python/C++): по lagrange_matrix_48x5 вычислить матрицу 5×5 базисных коэффициентов (полином по μ для каждого из 5 taps); проверить, что при μ = i/48, i=0..47 восстанавливаются исходные строки (или погрешность в пределах допуска).
4. **Переделка kernel:** заменить в ядре выборку строки `row = (uint)(μ*48)%48` на вычисление 5 коэффициентов по полиному от μ (схема Горнера), используя базисную матрицу 5×5 в `__constant`; логику границ и `sample_id < delay_samples` не менять.
5. **Тесты и эталон:** оставить CPU/NumPy эталон с 48×5 для сравнения; при необходимости добавить эталон с формулой Горнера и той же базисной матрицей; зафиксировать допуск по норме ошибки.

---

## 6. Диаграмма (текущая vs целевая)

```mermaid
flowchart LR
  subgraph current [Текущая реализация]
    A1[delay_us, fs]
    A2["D, μ; row = μ*48"]
    A3["L = matrix[row]"]
    A4["out = Σ L[k]*s[k]"]
    A1 --> A2 --> A3 --> A4
  end

  subgraph target [Целевая Farrow]
    B1[delay_us, fs]
    B2["D, μ"]
    B3["L[k] = Horner(μ, base_5x5)"]
    B4["out = Σ L[k]*s[k]"]
    B1 --> B2 --> B3 --> B4
  end
```

---

## Итог

- Семантика задержки явно документируется; текущее поведение (нули до delay_samples, первое значение в ceil(delay_samples)) сохраняется.
- Kernel получает подробное текстовое описание (входы, выходы, формулы, границы, отличия от Farrow).
- Переход на Farrow: исследование по CCRMA и локальным материалам, вывод базисной матрицы из 48×5, замена в ядре на расчёт по Горнеру; старые коэффициенты используются через производную базисную матрицу; эталон при необходимости оставить 48×5 с допуском.
- При рабочей авторизации GitHub — поиск референсного кода; при сложных решениях — sequential-thinking.

---

## 7. План реализации: модуль lch_farrow и аналитический генератор

### 7.1 Цель и независимость задач

- **lch_farrow** — полностью самостоятельный модуль. Не зависит от signal_generators. Связь только на уровне тестов (сравнение данных).
- **Аналитический генератор** — отдельная задача в `modules/signal_generators`. Независим от lch_farrow.
- Переносить только файлы, относящиеся к LCH-Farrow; не трогать файлы, от которых зависят другие генераторы.

### 7.2 Шаг 1: Создание и перенос (копирование) для lch_farrow

**Действия:**
- Создать `modules/lch_farrow/` со структурой по образцу `modules/fft_processor`:
  - `include/` — заголовки
  - `src/` — исходники
  - `tests/` — тесты
  - `CMakeLists.txt`
- Скопировать в lch_farrow всё, что относится к дробной задержке Farrow/Lagrange 48×5:
  - kernel дробной задержки, матрица 48×5
  - `lagrange_matrix_48x5.json` (или ссылка)
- Добавить `add_subdirectory(modules/lch_farrow)` в корневой `CMakeLists.txt`.
- Не трогать: `form_signal_generator`, `form_script_generator`, `signal_service`, `cw_generator`, `lfm_generator`, `noise_generator` и их зависимости.
- Результат: проект собирается (без настройки и тестов переноса).

**Входной интерфейс lch_farrow:**
- `input` — cl_mem или буфер комплексного сигнала (antennas × points)
- `delay_us[]` — задержки per-antenna в микросекундах
- `antennas`, `points`, `sample_rate` (fs)
- `lagrange_matrix` 48×5 (или базисная 5×5 при переходе на Farrow)
- опционально: `noise_amplitude`, `noise_seed`

**Выходной интерфейс lch_farrow:**
- `output` — cl_mem, задержанный сигнал + шум (antennas × points)

**Python API и тест:**
- Документация: `Doc/Python/lch_farrow_api.md` (или раздел в общем Doc)
- Тест: `Python_test/test_lch_farrow.py` — pytest, сравнение с NumPy-эталоном и/или LfmGeneratorAnalyticalDelay

**Профилирование (CLAUDE.md, Examples/GPUProfiler_SetGPUInfo.md):**
- Интеграция с `GPUProfiler` из DrvGPU
- Перед `profiler.Start()` — обязательно `profiler.SetGPUInfo(gpu_id, gpu_info)` (иначе «Unknown» в отчёте)
- `profiler.Record(gpu_id, "LchFarrow", "KernelName", pdata)` — для kernel, Upload, Download
- Вывод — через `console_output` из DrvGPU (мультиGPU-безопасно)

### 7.3 Шаг 2: Генератор дробной задержки (аналитический) — LfmGeneratorAnalyticalDelay

**Источник:** [Генератор дробной задержки аналит.md](Генератор%20дробной%20задержки%20аналит.md)

**Идея:** Для сигнала S(t) задержанный сигнал — S(t − τ). Задержка вводится подстановкой времени в формулу, без интерполяции. «Идеальный» сдвиг для beamforming и пеленгации.

**Класс:** `LfmGeneratorAnalyticalDelay` в `modules/signal_generators`. API как у DelayedFormSignalGenerator (SetParams, SetDelays, GenerateToCpu, GenerateInputData) — не ISignalGenerator из-за multi-antenna output. Использовать `LfmParams`, `SystemSampling` — не плодить сущности. Задержка: массив `delay_us[]` (мкс, значения от нс).

**Входной интерфейс:**
- `SystemSampling` — fs, length (points)
- `LfmParams` — f_start, f_end, amplitude, complex_iq
- `delay_us[]` — задержки per-antenna в микросекундах (float)
- `antennas` — количество каналов (размер delay_us)

**Выходной интерфейс:**
- **CPU:** `std::vector<std::vector<std::complex<float>>>` — [antenna][sample], как у DelayedFormSignalGenerator
- **GPU:** `InputData<cl_mem>` или `cl_mem` — буфер antennas × points × sizeof(complex<float>)

**Формула:** `chirp_rate = LfmParams::GetChirpRate(duration)`, `phase = π·chirp_rate·t_local² + 2π·f_start·t_local`, `t_local = t − τ`.

**Реализация (псевдокод):**
```cpp
// Для сэмпла n, антенны a (beam_id):
double t = n / fs;
double tau = delay_us[a] * 1e-6;  // секунды (мкс → с)
if (t < tau) output[gid] = 0;
else {
  double t_local = t - tau;
  double chirp_rate = params_.GetChirpRate(duration);  // как LfmGenerator
  double phase = M_PI * chirp_rate * t_local*t_local + 2*M_PI * f_start * t_local;
  output[gid] = std::polar(amplitude, phase);
}
```

**Python API и тест:**
- Документация: `Doc/Python/signal_generators_api.md` — добавить раздел `LfmGeneratorAnalyticalDelay`
- Тест: `Python_test/test_lfm_analytical_delay.py` — pytest, сравнение с NumPy-эталоном (фаза, граница t < τ, задержка 3.24 сэмпла → первый ненулевой в индексе 4)

### 7.4 Шаг 3: Возврат к lch_farrow — код и тесты

**Действия:**
- Реализовать/доработать kernel дробной задержки в `modules/lch_farrow` по [Doc_Addition/Info_FarrowFractionalDelay.md](../../Doc_Addition/Info_FarrowFractionalDelay.md).
- Использовать корректные формулы: `read_pos`, `frac`, `center`, `row` (см. [DelayedFormSignal_Kernel_CORRECT.md](DelayedFormSignal_Kernel_CORRECT.md)).
- **Профилирование:** включить GPUProfiler (SetGPUInfo до Start, Record для kernel/Upload/Download), вывод через console_output. Референс: [Examples/GPUProfiler_SetGPUInfo.md](../../Examples/GPUProfiler_SetGPUInfo.md).
- Написать код, оттестировать.
- **Связь на уровне тестов:** сравнивать выход lch_farrow с выходом LfmGeneratorAnalyticalDelay (одинаковые параметры ЛЧМ, задержка; сравнение по норме ошибки).

### 7.5 Порядок выполнения

```
1. Создание modules/lch_farrow + копирование файлов → сборка OK
2. LfmGeneratorAnalyticalDelay в signal_generators → создать, Doc/Python, Python_test
3. lch_farrow: реализация kernel, GPUProfiler, Doc/Python, Python_test
   Связь с аналитическим генератором — только в тестах (сравнение данных)
```

---

## 8. Реализация vs План — отличия

> Проверка 2026-02. Дополнять при выявлении расхождений.

### 8.1 LfmGeneratorAnalyticalDelay

| План (раздел 7.3) | Реализация | Статус |
|-------------------|------------|--------|
| API как DelayedFormSignalGenerator (SetParams, SetDelays, GenerateToCpu, GenerateInputData) | `SetParams(LfmParams)`, `SetSampling(SystemSampling)`, `SetDelays(delay_us)`, `GenerateToGpu()` → InputData, `GenerateToCpu()` → vector<vector<>> | ✅ Соответствует |
| Вход: SystemSampling, LfmParams, delay_us[] | LfmParams в конструкторе, SetSampling, SetDelays | ✅ |
| Выход CPU: vector<vector<complex<float>>> | Реализовано | ✅ |
| Выход GPU: InputData<cl_mem> | Реализовано | ✅ |
| Формула: chirp_rate = GetChirpRate, phase = π·k·t² + 2π·f_start·t | Реализовано в kernel | ✅ |
| Python: Doc/Python, test_lfm_analytical_delay.py | signal_generators_api.md, test_lfm_analytical_delay.py | ✅ |
| Графики | Добавлены 2026-02: Results/Plots/LfmAnalyticalDelay/ (3 PNG) | ✅ |

**Отличие:** Python API — `LfmAnalyticalDelay(ctx, f_start, f_end, amplitude)` (не SetParams из LfmParams). Функционально эквивалентно.

### 8.2 lch_farrow

| План (разделы 7.2, 7.4) | Реализация | Статус |
|-------------------------|------------|--------|
| Модуль modules/lch_farrow | Создан, CMake, add_subdirectory | ✅ |
| Kernel Lagrange 48×5 | lch_farrow_delay kernel в lch_farrow.cpp | ✅ |
| Вход: input, delay_us[], antennas, points, sample_rate | SetDelays, Process(input_buf, antennas, points), SetSampleRate | ✅ |
| Выход: output (cl_mem) | Process возвращает cl_mem | ✅ |
| Формулы read_pos, frac, center, row (DelayedFormSignal_Kernel_CORRECT) | Исправлено 2026-02: read_pos=sample_id−delay_samples, center=floor(read_pos), frac=read_pos−center, row=frac×48. Обновлены: delayed_form_signal.cl, lch_farrow.cpp (kernel + ProcessCpu). | ✅ Исправлено |
| GPUProfiler, SetGPUInfo, Record | Реализовано 2026-02: SetGPUInfo до Record, Record для Upload_delay_us и lch_farrow_delay. Референс: Examples/GPUProfiler_SetGPUInfo.md. | ✅ Реализовано |
| Doc/Python/lch_farrow_api.md | Создан | ✅ |
| Python_test/test_lch_farrow.py | Создан, 5 тестов | ✅ |

### 8.3 plot2_fractional_delay_boundary.png — амплитуды

**Вопрос:** почему разные амплитуды на двух subplot?

**Ответ:** Специально. Верхний subplot — |signal| (огибающая), шкала Y: 0..1. Нижний — Re и Im (компоненты), шкала Y: -1..1. Для комплексного сигнала |z|=1 везде, но Re и Im осциллируют. Разные шкалы по замыслу.

---

## Место для дополнений

<!-- Дополни план ниже или в отдельных разделах -->
