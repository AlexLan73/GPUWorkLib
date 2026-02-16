# FormSignalGenerator — Спецификация

> **Статус**: Черновик для обсуждения и редактирования
> **Дата**: 2026-02-11

---

## 1. Исходная формула (Python getX)

```python
def getX(fd, f0, a, an, ti, phi=0, fdev=0, tau=0):
    dt = 1 / fd
    N = int(ti * fd + np.finfo(np.float32).eps)
    t = np.linspace(0, ti - dt, N, dtype=np.float64) + tau
    bool_flag = np.logical_or(t < 0, t > ti - dt)
    X = a * norm * np.exp(1j*(2*pi*f0*t + pi*fdev/ti*((t-ti/2)**2) + phi)) + \
        an * norm * (np.random.randn(N) + 1j*np.random.randn(N))
    X[bool_flag] = 0
    return X, t
```

**Параметры**: fd (fs), f0, a (amplitude), an (noise), ti (duration), phi, fdev (chirp), tau (delay).
**norm**: по умолчанию `1/sqrt(2)` для complex IQ (подтверждено: ДА, параметр).

---

## 2. Мультиканальность и фазовая задержка

**Сигналы создаются параллельно на все антенны** (один kernel, gid = antenna_id * points + sample_id).

### 2.1 Задержка (tau) на каждый канал

Нужно задавать задержку **per-antenna**. Два варианта:

#### Вариант 1: Линейный шаг
Задержка увеличивается с фиксированным шагом на каждой антенне:
```
TAU_BASE = 0.0      # базовая задержка (с)
TAU_STEP = 0.0001  # шаг задержки на канал (с)
# Для антенны ID: tau = TAU_BASE + ID * TAU_STEP
```

**DSL [Params]**:
```
TAU_BASE = 0.0
TAU_STEP = 0.0001
```

**В [Defs]**:
```
float tau = TAU_BASE + (float)ID * TAU_STEP;
```

#### Вариант 2: Случайная задержка в диапазоне
Задержка меняется случайным образом в заданном диапазоне для каждой антенны:
```
TAU_MIN = 0.0
TAU_MAX = 0.001
TAU_SEED = 12345   # для воспроизводимости
# tau = TAU_MIN + (random in [0,1]) * (TAU_MAX - TAU_MIN)
```

**Реализация**: использовать Philox PRNG (как в NoiseGenerator) с ключом `(ID, TAU_SEED)` → uniform [0,1] → масштабировать в [TAU_MIN, TAU_MAX].

**DSL [Params]**:
```
TAU_MIN = 0.0
TAU_MAX = 0.001
TAU_SEED = 12345
```

**В [Defs]** (псевдокод, нужна функция philox_uniform):
```
float u = philox_uniform(ID, TAU_SEED);  // [0, 1)
float tau = TAU_MIN + u * (TAU_MAX - TAU_MIN);
```

---

### 2.2 Пример DSL с задержкой (полный)

```
[Params]
ANTENNAS = 8
POINTS = 4096
FS = 12e6
F0 = 1e6
A = 1.0
AN = 0.1
PHI = 0
FDEV = 0

// Вариант 1: линейный шаг
TAU_BASE = 0.0
TAU_STEP = 0.0001

// Вариант 2 (взаимоисключающий): случайный
// TAU_MIN = 0.0
// TAU_MAX = 0.001
// TAU_SEED = 12345

[Defs]
float dt = 1.0f / FS;
float ti = (float)POINTS * dt;
float tau = TAU_BASE + (float)ID * TAU_STEP;  // вариант 1
// float tau = TAU_MIN + philox_uniform(ID, TAU_SEED) * (TAU_MAX - TAU_MIN);  // вариант 2
float t = ((float)T * dt) + tau;
int in_window = (t >= 0.0f && t <= ti - dt) ? 1 : 0;
float phase = 2.0f * M_PI_F * F0 * t + M_PI_F * FDEV / ti * ((t - ti/2.0f) * (t - ti/2.0f)) + PHI;

[Signal]
float res_re = in_window ? (A * cos(phase) + AN * randn_re) : 0.0f;
float res_im = in_window ? (A * sin(phase) + AN * randn_im) : 0.0f;
```

---

## 3. Частотный диапазон

- **`f0`** — центральная частота (Hz). **По умолчанию `f0 = 0.0`** (DC).
- `freq_min`, `freq_max` — опционально для multi-beam: `freq_i = f0 + i * freq_step`, чтобы все частоты в [freq_min, freq_max].
- Диапазон -5..+5 MHz — параметр (валидация при необходимости).

---

## 4. Кэш кернелов — объяснение для начинающих

### 4.1 Что такое «кэш кернелов»?

**OpenCL работает так:**
1. Исходный код kernel (строка на C-подобном языке) → **компиляция** → бинарник для конкретной GPU
2. Компиляция занимает время (секунды)
3. Запуск скомпилированного kernel — миллисекунды

**Кэш** = сохранять результат компиляции, чтобы при повторном вызове **не компилировать заново**.

### 4.2 Два вида кэша

| Вид | Когда помогает | Где хранить |
|-----|----------------|-------------|
| **In-memory** (в памяти) | Один и тот же script вызывается много раз в рамках одной сессии (много `Generate()` подряд) | В объекте ScriptGenerator (поле `program_`) |
| **On-disk** (на диске) | Приложение перезапустили — загрузить готовый бинарник вместо компиляции | Файл в `Results/KernelCache/` или `~/.gpuworklib/` |

### 4.3 Что уже есть

**ScriptGenerator** уже делает in-memory кэш:
- `LoadScript()` → парсинг → компиляция → сохраняет `cl_program` в `program_`
- `Generate()` → использует тот же `program_`, **компиляция не повторяется**
- `LoadScript()` с новым текстом → `ReleaseGpuResources()` → компиляция заново

То есть **повторные вызовы Generate() с тем же script уже не компилируют** — это и есть «кэш повторного вызова».

### 4.4 Вопрос для обсуждения

**On-disk кэш** (сохранение бинарника между запусками приложения):
- Плюсы: быстрый старт при повторном запуске
- Минусы: нужна синхронизация (версия kernel, драйвер, устройство), место на диске
- **Где реализовать?** Варианты:
  - A) В `ScriptGenerator` / `FormScriptGenerator` — проверять файл при `LoadScript()`, загружать если есть
  - B) Отдельный сервис в DrvGPU — но вы сказали **НЕТ** добавлять в DrvGPU
  - C) Пока не делать on-disk, только in-memory (уже есть)

**Нужно решить**: делаем on-disk кэш или достаточно in-memory?

---

## 5. Расположение кернелов

**Кернелы создавать в своей библиотеке:**
```
modules/signal_generators/kernels/
├── form_signal.cl          # формула getX (OpenCL)
├── form_signal.hip         # ROCm stub
└── script_template.cl      # шаблон для DSL (если нужен)
```

Не inline в .cpp, а отдельные файлы. Загрузка: `#include` через CMake или чтение файла при сборке.

**DrvGPU**: НЕ добавлять `kernel_cache.hpp` в DrvGPU. Кэш (если нужен on-disk) — в modules/signal_generators.

---

## 6. Шум на GPU

### 6.1 Что уже есть

**NoiseGenerator** использует **Philox-2x32 PRNG + Box-Muller** на GPU (см. `noise_generator.cpp`).

### 6.2 Варианты для FormSignal

**Вариант A: Встроить в kernel FormSignal**
- Скопировать Philox + Box-Muller из NoiseGenerator в kernel form_signal
- Один kernel: сигнал + шум в одном проходе
- Плюс: один проход по памяти
- Минус: дублирование кода

**Вариант B: Два параллельных процесса, потом сложить**
1. Kernel 1: генерирует только сигнал (без шума) → buffer_signal
2. NoiseGenerator (или отдельный kernel): генерирует шум → buffer_noise
3. Kernel 2 (add): `output[i] = buffer_signal[i] + buffer_noise[i]`
4. Освободить buffer_noise

Плюсы: переиспользование NoiseGenerator, разделение ответственности.
Минусы: два прохода, дополнительная память.

**Рекомендация**: Вариант B — использовать существующий NoiseGenerator, затем kernel сложения. Или вынести Philox+Box-Muller в общий `kernels/prng.cl` и использовать в обоих.

---

## 7. Output (GPU / CPU)

- **GPU**: `cl_mem` (оставить на GPU, вернуть указатель)
- **CPU**: `std::vector<std::vector<std::complex<float>>>` — по каналам (подтверждено: ДА)

---

## 8. Python API и примеры

### 8.1 Пример в новом файле

Создать отдельный файл, например:
```
Python_test/example_form_signal.py
```
или
```
examples/form_signal_demo.py
```

### 8.2 Графики — «как на презентацию»

- Качественные графики (matplotlib или plotly)
- Несколько subplot: time domain (Re/Im), magnitude, спектр (FFT)
- Подписи, сетка, читаемые шрифты
- Сохранение в PNG/PDF для вставки в презентацию

---

## 9. Ответы на вопросы

| # | Вопрос | Ответ |
|---|--------|-------|
| 1 | norm — параметр или 1/sqrt(2)? | ДА, параметр (по умолчанию 1/sqrt(2)) |
| 2 | Шум: Box-Muller в kernel или отдельно? | Использовать существующий NoiseGenerator (Philox+Box-Muller) + kernel сложения ИЛИ встроить в form_signal |
| 3 | Кэш кернелов | Тонкое место — обсудить отдельно (см. раздел 4) |
| 4 | vector<vector<complex<float>>>? | ДА |
| 5 | Частотный диапазон -5..+5 MHz | Параметр f0 (и опционально freq_min, freq_max) |

---

## 10. Вопросы для детального обсуждения

### 10.1 On-disk кэш кернелов

- Нужен ли сохранение скомпилированных бинарников между запусками приложения?
- Если да — где хранить: `Results/KernelCache/`, `~/.gpuworklib/`, путь из configGPU.json?
- Ключ кэша: hash(kernel_source) + device_name? Как инвалидировать при смене драйвера?

### 10.2 Задержка: случайный вариант

- Для `TAU_RANDOM`: передавать pre-computed массив задержек с CPU (buffer) или генерировать в kernel через Philox?
- Если в kernel — нужно добавить `philox_uniform(ID, seed)` в DSL или отдельную функцию?

### 10.3 Шум: Вариант A vs B

- Вариант A (всё в одном kernel): проще вызов, один проход.
- Вариант B (NoiseGenerator + add): переиспользование кода, тестируемость.
- Какой предпочтительнее?

---

## 11. Этапы (без DrvGPU kernel_cache)

**Убрать из плана**: «Добавить DrvGPU/services/kernel_cache.hpp».

Кэш in-memory уже есть в ScriptGenerator. On-disk — по результатам обсуждения.

---

*Редактируйте этот файл, добавляйте ответы и пометки. После согласования — перенести в основной план.*

---

## 12. Анализ дополнений (sequential-thinking)

### 12.1 On-disk кэш кернелов — оценка предложения

**Предложение Alex**: имя кернела → сохранение .cl + binary; при коллизии старые файлы → `name_00`, `name_01`; загрузка по имени; manifest с комментариями; префикс `_opencl`/`_rocm`.

| Аспект | Оценка | Комментарий |
|--------|--------|-------------|
| **Именование** | ✅ Сильно | Человекочитаемые имена (work_sig0) удобнее hash |
| **Версионирование _00** | ✅ Хорошо | Сохраняет историю, ручная очистка — ок |
| **Путь** `modules/[module]/kernels/bin` | ✅ Масштабируемо | Каждый модуль сам за собой, нет общей кучи |
| **Manifest локально** | ✅ Верно | Описание рядом с кернелами модуля |
| **Префикс _opencl/_rocm** | ✅ Решает совместимость | Разные бинарники для разных бэкендов |
| **Переиспользование** | ✅ Важно | Один механизм для signal_generators, filters, и др. |

**Риски**:
- ~~Инвалидация при смене драйвера~~ — **отложено**, см. [DiscussionPlan/~6. KernelCache/Driver_Invalidation_Note.md](../DiscussionPlan/~6.%20KernelCache/Driver_Invalidation_Note.md)
- Конфликт имён между модулями: не критично, т.к. каждый модуль в своей папке.

**Рекомендация**: Принять предложение. Версию драйвера можно добавить в manifest (при загрузке — проверять совпадение).

---

### 12.2 Структура для 20+ модулей

```
modules/
├── signal_generators/
│   └── kernels/
│       ├── bin/                    # скомпилированные бинарники
│       │   ├── work_sig0_opencl.bin
│       │   ├── work_sig0_rocm.hsaco
│       │   └── work_sig0_opencl_00.bin   # старая версия
│       ├── work_sig0.cl            # исходник
│       ├── work_sig0_00.cl         # старая версия
│       ├── manifest.json           # имена, комментарии, дата, драйвер
│       └── README.md               # описание, ссылки (чтобы не путаться!)
├── filters/
│   └── kernels/
│       ├── bin/
│       ├── manifest.json
│       └── README.md
└── ...
```

**Избежание путаницы**: в каждом `modules/[module]/kernels/` — свой `README.md` с описанием кернелов модуля. В `MemoryBank/MASTER_INDEX.md` и `CLAUDE.md` — явные ссылки на структуру (например: «Profiler → `DrvGPU/services/gpu_profiler.hpp`», «Kernels signal_generators → `modules/signal_generators/kernels/`»).

**CMake — конкретно**:
```cmake
# В modules/signal_generators/CMakeLists.txt
set(SIGNAL_GENERATORS_KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
set(SIGNAL_GENERATORS_KERNELS_BIN ${SIGNAL_GENERATORS_KERNELS_DIR}/bin)

# Копировать .cl в build (опционально, если нужен runtime path)
configure_file(
  ${SIGNAL_GENERATORS_KERNELS_DIR}/form_signal.cl
  ${CMAKE_CURRENT_BINARY_DIR}/kernels/form_signal.cl
  COPYONLY
)

# Или: путь к исходникам для загрузки в runtime
target_compile_definitions(signal_generators PRIVATE
  KERNELS_SOURCE_DIR="${SIGNAL_GENERATORS_KERNELS_DIR}"
)
```
Каждый модуль использует `CMAKE_CURRENT_SOURCE_DIR` — без глобальных путей. При сборке большого проекта каждый submodule видит только свои `kernels/`.

---

### 12.3 Output: cl_mem + описание

**Требование**: GPU возвращает `cl_mem` + минимальное описание.

**Что нужно**: адрес (cl_mem), кол-во антенн, точек на антенну, размер в памяти. **Всё это уже есть** в DrvGPU (`BufferInfo`: num_elements, size_bytes; для антенн — num_antennas, points_per_antenna выводятся из контекста или передаются отдельно).

**Не добавлять**: kernel_name, backend_type — излишне. Использовать существующие структуры DrvGPU.

---

### 12.4 Задержка TAU_RANDOM

**Решение**: генерировать в kernel через Philox. Добавить в DSL встроенную функцию `philox_uniform(ID, seed)` → [0, 1). ScriptGenerator при генерации kernel-кода подставляет вызов Philox (код уже есть в noise_generator).

---

### 12.5 Шум: A vs B

**Критерий**: проще и надёжнее.

| | Вариант A (всё в kernel) | Вариант B (NoiseGen + add) |
|---|--------------------------|----------------------------|
| Простота вызова | ✅ Один вызов | ❌ Два kernel + add |
| Надёжность | ✅ Один источник правды | ✅ Переиспользование NoiseGen |
| Тестируемость | Сложнее изолировать | ✅ Шум отдельно |
| Память | ✅ Один буфер | ❌ Два буфера временно |

**Рекомендация**: **Вариант A** — встроить Philox+Box-Muller в form_signal kernel. Причины: один проход, меньше памяти, один вызов. Вынести Philox в `kernels/prng.cl` (общий include) — без дублирования.

---

### 12.6 Итоговые решения

| Вопрос | Решение |
|--------|---------|
| On-disk кэш | ДА. Имя → .cl + .bin в `modules/[module]/kernels/bin`. Manifest. Версионирование _00. |
| Префикс backend | `_opencl`, `_rocm` в имени файла |
| Output | cl_mem + существующие BufferInfo (адрес, size, num_antennas, points). Без новых полей. |
| TAU_RANDOM | Philox в kernel |
| Шум | Вариант A, prng.cl как общий include |
| Путь manifest | Локально: `modules/[module]/kernels/manifest.json` |

---

### Дополнения
## 4. Кэш кернелов — объяснение для начинающих
на обсуждение!!
!!!!!!!!!!!!!!!!!!!  хорошие вопросы!!!
**On-disk кэш** (сохранение бинарника между запусками приложения):
- Плюсы: быстрый старт при повторном запуске
- Минусы: нужна синхронизация (версия kernel, драйвер, устройство), место на диске
- **Где реализовать?** Варианты:
  - A) В `ScriptGenerator` / `FormScriptGenerator` — проверять файл при `LoadScript()`, загружать если есть
  - B) Отдельный сервис в DrvGPU — но вы сказали **НЕТ** добавлять в DrvGPU
  - C) Пока не делать on-disk, только in-memory (уже есть)

**Нужно решить**: делаем on-disk кэш или достаточно in-memory?
можетсохранять керналы? и это механисз у на еще будет с фильтрами много вариантов и эксперементов.
обсуждаем! предложи сама! мой вариант
для каждого кернел в процессе формирования  соэдавать имя и по нему записывать в папку с кернелами (может в подпапку что бы не мешали)
если человек формирует сигнал, сигнал имеет свое имя и параметры кернел собилается и храниться как cl код и двоичный. в следующий раз он хочет использовать сохраненый вариант просто пишет имя и загрузить. программа загружает соответственный кернел. если вызвать команду создать и имя будет такое же к старому варианту дописать номер _00 потом 01  и так далее - к примеру был кернел и бинарник work_sig0.cl и work_sig0.бинарный после повторной команды создать сигнал с именем work_sig0 у нас на диске старые файлы станут work_sig0.cl_00 и work_sig0.бинарный_00 и так далее 
потом в ручном режиме удалить. так жк можно добавить комментарии во время формирования 
во время создания задавать имя комментарии и код. можно вести файл с названием кернелов с комментариями
- обрати внимание этот механизм может быть использован в разных модулях! значит значит файл с описание должен лежать локально со своими кернелами

## 5. Расположение кернелов
Не inline в .cpp, а отдельные файлы. Загрузка: `#include` через CMake или чтение файла при сборке.
Расматривай задачу таким образом у нас будет > 20 модулей у всех свой CMake и это будет собираться в очень большой проект. нужно делать как можно проще и с маштабированием!

## 7. Output (GPU / CPU)

- **GPU**: `cl_mem` (оставить на GPU, вернуть указатель) 
и мета файл! описание должно быть в интерфейсе DrvGPU

## 8. Python API и примеры
### 8.1 Пример в новом файле
Создать отдельный файл, например:
```
Python_test/example_form_signal.py
```

## 10. ОТВЕТЫ

### 10.1 On-disk кэш кернелов

- Нужен ли сохранение скомпилированных бинарников между запусками приложения? - Да выше описал
- Если да — где хранить: `Results/KernelCache/`, `~/.gpuworklib/`, путь из configGPU.json?
предлагаю хранить modules/signal_generators/kernels/bin (или что то такое ) 
modules/[название модуля]/kernels/bin
- Ключ кэша: hash(kernel_source) + device_name? Как инвалидировать при смене драйвера?
 !!! пожет автоматически добавлять к кернелам прификс _opencl or _rocm?

### 10.2 Задержка: случайный вариант

- Для `TAU_RANDOM`: передавать pre-computed массив задержек с CPU (buffer) или генерировать в kernel через Philox?
- Если в kernel — нужно добавить `philox_uniform(ID, seed)` в DSL или отдельную функцию?
генерировать в kernel!!

### 10.3 Шум: Вариант A vs B

- Вариант A (всё в одном kernel): проще вызов, один проход.
- Вариант B (NoiseGenerator + add): переиспользование кода, тестируемость.
- Какой предпочтительнее?

- смотри как проще и надежней


