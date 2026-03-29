# Промт для создания полного Doxygen для GPUWorkLib

## Как вызвать

В Claude Code (или Claude в VS Code) открой проект `E:\C++\GPUWorkLib` и вставь промт ниже.

---

## Промт (скопируй и вставь)

```
Кодо, создай полную и красивую Doxygen документацию для проекта GPUWorkLib.
Используй sequential-thinking для планирования.
Используй агентов для параллельной работы над модулями.

Возьми за образец то, что мы сделали для discriminator_estimates в проекте E:\C++\Refactoring:
- Главная страница с описанием проекта, таблицей модулей, графиками
- Страница формул для каждого модуля (MathJax)
- Страница тестов (C++, GTest, Python) с картинками из Results/Plots/
- Страница архитектуры (Ref03, 6-слойная модель)
- Формулы @f$ в Doxygen комментариях всех публичных функций в заголовках

### ВАЖНО: Модульная структура + связи через TAGFILES!

Это большой проект — Doxygen организован **по папкам-модулям**.
Модули **связаны между собой** через механизм TAGFILES:
- Каждый модуль генерирует .tag файл (индекс своих символов)
- Главный Doxyfile подключает все .tag файлы → перекрёстные ссылки между модулями
- Из главной страницы кликаешь → переходишь в HTML модуля
- Из модуля кликаешь на класс DrvGPU → переходишь в HTML DrvGPU

### Схема связей

```
                    ┌──────────────────┐
                    │  Главный Doxyfile │
                    │  (TAGFILES = все) │
                    │  html/index.html │
                    └───────┬──────────┘
                            │ подключает .tag файлы
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌──────────────┐ ┌─────────────┐ ┌─────────────┐
    │   DrvGPU/    │ │  modules/   │ │  modules/   │
    │  Doxyfile    │ │ sig_gen/    │ │ fft_func/   │  ...
    │  .tag ←──────┤ │ Doxyfile    │ │ Doxyfile    │
    │  html/       │ │ .tag ←──────┤ │ .tag        │
    └──────────────┘ └─────────────┘ └─────────────┘
         ↑                  ↑               ↑
         └──────────────────┴───────────────┘
              модули ссылаются на DrvGPU
```

### Структура файлов

```
Doc/Doxygen/
├── Doxyfile                  # Главный (собирает ВСЁ, подключает все .tag)
├── build_docs.bat            # Скрипт сборки (порядок: модули → главный)
├── build_docs.sh             # То же для Linux
├── html/                     # HTML всего проекта (с перекрёстными ссылками)
│
├── DrvGPU/                   # --- DrvGPU (центральный компонент) ---
│   ├── Doxyfile              # INPUT = ../../../DrvGPU, GENERATE_TAGFILE = drvgpu.tag
│   ├── drvgpu.tag            # (генерируется) индекс символов DrvGPU
│   ├── pages/
│   │   ├── mainpage.md       # Описание: backends, memory, profiler
│   │   ├── architecture.md   # OpenCL/ROCm, CommandQueue, BatchManager
│   │   ├── gpu_profiler.md   # GPUProfiler API + примеры
│   │   └── console_output.md # ConsoleOutput (multi-GPU safe)
│   └── html/                 # HTML только DrvGPU
│
├── modules/                  # --- Модули (каждый со своим .tag) ---
│   ├── signal_generators/
│   │   ├── Doxyfile          # GENERATE_TAGFILE = signal_generators.tag
│   │   │                     # TAGFILES = ../../DrvGPU/drvgpu.tag=../../DrvGPU/html
│   │   ├── signal_generators.tag  # (генерируется)
│   │   ├── pages/
│   │   │   ├── overview.md   # CW, LFM, Noise, Script генераторы
│   │   │   ├── formulas.md   # Формулы MathJax
│   │   │   └── tests.md      # Тесты + @image html графики
│   │   └── html/
│   │
│   ├── fft_func/
│   │   ├── Doxyfile          # TAGFILES = ../../DrvGPU/drvgpu.tag=...
│   │   ├── fft_func.tag
│   │   ├── pages/
│   │   │   ├── overview.md   # hipFFT/clFFT
│   │   │   ├── formulas.md   # DFT, IDFT, окна
│   │   │   └── tests.md
│   │   └── html/
│   │
│   ├── filters/              # FIR, IIR
│   ├── statistics/           # Mean, Median, Welford
│   ├── heterodyne/           # NCO, Dechirp
│   ├── vector_algebra/       # Cholesky, матрицы
│   ├── capon/                # Capon beamforming
│   ├── range_angle/          # Дальность-угол
│   ├── fm_correlator/        # ЧМ корреляция
│   ├── strategies/           # Pipeline
│   └── lch_farrow/           # Фарроу интерполяция
│       ├── Doxyfile
│       ├── lch_farrow.tag
│       ├── pages/
│       └── html/
│
└── pages/                    # --- Общие страницы ---
    ├── mainpage.md           # Главная: таблица ВСЕХ модулей, ссылки
    ├── architecture.md       # Ref03: 6-слойная модель
    ├── build_guide.md        # Сборка: CMake, ROCm, OpenCL
    └── tests_overview.md     # Обзор тестирования
```

### Шаг 1: Модульные Doxyfile (с TAGFILES)

Каждый модульный Doxyfile должен содержать:

```
# modules/signal_generators/Doxyfile — ПРИМЕР

PROJECT_NAME     = "SignalGenerators"
PROJECT_BRIEF    = "Генераторы сигналов на GPU (CW, LFM, Noise, Script)"
INPUT            = ../../../../modules/signal_generators/include \
                   ../../../../modules/signal_generators/src \
                   pages
FILE_PATTERNS    = *.h *.hpp *.cpp *.cl *.md
RECURSIVE        = YES
OUTPUT_DIRECTORY = .
HTML_OUTPUT      = html

# --- Формулы ---
USE_MATHJAX      = YES
MATHJAX_VERSION  = MathJax_3

# --- Картинки ---
IMAGE_PATH       = ../../../../Results/Plots/signal_generators

# --- TAGFILE: генерируем свой .tag ---
GENERATE_TAGFILE = signal_generators.tag

# --- TAGFILES: подключаем DrvGPU (ссылки на классы DrvGPU) ---
TAGFILES         = ../../DrvGPU/drvgpu.tag=../../DrvGPU/html

# --- Общие настройки ---
OUTPUT_LANGUAGE  = Russian
EXTRACT_ALL      = YES
SOURCE_BROWSER   = YES
OPTIMIZE_OUTPUT_FOR_C = NO
HAVE_DOT         = NO
GENERATE_LATEX   = NO
JAVADOC_AUTOBRIEF = YES
```

DrvGPU Doxyfile аналогично, но БЕЗ TAGFILES (он ни от кого не зависит):
```
GENERATE_TAGFILE = drvgpu.tag
# TAGFILES не нужен — DrvGPU самодостаточен
```

### Шаг 2: Главный Doxyfile (подключает ВСЕ .tag)

```
# Doc/Doxygen/Doxyfile — ГЛАВНЫЙ

PROJECT_NAME     = "GPUWorkLib — GPU Signal Processing Library"
PROJECT_BRIEF    = "Библиотеки GPU-вычислений для ЦОС (OpenCL, ROCm/HIP)"
INPUT            = ../../include ../../DrvGPU ../../modules pages
FILE_PATTERNS    = *.h *.hpp *.cpp *.c *.cl *.md
RECURSIVE        = YES
OUTPUT_DIRECTORY = .
HTML_OUTPUT      = html

USE_MATHJAX      = YES
MATHJAX_VERSION  = MathJax_3
IMAGE_PATH       = ../../Results/Plots/signal_generators \
                   ../../Results/Plots/filters \
                   ../../Results/Plots/heterodyne \
                   ../../Results/Plots/statistics \
                   ../../Results/Plots/strategies \
                   ../../Results/Plots/fft_maxima \
                   ../../Results/Plots/integration

# --- Подключаем ВСЕ .tag файлы модулей ---
TAGFILES = DrvGPU/drvgpu.tag=DrvGPU/html \
           modules/signal_generators/signal_generators.tag=modules/signal_generators/html \
           modules/fft_func/fft_func.tag=modules/fft_func/html \
           modules/filters/filters.tag=modules/filters/html \
           modules/statistics/statistics.tag=modules/statistics/html \
           modules/heterodyne/heterodyne.tag=modules/heterodyne/html \
           modules/vector_algebra/vector_algebra.tag=modules/vector_algebra/html \
           modules/capon/capon.tag=modules/capon/html \
           modules/range_angle/range_angle.tag=modules/range_angle/html \
           modules/fm_correlator/fm_correlator.tag=modules/fm_correlator/html \
           modules/strategies/strategies.tag=modules/strategies/html \
           modules/lch_farrow/lch_farrow.tag=modules/lch_farrow/html

OUTPUT_LANGUAGE  = Russian
EXTRACT_ALL      = YES
SOURCE_BROWSER   = YES
HAVE_DOT         = NO
GENERATE_LATEX   = NO
```

### Шаг 3: Скрипт сборки (порядок важен!)

Создай `Doc/Doxygen/build_docs.bat`:
```bat
@echo off
REM Сборка Doxygen: сначала модули (генерируют .tag), потом главный (подключает .tag)
set DOXYGEN="C:\Program Files\doxygen\bin\doxygen.exe"

echo === Шаг 1: DrvGPU (базовый, без зависимостей) ===
cd DrvGPU && %DOXYGEN% Doxyfile && cd ..

echo === Шаг 2: Модули (зависят от DrvGPU) ===
for %%m in (signal_generators fft_func filters statistics heterodyne vector_algebra capon range_angle fm_correlator strategies lch_farrow) do (
    echo --- %%m ---
    cd modules\%%m && %DOXYGEN% Doxyfile && cd ..\..
)

echo === Шаг 3: Главный (подключает ВСЕ .tag) ===
%DOXYGEN% Doxyfile

echo === Готово! Открываю... ===
start html\index.html
```

Создай `Doc/Doxygen/build_docs.sh`:
```bash
#!/bin/bash
set -e
DOXYGEN="doxygen"

echo "=== Шаг 1: DrvGPU ==="
(cd DrvGPU && $DOXYGEN Doxyfile)

echo "=== Шаг 2: Модули ==="
for m in signal_generators fft_func filters statistics heterodyne vector_algebra capon range_angle fm_correlator strategies lch_farrow; do
    echo "--- $m ---"
    (cd "modules/$m" && $DOXYGEN Doxyfile)
done

echo "=== Шаг 3: Главный ==="
$DOXYGEN Doxyfile
echo "=== Готово! ==="
```

### Шаг 4: Общие страницы (pages/)

a) **mainpage.md** — главная со ссылками на модули:
   - Ссылка @ref signal_generators → переходит в HTML signal_generators
   - Таблица модулей с кликабельными ссылками
   - Графики (@image html)

b) **architecture.md** — Ref03 6-слойная модель

c) **build_guide.md** — сборка CMake + ROCm/OpenCL

d) **tests_overview.md** — обзор тестов

### Шаг 5: Страницы каждого модуля

Для каждого модуля создай pages/:
1. **overview.md** — описание + @ref на классы DrvGPU (ссылки через TAGFILES!)
2. **formulas.md** — формулы MathJax (@f$ ... @f$)
3. **tests.md** — тесты + @image html из Results/Plots/{module}/

Прочитай существующую документацию:
- Doc/Modules/{module}/Full.md — возьми формулы и описание
- Doc/DrvGPU/*.md — для DrvGPU страниц

### Шаг 6: Обогати заголовки формулами

Добавь @brief + @f$ в публичные функции/классы всех модулей.
НЕ МЕНЯТЬ сигнатуры! Только комментарии.

### Результат
- Главная страница → клик → модуль (через TAGFILES!)
- Модуль → клик на DrvGPU класс → переходит в DrvGPU HTML
- Каждый модуль можно пересобрать отдельно (секунды)
- Формулы MathJax, графики, тесты — везде
- Один скрипт build_docs.bat собирает ВСЁ в правильном порядке

Это нужно для заказчика — сделай красиво и подробно!
```

---

## Запуск

### Полная сборка (все модули + главный):
```powershell
cd E:\C++\GPUWorkLib\Doc\Doxygen
.\build_docs.bat
```

### Пересборка одного модуля (быстро):
```powershell
cd E:\C++\GPUWorkLib\Doc\Doxygen\modules\signal_generators
& "C:\Program Files\doxygen\bin\doxygen.exe" Doxyfile
start html\index.html
```

### Только главный (если .tag файлы уже есть):
```powershell
cd E:\C++\GPUWorkLib\Doc\Doxygen
& "C:\Program Files\doxygen\bin\doxygen.exe" Doxyfile
start html\index.html
```

## Добавить doxygen в PATH (один раз)

```powershell
# Для текущей сессии:
$env:PATH += ";C:\Program Files\doxygen\bin"

# Навсегда (от администратора):
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files\doxygen\bin", "Machine")
```

## Как работают связи (TAGFILES)

```
1. DrvGPU генерирует drvgpu.tag (индекс: GPUProfiler, ConsoleOutput, IBackend...)
2. signal_generators подключает drvgpu.tag → если в коде упоминается GPUProfiler,
   Doxygen создаст ССЫЛКУ на DrvGPU/html/classGPUProfiler.html
3. Главный подключает ВСЕ .tag → на главной странице все классы кликабельны,
   ведут в HTML соответствующего модуля
```
