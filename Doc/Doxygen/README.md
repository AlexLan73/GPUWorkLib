# Генерация Doxygen документации GPUWorkLib

## Требования

- **Doxygen 1.9+**
  ```bash
  sudo apt install doxygen
  # или на Windows: https://www.doxygen.nl/download.html
  ```
- **Graphviz** (для диаграмм классов и зависимостей)
  ```bash
  sudo apt install graphviz
  # или на Windows: https://graphviz.org/download/
  ```

## Модульная структура

Документация организована **по модулям** с перекрёстными ссылками через **TAGFILES**:

```
Doc/Doxygen/
├── Doxyfile                  # Главный (подключает ВСЕ .tag)
├── build_docs.bat            # Скрипт сборки (Windows)
├── build_docs.sh             # Скрипт сборки (Linux)
├── html/                     # HTML всего проекта
│
├── DrvGPU/                   # --- DrvGPU (центральный) ---
│   ├── Doxyfile              # GENERATE_TAGFILE = drvgpu.tag
│   ├── pages/                # mainpage, architecture, profiler, console
│   └── html/
│
├── modules/                  # --- 11 модулей ---
│   ├── signal_generators/    # CW, LFM, Noise, FormSignal
│   ├── fft_func/             # hipFFT + spectrum maxima
│   ├── filters/              # FIR, IIR, MA, Kalman, KAMA
│   ├── statistics/           # Welford, median
│   ├── heterodyne/           # LFM dechirp
│   ├── vector_algebra/       # Cholesky inversion
│   ├── capon/                # MVDR beamformer
│   ├── range_angle/          # 3D range-angle
│   ├── fm_correlator/        # M-sequence correlation
│   ├── strategies/           # Digital beamforming
│   └── lch_farrow/           # Farrow interpolation
│       ├── Doxyfile          # GENERATE_TAGFILE + TAGFILES
│       ├── pages/            # overview, formulas, tests
│       └── html/
│
└── pages/                    # --- Общие страницы ---
    ├── mainpage.md           # Главная: таблица модулей
    ├── architecture.md       # Ref03: 6-слойная модель
    ├── build_guide.md        # Сборка: CMake, ROCm, OpenCL
    ├── modules_overview.md   # Краткий обзор модулей
    ├── tests_overview.md     # Сводка тестов
    └── groups.md             # Группировка модулей
```

## Запуск

### Полная сборка (все модули + главный):
```powershell
cd Doc/Doxygen
.\build_docs.bat          # Windows
./build_docs.sh           # Linux
```

### Пересборка одного модуля (быстро):
```powershell
cd Doc/Doxygen/modules/signal_generators
doxygen Doxyfile
start html/index.html
```

### Только главный (если .tag файлы уже есть):
```powershell
cd Doc/Doxygen
doxygen Doxyfile
start html/index.html
```

## Как работают связи (TAGFILES)

```
1. DrvGPU генерирует drvgpu.tag (индекс символов)
2. Модули подключают drvgpu.tag → ссылки на классы DrvGPU
3. Главный подключает ВСЕ .tag → перекрёстные ссылки везде
```

Порядок сборки: **DrvGPU → модули → главный** (скрипт делает автоматически).

## Настройки

- **ENABLE_ROCM=1** — показывает ROCm/HIP-специфичный код
- **MathJax 3** — формулы прямо в браузере
- **Dark theme** — HTML_COLORSTYLE=DARK
- **SVG диаграммы** — интерактивные (Graphviz)
- **Поиск** — встроенный JavaScript

## Типовые проблемы

**`dot: command not found`**
→ Установите Graphviz или `HAVE_DOT = NO` в Doxyfile.

**Предупреждения о незадокументированных**
→ Нормально — `WARN_IF_UNDOCUMENTED = NO`.
