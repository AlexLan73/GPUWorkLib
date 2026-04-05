# Шпаргалка: Работа с Doxygen через агента

> **Агент**: `doxygen-maintainer`
> **Файл**: `.claude/agents/doxygen-maintainer.md`
> **Передаётся через GitHub**: да ✅

---

## Как вызвать агента

В Claude Code просто напиши задачу — агент `doxygen-maintainer` подхватит автоматически:

### Добавить новый модуль
```
Добавь модуль pulse_compression в Doxygen
```
Агент создаст: `Doc/Doxygen/modules/pulse_compression/Doxyfile` + `pages/` (overview, formulas, tests), обновит главный Doxyfile, mainpage, groups, build scripts.

### Обновить существующий модуль
```
Обнови Doxygen pages для модуля capon — добавились новые тесты
```
Агент прочитает код `modules/capon/tests/`, обновит `pages/tests.md`.

### Копировать графики
```
Скопируй графики и пересобери документацию
```
Агент запустит `copy_images.sh` (ищет в `Results/Plots/` И `build/**/Results/Plots/`), потом `build_docs.sh`.

### Проверить ссылки
```
Проверь битые ссылки в Doxygen
```
Агент найдёт все `@ref` без соответствующих `@page`, покажет что исправить.

---

## Структура Doxygen (модульная с TAGFILES)

```
Doc/Doxygen/
├── Doxyfile                  # Главный (TAGFILES = все модули)
├── build_docs.bat            # Windows: clean → DrvGPU → модули → главный
├── build_docs.sh             # Linux: то же
├── copy_images.sh            # Копирование графиков
│
├── DrvGPU/                   # Центральный компонент
│   ├── Doxyfile              # GENERATE_TAGFILE = drvgpu.tag
│   └── pages/                # 4 страницы
│
├── modules/{module}/         # 11 модулей
│   ├── Doxyfile              # GENERATE_TAGFILE + TAGFILES = DrvGPU
│   └── pages/                # overview.md, formulas.md, tests.md
│
└── pages/                    # Общие страницы
    ├── mainpage.md           # Главная: таблица всех модулей
    ├── architecture.md       # Ref03: 6-слойная модель
    ├── build_guide.md        # Сборка
    ├── groups.md             # Группировка модулей
    ├── modules_overview.md   # Обзор
    └── tests_overview.md     # Сводка тестов
```

## Графики — два источника (IMAGE_PATH)

```
1. Doc/Modules/{module}/images/    ← стабильные, в git (приоритет)
2. Results/Plots/{plots_dir}/      ← автогенерация Python тестов (fallback)
```

| Модуль | Results/Plots/ | Doc/Modules/.../images/ |
|--------|---------------|------------------------|
| fft_func | fft_maxima/ | fft_func/images/ |
| filters | filters/ | filters/images/ |
| heterodyne | heterodyne/ | heterodyne/images/ |
| signal_generators | signal_generators/ | signal_generators/images/ |
| statistics | statistics/ | statistics/images/ |
| strategies | strategies/ | strategies/images/ |
| lch_farrow | lch_farrow/ | lch_farrow/images/ |
| capon | — (нет графиков) | capon/images/ |
| vector_algebra | — (нет графиков) | vector_algebra/images/ |
| range_angle | — (нет графиков) | range_angle/images/ |
| fm_correlator | — (нет графиков) | fm_correlator/images/ |

## Сборка

```bash
# Linux
cd Doc/Doxygen
./copy_images.sh    # сначала скопировать графики
./build_docs.sh     # потом собрать

# Windows (copy_images.bat нет — графики копировать вручную или через WSL)
cd Doc\Doxygen
.\build_docs.bat
```

> **Windows**: `build_docs.bat` НЕ копирует графики автоматически.
> Для графиков на Windows — запустить `copy_images.sh` через WSL/Git Bash,
> или скопировать `Results/Plots/*` → `Doc/Modules/{module}/images/` вручную.

Порядок сборки: **clean → DrvGPU (.tag) → 11 модулей (.tag) → главный (TAGFILES)**

## Схема именования @page

| Тип | Формат ID | Пример |
|-----|-----------|--------|
| DrvGPU | `drvgpu_{name}` | `drvgpu_main`, `drvgpu_profiler` |
| Модуль overview | `{module}_overview` | `filters_overview` |
| Модуль формулы | `{module}_formulas` | `filters_formulas` |
| Модуль тесты | `{module}_tests` | `filters_tests` |
| Общие | `{name}_page` | `architecture_page`, `tests_overview_page` |

**НЕ использовать**: `mod_{module}_page` — устаревший формат!

## Правила

- Нет абсолютных путей в Doxyfile
- DOT_PATH пустой (Graphviz через PATH)
- Язык страниц — русский
- `@image html` — только имя файла (Doxygen ищет по IMAGE_PATH)
- Graphviz: `build_docs.bat` добавляет в PATH автоматически

## Дополнительно

- `Doc/Doxygen/CREATE_DOXYGEN_PROMPT.md` — промпт для создания Doxygen-конфигурации нового модуля с нуля (используется агентом как шаблон)

---

*Создано: 2026-03-29 | Обновлено: 2026-04-05 (code review)*
