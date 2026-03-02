# План: Агент и команда для тестирования

**Источник**: MemoryBank/specs/create_agent_test.md  
**Дата**: 2026-03-02  
**Статус**: РЕАЛИЗОВАНО (2026-03-02)

---

## Ответы Алекса (зафиксировано)

1. **Агент**: B — Cursor Agent (skill в .cursor/)
2. **GPU**: Оба — build-time (пресет) и run-time (rocminfo/nvidia-smi)
3. **Файл тестов**: Да — один модуль на строку
4. **DrvGPU**: Первый в порядке (3.0)

---

## Что непонятно было — вопросы (resolved)

### 1. Что такое «агент» и «команда»?

- **Вариант A**: Shell-скрипт `run_tests.sh` + Python-раннер `run_agent_tests.py` — запускаем вручную из терминала
- **Вариант B**: Cursor Agent (skill/rule в .cursor/) — вызывается через Cursor UI / Composer
- **Вариант C**: CMake target + отдельная утилита — `cmake --build . --target run_all_tests`

Какой вариант имеется в виду?

### 2. Когда определять GPU — при сборке или при запуске?

- **Build-time** (CMake): уже есть `TYPE_GPU` в CMakePresets (NVIDIA-RTX3060, AMD-Radeon9070). Можно отключить clFFT для AMD-пресета
- **Run-time**: `rocminfo` / `nvidia-smi` — скрипт сам решает, что тестировать

Сейчас в проекте разные пресеты под разное железо. Логично ли определять GPU на этапе конфигурации (пресет) и не менять это во время run?

### 3. Тест «через файл» (п. 2.3)

Какой формат? Пример:
```
# tests_to_run.txt
fft_processor
statistics
vector_algebra
```
Или там будут конкретные тесты внутри модулей (например, `test_fft_processor_rocm`)?

### 4. DrvGPU в списке?

В списке порядка (п. 3) нет DrvGPU. В main.cpp есть `drvgpu_all_test::run()`. Включать DrvGPU в «all» или он отдельно?

---

## Предлагаемая архитектура (предварительно)

### Компоненты

| Компонент | Назначение |
|-----------|------------|
| `scripts/run_agent_tests.sh` | Точка входа: парсит аргументы, вызывает C++ и Python |
| `scripts/run_agent_tests.py` | Оркестратор: порядок модулей, фильтры по GPU, запуск pytest |
| `config/tests_order.txt` | Порядок модулей для режима `all` |
| CMake: `CLFFT_DISABLED_FOR_AMD` | При AMD-пресете не линковать clFFT (опционально) |

### Команда

```bash
./scripts/run_agent_tests.sh all                    # все модули по порядку
./scripts/run_agent_tests.sh fft_processor          # один модуль
./scripts/run_agent_tests.sh --file my_tests.txt    # из файла
```

### Логика GPU

- Читать `TYPE_GPU` из CMakeCache.txt (если есть) или вызывать `rocminfo` / `nvidia-smi`
- AMD → пропускать C++-тесты с clFFT (fft_processor OpenCL, fft_maxima OpenCL, …), запускать ROCm
- NVIDIA → пропускать ROCm-тесты, запускать OpenCL

### Сохранение результатов

- C++: `configGPU.json` уже включает is_prof → Results/Profiler/…
- Python: pytest с `--basetemp` или фикс. каталог для графиков (Results/Plots/)

---

## Дальнейшие шаги после ответов

1. Зафиксировать формат файла тестов (п. 2.3)
2. Уточнить наличие DrvGPU в «all»
3. Выбрать: build-time vs run-time определение GPU
4. Реализовать `run_agent_tests.sh` / `run_agent_tests.py`
