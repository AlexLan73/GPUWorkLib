# Python_test — тесты и примеры GPUWorkLib

Краткое описание тестов и **где лежат сформированные картинки (графики)**.

---

## Запуск в PyCharm

1. **Сборка с Python:** из корня репо:
   ```bash
   cmake -B build -DBUILD_PYTHON=ON && cmake --build build
   ```
2. **Run Configuration** для скрипта (например `test_delayed_form_signal.py`):
   - **Working directory:** корень репозитория GPUWorkLib (где лежат `Python_test/`, `build/`).
   - **Python interpreter:** тот же, для которого собирали (см. `Python3_EXECUTABLE` в CMake).
   - **Path / PYTHONPATH:** добавьте `build/python` (или `build/python/Release` / `build/python/Debug`), чтобы подхватывался модуль `gpuworklib` из сборки.
     - В PyCharm: Run → Edit Configurations → выберите скрипт → вкладка "Path" или "Environment" → добавьте в PYTHONPATH путь `$ProjectFileDir$/build/python` (или через "Add content roots" к папке `build/python`).
3. Если видите `AttributeError: ... has no attribute 'DelayedFormSignalGenerator'` — в конфиге подхватывается не та сборка (старый модуль без этого класса). Убедитесь, что в PYTHONPATH только один путь к `gpuworklib` — из свежей сборки `build/python`.

---

## Включение/выключение графиков (plot)

- **По умолчанию графики строятся** (если тесты доходят до блока с `plot1_...`).
- **Выключить:** аргумент командной строки `--no-plot` или переменная окружения `GPUWORKLIB_PLOT=0`.
- **В PyCharm включить явно:** Run → Edit Configurations → Environment variables → добавьте `GPUWORKLIB_PLOT=1` (при необходимости; по умолчанию и так включено).

---

## Где сформированные картинки

| Тест / скрипт | Папка с графиками |
|---------------|--------------------|
| **DelayedFormSignalGenerator** (Farrow 48×5) | `Results/Plots/DelayedFormSignal/` |
| **FormSignalGenerator** | `Results/Plots/FormSignal/` |
| Общие тесты (test_gpuworklib и др.) | `Results/Plots/` (корень) |

Путь задаётся относительно корня репозитория: `GPUWorkLib/Results/Plots/...`

**Как сгенерировать графики DelayedFormSignal:** из корня репо (нужна сборка с Python-биндингами и DelayedFormSignalGenerator):
```bash
python3 Python_test/test_delayed_form_signal.py
```
Флаг `--no-plot` отключает построение графиков. По умолчанию создаются 4 PNG в `Results/Plots/DelayedFormSignal/`:
- `plot1_integer_delay.png` — целая задержка, GPU vs NumPy
- `plot2_fractional_delay.png` — дробная задержка, overlay
- `plot3_multichannel_waterfall.png` — мультиканал, waterfall
- `plot4_delay_sweep.png` — ошибка vs задержка (sweep)

**FormSignal:** запуск `Python_test/test_form_signal.py` (по умолчанию с графиками) создаёт PNG в `Results/Plots/FormSignal/`.

---

## Описание тестов

### test_delayed_form_signal.py — DelayedFormSignalGenerator (Farrow 48×5)

| № | Функция | Что тестирует |
|---|---------|----------------|
| 1 | `test_integer_delay()` | Целая задержка (5 сэмплов): GPU vs NumPy reference; max_error < 1e-2 |
| 2 | `test_fractional_delay()` | Дробная задержка (2.7 сэмпла): GPU vs NumPy (Lagrange 48×5); max_error < 1e-2 |
| 3 | `test_multichannel_delay()` | 8 антенн с задержками 0, 1.5, …, 10.5 мкс; каждый канал сверяется с эталоном. Допуск ослаблен (max_err < 1.0); при ошибке ~0.5 возможна известная разница — см. TODO в коде. |
| 4 | `test_zero_delay()` | Задержка 0 → результат совпадает с FormSignalGenerator (без шума); max_error < 1e-4 |
| 5 | `test_delay_with_noise()` | Задержка + шум: проверка мощности шума (noise_power vs expected), ratio в диапазоне 0.5–2.0 |

Графики (если не `--no-plot`): см. таблицу выше → `Results/Plots/DelayedFormSignal/`.

---

### test_form_signal.py — FormSignalGenerator

Тесты генератора форм-сигнала (getX, мультилуч, TAU_STEP/TAU_RANDOM и т.д.). Графики → `Results/Plots/FormSignal/`.

---

### test_gpuworklib.py

Сводный набор тестов модулей (FormSignal, Script, FFT, FindAllMaxima и др.). Часть графиков в `Results/Plots/` (test1_single_tone.png, test2_three_tones.png, …).

---

### test_find_all_maxima_maxvalue.py, test_spectrum_find_all_maxima.py

Тесты поиска максимумов в спектре (FindAllMaxima, batch).

---

### example_form_signal.py

Пример использования FormSignalGenerator (без обязательных assert’ов).

---

*Обновлено: 2026-02-11*
