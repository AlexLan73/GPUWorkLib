# Задача: DelayedFormSignalGenerator (Farrow 48×5)

> Класс: **DelayedFormSignalGenerator**. Модуль: **signal_generators**. Задержка в **микросекундах** (float). Матрица коэффициентов **48×5** (float).  
> Реализацию выполняет другой исполнитель; главный (ревьюер) затем проверяет и даёт заключение.

---

## Ссылки для прочтения и понимания

### План и спецификация в проекте

- **[Plan_FractionalDelay_Farrow.md](../DiscussionPlan/Plan_FractionalDelay_Farrow.md)** — основной план: цель, параметры (все float), этапы, решённые вопросы, подробное описание алгоритма формирования (раздел 7).
- **[lagrange_matrix_48x5.json](../DiscussionPlan/lagrange_matrix_48x5.json)** — матрица коэффициентов 48×5 (float); использовать при реализации и тестах.

### Референсная реализация (LCH-Farrow01)

- **`/home/alex/C++/LCH-Farrow01/include/GPU/fractional_delay_processor.hpp`** — API процессора дробной задержки: константы 48×5, DelayParams (целая часть + lagrange_row), конфиг, загрузка матрицы из JSON, Process(gpu_buffer, delays).
- **`/home/alex/C++/LCH-Farrow01/src/GPU/fractional_delay_processor.cpp`** — реализация: загрузка матрицы из JSON, OpenCL kernel (5-точечная интерполяция по строке матрицы), IN-PLACE через temp-буфер, буферы на GPU.
- **`/home/alex/C++/LCH-Farrow01/lagrange_matrix.json`** — пример формата JSON для матрицы 48×5 (в GPUWorkLib своя копия: `lagrange_matrix_48x5.json`).

### Материалы по алгоритму в GPUWorkLib

- **`MemoryBank/DiscussionPlan/Analiz_DROB_pause/Farrow_GPU_OpenCL_Algorithm.md`** — описание алгоритма Farrow для GPU/OpenCL.
- **`MemoryBank/DiscussionPlan/Analiz_DROB_pause/Анализ дробной задержки/`** — скрипты и отчёт `delay_methods_report.md`.

### Внешние источники (теория)

- **CCRMA JOS — Lagrange / Farrow:**  
  - [Lagrange Interpolation](https://ccrma.stanford.edu/~jos/Interpolation/Lagrange_Interpolation.html)  
  - [Farrow Structure for Variable Delay FIR Filters](https://ccrma.stanford.edu/~jos/Interpolation/Farrow_Structure_Variable_Delay.html)  
  - [Lagrange Interpolation Coefficients Orders 1, 2, and 3](https://ccrma.stanford.edu/~jos/Interpolation/Lagrange_Interpolation_Coefficients_Orders.html)
- **Методология анализа проекта:** план «Analysis methodology and fractional-delay» (Context7, GitHub, sequential-thinking, статьи) — см. `.cursor/plans/` или обсуждения в сессиях.

### Связанные спецификации проекта

- **FormSignalGenerator:** `MemoryBank/specs/Form_signals.md` — формула сигнала, мультиканальность, выход InputData\<cl_mem\>.
- **Разногласия и чеклист FormSignalGenerator:** `MemoryBank/tasks/FormSignalGenerator_Разногласия.md`, `MemoryBank/tasks/CHECKLIST_FormSignalGenerator.md`.

---

## Обязательные требования по задаче

- Задержка на луч: массив **delay_us** (float), микросекунды.
- Все параметры и коэффициенты — **float**.
- Матрица 48×5 — из `lagrange_matrix_48x5.json` (или эквивалент в коде).
- Выход: сдвинутый сигнал + шум (без TAU_STEP/TAU_RANDOM в первой версии).
- **Интеграция:** обязательна демонстрация с графиками; данные из Python передаются в генератор как переменные (параметры, массивы).
- Реализацию делает исполнитель; главный проверяет и даёт заключение.

---

## Чеклист (заполнять по ходу)

- [ ] Спека и API зафиксированы (InputData\<cl_mem\> или аналог FormSignalGenerator).
- [ ] Параметры: delay_us (float), sample_rate (float), использование матрицы 48×5.
- [ ] OpenCL kernel: целая задержка D + интерполяция по строке матрицы 48×5.
- [ ] Тесты: эталон (NumPy/C++), сравнение GPU vs CPU.
- [ ] Демо с графиками, данные из Python как переменные.
- [ ] Ревью и заключение главного.
