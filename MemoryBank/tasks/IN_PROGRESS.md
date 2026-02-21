# IN PROGRESS — Текущие задачи

> **Обновлено**: 2026-02-21

---

## 🔄 TASK-009: Heterodyne LFM Dechirp Module

**Статус**: 📝 ПЛАН НАПИСАН — ждёт ревью и правок от Alex

**Файл плана**: `tasks/PLAN_Heterodyne_LFM_Dechirp.md` (v2.0)

**Что запланировано**:
- Новый класс `LfmConjugateGenerator` в `modules/signal_generators/`
- Новый модуль `modules/heterodyne/` (OpenCL backend + ROCm заглушка)
- Два OpenCL ядра: `dechirp_multiply.cl` + `dechirp_correct.cl`
- C++ тесты (3 файла, 7 тестов): basic / pipeline / external_ctx
- Python тесты (3 файла): step-by-step / pytest / GPU vs CPU сравнение
- Python биндинги (pybind11)
- Параметры: fs=12e6, B=1e6, N=4000, 5 антенн

**Блокер**: Alex читает план, вносит правки → после одобрения начинаем реализацию

---

## Следующие кандидаты (BACKLOG)

См. [BACKLOG.md](BACKLOG.md):
- Filters Stage 2: Text->kernel pipeline (FormScriptGenerator-like)
- ~~Filters Stage 3: Groq AI micro-agent~~ -> **DONE** (TASK-008)
- Performance benchmarks: замеры GPU vs CPU на разных конфигурациях
- Streaming: поддержка непрерывного потока (state persistence)
- Overlap-Save/Add: для длинных FIR через FFT
- ROCm Backend (полная реализация)
- Оконные функции
- Code Style рефакторинг

---

*Последнее обновление: 2026-02-18*
