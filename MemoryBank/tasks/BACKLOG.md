# BACKLOG — Очередь задач

> **Обновлено**: 2026-02-18

---

## 🆕 Новые задачи (добавлено 2026-02-18)

### GPU FIR Filter — реализация (Высокий приоритет)

**Цель**: Полный GPU FIR фильтр с Python биндингами и AI-driven пайплайном

| ID | Задача | Описание | Приоритет |
|----|--------|----------|-----------|
| FIR-001 | `fir_filter.cl` — OpenCL ядро | Direct form: `y[n] = Σ h[k] * x[n-k]`. Один поток = один выходной отсчёт. `__constant` память для коэффициентов. Zero padding на границах. Input: float2 complex, float coeffs, num_taps, num_samples | 🔴 HIGH |
| FIR-002 | `PyFIRFilter` в `python/gpu_worklib_bindings.cpp` | Методы: `set_coefficients(h: np.ndarray)`, `process(signal: np.ndarray) -> np.ndarray`, `get_frequency_response(worN=4096) -> dict`. Регистрация в PYBIND11_MODULE | 🔴 HIGH |
| FIR-003 | Тест `test_ai_fir_demo.py` на реальном GPU | Требования: `pip install scipy matplotlib`. Сначала запустить MODE="none" (без AI). Файл готов: `Python_test/test_ai_fir_demo.py` | 🔴 HIGH |

### AI-Driven DSP Pipeline — расширение (Средний приоритет)

| ID | Задача | Описание | Приоритет |
|----|--------|----------|-----------|
| AI-001 | Groq API — подключить AI режим | Бесплатно: console.groq.com. `pip install groq`. `export GROQ_API_KEY="gsk_..."`. Изменить MODE="groq" в `test_ai_fir_demo.py` | 🟡 MEDIUM |
| AI-002 | Ollama — оффлайн AI режим | Установить: ollama.ai/download. `ollama pull qwen2.5-coder:7b`. Изменить MODE="ollama" в `test_ai_fir_demo.py` | 🟡 MEDIUM |

### AI-Driven DSP Pipeline — идеи (Низкий приоритет)

| ID | Задача | Описание | Приоритет |
|----|--------|----------|-----------|
| AI-003 | Multi-step AI pipeline | LangChain или AutoGen для цепочки агентов: анализ сигнала → подбор фильтра → валидация | 🟢 LOW |
| AI-004 | IIR фильтры в пайплайн | `scipy.signal.butter`/`cheby` + GPU IIR реализация | 🟢 LOW |
| AI-005 | FIR через FFT overlap-add | Для длинных сигналов: FFT свёртка вместо прямой формы | 🟢 LOW |

---

## Перспективные задачи

### FormSignalGenerator (Высокий приоритет)

**Цель**: Мультиканальный генератор комплексных сигналов (формула getX) с задержкой, амплитудой, шумом.

| Задача | Описание |
|--------|----------|
| FORM-001 | FormParams + FormSignalGenerator (OpenCL kernel) |
| FORM-002 | FormScriptGenerator + DSL + on-disk кэш по имени |
| FORM-003 | SignalService + Factory |
| FORM-004 | Python bindings + example с графиками |
| FORM-005 | ROCm заглушки |

**Checklist**: [CHECKLIST_FormSignalGenerator.md](CHECKLIST_FormSignalGenerator.md)  
**Спецификация**: [specs/Form_signals.md](../specs/Form_signals.md)

---

### ROCm Backend (Средний приоритет)

**Цель**: Добавить поддержку AMD GPU через ROCm/HIP

| Задача | Описание |
|--------|----------|
| ROCM-001 | Создать `SpectrumProcessorROCm` (hipFFT) |
| ROCM-002 | Интегрировать в SpectrumMaximaFinder через Strategy Pattern |
| ROCM-003 | Тесты на AMD GPU |

**Зависимости**: Нужен доступ к AMD GPU




### Code Style (Низкий приоритет)

**Цель**: Google C++ Style + 2-пробельная табуляция

**Сфера**: `DrvGPU/`

---

## Отложенные (после основного функционала)

| Задача | Описание |
|--------|----------|
| Filters модуль | FIR, IIR фильтры на GPU |
| Statistics модуль | mean, std, variance на GPU |
| Heterodyne модуль | NCO, MixDown/MixUp |

---

*Последнее обновление: 2026-02-18*
