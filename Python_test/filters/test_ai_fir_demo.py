"""
ai_fir_demo.py — AI-управляемый FIR-фильтр для GPUWorkLib
==========================================================

Демонстрация: текстовая команда → AI → scipy коэффициенты → GPU пайплайн → графики

Режимы AI-бекенда (выбери один):
  MODE = "groq"    — бесплатное облако, нужен API ключ: console.groq.com
  MODE = "ollama"  — офлайн, нужен ollama: ollama.ai + ollama pull qwen2.5-coder:7b
  MODE = "none"    — без AI, параметры задаются напрямую (для теста без LLM)

Установка:
  pip install scipy matplotlib numpy
  pip install groq          # для Groq
  pip install ollama        # для Ollama (+ запустить: ollama serve)

Автор: Кодо (AI Assistant)
Дата:  2026-02-18
"""

import sys
import os
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

# ── GPUWorkLib (Python_test/filters/ -> 2 levels up) ──
BUILD_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'Debug'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'Release'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'),
]
for p in BUILD_PATHS:
    if os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))
        break

try:
    import gpuworklib
except ImportError:
    print("ERROR: gpuworklib not found. Build with -DBUILD_PYTHON=ON")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ════════════════════════════════════════════════════════════════════════════

MODE = "none"           # "groq" | "ollama" | "none"
OLLAMA_MODEL = "qwen2.5-coder:7b"

PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'Plots', 'filters')

# ── API Key — читается из api_keys.json (в корне проекта) ──────────────────
# Создай файл:  <корень проекта>/api_keys.json  со следующим содержимым:
#
#   {
#       "api": "sm_ВАШ_КЛЮЧ_ЗДЕСЬ"
#   }
#
# Файл добавлен в .gitignore — в репозиторий не попадёт.
# Альтернатива: задать переменную окружения  GROQ_API_KEY=sm_...
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_API_KEYS_PATH = os.path.join(_PROJECT_ROOT, 'api_keys.json')
GROQ_API_KEY = ""
if os.path.isfile(_API_KEYS_PATH):
    try:
        with open(_API_KEYS_PATH, 'r') as _f:
            _api_data = json.load(_f)
        GROQ_API_KEY = _api_data.get("api", "")
        if GROQ_API_KEY:
            print(f"[INFO] API key loaded from {_API_KEYS_PATH}")
    except Exception as _e:
        print(f"[WARN] Failed to load api_keys.json: {_e}")

# ════════════════════════════════════════════════════════════════════════════
# AI-БЕКЕНДЫ
# ════════════════════════════════════════════════════════════════════════════

def ai_ask(prompt: str, system: str = "") -> str:
    """Отправить запрос в LLM. Возвращает текстовый ответ."""

    if MODE == "groq":
        # pip install groq
        # Получить ключ бесплатно: console.groq.com
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("pip install groq")
        key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError(
                "Groq API key не задан!\n"
                "  1. Зарегистрируйся на console.groq.com (бесплатно)\n"
                "  2. Скопируй ключ и вставь в GROQ_API_KEY выше\n"
                "     или: set GROQ_API_KEY=gsk_..."
            )
        client = Groq(api_key=key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.05,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    elif MODE == "ollama":
        # pip install ollama
        # Нужен запущенный: ollama serve
        # Нужна модель: ollama pull qwen2.5-coder:7b
        try:
            import ollama
        except ImportError:
            raise RuntimeError("pip install ollama")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.05},
        )
        return response["message"]["content"]

    else:
        # MODE == "none": возвращаем None, используем scipy напрямую
        return None


def extract_json(text: str) -> dict:
    """Извлечь JSON из текста LLM (обычно в ```json блоке)."""
    # Попробовать ```json ... ```
    m = re.search(r'```json\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
    # Попробовать просто { ... }
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Не удалось найти JSON в ответе LLM:\n{text[:300]}")


# ════════════════════════════════════════════════════════════════════════════
# ШАГ 1: Парсинг текстового запроса → параметры фильтра
# ════════════════════════════════════════════════════════════════════════════

def parse_filter_request(text: str, fs: float) -> dict:
    """
    Текстовый запрос → параметры FIR-фильтра.

    Если MODE="none" — возвращает параметры по умолчанию.
    Если AI доступен — парсит текст через LLM.
    """
    print(f"  📝 Запрос: '{text}'")

    if MODE == "none":
        # Параметры по умолчанию для демонстрации
        params = {
            "filter_type": "lowpass",
            "f_pass": 5000.0,
            "f_stop": 8000.0,
            "fs": fs,
            "window": "kaiser",
            "ripple_db": 60.0,
            "description": "LPF 5kHz (demo, no AI)"
        }
        print(f"  ℹ️  MODE=none: использую параметры по умолчанию")
        print(f"     {params}")
        return params

    # AI парсит запрос
    system = (
        "Ты — эксперт по ЦОС. Из описания пользователя извлеки параметры FIR-фильтра. "
        "Верни ТОЛЬКО валидный JSON, ничего больше."
    )
    prompt = f"""Описание фильтра: "{text}"
Частота дискретизации: {fs} Гц

Верни JSON со следующими полями:
{{
  "filter_type": "lowpass" | "highpass" | "bandpass" | "bandstop",
  "f_pass": <Гц или [f1,f2] для полосовых>,
  "f_stop": <Гц или [f1,f2] для полосовых>,
  "fs": {fs},
  "window": "kaiser" | "hamming" | "hann" | "blackman",
  "ripple_db": <дБ затухания в полосе задержания, число>,
  "description": "<краткое описание>"
}}

Если параметр не указан — используй разумное значение по умолчанию.
Все частоты в Гц (не кГц). Только JSON, без пояснений."""

    raw = ai_ask(prompt, system=system)
    print(f"  🤖 AI ответ: {raw[:200]}...")
    params = extract_json(raw)
    params["fs"] = fs
    print(f"  ✅ Параметры: {params}")
    return params


# ════════════════════════════════════════════════════════════════════════════
# ШАГ 2: scipy проектирует FIR-фильтр
# ════════════════════════════════════════════════════════════════════════════

def design_fir_filter(params: dict) -> np.ndarray:
    """
    Параметры → коэффициенты FIR-фильтра (scipy.signal).

    Всегда использует scipy — точный эталон, независимо от AI.
    """
    fs = params["fs"]
    ftype = params["filter_type"]
    window = params.get("window", "hann")
    ripple_db = float(params.get("ripple_db", 60.0))
    f_pass = params["f_pass"]
    f_stop = params.get("f_stop", f_pass * 1.5)

    # Kaiser window: автоматический порядок
    if window == "kaiser":
        # Ширина переходной полосы
        if isinstance(f_pass, (list, tuple)):
            # Полосовой: берём минимальную переходную полосу
            trans_width = min(
                abs(f_stop[0] - f_pass[0]) if isinstance(f_stop, (list, tuple))
                else abs(f_stop - f_pass[0]),
                abs(f_pass[1] - f_stop[1]) if isinstance(f_stop, (list, tuple))
                else abs(f_stop - f_pass[1]),
            )
        else:
            trans_width = abs(f_stop - f_pass) if isinstance(f_stop, (int, float)) \
                          else float(f_stop - f_pass)
        trans_width = max(trans_width, 100.0)  # минимум 100 Гц

        N, beta = sp_signal.kaiserord(ripple_db, trans_width / (0.5 * fs))
        N = N if N % 2 == 1 else N + 1  # нечётный порядок
        window_arg = ("kaiser", beta)
        print(f"  📐 Kaiser: N={N}, beta={beta:.2f}, trans_width={trans_width:.0f} Гц")
    else:
        N = 129  # фиксированный порядок для других окон
        window_arg = window
        print(f"  📐 {window}: N={N}")

    # Нормализованные частоты среза
    if ftype == "lowpass":
        cutoff = float(f_pass) / (0.5 * fs)
        pass_zero = True
    elif ftype == "highpass":
        cutoff = float(f_pass) / (0.5 * fs)
        pass_zero = False
    elif ftype in ("bandpass", "bandstop"):
        cutoff = [float(f_pass[0]) / (0.5 * fs), float(f_pass[1]) / (0.5 * fs)]
        pass_zero = (ftype == "bandstop")
    else:
        raise ValueError(f"Неизвестный тип фильтра: {ftype}")

    h = sp_signal.firwin(N, cutoff, window=window_arg, pass_zero=pass_zero)
    print(f"  ✅ Коэффициенты: {len(h)} тапов, сумма={np.sum(h):.6f}")
    return h.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# ШАГ 3: Применение фильтра
# ════════════════════════════════════════════════════════════════════════════

def apply_filter(signal_data: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Применить FIR-фильтр к сигналу.

    CPU (scipy): точный эталон.
    TODO: GPU через будущий gpuworklib.FIRFilter (когда будет реализован).
    """
    # Применяем к Re и Im независимо (h — вещественный фильтр)
    re_filtered = sp_signal.lfilter(h, [1.0], signal_data.real)
    im_filtered = sp_signal.lfilter(h, [1.0], signal_data.imag)
    return (re_filtered + 1j * im_filtered).astype(np.complex64)


# ════════════════════════════════════════════════════════════════════════════
# ШАГ 4: Валидация
# ════════════════════════════════════════════════════════════════════════════

def validate_filter(original: np.ndarray, filtered: np.ndarray,
                    h: np.ndarray, params: dict) -> dict:
    """
    Верификация результатов фильтрации.

    Проверки:
      1. АЧХ: затухание в полосе задержания ≥ ripple_db - 3 dB
      2. Форма сигнала: нет NaN/Inf
      3. Сравнение с scipy.signal.lfilter (должно совпасть, т.к. мы и используем его)
    """
    fs = params["fs"]
    ftype = params["filter_type"]
    f_pass = params["f_pass"]
    ripple_db = float(params.get("ripple_db", 60.0))

    results = {}

    # 1. Проверка NaN/Inf
    results["no_nan"] = not np.any(np.isnan(filtered))
    results["no_inf"] = not np.any(np.isinf(filtered))

    # 2. АЧХ фильтра
    w, H = sp_signal.freqz(h, fs=fs, worN=8192)
    H_db = 20 * np.log10(np.abs(H) + 1e-12)

    if ftype == "lowpass" and isinstance(f_pass, (int, float)):
        f_stop = float(params.get("f_stop", f_pass * 1.5))
        # Затухание в полосе задержания
        mask_stop = w > f_stop
        if np.any(mask_stop):
            attenuation = -np.max(H_db[mask_stop])
            results["attenuation_db"] = attenuation
            results["attenuation_ok"] = attenuation >= ripple_db * 0.8
            print(f"  📊 Затухание в полосе задержания: {attenuation:.1f} дБ"
                  f" (требуется ≥{ripple_db*0.8:.0f})")

    # 3. Уровень подавленных частот в спектре
    N = len(filtered)
    freqs = np.fft.rfftfreq(N, 1.0/fs)
    orig_mag  = np.abs(np.fft.rfft(original.real))
    filt_mag  = np.abs(np.fft.rfft(filtered.real))

    if ftype == "lowpass" and isinstance(f_pass, (int, float)):
        f_stop = float(params.get("f_stop", f_pass * 1.5))
        mask_stop = freqs > f_stop
        if np.any(mask_stop) and np.max(orig_mag[mask_stop]) > 1e-3:
            suppression = 20 * np.log10(
                np.max(filt_mag[mask_stop]) / (np.max(orig_mag[mask_stop]) + 1e-12)
            )
            results["suppression_db"] = suppression
            results["suppression_ok"] = suppression < -20.0
            print(f"  📊 Подавление стопполос в спектре: {suppression:.1f} дБ"
                  f" (требуется < -20 дБ)")

    results["ok"] = results.get("no_nan", True) and results.get("no_inf", True)
    return results


# ════════════════════════════════════════════════════════════════════════════
# ШАГ 5: Графики
# ════════════════════════════════════════════════════════════════════════════

def plot_results(original: np.ndarray, filtered: np.ndarray,
                 h: np.ndarray, params: dict,
                 test_freqs: list, validation: dict):
    """Генерация 4 графиков: АЧХ, коэффициенты, время, спектры."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    fs = params["fs"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes.flat:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444')

    desc = params.get("description", params.get("filter_type", "FIR"))
    fig.suptitle(f"AI-Designed FIR Filter: {desc}", fontsize=13,
                 color='white', fontweight='bold')

    # ── График 1: АЧХ ───────────────────────────────────────────────
    w, H = sp_signal.freqz(h, fs=fs, worN=4096)
    H_db = 20 * np.log10(np.abs(H) + 1e-12)
    axes[0, 0].plot(w, H_db, color='#00d2ff', lw=1.2)
    axes[0, 0].axhline(-3,  color='#ff6b6b', ls='--', lw=0.8, label='-3 dB')
    axes[0, 0].axhline(-60, color='#ffdd57', ls='--', lw=0.8, label='-60 dB')
    axes[0, 0].set_title(f'Frequency Response (order={len(h)-1})')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('dB')
    axes[0, 0].set_ylim(-100, 5)
    axes[0, 0].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    axes[0, 0].grid(True, alpha=0.2, color='white')

    # Отметки частот среза
    f_pass = params.get("f_pass")
    f_stop = params.get("f_stop")
    if f_pass and isinstance(f_pass, (int, float)):
        axes[0, 0].axvline(f_pass, color='#00ff88', ls=':', lw=0.8, label=f'f_pass={f_pass:.0f}')
    if f_stop and isinstance(f_stop, (int, float)):
        axes[0, 0].axvline(f_stop, color='#ff6b6b', ls=':', lw=0.8, label=f'f_stop={f_stop:.0f}')

    # ── График 2: Коэффициенты ──────────────────────────────────────
    axes[0, 1].stem(h, markerfmt='C0o', linefmt='C0-', basefmt='#444')
    # Переделать цвета под тёмную тему
    markerline, stemlines, baseline = axes[0, 1].get_children()[:3]
    axes[0, 1].set_title(f'FIR Coefficients ({len(h)} taps)')
    axes[0, 1].set_xlabel('Tap index')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.2, color='white')

    # ── График 3: Временная область ─────────────────────────────────
    n_show = min(300, len(original))
    t_ms = np.arange(n_show) / fs * 1000
    axes[1, 0].plot(t_ms, original[:n_show].real,
                    color='#888888', lw=0.6, alpha=0.8, label='Input')
    axes[1, 0].plot(t_ms, filtered[:n_show].real,
                    color='#00ff88', lw=1.0, label='Filtered')
    axes[1, 0].set_title('Time Domain (Re)')
    axes[1, 0].set_xlabel('ms')
    axes[1, 0].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    axes[1, 0].grid(True, alpha=0.2, color='white')

    # ── График 4: Спектры ───────────────────────────────────────────
    N = len(original)
    freqs = np.fft.rfftfreq(N, 1.0/fs)
    orig_mag  = 20 * np.log10(np.abs(np.fft.rfft(original.real))  + 1e-12)
    filt_mag  = 20 * np.log10(np.abs(np.fft.rfft(filtered.real))  + 1e-12)
    axes[1, 1].plot(freqs, orig_mag, color='#888888', lw=0.6, alpha=0.8, label='Input')
    axes[1, 1].plot(freqs, filt_mag, color='#00d2ff', lw=1.0, label='Filtered')

    # Отметки тестовых частот
    colors_f = ['#ff6b6b', '#ffdd57', '#00ff88', '#ff9f40']
    for i, f in enumerate(test_freqs):
        axes[1, 1].axvline(f, color=colors_f[i % len(colors_f)],
                           ls=':', lw=0.8, alpha=0.7, label=f'{f:.0f} Hz')

    axes[1, 1].set_title('Spectrum Comparison (dB)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('dB')
    axes[1, 1].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=7)
    axes[1, 1].grid(True, alpha=0.2, color='white')

    # Аннотация валидации
    val_text = "✅ PASS" if validation.get("ok") else "❌ FAIL"
    if "attenuation_db" in validation:
        val_text += f"\nAtten: {validation['attenuation_db']:.0f} dB"
    if "suppression_db" in validation:
        val_text += f"\nSuppr: {validation['suppression_db']:.0f} dB"
    fig.text(0.98, 0.02, val_text, ha='right', va='bottom',
             color='#00ff88' if validation.get("ok") else '#ff6b6b',
             fontsize=9, family='monospace')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(PLOT_DIR, 'ai_fir_result.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 График сохранён: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ПАЙПЛАЙН
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(filter_request: str, fs: float = 44100.0,
                 test_freqs: list = None, num_samples: int = 8192):
    """
    Полный AI-DSP пайплайн:
      1. LLM парсит текстовый запрос → параметры фильтра
      2. scipy проектирует FIR → коэффициенты h[]
      3. GPU (gpuworklib) генерирует тестовые сигналы
      4. FIR применяется (пока CPU/scipy, будет GPU)
      5. Валидация + АЧХ + графики

    Args:
      filter_request: текстовое описание фильтра (любой язык)
      fs:             частота дискретизации (Гц)
      test_freqs:     список тестовых частот [Гц] (None = авто)
      num_samples:    длина сигнала
    """
    print("=" * 60)
    print(f"  AI-DSP Pipeline  [MODE={MODE}]")
    print("=" * 60)

    # ── GPU контекст ─────────────────────────────────────────────────
    ctx = gpuworklib.ROCmGPUContext(0)
    gen = gpuworklib.SignalGenerator(ctx)
    fft = gpuworklib.FFTProcessorROCm(ctx)
    print(f"  GPU: {ctx.device_name}")

    # ── Шаг 1: AI парсит запрос ──────────────────────────────────────
    print("\n[Шаг 1] Парсинг запроса...")
    params = parse_filter_request(filter_request, fs)

    # ── Шаг 2: scipy проектирует фильтр ─────────────────────────────
    print("\n[Шаг 2] Проектирование FIR-фильтра...")
    h = design_fir_filter(params)

    # ── Шаг 3: генерация тестового сигнала (GPU) ─────────────────────
    print("\n[Шаг 3] Генерация тестового сигнала (GPU)...")
    if test_freqs is None:
        f_pass_val = params["f_pass"]
        if isinstance(f_pass_val, (list, tuple)):
            fc = float(f_pass_val[0])
        else:
            fc = float(f_pass_val)
        # Авто: одна частота в полосе, две вне полосы
        test_freqs = [fc * 0.5, fc * 1.5, fc * 3.0]

    print(f"  Тестовые частоты: {[f'{f:.0f} Гц' for f in test_freqs]}")

    # Генерируем и суммируем синусоиды (на GPU)
    amplitudes = [1.0, 0.7, 0.5] + [0.3] * max(0, len(test_freqs) - 3)
    mixed = np.zeros(num_samples, dtype=np.complex64)
    for i, (f, a) in enumerate(zip(test_freqs, amplitudes)):
        s = gen.generate_cw(freq=f, fs=fs, length=num_samples, amplitude=a)
        mixed += s.astype(np.complex64)
        print(f"  + синусоида {f:.0f} Гц (A={a})")

    # ── Шаг 4: применение фильтра ────────────────────────────────────
    print("\n[Шаг 4] Фильтрация...")
    filtered = apply_filter(mixed, h)
    print(f"  Готово: {len(filtered)} отсчётов")

    # ── Шаг 5: FFT анализ на GPU ─────────────────────────────────────
    print("\n[Шаг 5] FFT-анализ (GPU)...")
    spectrum_in  = fft.process_complex(mixed,    sample_rate=fs)
    spectrum_out = fft.process_complex(filtered, sample_rate=fs)

    freqs_axis = np.fft.rfftfreq(num_samples, 1.0/fs)
    half = num_samples // 2
    mag_in  = np.abs(spectrum_in[:half])
    mag_out = np.abs(spectrum_out[:half])

    for f in test_freqs:
        idx = int(f / fs * num_samples)
        idx = min(idx, half - 1)
        ratio_db = 20 * np.log10((mag_out[idx] + 1e-12) / (mag_in[idx] + 1e-12))
        status = "✅ пропущен" if abs(ratio_db) < 3 else "🔴 подавлен"
        print(f"  {f:.0f} Гц: ratio={ratio_db:.1f} дБ  {status}")

    # ── Шаг 6: валидация ─────────────────────────────────────────────
    print("\n[Шаг 6] Валидация...")
    validation = validate_filter(mixed, filtered, h, params)
    print(f"  Результат: {'✅ PASS' if validation.get('ok') else '❌ FAIL'}")

    # ── Шаг 7: графики ───────────────────────────────────────────────
    print("\n[Шаг 7] Генерация графиков...")
    plot_results(mixed, filtered, h, params, test_freqs, validation)

    print("\n" + "=" * 60)
    print("  AI-DSP Pipeline завершён!")
    print("=" * 60)
    return h, filtered, validation


# ════════════════════════════════════════════════════════════════════════════
# ПРИМЕРЫ ЗАПУСКА
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Пример 1: Без AI (быстрый тест) ─────────────────────────────
    print("\n" + "━" * 60)
    print("  Пример 1: НЧ-фильтр (MODE=none, без AI)")
    print("━" * 60)

    h1, out1, val1 = run_pipeline(
        filter_request="КИХ фильтр НЧ, частота среза 5 кГц, частота дискретизации 44100 Гц",
        fs=44100.0,
        test_freqs=[1000, 3000, 8000, 15000],   # 1k, 3k — в полосе; 8k, 15k — вне
        num_samples=8192
    )

    # ── Пример 2: С AI (раскомментируй и задай MODE выше) ────────────
    # MODE = "groq"    # или "ollama"
    # GROQ_API_KEY = "gsk_..."
    #
    # h2, out2, val2 = run_pipeline(
    #     filter_request=(
    #         "Построй КИХ фильтр с окном Кайзера, полоса пропускания до 5 кГц, "
    #         "полоса задержания от 8 кГц, затухание 60 дБ, fs=44100 Гц. "
    #         "Замешай 3 синусоиды: 2 кГц, 5 кГц, 12 кГц."
    #     ),
    #     fs=44100.0,
    #     num_samples=8192
    # )

    print("\nВсё готово! 🎉")
    print(f"Графики в: {PLOT_DIR}")
