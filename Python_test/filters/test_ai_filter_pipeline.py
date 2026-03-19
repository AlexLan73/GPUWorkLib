"""
AI Filter Pipeline — Stage 3: Natural Language -> GPU DSP
==========================================================

Full pipeline:
  1. User describes filter in natural language (Russian/English)
  2. AI (Groq / Ollama / none) parses -> JSON parameters
  3. scipy designs FIR or IIR filter coefficients
  4. gpuworklib filters on GPU (FirFilter / IirFilter)
  5. Beautiful 4-panel dark-theme plot + validation

AI Backends:
  MODE = "groq"    — free cloud, needs API key: console.groq.com
  MODE = "ollama"  — offline, needs: ollama pull qwen2.5-coder:7b
  MODE = "none"    — no AI, direct parameters (demo/test mode)

Install:
  pip install scipy matplotlib numpy
  pip install groq          # for Groq
  pip install ollama        # for Ollama (+ run: ollama serve)

Author: Kodo (AI Assistant)
Date:   2026-02-18
"""

import sys
import os
import json
import re
import numpy as np
import traceback

# ── Project root (Python_test/filters/ -> 2 levels up) ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for subdir in ["build/python", "build/debian-radeon9070/python", "build/python/Release", "build/python/Debug", "build/Release", "build/Debug"]:
    p = os.path.join(PROJECT_ROOT, subdir.replace("/", os.sep))
    if os.path.exists(p):
        sys.path.insert(0, p)
        break

# ── Imports ──
try:
    import gpuworklib as gw
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("WARNING: gpuworklib not found. GPU processing disabled.")

try:
    import scipy.signal as sig
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("WARNING: matplotlib not found.")


# ============================================================================
# CONFIGURATION
# ============================================================================

MODE = "groq"                # "groq" | "ollama" | "none"
GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "qwen2.5-coder:7b"

PLOT_DIR = os.path.join(PROJECT_ROOT, 'Results', 'Plots', 'filters')

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
_API_KEYS_PATH = os.path.join(PROJECT_ROOT, 'api_keys.json')
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


# ============================================================================
# AI BACKENDS
# ============================================================================

def _has_ai_backend():
    """Check if AI backend (groq/ollama) is available."""
    if MODE == "groq":
        try:
            from groq import Groq  # noqa: F401
            return True
        except ImportError:
            return False
    if MODE == "ollama":
        try:
            import ollama  # noqa: F401
            return True
        except ImportError:
            return False
    return True  # MODE="none" doesn't need backend


def ai_ask(prompt: str, system: str = "") -> str:
    """Send request to LLM. Returns text response or None if MODE='none'."""

    if MODE == "groq":
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("pip install groq")
        key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError(
                "Groq API key not set!\n"
                "  1. Register at console.groq.com (free)\n"
                "  2. Set: GROQ_API_KEY=gsk_... or pass in code"
            )
        client = Groq(api_key=key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.05,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    elif MODE == "ollama":
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
        return None


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response (usually in ```json block)."""
    if text is None:
        return {}
    # Try ```json ... ```
    m = re.search(r'```json\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
    # Try bare { ... }
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Cannot find JSON in LLM response:\n{text[:300]}")


# ============================================================================
# STEP 1: Parse natural language -> filter parameters
# ============================================================================

SYSTEM_PROMPT = """You are a DSP expert. Parse the user's filter description into JSON parameters.
Return ONLY valid JSON, nothing else.

The JSON must have these fields:
{
  "filter_class": "fir" or "iir",
  "filter_type": "lowpass" | "highpass" | "bandpass" | "bandstop",
  "f_cutoff": <Hz, single number for LP/HP, or [f_low, f_high] for BP/BS>,
  "fs": <sample rate Hz>,
  "order": <integer, filter order (for IIR: 2-10, for FIR: auto or 32-512)>,
  "window": "kaiser" | "hamming" | "hann" | "blackman" (FIR only),
  "ripple_db": <stopband attenuation in dB, default 60>,
  "description": "<brief description>"
}

Rules:
- If user says "butterworth", "biquad", "IIR", "recursive" -> filter_class = "iir"
- If user says "FIR", "convolution", "taps", "window" -> filter_class = "fir"
- If unclear, default to "iir" for simple LP/HP, "fir" for complex shapes
- All frequencies in Hz (not kHz, not normalized)
- For IIR: order is the filter order (each 2 -> one biquad section)
- Default order: IIR=4, FIR=auto (calculated from transition band)
"""


def parse_filter_request(text: str, fs: float) -> dict:
    """Parse natural language filter request -> parameter dict."""
    print(f"  Request: '{text}'")

    if MODE == "none":
        # Default demo parameters (detect keywords)
        text_lower = text.lower()

        # Detect filter class
        if any(w in text_lower for w in ['iir', 'butterworth', 'biquad', 'chebyshev', 'recursive']):
            fclass = "iir"
        elif any(w in text_lower for w in ['fir', 'taps', 'convolution', 'window']):
            fclass = "fir"
        else:
            fclass = "iir"  # default

        # Detect filter type
        if any(w in text_lower for w in ['bandpass', 'band-pass', 'polosa']):
            ftype = "bandpass"
        elif any(w in text_lower for w in ['bandstop', 'band-stop', 'notch', 'rezhektorn']):
            ftype = "bandstop"
        elif any(w in text_lower for w in ['highpass', 'high-pass', 'vch']):
            ftype = "highpass"
        else:
            ftype = "lowpass"

        # Extract frequency from text (simple regex)
        freq_match = re.findall(r'(\d+(?:\.\d+)?)\s*(?:hz|ghz|khz|gts|kgts)', text_lower)
        if not freq_match:
            freq_match = re.findall(r'(\d+(?:\.\d+)?)\s*(?:hertz|gertz|gerts)', text_lower)

        if ftype in ("bandpass", "bandstop") and len(freq_match) >= 2:
            f_cutoff = [float(freq_match[0]), float(freq_match[1])]
        elif freq_match:
            f_cutoff = float(freq_match[0])
            # Check if kHz
            if 'khz' in text_lower or 'kgts' in text_lower:
                f_cutoff *= 1000
        else:
            f_cutoff = fs / 10  # default 10% of fs

        # Extract order
        order_match = re.search(r'(?:order|poryadok|porjadok)\s*[=:]?\s*(\d+)', text_lower)
        if order_match:
            order = int(order_match.group(1))
        else:
            order = 6 if fclass == "iir" else 0  # 0 = auto for FIR

        params = {
            "filter_class": fclass,
            "filter_type": ftype,
            "f_cutoff": f_cutoff,
            "fs": fs,
            "order": order,
            "window": "kaiser",
            "ripple_db": 60.0,
            "description": f"{fclass.upper()} {ftype} fc={f_cutoff} Hz (demo, no AI)"
        }
        print(f"  MODE=none: using parsed parameters")
        print(f"     {json.dumps(params, indent=2, ensure_ascii=False)}")
        return params

    # AI parses request
    prompt = f"""Filter description: "{text}"
Sample rate: {fs} Hz

Return JSON with the fields described in the system prompt."""

    raw = ai_ask(prompt, system=SYSTEM_PROMPT)
    print(f"  AI response: {raw[:200]}...")
    params = extract_json(raw)
    params["fs"] = fs
    if "filter_class" not in params:
        params["filter_class"] = "fir"
    print(f"  Parameters: {json.dumps(params, indent=2, ensure_ascii=False)}")
    return params


# ============================================================================
# STEP 2: Design filter (scipy)
# ============================================================================

def sos_to_sections(sos):
    """Convert scipy SOS matrix to list of dicts for gpuworklib IirFilter."""
    sections = []
    for row in sos:
        sections.append({
            'b0': float(row[0]), 'b1': float(row[1]), 'b2': float(row[2]),
            'a1': float(row[4]), 'a2': float(row[5]),
        })
    return sections


def design_filter(params: dict):
    """
    Design FIR or IIR filter using scipy.

    Returns:
        filter_class: "fir" or "iir"
        coeffs: FIR taps (np.array) or IIR SOS matrix (np.array)
        sections: list of dicts for gpuworklib (IIR only, None for FIR)
    """
    fs = params["fs"]
    fclass = params.get("filter_class", "iir")
    ftype = params["filter_type"]
    f_cutoff = params["f_cutoff"]
    order = int(params.get("order", 4))
    window = params.get("window", "kaiser")
    ripple_db = float(params.get("ripple_db", 60.0))

    # Normalize cutoff for scipy butter (Wn in [0,1] where 1 = Nyquist)
    nyquist = fs / 2.0
    if isinstance(f_cutoff, (list, tuple)):
        wn = [f / nyquist for f in f_cutoff]
    else:
        wn = float(f_cutoff) / nyquist

    if fclass == "iir":
        # ── IIR Butterworth ──
        if order < 1:
            order = 4
        print(f"  Designing IIR Butterworth: order={order}, type={ftype}, "
              f"fc={f_cutoff} Hz, fs={fs} Hz")

        btype_map = {"lowpass": "low", "highpass": "high",
                     "bandpass": "band", "bandstop": "bandstop"}
        btype = btype_map.get(ftype, "low")

        sos = sig.butter(order, wn, btype=btype, output='sos').astype(np.float64)
        sections = sos_to_sections(sos)
        print(f"  Biquad sections: {len(sections)} (order {order})")
        return "iir", sos, sections

    else:
        # ── FIR ──
        if order <= 0:
            # Auto order via Kaiser
            if isinstance(f_cutoff, (list, tuple)):
                trans_width = max(abs(f_cutoff[1] - f_cutoff[0]) * 0.2, 200)
            else:
                trans_width = max(f_cutoff * 0.3, 200)
            N, beta = sig.kaiserord(ripple_db, trans_width / nyquist)
            N = N if N % 2 == 1 else N + 1
            window_arg = ("kaiser", beta)
            print(f"  FIR Kaiser auto: N={N}, beta={beta:.2f}, "
                  f"trans_width={trans_width:.0f} Hz")
        else:
            N = order if order % 2 == 1 else order + 1
            if window == "kaiser":
                window_arg = ("kaiser", 5.0)
            else:
                window_arg = window
            print(f"  FIR {window}: N={N}")

        # Cutoff for firwin
        if ftype == "lowpass":
            cutoff_norm = wn
            pass_zero = True
        elif ftype == "highpass":
            cutoff_norm = wn
            pass_zero = False
        elif ftype == "bandpass":
            cutoff_norm = wn
            pass_zero = False
        elif ftype == "bandstop":
            cutoff_norm = wn
            pass_zero = True
        else:
            cutoff_norm = wn
            pass_zero = True

        h = sig.firwin(N, cutoff_norm, window=window_arg,
                        pass_zero=pass_zero).astype(np.float32)
        print(f"  FIR taps: {len(h)}, sum={np.sum(h):.6f}")
        return "fir", h, None


# ============================================================================
# STEP 3: Generate test signal (GPU)
# ============================================================================

def generate_test_signal(fs: float, test_freqs: list, num_samples: int = 4096):
    """Generate multi-frequency complex test signal on GPU."""
    if HAS_GPU:
        ctx = gw.GPUContext(0)
        gen = gw.SignalGenerator(ctx)
        mixed = np.zeros(num_samples, dtype=np.complex64)
        amplitudes = [1.0, 0.7, 0.5, 0.3]
        for i, f in enumerate(test_freqs):
            a = amplitudes[i] if i < len(amplitudes) else 0.3
            s = gen.generate_cw(freq=f, fs=fs, length=num_samples, amplitude=a)
            mixed += s.astype(np.complex64)
        return mixed, ctx
    else:
        # CPU fallback
        t = np.arange(num_samples) / fs
        mixed = np.zeros(num_samples, dtype=np.complex64)
        amplitudes = [1.0, 0.7, 0.5, 0.3]
        for i, f in enumerate(test_freqs):
            a = amplitudes[i] if i < len(amplitudes) else 0.3
            mixed += a * np.exp(1j * 2 * np.pi * f * t).astype(np.complex64)
        return mixed, None


# ============================================================================
# STEP 4: Apply filter on GPU
# ============================================================================

def apply_filter_gpu(signal_1d: np.ndarray, fclass: str,
                     coeffs, sections, ctx):
    """
    Apply FIR or IIR filter on GPU via gpuworklib.

    Args:
        signal_1d: 1D complex64 array
        fclass: "fir" or "iir"
        coeffs: FIR taps (np.array) or IIR SOS matrix (np.array)
        sections: list of dicts for IIR (from sos_to_sections)
        ctx: GPUContext
    """
    if not HAS_GPU or ctx is None:
        # CPU fallback via scipy
        print("  [CPU fallback - scipy]")
        if fclass == "fir":
            return sig.lfilter(coeffs, [1.0], signal_1d).astype(np.complex64)
        else:
            return sig.sosfilt(coeffs, signal_1d).astype(np.complex64)

    if fclass == "fir":
        fir = gw.FirFilter(ctx)
        fir.set_coefficients(coeffs.tolist())
        result = fir.process(signal_1d)
        print(f"  GPU FirFilter: {fir.num_taps} taps")
        return result
    else:
        iir = gw.IirFilter(ctx)
        iir.set_sections(sections)
        result = iir.process(signal_1d)
        print(f"  GPU IirFilter: {iir.num_sections} biquad sections")
        return result


def apply_filter_scipy(signal_1d: np.ndarray, fclass: str, coeffs):
    """Apply filter with scipy (for validation/comparison)."""
    if fclass == "fir":
        return sig.lfilter(coeffs, [1.0], signal_1d).astype(np.complex64)
    else:
        return sig.sosfilt(coeffs, signal_1d).astype(np.complex64)


# ============================================================================
# STEP 5: Validation
# ============================================================================

def validate_results(signal_in, result_gpu, result_scipy,
                     fclass, coeffs, params, test_freqs):
    """Validate GPU result vs scipy reference + spectral checks."""
    results = {}
    fs = params["fs"]

    # 1. GPU vs scipy match
    max_err = float(np.max(np.abs(result_gpu - result_scipy)))
    results["gpu_vs_scipy_err"] = max_err
    results["gpu_match"] = max_err < 0.1  # generous threshold
    print(f"  GPU vs scipy: max_err = {max_err:.2e}  "
          f"{'PASS' if results['gpu_match'] else 'FAIL'}")

    # 2. No NaN/Inf
    results["no_nan"] = bool(not np.any(np.isnan(result_gpu)))
    results["no_inf"] = bool(not np.any(np.isinf(result_gpu)))

    # 3. Spectral suppression check (use .real for rfft, complex64 not supported)
    N = len(signal_in)
    freqs_axis = np.fft.rfftfreq(N, 1.0 / fs)
    mag_in = np.abs(np.fft.rfft(signal_in.real.astype(np.float64)))
    mag_out = np.abs(np.fft.rfft(result_gpu.real.astype(np.float64)))

    ftype = params["filter_type"]
    f_cutoff = params["f_cutoff"]

    for f in test_freqs:
        idx = np.argmin(np.abs(freqs_axis - f))
        if mag_in[idx] > 1e-6:
            ratio_db = 20 * np.log10((mag_out[idx] + 1e-12) / (mag_in[idx] + 1e-12))
        else:
            ratio_db = 0.0

        # Determine expected behavior
        if ftype == "lowpass":
            fc = float(f_cutoff) if not isinstance(f_cutoff, (list, tuple)) else f_cutoff[0]
            expected = "pass" if f < fc * 0.8 else "stop"
        elif ftype == "highpass":
            fc = float(f_cutoff) if not isinstance(f_cutoff, (list, tuple)) else f_cutoff[0]
            expected = "pass" if f > fc * 1.2 else "stop"
        elif ftype == "bandpass":
            expected = "pass" if f_cutoff[0] < f < f_cutoff[1] else "stop"
        else:
            expected = "unknown"

        if expected == "pass":
            ok = ratio_db > -6.0
            status = "PASS" if ok else "ATTENUATED"
        elif expected == "stop":
            ok = ratio_db < -10.0
            status = "SUPPRESSED" if ok else "LEAKED"
        else:
            ok = True
            status = "---"

        symbol = "+" if ok else "!"
        print(f"  [{symbol}] {f:.0f} Hz: {ratio_db:+.1f} dB  ({status})")

    results["ok"] = (results["gpu_match"] and results["no_nan"]
                     and results["no_inf"])
    return results


# ============================================================================
# STEP 6: Beautiful 4-panel dark-theme plot
# ============================================================================

def plot_ai_results(signal_in, result_gpu, fclass, coeffs, params,
                    test_freqs, validation, filename="ai_filter_result.png"):
    """Generate beautiful 4-panel dark-theme plot."""
    if not HAS_PLOT:
        print("  SKIP: matplotlib not found")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    fs = params["fs"]
    f_cutoff = params["f_cutoff"]
    desc = params.get("description", f"{fclass.upper()} filter")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Dark theme
    fig.patch.set_facecolor('#0f0f23')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a3e')
        ax.tick_params(colors='#cccccc', labelsize=9)
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.title.set_color('#ffffff')
        for spine in ax.spines.values():
            spine.set_color('#333366')

    # Title
    gpu_tag = "GPU" if HAS_GPU else "CPU"
    ai_tag = MODE.upper() if MODE != "none" else "Direct"
    fig.suptitle(
        f'AI Filter Pipeline  [{ai_tag}]  ->  scipy  ->  {gpu_tag}\n'
        f'{desc}',
        fontsize=14, fontweight='bold', color='#ffffff', y=0.98
    )

    t_ms = np.arange(len(signal_in)) / fs * 1000
    n_show = min(500, len(signal_in))

    # ── Panel 1: Time domain (input vs filtered) ──
    ax1 = axes[0, 0]
    ax1.plot(t_ms[:n_show], signal_in.real[:n_show],
             color='#555588', linewidth=0.6, alpha=0.7, label='Input Re')
    ax1.plot(t_ms[:n_show], result_gpu.real[:n_show],
             color='#00ff88', linewidth=1.0, alpha=0.9, label='Filtered Re')
    ax1.set_title('Time Domain (Real)', fontsize=11)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(facecolor='#1a1a3e', labelcolor='#cccccc', fontsize=8,
               edgecolor='#333366')
    ax1.grid(True, alpha=0.15, color='#666699')
    ax1.set_xlim([0, t_ms[n_show - 1]])

    # ── Panel 2: Frequency Response (magnitude dB) ──
    ax2 = axes[0, 1]

    if fclass == "fir":
        w, h_freq = sig.freqz(coeffs, worN=4096)
    else:
        w, h_freq = sig.sosfreqz(coeffs, worN=4096)
    freq_hz = w / np.pi * (fs / 2)
    mag_db = 20 * np.log10(np.abs(h_freq) + 1e-12)

    ax2.plot(freq_hz, mag_db, color='#00d2ff', linewidth=1.5, label='|H(f)|')
    ax2.axhline(-3, color='#ff6b6b', ls='--', lw=0.8, alpha=0.7, label='-3 dB')
    ax2.axhline(-60, color='#ffdd57', ls='--', lw=0.8, alpha=0.5, label='-60 dB')

    # Mark cutoff
    if isinstance(f_cutoff, (list, tuple)):
        for fc in f_cutoff:
            ax2.axvline(fc, color='#00ff88', ls=':', lw=1.0, alpha=0.8)
    else:
        ax2.axvline(float(f_cutoff), color='#00ff88', ls=':', lw=1.0,
                     alpha=0.8, label=f'fc={float(f_cutoff):.0f} Hz')

    # Mark test frequencies
    test_colors = ['#ff6b6b', '#ffdd57', '#ff9f40', '#cc66ff']
    for i, f in enumerate(test_freqs):
        c = test_colors[i % len(test_colors)]
        ax2.axvline(f, color=c, ls=':', lw=0.6, alpha=0.5)

    ax2.set_title('Frequency Response', fontsize=11)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_ylim([-100, 5])
    ax2.legend(facecolor='#1a1a3e', labelcolor='#cccccc', fontsize=8,
               edgecolor='#333366', loc='lower left')
    ax2.grid(True, alpha=0.15, color='#666699')

    # ── Panel 3: Spectrum comparison (input vs output) ──
    ax3 = axes[1, 0]
    N = len(signal_in)
    freqs_axis = np.fft.rfftfreq(N, 1.0 / fs)
    mag_in = 20 * np.log10(np.abs(np.fft.rfft(signal_in.real.astype(np.float64))) + 1e-12)
    mag_out = 20 * np.log10(np.abs(np.fft.rfft(result_gpu.real.astype(np.float64))) + 1e-12)

    ax3.plot(freqs_axis, mag_in, color='#555588', linewidth=0.6,
             alpha=0.7, label='Input')
    ax3.plot(freqs_axis, mag_out, color='#00d2ff', linewidth=1.0,
             alpha=0.9, label='Filtered')

    # Mark test frequencies with suppression values
    for i, f in enumerate(test_freqs):
        c = test_colors[i % len(test_colors)]
        ax3.axvline(f, color=c, ls=':', lw=0.8, alpha=0.6,
                     label=f'{f:.0f} Hz')

    ax3.set_title('Spectrum Comparison', fontsize=11)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.legend(facecolor='#1a1a3e', labelcolor='#cccccc', fontsize=7,
               edgecolor='#333366', loc='lower left')
    ax3.grid(True, alpha=0.15, color='#666699')

    # ── Panel 4: Pole-Zero / Impulse Response ──
    ax4 = axes[1, 1]

    if fclass == "iir":
        # Pole-Zero diagram
        z_all = np.array([], dtype=complex)
        p_all = np.array([], dtype=complex)
        for row in coeffs:  # coeffs = SOS matrix
            z_sec = np.roots(row[:3])
            p_sec = np.roots(row[3:])
            z_all = np.concatenate([z_all, z_sec])
            p_all = np.concatenate([p_all, p_sec])

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 256)
        ax4.plot(np.cos(theta), np.sin(theta), color='#444477',
                 linewidth=1.2, alpha=0.4)
        ax4.fill(np.cos(theta), np.sin(theta), color='#1a1a4e', alpha=0.3)

        # Zeros
        ax4.plot(z_all.real, z_all.imag, 'o', color='#00d2ff', markersize=9,
                 markerfacecolor='none', markeredgewidth=2,
                 label=f'Zeros ({len(z_all)})', zorder=5)
        # Poles
        ax4.plot(p_all.real, p_all.imag, 'x', color='#ff6b6b', markersize=11,
                 markeredgewidth=2.5, label=f'Poles ({len(p_all)})', zorder=5)

        ax4.axhline(0, color='#444477', linewidth=0.5)
        ax4.axvline(0, color='#444477', linewidth=0.5)
        ax4.set_title(f'Pole-Zero (IIR order {params.get("order", "?")})',
                       fontsize=11)
        ax4.set_xlabel('Real')
        ax4.set_ylabel('Imaginary')
        ax4.set_aspect('equal')
        lim = max(1.3, np.max(np.abs(p_all)) * 1.3,
                  np.max(np.abs(z_all)) * 1.3)
        ax4.set_xlim([-lim, lim])
        ax4.set_ylim([-lim, lim])
        ax4.legend(facecolor='#1a1a3e', labelcolor='#cccccc', fontsize=9,
                   edgecolor='#333366', loc='upper left')
    else:
        # FIR: Impulse response (stem plot)
        stem_container = ax4.stem(range(len(coeffs)), coeffs,
                                   linefmt='-', markerfmt='o', basefmt='-')
        stem_container.stemlines.set_color('#00d2ff')
        stem_container.stemlines.set_alpha(0.7)
        stem_container.markerline.set_color('#00ff88')
        stem_container.markerline.set_markersize(3)
        stem_container.baseline.set_color('#444477')
        ax4.set_title(f'Impulse Response ({len(coeffs)} taps)', fontsize=11)
        ax4.set_xlabel('Tap index')
        ax4.set_ylabel('h[n]')

    ax4.grid(True, alpha=0.15, color='#666699')

    # ── Validation badge ──
    err = validation.get("gpu_vs_scipy_err", 0)
    ok = validation.get("ok", False)
    badge_text = (f"{'PASS' if ok else 'FAIL'}  |  "
                  f"GPU err: {err:.2e}  |  "
                  f"Mode: {MODE}  |  Engine: {gpu_tag}")
    badge_color = '#00ff88' if ok else '#ff6b6b'
    fig.text(0.5, 0.01, badge_text, ha='center', va='bottom',
             color=badge_color, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f0f23',
                       edgecolor=badge_color, alpha=0.9))

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")
    return out_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_ai_pipeline(filter_request: str,
                    fs: float = 50000.0,
                    test_freqs: list = None,
                    num_samples: int = 4096,
                    plot_filename: str = "ai_filter_result.png"):
    """
    Full AI-DSP Pipeline:
      1. AI/Parser -> filter parameters
      2. scipy designs FIR/IIR
      3. GPU generates test signal
      4. GPU applies filter (FirFilter / IirFilter)
      5. Validation (GPU vs scipy)
      6. Beautiful 4-panel plot

    Args:
        filter_request: natural language description (Russian/English)
        fs: sample rate (Hz)
        test_freqs: list of test frequencies (Hz). None = auto
        num_samples: signal length
        plot_filename: output plot filename
    """
    print("\n" + "=" * 70)
    print(f"  AI-DSP Pipeline  [MODE={MODE}]")
    print("=" * 70)

    # ── Step 1: Parse request ──
    print("\n[Step 1] Parsing filter request...")
    params = parse_filter_request(filter_request, fs)

    # ── Step 2: Design filter ──
    print("\n[Step 2] Designing filter (scipy)...")
    fclass, coeffs, sections = design_filter(params)

    # ── Step 3: Generate test signal ──
    print("\n[Step 3] Generating test signal...")
    if test_freqs is None:
        fc = params["f_cutoff"]
        if isinstance(fc, (list, tuple)):
            fc_val = (fc[0] + fc[1]) / 2
        else:
            fc_val = float(fc)
        ftype = params["filter_type"]
        if ftype == "lowpass":
            test_freqs = [fc_val * 0.2, fc_val * 0.8, fc_val * 2.0, fc_val * 4.0]
        elif ftype == "highpass":
            test_freqs = [fc_val * 0.2, fc_val * 0.5, fc_val * 2.0, fc_val * 5.0]
        elif ftype == "bandpass":
            test_freqs = [fc[0] * 0.3, (fc[0] + fc[1]) / 2, fc[1] * 2.0, fc[1] * 4.0]
        else:
            test_freqs = [fc_val * 0.5, fc_val, fc_val * 2.0, fc_val * 4.0]

    # Clamp test freqs to Nyquist
    test_freqs = [f for f in test_freqs if f < fs / 2 * 0.95]

    signal_in, ctx = generate_test_signal(fs, test_freqs, num_samples)
    print(f"  Signal: {num_samples} samples, {len(test_freqs)} test frequencies")
    for f in test_freqs:
        print(f"    + {f:.0f} Hz")

    # ── Step 4: Apply filter on GPU ──
    print("\n[Step 4] Filtering on GPU...")
    result_gpu = apply_filter_gpu(signal_in, fclass, coeffs, sections, ctx)

    # ── Step 5: Scipy reference ──
    print("\n[Step 5] Scipy reference (for validation)...")
    result_scipy = apply_filter_scipy(signal_in, fclass, coeffs)

    # ── Step 6: Validation ──
    print("\n[Step 6] Validation...")
    validation = validate_results(signal_in, result_gpu, result_scipy,
                                   fclass, coeffs, params, test_freqs)

    # ── Step 7: Plot ──
    print("\n[Step 7] Generating plot...")
    plot_path = plot_ai_results(signal_in, result_gpu, fclass, coeffs,
                                 params, test_freqs, validation, plot_filename)

    print("\n" + "=" * 70)
    status = "PASSED" if validation.get("ok") else "FAILED"
    print(f"  Pipeline complete!  [{status}]")
    print("=" * 70)

    return {
        "params": params,
        "filter_class": fclass,
        "coeffs": coeffs,
        "sections": sections,
        "result_gpu": result_gpu,
        "result_scipy": result_scipy,
        "validation": validation,
        "plot_path": plot_path,
    }


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

def demo_iir_lowpass():
    """Demo: IIR Butterworth low-pass 2500 Hz"""
    return run_ai_pipeline(
        filter_request="IIR Butterworth low-pass filter, cutoff 2500 Hz, order 8",
        fs=50000.0,
        test_freqs=[500, 1500, 5000, 15000],
        num_samples=4096,
        plot_filename="ai_iir_lowpass.png"
    )


def demo_fir_lowpass():
    """Demo: FIR low-pass Kaiser window 5000 Hz"""
    return run_ai_pipeline(
        filter_request="FIR low-pass filter with Kaiser window, cutoff 5000 Hz, "
                        "stopband attenuation 60 dB",
        fs=50000.0,
        test_freqs=[1000, 3000, 8000, 18000],
        num_samples=4096,
        plot_filename="ai_fir_lowpass.png"
    )


def demo_iir_highpass():
    """Demo: IIR Butterworth high-pass 3000 Hz"""
    return run_ai_pipeline(
        filter_request="IIR highpass Butterworth, cutoff 3000 Hz, order 6",
        fs=50000.0,
        test_freqs=[500, 1500, 5000, 15000],
        num_samples=4096,
        plot_filename="ai_iir_highpass.png"
    )


def demo_russian_request():
    """Demo: request in Russian"""
    return run_ai_pipeline(
        filter_request="Фильтр Баттерворта нижних частот, частота среза 2000 Hz, "
                        "порядок 6",
        fs=44100.0,
        test_freqs=[500, 1500, 5000, 15000],
        num_samples=8192,
        plot_filename="ai_russian_request.png"
    )


# ============================================================================
# PYTEST TESTS
# ============================================================================

def test_ai_pipeline_iir_lowpass():
    """AI Pipeline: IIR low-pass produces valid filtered output"""
    if not HAS_GPU or not HAS_SCIPY:
        return  # missing gpuworklib or scipy — skip silently
    if not _has_ai_backend():
        return  # AI backend not available
    result = demo_iir_lowpass()
    assert result["validation"]["ok"], "IIR pipeline validation failed"
    print("  PASSED")


def test_ai_pipeline_fir_lowpass():
    """AI Pipeline: FIR low-pass produces valid filtered output"""
    if not HAS_GPU or not HAS_SCIPY:
        return
    if not _has_ai_backend():
        return
    result = demo_fir_lowpass()
    assert result["validation"]["ok"], "FIR pipeline validation failed"
    print("  PASSED")


def test_ai_pipeline_iir_highpass():
    """AI Pipeline: IIR high-pass produces valid filtered output"""
    if not HAS_GPU or not HAS_SCIPY:
        return
    if not _has_ai_backend():
        return
    result = demo_iir_highpass()
    assert result["validation"]["ok"], "IIR highpass pipeline validation failed"
    print("  PASSED")


def test_ai_pipeline_russian():
    """AI Pipeline: Russian language request works"""
    if not HAS_GPU or not HAS_SCIPY:
        return
    if not _has_ai_backend():
        return
    result = demo_russian_request()
    assert result["validation"]["ok"], "Russian request pipeline failed"
    print("  PASSED")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "*" * 70)
    print("*" + " " * 20 + "AI FILTER PIPELINE" + " " * 20 + " *")
    print("*" + f"  MODE={MODE}  |  GPU={'YES' if HAS_GPU else 'NO'}  "
          f"|  scipy={'YES' if HAS_SCIPY else 'NO'}".center(68) + "*")
    print("*" * 70)

    demos = [
        ("1. IIR Butterworth Low-Pass (2500 Hz, order 8)", demo_iir_lowpass),
        ("2. FIR Kaiser Low-Pass (5000 Hz, auto order)", demo_fir_lowpass),
        ("3. IIR Butterworth High-Pass (3000 Hz, order 6)", demo_iir_highpass),
        ("4. Russian Language Request", demo_russian_request),
    ]

    results = []
    for title, demo_fn in demos:
        print(f"\n{'#' * 70}")
        print(f"  {title}")
        print(f"{'#' * 70}")
        try:
            r = demo_fn()
            results.append((title, r["validation"].get("ok", False)))
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append((title, False))

    # Summary
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for title, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {title}")

    all_ok = all(ok for _, ok in results)
    print(f"\n  Overall: {'ALL PASSED!' if all_ok else 'SOME FAILED'}")
    print(f"  Plots saved to: {PLOT_DIR}")
    print("=" * 70)
