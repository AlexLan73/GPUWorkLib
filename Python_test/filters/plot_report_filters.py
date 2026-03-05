#!/usr/bin/env python3
"""
Графики для отчёта по Task_20 / Task_21 / Task_22 — фильтры ROCm GPU
======================================================================

  Task_20: MovingAverageFilterROCm  — SMA, EMA, MMA, DEMA, TEMA
  Task_21: KalmanFilterROCm          — 1D скалярный фильтр Калмана
  Task_22: KaufmanFilterROCm (KAMA)  — адаптивная скользящая средняя

Запуск (без GPU):
  python Python_test/filters/plot_report_filters.py

Выходные файлы:
  Results/Plots/filters/report_task20_moving_average.png
  Results/Plots/filters/report_task21_kalman_filter.png
  Results/Plots/filters/report_task22_kaufman_kama.png

Автор: Кодо
Дата: 2026-03-05
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as mticker

# ─── Пути ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR      = os.path.join(PROJECT_ROOT, 'Results', 'Plots', 'filters')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Стиль ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   12,
    'axes.labelsize':   10,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linewidth':   0.6,
    'lines.linewidth':  1.8,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# Цветовая палитра
C_INPUT = '#888888'
C_SMA   = '#3498db'   # синий
C_EMA   = '#e74c3c'   # красный
C_MMA   = '#27ae60'   # зелёный
C_DEMA  = '#f39c12'   # оранжевый
C_TEMA  = '#9b59b6'   # фиолетовый
C_KAMA  = '#1abc9c'   # бирюза
C_KALMAN= '#2c3e50'   # тёмно-синий
C_NOISY = '#bdc3c7'   # серый (зашумлённый)
C_COVAR = '#e67e22'   # оранжевый


# ═══════════════════════════════════════════════════════════════════════════════
# ЭТАЛОННЫЕ РЕАЛИЗАЦИИ (CPU, float32, совпадают с GPU-ядрами)
# ═══════════════════════════════════════════════════════════════════════════════

def _ema_1ch(data, alpha):
    """Одноканальный EMA: state = alpha*x + (1-alpha)*state."""
    a  = np.float32(alpha)
    om = np.float32(1.0) - a
    n   = len(data)
    out = np.empty(n, dtype=np.float32)
    s   = np.float32(data[0])
    out[0] = s
    for i in range(1, n):
        s = a * np.float32(data[i]) + om * s
        out[i] = s
    return out

def ema_ref(data, N):
    return _ema_1ch(data, 2.0 / (N + 1))

def mma_ref(data, N):
    return _ema_1ch(data, 1.0 / N)

def sma_ref(data, N):
    n   = len(data)
    out = np.empty(n, dtype=np.float32)
    buf = np.zeros(N, dtype=np.float32)
    s   = np.float32(0.0)
    head = 0
    inv_N = np.float32(1.0 / N)
    for i in range(n):
        x = np.float32(data[i])
        if i < N:
            buf[i] = x; s += x
            out[i] = s / np.float32(i + 1)
        else:
            old = buf[head]; buf[head] = x
            head = head + 1 if head + 1 < N else 0
            s += x - old
            out[i] = s * inv_N
    return out

def dema_ref(data, N):
    alpha = 2.0 / (N + 1)
    e1    = _ema_1ch(data, alpha)
    e2    = _ema_1ch(e1,   alpha)
    return (2.0 * e1 - e2).astype(np.float32)

def tema_ref(data, N):
    alpha = 2.0 / (N + 1)
    e1    = _ema_1ch(data, alpha)
    e2    = _ema_1ch(e1,   alpha)
    e3    = _ema_1ch(e2,   alpha)
    return (3.0 * e1 - 3.0 * e2 + e3).astype(np.float32)

def kalman_ref_scalar(data, Q=0.1, R=25.0, x0=0.0, P0=25.0):
    """1D скалярный фильтр Калмана."""
    Q_f, R_f = np.float32(Q), np.float32(R)
    n    = len(data)
    out  = np.empty(n, dtype=np.float32)
    Parr = np.empty(n, dtype=np.float32)
    Karr = np.empty(n, dtype=np.float32)
    xh   = np.float32(x0)
    P    = np.float32(P0)
    for i in range(n):
        z      = np.float32(data[i])
        P_pred = P + Q_f
        K      = P_pred / (P_pred + R_f)
        xh     = xh + K * (z - xh)
        P      = (np.float32(1.0) - K) * P_pred
        out[i]  = xh
        Parr[i] = P
        Karr[i] = K
    return out, Parr, Karr

def _kaufman_1ch_real(data, N, fast_sc, slow_sc):
    """KAMA для вещественного 1D сигнала (float32)."""
    n       = len(data)
    out     = np.empty(n, dtype=np.float32)
    er_arr  = np.zeros(n, dtype=np.float32)
    sc_arr  = np.zeros(n, dtype=np.float32)
    fast    = np.float32(fast_sc)
    slow    = np.float32(slow_sc)
    sc_rng  = fast - slow
    eps     = np.float32(1e-8)

    ring = np.array(data[:N], dtype=np.float32)
    out[:N] = ring
    if n <= N:
        return out, er_arr, sc_arr

    kama = ring[0]
    vol  = np.float32(sum(abs(float(ring[i]) - float(ring[i-1])) for i in range(1, N)))
    head = 0

    for i in range(N, n):
        x           = np.float32(data[i])
        prev_idx    = (head + N - 1) % N
        before_head = (head - 1 + N) % N
        direction   = abs(x - ring[head])
        old_d       = abs(ring[head] - ring[before_head])
        new_d       = abs(x - ring[prev_idx])
        vol         = vol - old_d + new_d
        er          = direction / vol if vol > eps else np.float32(0.0)
        sc          = (er * sc_rng + slow) ** 2
        kama        = kama + sc * (x - kama)
        ring[head]  = x
        head        = head + 1 if head + 1 < N else 0
        out[i]     = kama
        er_arr[i]  = er
        sc_arr[i]  = sc
    return out, er_arr, sc_arr

def kaufman_ref(data, N=10, fast=2, slow=30):
    fsc = 2.0 / (fast + 1)
    ssc = 2.0 / (slow + 1)
    return _kaufman_1ch_real(data, N, fsc, ssc)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Task_20: Скользящие средние
# ═══════════════════════════════════════════════════════════════════════════════

def plot_task20():
    print("  [Task_20] Строю графики скользящих средних...")
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Task_20 · MovingAverageFilterROCm · SMA / EMA / MMA / DEMA / TEMA\n"
        "GPU ROCm (hiprtc) · Radeon 9070 gfx1201 · Тесты: 6/6 PASSED",
        fontsize=13, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

    N      = 10
    points = 200   # достаточно для визуализации

    # ── Сигналы ──────────────────────────────────────────────────────────────
    sig_step = np.zeros(120, dtype=np.float32)
    sig_step[20:70] = 1.0

    t_step  = np.arange(120)
    out_sma  = sma_ref(sig_step, N)
    out_ema  = ema_ref(sig_step, N)
    out_mma  = mma_ref(sig_step, N)
    out_dema = dema_ref(sig_step, N)
    out_tema = tema_ref(sig_step, N)

    # ── [0,0] Step Response — все 5 фильтров ─────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.step(t_step, sig_step, where='post',
             color=C_INPUT, lw=1.2, ls='--', alpha=0.55, label='Входной сигнал')
    ax0.plot(t_step, out_sma,  color=C_SMA,  label=f'SMA (N={N})')
    ax0.plot(t_step, out_ema,  color=C_EMA,  label=f'EMA (N={N})')
    ax0.plot(t_step, out_mma,  color=C_MMA,  label=f'MMA / Wilder')
    ax0.plot(t_step, out_dema, color=C_DEMA, label='DEMA')
    ax0.plot(t_step, out_tema, color=C_TEMA, label='TEMA')
    ax0.set_title('Ступенчатый отклик (N=10)')
    ax0.set_xlabel('Отсчёт n')
    ax0.set_ylabel('Амплитуда')
    ax0.legend(loc='center right', fontsize=8)
    ax0.axvline(20, color='gray', ls=':', alpha=0.5, lw=1.0)
    ax0.axvline(70, color='gray', ls=':', alpha=0.5, lw=1.0)
    ax0.text(20, -0.12, 'ON', ha='center', fontsize=8, color='gray')
    ax0.text(70, -0.12, 'OFF', ha='center', fontsize=8, color='gray')
    ax0.set_ylim(-0.15, 1.22)
    ax0.set_xlim(-1, 120)

    # ── [0,1] Zoom: нарастающий фронт (t = 15..50) ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    t_zoom = np.arange(15, 50)
    ax1.step(t_zoom, sig_step[15:50], where='post',
             color=C_INPUT, lw=1.2, ls='--', alpha=0.55, label='Вход')
    ax1.plot(t_zoom, out_sma[15:50],  color=C_SMA,  label=f'SMA')
    ax1.plot(t_zoom, out_ema[15:50],  color=C_EMA,  label=f'EMA')
    ax1.plot(t_zoom, out_mma[15:50],  color=C_MMA,  label=f'MMA')
    ax1.plot(t_zoom, out_dema[15:50], color=C_DEMA, label='DEMA')
    ax1.plot(t_zoom, out_tema[15:50], color=C_TEMA, label='TEMA')
    # Горизонтальная отметка 0.9
    ax1.axhline(0.9, color='grey', ls=':', lw=0.9, alpha=0.7)
    ax1.text(48.3, 0.91, '0.9', fontsize=8, color='grey', va='bottom')
    ax1.set_title('Фронт нарастания (zoom)')
    ax1.set_xlabel('Отсчёт n')
    ax1.set_ylabel('Амплитуда')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(-0.05, 1.1)

    # ── [0,2] Сравнение скоростей — bar chart (отсчёт достижения 0.5) ────────
    ax2 = fig.add_subplot(gs[0, 2])

    def reach_level(arr, step_on, level=0.5):
        """Отсчёт, когда arr впервые ≥ level, считая от step_on."""
        for k in range(step_on, len(arr)):
            if arr[k] >= level:
                return k - step_on
        return len(arr)

    filters  = ['SMA',  'EMA',  'MMA',  'DEMA', 'TEMA']
    outs     = [out_sma, out_ema, out_mma, out_dema, out_tema]
    colors   = [C_SMA, C_EMA, C_MMA, C_DEMA, C_TEMA]
    lag_50   = [reach_level(o, 20, 0.5) for o in outs]
    lag_90   = [reach_level(o, 20, 0.9) for o in outs]

    x_pos = np.arange(len(filters))
    w     = 0.38
    bars50 = ax2.bar(x_pos - w/2, lag_50, w, label='≥ 0.5 плато', color=colors, alpha=0.75)
    bars90 = ax2.bar(x_pos + w/2, lag_90, w, label='≥ 0.9 плато', color=colors, alpha=0.45,
                     edgecolor=colors, linewidth=1.2)
    for bar, v in zip(bars50, lag_50):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
                 ha='center', va='bottom', fontsize=8)
    for bar, v in zip(bars90, lag_90):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
                 ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(filters)
    ax2.set_ylabel('Задержка (отсчёты)')
    ax2.set_title('Скорость реакции на ступеньку\n(N=10, отсчётов до плато)')
    ax2.legend(fontsize=8)

    # ── [1,0] Частотный отклик — |H(ω)| ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    omega = np.linspace(0, np.pi, 512, dtype=np.float64)

    # SMA: H(z) = (1/N)*(1-z^{-N})/(1-z^{-1})
    def H_sma(w, N):
        z = np.exp(1j * w)
        return np.abs((1.0/N) * (1 - z**(-N)) / (1 - z**(-1) + 1e-20))

    # EMA/MMA: H(z) = alpha / (1 - (1-alpha)*z^{-1})
    def H_ema(w, alpha):
        z = np.exp(1j * w)
        return np.abs(alpha / (1 - (1-alpha) * z**(-1)))

    # DEMA: H = 2*H_ema - H_ema^2
    def H_dema(w, alpha):
        z  = np.exp(1j * w)
        He = alpha / (1 - (1-alpha) * z**(-1))
        return np.abs(2*He - He**2)

    # TEMA: H = 3*H_ema - 3*H_ema^2 + H_ema^3
    def H_tema(w, alpha):
        z  = np.exp(1j * w)
        He = alpha / (1 - (1-alpha) * z**(-1))
        return np.abs(3*He - 3*He**2 + He**3)

    alpha_ema = 2.0 / (N + 1)
    alpha_mma = 1.0 / N
    freq_norm = omega / np.pi   # 0..1 (Nyquist)

    ax3.plot(freq_norm, H_sma(omega, N),    color=C_SMA,  label='SMA')
    ax3.plot(freq_norm, H_ema(omega, alpha_ema),  color=C_EMA,  label='EMA')
    ax3.plot(freq_norm, H_ema(omega, alpha_mma),  color=C_MMA,  label='MMA')
    ax3.plot(freq_norm, H_dema(omega, alpha_ema), color=C_DEMA, label='DEMA')
    ax3.plot(freq_norm, H_tema(omega, alpha_ema), color=C_TEMA, label='TEMA')
    ax3.axhline(0.707, color='grey', ls=':', lw=0.9, alpha=0.7)
    ax3.text(0.96, 0.72, '−3дБ', ha='right', fontsize=8, color='grey')
    ax3.set_xlabel('Нормированная частота (×π рад/отсчёт)')
    ax3.set_ylabel('|H(ω)|')
    ax3.set_title('АЧХ фильтров (N=10)')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.35)
    ax3.legend(loc='upper right', fontsize=8)

    # ── [1,1] Импульсная характеристика EMA ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    n_imp = 40
    imp   = np.zeros(n_imp, dtype=np.float32);  imp[0] = 1.0
    h_ema = ema_ref(imp, N)
    t_imp = np.arange(n_imp)
    # Теоретическая: y[n] = (1-alpha)^n
    one_minus = np.float32(1.0) - np.float32(alpha_ema)
    h_theory  = one_minus ** t_imp

    ax4.stem(t_imp, h_ema, linefmt=C_EMA, markerfmt=f'o', basefmt='k-',
             label='EMA GPU/CPU')
    ax4.plot(t_imp, h_theory, color=C_EMA, ls='--', lw=1.2, alpha=0.7,
             label=f'(1−α)ⁿ, α={alpha_ema:.3f}')
    ax4.set_xlabel('Отсчёт n')
    ax4.set_ylabel('h[n]')
    ax4.set_title(f'Импульсная характеристика EMA (N={N})')
    ax4.legend(fontsize=8)
    ax4.set_xlim(-1, n_imp)

    # Аннотация: tau = -1/ln(1-alpha) ≈ N/2 для EMA
    tau = -1.0 / np.log(float(one_minus))
    ax4.axvline(tau, color='grey', ls=':', lw=0.9, alpha=0.7)
    ax4.text(tau + 0.5, float(h_theory[int(tau)]) + 0.04,
             f'τ≈{tau:.1f}', fontsize=8, color='grey')

    # ── [1,2] Сравнение EMA vs MMA: плавность при случайном сигнале ──────────
    ax5 = fig.add_subplot(gs[1, 2])
    rng  = np.random.default_rng(42)
    noisy = rng.standard_normal(points).astype(np.float32)
    t_n   = np.arange(points)

    out_ema_n  = ema_ref(noisy, N)
    out_mma_n  = mma_ref(noisy, N)

    # Показываем первые 80 отсчётов для наглядности
    sl = slice(0, 80)
    ax5.plot(t_n[sl], noisy[sl],     color=C_INPUT, lw=0.8, alpha=0.5, label='Шум (вход)')
    ax5.plot(t_n[sl], out_ema_n[sl], color=C_EMA, label=f'EMA α={alpha_ema:.3f}')
    ax5.plot(t_n[sl], out_mma_n[sl], color=C_MMA, label=f'MMA α={1/N:.3f} (Wilder)')
    ax5.set_xlabel('Отсчёт n')
    ax5.set_ylabel('Амплитуда')
    ax5.set_title(f'EMA vs MMA — белый шум (N={N})\nМеньше alpha → больше сглаживание')
    ax5.legend(fontsize=8)

    # Подпись параметров в правом нижнем углу
    info  = (f"EMA: α = 2/(N+1) = {alpha_ema:.3f}\n"
             f"MMA: α = 1/N     = {1/N:.3f}")
    ax5.text(0.97, 0.04, info, transform=ax5.transAxes, fontsize=8,
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    out_path = os.path.join(OUT_DIR, 'report_task20_moving_average.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    ✓ Сохранено: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Task_21: Фильтр Калмана
# ═══════════════════════════════════════════════════════════════════════════════

def plot_task21():
    print("  [Task_21] Строю графики фильтра Калмана...")
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Task_21 · KalmanFilterROCm · 1D Скалярный фильтр Калмана\n"
        "GPU ROCm (hiprtc) · Radeon 9070 gfx1201 · Тесты: 5/5 PASSED",
        fontsize=13, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 3, hspace=0.44, wspace=0.36)

    # ─── Параметры ────────────────────────────────────────────────────────────
    Q_noise = 0.01;  R_noise = 25.0          # подавление шума (const сигнал)
    Q_step  = 1.0;   R_step  = 25.0          # быстрый отклик на ступеньку
    Q_radar = 0.001; R_radar = 0.09          # LFM радар

    NPTS = 800

    # ── [0,0] Постоянный сигнал + шум → Kalman ───────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    rng    = np.random.default_rng(7)
    sigma  = np.sqrt(R_noise)                # = 5
    const  = 100.0
    noisy  = (const + rng.standard_normal(NPTS) * sigma).astype(np.float32)
    flt, P_arr, K_arr = kalman_ref_scalar(noisy, Q_noise, R_noise, 0.0, R_noise)

    t = np.arange(NPTS)
    ax0.plot(t, noisy, color=C_NOISY, lw=0.7, alpha=0.6, label=f'Зашумлённый (σ={sigma:.0f})')
    ax0.plot(t, flt,   color=C_KALMAN, lw=1.8, label='Фильтр Калмана')
    ax0.axhline(const, color='grey', ls='--', lw=0.9, alpha=0.5, label=f'Истина = {const:.0f}')
    ax0.set_xlabel('Отсчёт n')
    ax0.set_ylabel('Амплитуда')
    ax0.set_title(f'Постоянный сигнал + AWGN\nQ={Q_noise}, R={R_noise}, σ={sigma:.0f}')
    ax0.legend(fontsize=8)
    # Аннотация SNR improvement
    raw_rms = float(np.std(noisy[100:]))
    flt_rms = float(np.std(flt[100:] - const))
    ax0.text(0.97, 0.04,
             f'RMS(raw)  = {raw_rms:.2f}\nRMS(filt) = {flt_rms:.2f}\nУлучш. ×{raw_rms/flt_rms:.1f}',
             transform=ax0.transAxes, fontsize=8, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

    # ── [0,1] P(t) — сходимость дисперсии ошибки ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    # Устойчивое состояние K_ss, P_ss
    P_ss  = Q_noise/2 + np.sqrt((Q_noise/2)**2 + Q_noise*R_noise)
    K_ss  = P_ss / (P_ss + R_noise)
    ax1.semilogy(t, P_arr, color=C_COVAR, lw=1.8, label='P(t) — ошибка оценки')
    ax1.axhline(P_ss, color='grey', ls='--', lw=0.9,
                label=f'P_ss = {P_ss:.3f}')
    ax1.set_xlabel('Отсчёт n')
    ax1.set_ylabel('P(n) [log]')
    ax1.set_title(f'Сходимость дисперсии ошибки\nP_ss={P_ss:.3f}, K_ss={K_ss:.4f}')
    ax1.legend(fontsize=8)

    # Нанесём K(t) на вторую ось
    ax1b = ax1.twinx()
    ax1b.plot(t, K_arr, color=C_KALMAN, lw=1.2, alpha=0.7, ls=':', label='K(t) — коэфф. Калмана')
    ax1b.axhline(K_ss, color=C_KALMAN, ls='--', lw=0.7, alpha=0.5)
    ax1b.set_ylabel('K(n)', color=C_KALMAN)
    ax1b.tick_params(axis='y', labelcolor=C_KALMAN)
    ax1b.legend(fontsize=8, loc='center right')
    ax1b.set_ylim(0, 1.05)
    # Аннотация: где достигается steady-state
    conv_idx = next((i for i in range(len(P_arr)) if abs(P_arr[i] - P_ss) < P_ss * 0.05), NPTS)
    ax1.axvline(conv_idx, color='green', ls=':', lw=1.0, alpha=0.7)
    ax1.text(conv_idx + 5, P_arr[0] * 0.7, f'n≈{conv_idx}', fontsize=8, color='green')

    # ── [0,2] Ступенчатый отклик (разные Q) ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    step_sig = np.zeros(NPTS, dtype=np.float32)
    step_sig[300:] = 100.0

    Qs = [0.01, 0.1, 1.0, 10.0]
    cmap_q = plt.cm.viridis(np.linspace(0.2, 0.9, len(Qs)))

    ax2.plot(np.arange(NPTS), step_sig, color=C_INPUT, lw=1.0, ls='--', alpha=0.5, label='Ступенька')
    for Q_v, col in zip(Qs, cmap_q):
        flt_s, _, _ = kalman_ref_scalar(step_sig, Q_v, R_step, 0.0, R_step)
        K_ss_v = (Q_v/2 + np.sqrt((Q_v/2)**2 + Q_v*R_step)) / \
                 (Q_v/2 + np.sqrt((Q_v/2)**2 + Q_v*R_step) + R_step)
        ax2.plot(np.arange(NPTS), flt_s, color=col, lw=1.6,
                 label=f'Q={Q_v} (K_ss={K_ss_v:.3f})')

    ax2.axvline(300, color='grey', ls=':', lw=0.9, alpha=0.5)
    ax2.set_xlabel('Отсчёт n')
    ax2.set_ylabel('Амплитуда')
    ax2.set_title(f'Ступенчатый отклик при разных Q\n(R={R_step}, x0=0)')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_xlim(250, 650)

    # ── [1,0..1] LFM радар: 5 антенн — beat tone + AWGN ──────────────────────
    ax3 = fig.add_subplot(gs[1, :2])

    fs        = 10e6
    fdev      = 2e6
    N_fft     = 4096
    Ti        = N_fft / fs
    mu        = fdev / Ti
    noise_sig = 0.30
    c         = 3e8

    tau_us  = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
    tau     = tau_us * 1e-6
    f_beat  = mu * tau
    n_ant   = 5
    colors_ant = plt.cm.tab10(np.linspace(0, 0.5, n_ant))

    rng = np.random.default_rng(42)
    signals  = np.zeros((n_ant, N_fft), dtype=np.float32)
    filtered = np.zeros((n_ant, N_fft), dtype=np.float32)

    for a in range(n_ant):
        omega    = 2.0 * np.pi * f_beat[a] / fs
        tone     = np.cos(omega * np.arange(N_fft, dtype=np.float32))
        noise_r  = rng.standard_normal(N_fft).astype(np.float32) * noise_sig
        signals[a]  = tone + noise_r
        filtered[a], _, _ = kalman_ref_scalar(signals[a], Q_radar, R_radar, 0.0, R_radar)

    # Показываем FFT до и после фильтрации для антенны 0 и 2
    bin_hz = fs / N_fft
    freq_bins = np.arange(N_fft // 2) * bin_hz / 1000.0  # кГц

    for a, col in zip([0, 2, 4], [colors_ant[0], colors_ant[2], colors_ant[4]]):
        fft_raw = np.abs(np.fft.fft(signals[a]))[:N_fft//2]
        fft_flt = np.abs(np.fft.fft(filtered[a]))[:N_fft//2]
        peak_expected = int(round(f_beat[a] / bin_hz))
        label_raw = f'Ант.{a} (raw)   f={f_beat[a]/1e3:.1f} кГц'
        label_flt = f'Ант.{a} (Kalman)'
        w = max(1, peak_expected - 50), min(len(freq_bins), peak_expected + 50)
        ax3.plot(freq_bins[w[0]:w[1]], fft_raw[w[0]:w[1]],
                 color=col, lw=0.8, alpha=0.35)
        ax3.plot(freq_bins[w[0]:w[1]], fft_flt[w[0]:w[1]],
                 color=col, lw=2.0, alpha=0.9, label=label_flt)
        # Метка бина
        ax3.axvline(f_beat[a]/1e3, color=col, ls=':', lw=0.9, alpha=0.5)

    ax3.set_xlabel('Частота биений (кГц)')
    ax3.set_ylabel('|FFT|')
    ax3.set_title(
        f'LFM радар — FFT до/после Калмана · 5 антенн\n'
        f'fs={fs/1e6:.0f} МГц, N={N_fft}, σ={noise_sig}, '
        f'Q={Q_radar}, R={R_radar}\n'
        f'Зоны: ант.0 (bin#100), ант.2 (bin#300), ант.4 (bin#500)'
    )
    ax3.legend(fontsize=8, loc='upper right')

    # ── [1,2] SNR improvement bar chart ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])

    snr_raw = []
    snr_flt = []
    for a in range(n_ant):
        fft_raw = np.abs(np.fft.fft(signals[a]))
        fft_flt = np.abs(np.fft.fft(filtered[a]))
        peak_b  = int(round(f_beat[a] / bin_hz))
        # Пиковая мощность / шумовая мощность (вне ±5 бинов от пика)
        mask    = np.ones(N_fft, dtype=bool)
        mask[max(0, peak_b-5):peak_b+6] = False
        noise_r  = np.sqrt(np.mean(fft_raw[mask]**2))
        noise_f  = np.sqrt(np.mean(fft_flt[mask]**2))
        snr_raw.append(20 * np.log10(fft_raw[peak_b] / noise_r))
        snr_flt.append(20 * np.log10(fft_flt[peak_b] / noise_f))

    x_pos    = np.arange(n_ant)
    snr_raw  = np.array(snr_raw)
    snr_flt  = np.array(snr_flt)
    delta_snr = snr_flt - snr_raw

    bars_r = ax4.bar(x_pos - 0.22, snr_raw, 0.38, label='Без Калмана',
                     color=C_NOISY, edgecolor='grey', lw=0.8)
    bars_f = ax4.bar(x_pos + 0.22, snr_flt, 0.38, label='С Калманом',
                     color=C_KALMAN, alpha=0.85)
    for i, (r, f, d) in enumerate(zip(snr_raw, snr_flt, delta_snr)):
        ax4.text(i + 0.22, f + 0.3, f'+{d:.1f}dB', ha='center', va='bottom',
                 fontsize=8, color=C_KALMAN, fontweight='bold')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Ант.{a}\n{f_beat[a]/1e3:.0f}кГц' for a in range(n_ant)],
                        fontsize=8)
    ax4.set_ylabel('SNR (дБ)')
    ax4.set_title('Улучшение SNR после Калмана\n5 антенн LFM радара')
    ax4.legend(fontsize=8)
    avg_delta = np.mean(delta_snr)
    ax4.text(0.97, 0.04, f'Среднее улучшение:\n+{avg_delta:.1f} дБ',
             transform=ax4.transAxes, fontsize=9, ha='right', va='bottom',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

    out_path = os.path.join(OUT_DIR, 'report_task21_kalman_filter.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    ✓ Сохранено: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Task_22: Kaufman KAMA
# ═══════════════════════════════════════════════════════════════════════════════

def plot_task22():
    print("  [Task_22] Строю графики KAMA (Kaufman)...")
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Task_22 · KaufmanFilterROCm (KAMA) · Адаптивная скользящая средняя Кауфмана\n"
        "GPU ROCm (hiprtc) · Radeon 9070 gfx1201 · Тесты: 5/5 PASSED",
        fontsize=13, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 3, hspace=0.44, wspace=0.36)

    N    = 10
    FAST = 2
    SLOW = 30
    fsc  = 2.0 / (FAST + 1)   # fast SC ≈ 0.667
    ssc  = 2.0 / (SLOW + 1)   # slow SC ≈ 0.065

    # ── [0,0] Ступенчатый отклик KAMA vs EMA ─────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    sig_step = np.zeros(120, dtype=np.float32)
    sig_step[20:70] = 1.0
    t_step = np.arange(120)

    out_kama_s, er_s, sc_s = kaufman_ref(sig_step, N, FAST, SLOW)
    out_ema_s  = ema_ref(sig_step, N)
    out_sma_s  = sma_ref(sig_step, N)

    ax0.step(t_step, sig_step, where='post', color=C_INPUT, lw=1.2, ls='--',
             alpha=0.5, label='Вход')
    ax0.plot(t_step, out_sma_s,    color=C_SMA,  lw=1.4, label=f'SMA (N={N})')
    ax0.plot(t_step, out_ema_s,    color=C_EMA,  lw=1.6, label=f'EMA (N={N})')
    ax0.plot(t_step, out_kama_s,   color=C_KAMA, lw=2.2, label=f'KAMA (N={N}, f={FAST}, s={SLOW})')
    ax0.set_title(f'Ступенчатый отклик: KAMA vs EMA vs SMA')
    ax0.set_xlabel('Отсчёт n')
    ax0.set_ylabel('Амплитуда')
    ax0.legend(fontsize=8)
    ax0.axvline(20, color='gray', ls=':', alpha=0.4, lw=1.0)
    ax0.axvline(70, color='gray', ls=':', alpha=0.4, lw=1.0)
    ax0.text(20, -0.12, 'ON',  ha='center', fontsize=8, color='gray')
    ax0.text(70, -0.12, 'OFF', ha='center', fontsize=8, color='gray')
    ax0.set_ylim(-0.15, 1.22)

    # Аннотация SC при скачке
    ax0.text(0.97, 0.45,
             f'При ER=1:\nSC = fast_sc² = {fsc**2:.3f}\n(быстрый EMA)\n\n'
             f'При ER=0:\nSC = slow_sc² = {ssc**2:.4f}\n(почти стоит)',
             transform=ax0.transAxes, fontsize=8, ha='right', va='center',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightcyan', alpha=0.9))

    # ── [0,1] Тренд → Шум → Ступенька: адаптивность ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    rng  = np.random.default_rng(42)
    L    = 300
    data_tnt = np.zeros(L, dtype=np.float32)

    # Phase 1 [0..99]:   тренд 0→1
    data_tnt[:100]   = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    # Phase 2 [100..199]: белый шум вокруг 1.0
    data_tnt[100:200] = 1.0 + rng.standard_normal(100).astype(np.float32) * 0.25
    # Phase 3 [200..299]: ступенька до 0.0 + малый шум
    data_tnt[200:]   = rng.standard_normal(100).astype(np.float32) * 0.05

    t_tnt = np.arange(L)
    out_kama_t, er_t, sc_t = kaufman_ref(data_tnt, N, FAST, SLOW)
    out_ema_t               = ema_ref(data_tnt, N)

    ax1.fill_betweenx([-0.5, 1.6], 0,   100, alpha=0.07, color='green',  label='_nolegend_')
    ax1.fill_betweenx([-0.5, 1.6], 100, 200, alpha=0.07, color='orange', label='_nolegend_')
    ax1.fill_betweenx([-0.5, 1.6], 200, 300, alpha=0.07, color='red',    label='_nolegend_')
    ax1.text(50,  1.52, 'Тренд↑',  ha='center', fontsize=9, color='darkgreen')
    ax1.text(150, 1.52, 'Шум',     ha='center', fontsize=9, color='darkorange')
    ax1.text(250, 1.52, 'Ступ.↓',  ha='center', fontsize=9, color='darkred')

    ax1.plot(t_tnt, data_tnt,    color=C_NOISY, lw=0.7, alpha=0.45, label='Вход')
    ax1.plot(t_tnt, out_ema_t,   color=C_EMA,   lw=1.5, alpha=0.75, label=f'EMA (N={N})')
    ax1.plot(t_tnt, out_kama_t,  color=C_KAMA,  lw=2.2, label=f'KAMA (N={N})')
    ax1.set_title('Адаптивность KAMA: тренд → шум → ступенька')
    ax1.set_xlabel('Отсчёт n')
    ax1.set_ylabel('Амплитуда')
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.5, 1.65)

    # ── [0,2] Efficiency Ratio и SC(t) ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t_tnt, er_t, color=C_KAMA, lw=1.6, label='ER(t) — Efficiency Ratio')
    ax2.set_ylabel('ER(t)', color=C_KAMA)
    ax2.tick_params(axis='y', labelcolor=C_KAMA)
    ax2.set_ylim(-0.05, 1.15)

    ax2b = ax2.twinx()
    ax2b.plot(t_tnt, sc_t, color=C_EMA, lw=1.4, ls='--', alpha=0.85,
              label='SC(t) = сглаживающая константа')
    ax2b.axhline(fsc**2, color=C_EMA, ls=':', lw=0.8, alpha=0.6)
    ax2b.axhline(ssc**2, color='grey', ls=':', lw=0.8, alpha=0.6)
    ax2b.text(295, fsc**2 + 0.005, f'fast²={fsc**2:.3f}', fontsize=7,
              ha='right', color=C_EMA)
    ax2b.text(295, ssc**2 + 0.001, f'slow²={ssc**2:.4f}', fontsize=7,
              ha='right', color='grey')
    ax2b.set_ylabel('SC(t)', color=C_EMA)
    ax2b.tick_params(axis='y', labelcolor=C_EMA)
    ax2b.set_ylim(-0.02, fsc**2 * 1.25)

    ax2.fill_betweenx([-0.05, 1.15], 0,   100, alpha=0.07, color='green')
    ax2.fill_betweenx([-0.05, 1.15], 100, 200, alpha=0.07, color='orange')
    ax2.fill_betweenx([-0.05, 1.15], 200, 300, alpha=0.07, color='red')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax2.set_xlabel('Отсчёт n')
    ax2.set_title('Efficiency Ratio и SC(t)\n(адаптивный коэффициент сглаживания)')

    # ── [1,0] Сравнение шумоподавления: KAMA vs EMA vs SMA ───────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    rng2  = np.random.default_rng(99)
    noise_only = rng2.standard_normal(500).astype(np.float32)
    t_nse      = np.arange(500)

    out_kama_n, _, _ = kaufman_ref(noise_only, N, FAST, SLOW)
    out_ema_n         = ema_ref(noise_only, N)
    out_sma_n         = sma_ref(noise_only, N)

    skip = N * 3
    # Показываем первые 150 отсчётов
    s = slice(0, 150)
    ax3.plot(t_nse[s], noise_only[s],   color=C_NOISY, lw=0.8, alpha=0.4, label='Шум (вход)')
    ax3.plot(t_nse[s], out_sma_n[s],    color=C_SMA,   lw=1.4, label='SMA')
    ax3.plot(t_nse[s], out_ema_n[s],    color=C_EMA,   lw=1.4, label='EMA')
    ax3.plot(t_nse[s], out_kama_n[s],   color=C_KAMA,  lw=2.0, label='KAMA')
    ax3.set_xlabel('Отсчёт n')
    ax3.set_ylabel('Амплитуда')
    ax3.set_title('Подавление белого шума: KAMA vs EMA vs SMA\n(чем меньше колебания — тем лучше)')
    ax3.legend(fontsize=8)

    # std после разогрева
    std_in   = float(np.std(noise_only[skip:]))
    std_sma  = float(np.std(out_sma_n[skip:]))
    std_ema  = float(np.std(out_ema_n[skip:]))
    std_kama = float(np.std(out_kama_n[skip:]))
    ax3.text(0.97, 0.04,
             f'std вход:  {std_in:.3f}\n'
             f'std SMA:   {std_sma:.3f} (×{std_in/std_sma:.1f}↓)\n'
             f'std EMA:   {std_ema:.3f} (×{std_in/std_ema:.1f}↓)\n'
             f'std KAMA:  {std_kama:.3f} (×{std_in/std_kama:.1f}↓)',
             transform=ax3.transAxes, fontsize=8, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

    # ── [1,1] Скорость реакции KAMA при разных (fast, slow) ──────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sig_s2 = np.zeros(120, dtype=np.float32)
    sig_s2[20:70] = 1.0
    t_s2   = np.arange(120)

    configs = [
        (10, 2,  5,  '#1a535c', 'fast=2, slow=5  (агрессивный)'),
        (10, 2,  30, C_KAMA,    'fast=2, slow=30 (стандарт)'),
        (10, 2,  60, '#f4a261', 'fast=2, slow=60 (осторожный)'),
        (10, 5,  30, '#264653', 'fast=5, slow=30 (медленный)'),
    ]

    ax4.step(t_s2, sig_s2, where='post', color=C_INPUT, lw=1.0, ls='--', alpha=0.4, label='Вход')
    for N_c, fast_c, slow_c, col, lbl in configs:
        out_c, _, _ = kaufman_ref(sig_s2, N_c, fast_c, slow_c)
        ax4.plot(t_s2, out_c, color=col, lw=1.8, label=lbl)

    ax4.set_xlabel('Отсчёт n')
    ax4.set_ylabel('Амплитуда')
    ax4.set_title('Влияние параметров (fast, slow)\nна ступенчатый отклик (N=10)')
    ax4.legend(fontsize=7.5)
    ax4.set_ylim(-0.1, 1.2)
    ax4.axvline(20, color='gray', ls=':', alpha=0.4, lw=0.9)
    ax4.axvline(70, color='gray', ls=':', alpha=0.4, lw=0.9)

    # ── [1,2] KAMA — математика: SC vs ER (параболическая зависимость) ────────
    ax5 = fig.add_subplot(gs[1, 2])
    er_vals = np.linspace(0, 1, 200)

    for fast_v, slow_v, col, lbl in [
        (2, 30, C_KAMA,    'fast=2, slow=30 (стандарт)'),
        (2,  5, '#1a535c', 'fast=2, slow=5'),
        (5, 30, '#264653', 'fast=5, slow=30'),
        (3, 20, C_EMA,     'fast=3, slow=20'),
    ]:
        fsc_v = 2.0 / (fast_v + 1)
        ssc_v = 2.0 / (slow_v + 1)
        sc_v  = (er_vals * (fsc_v - ssc_v) + ssc_v) ** 2
        ax5.plot(er_vals, sc_v, color=col, lw=1.8, label=lbl)

    # Аннотации
    ax5.axvline(0.0, color='red',   ls=':', lw=0.8, alpha=0.5)
    ax5.axvline(1.0, color='green', ls=':', lw=0.8, alpha=0.5)
    ax5.text(0.02, ax5.get_ylim()[1] * 0.95 if ax5.get_ylim()[1] > 0 else 0.45,
             'ER=0\n(шум)', fontsize=8, color='red', va='top')
    ax5.text(0.98, 0.02,
             'ER=1\n(тренд)', fontsize=8, color='green', ha='right')

    ax5.set_xlabel('Efficiency Ratio (ER)')
    ax5.set_ylabel('Smoothing Constant SC = (ER·(f−s)+s)²')
    ax5.set_title('SC(ER) — адаптивный коэффициент\n(квадратичная зависимость от ER)')
    ax5.legend(fontsize=8)
    ax5.set_xlim(-0.02, 1.05)

    out_path = os.path.join(OUT_DIR, 'report_task22_kaufman_kama.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    ✓ Сохранено: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  Графики для отчёта: Task_20 / Task_21 / Task_22")
    print("=" * 60)
    print(f"  Выходная папка: {OUT_DIR}")
    print()

    plot_task20()
    plot_task21()
    plot_task22()

    print()
    print("=" * 60)
    print("  Все графики сохранены! ✓")
    print()
    print("  Файлы:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith('report_task'):
            path = os.path.join(OUT_DIR, f)
            size = os.path.getsize(path)
            print(f"    {f}  ({size // 1024} кБ)")
    print("=" * 60)
