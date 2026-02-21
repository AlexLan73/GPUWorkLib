
import numpy as np

# The issue: at 12 MHz sampling, we're dealing with BASEBAND signal after 
# demodulation. The carrier frequencies f0=9 GHz are already removed.
# Let me redo with proper baseband LFM model.

fs = 12e6          # sampling frequency (after downconversion)
BW = 5e6           # baseband bandwidth (fits within fs/2)
T_pulse = 100e-6   # pulse duration
chirp_rate = BW / T_pulse  # 5e10 Hz/s = 50 GHz/s... 
# Actually for baseband: BW should be << fs/2
# Let's use realistic baseband parameters
BW = 4e6           # 4 MHz baseband bandwidth  
chirp_rate = BW / T_pulse  # 4e10 Hz/s

N = int(fs * T_pulse)  # 1200 samples
Ts = 1.0 / fs

# True fractional delay
true_delay_ns = 23.7  
true_delay_s = true_delay_ns * 1e-9
true_delay_samples = true_delay_s * fs  # 0.2844 samples

print(f"=== BASEBAND LFM Parameters ===")
print(f"  fs = {fs/1e6:.0f} MHz, BW = {BW/1e6:.0f} MHz, T = {T_pulse*1e6:.0f} us")
print(f"  N = {N} samples, chirp_rate = {chirp_rate:.3e} Hz/s")
print(f"  True delay = {true_delay_ns} ns = {true_delay_samples:.6f} samples")
print(f"  Time-bandwidth product = {BW * T_pulse:.0f}")
print()

n_arr = np.arange(N)
t_ref = n_arr / fs

# Baseband LFM: f goes from -BW/2 to +BW/2 (centered)
f_start = -BW/2
phase_ref = 2 * np.pi * (f_start * t_ref + 0.5 * chirp_rate * t_ref**2)
s_ref = np.exp(1j * phase_ref)

# Delayed signal (analytic exact computation)
t_del = t_ref - true_delay_s
phase_del = 2 * np.pi * (f_start * t_del + 0.5 * chirp_rate * t_del**2)
s_delayed_exact = np.exp(1j * phase_del)

# Add noise
SNR_dB = 20
noise_power = 10**(-SNR_dB/10)
rng = np.random.default_rng(42)
noise = np.sqrt(noise_power/2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
s_rx = s_delayed_exact + noise

# =============================================
# CRB
# =============================================
beta_rms_sq = (BW**2) / 12
SNR_lin = 10**(SNR_dB/10)
CRB_var = 1.0 / (8 * np.pi**2 * beta_rms_sq * N * SNR_lin)
CRB_std = np.sqrt(CRB_var)
print(f"CRB: std(tau) = {CRB_std*1e9:.6f} ns = {CRB_std*1e12:.3f} ps")
print()

# =============================================
# METHOD 1: Spectral (Analytic) — direct phase computation
# Мы знаем модель s(t) = exp(j*phi(t)), считаем phi(t-tau) и находим tau
# =============================================
print("=" * 70)
print("МЕТОД 1: Аналитический спектральный (прямая фаза)")
print("  s_model(t,tau) = exp(j*2*pi*(f0*(t-tau) + 0.5*k*(t-tau)^2))")
print("  tau считается напрямую из формулы фазы")
print("=" * 70)

# Heterodyne: beat = s_rx * conj(s_ref)
# phase_beat = phase_del - phase_ref = 2*pi*(-f_start*tau - chirp_rate*t*tau + 0.5*chirp_rate*tau^2 + f_start*tau... )
# Simplify: beat phase = 2*pi * chirp_rate * tau * t + const
# This is a single tone at frequency f_beat = chirp_rate * tau

beat = s_rx * np.conj(s_ref)
beat_phase = np.unwrap(np.angle(beat))

# Linear fit: beat_phase = 2*pi * chirp_rate * tau * t + phi0
A1 = np.vstack([t_ref, np.ones(N)]).T
coeffs1 = np.linalg.lstsq(A1, beat_phase, rcond=None)[0]
slope1 = coeffs1[0]  # = 2*pi * chirp_rate * tau

tau_method1 = slope1 / (2 * np.pi * chirp_rate)
err1 = abs(tau_method1 - true_delay_s)
print(f"  slope = {slope1:.6f} rad/s")
print(f"  tau = {tau_method1*1e9:.6f} ns")
print(f"  Error = {err1*1e12:.3f} ps ({err1/CRB_std:.2f}x CRB)")
print()

# =============================================
# METHOD 2: Cross-correlation + parabolic interpolation
# =============================================
print("=" * 70)
print("МЕТОД 2: Кросс-корреляция + параболическая интерполяция")
print("=" * 70)

xcorr = np.correlate(s_rx, s_ref, mode='full')
lags = np.arange(-N+1, N)
peak_idx = np.argmax(np.abs(xcorr))

y_m1 = np.abs(xcorr[peak_idx - 1])
y_0  = np.abs(xcorr[peak_idx])
y_p1 = np.abs(xcorr[peak_idx + 1])
delta_p = 0.5 * (y_m1 - y_p1) / (y_m1 - 2*y_0 + y_p1)
tau_method2_samples = lags[peak_idx] + delta_p
tau_method2 = tau_method2_samples / fs
err2 = abs(tau_method2 - true_delay_s)
print(f"  Peak lag = {lags[peak_idx]}, delta = {delta_p:.6f}")
print(f"  tau = {tau_method2*1e9:.6f} ns")
print(f"  Error = {err2*1e12:.3f} ps ({err2/CRB_std:.2f}x CRB)")
print()

# =============================================
# METHOD 3: Dechirp + FFT with zero-padding
# =============================================
print("=" * 70)
print("МЕТОД 3: Гетеродинирование (dechirp) + FFT с zero-padding")
print("=" * 70)

ZP_factors = [1, 8, 64, 256, 1024]
for zp in ZP_factors:
    N_fft = N * zp
    Beat_fft = np.fft.fft(beat, N_fft)
    freqs_fft = np.fft.fftfreq(N_fft, Ts)
    
    # Look in positive and negative frequencies
    mag = np.abs(Beat_fft)
    peak_bin = np.argmax(mag)
    f_beat_est = freqs_fft[peak_bin]
    tau_est = f_beat_est / chirp_rate
    err = abs(tau_est - true_delay_s)
    print(f"  ZP={zp:4d}x: f_beat={f_beat_est:12.3f} Hz, tau={tau_est*1e9:12.6f} ns, err={err*1e12:.1f} ps ({err/CRB_std:.1f}x CRB)")
print()

# =============================================
# METHOD 4: Dechirp + FFT + parabolic peak refinement
# =============================================
print("=" * 70)
print("МЕТОД 4: Dechirp + FFT + параболическая уточнение пика")
print("=" * 70)

N_fft = N * 8  # modest zero-padding
Beat_fft = np.fft.fft(beat, N_fft)
freqs_fft = np.fft.fftfreq(N_fft, Ts)
mag = np.abs(Beat_fft)
pk = np.argmax(mag)

# Parabolic refinement on FFT magnitude
ym1 = mag[pk-1]; y0 = mag[pk]; yp1 = mag[(pk+1) % N_fft]
dp = 0.5 * (ym1 - yp1) / (ym1 - 2*y0 + yp1)
f_refined = freqs_fft[pk] + dp * (fs / N_fft)
tau_method4 = f_refined / chirp_rate
err4 = abs(tau_method4 - true_delay_s)
print(f"  FFT peak bin = {pk}, delta = {dp:.6f}")
print(f"  f_beat refined = {f_refined:.6f} Hz")
print(f"  tau = {tau_method4*1e9:.6f} ns")
print(f"  Error = {err4*1e12:.3f} ps ({err4/CRB_std:.2f}x CRB)")
print()

# =============================================
# METHOD 5: ML/Newton model fitting
# =============================================
print("=" * 70)
print("МЕТОД 5: ML/Newton (подгонка модели МП-оценка)")
print("=" * 70)

# Start from rough estimate (method 3 with some ZP)
N_fft_init = N * 64
Beat_fft_init = np.fft.fft(beat, N_fft_init)
freqs_init = np.fft.fftfreq(N_fft_init, Ts)
pk_init = np.argmax(np.abs(Beat_fft_init))
tau_init = freqs_init[pk_init] / chirp_rate

tau_ml = tau_init
for it in range(50):
    t_m = t_ref - tau_ml
    ph_m = 2 * np.pi * (f_start * t_m + 0.5 * chirp_rate * t_m**2)
    s_m = np.exp(1j * ph_m)
    
    res = s_rx - s_m
    
    # ds/dtau = -j*2*pi*(f_start + chirp_rate*(t-tau)) * s_model
    inst_f = f_start + chirp_rate * t_m
    ds_dtau = -1j * 2 * np.pi * inst_f * s_m  # derivative
    
    # Gauss-Newton: dtau = Re(res . conj(ds_dtau)) / |ds_dtau|^2
    num = np.real(np.sum(res * np.conj(ds_dtau)))
    den = np.real(np.sum(np.abs(ds_dtau)**2))
    dtau = -num / den
    
    tau_ml += dtau
    if abs(dtau) < 1e-16:
        break

err5 = abs(tau_ml - true_delay_s)
print(f"  Converged in {it+1} iterations")
print(f"  tau = {tau_ml*1e9:.6f} ns")
print(f"  Error = {err5*1e12:.3f} ps ({err5/CRB_std:.2f}x CRB)")
print()

# =============================================
# METHOD 6: Phase difference at known instant frequency (single-point)
# =============================================
print("=" * 70)
print("МЕТОД 6: Фазовая разность (мгновенная оценка)")
print("=" * 70)
# Average phase difference weighted by instantaneous frequency
inst_freq_ref = f_start + chirp_rate * t_ref
phase_diff = np.angle(s_rx * np.conj(s_ref))  # wrapped
# Weighted average: tau ≈ phase_diff / (2*pi*inst_freq)
# But need to unwrap properly. Use median of tau estimates from each sample
tau_per_sample = np.unwrap(phase_diff) / (2 * np.pi * inst_freq_ref)
# Weighted by |inst_freq| to reduce noise at low frequencies
weights = inst_freq_ref**2
tau_method6 = np.sum(weights * tau_per_sample) / np.sum(weights)
err6 = abs(tau_method6 - true_delay_s)
print(f"  tau = {tau_method6*1e9:.6f} ns")
print(f"  Error = {err6*1e12:.3f} ps ({err6/CRB_std:.2f}x CRB)")
print()

# =============================================
# FINAL SUMMARY
# =============================================
print("=" * 70)
print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
print("=" * 70)
print(f"  Истинная задержка: {true_delay_ns} ns ({true_delay_samples:.4f} отсчётов)")
print(f"  СКО по КРБ: {CRB_std*1e12:.1f} ps")
print()
results = [
    ("1. MНК фазового наклона (дехирп)", err1),
    ("2. Кросс-корреляция + парабола", err2),
    ("3. Дехирп + FFT ZP (1024x)", abs(freqs_fft[np.argmax(np.abs(np.fft.fft(beat, N*1024)))]/chirp_rate * (N*1024 > 0) - true_delay_s) if False else None),
    ("4. Дехирп + FFT(8x) + парабола", err4),
    ("5. ML/Newton (подгонка модели)", err5),
    ("6. Взвешенная фаз. разность", err6),
]

# Redo method 3 best
N_fft_best = N * 1024
Beat_fft_best = np.fft.fft(beat, N_fft_best)
freqs_best = np.fft.fftfreq(N_fft_best, Ts)
pk_best = np.argmax(np.abs(Beat_fft_best))
tau_best = freqs_best[pk_best] / chirp_rate
err3_best = abs(tau_best - true_delay_s)

results_final = [
    ("1. МНК фазового наклона (beat)", err1, "O(N)", "Нет смещения"),
    ("2. Кросс-корр + парабола", err2, "O(N²)", "Смещение!"),
    ("3. Дехирп+FFT ZP(1024x)", err3_best, "O(NlogN·ZP)", "Нет смещения"),
    ("4. Дехирп+FFT(8x)+парабола", err4, "O(NlogN)", "Малое смещ."),
    ("5. ML/Newton (модель)", err5, "O(N·iter)", "Нет смещения"),
    ("6. Взвеш. фазовая разность", err6, "O(N)", "Малое смещ."),
]

print(f"{'Метод':<35} {'Ошибка ps':>10} {'x CRB':>8} {'Сложн.':>12} {'Bias':>15}")
print("-" * 82)
for name, err, compl, bias in results_final:
    if err is not None:
        print(f"{name:<35} {err*1e12:>10.1f} {err/CRB_std:>8.1f} {compl:>12} {bias:>15}")
print(f"{'КРБ (нижняя граница)':<35} {CRB_std*1e12:>10.1f} {'1.0':>8} {'—':>12} {'—':>15}")
