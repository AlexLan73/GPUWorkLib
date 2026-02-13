
import numpy as np

# The ML/Newton is diverging because of the wideband phase wrapping issue.
# Let me fix it: initialize better and use phase-only cost function.

# Also the cross-correlation is failing because the delay is sub-sample (0.045 samples)
# and parabolic fit doesn't work well for such small delays with wideband chirp.

# Let me fix Method 5 and add the CORRECT formulation.
# Also add Method 6: phase of cross-spectrum (frequency domain).

fs = 12e6; BW_rf = 500e6; T_pulse = 20e-6
chirp_rate = BW_rf / T_pulse  # 2.5e13
N = int(fs * T_pulse)  # 240
Ts = 1/fs

true_delay_ns = 3.75
true_delay_s = true_delay_ns * 1e-9

rng = np.random.default_rng(42)
n_arr = np.arange(N)
t = n_arr / fs

f0_bb = 0
s_ref = np.exp(1j * 2 * np.pi * (0.5 * chirp_rate * t**2))

t_d = t - true_delay_s
s_del = np.exp(1j * 2 * np.pi * (0.5 * chirp_rate * t_d**2))

beta_rms = BW_rf / np.sqrt(12)

SNR_dB_values = [10, 20, 30, 40]
n_trials = 1000

all_results = {}

for SNR_dB in SNR_dB_values:
    SNR_lin = 10**(SNR_dB/10)
    CRB_var = 1.0 / (8 * np.pi**2 * beta_rms**2 * N * SNR_lin)
    CRB_std_ps = np.sqrt(CRB_var) * 1e12
    
    errs = {name: [] for name in [
        'phase_slope', 'fft_zp256', 'fft8_parab', 'ml_phase', 
        'cross_spec_phase', 'hilbert_xcorr'
    ]}
    
    for trial in range(n_trials):
        noise_power = 10**(-SNR_dB/10)
        noise = np.sqrt(noise_power/2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        s_rx = s_del + noise
        beat = s_rx * np.conj(s_ref)
        
        # === Method 1: Phase slope of beat signal (МНК) ===
        bp = np.unwrap(np.angle(beat))
        c1 = np.polyfit(t, bp, 1)
        tau1 = -c1[0] / (2*np.pi*chirp_rate)
        errs['phase_slope'].append(tau1 - true_delay_s)
        
        # === Method 2: FFT ZP 256x ===
        ZP = 256
        Nf = N*ZP
        Bf = np.fft.fft(beat, Nf)
        ff = np.fft.fftfreq(Nf, Ts)
        pk = np.argmax(np.abs(Bf))
        tau2 = -ff[pk] / chirp_rate
        errs['fft_zp256'].append(tau2 - true_delay_s)
        
        # === Method 3: FFT(8x) + parabolic ===
        Nf3 = N*8
        Bf3 = np.fft.fft(beat, Nf3)
        ff3 = np.fft.fftfreq(Nf3, Ts)
        m3 = np.abs(Bf3)
        pk3 = np.argmax(m3)
        ym = m3[(pk3-1)%Nf3]; y0 = m3[pk3]; yp = m3[(pk3+1)%Nf3]
        den3 = ym - 2*y0 + yp
        dp3 = 0.5*(ym-yp)/den3 if abs(den3) > 1e-15 else 0
        fr3 = ff3[pk3] + dp3*(fs/Nf3)
        tau3 = -fr3 / chirp_rate
        errs['fft8_parab'].append(tau3 - true_delay_s)
        
        # === Method 4: ML phase-only Newton (fixed) ===
        # Initialize from phase slope
        tau_ml = tau1
        for it in range(30):
            t_m = t - tau_ml
            ph_m = 2*np.pi*(0.5*chirp_rate*t_m**2)
            s_m = np.exp(1j*ph_m)
            
            # Phase-only residual: angle(s_rx * conj(s_m))
            phase_res = np.angle(s_rx * np.conj(s_m))
            
            # Derivative of phase w.r.t. tau: d/dtau[angle(s_rx * conj(s_m))]
            # ≈ 2*pi*(chirp_rate * t_m) (instantaneous frequency of model)
            # Gauss-Newton on phase: dtau = sum(phase_res * dphi_dtau) / sum(dphi_dtau^2)
            dphi_dtau = 2*np.pi*chirp_rate*t_m
            
            num = np.sum(phase_res * dphi_dtau)
            den = np.sum(dphi_dtau**2)
            dtau = num / den
            tau_ml += dtau
            if abs(dtau) < 1e-16:
                break
        errs['ml_phase'].append(tau_ml - true_delay_s)
        
        # === Method 5: Cross-spectral phase slope ===
        S_rx = np.fft.fft(s_rx)
        S_ref = np.fft.fft(s_ref)
        CS = S_rx * np.conj(S_ref)
        phase_cs = np.unwrap(np.angle(CS))
        freqs_cs = np.fft.fftfreq(N, Ts)
        
        # Weighted linear fit (weight by |CS|)
        w = np.abs(CS)
        valid = w > 0.1 * np.max(w)
        if np.sum(valid) > 10:
            A5 = np.vstack([2*np.pi*freqs_cs[valid], np.ones(np.sum(valid))]).T
            W5 = np.diag(w[valid])
            c5 = np.linalg.lstsq(W5 @ A5, W5 @ phase_cs[valid], rcond=None)[0]
            tau5 = -c5[0]
        else:
            tau5 = tau1  # fallback
        errs['cross_spec_phase'].append(tau5 - true_delay_s)
        
        # === Method 6: Hilbert (analytic) correlation envelope peak ===
        # Compute cross-correlation of analytic signals
        from scipy.signal import hilbert as sig_hilbert
        xcorr_full = np.fft.ifft(S_rx * np.conj(S_ref))
        # The envelope of cross-correlation
        env = np.abs(xcorr_full)
        pk6 = np.argmax(env)
        if pk6 > N//2:
            pk6_lag = pk6 - N
        else:
            pk6_lag = pk6
        # Parabolic on envelope
        ym6 = env[(pk6-1)%N]; y06 = env[pk6]; yp6 = env[(pk6+1)%N]
        den6 = ym6 - 2*y06 + yp6
        dp6 = 0.5*(ym6-yp6)/den6 if abs(den6)>1e-15 else 0
        tau6 = (pk6_lag + dp6) / fs
        errs['hilbert_xcorr'].append(tau6 - true_delay_s)
    
    all_results[SNR_dB] = {}
    for name, el in errs.items():
        ea = np.array(el)
        # Remove outliers (>10x CRB for robust stats)
        rmse = np.sqrt(np.mean(ea**2))
        bias = np.mean(ea)
        std = np.std(ea)
        all_results[SNR_dB][name] = {
            'rmse_ps': rmse*1e12, 'bias_ps': bias*1e12, 'std_ps': std*1e12,
            'crb_ps': CRB_std_ps
        }

names_ru = {
    'phase_slope':      '1. МНК наклона фазы beat',
    'fft_zp256':        '2. FFT + zero-pad (256x)',
    'fft8_parab':       '3. FFT(8x) + парабола',
    'ml_phase':         '4. ML фазовый Newton',
    'cross_spec_phase': '5. Кросс-спектр фаз. наклон',
    'hilbert_xcorr':    '6. Огибающая корреляции',
}

print("=" * 95)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ: 1000 Монте-Карло реализаций")
print(f"Параметры: BW={BW_rf/1e6:.0f} МГц, T={T_pulse*1e6:.0f} мкс, fs={fs/1e6:.0f} МГц, N={N}")
print(f"Истинная задержка: {true_delay_ns} нс ({true_delay_s*fs:.3f} отсчётов)")
print("=" * 95)

for SNR_dB in SNR_dB_values:
    crb = all_results[SNR_dB]['phase_slope']['crb_ps']
    print(f"\n  SNR = {SNR_dB} дБ  |  КРБ std(τ) = {crb:.2f} пс")
    print(f"  {'Метод':<32} {'RMSE пс':>10} {'Bias пс':>10} {'Std пс':>10} {'RMSE/КРБ':>10}")
    print("  " + "-" * 77)
    for key in errs.keys():
        r = all_results[SNR_dB][key]
        ratio = r['rmse_ps']/r['crb_ps']
        mark = " ✓" if ratio < 2 else (" ○" if ratio < 5 else " ✗")
        print(f"  {names_ru[key]:<32} {r['rmse_ps']:>10.2f} {r['bias_ps']:>10.2f} {r['std_ps']:>10.2f} {ratio:>9.2f}x{mark}")
