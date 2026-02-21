
import numpy as np

# Methods 4,5,6 have issues. Let me debug and fix:
# - ML Newton diverges because it's wideband and phase wraps
# - Cross-spectral phase: the LFM signal occupies very different bandwidth in DFT vs RF
# - Hilbert envelope: peak is at 0 because delay < 1 sample

# Let me implement all methods CORRECTLY for the dechirp (stretch processing) scenario

fs = 12e6; BW_rf = 500e6; T_pulse = 20e-6
chirp_rate = BW_rf / T_pulse
N = int(fs * T_pulse)
Ts = 1/fs

true_delay_ns = 3.75
true_delay_s = true_delay_ns * 1e-9
f_beat_true = chirp_rate * true_delay_s  # 93750 Hz

rng = np.random.default_rng(42)
t = np.arange(N) / fs

# Reference and delayed signals
s_ref = np.exp(1j * np.pi * chirp_rate * t**2)
t_d = t - true_delay_s
s_del = np.exp(1j * np.pi * chirp_rate * t_d**2)

beta_rms = BW_rf / np.sqrt(12)

SNR_list = [10, 20, 30, 40]
n_trials = 1000

all_res = {}

for SNR_dB in SNR_list:
    SNR_lin = 10**(SNR_dB/10)
    CRB_var = 1/(8*np.pi**2 * beta_rms**2 * N * SNR_lin)
    CRB_std = np.sqrt(CRB_var)
    
    methods = ['phase_slope_beat', 'fft_zp256', 'fft8_parab', 'ml_phase_beat', 
               'complex_xcorr_env', 'czt_zoom']
    errs = {m: [] for m in methods}
    
    for trial in range(n_trials):
        np_pw = 10**(-SNR_dB/10)
        noise = np.sqrt(np_pw/2)*(rng.standard_normal(N)+1j*rng.standard_normal(N))
        s_rx = s_del + noise
        beat = s_rx * np.conj(s_ref)
        # beat ≈ exp(-j*2*pi*chirp_rate*tau*t) * exp(j*pi*chirp_rate*tau^2) + noise
        # = exp(-j*2*pi*f_beat*t + j*phi0) + noise
        
        # === 1. Phase slope of beat (МНК) ===
        bp = np.unwrap(np.angle(beat))
        c = np.polyfit(t, bp, 1)
        tau1 = -c[0]/(2*np.pi*chirp_rate)
        errs['phase_slope_beat'].append(tau1 - true_delay_s)
        
        # === 2. FFT ZP 256x ===
        Nf = N*256
        Bf = np.fft.fft(beat, Nf)
        ff = np.fft.fftfreq(Nf, Ts)
        pk = np.argmax(np.abs(Bf))
        tau2 = -ff[pk]/chirp_rate
        errs['fft_zp256'].append(tau2 - true_delay_s)
        
        # === 3. FFT(8x) + parabolic peak ===
        Nf3 = N*8
        Bf3 = np.fft.fft(beat, Nf3)
        ff3 = np.fft.fftfreq(Nf3, Ts)
        m3 = np.abs(Bf3)
        pk3 = np.argmax(m3)
        ym = m3[(pk3-1)%Nf3]; y0 = m3[pk3]; yp = m3[(pk3+1)%Nf3]
        d3 = ym-2*y0+yp
        dp3 = 0.5*(ym-yp)/d3 if abs(d3)>1e-15 else 0
        fr3 = ff3[pk3] + dp3*(fs/Nf3)
        tau3 = -fr3/chirp_rate
        errs['fft8_parab'].append(tau3 - true_delay_s)
        
        # === 4. ML on beat signal (single tone estimation) ===
        # Beat is a single complex exponential at f_beat
        # ML estimation of frequency of complex exponential = phase slope
        # But let's do it via Newton on the log-likelihood
        # For complex exponential in AWGN:
        # s_beat_model = A*exp(j*2*pi*f*t)
        # MLE of f maximizes |sum(beat * exp(-j*2*pi*f*t))|^2
        # This is equivalent to FFT peak, but we can refine with Newton
        
        # Initialize from FFT
        Nf4 = N*32
        Bf4 = np.fft.fft(beat, Nf4)
        pk4 = np.argmax(np.abs(Bf4))
        f_init = np.fft.fftfreq(Nf4, Ts)[pk4]
        
        # Newton refinement for frequency of complex exponential
        f_est = f_init
        for it in range(20):
            e = np.exp(-1j*2*np.pi*f_est*t)
            S = np.sum(beat * e)
            # dS/df = sum(beat * (-j*2*pi*t) * e)
            dS = np.sum(beat * (-1j*2*np.pi*t) * e)
            # d2S/df2 = sum(beat * (-j*2*pi*t)^2 * e)
            d2S = np.sum(beat * (-(2*np.pi*t)**2) * e)
            
            # Maximize |S|^2 => d|S|^2/df = 2*Re(S'*conj(S)) = 0
            # Newton: df = -g/H where g = d|S|^2/df, H = d2|S|^2/df2
            g = 2*np.real(dS * np.conj(S))
            H = 2*np.real(d2S*np.conj(S) + np.abs(dS)**2)
            
            if abs(H) < 1e-20: break
            df = -g/H
            f_est += df
            if abs(df) < 1e-8: break
        
        tau4 = -f_est/chirp_rate
        errs['ml_phase_beat'].append(tau4 - true_delay_s)
        
        # === 5. Complex cross-correlation envelope ===
        # Cross-correlate in frequency domain, find peak of ENVELOPE (not RF)
        # For wideband LFM, use magnitude of complex cross-correlation
        # Zero-pad for sub-sample resolution
        ZP5 = 32
        S_rx5 = np.fft.fft(s_rx, N*ZP5)
        S_ref5 = np.fft.fft(s_ref, N*ZP5)
        CC5 = np.fft.ifft(S_rx5 * np.conj(S_ref5))
        env5 = np.abs(CC5)
        pk5 = np.argmax(env5)
        Ntot5 = N*ZP5
        if pk5 > Ntot5//2:
            pk5_lag = pk5 - Ntot5
        else:
            pk5_lag = pk5
        tau5 = pk5_lag / (fs * ZP5)
        
        # Parabolic refine on envelope
        ym5 = env5[(pk5-1)%Ntot5]; y05 = env5[pk5]; yp5 = env5[(pk5+1)%Ntot5]
        d5 = ym5-2*y05+yp5
        dp5 = 0.5*(ym5-yp5)/d5 if abs(d5)>1e-15 else 0
        tau5 = (pk5_lag + dp5)/(fs*ZP5)
        errs['complex_xcorr_env'].append(tau5 - true_delay_s)
        
        # === 6. CZT / Zoom FFT around beat frequency ===
        # Chirp-Z Transform for fine frequency resolution around f_beat region
        # First: coarse estimate from FFT
        f_coarse = f_init  # from method 4 init
        
        # Fine grid around coarse estimate
        df_fine = fs/(N*10000)  # very fine resolution
        n_fine = 201
        f_grid = f_coarse + np.arange(-n_fine//2, n_fine//2+1) * df_fine
        # DFT at each frequency
        spec = np.array([np.abs(np.sum(beat * np.exp(-1j*2*np.pi*f*t)))**2 for f in f_grid])
        pk6 = np.argmax(spec)
        
        # Parabolic on fine grid
        if 0 < pk6 < len(spec)-1:
            ym6 = spec[pk6-1]; y06 = spec[pk6]; yp6 = spec[pk6+1]
            d6 = ym6-2*y06+yp6
            dp6 = 0.5*(ym6-yp6)/d6 if abs(d6)>1e-15 else 0
            f_czt = f_grid[pk6] + dp6*df_fine
        else:
            f_czt = f_grid[pk6]
        tau6 = -f_czt/chirp_rate
        errs['czt_zoom'].append(tau6 - true_delay_s)
    
    all_res[SNR_dB] = {}
    for m in methods:
        ea = np.array(errs[m])
        all_res[SNR_dB][m] = {
            'rmse': np.sqrt(np.mean(ea**2))*1e12,
            'bias': np.mean(ea)*1e12,
            'std': np.std(ea)*1e12,
            'crb': CRB_std*1e12
        }

names = {
    'phase_slope_beat': '1. МНК фазы beat-сигнала',
    'fft_zp256':        '2. FFT + ZP (256x)',
    'fft8_parab':       '3. FFT(8x) + парабола',
    'ml_phase_beat':    '4. ML Newton (частота beat)',
    'complex_xcorr_env':'5. Корр. огибающая (ZP=32)',
    'czt_zoom':         '6. Zoom-FFT (CZT)',
}

print("=" * 95)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ: 6 методов × 1000 Монте-Карло × 4 SNR")
print(f"BW={BW_rf/1e6:.0f} МГц, T={T_pulse*1e6:.0f} мкс, fs={fs/1e6:.0f} МГц, N={N}, τ={true_delay_ns} нс")
print("=" * 95)

for SNR_dB in SNR_list:
    crb = all_res[SNR_dB]['phase_slope_beat']['crb']
    print(f"\n  SNR = {SNR_dB} дБ  |  КРБ = {crb:.2f} пс")
    print(f"  {'Метод':<32} {'RMSE':>8} {'Bias':>8} {'Std':>8} {'RMSE/КРБ':>10}")
    print("  " + "-" * 72)
    for m in methods:
        r = all_res[SNR_dB][m]
        rat = r['rmse']/r['crb']
        s = "★" if rat < 1.5 else ("✓" if rat < 3 else "✗")
        print(f"  {names[m]:<32} {r['rmse']:>7.1f} {r['bias']:>7.1f} {r['std']:>7.1f} {rat:>9.2f}x {s}")
